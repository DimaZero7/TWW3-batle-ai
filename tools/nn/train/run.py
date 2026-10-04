"""Train the battle network with PPO in the simulator (docs/en/training/training.md).

    bash tools/nn/dock.sh tools.nn.train.run --minutes 15 --name r1          # in the container, on the GPU
    bash tools/nn/dock.sh tools.nn.train.run --minutes 60 --name r2 --init build/nn-train/runs/r1/best.pt

The network starts from build/nn-train/random.pt (written if missing) or --init, plays `--battles`
battles at once against itself, its past versions and the scripted opponents, and learns after
every `--steps` decisions. The networks decide as in the game (tools/nn/train/cadence.py: every
--decide-s 1 s of battle, the orders --order-latency 0.36 s late; the simulator's step stays 0.5 s);
--gamma (and GAE's lambda) are per 0.5 s of battle and become per decision here, the horizon in
seconds the same. Writes (build/ is not in Git), in build/nn-train/runs/<name>/:

    latest.pt, best.pt     the network (tools/nn/train/checkpoint.py); best: the best window of
                           training battles against the scripts (the worst opponent-role counts)
    pool/v*.pt             past versions, the opponents of self-play
    log.jsonl              one line per update: losses, entropy, reward, win rates by opponent and role
    eval.json              the final evaluation (tools/nn/train/evaluate.py)
    replays/*/             battles of the final network, like the game's recordings
and copies the final network to build/nn-train/latest.pt (the companion's default).
"""
import argparse
import dataclasses
import json
import math
import sys
import time

import numpy as np
import torch

from tools.nn.model import config as model_config
from tools.nn.model import critic as model_critic
from tools.nn.model import policy as model_policy
from tools.nn.train import cadence as cad
from tools.nn.train import checkpoint, drills, evaluate, league, matchups, ppo, randomise, reward, rollout, scenes
from tools.nn.train import teach_auto
from tools.nn.train.drills import source as drill_source
from tools.nn.train.drills import teach as drill_teach

SCRIPTED = ("nearest", "hold_shoot", "hold", "ai_like")


def compatible(path, preset):
    """The checkpoint exists and matches the current code's network (actor and critic)."""
    try:
        data = checkpoint.read(path)
        cfg = checkpoint.config_of(data)
        model_policy.Actor(cfg).load_state_dict(data["actor"])
        model_critic.Critic(cfg).load(data["critic"])
        return data.get("preset") == preset
    except (FileNotFoundError, ValueError, RuntimeError, KeyError):
        return False


def networks(preset, device, start=None, critic_from=None):
    """(actor, critic) of `start` (default: random.pt, written if missing). The critic comes from
    critic_from when given (e.g. the run's latest.pt beside a test5 m<minute>.pt saved without it); a
    checkpoint without one starts a fresh critic (then give --critic-warmup updates)."""
    if start is None:
        start = checkpoint.RANDOM
        if not compatible(start, preset):
            checkpoint.write_random(start, preset)
    data = checkpoint.read(start)
    cfg = checkpoint.config_of(data)
    actor, critic = model_policy.Actor(cfg), model_critic.Critic(cfg)
    actor.load_state_dict(data["actor"])
    state = checkpoint.read(critic_from)["critic"] if critic_from else data.get("critic")
    if state is None:
        print(f"{start}: no critic in the checkpoint, a fresh one (give it --critic-warmup updates)", flush=True)
    else:
        critic.load(state)
    return actor.to(device).eval(), critic.to(device).eval()


def sized(cfg, N, slots=22, width=1.0):
    """The PPO settings for battles of N slots: fewer decisions per minibatch for bigger battles, so a
    minibatch holds about as many unit tokens as at 22 slots (the memory of a 16 GB card). A wider
    network (width: its token width / the small one's 128) keeps the minibatch and computes it in
    ceil(width) parts (gradient accumulation): the widened network (x 2) with the small one's minibatch
    did not fit in 16 GB, and a minibatch halved instead (03.10) doubled the optimizer's steps per update:
    Adam's steps are about lr each, so the policy moved ~2 times as far and the KL (quadratic in the
    step) ~4 times (the first update's KL over the whole batch 0.033 against the small network's 0.0073
    on the same batch; with the parts 0.010) - the wide run w1 collapsed within 5 minutes."""
    return dataclasses.replace(cfg, minibatch=max(256, int(cfg.minibatch * min(1.0, slots / N))),
                               accum=max(1, math.ceil(width - 1e-9)))


def optimizer(actor, critic, lr, width=1.0):
    """Adam over the actor and the critic. In a widened network (width > 1, tools/nn/model/widen.py) the
    weights that read the copied token stream (widen.stream_readers: fan-in x width) take lr / width:
    every copy gets the gradient the small weight got and Adam moves each about lr, so their sum (what
    the network computes) moved width times as far as in the small network (as muP's lr ~ 1 / fan-in)."""
    if width <= 1.0:
        return torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr, eps=1e-5)
    from tools.nn.model import widen
    readers, rest = [], []
    for model, spec in ((actor, widen.ACTOR), (critic, widen.CRITIC)):
        names = widen.stream_readers(model, spec)
        for n, p in model.named_parameters():
            (readers if n in names else rest).append(p)
    return torch.optim.Adam([{"params": readers, "lr": lr / width}, {"params": rest, "lr": lr}], lr=lr, eps=1e-5)


@torch.no_grad()
def follow(reference, actor, share):
    """The reference moves `share` of the way to the actor (Polyak averaging of the weights: --anchor-ema)."""
    for r, a in zip(reference.state_dict().values(), actor.state_dict().values()):
        if r.is_floating_point():
            r.lerp_(a, share)
        else:
            r.copy_(a)


def schedule(start, end, share):
    """Linear from start (share 0) to end (share 1)."""
    share = min(1.0, max(0.0, share))
    return start + (end - start) * share


def rates(stats):
    return {k: round(w / g, 3) for k, (g, w, _) in stats.items() if g}


def score(window):
    """The worst win rate against the scripts over (opponent, role) with enough battles; None if too few."""
    rs = [w / g for k, (g, w, _) in window.items() if k.split("/")[0] in SCRIPTED and g >= 20]
    keys = [k for k, (g, _, _) in window.items() if k.split("/")[0] in SCRIPTED and g >= 20]
    return (min(rs), keys) if len(rs) >= 5 else (None, keys)


def normal_drills(spec):
    """--teach-normal -> ({drill: fixed share} or None for adaptive, [drill names]): "auto" every drills.TRAIN
    drill with moments; "kiting,hold_fire" those, adaptive; json {"kiting": 0.1} fixed shares; "" none."""
    if not spec:
        return None, []
    if spec.strip().startswith("{"):
        fixed = {k: float(v) for k, v in json.loads(spec).items()}
        return fixed, list(fixed)
    names = [n for n in drills.TRAIN] if spec == "auto" else [n.strip() for n in spec.split(",") if n.strip()]
    return None, names


def train(args, every=None, teacher=None, normal=None):
    """every: (minutes, hook) - hook(actor, critic, minute, update) after every `minutes` of training
    (its time not counted as training: e.g. a full evaluation). teacher: with --drill-teach auto, the
    adaptive teacher (teach_auto.Auto; default: one with the init checkpoint's last drill numbers,
    teach_auto.prior) - the hook may change its shares (Auto.observe), read before every update. normal: with
    an adaptive --teach-normal, the teacher in normal battles (teach_auto.Transfer; default: one with the init
    checkpoint's last transfer numbers), likewise."""
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    drills.BROAD = args.drill_broad              # the drills' broad frames: their share of every drill's battles
    drills.EMBED = args.drill_embed              # the embedded frames: their share (of the drills that have one)
    out = checkpoint.DIR / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    if args.preset is None:          # the record follows the network: --init's preset (e.g. "wide"), else small
        args.preset = checkpoint.read(args.init).get("preset", "small") if args.init else "small"
    actor, critic = networks(args.preset, device, args.init, args.critic_init)
    reference = None
    if args.anchor:
        reference = checkpoint.load_policy(args.reference or args.init, device)
        for p in reference.parameters():
            p.requires_grad_(False)
    # The network the run started from, frozen: the distance from the start (ppo.distance, start_kl in
    # the log) next to the rating, whatever the reference does (--anchor-roll moves it).
    start = None
    if args.init:
        start = checkpoint.load_policy(args.init, device)
        for p in start.parameters():
            p.requires_grad_(False)
    cadence = cad.of_args(args)
    # gamma and lambda per decision: the --gamma given per 0.5 s (and the default lambda) over the decision's
    # seconds, so the horizon in seconds stays whatever the cadence (cadence.py)
    cfg = ppo.PPOConfig(lr=args.lr, gamma=cadence.discount(args.gamma), lam=cadence.discount(ppo.PPOConfig.lam),
                        epochs=args.epochs, minibatch=args.minibatch,
                        entropy=args.entropy, anchor=args.anchor, adv_norm=args.adv_norm)
    width = actor.cfg.d / model_config.SMALL.d
    opt = optimizer(actor, critic, cfg.lr, width)
    weights = reward.Weights(order_change=args.order_cost, idle=args.idle, lord=args.lord, retarget=args.retarget,
                             idle_tau_s=args.idle_tau, idle_cap=args.idle_cap, idle_pause_s=args.idle_pause,
                             idle_step=args.idle_step, idle_rate=args.idle_rate, idle_window_s=args.idle_window,
                             gold=args.gold, rout_share=args.rout_share, lord_rout=args.lord_rout)

    def small_arg():
        if not args.small:
            return None
        share, units = args.small.split(":")
        return float(share), int(units)
    entropy_end = args.entropy if args.entropy_end is None else args.entropy_end
    anchor_end = args.anchor if args.anchor_end is None else args.anchor_end
    pool = league.Pool(out / "pool", size=args.pool)
    pool.add(checkpoint.RANDOM)
    if args.init:
        pool.add(args.init)
    for extra in filter(None, (args.pool_extra or "").split(",")):
        pool.add(extra)
    past = model_policy.Actor(actor.cfg).to(device).eval()
    pasts = {actor.cfg: past}
    past_path = None
    mix = json.loads(args.mix) if args.mix else league.MIX
    if args.drills:
        # drills: --drills of the battles, shared by --drill-weights (default: the verified ones equally)
        shares = json.loads(args.drill_weights) if args.drill_weights else {n: 1.0 for n in drills.TRAIN}
        mix = league.with_drills(mix, args.drills, shares)
    lay = league.layout(args.battles, len(scenes.SCENES), mix, scene_attacker=scenes.attackers())
    # the drills' teacher (drills/teach.py): manual {drill: the imitation term's starting weight}, annealed
    # to 0; or auto (teach_auto.py): every READY drill the run plays, a share of its battles from its deficit
    played = set(np.unique(lay.opponent).tolist())
    auto = args.drill_teach == "auto"
    if auto:
        names = [n for n in drills.READY if league.CODE[drills.opponent(n)] in played]
        if not names:
            raise SystemExit("--drill-teach auto: the run plays no READY drill (--drills, --drill-weights)")
        if teacher is None:
            teacher = teach_auto.Auto(args.drill_teach_k, args.drill_teach_cap, args.drill_teach_weight,
                                      *teach_auto.prior(args.init))
        rows = teacher.bind(names)
        teach0 = dict.fromkeys(names, teacher.weight)
        print(f"drill teacher auto (k {teacher.k:g}, cap {teacher.cap:g}, weight {teacher.weight:g}), from "
              f"{rows[0]['source'] if rows else '-'}:\n" + "\n".join(teach_auto.table(rows)), flush=True)
    else:
        teacher = None
        teach0 = json.loads(args.drill_teach) if args.drill_teach else {}
    absent = [n for n in teach0 if n not in drills.NAMES or league.CODE[drills.opponent(n)] not in played]
    if absent:
        raise SystemExit(f"--drill-teach {absent}: not drills the run plays (--drills, --drill-weights)")
    # the teacher in normal battles (--teach-normal): the drill's teacher script at its moments in the rows of no
    # drill, a share of those battles (fixed, or adaptive from the transfer gap: teach_auto.Transfer)
    fixed_normal, normal_names = normal_drills(args.teach_normal)
    loaded_normal = drills.load(normal_names) if normal_names else {}
    bad = [n for n in normal_names if n not in loaded_normal or loaded_normal[n].moments is None]
    if bad:
        raise SystemExit(f"--teach-normal {bad}: no such drill with moments (drills.Drill.moments)")
    if normal_names and fixed_normal is None:
        if normal is None:
            normal = teach_auto.Transfer(args.teach_normal_k, args.teach_normal_cap, args.teach_normal_weight,
                                         *teach_auto.prior(args.init, "transfer"), match=args.teach_normal_match)
        rows = normal.bind(normal_names)
        print(f"teacher in normal battles auto (k {normal.k:g}, cap {normal.cap:g}, weight {normal.weight:g}, match "
              f"{normal.match:g}), from {rows[0]['source'] if rows else '-'}:\n"
              + "\n".join(teach_auto.table(rows, ref="ai_like applied", net="net applied")), flush=True)
    else:
        normal = None
    normal_w = ({drills.normal_name(n): args.teach_normal_weight for n in normal_names} if fixed_normal is not None
                else {})
    normal_s = {drills.normal_name(n): v for n, v in (fixed_normal or {}).items()}
    params = rollout.params_with_limit(args.limit)
    rng = np.random.default_rng(args.seed)

    def source():
        """A bank of random battles (tools/nn/armies) from fresh train seeds, with the drills' banks."""
        from tools.nn.armies import generate
        seeds = rng.integers(generate.TRAIN_SEEDS.start, generate.TRAIN_SEEDS.stop, args.bank)
        if args.drills:
            return drill_source.Mixed(seeds, args.max_units, params, device, lay, seed=int(rng.integers(1 << 30)),
                                      small=small_arg(), per_drill=args.drill_bank)
        return scenes.Generated(seeds, args.max_units, params, device, seed=int(rng.integers(1 << 30)),
                                small=small_arg())

    env = rollout.Battles(lay, scenes.SCENES, device, params, randomise.Spread(), weights, seed=args.seed,
                          source=source(), cadence=cadence,
                          teach=drills.load(list(teach0)) if teach0 else None,
                          teach_normal={n: loaded_normal[n] for n in normal_names} if normal_names else None)
    if normal_s:
        env.set_teach_shares(normal_s)
    step_cfg = sized(cfg, env.N, width=width)
    log = (out / "log.jsonl").open("w", encoding="utf-8", newline="\n")
    print(f"{args.name}: battles {env.B} (learner rows {env.R}), slots {env.N}, up to {args.max_units} units a side, "
          f"steps per update {args.steps} ({args.steps * env.decision_s:g} s of battle), "
          f"minibatch {step_cfg.minibatch} decisions in {step_cfg.accum} part(s), {cadence.text(params.dt)}, "
          f"model {args.preset} {model_policy.parameters(actor) / 1e6:.2f} M actor, limit {args.limit:.0f} s, "
          f"ppo {dataclasses.asdict(cfg)}, reward {dataclasses.asdict(weights)}, mix {mix}"
          + (f", drill teacher {teach0} -> 0 over {args.drill_teach_minutes:g} min" if teach0 and not auto else "")
          + (f", drill teacher auto {teacher.meta()}" if auto else "")
          + (f", teacher in normal battles {normal.meta()}" if normal is not None else "")
          + (f", teacher in normal battles fixed shares {normal_s} weight {args.teach_normal_weight:g}" if normal_s else ""),
          flush=True)

    def pick_past():
        nonlocal past, past_path
        path, untrained = pool.sample()
        if path != past_path:
            data = checkpoint.read(path, device)
            cfg_past = checkpoint.config_of(data)
            # a pool version of another size (random.pt beside a widened learner): its own actor, made once
            if cfg_past not in pasts:
                pasts[cfg_past] = model_policy.Actor(cfg_past).to(device).eval()
            past = pasts[cfg_past]
            past.load_state_dict(data["actor"])
            past_path = path
        env.set_past(past, untrained)

    pick_past()
    t_warm = time.time()
    env.step(actor, critic)                       # compiles the simulator's step (not counted)
    env.take_stats()
    env.orders_per_minute()
    env.kinds()
    env.lords()
    print(f"warm-up {time.time() - t_warm:.0f} s", flush=True)
    t0 = time.time()
    t_bank = t0
    update, decisions, total, window, best = 0, 0, {}, {}, -1.0
    floor_w = None                                # the entropy floor's current weight (--entropy-target)
    paused = 0.0
    rolls, rolled_at = 0, 0.0                     # --anchor-roll: renewals of the reference, training s of the last
    followed_at = 0.0                             # --anchor-ema: training s of the reference's last step
    if args.anchor_ema and args.anchor_roll:
        raise SystemExit("--anchor-ema and --anchor-roll: one of them")
    next_mark = every[0] if every else None
    def share_done():
        """The share of the run done: of the updates when --updates is set, else of the minutes."""
        if args.updates:
            return update / args.updates
        return (time.time() - t0 - paused) / (args.minutes * 60)

    while share_done() < 1.0 and (not args.updates or time.time() - t0 - paused < args.minutes * 60):
        t_u = time.time()
        done_share = share_done()
        if time.time() - t_bank >= args.bank_refresh * 60:
            env.source = source()
            env.bank = env.source.bank
            t_bank = time.time()
        if teacher is not None:
            env.set_teach_shares(teacher.shares)
        if normal is not None:
            env.set_teach_shares(normal.shares)
        batch = rollout.collect(env, actor, critic, args.steps)
        t_c = time.time()
        # Schedules: the kind's entropy bonus and the KL to the reference go linearly from their start to
        # their end value over the run (e.g. exploration fades out).
        # With --entropy-target the weight is the floor's (ppo.entropy_weight), never below the schedule.
        scheduled = schedule(args.entropy, entropy_end, done_share)
        floor_w = scheduled if floor_w is None else max(scheduled, floor_w)
        u_cfg = dataclasses.replace(step_cfg, entropy=floor_w if args.entropy_target else scheduled,
                                    anchor=schedule(args.anchor, anchor_end, done_share))
        trains = update >= args.critic_warmup
        teach_w = (teacher.weights() if teacher is not None
                   else drill_teach.weights(teach0, args.drill_teach_minutes, (time.time() - t0 - paused) / 60))
        teach_w = {**teach_w, **(normal.weights() if normal is not None else normal_w)}
        st = ppo.update(actor, critic, opt, batch, u_cfg, train_policy=trains, reference=reference, teach=teach_w)
        taught = st.pop("teach", None)
        if start is not None:
            st["start_kl"] = ppo.distance(actor, start, batch, u_cfg.minibatch)
        if args.entropy_target and trains:
            floor_w = ppo.entropy_weight(floor_w, st["entropy"], args.entropy_target, scheduled,
                                         max(scheduled, args.entropy_max), args.entropy_rate)
        del batch
        update += 1
        if args.gpu_duty < 1.0:
            # --gpu-duty: rest the GPU for this share of the time (quieter fans, a responsive desktop);
            # the rest does not count as training time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rest = (time.time() - t_u) * (1.0 / max(args.gpu_duty, 0.1) - 1.0)
            time.sleep(rest)
            paused += rest
        # --anchor-roll: every that many seconds of training the reference becomes the current actor (a
        # leash that moves with the network: it bounds the drift within a window, not the whole run).
        trained_s = time.time() - t0 - paused
        # --anchor-ema: after every update the reference moves towards the actor, by the share that halves
        # their distance in --anchor-ema seconds of training (a slow leash: a sudden collapse moves it a
        # little and the KL pulls back; steady progress it follows)
        if args.anchor_ema and reference is not None:
            follow(reference, actor, 1.0 - 0.5 ** ((trained_s - followed_at) / args.anchor_ema))
            followed_at = trained_s
        if args.anchor_roll and reference is not None and trained_s - rolled_at >= args.anchor_roll:
            reference.load_state_dict(actor.state_dict())
            rolls, rolled_at = rolls + 1, trained_s
            print(f"   anchor roll {rolls}: the KL reference is now the actor of update {update} "
                  f"({trained_s / 60:.1f} min of training)", flush=True)
        decisions += env.B * args.steps
        stats = env.take_stats()
        lords = env.lords()
        for k, (g, w, s) in stats.items():
            for acc in (total, window):
                G, W, S = acc.get(k, (0, 0, 0.0))
                acc[k] = (G + g, W + w, S + s * g)
        row = {"update": update, "seconds": round(time.time() - t0, 1), "battles": env.battles,
               "timeouts": env.timeouts, "battle_steps_per_s": round(env.B * args.steps / (time.time() - t_u)),
               "battle_seconds_per_s": round(env.B * args.steps * env.decision_s / (time.time() - t_u)),
               "collect_s": round(t_c - t_u, 2), "update_s": round(time.time() - t_c, 2),
               **{k: round(v, 4) for k, v in st.items()},
               "entropy_weight": round(u_cfg.entropy, 5), "anchor_weight": round(u_cfg.anchor, 4),
               "anchor_rolls": rolls,
               "lord_dead_own": round(lords["own"], 3), "lord_dead_enemy": round(lords["enemy"], 3),
               "abilities_per_battle": round(env.abilities(), 2),
               "switches_per_minute": round(lords["switches_per_minute"], 2),
               "orders_per_minute": round(env.orders_per_minute(), 2), "kinds": env.kinds(),
               "reward_parts": {r: {k: round(v, 4) for k, v in p.items()} for r, p in env.reward_parts().items()},
               "games": {k: g for k, (g, _, _) in stats.items()}, "win_rate": rates(stats),
               "seconds_per_battle": {k: round(s) for k, (_, _, s) in stats.items()}, "past": str(past_path.name)}
        if taught:
            row["teach"] = taught
        log.write(json.dumps(row) + "\n")
        log.flush()
        if update % args.print_every == 0:
            print(f"u{update:4d} {row['seconds']:6.0f}s battles {env.battles:6d} (timeouts {env.timeouts:5d}) "
                  f"{row['battle_steps_per_s']:6d} st/s ({row['battle_seconds_per_s']} battle-s/s) pl {st['policy_loss']:+.3f} vl {st['value_loss']:.4f} "
                  f"ent {st['entropy']:.2f}/{st['entropy_all']:.2f} kl {st['kl']:.3f} R {st['reward']:+.3f} "
                  f"anchor {st['anchor_kl']:.3f} start {st.get('start_kl', 0.0):.3f} orders/min {row['orders_per_minute']:5.1f} "
                  f"lords dead own {lords['own']:.2f} enemy {lords['enemy']:.2f} abil {row['abilities_per_battle']:.1f} "
                  f"kinds " + " ".join(f"{k[:2]} {v:.2f}" for k, v in row["kinds"].items()), flush=True)
            print(f"      ev {st.get('ev', 0):.3f} (attack {st.get('ev_attack', 0):.3f} defend {st.get('ev_defend', 0):.3f}) "
                  f"adv std attack {st.get('adv_std_attack', 0):.3f} defend {st.get('adv_std_defend', 0):.3f}; "
                  "reward a minute " + "; ".join(f"{r} " + " ".join(f"{k} {v:+.3f}" for k, v in p.items())
                                                 for r, p in row["reward_parts"].items()), flush=True)
            if taught:
                active = (lambda v: "-" if v["agree_active"] is None else f"{v['agree_active']:.3f}")
                share = (lambda v: f"share {v['share']:.2f} (labelled {v['labelled']:.2f}) " if "share" in v else "")
                print("      teacher " + "; ".join(f"{k} {share(v)}weight {v['weight']:.3f} ce {v['ce']:.3f} agree {v['agree']:.3f} "
                                                  f"(kind {v['agree_kind']:.3f}, active {active(v)} of {v['active']}; "
                                                  f"{v['units']} unit-decisions)" for k, v in taught.items()), flush=True)
        if update % args.snapshot_every == 0:
            meta = {"update": update, "battles": env.battles, "seconds": row["seconds"], "run": args.name,
                    "cadence": cadence.meta()}
            pool.save(actor, None, args.preset, meta)
            checkpoint.save(out / "latest.pt", actor, critic, args.preset, meta)
            sc, _ = score(window)
            print(f"   window: " + ", ".join(f"{k} {w / g:.2f} ({g})" for k, (g, w, _) in sorted(window.items())),
                  flush=True)
            if sc is not None:
                if sc > best:
                    best = sc
                    checkpoint.save(out / "best.pt", actor, critic, args.preset, dict(meta, score=sc))
                    print(f"   best: worst scripted win rate {sc:.2f}", flush=True)
                window = {}
            pick_past()
        elif update % 2 == 0:
            pick_past()
        if every and time.time() - t0 - paused >= next_mark * 60 and time.time() - t0 - paused < args.minutes * 60:
            t_e = time.time()
            torch.cuda.empty_cache()
            every[1](actor, critic, next_mark, update)
            actor.eval()
            critic.eval()
            torch.cuda.empty_cache()
            paused += time.time() - t_e
            next_mark += every[0]
    seconds = time.time() - t0
    meta = {"update": update, "battles": env.battles, "seconds": round(seconds), "decisions": decisions,
            "battles_at_once": env.B, "steps_per_update": args.steps, "limit_s": args.limit, "run": args.name,
            "cadence": cadence.meta(),
            "ppo": dataclasses.asdict(cfg), "reward": dataclasses.asdict(weights)}
    checkpoint.save(out / "latest.pt", actor, critic, args.preset, meta)
    if not (out / "best.pt").exists():
        checkpoint.save(out / "best.pt", actor, critic, args.preset, meta)
    log.close()
    summary = {"updates": update, "seconds": round(seconds), "battles": env.battles, "timeouts": env.timeouts,
               "battle_steps_per_s": round(decisions / seconds),
               "battle_seconds_per_s": round(decisions * env.decision_s / seconds),
               "battles_per_min": round(env.battles / (seconds / 60)), "cadence": cadence.meta(),
               "win_rate_all": {k: round(w / g, 3) for k, (g, w, _) in sorted(total.items()) if g},
               "games": {k: g for k, (g, _, _) in sorted(total.items())},
               "mean_battle_s": {k: round(s / g) for k, (g, _, s) in sorted(total.items()) if g}, "best_score": best}
    print("trained:", json.dumps(summary), flush=True)
    return actor, summary, out


def show(name, r):
    print(f"{name}: orders/min {r['orders_per_minute']:.1f}, kinds "
          + ", ".join(f"{k} {v:.2f}" for k, v in r["kinds"].items()), flush=True)
    for opp, o in r["by_opponent"].items():
        for role, x in o["roles"].items():
            if not x.get("games"):
                continue
            print(f"  v {opp:10} {role:6} win {x['win_rate']:.3f} ({x['wins']}/{x['games']}), {x['seconds']:5.0f} s, "
                  f"HP lost own {x['hp_own_lost']:.2f} enemy {x['hp_enemy_lost']:.2f}, "
                  f"gold lost own {x.get('gold_lost', float('nan')):.0f} enemy {x.get('gold_destroyed', float('nan')):.0f} "
                  f"(ratio {x.get('gold_ratio', float('nan')):.2f}), timeouts {x['timeouts']:.2f}, "
                  f"lord dead own {x.get('lord_dead_own', float('nan')):.2f} enemy {x.get('lord_dead_enemy', float('nan')):.2f}")
        print(f"      kinds " + ", ".join(f"{k} {v:.2f}" for k, v in o["kinds"].items()), flush=True)
        if "matchups" in o:
            print(f"      by faction: {matchups.text(o)}", flush=True)
        if len(r["by_scene"][opp]) <= 12:
            print("      " + "; ".join(f"{k} {v['win_rate']:.2f}" for k, v in r["by_scene"][opp].items()), flush=True)


def final_eval(path, args, summary, out):
    device = torch.device(args.device)
    cadence = cad.of_args(args)
    actor = checkpoint.load_policy(path, device)
    untrained = checkpoint.load_policy(args.eval_past or checkpoint.RANDOM, device)
    res = {"training": summary, "checkpoint": str(path)}
    t = time.time()
    res["trained"] = evaluate.play(actor, past=untrained, device=device, limit_s=args.limit,
                                   generated=args.eval_generated, max_units=args.max_units, cadence=cadence)
    print(f"eval {path.name} {time.time() - t:.0f} s", flush=True)
    show(path.name, res["trained"])
    (out / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    paths = evaluate.record(actor, out / "replays", device=device, limit_s=args.limit, source=str(path), cadence=cadence)
    print(f"replays: {len(paths)} in {out / 'replays'}", flush=True)
    return res


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="run")
    ap.add_argument("--init", help="start from this checkpoint (default: random.pt)")
    ap.add_argument("--minutes", type=float, default=5.0, help="wall time of training (the warm-up not counted)")
    ap.add_argument("--updates", type=int, default=0,
                    help="train this many updates instead (--minutes then only caps the time; schedules follow the updates)")
    ap.add_argument("--battles", type=int, default=1024, help="battles at once")
    ap.add_argument("--steps", type=int, default=64, help="decisions per chunk (between updates)")
    ap.add_argument("--preset", choices=sorted(model_config.PRESETS),
                    help="the size of a network started from random.pt (default: --init's, else small; a checkpoint "
                         "brings its own sizes)")
    ap.add_argument("--limit", type=float, default=3600.0, help="battle time limit, s (the defender wins at it)")
    ap.add_argument("--lr", type=float, default=ppo.PPOConfig.lr)
    ap.add_argument("--gamma", type=float, default=ppo.PPOConfig.gamma,
                    help="discount per 0.5 s of battle (per decision: gamma ** (--decide-s / 0.5); GAE's lambda too)")
    cad.add_args(ap)
    ap.add_argument("--epochs", type=int, default=ppo.PPOConfig.epochs)
    ap.add_argument("--minibatch", type=int, default=ppo.PPOConfig.minibatch, help="decisions per minibatch")
    ap.add_argument("--entropy", type=float, default=ppo.PPOConfig.entropy, help="weight of the kind's entropy")
    ap.add_argument("--entropy-end", type=float, help="... at the end of the run (linear; default: no change)")
    ap.add_argument("--entropy-target", type=float, default=0.0,
                    help="> 0: an entropy floor - the weight goes up x --entropy-rate every update while the kind's "
                         "entropy is below this, back down to the schedule above it (0: the schedule only)")
    ap.add_argument("--entropy-max", type=float, default=0.1, help="the floor's weight at most this")
    ap.add_argument("--entropy-rate", type=float, default=1.25, help="the floor's factor per update")
    ap.add_argument("--order-cost", type=float, default=reward.Weights.order_change)
    ap.add_argument("--idle", type=float, default=reward.Weights.idle)
    ap.add_argument("--gold", type=float, default=reward.Weights.gold,
                    help="(enemy gold destroyed - own gold lost) / budget, per step")
    ap.add_argument("--rout-share", type=float, default=reward.Weights.rout_share,
                    help="a routing unit (it may rally) loses this share of the gold it has left")
    ap.add_argument("--lord", type=float, default=reward.Weights.lord, help="the enemy lord's death - own lord's death")
    ap.add_argument("--lord-rout", type=float, default=reward.Weights.lord_rout,
                    help="> 0: in the lord term a shattered lord counts as dead, a routing one as this share of a death")
    ap.add_argument("--retarget", type=float, default=reward.Weights.retarget,
                    help="cost of switching an attack to another target while the old one stands")
    ap.add_argument("--mix", help='opponent shares as json, e.g. {"self": 0.2, "nearest": 0.4}')
    ap.add_argument("--critic-warmup", type=int, default=0, help="first updates train only the critic")
    ap.add_argument("--anchor", type=float, default=ppo.PPOConfig.anchor, help="KL weight to the reference actor")
    ap.add_argument("--anchor-end", type=float, help="... at the end of the run (linear; default: no change)")
    ap.add_argument("--reference", help="the reference actor (default: --init)")
    ap.add_argument("--anchor-roll", type=float, default=0.0,
                    help="seconds of training between renewals of the KL reference (--reference or --init): it "
                         "becomes the current actor (a rolling anchor; 0: fixed for the run)")
    ap.add_argument("--anchor-ema", type=float, default=0.0,
                    help="> 0: after every update the reference moves towards the actor (Polyak averaging of the "
                         "weights) with this half-life in seconds of training (0: off; not with --anchor-roll)")
    ap.add_argument("--critic-init", help="take the critic from this checkpoint (default: --init's own)")
    ap.add_argument("--adv-norm", default=ppo.PPOConfig.adv_norm, choices=("batch", "role"),
                    help="normalise the side's advantage over the minibatch or over each role apart")
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--pool-extra", help="more past opponents for the pool (checkpoints, comma-separated)")
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--eval-past", help="the network behind 'past' in the final evaluation (default: random.pt)")
    ap.add_argument("--small", help="share:units - that share of every generated bank with at most `units` a side")
    ap.add_argument("--idle-tau", type=float, default=reward.Weights.idle_tau_s,
                    help="s: before its first damage the attacker's idle cost is --idle x (exp(t / this) - 1)")
    ap.add_argument("--idle-pause", type=float, default=reward.Weights.idle_pause_s,
                    help="s: after it, a step up every this many seconds without damage (0 before the first)")
    ap.add_argument("--idle-step", type=float, default=reward.Weights.idle_step,
                    help="... --idle x (exp(steps x this) - 1); new damage sets it back to 0")
    ap.add_argument("--idle-cap", type=float, default=reward.Weights.idle_cap,
                    help="the idle cost at most --idle x this")
    ap.add_argument("--idle-rate", type=float, default=reward.Weights.idle_rate,
                    help="0: any damage resets the idle timer; > 0: only a damage rate of at least this share of "
                         "the budget a minute (defender gold lost, mean over --idle-window)")
    ap.add_argument("--idle-window", type=float, default=reward.Weights.idle_window_s,
                    help="s: the time constant of that damage rate")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--max-units", type=int, default=19, help="random armies (tools/nn/armies): units a side at most")
    ap.add_argument("--bank", type=int, default=2048, help="generated battles ready at once")
    ap.add_argument("--drills", type=float, default=0.0,
                    help="share of the battles that are drills (tools/nn/train/drills; e.g. 0.08), taken from the "
                         "other opponents in proportion")
    ap.add_argument("--drill-weights", help="the drills' shares of --drills as json, e.g. {\"pincer\": 1, \"kiting\": 2} "
                                            "(default: drills.TRAIN equally - kiting, hold_fire; counter is READY but "
                                            "not trained by default: it costs in normal battles)")
    ap.add_argument("--drill-bank", type=int, default=256, help="battles of each drill ready at once (renewed with the bank)")
    ap.add_argument("--drill-broad", type=float, default=drills.BROAD_DEFAULT,
                    help="share of every drill's battles from its broad frame (the situation in a messier battle: more "
                         "unit types, uninvolved units, lords; tools/nn/train/drills), the rest from the clean frame; "
                         "also the drill evaluation's mix (test5)")
    ap.add_argument("--drill-embed", type=float, default=drills.EMBED_DEFAULT,
                    help="share of the battles of a drill that has an embedded frame (kiting, hold_fire) from it: the "
                         "situation inside a normal generated battle (both lords, normal sizes, no other cue); the "
                         "rest clean / broad by --drill-broad; also the drill evaluation's mix (test5); 0: the old mix")
    ap.add_argument("--teach-normal", default="",
                    help="the teacher in NORMAL battles (tools/nn/train/teach_auto.py Transfer): the drill's teacher "
                         "script labels our units in the ordinary training battles at the drill's moments only (its "
                         "transfer detector's situation), a share of those battles: 'auto' (every drills.TRAIN drill, "
                         "adaptive: share = min(cap, k x gap), gap = max(0, ai_like's applied share - the network's) / "
                         "ai_like's from every test5 evaluation's transfer block, 0 within --teach-normal-match), "
                         "'kiting' (names, adaptive) or json {\"kiting\": 0.1} (fixed shares)")
    ap.add_argument("--teach-normal-k", type=float, default=teach_auto.NORMAL_K, help="--teach-normal adaptive: k")
    ap.add_argument("--teach-normal-cap", type=float, default=teach_auto.NORMAL_CAP,
                    help="--teach-normal adaptive: the share of the normal battles labelled at most this")
    ap.add_argument("--teach-normal-weight", type=float, default=teach_auto.NORMAL_WEIGHT,
                    help="--teach-normal: the imitation weight on a labelled unit (the pull is weight x share)")
    ap.add_argument("--teach-normal-match", type=float, default=teach_auto.NORMAL_MATCH,
                    help="--teach-normal adaptive: no labels while the gap is at most this")
    ap.add_argument("--drill-teach", help="the drills' teacher: 'auto' (tools/nn/train/teach_auto.py: every READY drill "
                                          "the run plays, a share of its battles labelled from how far the network is "
                                          "behind the skilled script, renewed after every test5 evaluation) or, manual, "
                                          "json {drill: weight}, e.g. {\"kiting\": 0.5}: in that drill's battles + "
                                          "weight x cross-entropy of the policy against the drill's skilled script "
                                          "(labels only; tools/nn/train/drills/teach.py), the weight going linearly to 0 "
                                          "over --drill-teach-minutes")
    ap.add_argument("--drill-teach-k", type=float, default=teach_auto.K,
                    help="auto: share = min(cap, k x deficit), deficit = max(0, skilled - net win) / skilled")
    ap.add_argument("--drill-teach-cap", type=float, default=teach_auto.CAP, help="auto: the share at most this")
    ap.add_argument("--drill-teach-weight", type=float, default=teach_auto.WEIGHT,
                    help="auto: the imitation weight on a labelled unit (fixed; the pull is weight x share)")
    ap.add_argument("--drill-teach-minutes", type=float, default=10.0,
                    help="minutes of training over which the teacher's weight goes to 0")
    ap.add_argument("--bank-refresh", type=float, default=5.0, help="minutes between new banks of armies")
    ap.add_argument("--gpu-duty", type=float, default=1.0,
                    help="share of the time the GPU works (e.g. 0.9: after each update rest 1/9 of its time; "
                         "quieter fans, a responsive desktop; the rest is not counted as training time)")
    ap.add_argument("--eval-generated", type=int, default=512, help="random battles per opponent (EVAL_SEEDS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args()
    actor, summary, out = train(args)
    checkpoint.save(checkpoint.LATEST, actor, None, args.preset, checkpoint.meta(out / "latest.pt"))
    if not args.no_eval:
        final_eval(out / "latest.pt", args, summary, out)


if __name__ == "__main__":
    main()
