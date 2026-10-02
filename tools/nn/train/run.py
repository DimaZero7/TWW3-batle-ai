"""Train the battle network with PPO in the simulator (docs/en/training/training.md).

    bash tools/nn/dock.sh tools.nn.train.run --minutes 15 --name r1          # in the container, on the GPU
    bash tools/nn/dock.sh tools.nn.train.run --minutes 60 --name r2 --init build/nn-train/runs/r1/best.pt

The network starts from build/nn-train/random.pt (written if missing) or --init, plays `--battles`
battles at once against itself, its past versions and the scripted opponents, and learns after
every `--steps` decisions. Writes (build/ is not in Git), in build/nn-train/runs/<name>/:

    latest.pt, best.pt     the network (tools/nn/train/checkpoint.py); best: the best window of
                           training battles against the scripts (the worst opponent-role counts)
    pool/v*.pt             past versions, the opponents of self-play
    log.jsonl              one line per update: losses, entropy, reward, win rates by opponent and role
    eval.json              the final evaluation (tools/nn/train/evaluate.py)
    replays/*/             battles of the final network, like the game's recordings
and copies the final network to build/nn-train/latest.pt (the companion's default).
"""
import argparse
import copy
import dataclasses
import json
import shutil
import sys
import time

import numpy as np
import torch

from tools.nn.model import config as model_config
from tools.nn.model import critic as model_critic
from tools.nn.model import policy as model_policy
from tools.nn.train import checkpoint, drills, evaluate, league, matchups, ppo, randomise, reward, rollout, scenes
from tools.nn.train.drills import source as drill_source

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


def soften_kind(actor, temperature):
    """Divide the order kind's logits by `temperature` (the head's last layer), once: a policy that
    collapsed to one kind (long19: attack with probability 1.000, the entropy bonus's gradient
    vanishes there) gets the others back with a few per cent each (at 4: attack 0.92) and PPO can
    try them; the most likely kind stays the same."""
    with torch.no_grad():
        actor.heads.kind.weight.div_(temperature)
        actor.heads.kind.bias.div_(temperature)


def sized(cfg, N, slots=22):
    """The PPO settings for battles of N slots: fewer decisions per minibatch for bigger battles, so a
    minibatch holds about as many unit tokens as at 22 slots (the memory of a 16 GB card)."""
    return dataclasses.replace(cfg, minibatch=max(256, int(cfg.minibatch * min(1.0, slots / N))))


def schedule(start, end, share):
    """Linear from start (share 0) to end (share 1)."""
    share = min(1.0, max(0.0, share))
    return start + (end - start) * share


def curriculum(text):
    """'5:0.3,10:0.6,19:1' -> [(5, 0.3), (10, 0.6), (19, 1.0)]: max units a side until a share of the time."""
    return [(int(a), float(b)) for a, b in (part.split(":") for part in text.split(","))]


def rates(stats):
    return {k: round(w / g, 3) for k, (g, w, _) in stats.items() if g}


def score(window):
    """The worst win rate against the scripts over (opponent, role) with enough battles; None if too few."""
    rs = [w / g for k, (g, w, _) in window.items() if k.split("/")[0] in SCRIPTED and g >= 20]
    keys = [k for k, (g, _, _) in window.items() if k.split("/")[0] in SCRIPTED and g >= 20]
    return (min(rs), keys) if len(rs) >= 5 else (None, keys)


def train(args, every=None):
    """every: (minutes, hook) - hook(actor, critic, minute, update) after every `minutes` of training
    (its time not counted as training: e.g. a full evaluation)."""
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = checkpoint.DIR / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    actor, critic = networks(args.preset, device, args.init, args.critic_init)
    if args.kind_temperature != 1.0:
        soften_kind(actor, args.kind_temperature)
    reference = None
    own_reference = args.reference == "self"
    if args.anchor:
        if own_reference:
            # a trust region to the network's own recent version: a frozen copy of the actor, renewed
            # every --reference-every updates (it bounds how far noisy steps drift between renewals,
            # not where the policy may go)
            reference = copy.deepcopy(actor).eval()
        else:
            reference = checkpoint.load_policy(args.reference or args.init, device)
            if args.kind_temperature != 1.0:
                # the same softening, or the KL would pull the softened kind straight back to the sharp one
                soften_kind(reference, args.kind_temperature)
        for p in reference.parameters():
            p.requires_grad_(False)
    # The network the run started from, frozen: the distance from the start (ppo.distance, start_kl in
    # the log) next to the rating, whatever the reference does (--anchor-roll moves it).
    start = None
    if args.init:
        start = checkpoint.load_policy(args.init, device)
        if args.kind_temperature != 1.0:
            soften_kind(start, args.kind_temperature)
        for p in start.parameters():
            p.requires_grad_(False)
    cfg = ppo.PPOConfig(lr=args.lr, gamma=args.gamma, epochs=args.epochs, minibatch=args.minibatch,
                        entropy=args.entropy, anchor=args.anchor, unit_credit=args.unit_credit,
                        unit_value=args.unit_value,
                        adv_norm=args.adv_norm)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr, eps=1e-5)
    weights = reward.Weights(order_change=args.order_cost, timeout=args.timeout, idle=args.idle, hp=args.hp,
                             standing=args.standing, lord=args.lord, retarget=args.retarget,
                             idle_tau_s=args.idle_tau, idle_cap=args.idle_cap, idle_pause_s=args.idle_pause,
                             idle_step=args.idle_step, idle_share=args.idle_share, idle_rate=args.idle_rate,
                             idle_window_s=args.idle_window,
                             unit_gold=args.unit_gold, unit_attrib=args.unit_attrib, flanked=args.flanked, missile_melee=args.missile_melee,
                             crowd=args.crowd, flank_attack=args.flank_attack, neighbour=args.neighbour,
                             idle_near=args.idle_near, shirk=args.shirk, shirk_m=args.shirk_m, shirk_side=args.shirk_side, gold=args.gold, rout_share=args.rout_share,
                             unit_idle=args.unit_idle, lord_rout=args.lord_rout,
                             lord_exposed=args.lord_exposed, lord_exposed_hp=args.lord_exposed_hp,
                             lord_lead=args.lord_lead, lord_lead_m=args.lord_lead_m, lord_lead_near=args.lord_lead_near,
                             lord_fall=args.lord_fall)

    def small_arg():
        if not args.small:
            return None
        share, units = args.small.split(":")
        return float(share), int(units)
    entropy_end = args.entropy if args.entropy_end is None else args.entropy_end
    anchor_end = args.anchor if args.anchor_end is None else args.anchor_end
    unit_credit_end = args.unit_credit if args.unit_credit_end is None else args.unit_credit_end
    pool = league.Pool(out / "pool", size=args.pool)
    pool.add(checkpoint.RANDOM)
    if args.init:
        pool.add(args.init)
    for extra in filter(None, (args.pool_extra or "").split(",")):
        pool.add(extra)
    past = model_policy.Actor(actor.cfg).to(device).eval()
    past_path = None
    mix = json.loads(args.mix) if args.mix else league.MIX
    if args.drills:
        # drills: --drills of the battles, shared by --drill-weights (default: the verified ones equally)
        if args.armies != "generated":
            raise SystemExit("--drills needs --armies generated")
        shares = json.loads(args.drill_weights) if args.drill_weights else {n: 1.0 for n in drills.READY}
        mix = league.with_drills(mix, args.drills, shares)
    lay = league.layout(args.battles, len(scenes.SCENES), mix, scene_attacker=scenes.attackers())
    params = rollout.params_with_limit(args.limit)
    stages = curriculum(args.curriculum) if args.armies == "generated" else [(None, 1.0)]
    rng = np.random.default_rng(args.seed)

    def source(max_units):
        """A bank of random battles from fresh train seeds (None: the fixed scenes)."""
        if max_units is None:
            return None
        from tools.nn.armies import generate
        seeds = rng.integers(generate.TRAIN_SEEDS.start, generate.TRAIN_SEEDS.stop, args.bank)
        if args.drills:
            return drill_source.Mixed(seeds, max_units, params, device, lay, seed=int(rng.integers(1 << 30)),
                                      small=small_arg(), per_drill=args.drill_bank)
        return scenes.Generated(seeds, max_units, params, device, seed=int(rng.integers(1 << 30)), small=small_arg())

    def make_env(max_units):
        return rollout.Battles(lay, scenes.SCENES, device, params, randomise.Spread(), weights, seed=args.seed,
                               source=source(max_units))
    stage = 0
    env = make_env(stages[0][0])
    step_cfg = sized(cfg, env.N)
    log = (out / "log.jsonl").open("w", encoding="utf-8", newline="\n")
    print(f"{args.name}: battles {env.B} (learner rows {env.R}), slots {env.N}, armies {args.armies} {stages}, "
          f"steps per update {args.steps}, "
          f"model {args.preset} {model_policy.parameters(actor) / 1e6:.2f} M actor, limit {args.limit:.0f} s, "
          f"ppo {dataclasses.asdict(cfg)}, reward {dataclasses.asdict(weights)}, mix {mix}", flush=True)

    def pick_past():
        nonlocal past_path
        path, untrained = pool.sample()
        if path != past_path:
            past.load_state_dict(checkpoint.read(path, device)["actor"])
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
    best_eval, t_eval, paused = -1.0, time.time(), 0.0
    rolls, rolled_at = 0, 0.0                     # --anchor-roll: renewals of the reference, training s of the last
    next_mark = every[0] if every else None
    elog = (out / "eval_log.jsonl").open("w", encoding="utf-8", newline="\n")
    def share_done():
        """The share of the run done: of the updates when --updates is set, else of the minutes."""
        if args.updates:
            return update / args.updates
        return (time.time() - t0 - paused) / (args.minutes * 60)

    while share_done() < 1.0 and (not args.updates or time.time() - t0 - paused < args.minutes * 60):
        t_u = time.time()
        done_share = share_done()
        if stage + 1 < len(stages) and done_share >= stages[stage][1]:
            stage += 1
            del env
            torch.cuda.empty_cache()
            env = make_env(stages[stage][0])
            step_cfg = sized(cfg, env.N)
            pick_past()
            print(f"   curriculum: up to {stages[stage][0]} units a side (slots {env.N})", flush=True)
            t_bank = time.time()
        elif stages[stage][0] is not None and time.time() - t_bank >= args.bank_refresh * 60:
            env.source = source(stages[stage][0])
            env.bank = env.source.bank
            t_bank = time.time()
        batch = rollout.collect(env, actor, critic, args.steps)
        t_c = time.time()
        # Schedules: the kind's entropy bonus, the KL to the reference and the per-unit credit go linearly
        # from their start to their end value over the run (e.g. exploration and the warm start's hold fade
        # out; the units' own credit hands over to the side's, OpenAI Five's "team spirit").
        # With --entropy-target the weight is the floor's (ppo.entropy_weight), never below the schedule.
        scheduled = schedule(args.entropy, entropy_end, done_share)
        floor_w = scheduled if floor_w is None else max(scheduled, floor_w)
        u_cfg = dataclasses.replace(step_cfg, entropy=floor_w if args.entropy_target else scheduled,
                                    anchor=schedule(args.anchor, anchor_end, done_share),
                                    unit_credit=schedule(args.unit_credit, unit_credit_end, done_share))
        trains = update >= args.critic_warmup
        st = ppo.update(actor, critic, opt, batch, u_cfg, train_policy=trains, reference=reference)
        if start is not None:
            st["start_kl"] = ppo.distance(actor, start, batch, u_cfg.minibatch)
        if args.entropy_target and trains:
            floor_w = ppo.entropy_weight(floor_w, st["entropy"], args.entropy_target, scheduled,
                                         max(scheduled, args.entropy_max), args.entropy_rate)
        del batch
        update += 1
        if own_reference and reference is not None and update % args.reference_every == 0:
            reference.load_state_dict(actor.state_dict())
        # --anchor-roll: every that many seconds of training the reference becomes the current actor (a
        # leash that moves with the network: it bounds the drift within a window, not the whole run).
        trained_s = time.time() - t0 - paused
        if args.anchor_roll and reference is not None and not own_reference and trained_s - rolled_at >= args.anchor_roll:
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
               "collect_s": round(t_c - t_u, 2), "update_s": round(time.time() - t_c, 2),
               **{k: round(v, 4) for k, v in st.items()},
               "entropy_weight": round(u_cfg.entropy, 5), "anchor_weight": round(u_cfg.anchor, 4),
               "unit_credit": round(u_cfg.unit_credit, 4),
               "anchor_rolls": rolls,
               "lord_dead_own": round(lords["own"], 3), "lord_dead_enemy": round(lords["enemy"], 3),
               "abilities_per_battle": round(env.abilities(), 2),
               "switches_per_minute": round(lords["switches_per_minute"], 2),
               "orders_per_minute": round(env.orders_per_minute(), 2), "kinds": env.kinds(),
               "reward_parts": {r: {k: round(v, 4) for k, v in p.items()} for r, p in env.reward_parts().items()},
               "games": {k: g for k, (g, _, _) in stats.items()}, "win_rate": rates(stats),
               "seconds_per_battle": {k: round(s) for k, (_, _, s) in stats.items()}, "past": str(past_path.name)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if update % args.print_every == 0:
            print(f"u{update:4d} {row['seconds']:6.0f}s battles {env.battles:6d} (timeouts {env.timeouts:5d}) "
                  f"{row['battle_steps_per_s']:6d} st/s pl {st['policy_loss']:+.3f} vl {st['value_loss']:.4f} "
                  f"ent {st['entropy']:.2f}/{st['entropy_all']:.2f} kl {st['kl']:.3f} R {st['reward']:+.3f} "
                  f"anchor {st['anchor_kl']:.3f} start {st.get('start_kl', 0.0):.3f} unit A share {st['unit_adv_share']:.2f} orders/min {row['orders_per_minute']:5.1f} "
                  f"lords dead own {lords['own']:.2f} enemy {lords['enemy']:.2f} abil {row['abilities_per_battle']:.1f} "
                  f"kinds " + " ".join(f"{k[:2]} {v:.2f}" for k, v in row["kinds"].items()), flush=True)
            print(f"      ev {st.get('ev', 0):.3f} (attack {st.get('ev_attack', 0):.3f} defend {st.get('ev_defend', 0):.3f}) "
                  f"adv std attack {st.get('adv_std_attack', 0):.3f} defend {st.get('adv_std_defend', 0):.3f}; "
                  "reward a minute " + "; ".join(f"{r} " + " ".join(f"{k} {v:+.3f}" for k, v in p.items())
                                                 for r, p in row["reward_parts"].items()), flush=True)
        if update % args.snapshot_every == 0:
            meta = {"update": update, "battles": env.battles, "seconds": row["seconds"], "run": args.name}
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
        if args.eval_every and time.time() - t_eval >= args.eval_every * 60:
            # Select by evaluation: whole battles on EVAL_SEEDS against the given opponents, both roles;
            # the score is the mean win rate over (opponent, role). Its time is not training time.
            t_e = time.time()
            res = evaluate.play(actor, opponents=tuple(args.eval_opponents.split(",")), device=device,
                                limit_s=args.limit, generated=args.eval_battles, max_units=stages[-1][0] or 19,
                                small=small_arg())
            rates_ = {f"{o}/{r}": x["win_rate"] for o, v in res["by_opponent"].items()
                      for r, x in v["roles"].items() if x.get("games")}
            sc = float(np.mean(list(rates_.values())))
            meta = {"update": update, "seconds": round(t_e - t0 - paused), "run": args.name, "eval": rates_,
                    "eval_score": sc, "kinds": res["kinds"]}
            elog.write(json.dumps(meta) + "\n")
            elog.flush()
            if sc > best_eval:
                best_eval = sc
                checkpoint.save(out / "best_eval.pt", actor, critic, args.preset, meta)
            print(f"   eval u{update}: score {sc:.3f} (best {best_eval:.3f}) "
                  + ", ".join(f"{k} {v:.2f}" for k, v in rates_.items()) + " kinds "
                  + " ".join(f"{k[:2]} {v:.2f}" for k, v in res["kinds"].items())
                  + f" [{time.time() - t_e:.0f} s]", flush=True)
            paused += time.time() - t_e
            t_eval = time.time()
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
            "ppo": dataclasses.asdict(cfg), "reward": dataclasses.asdict(weights)}
    checkpoint.save(out / "latest.pt", actor, critic, args.preset, meta)
    if not (out / "best.pt").exists():
        checkpoint.save(out / "best.pt", actor, critic, args.preset, meta)
    log.close()
    summary = {"updates": update, "seconds": round(seconds), "battles": env.battles, "timeouts": env.timeouts,
               "battle_steps_per_s": round(decisions / seconds),
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
    actor = checkpoint.load_policy(path, device)
    untrained = checkpoint.load_policy(args.eval_past or checkpoint.RANDOM, device)
    res = {"training": summary, "checkpoint": str(path)}
    t = time.time()
    if args.armies == "generated":
        res["trained"] = evaluate.play(actor, past=untrained, device=device, limit_s=args.limit,
                                       generated=args.eval_generated, max_units=curriculum(args.curriculum)[-1][0])
    else:
        res["trained"] = evaluate.play(actor, per_scene=args.eval_per_scene, past=untrained, device=device,
                                       limit_s=args.limit)
    print(f"eval {path.name} {time.time() - t:.0f} s", flush=True)
    show(path.name, res["trained"])
    (out / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    paths = evaluate.record(actor, out / "replays", device=device, limit_s=args.limit, source=str(path))
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
    ap.add_argument("--preset", default="small", choices=sorted(model_config.PRESETS))
    ap.add_argument("--limit", type=float, default=3600.0, help="battle time limit, s (the defender wins at it)")
    ap.add_argument("--lr", type=float, default=ppo.PPOConfig.lr)
    ap.add_argument("--gamma", type=float, default=ppo.PPOConfig.gamma)
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
    ap.add_argument("--timeout", type=float, default=reward.Weights.timeout)
    ap.add_argument("--idle", type=float, default=reward.Weights.idle)
    ap.add_argument("--gold", type=float, default=reward.Weights.gold,
                    help="(enemy gold destroyed - own gold lost) / budget, per step")
    ap.add_argument("--rout-share", type=float, default=reward.Weights.rout_share,
                    help="a routing unit (it may rally) loses this share of the gold it has left")
    ap.add_argument("--hp", type=float, default=reward.Weights.hp, help="the old health trade (0: in gold)")
    ap.add_argument("--standing", type=float, default=reward.Weights.standing,
                    help="the old cost share that stopped standing (0: in gold)")
    ap.add_argument("--lord", type=float, default=reward.Weights.lord, help="the enemy lord's death - own lord's death")
    ap.add_argument("--lord-rout", type=float, default=reward.Weights.lord_rout,
                    help="> 0: in the lord term a shattered lord counts as dead, a routing one as this share of a death")
    ap.add_argument("--retarget", type=float, default=reward.Weights.retarget,
                    help="cost of switching an attack to another target while the old one stands")
    ap.add_argument("--mix", help='opponent shares as json, e.g. {"self": 0.2, "nearest": 0.4}')
    ap.add_argument("--critic-warmup", type=int, default=0, help="first updates train only the critic")
    ap.add_argument("--kind-temperature", type=float, default=1.0,
                    help="divide the starting actor's order-kind logits by this (> 1: softer, for exploration); "
                         "a --reference checkpoint gets the same")
    ap.add_argument("--anchor", type=float, default=ppo.PPOConfig.anchor, help="KL weight to the reference actor")
    ap.add_argument("--anchor-end", type=float, help="... at the end of the run (linear; default: no change)")
    ap.add_argument("--reference", help="the reference actor (default: --init); 'self': the network's own copy, "
                                        "renewed every --reference-every updates (a trust region, no fixed leash)")
    ap.add_argument("--reference-every", type=int, default=10, help="updates between renewals of --reference self")
    ap.add_argument("--anchor-roll", type=float, default=0.0,
                    help="seconds of training between renewals of the KL reference (--reference or --init): it "
                         "becomes the current actor (a rolling anchor; 0: fixed for the run)")
    ap.add_argument("--critic-init", help="take the critic from this checkpoint (default: --init's own)")
    ap.add_argument("--adv-norm", default=ppo.PPOConfig.adv_norm, choices=("batch", "role"),
                    help="normalise the side's advantage over the minibatch or over each role apart")
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--pool-extra", help="more past opponents for the pool (checkpoints, comma-separated)")
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--eval-per-scene", type=int, default=128)
    ap.add_argument("--eval-past", help="the network behind 'past' in the final evaluation (default: random.pt)")
    ap.add_argument("--eval-every", type=float, default=0,
                    help="minutes of training between evaluations that select best_eval.pt (0: none)")
    ap.add_argument("--eval-opponents", default="ai_like,nearest")
    ap.add_argument("--eval-battles", type=int, default=128, help="EVAL_SEEDS battles per opponent in those evaluations")
    ap.add_argument("--small", help="share:units - that share of every generated bank with at most `units` a side")
    ap.add_argument("--idle-tau", type=float, default=reward.Weights.idle_tau_s,
                    help="s: before its first damage the attacker's idle cost is --idle x (exp(t / this) - 1)")
    ap.add_argument("--idle-pause", type=float, default=reward.Weights.idle_pause_s,
                    help="s: after it, a step up every this many seconds without damage (0 before the first)")
    ap.add_argument("--idle-step", type=float, default=reward.Weights.idle_step,
                    help="... --idle x (exp(steps x this) - 1); new damage sets it back to 0")
    ap.add_argument("--idle-cap", type=float, default=reward.Weights.idle_cap,
                    help="the idle cost at most --idle x this")
    ap.add_argument("--idle-share", type=float, default=reward.Weights.idle_share,
                    help="0: idle while no unit fights or shoots; 1: x the share of the standing army (by cost) that does not")
    ap.add_argument("--idle-rate", type=float, default=reward.Weights.idle_rate,
                    help="0: any damage resets the idle timer; > 0: only a damage rate of at least this share of "
                         "the budget a minute (defender gold lost, mean over --idle-window)")
    ap.add_argument("--idle-window", type=float, default=reward.Weights.idle_window_s,
                    help="s: the time constant of that damage rate")
    ap.add_argument("--unit-credit", type=float, default=ppo.PPOConfig.unit_credit,
                    help="weight of each unit's own advantage beside the side's (0: the side's only)")
    ap.add_argument("--unit-credit-end", type=float,
                    help="... at the end of the run (linear from --unit-credit; default: no change)")
    ap.add_argument("--unit-value", type=float, default=ppo.PPOConfig.unit_value,
                    help="weight of the per-unit value loss beside the side's (its targets are x unit_scale)")
    ap.add_argument("--unit-gold", type=float, default=reward.Weights.unit_gold, help="per unit: its own gold trade")
    ap.add_argument("--unit-attrib", type=float, default=reward.Weights.unit_attrib,
                    help="in the unit's gold trade, what it destroyed: 1 the enemy's gold loss (routs, kills) split "
                         "among the units engaging it; 0 the old HP estimate")
    ap.add_argument("--flanked", type=float, default=reward.Weights.flanked,
                    help="per unit and decision: struck in the flank or rear")
    ap.add_argument("--missile-melee", type=float, default=reward.Weights.missile_melee,
                    help="per missile unit and decision in melee")
    ap.add_argument("--crowd", type=float, default=reward.Weights.crowd,
                    help="per unit and decision in a pile (more than 2 on one enemy while another flanks)")
    ap.add_argument("--idle-near", type=float, default=reward.Weights.idle_near,
                    help="per melee unit and decision standing by while a fellow within 60 m fights")
    ap.add_argument("--shirk", type=float, default=reward.Weights.shirk,
                    help="per melee unit and decision standing still out of melee while its side fights in melee "
                         "and an enemy is within --shirk-m")
    ap.add_argument("--shirk-m", type=float, default=reward.Weights.shirk_m)
    ap.add_argument("--shirk-side", type=float, default=reward.Weights.shirk_side,
                    help="the side (either role) pays this x the share of its army (by cost) shirking (as --shirk) "
                         "per decision; works at --unit-credit 0")
    ap.add_argument("--flank-attack", type=float, default=reward.Weights.flank_attack,
                    help="per unit and decision striking an enemy's flank or rear (bonus)")
    ap.add_argument("--unit-idle", type=float, default=reward.Weights.unit_idle,
                    help="per unit: x the attacker's idle m, per decision an attacking unit neither fights nor shoots")
    ap.add_argument("--lord-exposed", type=float, default=reward.Weights.lord_exposed,
                    help="per unit: per decision a lord fights in melee below --lord-exposed-hp of its health")
    ap.add_argument("--lord-exposed-hp", type=float, default=reward.Weights.lord_exposed_hp)
    ap.add_argument("--lord-lead", type=float, default=reward.Weights.lord_lead,
                    help="per unit: per decision a lord stands ahead of its line near the enemy (x 0-1: past "
                         "--lord-lead-m, full at twice it)")
    ap.add_argument("--lord-lead-m", type=float, default=reward.Weights.lord_lead_m)
    ap.add_argument("--lord-lead-near", type=float, default=reward.Weights.lord_lead_near,
                    help="m: only while a standing enemy is this near the lord")
    ap.add_argument("--lord-fall", type=float, default=reward.Weights.lord_fall,
                    help="per unit: the lord pays this x its fall in the step (standing 1, routing 1 - --lord-rout "
                         "(0.5 when 0), shattered or dead 0; a rally gives it back)")
    ap.add_argument("--neighbour", type=float, default=reward.Weights.neighbour,
                    help="+ this x the mean of own units' terms within 40 m")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--armies", default="scenes", choices=("scenes", "generated"),
                    help="the fixed arenas, or random armies of tools/nn/armies")
    ap.add_argument("--curriculum", default="5:0.3,10:0.6,19:1",
                    help="generated armies: max units a side, until this share of the time")
    ap.add_argument("--bank", type=int, default=2048, help="generated battles ready at once")
    ap.add_argument("--drills", type=float, default=0.0,
                    help="share of the battles that are drills (tools/nn/train/drills; e.g. 0.08), taken from the "
                         "other opponents in proportion; needs --armies generated")
    ap.add_argument("--drill-weights", help="the drills' shares of --drills as json, e.g. {\"pincer\": 1, \"kiting\": 2} "
                                            "(default: the verified drills, drills.READY, equally)")
    ap.add_argument("--drill-bank", type=int, default=256, help="battles of each drill ready at once (renewed with the bank)")
    ap.add_argument("--bank-refresh", type=float, default=5.0, help="minutes between new banks of armies")
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
