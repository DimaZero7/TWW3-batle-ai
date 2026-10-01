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
from tools.nn.train import checkpoint, evaluate, league, ppo, randomise, reward, rollout, scenes

SCRIPTED = ("nearest", "hold_shoot", "hold", "ai_like")


def compatible(path, preset):
    """The checkpoint exists and matches the current code's network (actor and critic)."""
    try:
        data = checkpoint.read(path)
        cfg = checkpoint.config_of(data)
        model_policy.Actor(cfg).load_state_dict(data["actor"])
        model_critic.Critic(cfg).load_state_dict(data["critic"])
        return data.get("preset") == preset
    except (FileNotFoundError, ValueError, RuntimeError, KeyError):
        return False


def networks(preset, device, start=None):
    if start is None:
        start = checkpoint.RANDOM
        if not compatible(start, preset):
            checkpoint.write_random(start, preset)
    data = checkpoint.read(start)
    cfg = checkpoint.config_of(data)
    actor, critic = model_policy.Actor(cfg), model_critic.Critic(cfg)
    actor.load_state_dict(data["actor"])
    critic.load_state_dict(data["critic"])
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


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = checkpoint.DIR / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    actor, critic = networks(args.preset, device, args.init)
    if args.kind_temperature != 1.0:
        soften_kind(actor, args.kind_temperature)
    reference = None
    if args.anchor:
        reference = checkpoint.load_policy(args.reference or args.init, device)
        for p in reference.parameters():
            p.requires_grad_(False)
    cfg = ppo.PPOConfig(lr=args.lr, gamma=args.gamma, epochs=args.epochs, minibatch=args.minibatch,
                        entropy=args.entropy, anchor=args.anchor)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr, eps=1e-5)
    weights = reward.Weights(order_change=args.order_cost, timeout=args.timeout, idle=args.idle, hp=args.hp,
                             standing=args.standing, lord=args.lord, retarget=args.retarget,
                             idle_ramp_s=args.idle_ramp, tempo=args.tempo, tempo_after_s=args.tempo_after)

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
    past_path = None
    mix = json.loads(args.mix) if args.mix else league.MIX
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
    best_eval, t_eval, paused = -1.0, time.time(), 0.0
    elog = (out / "eval_log.jsonl").open("w", encoding="utf-8", newline="\n")
    while time.time() - t0 - paused < args.minutes * 60:
        t_u = time.time()
        done_share = (t_u - t0 - paused) / (args.minutes * 60)
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
        # Schedules: the kind's entropy bonus and the KL to the reference go linearly from their start to
        # their end value over the run (e.g. exploration and the warm start's hold fade out).
        u_cfg = dataclasses.replace(step_cfg, entropy=schedule(args.entropy, entropy_end, done_share),
                                    anchor=schedule(args.anchor, anchor_end, done_share))
        st = ppo.update(actor, critic, opt, batch, u_cfg, train_policy=update >= args.critic_warmup,
                        reference=reference)
        del batch
        update += 1
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
               "lord_dead_own": round(lords["own"], 3), "lord_dead_enemy": round(lords["enemy"], 3),
               "switches_per_minute": round(lords["switches_per_minute"], 2),
               "orders_per_minute": round(env.orders_per_minute(), 2), "kinds": env.kinds(),
               "games": {k: g for k, (g, _, _) in stats.items()}, "win_rate": rates(stats),
               "seconds_per_battle": {k: round(s) for k, (_, _, s) in stats.items()}, "past": str(past_path.name)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if update % args.print_every == 0:
            print(f"u{update:4d} {row['seconds']:6.0f}s battles {env.battles:6d} (timeouts {env.timeouts:5d}) "
                  f"{row['battle_steps_per_s']:6d} st/s pl {st['policy_loss']:+.3f} vl {st['value_loss']:.4f} "
                  f"ent {st['entropy']:.2f}/{st['entropy_all']:.2f} kl {st['kl']:.3f} R {st['reward']:+.3f} "
                  f"anchor {st['anchor_kl']:.3f} orders/min {row['orders_per_minute']:5.1f} "
                  f"lords dead own {lords['own']:.2f} enemy {lords['enemy']:.2f} "
                  f"kinds " + " ".join(f"{k[:2]} {v:.2f}" for k, v in row["kinds"].items()), flush=True)
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
                  f"HP lost own {x['hp_own_lost']:.2f} enemy {x['hp_enemy_lost']:.2f}, timeouts {x['timeouts']:.2f}, "
                  f"lord dead own {x.get('lord_dead_own', float('nan')):.2f} enemy {x.get('lord_dead_enemy', float('nan')):.2f}")
        print(f"      kinds " + ", ".join(f"{k} {v:.2f}" for k, v in o["kinds"].items()), flush=True)
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="run")
    ap.add_argument("--init", help="start from this checkpoint (default: random.pt)")
    ap.add_argument("--minutes", type=float, default=5.0, help="wall time of training (the warm-up not counted)")
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
    ap.add_argument("--order-cost", type=float, default=reward.Weights.order_change)
    ap.add_argument("--timeout", type=float, default=reward.Weights.timeout)
    ap.add_argument("--idle", type=float, default=reward.Weights.idle)
    ap.add_argument("--hp", type=float, default=reward.Weights.hp)
    ap.add_argument("--standing", type=float, default=reward.Weights.standing)
    ap.add_argument("--lord", type=float, default=reward.Weights.lord, help="the enemy lord's death - own lord's death")
    ap.add_argument("--retarget", type=float, default=reward.Weights.retarget,
                    help="cost of switching an attack to another target while the old one stands")
    ap.add_argument("--mix", help='opponent shares as json, e.g. {"self": 0.2, "nearest": 0.4}')
    ap.add_argument("--critic-warmup", type=int, default=0, help="first updates train only the critic")
    ap.add_argument("--kind-temperature", type=float, default=1.0,
                    help="divide the starting actor's order-kind logits by this (> 1: softer, for exploration)")
    ap.add_argument("--anchor", type=float, default=ppo.PPOConfig.anchor, help="KL weight to the reference actor")
    ap.add_argument("--anchor-end", type=float, help="... at the end of the run (linear; default: no change)")
    ap.add_argument("--reference", help="the reference actor (default: --init)")
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
    ap.add_argument("--idle-ramp", type=float, default=reward.Weights.idle_ramp_s,
                    help="the attacker's idle cost grows x (1 + t / this) (s; 0: flat)")
    ap.add_argument("--tempo", type=float, default=reward.Weights.tempo,
                    help="the attacker's cost per decision past --tempo-after s of battle")
    ap.add_argument("--tempo-after", type=float, default=reward.Weights.tempo_after_s)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--armies", default="scenes", choices=("scenes", "generated"),
                    help="the fixed arenas, or random armies of tools/nn/armies")
    ap.add_argument("--curriculum", default="5:0.3,10:0.6,19:1",
                    help="generated armies: max units a side, until this share of the time")
    ap.add_argument("--bank", type=int, default=2048, help="generated battles ready at once")
    ap.add_argument("--bank-refresh", type=float, default=5.0, help="minutes between new banks of armies")
    ap.add_argument("--eval-generated", type=int, default=512, help="random battles per opponent (EVAL_SEEDS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    actor, summary, out = train(args)
    checkpoint.save(checkpoint.LATEST, actor, None, args.preset, checkpoint.meta(out / "latest.pt"))
    if not args.no_eval:
        final_eval(out / "latest.pt", args, summary, out)


if __name__ == "__main__":
    main()
