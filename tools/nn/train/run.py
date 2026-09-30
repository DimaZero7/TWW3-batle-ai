"""Train the battle network with PPO in the simulator (docs/en/training/training.md).

    bash tools/nn/dock.sh tools.nn.train.run --minutes 5          # in the container, on the GPU

The network starts from build/nn-train/random.pt (written if missing), plays `--battles` battles
at once against itself, its past versions and the scripted opponents, and learns after every
`--steps` decisions. Writes (build/ is not in Git):

    build/nn-train/latest.pt        the network (tools/nn/train/checkpoint.py)
    build/nn-train/pool/v*.pt       past versions, the opponents of self-play
    build/nn-train/log.jsonl        one line per update: losses, entropy, reward, win rates
    build/nn-train/eval.json        the final evaluation (tools/nn/train/evaluate.py)
    build/nn-train/replays/*/       a few battles of the final network, like the game's recordings
"""
import argparse
import dataclasses
import json
import sys
import time

import torch

from tools.nn.model import config as model_config
from tools.nn.model import critic as model_critic
from tools.nn.model import policy as model_policy
from tools.nn.train import checkpoint, evaluate, league, ppo, randomise, reward, rollout, scenes


def compatible(path, preset):
    """The checkpoint exists and matches the current code's network."""
    try:
        data = checkpoint.read(path)
        actor = model_policy.Actor(checkpoint.config_of(data))
        actor.load_state_dict(data["actor"])
        return data.get("preset") == preset and "critic" in data
    except (FileNotFoundError, ValueError, RuntimeError, KeyError):
        return False


def networks(preset, device, start=checkpoint.RANDOM):
    if not compatible(start, preset):
        checkpoint.write_random(start, preset)
    data = checkpoint.read(start)
    cfg = checkpoint.config_of(data)
    actor, critic = model_policy.Actor(cfg), model_critic.Critic(cfg)
    actor.load_state_dict(data["actor"])
    critic.load_state_dict(data["critic"])
    return actor.to(device).eval(), critic.to(device).eval()


def rates(stats):
    return {k: round(w / g, 3) for k, (g, w, _) in stats.items() if g}


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = checkpoint.DIR
    out.mkdir(parents=True, exist_ok=True)
    actor, critic = networks(args.preset, device)
    cfg = ppo.PPOConfig(lr=args.lr)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr, eps=1e-5)
    weights = reward.Weights(order_change=args.order_cost)
    pool = league.Pool(checkpoint.POOL, size=args.pool)
    pool.add(checkpoint.RANDOM)
    past = model_policy.Actor(actor.cfg).to(device).eval()
    past_path = None
    lay = league.layout(args.battles, len(scenes.SCENES))
    env = rollout.Battles(lay, scenes.SCENES, device, rollout.params_with_limit(args.limit), randomise.Spread(),
                          weights, seed=args.seed)
    log = (out / "log.jsonl").open("w", encoding="utf-8", newline="\n")
    print(f"battles {env.B} (learner rows {env.R}), slots {env.N}, steps per update {args.steps}, "
          f"model {args.preset} {model_policy.parameters(actor) / 1e6:.2f} M actor", flush=True)

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
    print(f"warm-up {time.time() - t_warm:.0f} s", flush=True)
    t0 = time.time()
    update, decisions, total = 0, 0, {}
    while time.time() - t0 < args.minutes * 60:
        t_u = time.time()
        batch = rollout.collect(env, actor, critic, args.steps)
        t_c = time.time()
        st = ppo.update(actor, critic, opt, batch, cfg)
        update += 1
        decisions += env.B * args.steps
        stats = env.take_stats()
        for k, (g, w, s) in stats.items():
            G, W, S = total.get(k, (0, 0, 0.0))
            total[k] = (G + g, W + w, S + s * g)
        row = {"update": update, "seconds": round(time.time() - t0, 1), "battles": env.battles,
               "battle_steps_per_s": round(env.B * args.steps / (time.time() - t_u)),
               "collect_s": round(t_c - t_u, 2), "update_s": round(time.time() - t_c, 2),
               **{k: round(v, 4) for k, v in st.items()},
               "orders_per_minute": round(env.orders_per_minute(), 2), "kinds": env.kinds(),
               "games": {k: g for k, (g, _, _) in stats.items()}, "win_rate": rates(stats),
               "past": str(past_path.name)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if update % args.print_every == 0:
            print(f"u{update:4d} {row['seconds']:6.0f}s battles {env.battles:6d} {row['battle_steps_per_s']:6d} st/s "
                  f"pl {st['policy_loss']:+.3f} vl {st['value_loss']:.4f} ent {st['entropy']:.2f}/{st['entropy_all']:.2f} "
                  f"kl {st['kl']:.3f} "
                  f"R {st['reward']:+.3f} orders/min {row['orders_per_minute']:5.1f} win {row['win_rate']}", flush=True)
        if update % args.snapshot_every == 0:
            meta = {"update": update, "battles": env.battles, "seconds": row["seconds"]}
            pool.save(actor, None, args.preset, meta)
            checkpoint.save(checkpoint.LATEST, actor, critic, args.preset, meta)
            pick_past()
        elif update % 2 == 0:
            pick_past()
    seconds = time.time() - t0
    meta = {"update": update, "battles": env.battles, "seconds": round(seconds), "decisions": decisions,
            "battles_at_once": env.B, "steps_per_update": args.steps, "limit_s": args.limit,
            "ppo": dataclasses.asdict(cfg), "reward": dataclasses.asdict(weights)}
    checkpoint.save(checkpoint.LATEST, actor, critic, args.preset, meta)
    log.close()
    summary = {"updates": update, "seconds": round(seconds), "battles": env.battles,
               "battle_steps_per_s": round(decisions / seconds),
               "win_rate_all": {k: round(w / g, 3) for k, (g, w, _) in total.items() if g},
               "games": {k: g for k, (g, _, _) in total.items()},
               "mean_battle_s": {k: round(s / g) for k, (g, _, s) in total.items() if g}}
    print("trained:", json.dumps(summary), flush=True)
    return actor, summary


def final_eval(actor, args, summary):
    device = torch.device(args.device)
    untrained = checkpoint.load_policy(checkpoint.RANDOM, device)
    res = {"training": summary}
    t = time.time()
    res["trained"] = evaluate.play(actor, per_scene=args.eval_per_scene, past=untrained, device=device,
                                   limit_s=args.limit)
    print(f"eval trained {time.time() - t:.0f} s", flush=True)
    t = time.time()
    res["untrained"] = evaluate.play(untrained, opponents=("nearest", "hold_shoot", "hold"),
                                     per_scene=args.eval_per_scene, device=device, limit_s=args.limit)
    print(f"eval untrained {time.time() - t:.0f} s", flush=True)
    (checkpoint.DIR / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    for who in ("trained", "untrained"):
        r = res[who]
        print(f"{who}: orders/min {r['orders_per_minute']:.1f}, kinds "
              + ", ".join(f"{k} {v:.2f}" for k, v in r["kinds"].items()))
        for name, o in r["by_opponent"].items():
            print(f"  v {name:10} win {o['win_rate']:.3f} ({o['wins']}/{o['games']}), {o['seconds']:.0f} s, "
                  f"HP lost own {o['hp_own_lost']:.2f} enemy {o['hp_enemy_lost']:.2f}, timeouts {o['timeouts']:.2f}")
            print("      " + "; ".join(f"{k} {v['win_rate']:.2f}" for k, v in r["by_scene"][name].items()))
    paths = evaluate.record(actor, device=device, limit_s=args.limit, source=str(checkpoint.LATEST))
    print(f"replays: {len(paths)} in {evaluate.REPLAYS}", flush=True)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=5.0, help="wall time of training (the warm-up not counted)")
    ap.add_argument("--battles", type=int, default=1024, help="battles at once")
    ap.add_argument("--steps", type=int, default=16, help="decisions per battle between updates")
    ap.add_argument("--preset", default="small", choices=sorted(model_config.PRESETS))
    ap.add_argument("--limit", type=float, default=900.0, help="battle time limit, s (the defender wins at it)")
    ap.add_argument("--lr", type=float, default=ppo.PPOConfig.lr)
    ap.add_argument("--order-cost", type=float, default=reward.Weights.order_change)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--eval-per-scene", type=int, default=512)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    actor, summary = train(args)
    if not args.no_eval:
        final_eval(actor, args, summary)


if __name__ == "__main__":
    main()
