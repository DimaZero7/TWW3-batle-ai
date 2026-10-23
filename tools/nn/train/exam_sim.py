"""The exam's proxy in the simulator (docs/en/training/training.md "The exam's proxy"): the network against a fixed
network (default itS27_4, the exam's opponent in the game) on the exam's 6 seeds (armies v2), both armies of each,
R battles each, our network always defending (as nn-arena), in the fighters lab (as the training, run.py --fighters).

    python -m tools.nn.train.exam_sim build/nn-train/test5/g23_general/m40.pt            # in the container
    python -m tools.nn.train.exam_sim <ckpt> --R 8 --enemy build/v2/itS27_4.pt --out build/exam_sim

A seed's two exam battles: base = our network on side 1 (the generator's army 1), side 2 attacks; swap = our network on
side 2 (army 2), side 1 attacks. The bank holds every seed 2R times in a row (scenes.Generated: side 1 attacks at even
places) and evaluate.play(defend=True) puts our network on each battle's defending side: R swap and R base battles a
seed, 12 R battles in all. The enemy's orders are sampled (as the game's Brain), so are ours; limit 3600 s (the defender
wins at it); the cadence the game's (or the training's in test5).

Read like the game's exam (12 battles, 6 pairs): per seed the win share on each side; the seed's pair 2-0 when both
shares are above 0.5, 0-2 when both are below, else 1-1; the expected 2-0 / 0-2 of one exam (sum over the seeds of
p_base p_swap and (1 - p_base)(1 - p_swap): steadier than the pairs by the majority); wins per 12 battles; the trade
(enemy gold destroyed - own gold lost) / budget (evaluate.play's per-battle "trade"), its mean with 95 %.
Calibration (build/diag_exam_sim/notes.md, fighters, R 4): the order of the game's exam g19 m20 = g23 m40 > g22 m30 >
the mirror itS27_4 against itself; the plain simulator put g23 at g22's level. The mirror's "zero": ~0.54-0.56 wins.

Plain at import (numpy only): tools/nn/train/skill.py reads summary() / cell() on the host; play() imports torch.
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SEEDS = (1000800014, 1000800163, 1000800092, 1000800042, 1000800127, 1000800133)   # the game's exam (nn-arena)
ENEMY = "build/v2/itS27_4.pt"
R = 8                     # battles per seed and side (12 R in all): ~90 tell a trade gap of 0.065 at 95 %
ARMIES = "v2"
MAX_UNITS = 19
LIMIT_S = 3600.0
Z95 = 1.959964
PAIRS = ("2-0", "1-1", "0-2")


def battle_seeds(seeds=SEEDS, r=R):
    """The bank's seeds: every seed 2r times in a row, so its battles start at an even place (side 1 attacks there):
    with our network defending, the even places are its swap battles, the odd ones its base battles."""
    if r < 1:
        raise ValueError(f"R {r} < 1")
    return [int(s) for s in seeds for _ in range(2 * r)]


def pair_of(p_base, p_swap):
    """The seed's pair as the exam reads it: 2-0 when both win shares are above 0.5, 0-2 when both are below, else
    1-1."""
    if p_base > 0.5 and p_swap > 0.5:
        return "2-0"
    if p_base < 0.5 and p_swap < 0.5:
        return "0-2"
    return "1-1"


def summary(won, side, seed, trade, timeouts=None):
    """The proxy's numbers of per-battle lists (won, our side 1 = base / 2 = swap, seed, trade) -> {battles, win,
    wins12 (wins in an exam of 12), pairs [2-0, 1-1, 0-2], expected [2-0, 0-2], trade {value, se, lo, hi},
    by_seed {seed: {base, swap (win shares), n_base, n_swap, trade_base, trade_swap, pair}}}. A seed without battles on
    both sides has no pair."""
    won = np.asarray(won, dtype=float)
    side, seed, trade = np.asarray(side), np.asarray(seed), np.asarray(trade, dtype=float)
    by_seed, counts, e20, e02 = {}, dict.fromkeys(PAIRS, 0), 0.0, 0.0
    for s in dict.fromkeys(seed.tolist()):
        b, w = (seed == s) & (side == 1), (seed == s) & (side == 2)
        row = {"n_base": int(b.sum()), "n_swap": int(w.sum())}
        if b.any() and w.any():
            pb, ps = float(won[b].mean()), float(won[w].mean())
            row.update(base=round(pb, 4), swap=round(ps, 4), trade_base=round(float(trade[b].mean()), 4),
                       trade_swap=round(float(trade[w].mean()), 4), pair=pair_of(pb, ps))
            counts[row["pair"]] += 1
            e20 += pb * ps
            e02 += (1 - pb) * (1 - ps)
        by_seed[str(s)] = row
    n = len(won)
    se = float(trade.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    mean = float(trade.mean()) if n else 0.0
    out = {"battles": n, "win": round(float(won.mean()), 4) if n else None,
           "wins12": round(float(won.mean()) * 12, 2) if n else None,
           "pairs": [counts[p] for p in PAIRS], "expected": [round(e20, 2), round(e02, 2)],
           "trade": {"value": round(mean, 4), "se": round(se, 4), "lo": round(mean - Z95 * se, 4),
                     "hi": round(mean + Z95 * se, 4)},
           "by_seed": by_seed}
    if timeouts is not None:
        out["timeouts"] = round(float(timeouts), 4)
    return out


def title(s):
    """The table row's title for a summary s (skill.rows)."""
    s = s or {}
    enemy = Path(s.get("enemy") or ENEMY).stem
    lab = "fighters" if s.get("fighters", True) else "plain sim"
    return (f"exam vs {enemy} ({lab}, {len(s.get('seeds') or SEEDS)} seeds x 2 sides x R {s.get('R', R)}, we defend): "
            f"pairs 2-0/1-1/0-2 (expected 2-0/0-2), wins /12, trade ± 95%")


def cell(s):
    """'2/4/0 (exp 2.4/0.1), 8.3/12, +0.082 ± 0.040' of a summary s; '-' without one."""
    if not s or not s.get("battles"):
        return "-"
    a, b, c = s["pairs"]
    e20, e02 = s["expected"]
    t = s["trade"]
    return (f"{a}/{b}/{c} (exp {e20:.1f}/{e02:.1f}), {s['wins12']:.1f}/12, "
            f"{t['value']:+.3f} ± {Z95 * t['se']:.3f}")


def line(s, name=""):
    """One line of a result: the cell, the per-seed base / swap win shares, the wall time."""
    seeds = " ".join(f"{k[-3:]} {v['base']:.2f}/{v['swap']:.2f}" for k, v in (s.get("by_seed") or {}).items()
                     if "base" in v)
    return (f"exam {name} vs {Path(s.get('enemy') or ENEMY).stem} R {s.get('R')}: {cell(s)}; base/swap by seed {seeds}; "
            f"timeouts {s.get('timeouts', 0):.2f}; {s.get('seconds', 0):.0f} s")


@contextlib.contextmanager
def lab(on, seed=1):
    """The fighters lab for the exam's battles -> evaluate.play's `fighters` (None: the plain simulator). Inside a
    training with --fighters (test5's points) the training's hook is in place and serves; else installed here."""
    if not on:
        yield None
        return
    from tools.nn.fighters import hook as fighters_hook
    from tools.nn.train import men
    ctx = (contextlib.nullcontext() if fighters_hook.HOOK is not None
           else fighters_hook.installed(seed=seed, shape="lazy", adopt=False, outside=False))
    with ctx:
        yield (lambda: men.Men(M=None, seed=seed))


def resolve(path):
    """A project path (relative to the repository) as a file."""
    from tools import config as project
    p = Path(path)
    return p if p.is_absolute() else project.ROOT / p


def play(actor, enemy=ENEMY, r=R, seeds=SEEDS, device="cpu", cadence=None, fighters=True, seed=1, limit_s=LIMIT_S):
    """The proxy of one network (an actor) -> summary() + {enemy, R, seeds, armies, fighters, seconds, lord_dead_own, per_battle:
    {won, side, seed, trade, margin, destroyed, lost, budget}}. enemy: a checkpoint path or an actor. The draws (the
    orders, the latency) come from their own random state: the caller's (a training's) is left as it was."""
    import torch

    from tools.nn.train import cadence as cad
    from tools.nn.train import checkpoint, evaluate
    t0 = time.time()
    if isinstance(enemy, (str, Path)):
        name = str(enemy)
        past = checkpoint.load_policy(resolve(enemy), device)
    else:
        name, past = "actor", enemy
    bank = battle_seeds(seeds, r)
    dev = torch.device(device)
    cuda = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda), lab(fighters, seed) as men:
        torch.manual_seed(seed)
        res = evaluate.play(actor, opponents=("past",), past=past, device=device, limit_s=limit_s, seed=seed,
                            generated=len(bank), max_units=MAX_UNITS, paired=False, baseline=False,
                            cadence=cadence or cad.GAME, measure=(), cuda_graph=not fighters, fighters=men,
                            seeds=bank, armies=ARMIES, defend=True)
    b = res["battles"]["past"]
    out = summary(b["won"], b["side"], b["seed"], b["trade"], res["by_opponent"]["past"].get("timeouts"))
    lord = res["by_opponent"]["past"].get("lord_dead_own")         # own lord deaths in its battles (test5's mandatory set)
    if lord is not None:
        out["lord_dead_own"] = round(float(lord), 4)
    out.update(enemy=name, R=int(r), seeds=[int(s) for s in seeds], armies=ARMIES, fighters=bool(fighters),
               seconds=round(time.time() - t0),
               per_battle={k: b[k] for k in ("won", "side", "seed", "trade", "margin", "destroyed", "lost", "budget")})
    return out


def main(argv=None):
    import torch

    from tools.nn.train import cadence as cad
    from tools.nn.train import checkpoint
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="the network to check (a checkpoint: test5's m<minute>.pt, a run's latest.pt)")
    ap.add_argument("--R", type=int, default=R, help="battles per seed and side (12 R in all)")
    ap.add_argument("--enemy", default=ENEMY, help="the opponent's checkpoint (default: the exam's itS27_4)")
    ap.add_argument("--plain", action="store_true", help="the plain simulator (no fighters lab; the calibration "
                                                         "says it does not follow the game)")
    ap.add_argument("--out", help="a folder for the result: exam_<checkpoint>_R<R>.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cad.add_args(ap)
    args = ap.parse_args(argv)
    actor = checkpoint.load_policy(resolve(args.checkpoint), args.device)
    res = play(actor, args.enemy, args.R, device=args.device, cadence=cad.of_args(args), fighters=not args.plain)
    res["checkpoint"] = args.checkpoint
    name = Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).stem
    print(line(res, name), flush=True)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"exam_{name.replace('/', '_')}_R{args.R}{'_plain' if args.plain else ''}.json"
        path.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
        print(f"written: {path}", flush=True)


if __name__ == "__main__":
    main()
