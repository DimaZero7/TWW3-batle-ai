"""The game-vs-sim gap card of an in-game gate (docs/en/training/workflow.md "Game-vs-sim gap card").

    DOCK_NAME=agent-gap bash tools/nn/dock.sh tools.ops.gapcard build/nn-gate/<time>
    DOCK_NAME=agent-gap bash tools/nn/dock.sh tools.ops.gapcard build/nn-gate/<time> --profile fatigue,lords --copies 8
    .venv/Scripts/python -m tools.ops.gapcard build/nn-gate/<time> --from-json --profile routs   # reprint, no torch

Replays the gate's battles in the simulator from the same recorded starts (tools/nn/train/gapsim.py:
the gate's checkpoint against ai_like at the game's cadence, --copies copies each, on the CPU) and
measures the game's recording and every simulated copy with the same code (tools/nn/battle_metrics.py,
up to the game battle's end time). One table: a row per metric of the mandatory set and the chosen
profiles (tools/nn/train/profiles.py GAP; default full), per battle "game / sim mean", pooled game
mean / sim mean and their difference. A mark when game and sim differ beyond the noise:

    one battle:  |game - sim| > max(Z x sd x sqrt(1 + 1 / n), TOL)   sd: the spread of the n copies (the
                 noise of one battle, the game's one battle counted with it)
    pooled:      |mean game - mean sim| > max(Z x sqrt(sum sd_b^2 (1 + 1 / n_b)) / k, TOL)   over k battles

TOL: the smallest difference that matters for the metric (a share 0.05, the trade 0.05, ...). Writes
<gate folder>/gapcard.json (the per-battle numbers; --from-json reprints from it on the host). Wall time
on 8 CPU cores: ~4-5 minutes for a 4-battle gate at 8 copies (20261005-161910: 182-246 s of simulation).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn import battle_metrics as M
from tools.nn.train import profiles

Z = 2.0
TOL = {"win": 0.25, "length_s": 30.0, "abilities_own": 0.5, "ability_first_s": 30.0,
       "lord_lost_own": 0.002, "lord_lost_enemy": 0.002, "routs_own": 0.15, "routs_enemy": 0.15,
       "lap_enemy": 15.0}
TOL_DEFAULT = 0.05          # shares and the trade


def stats(xs):
    """(mean, sd, n) of the numbers in xs that are not None (sd None below 2)."""
    v = [x for x in xs if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None), len(v)


def beyond(diff, noise, key):
    """Whether a difference is beyond the noise (Z x noise, noise None: TOL alone) and TOL."""
    if diff is None:
        return False
    tol = TOL.get(key, TOL_DEFAULT)
    return abs(diff) > max(tol, Z * noise if noise is not None else 0.0)


def compare(battles, chosen):
    """battles: [{"battle", "game": metrics, "sims": [metrics]}] -> [{key, title, fmt, cells: [{game, sim,
    sd, n, flag}], pooled: {game, sim, diff, noise, k, flag}}] for the rows of the mandatory set and chosen."""
    out = []
    for key, _, title, fmt in M.rows_for(chosen):
        cells, diffs, var = [], [], []
        for b in battles:
            g = (b.get("game") or {}).get(key)
            mean, sd, n = stats([s.get(key) for s in b.get("sims") or []])
            noise = sd * math.sqrt(1 + 1 / n) if sd is not None else None
            d = None if g is None or mean is None else mean - g
            cells.append({"game": g, "sim": mean, "sd": sd, "n": n, "flag": beyond(d, noise, key)})
            if d is not None:
                diffs.append((g, mean))
                var.append(noise ** 2 if noise is not None else None)
        pooled = {"game": None, "sim": None, "diff": None, "noise": None, "k": len(diffs), "flag": False}
        if diffs:
            k = len(diffs)
            pooled["game"] = float(np.mean([g for g, _ in diffs]))
            pooled["sim"] = float(np.mean([s for _, s in diffs]))
            pooled["diff"] = pooled["sim"] - pooled["game"]
            known = [v for v in var if v is not None]
            pooled["noise"] = math.sqrt(sum(known)) / k if known else None
            pooled["flag"] = beyond(pooled["diff"], pooled["noise"], key)
        out.append({"key": key, "title": title, "fmt": fmt, "cells": cells, "pooled": pooled})
    return out


def _f(v, fmt):
    return "-" if v is None else fmt.format(v)


def table(rows, battles, head=""):
    """Markdown lines of compare()'s rows; then the flagged rows in words."""
    heads = [f"b{b['battle']} {b.get('role', '')[:3]} game / sim".replace("  ", " ") for b in battles]
    lines = ([head, ""] if head else []) + [
        "| metric | " + " | ".join(heads) + " | pooled game / sim | sim - game |",
        "|---" * (len(heads) + 3) + "|"]
    flagged = []
    for r in rows:
        f = r["fmt"]
        cells = [f"{_f(c['game'], f)} / {_f(c['sim'], f)}{' !' if c['flag'] else ''}" for c in r["cells"]]
        p = r["pooled"]
        diff = "-" if p["diff"] is None else (f.replace(":", ":+") if "+" not in f else f).format(p["diff"])
        lines.append(f"| {r['title']} | " + " | ".join(cells) + f" | {_f(p['game'], f)} / {_f(p['sim'], f)} | "
                     f"{diff}{' *' if p['flag'] else ''} |")
        if p["flag"]:
            noise = "" if p["noise"] is None else f", noise {f.replace('+', '').format(Z * p['noise'])}"
            flagged.append(f"{r['title']}: game {_f(p['game'], f)}, sim {_f(p['sim'], f)}{noise}")
    lines += ["", "! one battle beyond its copies' noise; * pooled beyond noise", "",
              "GAME vs SIM differ beyond noise (pooled): " + ("; ".join(flagged) if flagged else "none")]
    return lines


def run(gate_dir, chosen, copies, seed, threads, ckpt=None, only=None, log=print):
    """Plays the replays and measures -> the card's document {gate, checkpoint, copies, profile, battles, wall_s}."""
    import torch
    from tools.nn.train import gapsim
    torch.set_num_threads(threads)
    t0 = time.time()
    doc, found = gapsim.gate_battles(gate_dir, only=only)
    if not found:
        raise SystemExit(f"no battles with a recording in {gate_dir}")
    ckpt = ckpt or doc.get("checkpoint")
    starts = [gapsim.start_of(d) for _, d in found]
    limit = float(doc.get("timeout_s") or 3600)
    sims = gapsim.play(starts, ckpt, copies=copies, seed=seed, greedy=bool(doc.get("greedy")), limit_s=limit, log=log)
    battles = []
    for (row, d), (g, _), copies_ in zip(found, starts, sims):
        winner = {"net": 1, "game_ai": 2}.get(row.get("winner"), 0)
        game = M.game_battle(d, winner, chosen=chosen)
        cut = float(g.t[-1])
        battles.append({"battle": row["battle"], "role": row.get("role"), "run": d.name, "game": game,
                        "sims": [M.measure(s["battle"], s["winner"], cut, s["abilities"], chosen, orders=s["orders"])
                                 for s in copies_]})
    return {"gate": str(gate_dir), "checkpoint": str(ckpt), "copies": copies, "seed": seed,
            "profile": profiles.text(chosen, profiles.GAP), "battles": battles, "wall_s": round(time.time() - t0)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", help="build/nn-gate/<time>")
    ap.add_argument("--profile", default=profiles.FULL,
                    help="metric profiles besides the mandatory set: full (default), mandatory or a comma list of "
                         + ", ".join(profiles.GAP))
    ap.add_argument("--copies", type=int, default=8, help="simulated copies of each battle")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--checkpoint", help="the network (default: the gate's)")
    ap.add_argument("--battles", default="", help="only these battle numbers, e.g. 1,3")
    ap.add_argument("--from-json", action="store_true", help="reprint from <gate>/gapcard.json (no simulator)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        chosen = profiles.parse(args.profile, profiles.GAP)
    except ValueError as e:
        ap.error(str(e))
    gate = Path(args.gate)
    if not gate.is_absolute() and not gate.exists():
        gate = project.ROOT / gate
    path = gate / "gapcard.json"
    if args.from_json:
        doc = json.loads(path.read_text(encoding="utf-8"))
    else:
        only = {int(x) for x in args.battles.split(",") if x.strip()} or None
        doc = run(gate, chosen, args.copies, args.seed, args.threads, args.checkpoint, only,
                  log=lambda s: print(s, flush=True))
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8", newline="\n")
    rows = compare(doc["battles"], chosen)
    head = (f"gap card: {gate.name}, {Path(doc['checkpoint']).parent.name}/{Path(doc['checkpoint']).name} vs "
            f"ai_like in the simulator, {doc['copies']} copies a battle, game cadence; "
            f"profile {profiles.text(chosen, profiles.GAP)}; measured up to the game's end time")
    print("\n".join(table(rows, doc["battles"], head)))
    if not args.from_json:
        print(f"\nwritten {path}; wall {doc['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
