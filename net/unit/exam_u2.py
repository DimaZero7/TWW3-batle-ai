"""The exam of U2 (u2.py) by the method (METHOD.md recipe steps 3 and 5): one exam sat twice - before any training by
the scripts, after it by the network. The players: unit (the unit's script under 'hold': the needed move), stand
(the default), random (the bad one), straight (the script before its after_melee reflex) - net/unit/scripts.py - and
the unit network (--ckpt; run2, weights/unit/mixmap_run2) sampling (run2) and greedy (run2g). Each on --seeds seeds (a
seed: new skirmishes or battles and the simulator's draws); every player plays the same ones, so the comparison is
paired: the mean of the per-battle differences, its error from their spread (rank); the per-seed sd in the table is
for reading only.

    bash ops/dock.sh net.unit.exam_u2 drill --device cuda --seeds 4 --n 256 --out tmp/u2/drill.json
    bash ops/dock.sh net.unit.exam_u2 general --device cuda --seeds 3 --n 256 --out tmp/u2/general.json

drill: the skirmishes of u2.specs (the exam's seeds from u2.SEED); general: stage 0's general battles of the unit level
(net/stage0/battle.play_units: no level above - each unit's frame its slot of the deployment, the same for every
player; the order gate on; the enemy a script: a spec's formula chain enemy drawn anew among the scripts by the
stage's shares). The judge is U2's usefulness - met head-on (head_on); the rank check: unit > stand > random; in the
general battles each player against stand (paired_vs_stand: trade - the stack 'not worse than stand').
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

PLAYERS = ("unit", "stand", "random", "straight", "run2", "run2g")
DRILL_KEYS = ("head_on", "contact", "in_frame_contact", "exec", "exec_all", "reached", "reach_s", "orders_min",
              "raw_orders_min", "t_contact", "t_end")
GEN_KEYS = ("head_on", "frame_exec", "game_orders_min", "raw_orders_min", "orders_min", "flank_taken", "gap_m",
            "trade", "win", "t_contact", "t_end")


def _mean(v):
    v = v.float()
    v = v[torch.isfinite(v)]
    return float(v.mean()) if v.numel() else float("nan")


def _duty(a):
    """--gpu-duty: the GPU busy at most this share of the time (net/unit/train.Duty)."""
    from net.unit.train import Duty
    return Duty(a.device, a.gpu_duty)


def _player(name, ckpt, device):
    from net.unit import scripts as SC
    if name in SC.PLAYERS:
        return SC.player(name), False
    from net.tactic import ckpt as TK
    return TK.unit_network(device, ckpt), name.endswith("g")


def summary(rows, keys):
    """{key: (mean over the seeds, sd over the seeds)} of rows [{key: a seed's mean}]."""
    out = {}
    for k in keys:
        xs = [r[k] for r in rows if k in r and r[k] == r[k]]
        if not xs:
            continue
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
        out[k] = (round(m, 4), round(sd, 4))
    return out


def rank(per, key="head_on", order=("unit", "stand", "random")):
    """Each neighbour pair of the order, PAIRED by skirmish / battle (the players played the same ones): the mean of the
    per-battle differences a - b over the battles where both are finite, its two errors (2 x sd / sqrt(n) of the
    differences), passed - the difference larger (METHOD.md recipe 4: 'лучше' - a paired difference over two errors);
    per {player: {key: [B] all seeds' battles in one order}}."""
    out = []
    for a, b in zip(order, order[1:]):
        if a not in per or b not in per or key not in per[a] or key not in per[b]:
            continue
        d = per[a][key].float() - per[b][key].float()
        d = d[torch.isfinite(d)]
        if d.numel() < 2:
            continue
        m, two = float(d.mean()), 2 * float(d.std()) / math.sqrt(d.numel())
        out.append({"pair": f"{a} - {b}", "diff": round(m, 4), "two_errors": round(two, 4), "n": int(d.numel()),
                    "passed": bool(m > two), "not_worse": bool(m > -two)})
    return out


def drill(a, P):
    from net.unit import u2
    table, by_case, per = {}, {}, {}
    for name in a.players.split(","):
        rows, allm, cases = [], {}, []
        wall = 0.0
        pl, greedy = _player(name, a.ckpt, a.device)
        for s in range(a.seeds):
            sp = u2.specs(a.n, torch.Generator().manual_seed(u2.SEED + s))
            if hasattr(pl, "gen"):
                pl.gen, pl.seed = None, s          # (the random script: its own draws a seed)
            torch.manual_seed(1000 + s)
            t0 = time.time()
            duty = _duty(a)
            duty.start()
            m = u2.play(sp, pl, P, a.device, greedy=greedy)
            duty.end()                                  # (--gpu-duty: u2.play has no duty of its own - per seed)
            wall += time.time() - t0
            rows.append({k: _mean(m[k]) for k in DRILL_KEYS if k in m})
            for k, v in m.items():
                allm.setdefault(k, []).append(v)
            cases += [("far" if x["far"] else "near") + ("_enemy" if x["enemy"] else "_none") for x in sp]
            print(json.dumps({"player": name, "seed": s, "wall_s": round(time.time() - t0, 1), **{
                k: round(v, 4) for k, v in rows[-1].items()}}), flush=True)
        allm = {k: torch.cat(v) for k, v in allm.items()}
        table[name] = {"mean": summary(rows, DRILL_KEYS), "seeds": rows, "wall_s": round(wall, 1),
                       "skirmishes_min": round(a.n * a.seeds / max(wall, 1e-6) * 60, 1)}
        by_case[name] = {c: {k: round(_mean(allm[k][torch.tensor([x == c for x in cases])]), 4)
                             for k in ("head_on", "contact", "exec", "reached", "orders_min")}
                         for c in sorted(set(cases))}
        per[name] = allm
    return {"what": "drill", "n": a.n, "seeds": a.seeds, "table": table, "by_case": by_case,
            "rank_head_on": rank(per), "rank_exec": rank(per, "exec")}


def general(a, P):
    from net.stage0 import battle as BA
    from net.stage0 import cfg as scfg
    from net.stage0 import specs as SP
    from net.tactic import ckpt as TK
    unit_net = TK.unit_network(a.device, a.ckpt)
    shares = {k: v for k, v in scfg()["general"]["enemies"].items() if k != "chain"}
    table, per = {}, {}
    for name in a.players.split(","):
        rows, wall = [], 0.0
        by_map, allm = {}, {}
        for s in range(a.seeds):
            sp = SP.generals(a.n, SP.seeded(9100 + s))
            rnd = random.Random(9200 + s)
            for x in sp:
                if x["enemy"] == "chain":                  # (no formula: a script by the stage's shares)
                    x["enemy"] = rnd.choices(list(shares), weights=list(shares.values()))[0]
                if a.limit:
                    x["limit"] = a.limit
            if name.startswith("run2"):
                level, greedy = None, name.endswith("g")
            else:
                level, greedy = _player(name, a.ckpt, a.device)[0], False
                if hasattr(level, "seed"):
                    level.seed = s
            torch.manual_seed(2000 + s)
            t0 = time.time()
            m, stt = BA.play_units(sp, level, unit_net, P, a.device, gate=True, unit_greedy=greedy, duty=_duty(a))
            wall += time.time() - t0
            rows.append({k: _mean(m[k]) for k in GEN_KEYS if k in m})
            for k in GEN_KEYS:
                if k in m:
                    allm.setdefault(k, []).append(m[k])
            for mp in ("flat", "ksl_plains_01"):
                sel = torch.tensor([x["map"] == mp for x in sp])
                by_map.setdefault(mp, []).append({k: _mean(m[k][sel]) for k in ("head_on", "frame_exec",
                                                                                 "game_orders_min") if k in m})
            print(json.dumps({"player": name, "seed": s, "wall_s": round(time.time() - t0, 1),
                              "sim_wall_s": stt["wall_s"], **{k: round(v, 4) for k, v in rows[-1].items()}}),
                  flush=True)
        table[name] = {"mean": summary(rows, GEN_KEYS), "seeds": rows, "wall_s": round(wall, 1),
                       "battles_min": round(a.n * a.seeds / max(wall, 1e-6) * 60, 1),
                       "by_map": {mp: summary(v, ("head_on", "frame_exec", "game_orders_min"))
                                  for mp, v in by_map.items()}}
        per[name] = {k: torch.cat(v) for k, v in allm.items()}
    pairs = [(p, "stand") for p in per if p != "stand"]
    return {"what": "general", "n": a.n, "seeds": a.seeds, "table": table,
            "paired_vs_stand": {k: sum((rank(per, k, pr) for pr in pairs), []) for k in
                                ("trade", "head_on", "frame_exec", "flank_taken", "game_orders_min")},
            "per_battle": {p: {k: [round(float(x), 4) for x in v.tolist()] for k, v in d.items()}
                           for p, d in per.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("drill", "general"))
    ap.add_argument("--players", default=",".join(PLAYERS))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="general: cut every battle at this many s (0: the stage's)")
    ap.add_argument("--ckpt", default=None, help="the unit network (default: net/config/tactic.json train.unit_ckpt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu-duty", type=float, default=1.0, help="the GPU busy at most this share of the time")
    a = ap.parse_args(argv)
    from sim.params import load
    P = load()
    t0 = time.time()
    res = drill(a, P) if a.what == "drill" else general(a, P)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in res if k.startswith("rank")}), flush=True)
    return res


if __name__ == "__main__":
    main()
