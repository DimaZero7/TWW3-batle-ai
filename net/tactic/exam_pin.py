"""Drill T0-2 'pin and strike' (tmp/tactic_slice/design_flank_v2.md): the players of tactic/pin.py through the tactic
level (stage0/battle.play_tactic, watch.PinWatch) on general battles, paired by battle.

    players   the pilot's 768 general battles (--set pilot: seeds 0-2 of the pilot's generator - stage0/specs.generals on
              SP.seeded(9100 + seed), the enemy drawn anew among the scripts with random.Random(9200 + seed), as
              tmp/tactic_slice/pilot/run.general_specs; torch.manual_seed(2000 + seed) before each seed's batch), each
              player's per-battle measures to <out stem>_<player>.json, then the table of every player file of that stem
              and of --ref (the pilot's own per-player files: their per_battle) in <out>
A player's name: a key of pin.PLAYERS, optionally with ':holes' (config orders.hole_fits off - the old rule, a frame on a
hole must move), ':line' (orders.zone_fit off - a line zone on the units' middle), ':norout' (orders.end_on_rout off)
and ':nocap' (orders.cap_all off). The judge's window: plain health
at the anchor + 300 s, the anchor = the battle's first contact under K (--anchor, default K; capped at the battle's
limit, the share cut written).

    DOCK_NAME=agent-pin1 DOCK_CPUS=12 DOCK_MEM=6g bash ops/dock.sh net.tactic.exam_pin players --players K,T,A1,D --out tmp/pin/stage1.json
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from net.stage0 import battle as BA
from net.stage0 import cfg as scfg
from net.stage0 import specs as SP
from net.stage0 import watch as W
from net.tactic import cfg as tcfg
from net.tactic import pin as PN

EVENTS = ("attack_orders", "intercepted", "pin_lost", "gone_before", "resumed_on_rallied", "resumed_way_m", "reached",
          "unpinned_attacks", "two_on_side", "melee_changes", "melee_blocked", "rejected", "forced_first",
          "forced_hole", "forced_zone", "hole_kept", "let_go", "changes", "ended_on_rout", "striker_routed",
          "ord_pinned", "ord_chasing", "ord_approaching", "ord_standing", "ord_other", "inter_nearer",
          "order60_inter_n", "order60_inter_sum", "order60_clean_n", "order60_clean_sum") + tuple(
          f"fc_{c}_{w}" for c in ("mine_charging", "enemy_onto_standing", "both_moving", "neither")
          for w in ("n", "n30", "s30", "n60", "s60"))
MEANS = ("hp_trade", "trade", "game_orders_min", "routs_own", "routs_enemy", "t_contact", "zone_worst_m") + EVENTS
WINDOW_S = 300


def pilot_specs(n, seed, limit=0):
    """The pilot's general battles of a seed (module doc); limit: each battle's length, s (0: the stage's)."""
    shares = {k: v for k, v in scfg()["general"]["enemies"].items() if k != "chain"}
    sp = SP.generals(n, SP.seeded(9100 + seed))
    rnd = random.Random(9200 + seed)
    for x in sp:
        if x["enemy"] == "chain" or x["enemy"] not in shares:
            x["enemy"] = rnd.choices(list(shares), weights=list(shares.values()))[0]
        if limit:
            x["limit"] = int(limit)
    return sp


def _variant(name):
    base, *opts = name.split(":")
    o = tcfg()["orders"]
    saved = dict(o)
    o["hole_fits"] = "holes" not in opts and bool(saved["hole_fits"])
    o["zone_fit"] = "line" not in opts and bool(saved["zone_fit"])
    o["end_on_rout"] = "norout" not in opts and bool(saved.get("end_on_rout", True))
    o["cap_all"] = "nocap" not in opts and bool(saved.get("cap_all", True))
    return base, saved


def run_player(name, a, P):
    base, saved = _variant(name)
    try:
        per, wall = {}, 0.0
        for s in range(a.seeds):
            sp = pilot_specs(a.n, s, a.limit)
            torch.manual_seed(2000 + s)
            m, st = BA.play_tactic(sp, lambda: PN.player(base), P, a.device, hygiene=base not in PN.HYGIENE_OFF,
                                   chunk=a.chunk, compile=None if a.device.startswith("cuda") else False,
                                   watch_cls=W.PinWatch)
            wall += st["wall_s"]
            for k, v in m.items():
                per.setdefault(k, []).append(v)
            print(json.dumps({"player": name, "seed": s, "wall_s": st["wall_s"],
                              **{k: round(float(m[k][torch.isfinite(m[k])].mean()), 4) for k in MEANS if k in m}}),
                  flush=True)
    finally:
        tcfg()["orders"].update(saved)
    per = {k: torch.cat(v) for k, v in per.items()}
    out = {"wall_s": round(wall, 1), "per_battle": {k: [round(float(x), 4) for x in v.tolist()]
                                                     for k, v in per.items() if v.dim() == 1},
           "hp_series": [[round(float(x), 4) for x in r] for r in per["hp_series"].tolist()]}
    return out


def paired(a, b):
    d = [x - y for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    n = len(d)
    if n < 2:
        return None
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return {"diff": round(m, 4), "two_errors": round(2 * sd / math.sqrt(n), 4), "n": n}


def window(per, anchor, limit):
    """[B] plain health at anchor + WINDOW_S (the series' step, capped at its last point) and the share cut."""
    out, cut = [], 0
    step = W.PinWatch.SERIES_S
    for b, ser in enumerate(per["hp_series"]):
        t = anchor[b] + WINDOW_S
        if t > limit:
            cut += 1
        i = min(int(round(t / step)), len(ser) - 1)
        out.append(ser[i])
    return out, cut / max(len(per["hp_series"]), 1)


def table(files, ref, anchor_name="K", limit=None):
    """{players: means and event shares, pairs: paired differences} of the player files and the pilot's ref files."""
    limit = limit or int(scfg()["general"]["limit_s"])
    pl = {n: json.loads(Path(f).read_text(encoding="utf-8")) for n, f in files.items()}
    rf = {n: json.loads(Path(f).read_text(encoding="utf-8")) for n, f in ref.items()}
    mean = lambda v: round(sum(x for x in v if math.isfinite(x)) / max(sum(1 for x in v if math.isfinite(x)), 1), 4)  # noqa
    out = {"players": {}, "pairs": {}}
    anchor = None
    if anchor_name in pl:
        anchor = pl[anchor_name]["per_battle"]["t_contact"]
    for n, d in pl.items():
        pb = d["per_battle"]
        row = {k: mean(pb[k]) for k in MEANS if k in pb}
        ao = sum(pb.get("attack_orders", [0]))
        for k in ("intercepted", "pin_lost", "gone_before", "resumed_on_rallied", "reached", "unpinned_attacks"):
            if k in pb and ao:
                row[k + "_share"] = round(sum(pb[k]) / ao, 4)
        for k in EVENTS:
            if k in pb:
                row[k + "_sum"] = round(sum(pb[k]), 1)
        if anchor is not None:
            w, cut = window(d, anchor, limit)
            pb["judge_300"] = w
            row["judge_300"], row["judge_300_cut_share"] = mean(w), round(cut, 4)
        row["wall_s"] = d.get("wall_s")
        out["players"][n] = row
    for n, d in rf.items():
        pb = d["per_battle"]
        out["players"]["pilot_" + n] = {k: mean(pb[k]) for k in ("hp_trade", "trade", "game_orders_min", "routs_own",
                                                                 "routs_enemy", "t_contact", "tasks") if k in pb}
    allp = {**{n: d["per_battle"] for n, d in pl.items()}, **{"pilot_" + n: d["per_battle"] for n, d in rf.items()}}
    names = list(allp)
    for a_ in names:
        for b_ in names:
            if a_ == b_:
                continue
            row = {}
            for k in ("hp_trade", "judge_300", "game_orders_min", "routs_own", "routs_enemy", "trade"):
                if k in allp[a_] and k in allp[b_]:
                    row[k] = paired(allp[a_][k], allp[b_][k])
            out["pairs"][f"{a_} - {b_}"] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("players", "table"))
    ap.add_argument("--set", default="pilot", choices=("pilot",))
    ap.add_argument("--players", default="K,T,A1,D")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="each battle's length, s (0: the stage's 600)")
    ap.add_argument("--ref", default="", help="the pilot's per-player files: name=path,...")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="tmp/pin/stage1.json")
    a = ap.parse_args(argv)
    from sim.params import load
    P = load()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = lambda n: out.with_name(out.stem + "_" + n.replace(":", "@") + ".json")       # noqa: E731
    t0 = time.time()
    if a.what == "players":
        for name in a.players.split(","):
            res = run_player(name, a, P)
            stem(name).write_text(json.dumps(res), encoding="utf-8")
    files = {p.stem[len(out.stem) + 1:].replace("@", ":"): str(p) for p in sorted(out.parent.glob(out.stem + "_*.json"))}
    ref = dict(x.split("=", 1) for x in a.ref.split(",") if x)
    res = table(files, ref, limit=a.limit or None)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["players"]), flush=True)


if __name__ == "__main__":
    main()
