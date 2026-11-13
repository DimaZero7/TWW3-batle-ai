"""The game's AI against the game's AI with stage 0 armies (net/stage0/specs.py: emp against skv, 5-8 infantry and a
lord a side, gold matched), and the same battles in the simulator (ai_game both sides). Side 1 (west) under CA's script
planner 'attack the force' (bridge nn-arena --own-ai ai, apps.orders.planner_adapter), side 2 (east) the game's battle AI
attacking or defending (BATTLES' role); f* - the flat mp_crossroads_flat with Kislev's zone gap 192 m, k* -
ksl_plains_01. Both sides in rows across their zones' centres (specs.rows), the lord LORD_BACK m behind.

    python -m game.battle.aivai make                  build/aivai/<name>.json (arenas for --arena-json), battles.json
    bash game/battle/aivai.sh [names]                 the series in the game (make first; default all eight)
    python -m game.battle.aivai game <runs...>        build/aivai/game_rows.json: the measures of the game's runs
    python -m game.battle.aivai sim [names]           build/aivai/sim_rows.json (torch: ops/dock.sh), ai_game both
                                                      sides from the game's first-second places (game_rows.json)

measure(): first contact, when each side set off and how far it walked, the lord's place and first melee, the share of
melee seconds through an enemy's flank or rear, rout onsets and the first, the end (the game: its result event).
Evidence of 13.11: game/probe/evidence/aivai_game.json.
"""
import json
import math
import random
import sys
from pathlib import Path

import game

ROOT = game.ROOT
OUT = game.BUILD / "aivai"
UNITS = json.loads((ROOT / "sim/config/units.json").read_text(encoding="utf-8"))["units"]
GAP = 192.0                     # Kislev's game zones: centres at x +-184, depth 176
LORD_BACK = 30.0
FWD = {1: 1.0, 2: -1.0}         # side 1 west facing +x, side 2 east facing -x

# (name, seed, map, n units a side, the game AI's role: 'attack' | 'defend')
BATTLES = (("f1", 101, "flat", 6, "attack"), ("f2", 102, "flat", 8, "attack"), ("f3", 103, "flat", 5, "defend"),
           ("f4", 104, "flat", 7, "defend"),
           ("k1", 201, "ksl_plains_01", 6, "attack"), ("k2", 202, "ksl_plains_01", 8, "attack"),
           ("k3", 203, "ksl_plains_01", 5, "defend"), ("k4", 204, "ksl_plains_01", 7, "defend"))


def spec_of(seed, mp, n):
    from net.stage0 import specs as S
    rnd = random.Random(seed)
    fac = ("emp", "skv") if seed % 2 else ("skv", "emp")        # (no mirror: two armies of one faction)
    f1, f2, mine, theirs = S._sides(rnd, n, fac=fac)
    mz, ez = S.zone(0), S.zone(1)
    return {"fac": {1: f1, 2: f2}, "mine": mine, "theirs": theirs, "map": mp, "mine_zone": mz, "enemy_zone": ez,
            "places": {1: S.rows(mz, n), 2: S.rows(ez, len(theirs))}, "gold": (S.gold(mine), S.gold(theirs))}


def lord_key(fac):
    from net.stage0 import cfg
    return cfg()["lords"][fac]


def world_units(sp):
    """{1: [unit], 2: [unit]} the sim's matchup shape: key, slot, name, x, z, b, general, width, men."""
    out = {}
    for s, keys in ((1, sp["mine"]), (2, sp["theirs"])):
        pl = sp["places"][s]
        rows = []
        for i, (k, (x, z, b)) in enumerate(zip(keys, pl)):
            rows.append({"key": k, "slot": f"u{i + 1}", "name": f"u{i + 1}", "x": x, "z": z, "b": b, "general": False,
                         "width": 30.0, "men": int(UNITS[k]["men"])})
        mx = sum(p[0] for p in pl) / len(pl)
        mz = sum(p[1] for p in pl) / len(pl)
        b = pl[0][2]
        a = math.radians(b)
        lk = lord_key(sp["fac"][s])
        rows.insert(0, {"key": lk, "slot": "lord", "name": "lord", "x": mx - LORD_BACK * math.sin(a),
                        "z": mz - LORD_BACK * math.cos(a), "b": b, "general": True, "width": 5.0, "men": 1})
        out[s] = rows
    return out


def arena_of(name, sp, units):
    """The arena dict bridge.build --arena-json takes (sim/arena.py shape)."""
    from sim import arena as A
    from net.stage0 import cfg
    base = A.load_arena()
    sides = {}
    for s, side in ((1, "own"), (2, "enemy")):
        rows = []
        for u in units[s]:
            if side == "own":
                fwd, lat = u["x"] + GAP / 2, -u["z"]
            else:
                fwd, lat = GAP / 2 - u["x"], u["z"]
            r = {"slot": u["slot"], "key": u["key"], "men": u["men"], "forward": round(fwd, 1), "lateral": round(lat, 1),
                 "width": u["width"]}
            if u["general"]:
                r["general"] = True
            rows.append(r)
        cost = sum(float(UNITS[u["key"]].get("multiplayer_cost") or 0) for u in units[s])
        sides[side] = {"faction": cfg()["factions"][sp["fac"][s]], "units": rows, "cost": cost}
    arena = {k: v for k, v in base.items() if k not in ("faction", "units", "description", "sides")}
    arena.update(name=f"aivai_{name}", gap_m=GAP, sides=sides, note="probe 3: stage 0 armies, the game's AI both sides")
    return arena


def make():
    OUT.mkdir(parents=True, exist_ok=True)
    allb = {}
    for name, seed, mp, n, role in BATTLES:
        sp = spec_of(seed, mp, n)
        units = world_units(sp)
        (OUT / f"{name}.json").write_text(json.dumps(arena_of(name, sp, units), indent=1), encoding="utf-8")
        allb[name] = {"seed": seed, "map": mp, "role": role, "fac": sp["fac"], "gold": sp["gold"],
                      "units": {str(s): v for s, v in units.items()}}
        print(name, mp, role, sp["fac"], [u["key"].split("_inf_")[-1] for u in units[1]],
              [u["key"].split("_inf_")[-1] for u in units[2]], [round(g) for g in sp["gold"]])
    (OUT / "battles.json").write_text(json.dumps(allb, indent=1), encoding="utf-8")



def _ang(a):
    return (a + 180.0) % 360.0 - 180.0


def measure(frames):
    """First contact (both sides' melee flags), when each side's line set off (the middle of its units 10 m from the
    start) and how far it walked to contact, its lord's place behind the line 10 s before contact and his first melee
    after it, the share of melee seconds struck through an enemy's flank or rear (the nearest enemy in melee within 40 m,
    the striker more than 60 deg off its facing), rout onsets a side and the first after contact, the end (a side with
    nothing standing) and the winner."""
    out = {}
    f0 = frames[0][1]
    contact = next((t for t, us in frames if any(u["m"] for u in us if u["side"] == 1)
                    and any(u["m"] for u in us if u["side"] == 2)), None)
    out["contact_s"] = contact
    for side in (1, 2):
        idx = [i for i, u in enumerate(f0) if u["side"] == side and not u["lord"]]
        sx = sum(f0[i]["x"] for i in idx) / len(idx)
        go = None
        for t, us in frames:
            mx = sum(us[i]["x"] for i in idx) / len(idx)
            if go is None and abs(mx - sx) > 10:
                go = t
            if contact is not None and t >= contact:
                out[f"s{side}_walked_m"] = round((mx - sx) * FWD[side], 1)
                break
        out[f"s{side}_off_s"] = go
        li = next((i for i, u in enumerate(f0) if u["side"] == side and u["lord"]), None)
        if li is not None and contact is not None:
            ref = [us for t, us in frames if contact - 10 <= t <= contact]
            if ref:
                d = [(sum(us[i]["x"] for i in idx) / len(idx) - us[li]["x"]) * FWD[side] for us in ref]
                out[f"s{side}_lord_back_m"] = round(sum(d) / len(d), 1)
            lm = next((t for t, us in frames if us[li]["m"] and t >= contact - 5), None)
            out[f"s{side}_lord_melee_s"] = None if lm is None else round(lm - contact, 1)
            out[f"s{side}_lord_routed"] = any(us[li]["r"] for t, us in frames)
    for side in (1, 2):
        fl = tot = 0
        for t, us in frames:
            mine = [u for u in us if u["side"] == side and u["m"] and not u["r"]]
            foes = [u for u in us if u["side"] != side and u["m"] and not u["r"]]
            for u in mine:
                if not foes:
                    continue
                e = min(foes, key=lambda e: (e["x"] - u["x"]) ** 2 + (e["z"] - u["z"]) ** 2)
                if (e["x"] - u["x"]) ** 2 + (e["z"] - u["z"]) ** 2 > 40 ** 2:
                    continue
                tot += 1
                to_me = math.degrees(math.atan2(u["x"] - e["x"], u["z"] - e["z"])) % 360
                if abs(_ang(to_me - e["b"])) > 60:
                    fl += 1
        out[f"s{side}_flank_share"] = round(fl / tot, 3) if tot else None
    for side in (1, 2):
        n, first, prev = 0, None, {}
        for t, us in frames:
            for i, u in enumerate(us):
                if u["side"] != side:
                    continue
                if u["r"] and not prev.get(i):
                    n += 1
                    if first is None:
                        first = t
                prev[i] = u["r"]
        out[f"s{side}_routs"] = n
        out[f"s{side}_first_rout_s"] = None if first is None or contact is None else round(first - contact, 1)
    end, winner = None, 0
    for t, us in frames:
        up = {sd: any(u["men"] > 0 and not u["r"] and not u["s"] for u in us if u["side"] == sd) for sd in (1, 2)}
        if not (up[1] and up[2]):
            end, winner = t, (1 if up[1] else 2 if up[2] else 0)
            break
    out["end_s"] = end if end is not None else frames[-1][0]
    out["winner"] = winner
    out["fight_s"] = None if contact is None else round(out["end_s"] - contact, 1)
    return out


def game_frames(run):
    """[(t, units)] of an nn-arena run's nn_sample events (t in ms in the event)."""
    frames = []
    with open(Path(run) / "events.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if '"nn_sample"' not in line:
                continue
            d = json.loads(line)
            if d.get("event") != "nn_sample":
                continue
            us = [{"n": u["n"], "side": u["side"], "lord": u["n"].endswith("_lord"), "x": u["x"], "z": u["z"], "b": u["b"],
                   "men": u["men"], "m": bool(u["m"]), "r": bool(u["r"]), "s": bool(u["s"])}
                  for u in sorted(d["units"], key=lambda u: u["n"])]
            frames.append((d["t"] / 1000.0, us))
    return frames


def game_runs(runs):
    rows = []
    for r in runs:
        m = json.loads((Path(r) / "manifest.json").read_text(encoding="utf-8"))
        name = Path(m["config"]["army"]["from_file"]).stem
        fr = game_frames(r)
        row = dict(measure(fr), battle=name, run=Path(r).name, frames=len(fr))
        for line in open(Path(r) / "events.jsonl", encoding="utf-8"):
            if '"event":"result"' in line:
                res = json.loads(line)
                # (the game's own end: an army's collapse routs it before every unit runs)
                row["end_s"] = round(res["duration_model_ms"] / 1000, 1)
                row["winner"] = int(res.get("winner") or 0) if res.get("status") == "completed" else 0
                row["status"] = res.get("status")
                row["fight_s"] = None if row["contact_s"] is None else round(row["end_s"] - row["contact_s"], 1)
        rows.append(row)
        print(json.dumps(row))
    return rows


def game_starts():
    """{battle: {(side, slot): (x, z, b)}} the units' places at the game's first second (the game's AI re-deploys its
    own side in the deployment phase), from game_rows.json's runs."""
    rows = json.loads((OUT / "game_rows.json").read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        run = game.BUILD / "nn-arena" / "runs" / r["run"]
        f0 = game_frames(run)[0][1]
        out[r["battle"]] = {(u["side"], u["n"].split("_", 1)[1]): (u["x"], u["z"], u["b"]) for u in f0}
    return out


def sim(names=None, copies=2):
    from net.matchup import duel as DU
    from net.matchup import armies as AR
    from net.stage0 import cfg
    from sim.params import load
    allb = json.loads((OUT / "battles.json").read_text(encoding="utf-8"))
    names = names or list(allb)
    orig = AR.description
    DU.AR.description = lambda n, b, attacker=None: orig(n, b, b.get("attacker"))
    starts = game_starts()
    names = [n for n in names if n in starts]
    battles, maps, meta = [], [], []
    for nm in names:
        b = allb[nm]
        for _ in range(copies):
            units = {int(s): [dict(u) for u in v] for s, v in b["units"].items()}
            for s_ in (1, 2):
                for u in units[s_]:
                    u["x"], u["z"], u["b"] = starts[nm][(s_, u["slot"])]
            units["factions"] = {s: cfg()["factions"][b["fac"][str(s)]] for s in (1, 2)}
            units["attacker"] = 2 if b["role"] == "attack" else 1
            battles.append(units)
            maps.append("flat" if b["map"] == "flat" else b["map"])
            meta.append(nm)
    rec = [[] for _ in battles]

    def watch(sec, st):
        u = st.u
        keys = ("side", "lord", "x", "z", "b", "men", "m", "r", "gone")
        cpu = {k: u[k].detach().cpu() for k in keys}
        for bi in range(len(battles)):
            us = []
            for i in range(cpu["side"].shape[1]):
                sd = int(cpu["side"][bi, i])
                if sd <= 0:
                    continue
                us.append({"side": sd, "lord": bool(cpu["lord"][bi, i]), "x": float(cpu["x"][bi, i]),
                           "z": float(cpu["z"][bi, i]), "b": float(cpu["b"][bi, i]), "men": float(cpu["men"][bi, i]),
                           "m": bool(cpu["m"][bi, i]), "r": bool(cpu["r"][bi, i]),
                           "s": bool(cpu["gone"][bi, i]) or float(cpu["men"][bi, i]) <= 0})
            rec[bi].append((float(sec), us))
    sides = {1: DU.Script(1, "ai_game"), 2: DU.Script(2, "ai_game")}
    DU.play("m1", battles, sides, None, P=load(), device="cpu", limit_s=600, watch=watch, maps=maps, compile=False)
    rows = []
    for nm, fr in zip(meta, rec):
        row = dict(measure(fr), battle=nm, run="sim")
        rows.append(row)
        print(json.dumps(row))
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "make"
    if cmd == "make":
        make()
    elif cmd == "game":
        rows = game_runs(sys.argv[2:])
        (OUT / "game_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    elif cmd == "sim":
        rows = sim(sys.argv[2:] or None)
        (OUT / "sim_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
