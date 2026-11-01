"""The projectiles from the game's memory in a missile probe run (memreader Plus; docs/ru/game/memreader.md) against
what the game's script shows of the same battle: each lane's ammo (the projectiles it fired) and its units' places;
and (--men) the men of a charge probe run's units from memory against the script.

    python -m tools.nn.mem_probe [run folders...]      # default: the newest build/missile-probe/runs/* with mem rows
    python -m tools.nn.mem_probe --men [run folders...]   # default: the newest build/charge-probe/runs/* with men rows
    python -m tools.nn.mem_probe --morale [run folders...]  # default: the newest build/morale-probe/runs/* with mem rows

A morale probe run built with --mem (tools/nn/morale_probe.py plan mem) has mem_attach {status, bound} and every poll
(0.1 s of battle) mem_morale {c, tick, cost, t, units = {name = {st, pts, tgt, n, ok, same, fx [id, points, ...], tm [id,
value, until, ...], ct [enemy, side, ...], tim [surprise, waver, charge, rally], by [16 flag bytes], uf {field: value},
mp, ms, w, r, s, sn}}} (src/apps/mem/morale.lua; the script's CCO MoralePercent / MoraleState and flags read in the same
callback); morale_check: the points against MoralePercent (mp x base = points, base = target - the effects' sum), the
state against the script's flags, and every effect seen: its number, the game's name (EFFECT), its points, where.

A charge probe run built with --mem has mem_attach {status, bound} and every poll (0.1 s of battle) mem_men {c, tick,
tick_end, cost, t, units = {name = {n, rows (MAN fields a man, flat), sn, sx, sz}}, new = [[id, unit, address]]}
(src/apps/mem/entities.lua); men_check: in each poll and unit, the men alive from memory (the men vector: the dead
leave it) against the script's number_of_men_alive, and the script's unit:position against the men's places (the game
puts the unit at one of its men: the nearest man is 0.00 m off), with the share of (poll, unit) matching both.

A run built with `tools.build missile-probe --mem` has, besides the probe's rows (tools/nn/missile_probe.py):
  mem_attach  {status}: memreader's version, each code site's check, hooked or why not
  mem_proj    every probe tick (model_ms): calls (the hook's call count: one a simulation tick), spawn / state / gone
              (apps.mem.projectiles: {id, c, ...} with the call number c), fly (the projectiles flying now), hook
              (memreader's counters: calls, other_thread_calls)

Measures (load + compare):
  hook       did it fire on the script thread (calls > 0, other_thread_calls), its errors
  axes       the spawn points against the shooters' script places: the median distance (m) to the nearest shooter
             (memory x, z = the script's x, z if small)
  per lane   projectiles from memory attributed to the lane (the nearest shooter within ATTRIB_M at the spawn) against
             the script's ammo drop; matched = the sum over 0.5 s bins of min(memory, script) with one bin of slack
             (the hook counts at the simulation tick, the script reads ammo every tick_ms); share = matched / max(both)
  ends       the share of spawned projectiles that ended (state 5 / 6 lying, 7 done, or gone) and the flight time at the
             end; where they came down against the target's centre
Writes build/mem/projectiles.json (not in Git).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from tools import config as project

RUNS = project.BUILD / "missile-probe" / "runs"
MEN_RUNS = project.BUILD / "charge-probe" / "runs"
OUT = project.BUILD / "mem" / "projectiles.json"
MEN_OUT = project.BUILD / "mem" / "men.json"
MORALE_RUNS = project.BUILD / "morale-probe" / "runs"
MORALE_OUT = project.BUILD / "diag_mem" / "morale" / "morale.json"
# The morale effects by their number in a unit's list (the game's own name table of the effects, read from the program:
# R-work notes/mem/morale.md); the points of each are the database's _kv_morale rows (config/nn/game_rules.json)
EFFECT = ("none", "army_supported", "general_nearby", "general_rallying_nearby", "general_alive", "winning", "kills",
          "flanks_secure", "fortification", "fortification_breach", "settlement", "settlement_plaza", "on_the_hill",
          "enemy_routing", "overwhelming_friendly_units", "charging", "inspired", "inspired_by_unit_ability",
          "rallied_by_unit_ability", "by_ground_type", "num_positive", "army_destruction", "general_recently_died",
          "general_dead", "general_fled_recently", "losing", "melee_contact", "flank_attacked", "rear_attacked",
          "flanks_exposed_single", "flanks_exposed_multiple", "fatigue_tired", "fatigue_very_tired", "fatigue_exhausted",
          "casualties", "recent_casualties", "extended_casualties", "damaged_by_artillery", "attacked_by_artillery",
          "attacked_by_projectile", "under_friendly_fire", "fighting_cavalry", "friends_routing", "enemy_numbers",
          "surprised", "frightened_horses", "fear_enemy_unit", "discouraged_by_unit_ability", "projectiles_overhead",
          "night_battle_unprepared", "enemy_controls_land", "settlement_destruction", "concerned_by_ground_type",
          "terror_enemy_unit", "immunity_terror")
# a man's row from memory (src/apps/mem/entities.lua M.ROW)
MAN = ("id", "live", "x", "z", "y", "vx", "vz", "face", "hp", "anim", "mode", "react", "mark", "fit", "flags", "brace",
       "speed", "tgt", "tgt2", "carried", "chg")
COL = {k: i for i, k in enumerate(MAN)}
# reaction states (build/diag_cav/mechanic.md M8, M31; R-work notes/mem/entities.md)
REACT = {0xd: "step_back", 0xe: "knocked_back", 0xf: "down", 0x10: "flying", 0x11: "getting_up", 0x12: "getting_up_b"}
NOT_TARGET = (4, 6, 7, 0xc, 0xe, 0xf, 0x10, 0x11)
PLACE_EPS = 0.05
ATTRIB_M = 40.0
BIN_S = 0.5
LYING, DONE = (5, 6), 7
# The objects in the vector by their first state (the game's run, dist 2): a launched projectile appears in
# state 0 with flight time 0 and goes to 7 when it comes down; where it came down the game puts a NEW object lying on
# the ground (first seen in 6, ~20 s); one that hit a man is a new object too (first seen in 2, then 3, 7: the arrow in
# the body, its shooter kept)
LAUNCH, BODY = 0, 2


def kind(s):
    return {LAUNCH: "launch", BODY: "body", 5: "lying", 6: "lying"}.get(s["st"], "other")
# the spawn / state / fly rows (apps.mem.projectiles)
SPAWN = ("id", "c", "st", "x", "y", "z", "vx", "vy", "vz", "t", "rec", "sx", "sy", "sz")
STATE = ("id", "c", "st", "x", "y", "z", "t")
FLY = ("id", "st", "x", "y", "z", "vx", "vy", "vz", "t")


def rows(run_dir):
    for line in (Path(run_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def has_mem(run_dir):
    p = Path(run_dir) / "events.jsonl"
    return p.is_file() and '"mem_attach"' in p.read_text(encoding="utf-8")


def as_dicts(items, names):
    return [dict(zip(names, it)) for it in (items or [])]


def load(run_dir):
    """{attach, ticks [(model_ms, calls)], spawn, state, gone, fly [(model_ms, [..])], hook (the last), errors, records,
    lanes {name: {spec, t [model s], s_x, s_z, t_x, t_z, ammo}}} of one run."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    out = {"run": run_dir.name, "attach": None, "ticks": [], "spawn": [], "state": [], "gone": [], "fly": [],
           "hook": None, "errors": [], "records": [], "lanes": {}, "ammo_hook": [], "script_ammo": {}}
    lanes = {l["name"]: {"spec": l, "t": [], "s_x": [], "s_z": [], "t_x": [], "t_z": [], "ammo": [], "t_hp": []}
             for l in cfg.get("lanes", [])}
    for r in rows(run_dir):
        ev = r.get("event")
        if ev == "mem_attach":
            out["attach"] = r.get("status")
        elif ev == "mem_proj":
            if r.get("calls") is not None:
                out["ticks"].append((float(r["model_ms"]), int(r["calls"])))
            out["spawn"] += as_dicts(r.get("spawn"), SPAWN)
            out["state"] += as_dicts(r.get("state"), STATE)
            out["gone"] += [dict(zip(("id", "c"), g)) for g in (r.get("gone") or [])]
            if r.get("fly"):
                out["fly"].append((float(r["model_ms"]), as_dicts(r["fly"], FLY)))
            out["errors"] += list(r.get("errors") or [])
            out["records"] += list(r.get("records") or [])
            if r.get("hook"):
                out["hook"] = r["hook"]
            if r.get("discover"):
                out["discover"] = r["discover"]
            if r.get("chain"):
                out.setdefault("chains", []).append(r["chain"])
            if r.get("ammo"):
                out["ammo_hook"].append((float(r["model_ms"]), [a[1] for a in r["ammo"] if a and a[0]]))
        elif ev == "probe_sample":
            out["script_ammo"][float(r["model_ms"])] = {(x.get("s") or {}).get("ammo") for x in r["lanes"]} | {
                (x.get("f") or {}).get("ammo") for x in r["lanes"]}
            for x in r["lanes"]:
                L = lanes.get(x["lane"])
                if L is None:
                    continue
                s, tg = x.get("s") or {}, x.get("tg") or {}
                L["t"].append(float(r["model_ms"]) / 1000)
                for k, src, key in (("s_x", s, "x"), ("s_z", s, "z"), ("t_x", tg, "x"), ("t_z", tg, "z"),
                                    ("ammo", s, "ammo"), ("t_hp", tg, "hp")):
                    v = src.get(key)
                    L[k].append(np.nan if v is None else float(v))
    out["lanes"] = {k: {kk: (np.asarray(vv, float) if kk != "spec" else vv) for kk, vv in v.items()}
                    for k, v in lanes.items() if v["t"]}
    return out


def call_time(ticks):
    """f(c) -> model s of the hook's call number c, linear between the probe ticks (calls counted at each)."""
    if len(ticks) < 2:
        return lambda c: np.nan
    ms = np.array([t[0] for t in ticks]) / 1000
    calls = np.array([t[1] for t in ticks], float)
    keep = np.concatenate([[True], np.diff(calls) > 0])
    ms, calls = ms[keep], calls[keep]
    return lambda c: float(np.interp(c, calls, ms))


def attribute(spawn, lanes, at):
    """Each spawn's lane (the nearest shooter centre at the spawn's time within ATTRIB_M; None) and that distance."""
    out = []
    for s in spawn:
        t = at(s["c"])
        best, dist = None, math.inf
        for name, L in lanes.items():
            k = int(np.clip(np.searchsorted(L["t"], t), 0, len(L["t"]) - 1))
            d = math.hypot(s["x"] - L["s_x"][k], s["z"] - L["s_z"][k])
            if d < dist:
                best, dist = name, d
        out.append((best if dist <= ATTRIB_M else None, dist, t))
    return out


def bins_match(a, b, slack=1):
    """matched count of two count series on the same bins (greedy, a bin of a may take from b's bins +- slack)."""
    a, b = np.asarray(a, float).copy(), np.asarray(b, float).copy()
    matched = 0.0
    for d in [0] + [x for k in range(1, slack + 1) for x in (-k, k)]:
        for i in range(len(a)):
            j = i + d
            if 0 <= j < len(b):
                m = min(a[i], b[j])
                matched += m
                a[i] -= m
                b[j] -= m
    return matched


def compare(run):
    """The run's numbers (module doc)."""
    at = call_time(run["ticks"])
    launches = [s for s in run["spawn"] if kind(s) == "launch"]
    att = attribute(launches, run["lanes"], at)
    hook = run["hook"] or {}
    out = {"run": run["run"], "attach_ok": bool((run["attach"] or {}).get("ok")),
           "attach": run["attach"], "hook": hook, "errors": run["errors"][:10], "n_errors": len(run["errors"]),
           "spawned": len(run["spawn"]), "records": run["records"],
           "kinds": dict(sorted(__import__("collections").Counter(kind(x) for x in run["spawn"]).items())),
           "discover": run.get("discover"), "chain_first": (run.get("chains") or [None])[0], "chain_last": (run.get("chains") or [None])[-1]}
    # the ammo_left hook: each value it read from memory against the script's ammo_left values of the same tick
    vals = [(ms, v) for ms, vs in run["ammo_hook"] for v in vs]
    if vals:
        same = sum(v in run["script_ammo"].get(ms, set()) for ms, v in vals)
        out["ammo_hook"] = {"reads": len(vals), "equal_to_script": same, "first": vals[:4]}
    if not launches:
        return out
    dists = np.array([a[1] for a in att])
    out["spawn_to_shooter_m"] = {"median": float(np.median(dists)), "p90": float(np.percentile(dists, 90)),
                                 "unattributed": int(sum(a[0] is None for a in att))}
    # ends
    last = {}
    for s in run["state"]:
        last[s["id"]] = s
    gone = {g["id"] for g in run["gone"]}
    ended = [s for s in launches if s["id"] in gone or (s["id"] in last and last[s["id"]]["st"] in LYING + (DONE,))]
    out["ended_share"] = len(ended) / len(launches)
    bodies = [s for s in run["spawn"] if kind(s) == "body"]
    out["states_seen"] = sorted({s["st"] for s in run["spawn"]} | {s["st"] for s in run["state"]})
    lanes = {}
    for name, L in run["lanes"].items():
        mine = [(s, a) for s, a in zip(launches, att) if a[0] == name]
        t = L["t"]
        ammo = L["ammo"]
        ok = np.isfinite(ammo)
        script = float(np.nanmax(ammo) - np.nanmin(ammo)) if ok.sum() > 1 else 0.0
        edges = np.concatenate([[t[0] - BIN_S], t])
        drop = np.concatenate([[0.0], np.clip(-np.diff(np.where(ok, ammo, np.nan)), 0, None)])
        drop = np.nan_to_num(drop)
        mem_t = np.array([a[2] for _, a in mine])
        mem_bins = np.histogram(mem_t, bins=edges)[0] if len(mem_t) else np.zeros(len(t))
        matched = bins_match(mem_bins, drop)
        row = {"memory": len(mine), "script": script, "matched": matched,
               "share": matched / max(1.0, max(script, float(len(mine))))}
        # where they came down: the last known point of each against the target's centre at that time
        ends = []
        for s, a in mine:
            e = last.get(s["id"])
            if e and e["st"] in LYING + (DONE,):
                k = int(np.clip(np.searchsorted(t, at(e["c"])), 0, len(t) - 1))
                ends.append((math.hypot(e["x"] - L["t_x"][k], e["z"] - L["t_z"][k]), e["t"]))
        if ends:
            d = np.array([x[0] for x in ends])
            ft = np.array([x[1] for x in ends])
            row.update(end_to_target_m_median=float(np.median(d)), within_15m=float(np.mean(d <= 15)),
                       flight_s_median=float(np.median(ft)))
        if mine:
            v = np.array([math.hypot(s["vx"], s["vz"]) for s, _ in mine])
            row["launch_speed_median"] = float(np.median(v))
        # arrows in bodies (new 'body' objects) within 20 m of this lane's target, against the target's HP lost
        nb = 0
        for b in bodies:
            k = int(np.clip(np.searchsorted(t, at(b["c"])), 0, len(t) - 1))
            nb += math.hypot(b["x"] - L["t_x"][k], b["z"] - L["t_z"][k]) <= 20
        hp = L["t_hp"][np.isfinite(L["t_hp"])]
        row["bodies_at_target"] = nb
        row["target_hp_lost"] = float(hp[0] - hp[-1]) if len(hp) > 1 else None
        lanes[name] = row
    out["lanes"] = lanes
    mem_total = sum(r["memory"] for r in lanes.values())
    script_total = sum(r["script"] for r in lanes.values())
    out["matched_share"] = sum(r["matched"] for r in lanes.values()) / max(1.0, max(mem_total, script_total))
    out["fly_ticks"] = len(run["fly"])
    out["fly_max"] = max((len(f) for _, f in run["fly"]), default=0)
    return out


def facing(face):
    """The facing vector (x, z) of a man's facing number: the game's table gives (sin, cos) of face / 65536 turns
    (read at the attach: 0 -> (0, 1), 16384 -> (1, 0))."""
    a = np.asarray(face, float) / 65536 * 2 * np.pi
    return np.sin(a), np.cos(a)


def load_men(run_dir):
    """{run, attach, bound, polls [{c, tick, tick_end, cost, t (model s), units {name: {n, sn, sx, sz, m [men, MAN]}}}],
    ids {id: (unit, address)}, discover [rows], errors} of one charge probe run."""
    run_dir = Path(run_dir)
    out = {"run": run_dir.name, "attach": None, "bound": None, "polls": [], "ids": {}, "discover": [], "errors": []}
    for r in rows(run_dir):
        ev = r.get("event")
        if ev == "mem_attach":
            out["attach"], out["bound"] = r.get("status"), r.get("bound")
        elif ev == "mem_men":
            units = {}
            for name, u in (r.get("units") or {}).items():
                m = np.asarray(u.get("rows") or [], float)
                w = len(m) // u["n"] if u.get("n") else len(MAN)
                m = m.reshape(-1, w)
                if w < len(MAN):                # (a run of an older row: the fields it had, nan for the rest)
                    m = np.concatenate([m, np.full((len(m), len(MAN) - w), np.nan)], 1)
                units[name] = {"n": u.get("n"), "sn": u.get("sn"), "sx": u.get("sx"), "sz": u.get("sz"), "m": m}
            out["polls"].append({"c": r.get("c"), "tick": r.get("tick"), "tick_end": r.get("tick_end"),
                                 "cost": r.get("cost"), "t": float(r.get("t", r["model_ms"])) / 1000, "units": units})
            for i, name, addr in r.get("new") or []:
                out["ids"][int(i)] = (name, addr)
            out["errors"] += list(r.get("errors") or [])
        elif ev == "mem_men_discover":
            out["discover"].append({k: r.get(k) for k in ("lane", "s", "hits")})
    return out


def men_check(men):
    """The men from memory against the script (module doc): per unit and overall."""
    per, all_ok = {}, []
    for p in men["polls"]:
        for name, u in p["units"].items():
            m = u["m"]
            row = per.setdefault(name, {"polls": 0, "alive_equal": 0, "place_on_a_man": 0, "both": 0, "d": []})
            row["polls"] += 1
            alive = int((m[:, COL["live"]] == 1).sum()) if len(m) else 0
            eq = u["sn"] is not None and alive == u["sn"]
            d = (float(np.hypot(m[:, COL["x"]] - u["sx"], m[:, COL["z"]] - u["sz"]).min())
                 if len(m) and u["sx"] is not None else math.inf)
            on = d <= PLACE_EPS + 0.1           # rows are rounded to 0.1 m
            row["alive_equal"] += eq
            row["place_on_a_man"] += on
            row["both"] += eq and on
            row["d"].append(d)
            all_ok.append(eq and on)
    for row in per.values():
        d = np.asarray(row.pop("d"))
        row["place_d_median"], row["place_d_max"] = float(np.median(d)), float(d.max())
        row["share"] = row["both"] / max(1, row["polls"])
    cost = [p["cost"] for p in men["polls"] if p["cost"] is not None]
    drift = sum(1 for p in men["polls"] if p["tick_end"] is not None and p["tick_end"] != p["tick"])
    return {"run": men["run"], "attach_ok": bool((men["attach"] or {}).get("ok")), "bound": men["bound"],
            "polls": len(men["polls"]), "share": float(np.mean(all_ok)) if all_ok else None, "units": per,
            "cost_ms_median": float(np.median(cost)) if cost else None, "cost_ms_max": max(cost) if cost else None,
            "polls_spanning_ticks": drift, "errors": men["errors"][:10], "discover": men["discover"]}


def effect_name(i):
    return EFFECT[i] if 0 <= i < len(EFFECT) else f"effect_{i}"


def load_morale(run_dir):
    """{run, cfg, attach, bound, go_ms, polls [{t (s from the go), tick, cost, units {name: row}}], errors} of one
    morale probe run with mem rows."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    out = {"run": run_dir.name, "cfg": cfg, "attach": None, "bound": None, "go_ms": None, "polls": [], "errors": []}
    for r in rows(run_dir):
        ev = r.get("event")
        if ev == "mem_attach":
            out["attach"], out["bound"] = r.get("status"), r.get("bound")
        elif ev == "probe_go":
            out["go_ms"] = r["model_ms"]
        elif ev == "mem_morale":
            out["polls"].append({"t": float(r.get("t", r["model_ms"])) / 1000, "tick": r.get("tick"),
                                 "cost": r.get("cost"), "units": r.get("units") or {}})
            out["errors"] += list(r.get("errors") or [])
    g = (out["go_ms"] or 0) / 1000
    for p in out["polls"]:
        p["t"] = round(p["t"] - g, 2)
    return out


def _pairs(xs, k=2):
    xs = list(xs.values()) if isinstance(xs, dict) else list(xs or [])
    return [tuple(xs[i:i + k]) for i in range(0, len(xs) - k + 1, k)]


def unit_roles(cfg):
    """{unit name: (lane, kind, role, fearless)} of a morale probe run's lanes."""
    return {un["name"]: (ln["name"], ln.get("kind"), un["role"], bool(un.get("fearless")))
            for ln in cfg.get("lanes", []) for un in ln["units"]}


def morale_check(mor):
    """The morale from memory against the script and the effects seen (module doc)."""
    roles = unit_roles(mor["cfg"])
    per = {}
    effects = {}
    for p in mor["polls"]:
        for name, u in p["units"].items():
            lane, kind, role, fearless = roles.get(name, (None, None, None, None))
            row = per.setdefault(name, {"lane": lane, "kind": kind, "role": role, "fearless": fearless, "polls": 0,
                                        "list_ok": 0, "same": 0, "mp_polls": 0, "mp_equal": 0, "mp_err_max": 0.0,
                                        "base": {}, "states": {}, "flag_polls": 0, "flag_equal": 0})
            row["polls"] += 1
            row["same"] += int(u.get("same", 0))
            fx = _pairs(u.get("fx"))
            if not u.get("ok"):
                continue
            row["list_ok"] += 1
            base = int(u["tgt"]) - sum(v for _, v in fx)
            row["base"][base] = row["base"].get(base, 0) + 1
            key = f"{u['st']}:{u.get('ms')}"
            row["states"][key] = row["states"].get(key, 0) + 1
            # the script's flags against the state: wavering 5, routing 6 / shattered 7
            if u.get("w") is not None and u.get("r") is not None:
                w, r = bool(u["w"]), bool(u["r"])
                row["flag_polls"] += 1
                row["flag_equal"] += int((w == (u["st"] == 5)) and (r == (u["st"] in (6, 7))))
            mp = u.get("mp")
            if mp is not None and base > 0:
                err = abs(float(mp) - u["pts"] / base)
                row["mp_polls"] += 1
                row["mp_equal"] += int(err < 6e-4)
                row["mp_err_max"] = max(row["mp_err_max"], err)
            for i, v in fx:
                e = effects.setdefault(int(i), {"name": effect_name(int(i)), "values": {}, "units": set(), "polls": 0})
                e["values"][int(v)] = e["values"].get(int(v), 0) + 1
                e["units"].add(f"{lane}:{kind}:{role}")
                e["polls"] += 1
    for e in effects.values():
        e["units"] = sorted(e["units"])
    mp_polls = sum(r["mp_polls"] for r in per.values())
    flag_polls = sum(r["flag_polls"] for r in per.values())
    cost = [p["cost"] for p in mor["polls"] if p.get("cost") is not None]
    return {"run": mor["run"], "attach_ok": bool((mor["attach"] or {}).get("ok")), "bound": mor["bound"],
            "polls": len(mor["polls"]), "errors": mor["errors"][:10], "units": per,
            "mp_share": sum(r["mp_equal"] for r in per.values()) / max(1, mp_polls), "mp_polls": mp_polls,
            "flag_share": sum(r["flag_equal"] for r in per.values()) / max(1, flag_polls), "flag_polls": flag_polls,
            "list_ok_share": sum(r["list_ok"] for r in per.values()) / max(1, sum(r["polls"] for r in per.values())),
            "effects": {f"{i}:{e['name']}": {k: e[k] for k in ("values", "units", "polls")}
                        for i, e in sorted(effects.items())},
            "cost_ms_median": float(np.median(cost)) if cost else None}


def morale_series(mor, name):
    """{t, st, pts, tgt, mp, ok, w, r, fx {effect name: [points per poll; None where the list was not read]}, tm, tim,
    uf {field: [..]}} of one unit."""
    out = {"t": [], "st": [], "pts": [], "tgt": [], "mp": [], "ok": [], "w": [], "r": [], "fx": {}, "tm": [], "tim": [],
           "uf": {}}
    n = 0
    for p in mor["polls"]:
        u = p["units"].get(name)
        if u is None:
            continue
        out["t"].append(p["t"])
        for k in ("st", "pts", "tgt", "mp", "ok", "w", "r"):
            out[k].append(u.get(k))
        seen = {}
        for i, v in _pairs(u.get("fx")):
            seen[effect_name(int(i))] = seen.get(effect_name(int(i)), 0) + v
        for k in set(out["fx"]) | set(seen):
            out["fx"].setdefault(k, [0] * n).append(seen.get(k, 0) if u.get("ok") else None)
        out["tm"].append(_pairs(u.get("tm"), 3))
        out["tim"].append(list(u["tim"].values()) if isinstance(u.get("tim"), dict) else u.get("tim"))
        for k, v in (u.get("uf") or {}).items():
            out["uf"].setdefault(k, [None] * n).append(v)
        for k in out["uf"]:
            if len(out["uf"][k]) < n + 1:
                out["uf"][k].append(None)
        n += 1
    return out


def spans(t, on):
    """[(from, to)] the runs of True in `on` over the times t."""
    out, start = [], None
    for ti, x in zip(t, on):
        if x and start is None:
            start = ti
        if not x and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--men", action="store_true", help="the men of a charge probe run (module doc)")
    parser.add_argument("--morale", action="store_true", help="the morale of a morale probe run (module doc)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.morale:
        runs = args.runs or [p for p in sorted(MORALE_RUNS.glob("*")) if has_mem(p)][-1:]
        res = []
        for r in runs:
            mor = load_morale(r)
            chk = morale_check(mor)
            chk["series"] = {name: morale_series(mor, name) for name in chk["units"]}
            res.append(chk)
            print(f"{chk['run']}: attach {chk['attach_ok']}, polls {chk['polls']}, list ok {chk['list_ok_share']:.4f}, "
                  f"MoralePercent = points / base {chk['mp_share']:.4f} of {chk['mp_polls']}, flags = state "
                  f"{chk['flag_share']:.4f} of {chk['flag_polls']}, cost ms median {chk['cost_ms_median']}, "
                  f"errors {chk['errors']}")
            for name, row in chk["units"].items():
                print("  ", name, row)
            for k, e in chk["effects"].items():
                print("  ", k, e)
        out = args.out or MORALE_OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(out)
        return 0
    if args.men:
        runs = args.runs or [p for p in sorted(MEN_RUNS.glob("*")) if has_mem(p)][-1:]
        res = [men_check(load_men(r)) for r in runs]
        for r in res:
            print(f"{r['run']}: attach {r['attach_ok']}, polls {r['polls']}, share {r['share']}, cost ms median "
                  f"{r['cost_ms_median']} max {r['cost_ms_max']}, polls over a tick {r['polls_spanning_ticks']}, "
                  f"errors {r['errors']}")
            for name, row in r["units"].items():
                print("  ", name, row)
        out = args.out or MEN_OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(out)
        return 0
    args.out = args.out or OUT
    runs = args.runs or [p for p in sorted(RUNS.glob("*")) if has_mem(p)][-1:]
    res = [compare(load(r)) for r in runs]
    for r in res:
        print(f"{r['run']}: attach {r['attach_ok']}, hook {r['hook']}, ammo hook {r.get('ammo_hook')}, spawned {r['spawned']}, errors {r['n_errors']}")
        print("  chain first", r["chain_first"], "last", r["chain_last"], "discover", r["discover"])
        if not r["attach_ok"]:
            print("  attach:", json.dumps(r["attach"], ensure_ascii=False))
        if "lanes" in r:
            print(f"  spawn->shooter {r['spawn_to_shooter_m']}, ended {r['ended_share']:.3f}, states {r['states_seen']},"
                  f" matched {r['matched_share']:.3f}, fly ticks {r['fly_ticks']} max {r['fly_max']}")
            for name, row in r["lanes"].items():
                print("  ", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
