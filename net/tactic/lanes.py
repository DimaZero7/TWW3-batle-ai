"""Set P of drill T0-2 (tmp/tactic_slice/design_attack_v23.md section 4 'Sets', check 11; review_attack_v23 F, G.6, H):
copies of the game's recorded starts as scenes, played by the scene runner (stage0/battle.play_tactic, scene_battles,
SceneEnemy). A lane's start = the game's recorded rows at the probe's go (as charge_probe.sim_lanes reads them): mine
A (the anvil, the lane's first extra; slot 1) and B (the lane's attacker; slot 2), theirs E1 (the target; PER_SIDE + 1)
and E2 (the second target; + 2) - flank2v1: mine the anvil and the hammer, theirs the target; the ordered widths, the
recorded men; fearless as sim_lanes makes them (leadership and morale + 1e4; the units with their morale: a_morale,
extras' morale, t_morale); E1 / E2 / the target attack A (the anvil) from the go; the lords lord_back_m behind each side
(the twin's far 'keep' units); the flat field; `copies` a start, x / z jittered +- jitter_m as the twin's (config
drills.attack.sets.P). Plans: tight (the night's 200 ms lanes hold, tight, pinE1, pinE2, e1now), tight01 (its 100 ms
rerun: hold, tight, pinE1), intercept (meetfar part 2: hold, far, near, pinE1, pinE2), flank2v1.

    python -m net.tactic.lanes all --out tmp/tactic_slice/v23/stage3/P      (in the container: bash ops/dock.sh ...)

    built    the built state (review G.6 'exact'): each start without jitter, the runner's battle as built
             (matchup.duel.build of scene_battles) against sim_lanes' (its battle.run wrapped: the state taken, not run),
             unit by unit (same key and side, nearest place): x, z, b, men, hp, front, depth; leadership after the start
             (the runner's TacticSide.start against the twin's fearless rule)
    lanes    check 11: the lanes' own scripted orders through the scene runner - C plays every copy (the first melee
             flags, B - E2 distance), then each kind's order is put in by replay.Branch 'at' the first tactic tick at or
             after the lane's trigger (masks off: the rest is C's) - measured by unit_probe.intercept_measure on the
             runner's rows (lane-shaped, 1 s), against the twin (unit_probe._mf_lanes 'twin') on the same starts, per
             kind: within two errors (2 sqrt(se_runner^2 + se_twin^2), copies as units) on trade 30 / 60 / 120 s
    teach    TEACH on every copy beside C: TEACH - C by plan and kind (the plain trade at the end and the lane's trade);
             on tight's starts B's first attack order (its target, its tick from A's first melee flag) against the lanes'
             scripted ones, by cell (D, dx)
    game     the game's numbers beside the twin's on HEAD's simulator: tight01 (100 ms) by battle, flank2v1 against stand
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from net.stage0 import battle as BA
from net.stage0 import cfg as scfg
from net.tactic import exam_pin as EP

PS = BA.PER_SIDE
ROLES = {"lane": (("A", 1), ("B", 2), ("E1", PS + 1), ("E2", PS + 2)),
         "flank2v1": (("AN", 1), ("H", 2), ("T", PS + 1))}
KINDS = {"tight": ("hold", "tight", "pinE1", "pinE2", "e1now"), "tight01": ("hold", "tight", "pinE1"),
         "intercept": ("hold", "far", "near", "pinE1", "pinE2"), "flank2v1": None}
FIELDS = ("x", "z", "b", "men", "hp_abs", "m", "r", "w", "charge", "flank_hit", "cmb")
OUTF = {"hp_abs": "hp", "charge": "chg", "flank_hit": "fh"}


def _c():
    return scfg()["drills"]["attack"]["sets"]["P"]


def runs_of(plan):
    """The game's run folders of a plan (the evidence files'; written on Windows: '/' put in)."""
    return [d.replace("\\", "/") for d in _runs_of(plan)]


def _runs_of(plan):
    ev = Path("game/probe/evidence")
    if plan in ("tight", "tight01"):
        d = json.loads((ev / "tight_game.json").read_text(encoding="utf-8"))
        return d["runs"] if plan == "tight" else [x["run"] for x in d["step01"]["battles"].values()]
    if plan == "intercept":
        return json.loads((ev / "meetfar_game.json").read_text(encoding="utf-8"))["runs"]["intercept"]
    return json.loads((ev / "flank2v1_game.json").read_text(encoding="utf-8"))["runs"]


def game_lanes(plan):
    from game.probe import charge_probe as cp
    out = []
    for d in runs_of(plan):
        for ln in cp.load_run(Path(d)):
            if ln.get("plan") != plan or ln["spec"]["kind"] == "clock":
                continue
            k = ln["spec"]["kind"].replace("_sw", "")
            if KINDS[plan] is None or k in KINDS[plan]:
                ln["run_dir"] = Path(d).name
                out.append(ln)
    return out


def _t2_key(spec):
    from game.probe import charge_probe as cp
    t2 = spec["target2"]
    t2 = t2 if t2 in cp.UNITS else t2.split("_", 1)[1].rsplit("_", 1)[0][:-1]
    return cp.UNITS[t2][0]


def units_of(ln):
    """[(role, side, key, row, width, keeps morale)] of a lane's start (sim_lanes' rule)."""
    from game.probe import charge_probe as cp
    sp, s0 = ln["spec"], ln["samples"][0]
    ex = ln["extras"][0][1][0] if ln.get("extras") else None
    e0 = (sp.get("extras") or [{}])[0]
    both = bool(sp.get("morale_both"))
    a_keep = bool(sp.get("a_morale")) or both
    t_keep = bool(sp.get("t_morale")) or both
    anvil = (cp.UNITS[e0["short"]][0], ex, e0.get("width", sp["a_width"]), bool(e0.get("morale")))
    if ln["plan"] == "flank2v1":
        return [("AN", 1, *anvil), ("H", 1, sp["a_key"], s0[1], sp["a_width"], a_keep),
                ("T", 2, sp["t_key"], s0[2], sp["t_width"], t_keep)]
    e2 = ln["t2"][0][1]
    return [("A", 1, *anvil), ("B", 1, sp["a_key"], s0[1], sp["a_width"], a_keep),
            ("E1", 2, sp["t_key"], s0[2], sp["t_width"], t_keep), ("E2", 2, _t2_key(sp), e2, sp["t_width"], False)]


def scene_of(ln, rnd, jitter):
    """The scene spec of a lane's start (module doc)."""
    from net.tactic import scenes as SN
    us = units_of(ln)
    sp = ln["spec"]
    side = {1: [], 2: []}
    for role, s, key, row, width, keep in us:
        x = float(row["x"]) + (rnd.uniform(-1, 1) * jitter if jitter else 0.0)
        z = float(row["z"]) + (rnd.uniform(-1, 1) * jitter if jitter else 0.0)
        m0 = SN._men0(key)
        worn = None if float(row["men"]) >= m0 else float(row["men"]) / m0
        side[s].append({"role": role, "key": key, "place": (x, z, float(row["b"]), False, float(width)), "worn": worn,
                        "fearless": not keep})
    back = float(_c()["lord_back_m"])
    cen = {s: (np.mean([u["place"][0] for u in side[s]]), np.mean([u["place"][1] for u in side[s]])) for s in (1, 2)}
    lords = {}
    for s, o in ((1, 2), (2, 1)):
        dx, dz = cen[s][0] - cen[o][0], cen[s][1] - cen[o][1]
        n = math.hypot(dx, dz) or 1.0
        lords[s] = (float(cen[s][0] + back * dx / n), float(cen[s][1] + back * dz / n),
                    math.degrees(math.atan2(-dx, -dz)) % 360.0)
    run = sp.get("t_attack_extra") == "run"
    return {"kind": "lane", "plan": ln["plan"], "lane_kind": sp["kind"], "recipe": "P_" + ln["plan"],
            "fac": {1: "emp", 2: "skv"}, "factions": {1: scfg()["factions"]["emp"], 2: scfg()["factions"]["skv"]},
            "map": "flat", "east": False, "enemy": "scene", "limit": int(sp["max_s"]), "reserve": [],
            "mine": [u["key"] for u in side[1]], "theirs": [u["key"] for u in side[2]],
            "places_mine": [u["place"] for u in side[1]], "places_theirs": [u["place"] for u in side[2]],
            "worn": {"mine": [u["worn"] for u in side[1]], "theirs": [u["worn"] for u in side[2]]},
            "fearless": {"mine": [u["fearless"] for u in side[1]], "theirs": [u["fearless"] for u in side[2]]},
            "lords": lords, "plan_theirs": [{"kind": "at", "k": 0, "run": run} for _ in side[2]],
            "D": sp.get("D"), "dx": sp.get("dx"), "run_dir": ln.get("run_dir"), "battle": ln.get("battle"),
            "lane": sp.get("name")}


def p_specs(plan, copies=None, jitter=None):
    """(lanes, specs): every lane of the plan, `copies` scenes each (config P)."""
    c = _c()
    copies = int(c["copies"] if copies is None else copies)
    jitter = float(c["jitter_m"] if jitter is None else jitter)
    lanes = game_lanes(plan)
    out = []
    for j, ln in enumerate(lanes):
        for k in range(copies):
            out.append(scene_of(ln, random.Random(int(c["seed"]) + 1000 * j + k), jitter))
    return lanes, out


# ------------------------------------------------------------------ the runner's rows
class LaneWatch(EP.SceneWatch):
    """SceneWatch + the lanes' units' rows every second: rows [T][B, role, FIELDS] (numpy, by the plan's ROLES)."""
    ROLE_SLOTS = None

    def _init(self, st):
        super()._init(st)
        self.rows, self.secs = [], []
        self.lead0 = None

    def __call__(self, sec, st):
        super().__call__(sec, st)
        u = st.u
        sl = torch.tensor(self.ROLE_SLOTS, device=self.dev)
        if self.lead0 is None:
            self.lead0 = u["leadership"][:, sl].cpu().numpy()
        cols = []
        for k in FIELDS:
            v = u.get(k)
            cols.append(torch.full((self.B, len(self.ROLE_SLOTS)), float("nan"), device=self.dev) if v is None
                        else v[:, sl].float())
        self.rows.append(torch.stack(cols, -1).cpu().numpy())
        self.secs.append(int(sec))


def play(name, specs, a, P, plan=None, kind="lane"):
    """(the player file, rows [B, T, role, F], secs): `name` a player of pin.PLAYERS, or plan {b: Branch state} for
    'C' with the lanes' scripted orders."""
    from net.tactic import replay as RP
    slots = [s for _, s in ROLES[kind]]

    class W_(LaneWatch):
        ROLE_SLOTS = slots
    got = {}

    def look(ii, watch, side):
        R = np.stack(watch.rows, 1)                                      # [B, T, role, F]
        for k, b in enumerate(ii):
            got[b] = R[k]
        if len(watch.secs) > len(got.get("secs", [])):
            got["secs"] = watch.secs
        got.setdefault("lead0", {}).update({b: watch.lead0[k].tolist() for k, b in enumerate(ii)})
    chunk = 256
    seeds = [int(_c()["tseed"]) + k for k in range((len(specs) + chunk - 1) // chunk)]
    make = None
    if plan is not None:
        def make(ii):
            return RP.Branch("C", {k: plan[b] for k, b in enumerate(ii) if b in plan}, 1e6, masks=False)
    res = EP.run_player("C" if plan is not None else name, specs, seeds, chunk, a, P, make=make, on_chunk=look,
                        watch_cls=(lambda P_, g, chain, ii: W_(P_, g, chain=chain)) if plan is not None else W_,
                        chunk_args=plan is not None, full=True, by_limit=True, battles=BA.scene_battles,
                        enemy=BA.SceneEnemy)
    T = max(len(got[b]) for b in range(len(specs)))
    rows = np.full((len(specs), T, len(slots), len(FIELDS)), np.nan, dtype=np.float32)
    for b in range(len(specs)):
        rows[b, :len(got[b])] = got[b]
    return res, rows, got["secs"], got["lead0"]


def lane_of(ln, spec, rows, secs, order=None):
    """The runner's battle as a game-shaped lane (unit_probe._roles reads it): samples (t, B, E1, None), t2, extras;
    phases a_go at the order (sec, centre distance)."""
    def row(r):
        d = {}
        for j, k in enumerate(FIELDS):
            v = float(r[j])
            if math.isfinite(v):
                d[OUTF.get(k, k)] = bool(v) if k in ("m", "r", "w") else v
        return d if d else None
    t = [float(s) for s in secs[:rows.shape[0]]]
    smp, t2, ex = [], [], []
    for i, tt in enumerate(t):
        A, B, E1, E2 = (row(rows[i, j]) for j in range(4))
        smp.append((tt, B, E1, None))
        t2.append((tt, E2))
        ex.append((tt, [A]))
    out = {"run": "runner", "plan": ln["plan"], "spec": ln["spec"], "samples": smp, "t2": t2, "extras": ex,
           "contacts": {}, "phases": [], "men": []}
    if order is not None:
        sec, tg = order
        i = min(int(sec), rows.shape[0] - 1)
        d = math.hypot(float(rows[i, 1, 0] - rows[i, tg, 0]), float(rows[i, 1, 1] - rows[i, tg, 1]))
        out["phases"] = [{"phase": "a_go", "t": float(sec) * 1000, "d": d, "why": "runner"}]
    return out


def _first(v):
    k = np.nonzero(v)[0]
    return int(k[0]) if len(k) else None


def triggers(lanes, specs, rows, secs, copies):
    """{b: Branch state} of the lanes' scripted orders from C's run (module doc): the trigger second by the lane's own
    rule, the order at the first tactic tick at or after it."""
    from game.probe import charge_probe as cp
    tick = int(EP.tcfg()["decide_s"])
    plan, info = {}, {}
    for b, s in enumerate(specs):
        ln = lanes[b // copies]
        sp = ln["spec"]
        if sp["mode"] not in ("attack_t2", "attack_run"):
            continue
        tg = 3 if sp["mode"] == "attack_t2" else 2                          # role index: E2 / E1
        R = rows[b]
        m = np.nan_to_num(R[:, :, FIELDS.index("m")]) > 0
        trig = None
        if sp.get("a_go_melee_s") is not None:
            of = 0 if sp.get("a_go_melee_of") == "e1" else tg               # A's flag / the target's own
            f0 = _first(m[:, of])
            trig = None if f0 is None else secs[f0] + float(sp["a_go_melee_s"])
        elif sp.get("a_go_m") is not None:
            d = np.hypot(R[:, 1, 0] - R[:, tg, 0], R[:, 1, 1] - R[:, tg, 1])
            within = _first(d <= float(sp["a_go_m"]))
            if within is not None:
                if sp.get("a_go_closest"):
                    lo, k = d[within], within
                    for j in range(within, len(d)):
                        lo = min(lo, d[j])
                        if d[j] > lo + 2.0 or m[j, tg]:
                            k = j
                            break
                    trig = secs[k]
                else:
                    trig = secs[within]
        if trig is None:
            info[b] = {"trigger_s": None}
            continue
        at = int(math.ceil(trig / tick) * tick)
        plan[b] = {"t": at, "i": 2, "e": PS + 1 + (tg - 2), "u": -1, "kind": "at", "at": at, "cause": 3}
        info[b] = {"trigger_s": float(trig), "order_tick": at}
    return plan, info


def measure(lanes, specs, rows, secs, copies, orders=None):
    from game.probe import unit_probe as up
    out = []
    for b, s in enumerate(specs):
        ln = lanes[b // copies]
        o = (orders or {}).get(b)
        r = up.intercept_measure(lane_of(ln, s, rows[b], secs, o), up.TG_WINDOWS)
        r.update(battle=ln.get("battle"), lane=ln["spec"]["name"], run=ln.get("run_dir"), b=b)
        out.append(r)
    return out


def _ms(v):
    v = [float(x) for x in v if x is not None and math.isfinite(float(x))]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return [round(m, 4), round(sd / math.sqrt(len(v)), 4), len(v)]


# ------------------------------------------------------------------ the commands
def built(a, P, D):
    """The built state against the twin's (module doc)."""
    from game.probe import charge_probe as cp
    from net.matchup import duel as DU
    from sim import battle as SB
    from sim import geometry
    out = {}
    for plan in ("tight", "tight01", "intercept", "flank2v1"):
        lanes, specs = p_specs(plan, copies=1, jitter=0.0)
        st, _ = DU.build(BA.NAME, BA.scene_battles(specs), a.device, P, per_side=PS, maps=["flat"] * len(specs))
        cap = {}
        real = SB.run

        def grab(st_, *x, **k):
            cap["u"] = {kk: v.detach().cpu().clone() for kk, v in st_.u.items() if torch.is_tensor(v)}
        SB.run = grab
        try:
            cp.sim_lanes(lanes, params=P, device=a.device, copies=1, jitter_m=0.0, seed=0)
        finally:
            SB.run = real
        tw = cap["u"]
        mu = {k: v.detach().cpu() for k, v in st.u.items() if torch.is_tensor(v)}
        sp = P.sim["formation"]["spacing_m"]
        fm, dm = geometry.dims(st.u, sp)
        ft, dt = geometry.dims({k: v.to(a.device) for k, v in tw.items()}, sp)
        fm, dm, ft, dt = fm.cpu(), dm.cpu(), ft.cpu(), dt.cpu()
        worst = {k: 0.0 for k in ("x", "z", "b", "men", "hp_abs", "front", "depth")}
        miss = 0
        for b in range(len(specs)):
            for i in range(mu["x"].shape[1]):
                if float(mu["men"][b, i]) <= 0 or bool(mu["lord"][b, i]):
                    continue
                cand = [(float((tw["x"][b, j] - mu["x"][b, i]) ** 2 + (tw["z"][b, j] - mu["z"][b, i]) ** 2), j)
                        for j in range(tw["x"].shape[1]) if float(tw["men"][b, j]) > 0
                        and int(tw["side"][b, j]) == int(mu["side"][b, i]) and not bool(tw["lord"][b, j])]
                if not cand:
                    miss += 1
                    continue
                j = min(cand)[1]
                for k in ("x", "z", "men", "hp_abs"):
                    worst[k] = max(worst[k], abs(float(mu[k][b, i] - tw[k][b, j])))
                worst["b"] = max(worst["b"], abs((float(mu["b"][b, i] - tw["b"][b, j]) + 180) % 360 - 180))
                worst["front"] = max(worst["front"], abs(float(fm[b, i] - ft[b, j])))
                worst["depth"] = max(worst["depth"], abs(float(dm[b, i] - dt[b, j])))
        out[plan] = {"starts": len(specs), "units_unmatched": miss, "worst_abs_diff": {k: round(v, 4) for k, v in worst.items()},
                     "exact": miss == 0 and all(v < 1e-3 for v in worst.values())}
        print(json.dumps({"built": plan, **out[plan]}), flush=True)
    (D / "built.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _rows_file(D, plan, who):
    return D / f"{plan}_{who}.npz"


def runner(a, P, D):
    """C, the scripted lanes and TEACH on every copy of each plan (module doc); files <D>/<plan>_<who>.npz / .json."""
    copies = int(_c()["copies"])
    for plan in ("tight", "tight01", "intercept", "flank2v1"):
        lanes, specs = p_specs(plan)
        kind = "flank2v1" if plan == "flank2v1" else "lane"
        for who in ("C", "scripted", "TEACH"):
            if who == "scripted" and plan == "flank2v1":
                continue                        # (the hammer's scripted path is a move, not a tactic order)
            f = _rows_file(D, plan, who)
            if f.exists() and f.with_suffix(".json").exists():
                continue
            t0 = time.time()
            pl, info = None, {}
            if who == "scripted":
                z = np.load(_rows_file(D, plan, "C"))
                pl, info = triggers(lanes, specs, z["rows"], [int(x) for x in z["secs"]], copies)
            res, rows, secs, lead0 = play(who, specs, a, P, plan=pl, kind=kind)
            np.savez_compressed(f, rows=rows, secs=np.array(secs))
            res["triggers"] = {str(b): v for b, v in info.items()}
            res["lead0"] = {str(b): v for b, v in lead0.items()}
            res["specs_meta"] = [{k: s.get(k) for k in ("plan", "lane_kind", "D", "dx", "run_dir", "battle", "lane")}
                                 for s in specs]
            f.with_suffix(".json").write_text(json.dumps(res), encoding="utf-8")
            print(json.dumps({"plan": plan, "who": who, "battles": len(specs), "min": round((time.time() - t0) / 60, 1)}),
                  flush=True)


def twin(a, P, D):
    """The twin (unit_probe on the recorded starts) on the CPU, as its evidence runs (one batch of every lane's copies:
    on the GPU each new batch shape would be compiled anew)."""
    from game.probe import unit_probe as up
    for plan in ("tight", "tight01", "intercept"):
        f = D / f"{plan}_twin.json"
        if f.exists():
            continue
        rows = up._mf_lanes(plan, "twin", runs_of(plan), copies=int(_c()["copies"]), device="cpu", params=P)
        f.write_text(json.dumps(rows, default=float), encoding="utf-8")
    f = D / "flank2v1_twin.json"
    if not f.exists():
        rows = up.flank2v1_lanes("twin", runs_of("flank2v1"), copies=int(_c()["copies"]), device="cpu", params=P)
        f.write_text(json.dumps(rows, default=float), encoding="utf-8")


def _kind(k):
    return k.replace("_sw", "")


def check11(D):
    """The table of check 11 and of TEACH on P (module doc)."""
    copies = int(_c()["copies"])
    keys = ("trade_30", "trade_60", "trade_120")
    out = {"rule": "per kind: |runner - twin| <= 2 sqrt(se_r^2 + se_t^2) on trade 30 / 60 / 120 s (copies as units)"}
    for plan in ("tight", "tight01", "intercept"):
        lanes, specs = p_specs(plan)
        tw = [r for r in json.loads((D / f"{plan}_twin.json").read_text(encoding="utf-8"))
              if KINDS[plan] is None or _kind(r["kind"]) in KINDS[plan]]
        z = np.load(_rows_file(D, plan, "scripted"))
        sj = json.loads(_rows_file(D, plan, "scripted").with_suffix(".json").read_text(encoding="utf-8"))
        orders = {int(b): (v["order_tick"], 3 if lanes[int(b) // copies]["spec"]["mode"] == "attack_t2" else 2)
                  for b, v in sj["triggers"].items() if v.get("order_tick") is not None}
        mr = measure(lanes, specs, z["rows"], [int(x) for x in z["secs"]], copies, orders)
        tab = {}
        for k in dict.fromkeys(_kind(s["lane_kind"]) for s in specs):
            R = [r for r in mr if _kind(r["kind"]) == k]
            T = [r for r in tw if _kind(r["kind"]) == k]
            row = {"n_runner": len(R), "n_twin": len(T)}
            ok = True
            for f in keys + ("order_s", "m_E2", "b_e2_s", "A_rout_s", "B_rout_s"):
                x, y = _ms([r.get(f) for r in R]), _ms([r.get(f) for r in T])
                row[f] = {"runner": x, "twin": y}
                if f in keys and x and y:
                    e = 2 * math.sqrt(x[1] ** 2 + y[1] ** 2)
                    row[f]["diff"] = round(x[0] - y[0], 4)
                    row[f]["two_errors"] = round(e, 4)
                    row[f]["within"] = abs(x[0] - y[0]) <= e
                    ok &= row[f]["within"]
            row["intercepted"] = {"runner": _ms([float(bool(r.get("intercepted"))) for r in R]),
                                  "twin": _ms([float(bool(r.get("intercepted"))) for r in T])}
            row["within_all"] = ok
            tab[k] = row
        out[plan] = tab
    # TEACH on P
    out["teach"] = {}
    for plan in ("tight", "tight01", "intercept", "flank2v1"):
        lanes, specs = p_specs(plan)
        dC = json.loads(_rows_file(D, plan, "C").with_suffix(".json").read_text(encoding="utf-8"))
        dT = json.loads(_rows_file(D, plan, "TEACH").with_suffix(".json").read_text(encoding="utf-8"))
        hpC, hpT = dC["per_battle"]["hp_trade"], dT["per_battle"]["hp_trade"]
        tb = {"all": EP.paired(hpT, hpC)}
        if plan != "flank2v1":
            zC, zT = np.load(_rows_file(D, plan, "C")), np.load(_rows_file(D, plan, "TEACH"))
            mC = measure(lanes, specs, zC["rows"], [int(x) for x in zC["secs"]], copies)
            mT = measure(lanes, specs, zT["rows"], [int(x) for x in zT["secs"]], copies)
            for w in (60, 120):
                tb[f"lane_trade_{w}"] = EP.paired([r.get(f"trade_{w}") if r.get(f"trade_{w}") is not None else math.nan
                                                   for r in mT],
                                                  [r.get(f"trade_{w}") if r.get(f"trade_{w}") is not None else math.nan
                                                   for r in mC])
        for k in dict.fromkeys(_kind(s["lane_kind"]) for s in specs):
            sel = [b for b, s in enumerate(specs) if _kind(s["lane_kind"]) == k]
            tb[f"kind {k}"] = EP.paired(hpT, hpC, sel)
        tb["orders_unit_min"] = EP.paired(dT["per_battle"]["game_orders_min"], dC["per_battle"]["game_orders_min"])
        out["teach"][plan] = tb
        if plan in ("tight", "tight01"):
            out["teach"][plan]["B_first_order_by_cell"] = b_first(dT, specs, np.load(_rows_file(D, plan, "TEACH")))
    (D / "check11.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def b_first(dT, specs, z):
    """On tight's starts: TEACH's B (slot 2) - its first attack order: target E1 / E2, its tick from A's first melee
    flag (and from E2's), by cell (D, dx); the lane it reproduces (scripted: tight E2 at A + 2.5 s, e1now E1 at A + 2.5,
    pinE1 E1 at A + 8, pinE2 E2 at E2 + 8; hold none) - the nearest by the tick on the same target."""
    ci = {k: i for i, k in enumerate(dT["order_log_cols"])}
    first = {}
    for r in sorted(dT["order_log"], key=lambda r: r[ci["sec"]]):
        b, i = int(r[ci["b"]]), int(r[ci["slot"]])
        if i == 2 and b not in first:
            first[b] = (float(r[ci["sec"]]), int(r[ci["target"]]))
    rows, secs = z["rows"], [int(x) for x in z["secs"]]
    m = np.nan_to_num(rows[..., FIELDS.index("m")]) > 0
    cells = {}
    for b, s in enumerate(specs):
        key = f"D{s['D']} dx{s['dx']}"
        c = cells.setdefault(key, {"copies": 0, "no_order": 0, "E1": [], "E2": [], "other": 0, "lane": {}})
        c["copies"] += 1
        fa, fe2 = _first(m[b, :, 0]), _first(m[b, :, 3])
        if b not in first:
            c["no_order"] += 1
            c["lane"]["hold"] = c["lane"].get("hold", 0) + 1
            continue
        sec, tg = first[b]
        rel_a = None if fa is None else sec - secs[fa]
        rel_e2 = None if fe2 is None else sec - secs[fe2]
        if tg == PS + 1:
            c["E1"].append(rel_a)
            lane = "e1now" if rel_a is not None and rel_a < 5.25 else "pinE1"
        elif tg == PS + 2:
            c["E2"].append([rel_a, rel_e2])
            lane = "tight" if (rel_a is not None and rel_a < 5.25 and (rel_e2 is None or rel_e2 < 0)) else "pinE2"
        else:
            c["other"] += 1
            lane = "other"
        c["lane"][lane] = c["lane"].get(lane, 0) + 1
    for c in cells.values():
        c["E1_from_A_flag_s"] = _ms(c.pop("E1"))
        e2 = c.pop("E2")
        c["E2_from_A_flag_s"] = _ms([x[0] for x in e2])
        c["E2_from_E2_flag_s"] = _ms([x[1] for x in e2])
    return cells


def game_vs_twin(D):
    """tight01 (100 ms) by battle and flank2v1 against stand: the game's (evidence files) beside the twin's (HEAD)."""
    from game.probe import unit_probe as up
    ev = Path("game/probe/evidence")
    tg = json.loads((ev / "tight_game.json").read_text(encoding="utf-8"))["step01"]["game"]["tables_by_battle"]
    tw = json.loads((D / "tight01_twin.json").read_text(encoding="utf-8"))
    out = {"tight01": {"game_0.1s": tg.get("game_0.1s"), "twin_HEAD": {}}}
    for x, base in (("tight", "pinE1"), ("tight", "hold"), ("pinE1", "hold")):
        p = up.mf_paired(tw, base, ("trade_60", "trade_120", "trade_240"), sim_within=True).get(x, {})
        for w in (60, 120, 240):
            out["tight01"]["twin_HEAD"][f"{x}-{base}_{w}"] = p.get(f"trade_{w}")
    f2 = json.loads((ev / "flank2v1_game.json").read_text(encoding="utf-8"))
    rows = json.loads((D / "flank2v1_twin.json").read_text(encoding="utf-8"))
    for r in rows:
        r["run"] = "sim"
    keys = tuple(f2["paired_game"])
    pt = up.mf_paired(rows, lambda k: k.split("_")[0] + "_stand", keys)
    out["flank2v1"] = {"game": f2["paired_game"], "twin_HEAD": {f: {k: v.get(f) for k, v in pt.items()} for f in keys},
                       "twin_evidence": f2.get("paired_twin"), "evidence_twin_sim": f2.get("sim")}
    (D / "game_vs_twin.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("built", "run", "twin", "table", "all"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="tmp/tactic_slice/v23/stage3/P")
    ap.add_argument("--sim", default="")
    a = ap.parse_args(argv)
    from sim.params import load
    P = load()
    D = Path(a.out)
    D.mkdir(parents=True, exist_ok=True)
    if a.what in ("built", "all"):
        built(a, P, D)
    if a.what in ("run", "all"):
        runner(a, P, D)
    if a.what in ("twin", "all"):
        twin(a, P, D)
    if a.what in ("table", "all"):
        out = check11(D)
        g = game_vs_twin(D)
        print(json.dumps(out, indent=1, default=float), flush=True)
        print(json.dumps(g, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
