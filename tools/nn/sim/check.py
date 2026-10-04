"""Checks of the simulator against the game (docs/en/training/simulator.md).

Every fair recorded run of the measurement arenas and of the mirror arena is replayed in the
simulator open-loop (tools/nn/sim/replay.py: the recorded orders, from the recorded start), the
simulated battle is written down per second like a recording (a gamedata.Battle) and measured
by the same code as the game (tools/nn/measure.py). Then the speed: many battles at once.

    python -m tools.nn.sim.check                    # all checks, CPU
    bash tools/nn/dock.sh tools.nn.sim.check        # in the container: CPU and GPU
    python -m tools.nn.sim.check --only mechanics   # mechanics | battles | speed

Prints a table; writes build/nn-sim/check.json (not in Git).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools import config as project
from tools.nn import gamedata, measure
from tools.nn import scenario as arena_scenario
from tools.nn.sim import battle, replay, scenario
from tools.nn.sim.params import load

OUT = project.BUILD / "nn-sim" / "check.json"
TOLERANCE = 0.2
FIGHT_NEAREST = True             # replay: a unit in melee without a recorded target attacks the nearest enemy
PLANNER = ("attack", "defend")   # battles of CA's planner against the game's AI (not the network's own runs)
NET = "net"          # the network's battles against the game's AI on generated armies (the gate): reported apart
COPIES = 8           # whole battles: replays of each recorded battle
CURVE_S = (60, 120, 180)   # share of HP lost this long after the first contact
JITTER_M = 2.0       # ... from starts moved by up to this much
BOOLS = gamedata.BOOL_FIELDS
FLOATS = gamedata.FLOAT_FIELDS


class Recorder:
    """Writes the batch down once a simulated second, like the game's recordings."""

    def __init__(self, st, every_s=1.0):
        self.every = every_s
        self.next = 0.0
        self.t = []
        self.rows = []
        self.done_at = [None] * st.B
        self.take(st)

    def take(self, st):
        u = st.u
        snap = {k: u[k].detach().cpu().numpy().copy() for k in FLOATS + BOOLS + ("target", "fat")}
        self.rows.append(snap)
        self.t.append(st.t.detach().cpu().numpy().copy())

    def __call__(self, st):
        t = float(st.t.max())
        if t + 1e-6 >= self.next + self.every:
            self.next += self.every
            self.take(st)
        done = st.done.cpu().numpy()
        for b in range(st.B):
            if done[b] and self.done_at[b] is None:
                self.done_at[b] = len(self.rows)

    def battle(self, b, names, keys, side, slot_of, winner, own_ai, arena):
        """The simulated battle b as a gamedata.Battle in the recording's unit order."""
        end = self.done_at[b] or len(self.rows)
        rows = self.rows[:end + 1]
        idx = np.array(slot_of)
        f = {k: np.stack([r[k][b, idx] for r in rows]).astype(bool if k in BOOLS else float) for k in FLOATS + BOOLS}
        f["fat"] = np.stack([r["fat"][b, idx] for r in rows]).astype(float)
        inv = {s: i for i, s in enumerate(slot_of)}
        tg = np.stack([r["target"][b, idx] for r in rows])
        target = np.vectorize(lambda s: inv.get(int(s), -1))(tg) if tg.size else tg
        t = np.array([r[b] for r in self.t[:end + 1]])
        return gamedata.Battle(run=f"sim-{b}", own_ai=own_ai, enemy_role="?", result={"winner": int(winner),
                               "status": "completed"}, t=t, f=f, target=target, arena=arena, names=names,
                               keys=keys, side=side)


def factions_of(run_dir, arena):
    cfg = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))["config"]
    fac = cfg.get("factions") or {}
    if not fac:
        base = arena_scenario.load_arena(arena)
        fac = {"own": base.get("faction"), "enemy": base.get("faction")}
    return fac


def ahead(st):
    """[B] the side with more of its starting health left in standing units (1 or 2)."""
    u = st.u
    stand = (u["men"] > 0) & ~u["gone"] & ~u["r"]
    share = [((u["hp_abs"] * (stand & (u["side"] == s))).sum(1) / (u["hp0"] * (u["side"] == s)).sum(1).clamp(min=1))
             for s in (1, 2)]
    return torch.where(share[0] >= share[1], 1, 2)


def simulate(run_dirs, params=None, device="cpu", copies=1, jitter_m=0.0, seed=0, end_at_recording=False):
    """Replay recorded runs in one batch, `copies` times each (start places moved by up to
    jitter_m, so the copies differ); returns [(game Battle, [sim Battle per copy], factions)].
    end_at_recording: a simulated battle stops when its recording ends; if it is not over by
    then, the side with more standing health is taken as the winner (Battle.result "cut": True).
    Otherwise the last recorded orders go on until the battle ends."""
    params = params or load()
    games = [gamedata.load(d) for d in run_dirs]
    armies, facs = [], []
    for d, g in zip(run_dirs, games):
        fac = factions_of(d, g.arena)
        army = scenario.from_recording(g, d)
        army["sides"][1]["faction"] = fac.get("own")
        army["sides"][2]["faction"] = fac.get("enemy")
        armies.append(army)
        facs.append(fac)
    H = max(len(a["sides"][s]["units"]) for a in armies for s in (1, 2))
    batch = [a for a in armies for _ in range(copies)]
    st = scenario.build(batch, params, device=device, per_side=H)
    if jitter_m:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for k in ("x", "z"):
            noise = (torch.rand(st.u[k].shape, generator=gen) * 2 - 1) * jitter_m
            st.u[k] = st.u[k] + noise.to(device)
    slot_maps, rows = [], []
    for a, g in zip(armies, games):
        m = scenario.slots(a, H)
        slot_of = [m[n] for n in g.names]
        slot_maps.append(slot_of)
        width = {x["name"]: x.get("width") for side in (1, 2) for x in a["sides"][side]["units"]}
        # who leaves melee on a far recorded point: missile units, and the network's own units (side 1
        # of its runs: their point is the network's move order; replay.recorded_orders)
        missile = [bool((params.units.get(k) or {}).get("missile")) for k in g.keys] if g.keys else [False] * len(g.names)
        leavers = [bool(m) or (g.own_ai == NET and int(s) == 1) for m, s in zip(missile, g.side)]
        # the network's units in melee without a recorded target are under its HOLD (replay.recorded_orders)
        nearest = [FIGHT_NEAREST and not (g.own_ai == NET and int(s) == 1) for s in g.side]
        orders = replay.recorded_orders(g, slot_of, 2 * H, [width.get(n) for n in g.names],
                                        params.sim["formation"]["spacing_m"], nearest,
                                        params.sim["contact"].get("leave_m", 0.0), leavers)
        rows.extend([orders] * copies)
    rec = Recorder(st)
    ends = torch.tensor([float(g.t[-1]) for g in games for _ in range(copies)], device=device)
    cut = torch.zeros(st.B, dtype=torch.bool, device=device)

    def record(s):
        if end_at_recording:
            over = ~s.done & (s.t >= ends - 1e-6)
            if bool(over.any()):
                s.winner = torch.where(over, ahead(s), s.winner)
                s.done = s.done | over
                cut.copy_(cut | over)
        rec(s)
    policy = replay.Replay(rows, device=device, grace_s=0.0 if end_at_recording else 1e9)
    battle.run(st, policy, params, record=record)
    winners = st.winner.cpu().numpy()
    cuts = cut.cpu().numpy()
    out = []
    for k, g in enumerate(games):
        sims = [rec.battle(k * copies + c, g.names, g.keys, g.side, slot_maps[k], winners[k * copies + c], g.own_ai,
                           g.arena) for c in range(copies)]
        for c, sim in enumerate(sims):
            sim.result["cut"] = bool(cuts[k * copies + c])
        out.append((g, sims, facs[k]))
    return out


def rel_err(sim, game):
    if sim is None or game is None or not np.isfinite(sim) or not np.isfinite(game) or game == 0:
        return None
    return (sim - game) / abs(game)


def mean(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def mechanics(params=None, device="cpu"):
    """The melee pairs and the shooting runs: game against simulator, per measured number."""
    params = params or load()
    p = measure.passports()
    runs = [d for d in gamedata.runs() if measure.kind(gamedata.load(d).arena) in ("pair", "missile")]
    pairs = simulate(runs, params, device)
    by_arena = {}
    for g, (s,), _ in pairs:
        k = measure.kind(g.arena)
        if k == "pair":
            gm, sm = measure.melee_pair(g, p), measure.melee_pair(s, p)
        else:
            gm, sm = measure.missile_run(g, p), measure.missile_run(s, p)
        by_arena.setdefault(g.arena, []).append((gm, sm))
    rows = []
    for arena, items in sorted(by_arena.items()):
        k = measure.kind(arena)
        if k == "pair":
            def pick(m, i, key):
                if not m.get("contact"):
                    return None
                return m["units"][i].get(key)
            metrics = [("fight_s", lambda m: m.get("fight_s"))]
            for i in range(2):
                name = items[0][0]["units"][i]["slot"] if items[0][0].get("contact") else str(i)
                for key in ("steady_hp_per_s", "charge_hp_lost", "waver_after_s", "rout_after_s"):
                    metrics.append((f"{name}.{key}", lambda m, i=i, key=key: pick(m, i, key)))
            metrics.append(("winner_side", lambda m: m.get("winner_side")))
        else:
            metrics = [(key, lambda m, key=key: m.get(key)) for key in (
                "first_shot_after_stop_s", "shots_per_man_per_s", "implied_reload_s", "hp_per_s_while_shooting",
                "target_waver_after_first_shot_s", "target_rout_after_first_shot_s")]
            metrics.append(("hit_rate", lambda m: (m.get("by_distance") or [{}])[-1].get("implied_hit_rate")))
        for name, get in metrics:
            game = [get(gm) for gm, _ in items]
            sim = [get(sm) for _, sm in items]
            if name == "winner_side":
                agree = sum(a == b for a, b in zip(game, sim))
                rows.append({"arena": arena, "metric": name, "game": game, "sim": sim, "match": f"{agree}/{len(game)}",
                             "ok": agree == len(game)})
                continue
            gmean, smean = mean(game), mean(sim)
            err = rel_err(smean, gmean)
            if gmean is None:       # the game never saw it (the winner never wavered): nor should the simulator
                rows.append({"arena": arena, "metric": name, "game": None, "game_range": [None, None], "sim": smean,
                             "rel_err": None, "ok": smean is None})
                continue
            rows.append({"arena": arena, "metric": name, "game": gmean,
                         "game_range": [min((x for x in game if x is not None), default=None),
                                        max((x for x in game if x is not None), default=None)],
                         "sim": smean, "rel_err": err, "ok": err is not None and abs(err) <= TOLERANCE})
    return rows


def routs(b):
    """Routs, rallies and how long each rally took (s) over a battle's units."""
    n_rout, n_rally, spans = 0, 0, []
    for i in range(len(b.names)):
        r = b.f["r"][:, i].astype(bool)
        start = None
        for t in range(1, len(r)):
            if r[t] and not r[t - 1]:
                n_rout += 1
                start = t
            elif r[t - 1] and not r[t] and start is not None and b.f["men"][t, i] > 0 and not b.f["s"][t, i]:
                n_rally += 1
                spans.append(float(b.t[t] - b.t[start]))
                start = None
    return n_rout, n_rally, spans


def battles(params=None, device="cpu", copies=COPIES, jitter_m=JITTER_M, net=False):
    """Whole battles: the 10 Empire-Skaven runs and the fair mirror-arena runs, each replayed
    `copies` times from slightly moved starts; the simulator's winner is the majority's.
    net: instead the network's battles against the game's AI on generated armies (both sides'
    recorded orders replayed)."""
    params = params or load()
    p = measure.passports()
    if net:
        runs = [d for d in gamedata.runs(own_ai=NET) if gamedata.load(d).arena.startswith("random")]
    else:
        runs = [d for d in gamedata.runs() if (gamedata.load(d).arena.startswith("whole")
                                               or gamedata.load(d).arena == "arena") and gamedata.load(d).own_ai in PLANNER]
    out = []
    for g, sims, fac in simulate(runs, params, device, copies, jitter_m, end_at_recording=True):
        names = {1: fac.get("own", "?"), 2: fac.get("enemy", "?")}
        votes = [int(x.winner) for x in sims]
        sw = max((1, 2), key=votes.count)
        s = next(x for x in sims if int(x.winner) == sw)
        gw = g.winner
        row = {"run": g.run, "arena": g.arena, "own_ai": g.own_ai, "game_winner": int(gw), "sim_winner": int(sw),
               "sim_winner_share": round(votes.count(sw) / len(votes), 2),
               "sim_cut": round(sum(x.result.get("cut", False) for x in sims) / len(sims), 2),
               "game_s": float(g.t[-1]), "sim_s": float(np.median([x.t[-1] for x in sims])),
               "match": bool(gw == sw) if gw else None}
        for side in (1, 2):
            idx = np.nonzero(g.side == side)[0]
            for tag, b in (("game", g), ("sim", s)):
                hp = sum(measure.hp_abs(b, i, p) for i in idx)
                row[f"{tag}_hp_lost_{side}"] = round(float(1 - hp[-1] / hp[0]), 3)
                row[f"{tag}_men_end_{side}"] = float(np.nansum(b.f["men"][-1, idx]))
                row[f"{tag}_routed_{side}"] = int(sum(bool(b.f["r"][:, i].any()) for i in idx))
        for tag, b in (("game", g), ("sim", s)):
            c = measure.first(b.f["m"].any(axis=1))
            for side in (1, 2):
                idx = np.nonzero(g.side == side)[0]
                hp = sum(measure.hp_abs(b, i, p) for i in idx)
                curve = []
                for after in CURVE_S:
                    k = None if c is None else measure.first(b.t >= b.t[c] + after)
                    k = len(b.t) - 1 if k is None else k
                    curve.append(round(float(1 - hp[k] / hp[0]), 3))
                row[f"{tag}_hp_lost_after_contact_{side}"] = curve
            n_rout, n_rally, spans = routs(b)
            row[f"{tag}_routs"], row[f"{tag}_rallies"], row[f"{tag}_rally_s"] = n_rout, n_rally, spans
        row["factions"] = names
        out.append(row)
    return out


def speed(device="cpu", batch=1024, params=None, seconds=None):
    """Battles a second: `batch` copies of the Empire-Skaven battle (11 slots a side), every unit
    attacking the nearest enemy, run to the end."""
    params = params or load()
    army = scenario.from_arena("whole_emp_v_skv", "attack")

    def fresh():
        st = scenario.build([army] * batch, params, device=device, per_side=20)
        # Different battles: move the starts by up to 10 m.
        g = torch.Generator(device="cpu").manual_seed(0)
        st.u["x"] = st.u["x"] + (torch.rand(st.u["x"].shape, generator=g) * 20 - 10).to(device)
        st.u["z"] = st.u["z"] + (torch.rand(st.u["z"].shape, generator=g) * 20 - 10).to(device)
        return st
    # Warm up (on CUDA the first steps compile the step), then time a fresh batch.
    battle.run(fresh(), replay.nearest_attack, params, until_s=2.0)
    st = fresh()
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0

    def count(_):
        nonlocal steps
        steps += 1
    battle.run(st, replay.nearest_attack, params, until_s=seconds, every_s=1.0, record=count)
    if device != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {"device": device, "batch": batch, "slots": st.N, "steps": steps, "wall_s": round(wall, 2),
            "sim_s": round(float(st.t.max()), 1), "battles_per_s": round(batch / wall, 1),
            "steps_per_s": round(steps / wall, 1), "done": int(st.done.sum()),
            "winner_side2": int((st.winner == 2).sum())}


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("mechanics", "battles", "speed"))
    parser.add_argument("--device", default=None, help="cpu or cuda (default: both where cuda exists, for speed)")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    params = load()
    report = {}
    dev = args.device or "cpu"
    if args.only in (None, "mechanics"):
        rows = mechanics(params, dev)
        report["mechanics"] = rows
        print("== mechanics: game (mean of 3 runs) against the simulator (same recorded orders)")
        print(f"{'arena':26} {'metric':34} {'game':>10} {'range':>17} {'sim':>10} {'err':>7}  ok")
        for r in rows:
            if r["metric"] == "winner_side":
                print(f"{r['arena']:26} {r['metric']:34} {str(r['game']):>10} {'':>17} {str(r['sim']):>10} "
                      f"{r['match']:>7}  {'yes' if r['ok'] else 'NO'}")
                continue
            rng = r["game_range"]
            print(f"{r['arena']:26} {r['metric']:34} {fmt(r['game']):>10} {fmt(rng[0]) + '-' + fmt(rng[1]):>17} "
                  f"{fmt(r['sim']):>10} {fmt(r['rel_err'] and 100 * r['rel_err'], 0) + '%':>7}  "
                  f"{'yes' if r['ok'] else 'NO'}")
        ok = [r["ok"] for r in rows]
        print(f"within {int(TOLERANCE * 100)} %: {sum(ok)} of {len(ok)}")
    if args.only in (None, "battles"):
        rows = battles(params, dev)
        report["battles"] = rows
        print("== whole battles: recorded orders replayed open-loop from the recorded start")
        print(f"{'run':16} {'arena':16} {'own_ai':6} {'winner game/sim (share)':>24} {'s game/sim':>12} "
              f"{'HP lost 1 game/sim':>19} {'HP lost 2 game/sim':>19}")
        for r in rows:
            print(f"{r['run']:16} {r['arena']:16} {r['own_ai']:6} {r['game_winner']:>9}/{r['sim_winner']} "
                  f"({r['sim_winner_share']:.2f})       "
                  f"{r['game_s']:>6.0f}/{r['sim_s']:<5.0f} {r['game_hp_lost_1']:>9.2f}/{r['sim_hp_lost_1']:<9.2f} "
                  f"{r['game_hp_lost_2']:>9.2f}/{r['sim_hp_lost_2']:<9.2f}")
        summary = {}
        for group in ("whole", "arena"):
            rs = [r for r in rows if (r["arena"].startswith("whole") if group == "whole" else r["arena"] == "arena")
                  and r["match"] is not None]
            if rs:
                summary[group] = {"decided": len(rs), "same_winner": sum(r["match"] for r in rs),
                                  "share": round(sum(r["match"] for r in rs) / len(rs), 3)}
        allr = [r for r in rows if r["match"] is not None]
        summary["all"] = {"decided": len(allr), "same_winner": sum(r["match"] for r in allr),
                          "share": round(sum(r["match"] for r in allr) / max(len(allr), 1), 3)}
        for tag in ("game", "sim"):
            spans = [x for r in rows for x in r[f"{tag}_rally_s"]]
            summary[f"{tag}_routs_per_battle"] = round(float(np.mean([r[f"{tag}_routs"] for r in rows])), 1)
            summary[f"{tag}_rallies_per_battle"] = round(float(np.mean([r[f"{tag}_rallies"] for r in rows])), 1)
            summary[f"{tag}_rally_median_s"] = float(np.median(spans)) if spans else None
            summary[f"{tag}_duration_mean_s"] = round(float(np.mean([r[f"{tag}_s"] for r in rows])), 0)
        curves = {}
        for r in rows:
            if not r["arena"].startswith("whole"):
                continue
            for side in (1, 2):
                name = r["factions"][side]
                for tag in ("game", "sim"):
                    curves.setdefault(name, {}).setdefault(tag, []).append(r[f"{tag}_hp_lost_after_contact_{side}"])
        summary["hp_lost_after_contact"] = {
            name: {tag: [round(float(v), 3) for v in np.mean(c, axis=0)] for tag, c in d.items()}
            for name, d in curves.items()}
        for name, d in summary["hp_lost_after_contact"].items():
            print(f"{name}: HP lost {', '.join(str(x) for x in CURVE_S)} s after the first contact: "
                  f"game {d['game']}, sim {d['sim']}")
        print(f"routs / rallies a battle: game {summary['game_routs_per_battle']} / {summary['game_rallies_per_battle']}, "
              f"sim {summary['sim_routs_per_battle']} / {summary['sim_rallies_per_battle']}; rally takes (median): "
              f"game {summary['game_rally_median_s']} s, sim {summary['sim_rally_median_s']} s; battle (mean): "
              f"game {summary['game_duration_mean_s']} s, sim {summary['sim_duration_mean_s']} s")
        summary["sim_not_over_at_recording_end"] = round(float(np.mean([r["sim_cut"] for r in rows])), 2)
        print(f"simulated battles not over when the recording ended (winner = side with more standing health): "
              f"{100 * summary['sim_not_over_at_recording_end']:.0f} %")
        report["battles_summary"] = summary
        for k in ("whole", "arena", "all"):
            if k in summary:
                v = summary[k]
                print(f"same winner, {k}: {v['same_winner']} of {v['decided']} ({100 * v['share']:.0f} %)")
        # The network's gate battles (generated armies), apart: both sides' recorded orders replayed.
        net_rows = battles(params, dev, net=True)
        report["battles_net"] = net_rows
        print("== the network's battles against the game's AI (generated armies), replayed open-loop")
        for r in net_rows:
            print(f"{r['run']:16} {r['arena']:20} winner game/sim {r['game_winner']}/{r['sim_winner']} "
                  f"({r['sim_winner_share']:.2f})  s {r['game_s']:.0f}/{r['sim_s']:.0f}  HP lost 1 "
                  f"{r['game_hp_lost_1']:.2f}/{r['sim_hp_lost_1']:.2f}  2 {r['game_hp_lost_2']:.2f}/{r['sim_hp_lost_2']:.2f}")
        decided = [r for r in net_rows if r["match"] is not None]
        report["battles_net_summary"] = {"decided": len(decided), "same_winner": sum(r["match"] for r in decided)}
        print(f"same winner, the network's battles: {sum(r['match'] for r in decided)} of {len(decided)}")
    if args.only in (None, "speed"):
        devices = [args.device] if args.device else (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
        report["speed"] = []
        for d in devices:
            r = speed(d, args.batch, params)
            report["speed"].append(r)
            print(f"== speed {d}: {r['batch']} battles x {r['slots']} slots, {r['sim_s']} s simulated, "
                  f"{r['wall_s']} s wall -> {r['battles_per_s']} battles/s, {r['steps_per_s']} steps/s "
                  f"({r['done']} finished)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
