"""Magic in the game before the simulator knows it (build/magic/plan.md 6-7): the Gold Wizard (Lore of Metal,
Searing Doom) and the Warlock Engineer (Lore of Ruin, Warp Lightning), each casting his level-1 spell on a held,
fearless target at a set second. The lanes run on the melee probe (tools/nn/charge_probe.py plan 'magic',
src/entries/charge_probe.lua lane mode 'cast'); far apart, the caster gap_m in front of his target (inside the
spells' 200 m), nobody else ordered.

  battle 1  the game's own winds (P0, P1): each side's caster casts at 40 s and tries again every second from 41 s to
            150 s (when the second and third casts go: the cooldown from the cast or from the end, the cost); a second
            caster each side without <unit_capabilities> (the game's default abilities, never casting): what the
            list changes (probe_mage). Targets clanrats and Empire spearmen;
  battle 2  scripted winds (P1): the Empire side +30 current +40 reserve (are the current and the reserve capped?), the
            Skaven side +30 current -60 reserve (does the flow slow with a small reserve?); two casters a side, casting
            at 10 / 20 s and again from 35 / 45 s: Searing Doom on stormvermin (armour) and skavenslaves (none), Warp
            Lightning on swordsmen and greatswords (P2, P3 damage by armour);
  battle 3  the cast's rules and a moving target (P2, P4): Searing Doom on clanrats walking at the caster, Warp
            Lightning ordered on spearmen out of range (260 m: does the caster walk in?), the stormvermin and the
            swordsmen again (a second sample).

Every lane samples both units every 0.5 s (men, health), the soldiers' places every 1 s and each side's winds every
1 s (probe_winds). The casters' abilities are exactly the spell and their innate ones (tools/nn/scenario.py
capabilities).

    python -m tools.nn.charge_probe plan --plan magic             # the battles
    python -m tools.nn.charge_probe run --plan magic --speed 3    # build + launch each, in turn
    python -m tools.nn.magic_probe [runs...] [--out build/magic/probe.json]   # the casts, the winds, the damage
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tools.nn import charge_probe as cp

SEARING_DOOM = "wh_main_spell_metal_searing_doom"
WARP_LIGHTNING = "wh2_main_spell_ruin_warp_lightning"
GOLD, WARLOCK = "goldwiz", "warlock"
SPELL = {GOLD: SEARING_DOOM, WARLOCK: WARP_LIGHTNING}
GAP_M = 120              # caster to the target's front: inside the spells' 200 m (target_intercept_range)
FAR_M = 260              # ... and out of it (battle 3)
WINDS_MS = 1000
MEN_NEAR_M = 400         # the soldiers' places whenever caster and target are this close (always: no contact)
# battle -> the scripted winds at the start: [{side (1 Empire, 2 Skaven), current, reserve}] (amounts added)
WINDS_SET = {2: [{"side": 1, "current": 30, "reserve": 40}, {"side": 2, "current": 30, "reserve": -60}]}
WINDOWS = ((0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 20))   # s after the cast (its winds drop)
DROP = 2.5               # a fall of the side's current winds this big in one second is a cast


def tries(first_s, from_s, to_s, every_s=1):
    """A caster's casts: one at first_s, then a try every every_s from from_s to to_s (the cooldown's end)."""
    out = [first_s] + list(range(from_s, to_s + 1, every_s))
    return sorted(set(out))


def caster(who, target, at, gap_m=GAP_M, max_s=160, kind="", target_mode="hold", abilities=True, **extra):
    """One lane: the caster `who` (GOLD / WARLOCK) and its target; at: the cast seconds after the go."""
    key = SPELL[who]
    if abilities:                         # (none: the game's defaults; the entry config takes no None)
        extra["a_abilities"] = [key]
    return cp.lane(who, target, "cast", target_mode, gap_m=gap_m, fight_s=max_s, max_s=max_s, answer=False,
                   casts=[{"at_s": float(s), "key": key, "tg": "t"} for s in at],
                   kind=kind or f"{who}>{target}", **extra)


def battles():
    """[[lane spec]] of the magic plan (tools/nn/charge_probe.battles hands these over)."""
    b1 = [caster(GOLD, "clanrat", tries(40, 41, 150), kind="b1_sd_clanrat"),
          caster(WARLOCK, "spear", tries(40, 41, 150), kind="b1_wl_spear"),
          caster(GOLD, "slave", [], abilities=False, kind="b1_default_gold"),
          caster(WARLOCK, "swords", [], abilities=False, kind="b1_default_warlock")]
    b2 = [caster(GOLD, "svsh", tries(10, 35, 80), max_s=130, kind="b2_sd_svsh"),
          caster(GOLD, "slave", tries(20, 45, 90), max_s=130, kind="b2_sd_slave"),
          caster(WARLOCK, "swords", tries(10, 35, 80), max_s=130, kind="b2_wl_swords"),
          caster(WARLOCK, "gs", tries(20, 45, 90), max_s=130, kind="b2_wl_gs")]
    b3 = [caster(GOLD, "clanrat", [5], max_s=100, target_mode="push", push_m=60, kind="b3_sd_walking"),
          caster(WARLOCK, "spear", [5], gap_m=FAR_M, max_s=100, kind="b3_wl_far"),
          caster(GOLD, "svsh", [20], max_s=100, kind="b3_sd_svsh"),
          caster(WARLOCK, "swords", [30], max_s=100, kind="b3_wl_swords")]
    return [b1, b2, b3]


def config_extra(index):
    """The entry config's magic fields of battle `index` (1-based)."""
    return {"winds_ms": WINDS_MS, "winds_set": WINDS_SET.get(index, []), "men_ms": 1000, "men_near_m": MEN_NEAR_M}


# ---------------------------------------------------------------- the recordings

def load_magic(run_dir):
    """The magic events of one run: {winds [(t s, {side: row})], api, set, mages, tries [probe_ability rows]}."""
    out = {"winds": [], "api": {}, "set": [], "mages": {}, "tries": []}
    for line in (Path(run_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if '"probe_' not in line:
            continue
        r = json.loads(line)
        ev = r["event"]
        if ev == "probe_winds":
            out["winds"].append((r["t"] / 1000, {1: r.get("s1") or {}, 2: r.get("s2") or {}}, r.get("m")))
        elif ev == "probe_winds_api":
            out["api"][r["side"]] = r
        elif ev == "probe_winds_set":
            out["set"].append(r)
        elif ev == "probe_mage":
            out["mages"][r["lane"]] = r
        elif ev == "probe_ability":
            out["tries"].append(r)
    return out


def side_of(lane):
    """1 (Empire, alliance 0) or 2 (Skaven) for the lane's caster."""
    return 1 if lane["spec"]["attacker"].startswith("own_") else 2


def casts_of(winds, side, drop=DROP):
    """[(t s, drop)] of a side: the 1-s falls of its current winds of at least `drop` (the casts)."""
    pts = [(t, w[side].get("cur")) for t, w, _ in winds if w[side].get("cur") is not None]
    out = []
    for (t0, a), (t1, b) in zip(pts, pts[1:]):
        if a - b >= drop:
            out.append((t1, round(a - b, 3)))
    return out


def winds_summary(winds, side, costs=()):
    """A side's winds: the first, highest and last current (and reserve), and the flow a second from the reserve into
    the current: (last - first + the casts' costs) / the time (the game reads whole numbers: a rise of 1 at a time, so
    the flow is taken over the whole recording, not second by second); the reserve's fall a second likewise."""
    pts = [(t, w[side].get("cur"), w[side].get("res"), w[side].get("rate")) for t, w, _ in winds
           if w[side].get("cur") is not None]
    if len(pts) < 2:
        return None
    t = np.array([p[0] for p in pts])
    cur = np.array([p[1] for p in pts], float)
    span = float(t[-1] - t[0])
    res = [p[2] for p in pts if p[2] is not None]
    rates = [p[3] for p in pts if p[3] is not None]
    return {"first": float(cur[0]), "max": float(cur.max()), "last": float(cur[-1]),
            "flow": round((cur[-1] - cur[0] + sum(costs)) / span, 4) if span > 0 else None,
            "res_first": res[0] if res else None, "res_last": res[-1] if res else None,
            "res_fall": round((res[0] - res[-1]) / span, 4) if res and span > 0 else None,
            "rate_first": rates[0] if rates else None, "rate_last": rates[-1] if rates else None}


def damage_after(samples, t_cast, windows=WINDOWS):
    """The target's health and men lost in each window after t_cast ({"0-2": (hp, men)}), and in total to 20 s."""
    t = np.array([s[0] for s in samples])
    hp = cp.series(samples, "tg", "hp")
    men = cp.series(samples, "tg", "men")
    out = {}
    for lo, hi in windows:
        a, b = cp.at(t, hp, t_cast + lo), cp.at(t, hp, t_cast + hi)
        ma, mb = cp.at(t, men, t_cast + lo), cp.at(t, men, t_cast + hi)
        out[f"{lo}-{hi}"] = (None if np.isnan(a - b) else round(a - b), None if np.isnan(ma - mb) else round(ma - mb))
    hp0, men0 = cp.at(t, hp, t_cast), cp.at(t, men, t_cast)
    return {"hp0": None if np.isnan(hp0) else round(hp0), "men0": None if np.isnan(men0) else round(men0),
            "windows": out}


def measure(run_dir):
    """Rows of one run: per lane its casts (from the side's winds drops, matched to the lane's last try before them)
    with the damage after each; per side the winds."""
    lanes = cp.load_run(run_dir)
    magic = load_magic(run_dir)
    by_side = {}
    for ln in lanes:
        if ln["spec"].get("mode") == "cast":
            by_side.setdefault(side_of(ln), []).append(ln)
    rows, sides = [], {}
    for side in (1, 2):
        drops = casts_of(magic["winds"], side)
        sides[side] = {"api": (magic["api"].get(side) or {}).get("found"),
                       "set": [s for s in magic["set"] if s["side"] == side],
                       "casts": drops, "winds": winds_summary(magic["winds"], side, [d[1] for d in drops])}
        for t_drop, cost in drops:
            # the lane of this side whose try came last before the drop's sample (its given tries only; a try in the same
            # second as the sample is taken as after it: battle 1, the first try at 40 s showed at the 41 s sample)
            best = None
            for ln in by_side.get(side, []):
                ts = [a["t"] / 1000 for a in ln["abilities"] if a.get("status") == "given" and a["t"] / 1000 < t_drop - 0.25]
                if ts and (best is None or max(ts) > best[1]):
                    best = (ln, max(ts))
            if best is None:
                continue
            ln, t_try = best
            rows.append({"run": ln["run"], "battle": ln["battle"], "lane": ln["spec"]["name"],
                         "kind": ln["spec"].get("kind"), "side": side, "try_s": round(t_try, 1), "drop_s": t_drop,
                         "cost": cost, **damage_after(ln["samples"], t_try)})
    mages = {k: {f: v for f, v in m.items() if f in ("u", "active", "passive", "magic", "can", "cco")}
             for k, m in magic["mages"].items()}
    return {"run": Path(run_dir).name, "casts": rows, "sides": sides, "mages": mages}


def report(run_dirs, out=None):
    docs = [measure(r) for r in run_dirs]
    for d in docs:
        print(f"== {d['run']}")
        for side, s in d["sides"].items():
            print(f"  side {side}: readers {s['api']}, set {[(x['current'], x['reserve'], x['status']) for x in s['set']]}")
            print(f"    winds {s['winds']}")
            print(f"    drops {s['casts']}")
        for lane, m in d["mages"].items():
            print(f"  {lane} {m['u']}: active {m.get('active')} passive {m.get('passive')} magic {m.get('magic')} "
                  f"can {m.get('can')} cco {m.get('cco')}")
        for c in d["casts"]:
            w = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in c["windows"].items())
            print(f"  {c['kind']:<20} try {c['try_s']:>6} drop {c['drop_s']:>6} cost {c['cost']:>5}  hp0 {c['hp0']} "
                  f"men0 {c['men0']}  lost hp/men {w}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    return docs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run folders (default: the charge probe's magic runs)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    run_dirs = args.runs or [r for r in cp.runs() if json.loads((r / "manifest.json").read_text(encoding="utf-8"))
                             ["config"].get("plan") == "magic"]
    report(run_dirs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
