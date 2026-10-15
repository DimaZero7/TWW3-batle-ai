"""Target's HP lost a second in windows after a charge's impact, game vs simulator variants, over the busycharge,
the melee probe's charge plan and the recontact probe's first contacts.
DOCK_BUILD=<main>/build bash tools/nn/dock.sh firstblow_tmp.r2 [--copies N] [--groups busy,melee,recon] name:sec.key=v;... """
import sys
from pathlib import Path
import numpy as np
from tools.nn import charge_probe as cp, busycharge_probe as bp
from tools.nn.sim.stepreplay import _params

GROUPS = {"busy": ["build/charge-probe/runs/20261014-223056", "build/charge-probe/runs/20261014-223214",
                   "build/charge-probe/runs/20261014-223324"],
          "melee": sorted(str(d) for d in Path("build/meleetests/runs/charge").iterdir()),
          "recon": ["build/charge-probe/runs/20261011-143904", "build/charge-probe/runs/20261011-144020",
                    "build/charge-probe/runs/20261011-144146"]}
WIN = [(0, 1), (1, 2), (0, 2), (2, 5), (0, 5), (5, 10), (10, 20)]
argv = sys.argv[1:]
copies = 4
groups = list(GROUPS)
if "--copies" in argv:
    i = argv.index("--copies"); copies = int(argv[i + 1]); del argv[i:i + 2]
if "--groups" in argv:
    i = argv.index("--groups"); groups = argv[i + 1].split(","); del argv[i:i + 2]
VARIANTS = {k: (v.split(";") if v else []) for k, v in (a.split(":", 1) for a in argv)} or {"base": []}


def impact_generic(ln):
    c = ln["contacts"].get(1)
    if c is None:
        return None
    s = ln["samples"]
    t = np.array([x[0] for x in s]); hp = cp.series(s, "tg", "hp")
    for i in range(1, len(t)):
        if t[i - 1] < c - 2.0 or t[i - 1] > c + 1.5:
            continue
        drop = hp[i - 1] - hp[i]; dt = t[i] - t[i - 1]
        before = (cp.at(t, hp, t[i - 1] - 3.0) - hp[i - 1]) / 3.0 if t[i - 1] - 3.0 >= t[0] else 0.0
        if np.isfinite(drop) and drop > 50 * dt / 0.5 and drop > 2.5 * max(before, 0.0) * dt:
            return float(t[i - 1])
    return c


def key(g, ln):
    sp = ln["spec"]
    if g == "busy":
        return (g, cp.SHORT[sp["a_key"]], str(sp.get("kind")))
    return (g, cp.SHORT[sp["a_key"]] + ">" + cp.SHORT[sp["t_key"]], str(sp.get("mode", "")) + "/" + str(sp.get("target_mode", "")))


def rates(g, ls):
    out = {}
    for ln in ls:
        c = bp.impact(ln) if g == "busy" else impact_generic(ln)
        if c is None:
            continue
        s = ln["samples"]; t = np.array([x[0] for x in s]); hp = cp.series(s, "tg", "hp")
        out.setdefault(key(g, ln), []).append([(cp.at(t, hp, c + a) - cp.at(t, hp, c + b)) / (b - a) for a, b in WIN])
    return {k: (np.nanmean(np.array(v), axis=0), len(v)) for k, v in out.items()}


print("windows s:", WIN, "copies", copies, flush=True)
for g in groups:
    lanes = [ln for d in GROUPS[g] for ln in cp.load_run(d)]
    if g == "busy":
        lanes = [ln for ln in lanes if ln.get("plan") == bp.PLAN]
    lanes = [ln for ln in lanes if ln["contacts"].get(1) is not None and cp.SHORT.get(ln["spec"]["a_key"]) != "general"]
    res = {"game": rates(g, lanes)}
    for name, sets in VARIANTS.items():
        res[name] = rates(g, cp.sim_lanes(lanes, params=_params(sets), device="cpu", copies=copies))
    for k in sorted(res["game"]):
        for name, r in res.items():
            if k in r:
                v, n = r[k]
                g0 = res["game"][k][0]
                rat = "" if name == "game" else " | x " + " ".join(f"{a / b:5.2f}" if b > 0 else "  -  " for a, b in zip(v, g0))
                print(f"{g:5} {k[1]:16} {k[2]:22} {name:9} {n:3d} " + " ".join(f"{x:6.0f}" for x in v) + rat, flush=True)
