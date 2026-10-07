"""The armies of an old arena run again (tools.build nn-arena --army-from-run <run folder>): both sides'
units, keys and deployment read back from that run's manifest.json (build/nn-arena/runs/<id>), not from the
army generator, so a battle of an older generator can be played again after the generator changed.

The manifest keeps each unit's key, slot, x/z, bearing and width, the sides' factions, the gap and the defend
radius; the map, its zones and the deployment size are config/nn/arena.json's (the generated battles' base).
The men of a unit are the pool's for its key (config/nn/pools.json); their sum must equal the manifest's
"army.men" when it has one, else the pool changed and the battle would not be the same.

    arena, army = army_from_run.load(Path("build/nn-arena/runs/20261007-091416"))

Outside tools/nn/armies on purpose: the simulator's version (tools/nn/train/version.py VERSION_FILES)
does not change. Plain python: no numpy, no torch.
"""
import json
from pathlib import Path

from tools.nn import scenario as nn_scenario
from tools.nn.armies import pools as P

SIDES = nn_scenario.SIDES
LORD_SLOT = "lord"


def side_frame(side, x, z, gap_m):
    """World x/z of a unit -> (forward, lateral) in its side's frame (the inverse of scenario.placements)."""
    ax, az, fwd, right, _ = nn_scenario.frame(side, gap_m)
    dx, dz = x - ax, z - az
    return round(dx * fwd[0] + dz * fwd[1], 1) + 0.0, round(dx * right[0] + dz * right[1], 1) + 0.0


def _men_by_key(pools):
    out = {}
    for pool in pools.values():
        for u in [pool.lord, *pool.units]:
            out[u.key] = u.men
    return out


def load(run_dir, swap=False, pools=None, base=None):
    """(arena, army): the arena dict tools/nn/scenario.py takes, rebuilt from the run's manifest, and the
    run_config "army" entry (the old run's, plus "from_run"). swap: the sides exchanged, as --army-swap does
    with the generator's (the network takes the other army); "army.swap" is then the old one flipped."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest.get("config") or {}
    if manifest.get("target") != "nn-arena" or not cfg.get("units") or not cfg.get("factions"):
        raise ValueError(f"{run_dir}: not an nn-arena run with units and factions in its manifest")
    men = _men_by_key(pools or P.load())
    gap_m = cfg["gap_m"]
    sides = {}
    for side in SIDES:
        units = []
        for u in cfg["units"][side]:
            if u["key"] not in men:
                raise ValueError(f"{run_dir}: {side} unit {u['key']} is in no faction pool (config/nn/pools.json)")
            forward, lateral = side_frame(side, u["x"], u["z"], gap_m)
            row = {"slot": u["slot"], "key": u["key"], "men": men[u["key"]], "forward": forward, "lateral": lateral,
                   "width": u["width"]}
            if u["slot"] == LORD_SLOT:
                row["general"] = True
            units.append(row)
        old = cfg.get("army") or {}
        want = (old.get("men") or {}).get(side)
        if want is not None and sum(u["men"] for u in units) != want:
            raise ValueError(f"{run_dir}: {side} has {sum(u['men'] for u in units)} men by the pools, "
                             f"the run had {want}")
        sides[side] = {"faction": cfg["factions"][side], "units": units}
        for k in ("template", "cost", "side_budget"):
            if k in old:
                sides[side][{"template": "army", "side_budget": "budget"}.get(k, k)] = old[k][side]
    base = base or nn_scenario.load_arena()
    arena = {k: v for k, v in base.items() if k not in ("faction", "units", "description", "sides")}
    arena.update(gap_m=gap_m, defend_radius_m=cfg["defend_radius_m"], name=f"{cfg['arena']}_run{run_dir.name}",
                 note=f"armies of run {run_dir.name}", sides=sides)
    army = dict(cfg.get("army") or {}, from_run=run_dir.name)
    if swap:
        arena = dict(arena, name=f"{arena['name']}_swap", sides={"own": sides["enemy"], "enemy": sides["own"]})
        army["swap"] = not army.get("swap", False)
        for k in ("side_budget", "template", "cost", "men"):
            if k in army:
                army[k] = {"own": army[k]["enemy"], "enemy": army[k]["own"]}
    return arena, army
