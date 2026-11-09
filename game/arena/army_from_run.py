"""The armies of an old arena run again (bridge.build nn-arena --army-from-run <run folder>): both sides'
units, keys and deployment read back from that run's manifest.json (build/nn-arena/runs/<id>, the kept: data/nn-arena/runs, data/human/runs), not from the
army generator, so a battle of an older generator can be played again after the generator changed.

The manifest keeps each unit's key, slot, x/z, bearing and width, the sides' factions, the gap and the defend
radius; the map, its zones and the deployment size are sim/config/arena.json's (the generated battles' base).
The men of a unit are the pool's for its key (sim/config/pools.json); their sum must equal the manifest's
"army.men" when it has one, else the pool changed and the battle would not be the same.

    arena, army = army_from_run.load(Path("data/human/runs/20261025-192124"))

Outside sim/armies on purpose: a battle of the game only, the simulator's armies do not change. Plain python: no numpy, no torch.
"""
import json
from pathlib import Path

from sim import CONFIG as SIM_CONFIG
from sim import arena as nn_scenario
from sim.armies import pools as P

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


def _men_of_run(run_dir):
    """{script name: men} of the run's first nn_sample (events.jsonl); {} without one."""
    f = Path(run_dir) / "events.jsonl"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if '"nn_sample"' not in line:
                continue
            d = json.loads(line)
            if d.get("event") == "nn_sample":
                return {u["n"]: int(u.get("men") or 0) for u in d.get("units") or []}
    return {}


def _men_of_units(key):
    """The men of a key by sim/config/units.json (the game's database at the unit size of the project's runs); 0 unknown."""
    f = SIM_CONFIG / "units.json"
    try:
        return int(json.loads(f.read_text(encoding="utf-8"))["units"][key]["men"])
    except (OSError, KeyError, ValueError, TypeError):
        return 0


def _slot_of(key, taken):
    """A slot name for a new key: its key's last two words and the first free number ('long_rifles_1')."""
    stem = "_".join(key.split("_")[-2:])
    stem = stem[:-2] if stem.endswith("_0") else stem
    n = 1
    while f"{stem}_{n}" in taken:
        n += 1
    return f"{stem}_{n}"


def load(run_dir, swap=False, pools=None, base=None, replace=None):
    """(arena, army): the arena dict sim/arena.py takes, rebuilt from the run's manifest, and the
    run_config "army" entry (the old run's, plus "from_run"). swap: the sides exchanged, as --army-swap does
    with the generator's (the network takes the other army); "army.swap" is then the old one flipped.
    replace {old key: new key}: every unit of old key becomes one of new key on the same place, width and side - its men
    the pool's (else sim/config/units.json, the game's database), its slot new_key's name ('<last words>_<n>'); the
    run's men total is then not checked (army.replaced lists the change)."""
    replace = dict(replace or {})
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest.get("config") or {}
    if manifest.get("target") not in ("nn-arena", "human") or not cfg.get("units") or not cfg.get("factions"):
        raise ValueError(f"{run_dir}: not an nn-arena / human run with units and factions in its manifest")
    men = _men_by_key(pools or P.load())
    # a key in no pool (a battle the person played: any key the game knows): its men as the run's first state had them
    run_men = _men_of_run(run_dir)
    gap_m = cfg["gap_m"]
    sides = {}
    for side in SIDES:
        units = []
        for u in cfg["units"][side]:
            if u["key"] in replace:
                u = dict(u, key=replace[u["key"]], script_name=None,
                         slot=_slot_of(replace[u["key"]], [x["slot"] for x in units] + [x["slot"] for x in cfg["units"][side]]))
            n = men.get(u["key"]) or run_men.get(u["script_name"]) or _men_of_units(u["key"])
            if not n:
                raise ValueError(f"{run_dir}: {side} unit {u['key']} is in no faction pool (sim/config/pools.json) "
                                 f"and not in the run's first state")
            forward, lateral = side_frame(side, u["x"], u["z"], gap_m)
            row = {"slot": u["slot"], "key": u["key"], "men": n, "forward": forward, "lateral": lateral,
                   "width": u["width"]}
            if u["slot"] == LORD_SLOT:
                row["general"] = True
            units.append(row)
        old = cfg.get("army") or {}
        want = (old.get("men") or {}).get(side)
        if want is not None and not replace and sum(u["men"] for u in units) != want:
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
    if replace:
        army["replaced"] = replace
        for side in SIDES:
            army.setdefault("men", {})
            army["men"] = dict(army["men"], **{side: sum(u["men"] for u in sides[side]["units"])})
    if swap:
        arena = dict(arena, name=f"{arena['name']}_swap", sides={"own": sides["enemy"], "enemy": sides["own"]})
        army["swap"] = not army.get("swap", False)
        for k in ("side_budget", "template", "cost", "men"):
            if k in army:
                army[k] = {"own": army[k]["enemy"], "enemy": army[k]["own"]}
    return arena, army
