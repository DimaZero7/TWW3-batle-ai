"""Armies -> the first state of a batch (tools/nn/sim/state.py).

An army description per battle (plain python):

    {"attacker": 1 or 2,
     "sides": {1: {"faction": "...", "units": [{"key", "x", "z", "b", "width", "general"}...]},
               2: {...}}}

`b` is the bearing in degrees, `width` the frontage in m (default: men / rank_depth files),
`men` optional (default: the passport's). Side 1 goes to slots [0, H), side 2 to [H, 2H); a
side's general first. from_arena() and from_recording() make descriptions from
config/nn/arenas.json and recorded runs.
"""
import json
import math
from pathlib import Path

import torch

from tools.nn import scenario as arena_scenario
from tools.nn.sim import replay
from tools.nn.sim import state as S
from tools.nn.sim.params import load

MAX_PER_SIDE = 20        # 1 lord + 19 units


def build(armies, params=None, device="cpu", per_side=None):
    """State [B, 2H] from a list of army descriptions."""
    params = params or load()
    B = len(armies)
    H = per_side or max(max(len(a["sides"][s]["units"]) for s in (1, 2)) for a in armies)
    st = S.empty(B, 2 * H, device=device)
    rows = {k: [[0.0] * (2 * H) for _ in range(B)] for k in S.STATIC}
    obs = {k: [[0.0] * (2 * H) for _ in range(B)] for k in ("x", "z", "b", "men")}
    attacker = []
    for bi, army in enumerate(armies):
        attacker.append(int(army.get("attacker", 1)))
        for side in (1, 2):
            spec = army["sides"][side]
            units = sorted(spec["units"], key=lambda u: not u.get("general"))
            assert len(units) <= H, f"battle {bi} side {side}: {len(units)} units > {H}"
            for k, unit in enumerate(units):
                slot = (side - 1) * H + k
                row = params.static(unit["key"], spec.get("faction"))
                if unit.get("width"):
                    row["width"] = float(unit["width"]) if row["men0"] > 1 else row["width"]
                row["side"] = side
                row["lord"] = bool(unit.get("general"))
                for name in S.STATIC:
                    rows[name][bi][slot] = row[name]
                obs["x"][bi][slot] = unit["x"]
                obs["z"][bi][slot] = unit["z"]
                obs["b"][bi][slot] = unit.get("b", 90.0 if side == 1 else 270.0)
                obs["men"][bi][slot] = unit.get("men", row["men0"])
                st.keys[bi][slot] = unit["key"]
    u = st.u
    for name, (code, _) in S.STATIC.items():
        u[name] = torch.tensor(rows[name], dtype=S._dtype(code), device=device)
    for name in obs:
        u[name] = torch.tensor(obs[name], dtype=torch.float32, device=device)
    present = u["side"] > 0
    u["men"] = torch.where(present, u["men"], torch.zeros_like(u["men"]))
    u["hp_abs"] = u["hp0"] * u["men"] / u["men0"].clamp(min=1)
    u["hp"] = torch.where(present, torch.ones_like(u["hp"]), torch.zeros_like(u["hp"]))
    u["morale"] = u["leadership"].clone()
    u["mp"] = torch.where(present, torch.ones_like(u["mp"]), torch.zeros_like(u["mp"]))
    u["ms"] = torch.where(present, torch.full_like(u["ms"], 2.0), torch.zeros_like(u["ms"]))
    u["a"] = u["ammo0"].clone()
    u["ox"], u["oz"] = u["x"].clone(), u["z"].clone()
    u["rally_s"].fill_(1e6)
    u["under_fire_s"].fill_(1e6)
    u["vis"] = present.clone()
    st.attacker = torch.tensor(attacker, dtype=torch.int64, device=device)
    st.bounds = params.map_half
    return st


def arena_of(arena, arena_path=None):
    """An arena by name (config/nn/arenas.json, or 'arena'), or given as a dict: an entry in
    arenas.json's format ({"gap_m", "sides": {"own", "enemy"}}, ...) laid over the base arena
    (config/nn/arena.json: map, zones), or a whole arena as tools.nn.scenario.load_arena gives."""
    if arena is None or isinstance(arena, str):
        return arena_scenario.load_arena(arena, arenas_path=arena_path)
    if "sides" not in arena and "units" not in arena:
        raise ValueError("an arena dict sets its armies: 'sides' (own, enemy) or 'units'")
    base = json.loads(arena_scenario.ARENA.read_text(encoding="utf-8"))
    merged = {k: v for k, v in base.items() if k not in ("faction", "units", "description")} if "sides" in arena         else dict(base)
    merged.update(arena)
    merged.setdefault("name", "custom")
    return merged


def from_arena(name, own_ai="attack", arena_path=None):
    """An army description from an arena: a name in config/nn/arenas.json (or 'arena'), or an
    arena dict (arena_of). Our side is side 1. own_ai attack -> side 1 attacks; defend or hold ->
    side 2 attacks."""
    arena = arena_of(name, arena_path)
    places = arena_scenario.placements(arena)
    sides = arena_scenario.armies(arena)
    params = load()
    spacing = params.sim["formation"]["spacing_m"]
    out = {"attacker": 1 if own_ai == "attack" else 2, "sides": {}}
    for side, tag in ((1, "own"), (2, "enemy")):
        units = []
        for p in places[tag]:
            # A placement is the formation's front centre (as the game's order point): step back
            # half a depth to the centre.
            back = float(replay.half_depth(p["men"], p["width"], spacing)) if p["men"] > 1 else 0.0
            br = math.radians(p["bearing_deg"])
            units.append({"key": p["key"], "x": p["x"] - back * math.sin(br), "z": p["z"] - back * math.cos(br),
                          "b": float(p["bearing_deg"]), "width": p["width"], "general": p["general"],
                          "men": p["men"], "name": p["script_name"]})
        out["sides"][side] = {"faction": sides[tag]["faction"], "units": units}
    return out


def attacker_of(battle, cfg=None):
    """The attacking side of a recorded battle: the manifest's own_role or enemy_role (a run the
    network played has own_ai "net"), else own_ai (attack -> side 1)."""
    cfg = cfg or {}
    own = cfg.get("own_role") or {"attack": "defend", "defend": "attack"}.get(cfg.get("enemy_role") or battle.enemy_role)
    if own in ("attack", "defend"):
        return 1 if own == "attack" else 2
    return 1 if battle.own_ai == "attack" else 2


def from_recording(battle, run_dir=None):
    """An army description from a recorded battle (tools/nn/gamedata.Battle): the units' places
    and bearings at the first record, widths and factions from the run's manifest. Also returns
    the slot of each recorded unit, in the recording's order."""
    widths, factions, generals, cfg = {}, {}, set(), {}
    if run_dir is not None:
        cfg = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))["config"]
        factions = cfg.get("factions", {})
        for tag in ("own", "enemy"):
            for unit in cfg.get("units", {}).get(tag, []):
                widths[unit["script_name"]] = unit.get("width")
                if unit.get("slot") == "lord" or unit["script_name"].endswith("_lord"):
                    generals.add(unit["script_name"])
    out = {"attacker": attacker_of(battle, cfg), "sides": {1: {"faction": factions.get("own"), "units": []},
                                                                     2: {"faction": factions.get("enemy"), "units": []}}}
    for i, name in enumerate(battle.names):
        side = int(battle.side[i])
        out["sides"][side]["units"].append({
            "key": battle.keys[i], "x": float(battle.f["x"][0, i]), "z": float(battle.f["z"][0, i]),
            "b": float(battle.f["b"][0, i]), "width": widths.get(name), "general": name in generals,
            "men": float(battle.f["men"][0, i]), "name": name})
    return out


def slots(army, per_side):
    """Slot of each unit in an army description, in its listed order: {name: slot}."""
    out = {}
    for side in (1, 2):
        units = sorted(army["sides"][side]["units"], key=lambda u: not u.get("general"))
        for k, unit in enumerate(units):
            out[unit.get("name", f"{side}_{k}")] = (side - 1) * per_side + k
    return out
