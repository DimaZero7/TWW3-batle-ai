"""A generated battle with one side's army changed by hand (tools.build nn-arena --army-add/--army-remove):
units of the faction's pool (config/nn/pools.json) added or removed, the side deployed again by the
generator's rule (tools/nn/armies/place.py: melee lines in front, missile lines behind, the lord behind
all), its cost recounted. The side's budget stays the generator's.

    arena = army_edit.edit(generate.battle(seed), "own", add=[("wh2_dlc13_emp_inf_archers_0", 3)],
                           remove=[("wh_main_emp_inf_spearmen_0", 1)])

Outside tools/nn/armies on purpose: a hand-made battle for the game, the simulator's version
(tools/nn/train/version.py VERSION_FILES) does not change. Plain python: no numpy, no torch.
"""
from tools.nn.armies import place as placement
from tools.nn.armies import pools as P

MAX_UNITS = 19           # besides the lord (tools/nn/armies/generate.py; the game: 20 with the lord)


def parse(spec):
    """'KEY' or 'KEY=N' -> (KEY, N), N >= 1."""
    key, _, n = spec.partition("=")
    count = int(n) if n else 1
    if not key or count < 1:
        raise ValueError(f"{spec!r}: KEY or KEY=N with N >= 1")
    return key, count


def _counts(pairs):
    out = {}
    for key, n in pairs:
        out[key] = out.get(key, 0) + n
    return out


def edit(arena, side, add=(), remove=(), pools=None):
    """A copy of `arena` with `side`'s army changed: remove [(key, n)] then add [(key, n)] (the side's
    faction's pool units). The name gets '_edit', the side "edit": {"add": {key: n}, "remove": {key: n}}."""
    pools = pools or P.load()
    army = arena["sides"][side]
    pool = pools[army["faction"]]
    by_key = {u.key: u for u in pool.units}
    lord = [u for u in army["units"] if u.get("general")]
    assert len(lord) == 1 and lord[0]["key"] == pool.lord.key, f"{side}: the lord is not the pool's"
    units = []
    for u in army["units"]:
        if not u.get("general"):
            assert u["key"] in by_key, f"{side}: {u['key']} is not in the {army['faction']} pool"
            units.append(by_key[u["key"]])
    for key, n in remove:
        have = sum(u.key == key for u in units)
        if have < n:
            raise ValueError(f"{side}: cannot remove {n} x {key}, the army has {have}")
        for _ in range(n):
            units.remove(by_key[key])
    for key, n in add:
        if key not in by_key:
            raise ValueError(f"{key} is not in the {army['faction']} pool: {', '.join(sorted(by_key))}")
        units += [by_key[key]] * n
    if len(units) > MAX_UNITS:
        raise ValueError(f"{side}: {len(units)} units besides the lord, the game takes {MAX_UNITS}")
    edited = dict(army, cost=pool.lord.cost + sum(u.cost for u in units),
                  units=placement.place(pool.lord, units, arena["deployment_m"]),
                  edit={"add": _counts(add), "remove": _counts(remove)})
    return dict(arena, name=f"{arena['name']}_edit", sides=dict(arena["sides"], **{side: edited}),
                note=f"{arena.get('note', '')}; {side} edited by hand".lstrip("; "))
