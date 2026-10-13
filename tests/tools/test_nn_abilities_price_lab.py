"""The bonus lab's numpy side (tools/nn/abilities_price/lab.py): its abilities are layer 2's parts of the real ones, the
scenario draw, the rate and its interval, the curves, the morale curve's plug into layer 3."""
import math

import numpy as np
import pytest

from tools.nn.abilities_price import ability as abl
from tools.nn.abilities_price import bonuses as bz
from tools.nn.abilities_price import lab
from tools.nn.abilities_price import validate as va


@pytest.fixture(scope="module")
def cfg():
    pools, units, passports, sim, rules = lab.load_config()
    bricks = bz.bricks(bz.rules(rules, sim))
    return pools, units, passports, bricks, lab.variants(passports, bricks, units)


def test_every_kind_has_variants_carried_by_a_lord(cfg):
    _, units, _, _, var = cfg
    assert set(lab.KINDS) <= set(var)
    for kind, vs in var.items():
        for v in vs:
            assert v.lords and all(units[k]["caste"] == "lord" for k in v.lords), v


def test_a_lab_ability_at_dose_1_is_exactly_its_real_part(cfg):
    """The lab ability prices through layer 2 as the real ability's part of that kind: the same kind, value, groups,
    duration, radius and targets - so its counter is layer 1's counter of that part on any trajectory."""
    _, _, passports, bricks, var = cfg
    real = abl.abilities(passports, bricks)
    rng = np.random.default_rng(0)
    for kind, vs in var.items():
        for v in vs:
            a = lab.ability_of(lab.lab_passport(v, 1.0, passports, bricks), bricks)
            r = real[v.source]
            part = next(p for p in r.parts if p.kind == kind and p.groups == v.groups)
            assert a.parts == (part,) and not a.unpriced
            assert (a.active_s, a.range_m, a.update_targets) == (r.active_s, r.range_m, r.update_targets)
            tr, _ = va.random_trace(rng, active=int(a.active_s))
            m, mr = a.masks(tr), r.masks(tr)
            assert all(np.array_equal(m[g], mr[g]) for g in m)
            mask = np.zeros_like(tr["alive"])
            for g in part.groups:
                mask |= m[g]
            assert a.counters(tr, bricks)[kind] == pytest.approx(bricks[kind].count(tr, mask, part.value))


def test_a_doubled_dose_doubles_a_proportional_counter(cfg):
    _, _, passports, bricks, var = cfg
    rng = np.random.default_rng(1)
    for kind in ("melee_damage", "charge", "melee_defence", "speed", "charge_speed"):
        v = var[kind][0]
        tr, _ = va.random_trace(rng, active=30)
        c1 = lab.ability_of(lab.lab_passport(v, 1.0, passports, bricks), bricks).counters(tr, bricks)[kind]
        c2 = lab.ability_of(lab.lab_passport(v, 2.0, passports, bricks), bricks).counters(tr, bricks)[kind]
        assert c2 == pytest.approx(2 * c1)


def test_a_dose_0_lab_ability_is_neutral(cfg):
    _, _, passports, bricks, var = cfg
    for vs in var.values():
        for v in vs:
            p = lab.lab_passport(v, 0.0, passports, bricks)
            assert p["vigour_per_s"] == 0
            assert all(e["value"] == (1.0 if e["how"] == "mult" else 0.0) for e in p["effects"])


@pytest.mark.parametrize("kind", lab.KINDS)
def test_the_draw_fits_the_state_and_places_the_run_up(cfg, kind):
    pools, units, _, _, var = cfg
    rng = np.random.default_rng(2)
    scen = lab.draw(kind, 200, rng, lab.pool_of(pools), units, var[kind], np.array([0.0, 30000.0]))
    for s in scen:
        for side in (1, 2):
            us = s["army"]["sides"][side]["units"]
            assert len(us) <= lab.PER_SIDE
            assert sum(bool(u.get("general")) for u in us) == 1
        names = [u["name"] for sd in (1, 2) for u in s["army"]["sides"][sd]["units"]]
        assert len(names) == len(set(names)) and set(s["mode"]) == set(names)
        assert s["army"]["sides"][1]["units"][0]["name"] == "lord" and s["lord"] in var[kind][0].lords + tuple(
            lord for v in var[kind] for lord in v.lords)
        lab_units = [n for n in names if s["mode"][n] != lab.RESERVE_HOLD]
        assert 1 + 4 + 2 >= sum(1 for n in lab_units if not n.startswith("e")) and len(s["foe_keys"]) >= 1
        if s["family"] == "runup":
            lo, hi = lab.RUNUP[kind]
            assert lo <= s["runup"] <= hi
            lord = s["army"]["sides"][1]["units"][0]
            e0 = next(u for u in s["army"]["sides"][2]["units"] if u["name"] == "e0")
            d = math.hypot(lord["x"] - e0["x"], lord["z"] - e0["z"])
            assert s["runup"] <= d <= s["runup"] + 25.0
            assert s["mode"]["lord"] == lab.TARGET and s["target"]["lord"] == "e0"
        else:
            assert lab.PRESS_LINE[0] <= s["press"] <= lab.PRESS_LINE[1]
        if kind == "vigour":
            assert 0 <= s["fatigue"]["lord"] <= 30000
        else:
            assert s["fatigue"]["lord"] in (0.0, 30000.0)


def test_the_rate_comes_back_with_an_interval_that_covers_it():
    rng = np.random.default_rng(3)
    c = rng.exponential(50, 4000) * (rng.random(4000) < 0.6)
    y = 0.02 * c + rng.normal(0, 1, 4000) * np.sqrt(c / 50 + 0.01)
    r = lab.rate(y, c, boot=300)
    assert r["k"] == pytest.approx(0.02, rel=0.05) and r["lo"] < 0.02 < r["hi"] and r["hi"] - r["lo"] < 0.004
    assert r["n_pos"] == int((c > 0).sum()) and abs(r["y0"]) < 0.05
    assert r["ls"] == pytest.approx(0.02, rel=0.05)


def test_curves_find_a_rate_that_depends_on_the_condition():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 100, 6000)
    c = rng.exponential(10, 6000)
    y = np.where(x < 50, 0.05, 0.01) * c + rng.normal(0, 0.2, 6000)
    rows = lab.curve(y, c, x, (0, 50, 101), boot=200)
    assert len(rows) == 2
    assert rows[0]["k"] == pytest.approx(0.05, rel=0.1) and rows[1]["k"] == pytest.approx(0.01, rel=0.2)
    assert lab.separated(rows)
    flat = lab.curve(0.03 * c + rng.normal(0, 0.2, 6000), c, x, (0, 50, 101), boot=200)
    assert not lab.separated(flat)


def _presses(rng, abilities, names, P=12):
    """collect.py's file shape from random trajectories (one copy)."""
    fields = ("x", "z", "alive", "r", "m", "mv", "sprint", "charge", "morale", "fatigue")
    d = {f: [] for f in fields + ("on", "mmax")}
    for _ in range(P):
        tr, _ = va.random_trace(rng, active=20)
        for f in fields:
            d[f].append(tr[f][None])
        d["on"].append(tr["on"][None])
        d["mmax"].append(tr["mmax"][None])
    out = {k: np.stack(v) for k, v in d.items()}
    out["owner"] = np.zeros(P, dtype=int)
    out["key"] = np.array([names.index(k) for k in list(abilities)[:P]] * P)[:P]
    out["diff"] = np.zeros((P, 1))
    return out


def test_a_flat_morale_curve_prices_as_the_flat_rate(cfg):
    _, _, passports, bricks, _ = cfg
    abilities = abl.abilities(passports, bricks)
    names = sorted(passports)
    mor = [k for k, a in abilities.items() if "morale" in a.kinds()]
    rng = np.random.default_rng(5)
    data = _presses(rng, {k: abilities[k] for k in mor}, names, P=len(mor) * 3)
    data["key"] = np.array([names.index(mor[p % len(mor)]) for p in range(len(data["owner"]))])
    kinds, C, _ = va.counters(data, names, abilities, bricks)
    j = kinds.index("morale")
    flat = lab.morale_curve_counters(data, names, abilities, bricks, np.array([-1e9, 16.0, 1e9]), [0.1, 0.1], 0.1)
    assert flat == pytest.approx(C[:, j])
    steep = lab.morale_curve_counters(data, names, abilities, bricks, np.array([-1e9, 1e9]), [0.2], 0.1)
    assert steep == pytest.approx(2 * C[:, j])
