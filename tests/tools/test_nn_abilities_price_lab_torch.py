"""The bonus lab's simulator pairs (tools/nn/abilities_price/lab_sim.py): the two copies differ only by the bonus, the
lab masks are the simulator's own.

torch is not in the project's .venv: skipped there, run in the training container.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.abilities_price import bonuses as bz  # noqa: E402
from tools.nn.abilities_price import lab  # noqa: E402
from tools.nn.abilities_price import lab_sim  # noqa: E402
from tools.nn.sim import abilities as sab  # noqa: E402

FIELDS = lab_sim.FLAGS + lab_sim.FLOATS


def setup(kinds, n, dose=None, seed=3):
    pools, units, passports, sim, rules = lab.load_config()
    bricks = bz.bricks(bz.rules(rules, sim))
    var = lab.variants(passports, bricks, units)
    plab = {lab.lab_key(v, d): lab.lab_passport(v, d, passports, bricks)
            for vs in var.values() for v in vs for d in lab.DOSES + (0.0,)}
    rng = np.random.default_rng(seed)
    scen = []
    for i, k in enumerate(kinds):
        s = lab.draw(k, n, rng, lab.pool_of(pools), units, var[k], None, seed_base=1000 * i)
        for x in s:
            x["press"] = min(x["press"], 6)                          # short runs
            if dose is not None:
                v = next(v for v in var[k] if v.source == x["variant"])
                x["key"], x["dose"] = lab.lab_key(v, dose), dose
        scen += s
    return scen, lab_sim.params_with(plab), plab, bricks


def test_a_bonus_of_size_zero_leaves_the_copies_identical():
    """Copy on fires the lab ability (its timer runs) but with a bonus of size 0 every trace field, the gold and the
    off copy are the same bit for bit: the copies differ only by the bonus."""
    scen, params, _, _ = setup(lab.KINDS, 4, dose=0.0)
    r = lab_sim.play(scen, params, "cpu", horizon_s=12, keep_off=True)
    assert r["fired"].all()
    for f in FIELDS:
        assert np.array_equal(r[f], r["off_" + f]), f
    assert (r["y"] == 0).all() and (r["y_own"] == 0).all() and (r["y_foe"] == 0).all()
    assert (r["off_on"] == 0).all() and (r["on"][:, 1] > 0).all()


def test_a_whole_battle_pair_with_a_bonus_of_size_zero_stays_identical():
    """The battle family: the battle run once to the press, doubled; with a bonus of size 0 the copies stay the same."""
    _, params, plab, bricks = setup(("speed",), 1)
    pools, units, passports, sim, rules = lab.load_config()
    var = lab.variants(passports, bricks, units)
    batches = lab_sim.draw_battles("melee_defence", 3, np.random.default_rng(7), var["melee_defence"],
                                   np.array([4.0]), per_batch=3)
    t, scen = batches[0]
    for s in scen:
        v = next(v for v in var["melee_defence"] if v.source == s["variant"])
        s["key"] = lab.lab_key(v, 0.0)
    r = lab_sim.play_battle(scen, t, params, "cpu", horizon_s=6, keep_off=True, kind="melee_defence", wait_s=4)
    assert (r["press"] <= 4).all()
    assert r["fired"].all() and r["x"].shape[2] == lab_sim.BATTLE_PER_SIDE
    for f in FIELDS:
        assert np.array_equal(r[f], r["off_" + f]), f
    assert (r["y"] == 0).all()


def test_with_the_bonus_the_copies_part_only_after_the_press():
    scen, params, _, _ = setup(("melee_damage", "speed", "morale"), 6, dose=2.0)
    r = lab_sim.play(scen, params, "cpu", horizon_s=20, keep_off=True)
    assert (r["y0"] == 0).all()
    for f in FIELDS:
        assert np.array_equal(r[f][:, 0], r["off_" + f][:, 0]), f       # sample 0: the press, before it acts
    moved = np.abs(r["x"].astype(float) - r["off_x"].astype(float)).max((1, 2)) > 0
    assert moved.any() or (r["y"] != 0).any()


def test_the_lab_masks_are_the_simulators():
    """Who gets the bonus by layer 2 (the lab passport's Ability on the trace) against the simulator's own record
    (abilities.RECEIVED) of copy on, every second after the press."""
    scen, params, plab, bricks = setup(("melee_defence", "morale", "speed"), 6)
    sab.RECEIVED = []
    try:
        r = lab_sim.play(scen, params, "cpu", horizon_s=20)
        got = list(sab.RECEIVED)
    finally:
        sab.RECEIVED = None
    k = max(1, int(round(1.0 / params.dt)))
    agree = total = 0
    for i, s in enumerate(scen):
        a = lab.ability_of(plab[s["key"]], bricks)
        tr = lab.trace_of(r, i)
        m = a.masks(tr)
        mine = m["self"] | m["friends"]
        for sec in range(1, 21):
            step = (s["press"] + sec - 1) * k + k - 1
            sim = got[step].get(s["key"])
            sim = sim[i, :lab.TRACE_H].numpy() if sim is not None else np.zeros(lab.TRACE_H, dtype=bool)
            agree += int((sim == mine[sec]).sum())
            total += lab.TRACE_H
    assert agree / total >= 0.97, agree / total
