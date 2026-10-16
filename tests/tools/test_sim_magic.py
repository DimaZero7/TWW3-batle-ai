"""tools.nn.sim.magic: the side's winds, a cast at a unit and its bombardment (docs/en/training/simulator.md, "Magic").
Needs torch: skipped in the project's .venv; run in the container."""
import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from tools.nn.charge_probe import PROBE_UNITS_JSON  # noqa: E402
from tools.nn.sim import battle, magic, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

GOLD, WARLOCK = "wh3_dlc25_emp_cha_wizard_metal", "wh2_main_skv_cha_warlock_engineer"
SD, WL = "wh_main_spell_metal_searing_doom", "wh2_main_spell_ruin_warp_lightning"
SPEAR, SWORD, GREAT = "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_swordsmen", "wh_main_emp_inf_greatswords"
CLANRAT, SLAVE, STORM = "wh2_main_skv_inf_clanrats_1", "wh2_main_skv_inf_skavenslaves_0", "wh2_main_skv_inf_stormvermin_1"


def params(**magic_cal):
    p = load()
    probe = json.loads(PROBE_UNITS_JSON.read_text(encoding="utf-8"))["units"]
    p = dataclasses.replace(p, units={**probe, **p.units})
    return p.with_cal("magic", **magic_cal) if magic_cal else p


P = params()


def duel(caster, target, gap=120.0, width=30.0, extra=None, p=P):
    """A caster at (0, -gap) facing north and a target unit at (0, 0) facing south, each on its own side (and extra
    units of the target's side: [(key, x, z)])."""
    emp = caster == GOLD
    c = {"key": caster, "x": 0.0, "z": -gap, "b": 0.0, "general": False}
    t = [{"key": target, "x": 0.0, "z": 0.0, "b": 180.0, "width": width, "general": False}]
    t += [{"key": k, "x": x, "z": z, "b": 180.0, "width": width, "general": False} for k, x, z in extra or ()]
    sides = {1: {"faction": "wh_main_emp_empire", "ai": False, "units": [c] if emp else t},
             2: {"faction": "wh2_main_skv_skaven", "ai": False, "units": t if emp else [c]}}
    st = scenario.build([{"attacker": 1, "sides": sides}], p, per_side=1 + len(extra or ()))
    a = 0 if emp else st.N // 2
    tg = st.N // 2 if emp else 0
    st.u["leadership"] = st.u["leadership"] + 1e4          # fearless: nobody routs
    st.u["morale"] = st.u["morale"] + 1e4
    return st, a, tg


def run(st, a, tg, casts=(0.0,), until=20.0, p=P, keep_caster=True):
    """Ask the caster to cast at the target at the given seconds; every unit holds (the caster keeps his order)."""
    asked = list(casts)

    def policy(st):
        o = O.hold(st.B, st.N, st.device)
        if keep_caster:
            o.kind[:, a] = O.KEEP
        slot = torch.full_like(o.kind, -1)
        target = torch.full_like(o.kind, -1)
        if asked and float(st.t[0]) >= asked[0] - 1e-6:
            asked.pop(0)
            slot[:, a], target[:, a] = 0, tg
        magic.request(st.u, slot, target)
        return o
    rec = []
    battle.run(st, policy, p, until_s=until, record=lambda s: rec.append(
        (float(s.t[0]), float(s.u["hp_abs"][0, tg]), float(s.u["men"][0, tg]), float(s.u["wm_cur"][0, a]),
         float(s.u["x"][0, a]), float(s.u["z"][0, a]))))
    return rec


def test_the_spell_passports_are_the_database():
    sd, wl = P.spells[SD], P.spells[WL]
    assert (sd["mana"], sd["wind_up_s"], sd["range_m"], sd["active_s"], sd["recharge_s"]) == (6, 1, 200, 8, 30)
    b = sd["bombardment"]
    assert (b["projectiles"], b["start_s"], b["window_s"], b["spread_m"]) == (40, 2, 3, 10)
    assert (b["projectile"]["damage"], b["projectile"]["ap"], b["explosion"]["ap"], b["explosion"]["radius_m"]) == \
        (24, 12, 30, 3)
    assert wl["bombardment"]["projectile"]["ap"] == 224 and wl["bombardment"]["projectile"]["penetration"] == 20
    # the casters own their rank-1 spell
    assert P.static(GOLD)["sp0"] == magic.keys(P).index(SD) and P.static(WARLOCK)["sp0"] == magic.keys(P).index(WL)
    assert P.static(SPEAR)["sp0"] == -1


def test_winds_start_as_the_database_and_flow_one_in_ten_seconds():
    st, a, tg = duel(GOLD, CLANRAT)
    assert float(st.u["wm_cur"][0, a]) == 12 and float(st.u["wm_res"][0, a]) == 68
    rec = run(st, a, tg, casts=(), until=20.0)
    assert rec[-1][3] == pytest.approx(14.0, abs=0.06)          # +0.1 a second


def test_a_cast_costs_its_mana_and_the_cooldown_runs_from_the_effects_end():
    st, a, tg = duel(GOLD, CLANRAT)
    rec = run(st, a, tg, casts=[0.0] + [float(t) for t in range(1, 50)], until=50.0)
    rec = [(0.0, None, None, 12.0)] + rec
    drops = [r[0] for prev, r in zip(rec, rec[1:]) if prev[3] - r[3] > 2.5]
    assert len(drops) == 2
    # wind-up 1 + active 8 + recharge 30 from the first cast (the probe: 40 s with the sampling's second)
    assert 39.0 <= drops[1] - drops[0] <= 40.0
    assert rec[2][3] == pytest.approx(12 - 6 + 0.1, abs=0.06)


def test_the_first_hit_comes_after_the_wind_up_the_start_and_the_fall():
    for caster, target, first in ((GOLD, CLANRAT, 5.96), (WARLOCK, SPEAR, 5.17)):
        st, a, tg = duel(caster, target)
        rec = run(st, a, tg, until=12.0)
        hit = next(t for t, hp, *_ in rec if hp < rec[0][1] - 0.5)
        assert first <= hit <= first + 0.5 + 1e-6


def test_out_of_range_the_caster_runs_in_then_casts():
    st, a, tg = duel(WARLOCK, SPEAR, gap=260.0)
    rec = run(st, a, tg, until=40.0)
    start = next(r for prev, r in zip(rec, rec[1:]) if prev[3] - r[3] > 2.5)
    assert 10.0 < start[0] < 25.0
    assert abs(start[5]) <= 200.0 + 5.0 + 1.0                  # within range of the target's centre (0, 0)
    assert rec[-1][1] < rec[0][1]                              # and the bolts struck


def test_no_cast_without_winds_or_on_a_friend():
    st, a, tg = duel(GOLD, CLANRAT)
    st.u["wm_cur"].fill_(3.0)
    st.u["wm_res"].fill_(0.0)
    rec = run(st, a, tg, until=12.0)
    assert rec[-1][1] == rec[0][1] and rec[-1][3] == pytest.approx(3.0)
    st, a, tg = duel(GOLD, CLANRAT)
    rec = run(st, a, a, until=12.0)                           # at himself
    assert rec[-1][3] > 12.0 - 1e-6


def test_armour_cuts_searing_doom_not_warp_lightning():
    hp = {}
    for caster, target in ((GOLD, SLAVE), (GOLD, STORM), (WARLOCK, SWORD), (WARLOCK, GREAT)):
        st, a, tg = duel(caster, target)
        rec = run(st, a, tg, until=15.0)
        hp[target] = rec[0][1] - rec[-1][1]
    assert hp[STORM] < 0.9 * hp[SLAVE]                        # armour 90 against 0: the base part cut
    assert hp[GREAT] > 0.8 * hp[SWORD]                        # all armour-piercing


def test_the_damage_is_the_probes_order_of_magnitude():
    # the probe (build/magic/probe1.md): Searing Doom on clanrats 2628 HP 4 dead, Warp Lightning on spearmen 2480 / 24
    for caster, target, hp_game, men_game in ((GOLD, CLANRAT, 2628, 4), (WARLOCK, SPEAR, 2480, 24)):
        st, a, tg = duel(caster, target)
        rec = run(st, a, tg, until=15.0)
        assert 0.6 * hp_game < rec[0][1] - rec[-1][1] < 1.4 * hp_game
        assert rec[0][2] - rec[-1][2] < 5 * men_game + 5


def test_landings_peak_at_the_centre():
    e, f_r, _, _ = magic.profile(10.0, 2.0, 1.0, "radius", "cpu")
    _, f_a, _, _ = magic.profile(10.0, 2.0, 1.0, "area", "cpu")
    assert float(f_r[0]) > 3 * float(f_r[4])                   # density 1 / r
    assert float(f_a[0]) == pytest.approx(float(f_a[3]), rel=0.1)   # uniform
    # the landings' density integrates to one projectile over the disc (radius 10)
    area = torch.pi * (e[1:] ** 2 - e[:-1] ** 2)
    assert float((f_r * area).sum()) == pytest.approx(1.0, rel=0.05)


def test_the_explosion_strikes_friends_near_the_target():
    st, a, tg = duel(WARLOCK, SPEAR, extra=[(SWORD, 0.0, 9.0)])      # swordsmen just behind the spearmen
    friend = tg + 1
    run(st, a, tg, until=12.0)
    assert float(st.u["hp_abs"][0, friend]) < float(st.u["hp0"][0, friend])
    st, a, tg = duel(WARLOCK, SPEAR, extra=[(SWORD, 120.0, 0.0)])    # far off: untouched
    run(st, a, tg, until=12.0)
    assert float(st.u["hp_abs"][0, tg + 1]) == float(st.u["hp0"][0, tg + 1])


def test_the_strike_falls_where_a_walking_target_was_at_the_launch():
    st, a, tg = duel(GOLD, CLANRAT)

    def walk(st):
        o = O.hold(st.B, st.N, st.device)
        o.kind[:, a] = O.KEEP
        o.kind[:, tg], o.x[:, tg], o.z[:, tg] = O.MOVE, 0.0, -60.0
        slot = torch.full_like(o.kind, -1)
        target = torch.full_like(o.kind, -1)
        if float(st.t[0]) < 1e-6:
            slot[:, a], target[:, a] = 0, tg
        magic.request(st.u, slot, target)
        return o
    z_at = {}
    battle.run(st, walk, P, until_s=4.0, record=lambda s: z_at.__setitem__(round(float(s.t[0]), 1),
                                                                         float(s.u["z"][0, tg])))
    assert bool(st.u["bm0_fix"][0, a])
    # wind-up 1 s + start 2 s: where the target was ~3 s after the order, not at the order nor later
    assert z_at[3.5] <= float(st.u["bm0_z"][0, a]) <= z_at[2.0]
    assert z_at[0.5] - float(st.u["bm0_z"][0, a]) > 1.0


def test_a_scripted_sides_caster_leaves_and_the_networks_stays():
    # battle 0: the Empire's caster on side 1 (a network), battle 1: the same with side 1 scripted; side 2 always a
    # script with its own caster beside clanrats
    def army():
        return {"attacker": 1, "sides": {
            1: {"faction": "wh_main_emp_empire", "units": [{"key": GOLD, "x": 0.0, "z": -120.0, "b": 0.0},
                                                          {"key": SPEAR, "x": 40.0, "z": -100.0, "b": 0.0}]},
            2: {"faction": "wh2_main_skv_skaven", "units": [{"key": CLANRAT, "x": 0.0, "z": 0.0, "b": 180.0},
                                                           {"key": WARLOCK, "x": 0.0, "z": 120.0, "b": 180.0}]}}}
    st = scenario.build([army(), army()], P, per_side=2)
    by_script = torch.tensor([[False, True], [True, True]])
    magic.leave_scripted(st.u, by_script)
    u = st.u
    gold, spear, clanrat, warlock = 0, 1, 2, 3
    assert int(u["side"][0, gold]) == 1 and float(u["men"][0, gold]) == 1 and int(u["sp0"][0, gold]) >= 0
    for b, slot in ((0, warlock), (1, warlock), (1, gold)):
        assert int(u["side"][b, slot]) == 0 and float(u["men"][b, slot]) == 0 and int(u["sp0"][b, slot]) == -1
    assert int(u["side"][1, spear]) == 1 and float(u["men"][1, spear]) == 120        # the other units stay
    assert int(u["side"][0, clanrat]) == 2
    before = {k: v.clone() for k, v in u.items() if torch.is_tensor(v)}
    magic.leave_scripted(st.u, by_script)                                              # again: nothing changes
    assert all(torch.equal(before[k], u[k]) for k in before)
    # the network's caster still casts on battle 0 (the bombardment strikes the clanrats)
    st0 = scenario.build([army()], P, per_side=2)
    magic.leave_scripted(st0.u, torch.tensor([[False, True]]))
    st0.u["leadership"] = st0.u["leadership"] + 1e4
    st0.u["morale"] = st0.u["morale"] + 1e4
    rec = run(st0, gold, clanrat, until=12.0)
    assert rec[-1][1] < rec[0][1] - 500


def test_magic_off_changes_nothing():
    st, a, tg = duel(GOLD, CLANRAT, p=params(on=0))
    rec = run(st, a, tg, until=12.0, p=params(on=0))
    assert rec[-1][1] == rec[0][1]
