"""The Pusbags' rules in the simulator (31.10.2026; build/diag_roster/wh3_dlc29_skv_inf_pusbags/mechanic.md; config/nn/
sim.json direct, vortex): the direct damage on itself in melee (tools/nn/sim/direct.py) and the vortex at a man's death
(tools/nn/sim/vortex.py). Needs torch (the container)."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, direct, effects, scenario, vortex  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import opponents  # noqa: E402

P = load()
PUS, SWORD, SLAVES = "wh3_dlc29_skv_inf_pusbags", "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_skavenslaves_0"
GENERAL, CENSER = "wh_main_emp_cha_general_0", "wh2_main_unit_contact_plague_monk_censer_morale"


def army(side1, side2):
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b} for k, x, z, b in rows]
    return {"attacker": 2, "sides": {1: {"faction": "wh_main_emp_empire", "ai": True, "units": units(side1)},
                                     2: {"faction": "wh2_main_skv_skaven", "ai": True, "units": units(side2)}}}


def build(*armies, params=P):
    return scenario.build(list(armies), params)


def fight(skaven=PUS):
    return army([(SWORD, 0, -20, 0)], [(skaven, 0, 20, 180)])


def censer_slot(params=P):
    return next(j for j, _, k in effects.contact_keys(params) if k == CENSER)


def test_the_statics_point_at_the_two_effects():
    st = build(fight())
    u, H = st.u, st.N // 2
    order = effects.order(P)
    assert order[int(u["dd_fx"][0, H])] == "wh3_dlc29_unit_passive_unshakable_pathogen"
    assert order[int(u["vx_fx"][0, H])] == "wh3_dlc29_unit_passive_pestilent_perfection"
    assert int(u["dd_fx"][0, 0]) == -1 and int(u["vx_fx"][0, 0]) == -1


def test_a_hit_is_2_3_4_hp_and_magic_resistance_cuts_it():
    one = torch.ones(1, 1)
    assert float(direct.hit_mean(5 * one, 0.5 * one, one, 5)) == pytest.approx(3.2, abs=1e-5)
    # 25 % magic: round(2 x 0.75) = 2, round(2.25) = 2, round(3) = 3 -> 0.2 x 2 + 0.4 x 2 + 0.4 x 3
    assert float(direct.hit_mean(5 * one, 0.5 * one, 0.75 * one, 5)) == pytest.approx(2.4, abs=1e-5)


class TestTheDisease:
    def step(self, st, melee, n=1):
        u = st.u
        out = []
        for _ in range(n):
            u["fx_on"] = torch.where(torch.tensor(melee), u["fx"], torch.zeros_like(u["fx"]))
            out.append(direct.strike(u, P, P.dt))
        return out

    def test_a_tick_at_once_then_every_second_off_out_of_melee(self):
        st = build(fight())
        H = st.N // 2
        hp = [float(h[0, H]) for h, _, _ in self.step(st, True, 6)]
        assert hp == pytest.approx([9.6, 0, 9.6, 0, 9.6, 0], abs=1e-4)          # 3 men x 3.2, every 2 steps of 0.5 s
        assert [float(h[0, H]) for h, _, _ in self.step(st, False, 2)] == [0.0, 0.0]
        # back in melee: a tick at once (the phase starts again)
        assert float(self.step(st, True)[0][0][0, H]) == pytest.approx(9.6, abs=1e-4)
        assert float(st.u["dd_sum"][0, H]) == pytest.approx(4 * 9.6, abs=1e-3)
        assert float(st.u["dd_sum"][0, 0]) == 0.0

    def test_it_kills_wounded_men_and_spares_fresh_ones(self):
        st = build(fight())
        H = st.N // 2
        u = st.u
        _, k_fresh, hits = self.step(st, True)[0]
        assert float(hits[0, H]) == 3 and float(k_fresh[0, H]) < 1e-3
        st = build(fight())
        u = st.u
        u["hp_abs"][0, H] = u["men"][0, H] * 4.0                                  # every man at 4 HP
        _, k_low, _ = self.step(st, True)[0]
        assert float(k_low[0, H]) > 1.0


def test_in_a_fight_the_pusbags_lose_9_6_hp_a_second_to_themselves():
    st = build(fight())
    H = st.N // 2
    u = st.u
    melee_s = 0.0
    for _ in range(int(40 / P.dt)):
        battle.step(st, opponents.ai_like(st), P, P.dt)
        melee_s += P.dt * float(u["m"][0, H])
    assert melee_s > 15
    assert float(u["dd_sum"][0, H]) == pytest.approx(9.6 * melee_s, rel=0.08)
    # the skavenslaves (no such effect) none
    st2 = build(fight(SLAVES))
    for _ in range(int(40 / P.dt)):
        battle.step(st2, opponents.ai_like(st2), P, P.dt)
    assert float(st2.u["dd_sum"].abs().sum()) == 0.0


class TestTheVortex:
    def test_a_strike_takes_5_through_the_armours_roll_and_magic_skips_the_physical_resistance(self):
        st = build(fight())
        u = st.u
        one = torch.ones(1, 1, 1)
        u["armour"][0, 0] = 0.0
        assert float(vortex.hit_hp(u, 5 * one, 0 * one, one, 0 * one, 0 * one, 11)[0, 0, 0]) == pytest.approx(5.0)
        u["armour"][0, 0] = 35.0
        # cut U[0.175, 0.35]: 5 (1 - cut) below 3.5 (-> 3) when cut > 0.3: 0.05 / 0.175 of the rolls, else 4
        want = 3 * (0.05 / 0.175) + 4 * (1 - 0.05 / 0.175)
        assert float(vortex.hit_hp(u, 5 * one, 0 * one, one, 0 * one, 0 * one, 11)[0, 0, 0]) == pytest.approx(want, abs=1e-4)
        u["armour"][0, 0] = 0.0
        u["resist_physical"][0, 0] = 0.5
        assert float(vortex.hit_hp(u, 5 * one, 0 * one, one, 0 * one, 0 * one, 11)[0, 0, 0]) == pytest.approx(5.0)
        # a vortex that is not magical meets it: round(2.5) = 3
        assert float(vortex.hit_hp(u, 5 * one, 0 * one, 0 * one, 0 * one, 0 * one, 11)[0, 0, 0]) == pytest.approx(3.0)

    def test_a_front_death_reaches_one_swordsman_and_a_lone_body(self):
        st = build(army([(SWORD, 0, -20, 0), (GENERAL, 40, -20, 0)], [(PUS, 0, 20, 180)]))
        u, H = st.u, st.N // 2
        R = torch.full((1, st.N, 1), 2.0)
        gap = P.sim["vortex"]["contact_gap_m"]
        n = vortex.men_within(u, R, gap)
        assert float(n[0, H, 0]) == pytest.approx(1.03, abs=0.03)                # the probe's places: 1.0
        # the General's body (r 0.7): reached where r_p + r + the gap is below sqrt(R^2 + r^2) - 5 of the 8 gap points
        assert float(n[0, H, 1]) == pytest.approx(0.625)
        kn = vortex.men_within(u, R, gap, inner=0.5 * R + u["radius"][:, None, :])
        assert 0.15 < float(kn[0, H, 0]) < 0.5

    def test_the_dead_strike_two_steps_later_lay_the_censers_morale_and_knock(self):
        st = build(fight())
        u, H, N = st.u, st.N // 2, st.N
        u["fx_on"] = u["fx"].clone()
        q = torch.zeros(1, N, N)
        q[0, H, 0] = 4.0                                                          # 4 dead next to the swordsmen
        vortex.queue(u, P, q)
        contact = torch.zeros(1, N, N, dtype=torch.bool)
        contact[0, H, 0] = contact[0, 0, H] = True
        hp1, _, hits1, _ = vortex.strike(u, P, P.dt, contact)
        assert float(hits1.sum()) == 0.0                                         # 0.5 s: waiting (delay 1 s)
        vortex.queue(u, P, torch.zeros(1, N, N))
        hp, kills, hits, kn = vortex.strike(u, P, P.dt, contact)
        assert float(hits[0, H, 0]) == pytest.approx(4 * 1.03, abs=0.15)
        assert float(hp[0, H, 0]) == pytest.approx(float(hits[0, H, 0]) * 5.0 * (1 - 0.75 * u["armour"][0, 0] / 100), rel=0.1)
        assert float(kn[0, 0]) > 0.5 and float(hits[0, H, H]) == 0.0
        assert float(u["vx_hits"][0, H]) == pytest.approx(float(hits.sum()))
        # out of contact: nothing
        vortex.queue(u, P, torch.zeros(1, N, N))
        vortex.queue(u, P, q)
        vortex.queue(u, P, torch.zeros(1, N, N))
        assert float(vortex.strike(u, P, P.dt, torch.zeros_like(contact))[2].sum()) == 0.0

    def test_knocked_men_strike_less(self):
        st = build(fight())
        u = st.u
        rate = torch.ones(1, st.N, st.N)
        vortex.lie(u, P, P.dt, torch.tensor([[10.0] + [0.0] * (st.N - 1)]), torch.full((1, st.N), 20.0))
        lying = float(u["vk_p"][0, 0])
        assert lying == pytest.approx(0.5 * math.exp(-P.dt / 1.85), abs=1e-4)
        vortex.up_share(u, rate)
        assert float(rate[0, 0, 1]) == pytest.approx(1 - lying) and float(rate[0, 1, 0]) == 1.0

    def test_the_dead_by_blows_lie_next_to_the_killer_the_others_by_the_contact_share(self):
        st = build(fight())
        u, H, N = st.u, st.N // 2, st.N
        own = torch.zeros(1, N, dtype=torch.bool)
        own[0, H] = True
        enemy = (u["side"][:, :, None] != u["side"][:, None, :]) & (u["side"][:, :, None] > 0) & (u["side"][:, None, :] > 0)
        deaths = torch.zeros(1, N, N)
        deaths[0, 0, H], deaths[0, H, H] = 3.0, 1.0                              # 3 by the swordsmen's blows, 1 disease
        share = torch.zeros(1, N, N)
        share[0, 0, H] = 1.0
        F = torch.zeros(1, N, N)
        F[0, H, 0] = 45.0                                                         # a quarter of the 180 in the contact
        drop = torch.zeros(1, N)
        drop[0, H] = 4.0
        near = vortex.dead_near(u, P, own, drop, deaths, share, F, u["men"], enemy)
        assert float(near[0, H, 0]) == pytest.approx(3 + 1 * 45 / 180)
        assert float(near.sum()) == pytest.approx(3.25)


def test_in_a_fight_the_vortices_hold_the_swordsmen_under_the_censers_minus_10():
    st = build(fight())
    u = st.u
    j = censer_slot()
    on = 0
    for i in range(int(50 / P.dt)):
        battle.step(st, opponents.ai_like(st), P, P.dt)
        if i * P.dt > 25:
            on += int(float(u[f"cfx{j}_s"][0, 0]) > 0)
    H = st.N // 2
    assert float(u["vx_hits"][0, H]) > 10 and float(u["vx_kn"][0, H]) > 1
    assert on / ((50 - 25) / P.dt) > 0.8
    # the swordsmen lay nothing on the Pusbags (their own side is never struck)
    assert float(u[f"cfx{j}_s"][0, H]) == 0.0


def test_a_battle_without_pusbags_is_the_same_with_the_rules_off():
    a = army([(SWORD, 0, -20, 0), (GENERAL, 0, -40, 0)], [(SLAVES, 0, 20, 180)])
    off = P.with_cal("vortex", on=0).with_cal("direct", on=0)
    s1, s2 = build(a), build(a, params=off)
    for _ in range(int(40 / P.dt)):
        battle.step(s1, opponents.ai_like(s1), P, P.dt)
        battle.step(s2, opponents.ai_like(s2), off, off.dt)
    for k, v in s1.u.items():
        if torch.is_tensor(v) and k in s2.u:
            assert torch.equal(v, s2.u[k]), k
