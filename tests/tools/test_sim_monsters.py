"""The Skaven monsters in the simulator (29.10.2026; build/diag_skv_mon/mechanic.md, sim_gap.md; config/nn/sim.json
contact.ring_big, contact.impact lane_body / knock_ignore, melee.resist_sum, missile.big_single_in_melee, morale.terror,
regen): the ring of men round a large lone body, its box, the shots at it in melee, the summed resistances (fire, magic),
regeneration and the fire, terror, the body's lane in a charge, the knock resistance, the Rat Ogres' brace. Needs torch
(the container)."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, effects, geometry, impact, melee, missile, regen, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
OGRES, HORROR, MUTANT = ("wh2_main_skv_mon_rat_ogres", "wh2_dlc16_skv_mon_brood_horror_0",
                         "wh2_dlc16_skv_mon_rat_ogre_mutant")
SPEAR, SWORD, ARCHERS = "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_swordsmen", "wh2_dlc13_emp_inf_archers_0"
SUN, GENERAL, WARLORD = "wh_dlc04_emp_cav_knights_blazing_sun_0", "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
CENSER, CLANRAT = "wh2_main_skv_inf_plague_monk_censer_bearer", "wh2_main_skv_inf_clanrats_0"
SP = P.sim["formation"]["spacing_m"]


def army(side1, side2, f1="wh_main_emp_empire", f2="wh2_main_skv_skaven"):
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b, **(extra[0] if extra else {})} for k, x, z, b, *extra in rows]
    return {"attacker": 1, "sides": {1: {"faction": f1, "units": units(side1)}, 2: {"faction": f2, "units": units(side2)}}}


def build(side1, side2, params=P, **kw):
    return scenario.build([army(side1, side2, **kw)], params)


def run(st, seconds, params=P, orders=None):
    for _ in range(int(round(seconds / params.dt))):
        battle.step(st, orders(st) if orders else O.hold(st.B, st.N), params, params.dt)


class TestTheRing:
    def test_the_ring_formula(self):
        # the lord's (R 0.7) ring of spearmen (r 0.65) is the lord swarm probe's 6.5; the Mutant 12.8, the Horror's capsule 17.2
        st = build([(SPEAR, 0, 0, 0), (SPEAR, 50, 0, 0)], [(WARLORD, 0, 5, 180), (MUTANT, 50, 5, 180), (HORROR, 100, 5, 180)])
        H = st.N // 2
        n = melee.ring(st.u, (st.u["men0"] <= 1)[:, None, :])
        assert float(n[0, 0, H]) == pytest.approx(2 * math.pi * 1.35 / 1.3, abs=1e-4) == pytest.approx(6.52, abs=0.01)
        assert float(n[0, 0, H + 1]) == pytest.approx(2 * math.pi * 2.65 / 1.3, abs=1e-4)          # 12.81
        R, q = 3.8, 0.36
        want = (4 * R * (1 - q) + 2 * math.pi * (R * q + 0.65)) / 1.3
        assert float(n[0, 0, H + 2]) == pytest.approx(want, abs=1e-3) and 17.0 < want < 17.5
        big = melee.big_lone(st.u)
        assert big[0, H + 1] and big[0, H + 2] and not big[0, H] and not big[0, 0]

    def test_infantry_round_a_monster_strike_with_the_ring_a_lord_keeps_6_5(self):
        rows = {}
        for target in (WARLORD, MUTANT, HORROR):
            st = build([(SPEAR, 0, -6, 0), (SPEAR, 0, 6, 180)], [(target, 0, 0, 90)])
            H = st.N // 2
            pw = geometry.pairwise(st.u, SP)
            contact = pw["enemy"] & (pw["gap"] <= 1.0)
            assert contact[0, 0, H] and contact[0, 1, H]
            _, _, _, F = melee.strikes(st.u, pw, contact, P, torch.zeros_like(st.u["men"]))
            rows[target] = float(F[0, :2, H].sum())
            n = float(melee.ring(st.u, (st.u["men0"] <= 1)[:, None, :])[0, 0, H])
            assert rows[target] == pytest.approx(min(n, 240.0) if target != WARLORD else 6.5, rel=1e-4), target
        assert rows[WARLORD] < rows[MUTANT] < rows[HORROR]

    def test_off_the_monster_takes_the_lords_cap(self):
        st = build([(SPEAR, 0, -6, 0)], [(HORROR, 0, 0, 90)])
        H = st.N // 2
        pw = geometry.pairwise(st.u, SP)
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        off = P.with_cal("contact", ring_big={"on": 0})
        F = melee.strikes(st.u, pw, contact, off, torch.zeros_like(st.u["men"]))[3]
        assert float(F[0, 0, H]) == pytest.approx(6.5)


def test_a_lone_bodys_box_is_its_length_and_width():
    st = build([(GENERAL, 0, 0, 0)], [(HORROR, 0, 50, 180), (MUTANT, 30, 50, 180)])
    H = st.N // 2
    front, depth = geometry.dims(st.u, SP)
    assert float(front[0, H]) == pytest.approx(2 * 3.8 * 0.36) and float(depth[0, H]) == pytest.approx(7.6)
    assert float(front[0, H + 1]) == float(depth[0, H + 1]) == pytest.approx(4.0)
    assert float(front[0, 0]) == float(depth[0, 0]) == pytest.approx(1.4)


def test_shots_at_a_monster_in_melee_land_on_it_a_lord_in_melee_takes_none():
    hits = {}
    for target in (WARLORD, HORROR):
        st = build([(ARCHERS, 0, -120, 0), (SWORD, 0, 6, 180)], [(target, 0, 0, 0)])
        H = st.N // 2
        pw = geometry.pairwise(st.u, SP)
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        assert contact[0, 1, H]
        aim = torch.full_like(st.u["men"], -1, dtype=torch.long)
        aim[0, 0] = H
        _, hp, _ = missile.volley(st.u, pw, aim, 1.0, P, contact=contact)
        hits[target] = float(hp[0, 0, H])
    assert hits[WARLORD] == 0.0 and hits[HORROR] > 0.0
    off = P.with_cal("missile", big_single_in_melee=0)
    st = build([(ARCHERS, 0, -120, 0), (SWORD, 0, 6, 180)], [(HORROR, 0, 0, 0)])
    H = st.N // 2
    pw = geometry.pairwise(st.u, SP)
    aim = torch.full_like(st.u["men"], -1, dtype=torch.long)
    aim[0, 0] = H
    _, hp, _ = missile.volley(st.u, pw, aim, 1.0, off, contact=pw["enemy"] & (pw["gap"] <= 1.0))
    assert float(hp[0, 0, H]) == 0.0


class TestResistances:
    def resist(self, a, t, params=P, f1="wh_main_emp_empire", f2="wh2_main_skv_skaven"):
        st = build([(a, 0, 0, 0)], [(t, 0, 10, 180)], f1=f1, f2=f2)
        H = st.N // 2
        u = st.u
        old = effects.apply(u, params, params.dt, torch.ones_like(u["r"]), torch.zeros_like(u["r"]),
                            geometry.pairwise(u, SP)["dist"], u["side"][:, :, None] == u["side"][:, None, :])
        out = melee.resist_of(u, params.sim["melee"])
        effects.restore(u, old)
        return float(out[0, 0, H]), float(out[0, H, 0])

    def test_fire_on_the_horror_and_the_mutants_magic(self):
        # the Blazing Sun's flaming lances on the Horror: its fire weakness -20 % (Regeneration): x1.2
        on_horror, horror_on_sun = self.resist(SUN, HORROR)
        assert on_horror == pytest.approx(-0.2) and horror_on_sun == 0.0
        # the Mutant's magical claws on the Blazing Sun: its magic resistance, not its physical; the Sun's lances on the
        # Mutant: its 20 % physical
        sun_on_mutant, mutant_on_sun = self.resist(SUN, MUTANT)
        sun = P.units[SUN]["damage_resist"]
        assert sun_on_mutant == pytest.approx(0.2)
        assert mutant_on_sun == pytest.approx(min(0.9, (sun["magic"] + sun["all"]) / 100))
        # a spearman on the Horror: nothing; the censer (magical, small) on the Blazing Sun meets its magic resistance too
        # (magic_scope "all", the game's rule) - with "large" its physical
        assert self.resist(SPEAR, HORROR) == (0.0, 0.0)
        assert self.resist(SUN, CENSER)[1] == pytest.approx(min(0.9, (sun["magic"] + sun["all"]) / 100))
        large = P.with_cal("melee", magic_scope="large")
        assert self.resist(SUN, CENSER, params=large)[1] == pytest.approx((sun["physical"] + sun["all"]) / 100)
        # a plain (not magical) striker on the Blazing Sun keeps the physical
        assert self.resist(SUN, CLANRAT)[1] == pytest.approx((sun["physical"] + sun["all"]) / 100)

    def test_a_flaming_blow_on_the_horror_takes_a_fifth_more(self):
        st = build([(SUN, 0, -10, 0)], [(HORROR, 0, 0, 180)])
        H = st.N // 2
        u = st.u
        plain = melee.per_hit(torch.tensor(100.0), torch.tensor(50.0), u["armour"][0, H], u["hp_man"][0, H], 0.0,
                              single=torch.tensor(True))
        fire = melee.per_hit(torch.tensor(100.0), torch.tensor(50.0), u["armour"][0, H], u["hp_man"][0, H], -0.2,
                             single=torch.tensor(True))
        assert float(fire) == pytest.approx(1.2 * float(plain))


class TestRegeneration:
    def test_six_hp_a_second_half_after_fire_the_pool_three_quarters(self):
        st = build([(SPEAR, 0, -200, 0)], [(HORROR, 0, 0, 180), (MUTANT, 50, 0, 180)])
        H = st.N // 2
        u = st.u
        assert float(u["regen"][0, H]) == pytest.approx(0.001) and float(u["regen"][0, H + 1]) == 0.0
        assert float(u["heal_cap"][0, H]) == 0.75 and float(u["heal_cap"][0, 0]) == 0.0
        u["hp_abs"][0, H] = 3000.0
        u["hp_abs"][0, H + 1] = 3000.0
        got = regen.heal(u, P, 1.0)
        assert float(got[0, H]) == 6.0 and float(u["hp_abs"][0, H]) == 3006.0      # round(0.001 x 6009)
        assert float(got[0, H + 1]) == 0.0                                          # the Mutant has none
        u["burn_s"][0, H] = 5.0
        assert float(regen.heal(u, P, 1.0)[0, H]) == 3.0                            # x0.5 under fire
        u["burn_s"][0, H] = 0.0
        u["healed"][0, H] = 0.75 * 6009 - 2.0
        assert float(regen.heal(u, P, 1.0)[0, H]) == pytest.approx(2.0)             # the pool's last 2 HP
        assert float(regen.heal(u, P, 1.0)[0, H]) == 0.0                            # then none, ever
        st = build([(SPEAR, 0, -200, 0)], [(HORROR, 0, 0, 180)])
        H = st.N // 2
        st.u["hp_abs"][0, H] = 6007.0
        assert float(regen.heal(st.u, P, 1.0)[0, H]) == 2.0                         # no more than its health
        st.u["men"][0, H], st.u["hp_abs"][0, H] = 0.0, 0.0
        assert float(regen.heal(st.u, P, 1.0)[0, H]) == 0.0                         # dead: none

    def test_a_flaming_blow_starts_the_fire_clock(self):
        st = build([(SUN, 0, -10, 0), (SPEAR, 30, -10, 0)], [(HORROR, 0, 0, 180), (HORROR, 30, 0, 180)])
        H = st.N // 2
        u = st.u
        hp = torch.zeros(1, st.N, st.N)
        hp[0, 0, H] = 50.0                     # the Blazing Sun's lances on the first Horror
        hp[0, 1, H + 1] = 50.0                 # spears on the second
        regen.burn(u, P, 0.5, hp, geometry.pairwise(u, SP)["enemy"].float())
        assert float(u["burn_s"][0, H]) == 10.0 and float(u["burn_s"][0, H + 1]) == 0.0
        regen.burn(u, P, 0.5, torch.zeros_like(hp), geometry.pairwise(u, SP)["enemy"].float())
        assert float(u["burn_s"][0, H]) == 9.5

    def test_a_battle_heals_the_horror_alone(self):
        st = build([(SPEAR, 0, -300, 0)], [(HORROR, 0, 0, 180)])
        H = st.N // 2
        st.u["hp_abs"][0, H] = 3000.0
        run(st, 10.0)
        assert float(st.u["hp_abs"][0, H]) == pytest.approx(3060.0, abs=0.5)
        assert float(st.u["hp"][0, H]) == pytest.approx(3060.0 / 6009, abs=1e-4)


class TestTerror:
    def scene(self, morale, key=SPEAR, gap=3.0):
        # (the General far off: the battle goes on while the unit runs)
        st = build([(key, 0, 0, 0), (GENERAL, -400, -400, 0)], [(HORROR, 0, 0, 180)])
        H = st.N // 2
        _, depth = geometry.dims(st.u, SP)
        st.u["z"][0, H] = depth[0, 0] / 2 + 3.8 + gap
        st.u["morale"][0, 0] = morale
        return st, H

    def test_a_low_unit_within_5_m_runs_14_s_uncounted_then_is_immune(self):
        st, H = self.scene(5.0)
        run(st, P.dt)
        u = st.u
        assert bool(u["r"][0, 0]) and float(u["rout_count"][0, 0]) == 0.0
        assert float(u["ter_s"][0, 0]) == pytest.approx(14.0) and float(u["ter_imm"][0, 0]) == pytest.approx(99.0)
        # its 14 s over, it stops running (its morale above 0) - the immunity keeps it from running again at once
        u["ter_s"][0, 0] = P.dt
        u["morale"][0, 0] = 5.0
        u["x"][0, H], u["z"][0, H] = 0.0, float(u["z"][0, 0]) + 10.0
        run(st, P.dt)
        u = st.u
        assert float(u["ter_s"][0, 0]) == 0.0 and float(u["rout_count"][0, 0]) == 0.0
        assert not bool(u["r"][0, 0]) or float(u["morale"][0, 0]) <= 0
        assert float(u["ter_imm"][0, 0]) > 80

    def test_steady_or_far_or_immune_units_stay(self):
        st, _ = self.scene(40.0)                       # 40 points: not afraid enough
        run(st, P.dt)
        assert not bool(st.u["r"][0, 0]) and float(st.u["ter_s"][0, 0]) == 0.0
        st, _ = self.scene(5.0, gap=12.0)              # 12 m off: beyond terror_effect_range 5
        run(st, P.dt)
        assert float(st.u["ter_s"][0, 0]) == 0.0
        st, _ = self.scene(5.0, key="wh_dlc04_emp_inf_flagellants_0")    # unbreakable
        run(st, P.dt)
        assert not bool(st.u["r"][0, 0]) and float(st.u["ter_s"][0, 0]) == 0.0

    def test_the_monster_is_immune_while_frenzied(self):
        st = build([(SPEAR, 0, 0, 0)], [(HORROR, 0, 5, 180), (MUTANT, 10, 5, 180)])
        H = st.N // 2
        u = st.u
        old = effects.apply(u, P, P.dt, torch.ones_like(u["r"]), torch.zeros_like(u["r"]),
                            geometry.pairwise(u, SP)["dist"], u["side"][:, :, None] == u["side"][:, None, :])
        assert bool(u["terror"][0, H]) and bool(u["terror"][0, H + 1]) and not bool(u["terror"][0, 0])
        assert bool(u["itp"][0, H]) and bool(u["fear"][0, H])          # frenzy: immune to psychology
        effects.restore(u, old)


def test_the_horrors_blows_lay_poison():
    st = build([(SPEAR, 0, 0, 0)], [(HORROR, 0, 5, 180)])
    H = st.N // 2
    keys = [k for _, _, k in effects.contact_keys(P)]
    assert int(st.u["mfx"][0, H]) == effects.index(P)["wh_main_unit_contact_poison"]
    assert "wh_main_unit_contact_poison" in keys


class TestImpact:
    def test_the_lane_is_the_body_not_the_template(self):
        st = build([(SPEAR, 0, 0, 0), ("wh_main_emp_cav_empire_knights", 100, 0, 0)],
                   [(HORROR, 0, 30, 180), (MUTANT, 30, 30, 180), (OGRES, 60, 30, 180)])
        H = st.N // 2
        c = impact.per_rank(st.u, P)
        h = P.spacing_of(SPEAR)[0]
        assert float(c[0, H, 0]) == pytest.approx(2 * 3.8 * 0.36 / h)           # 1.85, the template's 18 m gave 12.2
        assert float(c[0, H + 1, 0]) == pytest.approx(4.0 / h)
        assert float(c[0, H + 2, 0]) == pytest.approx(2.0 / h)                   # 1.35, the template's 6.5 m 4.4
        # the cavalry keeps its template's place (bit for bit)
        off = impact.per_rank(st.u, P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], lane_body=0)))
        assert float(c[0, 1, H]) == float(off[0, 1, H])

    def test_monsters_and_lords_keep_their_feet_infantry_falls(self):
        st = build([(GENERAL, 0, 0, 0), (SPEAR, 40, 0, 0)],
                   [(HORROR, 0, 30, 180), (MUTANT, 30, 30, 180), (OGRES, 60, 30, 180), (WARLORD, 90, 30, 180)])
        H = st.N // 2
        stay = impact.ignore(st.u, P)                          # knock_ignore "all": the game's rule for every body
        assert [round(float(stay[0, H + k]), 2) for k in range(4)] == [0.6, 0.65, 0.2, 0.75]
        assert float(stay[0, 0]) == pytest.approx(0.4) and float(stay[0, 1]) == 0.0
        large = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], knock_ignore="large"))
        assert float(impact.ignore(st.u, large)[0, 0]) == 0.0

    def test_the_knights_knock_the_warlord_less_by_his_75_percent(self):
        st = build([("wh_main_emp_cav_empire_knights", 0, 0, 0)], [(WARLORD, 0, 10, 180)])
        H = st.N // 2
        rate = torch.zeros(1, st.N, st.N)
        rate[0, 0, H] = 50.0
        mhit = torch.full((1, st.N, st.N), 10.0)
        F = torch.ones(1, st.N, st.N)
        z = torch.zeros_like(st.u["men"])
        out = {}
        for scope in ("all", None):
            u = {k: v.clone() for k, v in st.u.items()}
            params = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], knock_ignore=scope))
            impact.knock(u, params, 1.0, rate, torch.zeros_like(rate), mhit, F, z)
            out[scope] = float(u["kd_n"][0, H])
        assert out[None] > 0 and out["all"] == pytest.approx(out[None] * (1 - 0.75), rel=1e-5)

    def test_infantry_never_knocks_a_monster_its_blows_always_knock_a_man(self):
        st = build([(SPEAR, 0, 0, 0)], [(HORROR, 0, 5, 180)])
        H = st.N // 2
        big = impact.bigger(st.u)
        assert big[0, H, 0] and not big[0, 0, H]
        r = float(st.u["mass"][0, H] / st.u["mass"][0, 0])
        assert r == 25.0 and 0.05 * r * 1.0 >= 1.0           # melee_on_damage_effect_knockdown_chance x r: always

    def test_the_rat_ogres_brace_at_2786(self):
        st = build([(SPEAR, 0, 0, 0)], [(OGRES, 0, 50, 180, {"width": 39})])
        H = st.N // 2
        speed = torch.zeros_like(st.u["men"])
        B = impact.bracing(st.u, P, speed, torch.zeros_like(st.u["men"]))
        files, ranks = impact.ranks_of(st.u, SP)
        assert float(files[0, H]) == 6 and float(ranks[0, H]) == 3
        assert float(B[0, H]) * 1500 == pytest.approx(2786, abs=1)               # 1500 x (1 + 3 x 2 / 7)


def test_the_horrors_charge_meets_the_men_its_body_covers():
    # the Brood Horror runs at 12 m/s into held Empire spearmen: each rank it knocks down is ~1.85 men (its body 2.74 m over
    # their 1.48 m files), not the 12 of its template's 18 m; the Rat Ogres' first rank meets 6 lanes x 1.35 = 8.1 men
    out = {}
    for key in (HORROR, OGRES):
        st = build([(SPEAR, 0, 0, 0)], [(key, 0, 20, 180, {"width": 39})])
        H = st.N // 2
        u = st.u
        front, depth = geometry.dims(u, SP)
        u["z"][0, H] = depth[0, 0] / 2 + depth[0, H] / 2
        u["vz"][0, H] = -12.0 if key == HORROR else -8.0
        pw = geometry.pairwise(u, SP)
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        fire = torch.zeros_like(u["r"])
        fire[0, H] = True
        speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
        charge = fire.float()
        impact.land(u, pw, contact, fire, P, P.dt, speed, charge)
        out[key] = float(u["kd_n"][0, 0])
    h = P.spacing_of(SPEAR)[0]
    assert 0 < out[HORROR] <= 8 * 2 * 3.8 * 0.36 / h + 1e-3          # at most max_ranks 8 ranks of 1.85
    place = impact.lane_place(st.u, P)
    assert float(place[0, H]) == pytest.approx(2.0) and float(place[0, 0]) == pytest.approx(h)
    assert out[OGRES] <= 8 * 6 * 2.0 / h + 1e-3


def test_the_general_keeps_his_feet_and_his_health_against_the_horrors_charge_by_40_percent():
    # knock_ignore "all": the Brood Horror's charge would knock the General down (dv 12.6 m/s over 7); he stays up with
    # his 40 % - fewer down, and the large charger's collision damage (only on the men it knocks down) x0.6 too
    out = {}
    for scope in ("all", None):
        st = build([(GENERAL, 0, 0, 0)], [(HORROR, 0, 20, 180)])
        H = st.N // 2
        u = st.u
        front, depth = geometry.dims(u, SP)
        u["z"][0, H] = depth[0, 0] / 2 + depth[0, H] / 2
        u["vz"][0, H] = -12.0
        params = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], knock_ignore=scope))
        pw = geometry.pairwise(u, SP)
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        fire = torch.zeros_like(u["r"])
        fire[0, H] = True
        speed = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)
        hp = impact.land(u, pw, contact, fire, params, params.dt, speed, fire.float())
        out[scope] = (float(hp[0, H, 0]), float(u["kd_n"][0, 0]), float(u["kd_q"][0, 0]))
    assert out[None][0] > 0 and out[None][1] > 0
    assert out["all"][0] == pytest.approx(0.6 * out[None][0], rel=1e-5)
    assert out["all"][1] == pytest.approx(0.6 * out[None][1], rel=1e-5)
    assert out["all"][2] == pytest.approx(0.6 * out[None][2], rel=1e-5)
