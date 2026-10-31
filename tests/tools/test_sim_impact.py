"""The cavalry's second step in the simulator (tools/nn/sim/impact.py; build/diag_cav/mechanic.md M1-M11): the impact of
a large charger's landing (bracing, the speed change, ranks, collision damage, the men lying), the knock of a large
striker's blows by the mass ratio, the charge in a large unit's pursuit, fear, the cavalry's run fatigue. Every expected
number is the database's formula worked out here. Needs torch (the container)."""
import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, impact, melee, morale, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
R = P.battle
KNIGHTS, DEMI, ROSE = ("wh_main_emp_cav_empire_knights", "wh_main_emp_cav_demigryph_knights_0",
                       "wh3_dlc25_emp_cav_knights_of_the_black_rose")
SPEAR, SWORD = "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_swordsmen"
CSPEAR, CLANRATS, SVHALB = "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_clanrats_1", \
    "wh2_main_skv_inf_stormvermin_0"
SP = P.sim["formation"]["spacing_m"]


def army(side1, side2):
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b, "width": 30} for k, x, z, b in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def touching(striker, target, t_bearing=270.0, params=P, gap=0.0):
    """The striker (side 1, slot 0, facing east) and the target (side 2, its first slot) `gap` m edge to edge."""
    st = scenario.build([army([(striker, 0, 0, 90)], [(target, 60, 0, t_bearing)])], params)
    H = st.N // 2
    front, depth = geometry.dims(st.u, SP)
    reach = float(depth[0, H]) if t_bearing in (90.0, 270.0) else float(front[0, H])
    st.u["x"][0, H] = (float(depth[0, 0]) + reach) / 2 + gap
    st.u["order_kind"][0, 0], st.u["order_target"][0, 0] = O.ATTACK, H
    return st, H


def landing(striker, target, v=10.8, t_bearing=270.0, target_v=0.0, params=P):
    """impact.land of the striker charging east at v m/s into the target (moving target_v m/s east): (hp, st, H)."""
    st, H = touching(striker, target, t_bearing, params)
    st.u["vx"][0, 0] = v
    st.u["vx"][0, H] = target_v
    pw = geometry.pairwise(st.u, SP)
    fire = torch.zeros_like(st.u["r"])
    fire[0, 0] = True
    speed = torch.sqrt(st.u["vx"] ** 2 + st.u["vz"] ** 2)
    hp = impact.land(st.u, pw, pw["enemy"], fire, params, params.dt, speed, torch.zeros_like(st.u["men"]))
    return float(hp[0, 0, H]), st, H


def keep(st):
    """Every unit keeps its order in force."""
    o = O.hold(st.B, st.N)
    o.kind.fill_(O.KEEP)
    return o


def lane_of(st, H):
    h, _ = melee.spacing_of(st.u, SP)
    files, ranks = impact.ranks_of(st.u, SP)
    return min(float(files[0, 0]) * float(h[0, 0]) / float(h[0, H]), float(files[0, H])), float(ranks[0, H])


def collision(v, M, mB, st, H):
    """HP of one man met: min(70, (v M / mB)^0.8 x 0.6), 70 % past the armour (melee.per_hit's mean)."""
    d = min(R["collision_damage_maximum"], (v * M / mB) ** R["collision_damage_normaliser"] * R["collision_damage_modifier"])
    ap = R["collision_damage_armour_penetration_ratio"]
    per = melee.per_hit(torch.tensor(d * (1 - ap)), torch.tensor(d * ap), st.u["armour"][0, H], st.u["hp_man"][0, H],
                        st.u["resist_physical"][0, H])
    return min(float(per), d)


class TestBracing:
    def test_the_multiplier_by_ranks_and_charge_defence(self):
        st, H = touching(KNIGHTS, CSPEAR)
        still = torch.zeros_like(st.u["men"])
        B = impact.bracing(st.u, P, still, still)
        _, ranks = impact.ranks_of(st.u, SP)
        n = float(ranks[0, H])
        expect = min(10.0, (1 + 3 * min(max((n - 1) / 7, 0), 1)) * 2)          # charge_defense_vs_large: x2
        assert float(B[0, H]) == pytest.approx(expect)
        assert float(B[0, 0]) == 1.0                                              # the knights do not brace (can_brace 0)
        st2, H2 = touching(KNIGHTS, CLANRATS)
        _, r2 = impact.ranks_of(st2.u, SP)
        assert float(impact.bracing(st2.u, P, still, still)[0, H2]) == pytest.approx(
            1 + 3 * min(max((float(r2[0, H2]) - 1) / 7, 0), 1))                  # no charge defence: x1
        moving = still.clone()
        moving[0, H2] = 1.0
        assert float(impact.bracing(st2.u, P, moving, still)[0, H2]) == 1.0      # on the move: no bracing
        charging = still.clone()
        charging[0, H2] = 0.5
        assert float(impact.bracing(st2.u, P, still, charging)[0, H2]) == 1.0    # in its own charge: no bracing


class TestLanding:
    def test_braced_spearmen_go_down_in_the_first_rank_only(self):
        hp, st, H = landing(KNIGHTS, CSPEAR)
        B = float(impact.bracing(st.u, P, torch.zeros_like(st.u["men"]), torch.zeros_like(st.u["men"]))[0, H])
        mB = 100.0 * B
        dv = 1.3 * 1000 / (mB + 1000) * 10.8
        assert 7.0 < dv < 10.5                                                   # M9: 8.6, knocked down
        lane, _ = lane_of(st, H)
        h, _ = melee.spacing_of(st.u, SP)
        c = float(h[0, 0] / h[0, H])
        v2 = 10.8 * (1 - 1.3 * mB / (mB + 1000) * 0.8) ** c
        assert 1.3 * 1000 / (mB + 1000) * v2 < 7.0                               # the 2nd rank stays up
        assert float(st.u["kd_q"][0, H]) == 1.0 and float(st.u["kd_t"][0, H]) == 0.0
        assert float(st.u["kd_n"][0, H]) == pytest.approx(lane)
        assert hp == pytest.approx(lane * (collision(10.8, 1000, mB, st, H) + collision(v2, 1000, mB, st, H)), rel=1e-4)

    def test_braced_stormvermin_stay_up(self):
        hp, st, H = landing(KNIGHTS, SVHALB)
        assert float(st.u["kd_q"][0, H]) == 0.0 and float(st.u["kd_n"][0, H]) == 0.0      # dv 6.4: knocked back only
        lane, _ = lane_of(st, H)
        B = float(impact.bracing(st.u, P, torch.zeros_like(st.u["men"]), torch.zeros_like(st.u["men"]))[0, H])
        assert 150.0 * B == pytest.approx(1200.0)                                # 8 ranks x 2: B 8
        assert hp == pytest.approx(lane * collision(10.8, 1000, 1200.0, st, H), rel=1e-4)   # only the first rank met

    def test_an_unbraced_target_goes_down_rank_after_rank(self):
        hp, st, H = landing(KNIGHTS, CLANRATS, v=10.8, target_v=-1.0)           # walking into the charge: no bracing
        lane, ranks = lane_of(st, H)
        h, _ = melee.spacing_of(st.u, SP)
        c = float(h[0, 0] / h[0, H])
        v, down, total = 11.8, 0, 0.0
        for _ in range(int(ranks)):
            total += lane * collision(v, 1000, 100.0, st, H)
            if 1.3 * 1000 / 1100 * v <= 7.0:
                break
            down += 1
            v *= (1 - 1.3 * 100 / 1100 * 0.8) ** c
        assert down >= 3                                                          # M9: 12.8 m/s into the first man
        assert float(st.u["kd_n"][0, H]) == pytest.approx(down * lane)
        assert hp == pytest.approx(total, rel=1e-4)

    def test_from_the_flank_the_brace_does_not_count(self):
        _, st, H = landing(KNIGHTS, SVHALB, t_bearing=0.0)                       # the target faces north
        assert float(st.u["kd_q"][0, H]) == 1.0                                   # B 1: dv 11.8, down

    @pytest.mark.parametrize("t_bearing, deg, braced", [(0.0, 120, True), (350.0, 120, True), (0.0, 75, False),
                                                         (120.0, 120, False)])
    def test_the_damage_keeps_the_brace_within_120_deg(self, t_bearing, deg, braced):
        """damage_brace_deg (build/diag_cav/mechanic.md M30): the speed change takes B only within brace_deg 75 - from the
        flank B 1, the men go down rank after rank - while each man met takes the collision damage divided by his brace B
        (stormvermin 150 x 8) within damage_brace_deg 120 of his facing: the charger (from the west) 90 deg off a target
        facing north, 80 deg off one facing 350; from the rear cone (facing 120: 150 deg off) by 150."""
        imp = P.sim["contact"]["impact"]
        assert imp.get("damage_brace_deg") == 120
        p = P.with_cal("contact", impact=dict(imp, damage_brace_deg=deg))
        hp, st, H = landing(KNIGHTS, SVHALB, t_bearing=t_bearing, params=p)
        _, ranks = lane_of(st, H)
        h, _ = melee.spacing_of(st.u, SP)
        c = float(h[0, 0] / h[0, H])
        # the horses along the side the knights meet, their lanes' men a rank
        pw = geometry.pairwise(st.u, SP)
        assert abs(float(pw["rel_j"][0, 0, H])) / geometry.DEG > 75.0
        files, _ = impact.ranks_of(st.u, SP)
        face = min(float(pw["face_i"][0, 0, H]), float(pw["face_j"][0, 0, H]))
        lane = min(min(face / float(h[0, 0]), float(files[0, 0])) * c, float(files[0, H]))
        mB = 1200.0 if braced else 150.0
        v, total = 10.8, 0.0
        for _ in range(int(ranks)):
            total += lane * collision(v, 1000, mB, st, H)
            if 1.3 * 1000 / 1150 * v <= 7.0:
                break
            v *= (1 - 1.3 * 150 / 1150 * 0.8) ** c
        assert float(st.u["kd_q"][0, H]) == 1.0                                    # B 1 in the speed change: down
        assert hp == pytest.approx(total, rel=1e-4)

    def test_no_landing_no_impact_and_infantry_never(self):
        hp, st, H = landing(SWORD, CLANRATS)                                     # a small charger: nothing
        assert hp == 0.0 and float(st.u["kd_n"][0, H]) == 0.0

    def test_a_new_charge_lands_again(self):
        _, st, H = landing(KNIGHTS, CSPEAR)
        st.u["kd_t"][0, H] = 5.0                                                  # up again
        pw = geometry.pairwise(st.u, SP)
        fire = torch.zeros_like(st.u["r"])
        fire[0, 0] = True
        speed = torch.sqrt(st.u["vx"] ** 2 + st.u["vz"] ** 2)
        impact.land(st.u, pw, pw["enemy"], fire, P, P.dt, speed, torch.zeros_like(st.u["men"]))
        assert float(st.u["kd_t"][0, H]) == 0.0 and float(impact.lying(st.u, P)[0, H]) == 1.0


class TestLying:
    def test_the_men_get_up_over_the_database_s_1_0_to_2_7_s(self):
        # lying U(1.0, 2.7) s (the database) + the getting-up animation (contact.impact wave rise_s 1.1, measured: M33)
        st, H = touching(KNIGHTS, CSPEAR)
        st.u["kd_q"][0, H] = 1.0
        for t, share in ((0.5, 1.0), (2.1, 1.0), (2.95, 0.5), (3.8, 0.0), (5.0, 0.0)):
            st.u["kd_t"][0, H] = t
            assert float(impact.lying(st.u, P)[0, H]) == pytest.approx(share, abs=1e-6), t

    def test_without_the_wave_no_getting_up_time(self):
        p = P.with_cal("contact", impact=dict(P.sim["contact"]["impact"], wave={"on": 0, "rise_s": 1.1}))
        assert impact.rise(p) == 0.0 and impact.rise(P) == pytest.approx(1.1)
        st, H = touching(KNIGHTS, CSPEAR, params=p)
        st.u["kd_q"][0, H] = 1.0
        for t, share in ((1.0, 1.0), (1.85, 0.5), (2.7, 0.0)):
            st.u["kd_t"][0, H] = t
            assert float(impact.lying(st.u, p)[0, H]) == pytest.approx(share, abs=1e-6), t

    def test_the_lying_men_do_not_strike_the_large_enemy(self):
        st, H = touching(KNIGHTS, CSPEAR)
        st.u["kd_q"][0, H] = 1.0
        cut = impact.up_share(st.u, P, torch.ones(1, st.N, st.N))           # (in place: a fresh rate each call)
        assert float(cut[0, H, 0]) == 0.0                                         # the spearmen on the knights
        assert float(cut[0, 0, H]) == 1.0                                         # the knights' blows land
        st.u["kd_t"][0, H] = 1.85 + impact.rise(P)
        assert float(impact.up_share(st.u, P, torch.ones(1, st.N, st.N))[0, H, 0]) == pytest.approx(0.5)


class TestKnockByBlows:
    def strikes(self, striker, target, charge):
        st, H = touching(striker, target)
        pw = geometry.pairwise(st.u, SP)
        ch = torch.zeros_like(st.u["men"])
        ch[0, 0] = charge
        rate, mhit, _, F = melee.strikes(st.u, pw, pw["enemy"], P, ch)
        return st, H, rate, mhit, F, ch

    @pytest.mark.parametrize("charge, p", [(0.0, 0.5), (1.0, 1.0)])
    def test_a_horse_knocks_a_man_at_0_05_r_1_plus_2f(self, charge, p):
        st, H, rate, mhit, F, ch = self.strikes(KNIGHTS, CLANRATS, charge)
        dt = P.dt
        impact.knock(st.u, P, dt, rate, torch.zeros_like(rate), mhit, F, ch)
        hits = float(rate[0, 0, H]) * dt / float(mhit[0, 0, H])
        kill = min(1.0, float(mhit[0, 0, H]) / float(st.u["hp_man"][0, H]))
        ff = melee.files_share(P.sim["melee"])
        front = float(F[0, H, 0]) / ff
        expect = min(1.0, hits * p * (1 - kill) / max(front, 1.0)) * math.exp(-dt / 1.85)
        assert float(st.u["kd_p"][0, H]) == pytest.approx(expect, rel=1e-4)
        assert float(st.u["kd_p"][0, 0]) == 0.0                                   # the infantry knocks no horse

    def test_infantry_on_infantry_knocks_nobody(self):
        st, H, rate, mhit, F, ch = self.strikes(SWORD, CLANRATS, 1.0)
        impact.knock(st.u, P, P.dt, rate, torch.zeros_like(rate), mhit, F, ch)
        assert float(st.u["kd_p"].abs().sum()) == 0.0 and float(st.u["kd_n"].abs().sum()) == 0.0


class TestPursuitCharge:
    def chase(self, striker, params):
        st, H = touching(striker, CLANRATS, params=params)
        st.u["r"][0, H] = True
        pw = geometry.pairwise(st.u, SP)
        return float(melee.strikes(st.u, pw, pw["enemy"], params, torch.zeros_like(st.u["men"]))[0][0, 0, H])

    def test_a_horse_strikes_a_router_with_its_whole_charge(self):
        off = P.with_cal("contact", pursuit_large_charge=0)
        assert self.chase(KNIGHTS, P) > 1.2 * self.chase(KNIGHTS, off)
        assert self.chase(SWORD, P) == self.chase(SWORD, off)                     # infantry: the measured pursuit

    def test_a_horse_sprints_after_a_router(self):
        def run(params):
            st, H = touching(KNIGHTS, CLANRATS, params=params, gap=20.0)
            st.u["r"][0, H] = True
            st.u["vx"][0, 0] = float(st.u["run"][0, 0])
            st.u["leadership"] = st.u["leadership"] + 1e4
            for _ in range(4):
                battle.step(st, keep(st), params, params.dt)
            return float(torch.sqrt(st.u["vx"][0, 0] ** 2 + st.u["vz"][0, 0] ** 2))
        assert run(P) > float(P.units[KNIGHTS]["speed"]["run"]) + 0.1
        assert run(P.with_cal("contact", pursuit_large_charge=0)) <= float(P.units[KNIGHTS]["speed"]["run"]) + 1e-3


class TestFear:
    def test_the_term_is_the_database_s_minus_8(self):
        st, H = touching(KNIGHTS, CLANRATS)
        assert P.morale["ume_concerned_unit_frightened"] == -8.0
        assert P.morale["fear_effect_range"] == 20.0
        assert bool(st.u["fear"][0, 0]) is False

    def test_demigryphs_frighten_within_20_m_of_their_edge_not_the_immune(self):
        def points(striker, target, gap, fear_on=True):
            params = P if fear_on else P.with_cal("morale", fear={"on": 0})
            st = scenario.build([army([(striker, 0, 0, 90)], [(target, 60, 0, 270)])], params)
            H = st.N // 2
            front, depth = geometry.dims(st.u, SP)
            st.u["x"][0, H] = (float(depth[0, 0]) + float(depth[0, H])) / 2 + gap
            st.u["leadership"] = st.u["leadership"] + 1e4
            battle.step(st, keep(st), params, params.dt)
            return st, H
        st_on, H = points(DEMI, CLANRATS, 10.0)
        st_off, _ = points(DEMI, CLANRATS, 10.0, fear_on=False)
        assert float(st_on.u["morale"][0, H]) < float(st_off.u["morale"][0, H])  # within 20 m: afraid
        st_far, _ = points(DEMI, CLANRATS, 30.0)
        st_far_off, _ = points(DEMI, CLANRATS, 30.0, fear_on=False)
        assert float(st_far.u["morale"][0, H]) == float(st_far_off.u["morale"][0, H])
        assert bool(st_on.u["fear"][0, 0]) and not bool(st_on.u["itp"][0, 0])
        # the Black Rose is immune to psychology: a Skaven fear-causer made by hand (none in the pool) 5 m from the Black
        # Rose and the Empire Knights side by side
        fake = dict(P.units[CLANRATS], attributes=list(P.units[CLANRATS]["attributes"]) + ["causes_fear"])
        params = dataclasses.replace(P, units={**P.units, "fear_rats": fake})
        st = scenario.build([army([(ROSE, 0, 0, 90), (KNIGHTS, 0, -40, 90)], [("fear_rats", 60, -20, 270)])], params)
        H = st.N // 2
        _, depth = geometry.dims(st.u, SP)
        st.u["x"][0, H] = (float(depth[0, 0]) + float(depth[0, H])) / 2 + 5.0
        assert bool(st.u["itp"][0, 0]) and not bool(st.u["itp"][0, 1]) and bool(st.u["fear"][0, H])
        seen = {}

        def spy(u, ctx, params_, dt, _orig=morale.step):
            seen["fear"] = ctx.get("fear")
            return _orig(u, ctx, params_, dt)
        orig, battle.morale.step = battle.morale.step, spy
        try:
            battle.step(st, keep(st), params, params.dt)
        finally:
            battle.morale.step = orig
        assert seen["fear"] is not None and not bool(seen["fear"][0, 0]) and bool(seen["fear"][0, 1])
        assert not bool(seen["fear"][0, H])                                      # a fear-causer is not afraid


class TestRunFatigue:
    def test_the_cavalry_runs_at_running_cavalry_6(self):
        st, H = touching(KNIGHTS, CLANRATS)
        assert P.fatigue["running_cavalry"] == 6.0 and P.fatigue["running"] == 4.0
        assert float(st.u["fat_run"][0, 0]) == 6.0 and float(st.u["fat_run"][0, H]) == 4.0


class TestNoLargeUnitNoChange:
    def test_an_infantry_battle_is_the_same_bit_for_bit(self):
        off = P.with_cal("contact", impact={"on": 0}, pursuit_large_charge=0).with_cal("morale", fear={"on": 0})
        def run(params):
            st = scenario.build([army([(SWORD, 0, 0, 90), (SPEAR, 0, 40, 90)], [(CLANRATS, 40, 0, 270),
                                                                               (CSPEAR, 40, 40, 270)])], params)
            st.u["order_kind"][0, 0], st.u["order_target"][0, 0] = O.ATTACK, st.N // 2
            st.u["order_kind"][0, 1], st.u["order_target"][0, 1] = O.ATTACK, st.N // 2 + 1
            for _ in range(60):
                battle.step(st, keep(st), params, params.dt)
            return st
        a, b = run(P), run(off)
        for k, v in a.u.items():
            if k.startswith("kd_"):
                continue
            assert torch.equal(v, b.u[k]), k
