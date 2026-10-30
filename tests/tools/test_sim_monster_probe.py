"""The simulator against the monster and artillery probes (30.10.2026; build/diag_skv_mon/probe_results.md, the twin
build/diag_simfix): a formation meets a monster with its face (half the ring, contact.ring_big face), the men round it
strike in full in gather_s, the rat ogres' sparse formation strikes with two bodies a place (contact.sparse), the
artillery's morale (-8 attacked / -10 damaged, artillery.morale), a magical projectile's own hit against spell
resistance. Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import artillery, battle, geometry, melee, missile, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
OGRES, HORROR, MUTANT = ("wh2_main_skv_mon_rat_ogres", "wh2_dlc16_skv_mon_brood_horror_0",
                         "wh2_dlc16_skv_mon_rat_ogre_mutant")
SPEAR, KNIGHTS = "wh_main_emp_inf_spearmen_0", "wh_main_emp_cav_empire_knights"
SUN, CLANRAT, CSPEAR = ("wh_dlc04_emp_cav_knights_blazing_sun_0", "wh2_main_skv_inf_clanrats_0",
                        "wh2_main_skv_inf_clanrat_spearmen_0")
WARLORD = "wh2_main_skv_cha_warlord_0"
MORTAR, CANNON, WARP = "wh_main_emp_art_mortar", "wh_main_emp_art_great_cannon", "wh2_main_skv_art_warp_lightning_cannon"
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"
SP = P.sim["formation"]["spacing_m"]


def army(side1, side2, f1=EMP, f2=SKV):
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b, **(extra[0] if extra else {})} for k, x, z, b, *extra in rows]
    return {"attacker": 1, "sides": {1: {"faction": f1, "units": units(side1)}, 2: {"faction": f2, "units": units(side2)}}}


def build(side1, side2, params=P, **kw):
    return scenario.build([army(side1, side2, **kw)], params)


def strikes(st, params=P):
    pw = geometry.pairwise(st.u, SP)
    contact = pw["enemy"] & (pw["gap"] <= 1.0)
    return melee.strikes(st.u, pw, contact, params, torch.zeros_like(st.u["men"])), contact


class TestTheRingsFace:
    def test_one_formation_brings_half_the_ring_two_round_it_the_whole(self):
        for target in (MUTANT, HORROR):
            st = build([(SPEAR, 0, -6, 0)], [(target, 0, 0, 90)])
            H = st.N // 2
            (_, _, _, F), contact = strikes(st)
            assert contact[0, 0, H]
            n = float(melee.ring(st.u, (st.u["men0"] <= 1)[:, None, :])[0, 0, H])
            assert float(F[0, 0, H]) == pytest.approx(0.5 * n, rel=1e-4), target         # 6.4 / 8.6
            st = build([(SPEAR, 0, -6, 0), (SPEAR, 0, 6, 180)], [(target, 0, 0, 90)])
            H = st.N // 2
            (_, _, _, F), contact = strikes(st)
            assert contact[0, 0, H] and contact[0, 1, H]
            assert float(F[0, :2, H].sum()) == pytest.approx(n, rel=1e-4)
        assert melee.ring_face(P.sim["contact"]) == 0.5
        assert melee.ring_face({"ring_big": {"on": 1}}) == 1.0                         # (absent: the whole ring)

    def test_the_men_round_a_monster_strike_in_full_in_gather_s_a_lord_in_20(self):
        rates = {}
        for target in (MUTANT, WARLORD):
            st = build([(SPEAR, 0, -6, 0)], [(target, 0, 0, 180)])
            H = st.N // 2
            st.u["ran_in"][0, H] = True
            out = {}
            for t in (1.5, 3.0, 10.0, 20.0):
                st.u["contact_s"][0, :] = t
                (rate, _, _, _), _ = strikes(st)
                out[t] = float(rate[0, 0, H])
            rates[target] = out
        assert P.sim["contact"]["ring_big"]["gather_s"] == 3.0
        m = rates[MUTANT]
        assert m[1.5] == pytest.approx(0.5 * m[3.0], rel=1e-3) and m[3.0] == pytest.approx(m[20.0], rel=1e-3)
        w = rates[WARLORD]
        assert w[10.0] == pytest.approx(0.5 * w[20.0], rel=1e-3)                      # the lord's 20 s stays


class TestTheSparseFormation:
    def test_two_ogres_a_place_dense_infantry_and_horses_keep_their_files(self):
        st = build([(SPEAR, 0, -10, 0), (KNIGHTS, 100, -10, 0)], [(OGRES, 0, 0, 180, {"width": 39.0}),
                                                                   (CLANRAT, 100, 0, 180)])
        H = st.N // 2
        h, _ = melee.spacing_of(st.u, SP)
        hc, _ = melee.contact_spacing(st.u, h, h, P.sim["melee"].get("body_across"))
        k = melee.sparse_files(st.u, hc, P.sim["contact"])[0, :, 0]
        assert float(k[H]) == 2.0                                    # floor(6.5 / (2 x 1.0 + 0.5))
        assert float(k[0]) == float(k[1]) == float(k[H + 1]) == 0.0   # spearmen, knights (1.6 m across a 3 m place), clanrats
        assert melee.sparse_files(st.u, hc, {}) is None

    def test_the_ogres_bring_k_fit_x_the_contact_over_h(self):
        F = {}
        for on in (1, 0):
            params = P.with_cal("contact", sparse=dict(P.sim["contact"]["sparse"], on=on))
            st = build([(SPEAR, 0, -12, 0, {"width": 30.0})], [(OGRES, 0, 0, 180, {"width": 39.0})], params=params)
            H = st.N // 2
            st.u["contact_s"].fill_(100.0)                # (the steady fight: no opening wave)
            st.u["melee_s"].fill_(100.0)
            (_, _, _, f), contact = strikes(st, params)
            assert contact[0, H, 0]
            F[on] = float(f[0, H, 0])
        # off: the fighting_files share of the files on the contact (0.5 x (30 + the wrap at the ends) / 6.5 = 3.0); on: two
        # bodies a place, 4x (12 of 16)
        assert 2.3 <= F[0] < 4
        assert F[1] == pytest.approx(2 / P.sim["melee"]["fighting_files"] * F[0], rel=1e-4) and F[1] <= 16


class TestArtilleryMorale:
    def test_the_marks_and_their_points(self):
        st = build([(MORTAR, 0, 0, 0)], [(CSPEAR, 0, 200, 180), (CLANRAT, 100, 200, 180), (CLANRAT, 300, 200, 180)])
        u, H, dt = st.u, st.N // 2, P.dt
        assert float(u["art_s"][0, H]) == 1e6 and float(artillery.morale_points(u, P)[0, H]) == 0.0
        fired, near, hit = (torch.zeros_like(u["r"]) for _ in range(3))
        fired[0, H] = True
        near[0, H + 1] = True
        hit[0, H + 2] = True
        artillery.morale_mark(u, P, dt, fired, near, hit)
        pts = artillery.morale_points(u, P)[0]
        assert float(u["art_s"][0, H]) == -2.0 and float(pts[H]) == 0.0              # fired at: marked 2 s later
        assert float(pts[H + 1]) == -8.0 and float(pts[H + 2]) == -10.0              # a near miss, a hit
        none = torch.zeros_like(u["r"])
        for _ in range(int(round(2.0 / dt))):
            artillery.morale_mark(u, P, dt, none, none, none)
        assert float(artillery.morale_points(u, P)[0, H]) == -8.0
        for _ in range(int(round(30.0 / dt))):
            artillery.morale_mark(u, P, dt, none, none, none)
        pts = artillery.morale_points(u, P)[0]
        assert float(pts[H]) == float(pts[H + 1]) == float(pts[H + 2]) == 0.0       # 30 s without a new shot
        assert artillery.near_m(P) == 12.0

    def test_a_block_under_the_mortar_takes_the_term_one_far_off_does_not(self):
        st = build([(MORTAR, 0, 0, 0)], [(CSPEAR, 0, 200, 180), (CLANRAT, 400, 200, 180)])
        H = st.N // 2
        for _ in range(int(round(20 / P.dt))):
            battle.step(st, O.hold(st.B, st.N), P, P.dt)
        pts = artillery.morale_points(st.u, P)[0]
        assert float(pts[H]) in (-8.0, -10.0) and float(pts[H + 1]) == 0.0
        assert float(st.u["art_s"][0, H + 1]) == 1e6
        off = P.with_cal("artillery", morale=dict(P.sim["artillery"]["morale"], on=0))
        assert artillery.morale_points(st.u, off) is None


def test_a_magical_projectiles_own_hit_meets_spell_resistance_not_the_physical():
    hit = {}
    for gun, f1, f2 in ((WARP, SKV, EMP), (CANNON, EMP, SKV)):
        target_key = SUN if f2 == EMP else CLANRAT
        st = build([(gun, 0, 0, 0)], [(target_key, 0, 200, 180)], f1=f1, f2=f2)
        u, H = st.u, st.N // 2
        pw = geometry.pairwise(u, SP)
        target = torch.full_like(u["side"], -1)
        target[0, 0] = H
        u["hp_man"][0, H] = 10000.0
        base = float(missile.volley(u, pw, target, P.dt, P, loaded=torch.ones_like(u["men"]))[2][0, 0, H])
        u["resist_magic"][0, H] = 0.5
        hit[gun] = (base, float(missile.volley(u, pw, target, P.dt, P, loaded=torch.ones_like(u["men"]))[2][0, 0, H]))
        assert bool(u["p_magic"][0, 0]) is (gun == WARP)
    w0, w1 = hit[WARP]
    assert w1 < w0                                     # the warp lightning: the knights' spell resistance cuts it
    c0, c1 = hit[CANNON]
    assert c1 == c0                                    # a cannonball is not magical
