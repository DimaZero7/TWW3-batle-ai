"""The cavalry's first rules in the simulator (build/diag_cav/sim_gap.md 1, 10, 15): charge defence against large
chargers, a splash blow by the target's size, the charge's front rank by the formation template. The cavalry is the
probes' only (data/units/probe-units.json): the passports join the pool's here. Needs torch (the container)."""
import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from tools import config as project  # noqa: E402
from tools.nn.sim import geometry, melee, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

PROBE = json.loads((project.ROOT / "data/units/probe-units.json").read_text(encoding="utf-8"))["units"]
P = dataclasses.replace(load(), units={**PROBE, **load().units})
KNIGHTS, DEMI = "wh_main_emp_cav_empire_knights", "wh_main_emp_cav_demigryph_knights_0"
CSPEAR, CLANRATS, SVHALB = "wh2_main_skv_inf_clanrat_spearmen_0", "wh2_main_skv_inf_clanrats_1", \
    "wh2_main_skv_inf_stormvermin_0"
SWORD, SPEAR, GENERAL = "wh_main_emp_inf_swordsmen", "wh_main_emp_inf_spearmen_0", "wh_main_emp_cha_general_0"
WOLF = "wh2_dlc16_skv_mon_wolf_rats_0"


def army(side1, side2):
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b, "width": 30, "general": k == GENERAL} for k, x, z, b in rows]
    return {"attacker": 1, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                     2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def touching(striker, target, t_bearing=270.0, params=P):
    """The striker (side 1, slot 0, facing east) and the target (side 2, its first slot) edge to edge: (state, H)."""
    st = scenario.build([army([(striker, 0, 0, 90)], [(target, 60, 0, t_bearing)])], params)
    H = st.N // 2
    front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
    reach = float(depth[0, H]) if t_bearing in (90.0, 270.0) else float(front[0, H])
    st.u["x"][0, H] = (float(depth[0, 0]) + reach) / 2
    st.u["order_kind"][0, 0], st.u["order_target"][0, 0] = O.ATTACK, H
    return st, H


def blow(striker, target, charge, t_bearing=270.0, params=P, moving=False, cd=False, out=0):
    """melee.strikes of the striker on the target with its charge left `charge` (0-1): out 0 the HP a second, 1 the
    HP a hit."""
    st, H = touching(striker, target, t_bearing, params)
    if moving:
        st.u["vx"][0, H] = 1.0
    if cd:
        st.u["cd"][0, H] = True
    pw = geometry.pairwise(st.u, params.sim["formation"]["spacing_m"])
    ch = torch.zeros_like(st.u["men"])
    ch[0, 0] = charge
    return float(melee.strikes(st.u, pw, pw["enemy"], params, ch)[out][0, 0, H])


class TestPassportsInTheState:
    def test_the_knights_are_large_and_spearmen_defend_against_them(self):
        st, H = touching(KNIGHTS, CSPEAR)
        assert bool(st.u["large"][0, 0]) and not bool(st.u["large"][0, H])
        assert float(st.u["hp_man"][0, 0]) == 108 and float(st.u["hp0"][0, 0]) == 6480
        assert bool(st.u["cdl"][0, H]) and bool(st.u["reflect"][0, H]) and not bool(st.u["cdl"][0, 0])
        assert not bool(st.u["cd"][0, H])          # charge_defense (against all): no unit of ours has it


class TestChargeDefence:
    def test_braced_spearmen_cancel_a_large_chargers_bonus_from_the_front(self):
        for target in (CSPEAR, SVHALB):
            assert blow(KNIGHTS, target, 1.0) == pytest.approx(blow(KNIGHTS, target, 0.0), rel=1e-6), target
            assert blow(KNIGHTS, target, 1.0, out=1) == pytest.approx(blow(KNIGHTS, target, 0.0, out=1), rel=1e-6)

    def test_only_braced_only_from_the_front_only_with_the_attribute(self):
        base = blow(KNIGHTS, CSPEAR, 0.0)
        assert blow(KNIGHTS, CSPEAR, 1.0, moving=True) > 1.2 * blow(KNIGHTS, CSPEAR, 0.0, moving=True)   # not braced
        # the knights on the spearmen's flank (the target faces north: the knights 90 deg off its front)
        assert blow(KNIGHTS, CSPEAR, 1.0, t_bearing=0.0) > 1.2 * blow(KNIGHTS, CSPEAR, 0.0, t_bearing=0.0)
        assert blow(KNIGHTS, CLANRATS, 1.0) > 1.2 * blow(KNIGHTS, CLANRATS, 0.0)           # no charge_defense_vs_large
        assert base > 0
        off = P.with_cal("contact", charge_defence={"on": False, "front_deg": 45})
        assert blow(KNIGHTS, CSPEAR, 1.0, params=off) > 1.2 * blow(KNIGHTS, CSPEAR, 0.0, params=off)

    def test_against_infantry_only_charge_defence_against_all(self):
        # an infantry charger keeps its bonus against charge_defense_vs_large; charge_defense cancels any charger's
        assert blow(SWORD, CSPEAR, 1.0) > 1.1 * blow(SWORD, CSPEAR, 0.0)
        assert blow(SWORD, CLANRATS, 1.0, cd=True) == pytest.approx(blow(SWORD, CLANRATS, 0.0, cd=True), rel=1e-6)
        assert blow(KNIGHTS, CLANRATS, 1.0, cd=True) == pytest.approx(blow(KNIGHTS, CLANRATS, 0.0, cd=True), rel=1e-6)


class TestSplashBySize:
    def test_a_demigryphs_blow_splashes_on_infantry_not_on_cavalry(self):
        on, off = P, P.with_cal("melee", splash_by_size=False)
        # on infantry (small, the weapon's splash_attack_target_size small): two men a blow, each half of it - unchanged
        assert blow(DEMI, CLANRATS, 0.0, out=1) == pytest.approx(blow(DEMI, CLANRATS, 0.0, params=off, out=1), rel=1e-6)
        assert blow(DEMI, CLANRATS, 0.0) == pytest.approx(blow(DEMI, CLANRATS, 0.0, params=off), rel=1e-6)
        # on knights (medium): one man, the whole blow
        st, H = touching(DEMI, KNIGHTS, params=on)
        st.u["splash"][0, 0] = 1.0
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        single = melee.strikes(st.u, pw, pw["enemy"], on, torch.zeros_like(st.u["men"]))
        assert blow(DEMI, KNIGHTS, 0.0, out=1) == pytest.approx(float(single[1][0, 0, H]), rel=1e-6)
        assert blow(DEMI, KNIGHTS, 0.0) == pytest.approx(float(single[0][0, 0, H]), rel=1e-6)
        assert blow(DEMI, KNIGHTS, 0.0, params=off, out=1) < 0.75 * blow(DEMI, KNIGHTS, 0.0, out=1)   # (halved before)

    def test_the_lords_splash_on_infantry_is_as_before(self):
        off = P.with_cal("melee", splash_by_size=False)
        assert blow(GENERAL, CLANRATS, 0.0) == pytest.approx(blow(GENERAL, CLANRATS, 0.0, params=off), rel=1e-6)
        st, _ = touching(GENERAL, CLANRATS)
        assert float(st.u["splash_cap"][0, 0]) == 2.0           # medium (the database)
        st, _ = touching(KNIGHTS, CLANRATS)
        assert float(st.u["splash_cap"][0, 0]) == 1.0           # none given: smaller than the medium rider


def front_swing(striker, target, params):
    """The striker's first-strike swing on the target as its charge lands (ran)."""
    st, H = touching(striker, target, params=params)
    pw = geometry.pairwise(st.u, params.sim["formation"]["spacing_m"])
    st.u["contact_s"] = torch.full_like(st.u["men"], 1e4)
    ran = torch.zeros_like(st.u["men"], dtype=torch.bool)
    ran[0, 0] = True
    sw = melee.strikes(st.u, pw, pw["enemy"], params, torch.zeros_like(st.u["men"]), first=True, ran=ran)[4]
    return float(sw[0, 0, H])


class TestFrontRankByTemplate:
    def test_the_knights_front_rank_is_their_files_not_their_bodies(self):
        old = P.with_cal("contact", first_strike_rank_template=False)
        allv = P.with_cal("contact", first_strike_rank_template="all")
        k = P.units[KNIGHTS]
        h = k["spacing"]["h"]
        body = h - 2 * k["radius_m"] * (1 - k["radii_ratio"])                 # 3.0 - 2 x 1.4 x 0.5 = 1.6 m
        assert body == pytest.approx(1.6)
        assert front_swing(KNIGHTS, CLANRATS, P) == pytest.approx(front_swing(KNIGHTS, CLANRATS, old) * body / h, rel=1e-4)
        assert front_swing(KNIGHTS, CLANRATS, allv) == pytest.approx(front_swing(KNIGHTS, CLANRATS, P), rel=1e-6)

    def test_small_bodies_unchanged_unless_all(self):
        old = P.with_cal("contact", first_strike_rank_template=False)
        allv = P.with_cal("contact", first_strike_rank_template="all")
        # infantry (circles): the same under every setting; the wolf rats (an ellipse, small) only under "all"
        assert front_swing(SWORD, CLANRATS, allv) == pytest.approx(front_swing(SWORD, CLANRATS, old), rel=1e-6)
        assert front_swing(WOLF, SPEAR, P) == front_swing(WOLF, SPEAR, old)
        w = P.units[WOLF]
        k = (w["spacing"]["h"] - 2 * w["radius_m"] * (1 - w["radii_ratio"])) / w["spacing"]["h"]   # 0.92 / 2.0
        assert front_swing(WOLF, SPEAR, allv) == pytest.approx(front_swing(WOLF, SPEAR, old) * k, rel=1e-4)
