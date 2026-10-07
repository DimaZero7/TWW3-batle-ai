"""The melee core by the game's rules (docs/en/game/mechanics/melee.md, docs/en/training/simulator.md): the damage of
a blow (armour roll, overkill, a lord's splash), the hit chance, the charge (attack order, run-up, its own clock,
reflection), the melee clock and the formation spacing from the database. Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, melee, missile, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR, SLAVE, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0", \
    "wh2_main_skv_inf_clanrat_spearmen_0"
SWORD, CLANRATS, SLINGER = "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1", \
    "wh2_main_skv_inf_skavenslave_slingers_0"
GENERAL, WARLORD = "wh_main_emp_cha_general_0", "wh2_main_skv_cha_warlord_0"
T = torch.tensor


def hit(base, ap, armour, hp, resist=0.0, single=False):
    return float(melee.per_hit(T(float(base)), T(float(ap)), T(float(armour)), T(float(hp)), T(float(resist)),
                               single=T(single)))


def army(side1, side2, attacker=1):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": len(r) > 4 and r[4]} for r in rows]
    return {"attacker": attacker, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                            2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


# --- the damage of a blow (build/damage/spec.md D1-D6) ---

class TestDamage:
    def test_a_lone_man_takes_the_mean(self):                     # T1, T2, T4, T11
        assert hit(20, 5, 0, 100, single=True) == pytest.approx(25)
        assert hit(20, 5, 100, 100, single=True) == pytest.approx(5 + 20 * 0.25)
        assert hit(20, 5, 0, 100, resist=0.2, single=True) == pytest.approx(20)
        assert hit(290, 140, 90, 4068, single=True) == pytest.approx(140 + 290 * (1 - 0.675))

    def test_armour_above_100(self):                              # T3
        cut = melee.armour_cut(T([150.0, 200.0, 250.0])).tolist()
        assert cut == pytest.approx([2 - 1 / 1.5 - 1.5 / 4, 1.0, 1.0], abs=1e-4)
        assert hit(20, 5, 150, 100, single=True) == pytest.approx(5 + 20 * (1 - (2 - 1 / 1.5 - 1.5 / 4)), rel=1e-4)

    def test_overkill(self):                                       # T5-T9
        assert hit(72.5, 35, 30, 69) == pytest.approx(69)           # always kills: a man's health
        assert hit(72.5, 35, 95, 76) == pytest.approx(38.0, abs=0.3)  # always two blows
        assert hit(14, 35, 10, 56, resist=0.2) == pytest.approx(28.0, abs=0.3)   # greatswords on night runners
        assert hit(17, 2, 30, 69) == pytest.approx(13.7, abs=0.2)   # an arrow on spearmen: many blows
        assert hit(19, 6, 0, 50) == pytest.approx(20.0, abs=0.05)   # the step smoothed: not 25 nor 16.7

    def test_monotone(self):                                       # T10
        dmg = torch.linspace(5, 120, 40)
        arm = torch.linspace(0, 150, 40)
        by_dmg = melee.per_hit(dmg, T(10.0), T(40.0), T(70.0))
        by_arm = melee.per_hit(T(40.0), T(10.0), arm, T(70.0))
        assert bool((by_dmg[1:] >= by_dmg[:-1] - 1e-4).all()) and bool((by_arm[1:] <= by_arm[:-1] + 1e-4).all())

    def test_kill_share_from_the_blows_to_kill(self):              # T14
        h = melee.per_hit(T(72.5), T(35.0), T(95.0), T(76.0))
        assert float(melee.kill_share(T(76.0), h, 0.5)) == pytest.approx(2 ** -0.5, rel=0.02)

    def test_a_lords_blow_strikes_two_men_a_share_each(self):    # T12, T13
        st = scenario.build([army([(GENERAL, 0, 0, 90, True)], [(SPEAR, 6, 0, 270), (WARLORD, 0, 3, 180, True)])], P)
        W = st.N // 2                                              # the Warlord (a general comes first)
        H = W + 1                                                  # the spearmen
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        pair = torch.zeros((1, st.N, st.N), dtype=torch.bool)
        pair[0, 0, H] = pair[0, H, 0] = True                       # the General and the spearmen only
        contact = pw["enemy"] & pair
        z = torch.zeros_like(st.u["men"])
        st.u["order_kind"][0, 0] = O.ATTACK
        st.u["order_target"][0, 0] = H
        rate, h, _, _ = melee.strikes(st.u, pw, contact, P, z)
        g, sp = P.units[GENERAL], P.units[SPEAR]
        p = (35 + g["melee"]["attack"] - sp["melee"]["defence"]) / 100
        share = melee.per_hit(T(g["melee"]["damage"] / 4), T(g["melee"]["ap_damage"] / 4), T(float(sp["armour"])),
                              T(float(sp["hp_per_man"])))
        assert float(h[0, 0, H]) == pytest.approx(float(share), rel=1e-4)
        per_s = p / (p * 4.0 + P.sim["melee"]["miss_s"])
        assert float(rate[0, 0, H]) == pytest.approx(P.sim["contact"]["lord_splash_struck"] * per_s * float(share),
                                                     rel=1e-4)
        # the other lord: the whole blow (no share, no overkill)
        assert float(h[0, 0, W]) == pytest.approx(g["melee"]["ap_damage"] + g["melee"]["damage"] * (1 - 0.75 * 0.9),
                                                      rel=1e-4)

    def test_a_missile_hit_by_the_same_rule(self):                 # T15
        st = scenario.build([army([("wh2_dlc13_emp_inf_archers_0", 0, 0, 90)], [(SPEAR, 100, 0, 270)])], P)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        target = torch.full_like(st.u["men"], -1).long()
        target[0, 0] = st.N // 2
        _, _, h = missile.volley(st.u, pw, target, 1.0, P)
        a, sp = P.units["wh2_dlc13_emp_inf_archers_0"]["missile"], P.units[SPEAR]
        want = hit(a["damage"], a["ap_damage"], sp["armour"], sp["hp_per_man"])
        assert float(h[0, 0, st.N // 2]) == pytest.approx(want, rel=1e-4)


# --- formation spacing from the database (unit_spacings) ---

class TestSpacing:
    def test_the_template_spacing(self):
        assert P.spacing_of(SPEAR) == (1.48, 1.6) and P.spacing_of(SLAVE) == (1.6, 1.8)
        st = scenario.build([{"attacker": 1, "sides": {
            1: {"faction": "wh_main_emp_empire", "units": [{"key": SPEAR, "x": 0, "z": 0, "b": 90, "width": 30}]},
            2: {"faction": "wh2_main_skv_skaven", "units": [{"key": SLAVE, "x": 99, "z": 0, "b": 270, "width": 30}]}}}],
            P)
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        H = st.N // 2
        # spearmen 120 at 30 m: 20 files of 1.48 m, 6 ranks of 1.6 m; slaves 180: 18 files of 1.6, 10 ranks of 1.8
        assert float(front[0, 0]) == pytest.approx(29.6) and float(depth[0, 0]) == pytest.approx(9.6)
        assert float(front[0, H]) == pytest.approx(28.8) and float(depth[0, H]) == pytest.approx(18.0)
        assert float(replay.half_depth(120, 30, P.spacing_of(SPEAR))) == pytest.approx(4.8)

    def test_men_striking_by_the_strikers_spacing(self):
        st = scenario.build([army([(SPEAR, -50, 0, 90)], [(CLANRAT, 50, 0, 270)])], P)
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        H = st.N // 2
        st.u["x"][0, 0], st.u["x"][0, H] = -float(depth[0, 0]) / 2, float(depth[0, H]) / 2
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        _, _, _, F = melee.strikes(st.u, pw, contact, P, torch.zeros_like(st.u["men"]))
        length = min(float(front[0, 0]), float(front[0, H]))
        ff = P.sim["melee"]["fighting_files"]
        assert float(F[0, 0, H]) == pytest.approx(ff * length / 1.48, rel=1e-4)
        assert float(F[0, H, 0]) == pytest.approx(ff * length / 1.6, rel=1e-4)


# --- the charge (build/charge/spec.md) and the melee clock (build/cyclecharge) ---

def run_in(attacker, defender, kind=O.ATTACK, run=True, gap=40.0, steps=80, params=P, until_contact=True):
    """attacker (side 2) comes at a standing defender (side 1) from gap m; returns (state, attacker slot)."""
    st = scenario.build([army([(defender, 0, 0, 90)], [(attacker, 60, 0, 270)])], params)
    H = st.N // 2
    front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
    st.u["x"][0, H] = float(depth[0, 0]) / 2 + float(depth[0, H]) / 2 + gap
    for _ in range(steps):
        o = replay.hold(st)
        o.kind[0, H], o.target[0, H], o.run[0, H] = kind, 0, run
        if kind == O.MOVE:
            o.x[0, H], o.z[0, H] = -40.0, 0.0
        battle.step(st, o, params)
        if until_contact and bool(st.u["m"][0, H]):
            return st, H
    return st, H


class TestCharge:
    def test_an_attack_order_charges_at_a_run_or_a_walk(self):    # 1, 2 (melee core 2: the sprint)
        st, H = run_in(SWORD, CLANRATS)
        assert float(st.u["charge"][0, H]) == pytest.approx(1 - 0.0, abs=0.05)
        st, H = run_in(SWORD, CLANRATS, kind=O.MOVE)
        assert float(st.u["charge"][0, H]) == 0.0
        st, H = run_in(SWORD, CLANRATS, run=False, steps=200)      # walks, sprints the last 30 m: a charge
        assert float(st.u["charge"][0, H]) == pytest.approx(1 - 0.0, abs=0.05)

    def test_a_run_up_is_needed(self):                            # 3
        st, H = run_in(SWORD, CLANRATS, gap=4.0)
        assert float(st.u["charge"][0, H]) == 0.0

    def test_the_charge_clock(self):                               # 4
        st, H = run_in(SWORD, CLANRATS)
        c0 = float(st.u["charge"][0, H])
        o = replay.hold(st)
        o.kind[0, H], o.target[0, H], o.run[0, H] = O.ATTACK, 0, True
        for _ in range(13):                                        # 6.5 s
            battle.step(st, o, P)
        assert float(st.u["charge"][0, H]) == pytest.approx(c0 - 0.5, abs=1e-4)

    def test_out_of_contact_the_clock_goes_on_and_the_fight_does_not_restart(self):
        st = scenario.build([army([(CLANRATS, 0, 0, 90)], [(SWORD, 30, 0, 270)])], P)
        H = st.N // 2
        st.u["charge"][0, H] = 1.0
        st.u["contact_s"][0, H] = 20.0
        st.u["x"][0, H] = 60.0                                     # out of contact
        o = replay.hold(st)
        for _ in range(4):                                         # 2 s out
            battle.step(st, o, P)
        assert float(st.u["charge"][0, H]) == pytest.approx(1 - 2 / 13, abs=1e-4)
        assert float(st.u["contact_s"][0, H]) == 20.0              # the clock waits (contact.reset_s)
        for _ in range(int(P.sim["contact"]["reset_s"] / 0.5)):
            battle.step(st, o, P)
        assert float(st.u["contact_s"][0, H]) == 0.0

    def test_no_ramp_the_receiver_strikes_in_full_at_once(self):  # 6
        st, H = run_in(SWORD, CLANRATS)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        now = melee.strikes(st.u, pw, contact, P, st.u["charge"])[0][0, 0, H]
        st.u["contact_s"] = torch.full_like(st.u["men"], 60.0)
        later = melee.strikes(st.u, pw, contact, P, st.u["charge"])[0][0, 0, H]
        assert float(now) > 0 and float(now) == pytest.approx(float(later), rel=1e-5)

    def test_the_charge_bonus_weighs_in_full(self):               # 9
        st = scenario.build([army([(SPEAR, 0, 0, 90)], [(GENERAL, 6, 0, 270, True)])], P)
        H = st.N // 2
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"]
        ch = torch.zeros_like(st.u["men"])
        ch[0, H] = 1.0
        st.u["order_kind"][0, H], st.u["order_target"][0, H] = O.ATTACK, 0
        g, sp = P.units[GENERAL]["melee"], P.units[SPEAR]["melee"]
        rate = float(melee.strikes(st.u, pw, contact, P, ch)[0][0, H, 0])
        p = min(90, 35 + g["attack"] + g["charge_bonus"] - sp["defence"]) / 100
        d, a, cb = g["damage"], g["ap_damage"], g["charge_bonus"]
        share = melee.per_hit(T((d + cb * d / (d + a)) / 4), T((a + cb * a / (d + a)) / 4),
                              T(float(P.units[SPEAR]["armour"])), T(float(P.units[SPEAR]["hp_per_man"])))
        want = P.sim["contact"]["lord_splash_struck"] * p / (p * 4.0 + P.sim["melee"]["miss_s"]) * float(share)
        assert rate == pytest.approx(want, rel=1e-4)

    def test_braced_spearmen_reflect_a_charge(self):              # 7
        def rate(defender, charge, moving=False, behind=False, per=False):
            st = scenario.build([army([(defender, 0, 0, 90)], [(CLANRATS, 30, 0, 270)])], P)
            H = st.N // 2
            front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
            st.u["x"][0, H] = (float(depth[0, 0]) + float(depth[0, H])) / 2
            if behind:
                st.u["b"][0, 0] = 270.0
            if moving:
                st.u["vx"][0, 0] = 1.0
            pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
            ch = torch.zeros_like(st.u["men"])
            ch[0, H] = charge
            out = melee.strikes(st.u, pw, pw["enemy"], P, ch)
            return float(out[1 if per else 0][0, 0, H])
        base = rate(SPEAR, 0.0)
        sp, cr = P.units[SPEAR]["melee"], P.units[CLANRATS]
        double = melee.per_hit(T(2.0 * sp["damage"]), T(2.0 * sp["ap_damage"]), T(float(cr["armour"])),
                               T(float(cr["hp_per_man"])))
        assert rate(SPEAR, 0.70, per=True) == pytest.approx(float(double), rel=1e-4)   # x2 damage, then overkill
        assert rate(SPEAR, 0.70) > 1.3 * base
        assert rate(SPEAR, 0.69) == pytest.approx(base, rel=1e-5)  # 13 x 0.3 = 3.9 s of the charge
        assert rate(SPEAR, 1.0, moving=True) == pytest.approx(base, rel=1e-5)
        assert rate(SPEAR, 1.0, behind=True) == pytest.approx(rate(SPEAR, 0.0, behind=True), rel=1e-5)
        assert rate(SLAVE, 1.0) == pytest.approx(rate(SLAVE, 0.0), rel=1e-5)   # no charge_reflection
        st, H = run_in(CLANRATS, SPEAR)
        assert float(st.u["charge"][0, 0]) == 0.0                  # the braced unit gets no charge of its own

    def test_the_charge_costs_fatigue_two_seconds(self):          # 8
        st, H = run_in(SWORD, CLANRATS)
        f0 = float(st.u["fatigue"][0, H])
        o = replay.hold(st)
        o.kind[0, H], o.target[0, H] = O.ATTACK, 0
        battle.step(st, o, P)
        battle.step(st, o, P)
        early = (float(st.u["fatigue"][0, H]) - f0) / 1.0          # the charge's first 2 s: +34 a tick
        battle.step(st, o, P)
        battle.step(st, o, P)
        f1 = float(st.u["fatigue"][0, H])
        for _ in range(6):
            battle.step(st, o, P)
        late = (float(st.u["fatigue"][0, H]) - f1) / 3.0           # then the melee's 13.7
        tick = P.sim["fatigue"]["calibration"]["per_second"]
        assert early == pytest.approx(34 * tick, rel=0.3) and late == pytest.approx(13.7 * tick, rel=1e-3)

    def test_an_attack_sprints_the_last_charge_distance(self):
        # The database's battle_entities: charge distance 30 m (lords 35) at the charge speed, at a run or a walk;
        # a move order gives none (the melee probe, build/meleetests).
        assert "rush_m" not in P.sim["charge"]
        sp = P.units[SWORD]["speed"]
        assert sp["charge_distance"] == 30 and P.units[GENERAL]["speed"]["charge_distance"] == 35
        for run in (True, False):
            st, H = run_in(SWORD, CLANRATS, gap=20.0, steps=6, run=run, until_contact=False)
            speed = float(torch.sqrt(st.u["vx"][0, H] ** 2 + st.u["vz"][0, H] ** 2))
            assert speed == pytest.approx(sp["charge"], abs=0.05), run
        st, H = run_in(SWORD, CLANRATS, gap=50.0, steps=4, run=False, until_contact=False)     # 50 m: walks
        assert float(torch.sqrt(st.u["vx"][0, H] ** 2 + st.u["vz"][0, H] ** 2)) == pytest.approx(sp["walk"], abs=0.05)
        st, H = run_in(SWORD, CLANRATS, kind=O.MOVE, gap=20.0, steps=6, until_contact=False)
        assert float(torch.sqrt(st.u["vx"][0, H] ** 2 + st.u["vz"][0, H] ** 2)) == pytest.approx(sp["run"], abs=0.05)


# --- melee core 2 (build/melee2impl): the first strike, the lord's gather, the initial recharge ---

def lost_per_step(attacker, defender, kind=O.ATTACK, run=True, gap=40.0, steps=4, params=P):
    """attacker (side 2) comes at a standing defender; HP each side loses on the contact step and the steps after."""
    st = scenario.build([army([(defender, 0, 0, 90)], [(attacker, 60, 0, 270)])], P)
    H = st.N // 2
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    st.u["x"][0, H] = float(depth[0, 0]) / 2 + float(depth[0, H]) / 2 + gap
    out = []
    for _ in range(400):
        o = replay.hold(st)
        o.kind[0, H], o.target[0, H], o.run[0, H] = kind, 0, run
        if kind == O.MOVE:
            o.x[0, H], o.z[0, H] = 3.0, 0.0          # into the enemy, not beyond contact.leave_m (it would leave)
        hp_d, hp_a = float(st.u["hp_abs"][0, 0]), float(st.u["hp_abs"][0, H])
        battle.step(st, o, params)
        if out or bool(st.u["m"][0, H]):
            out.append((hp_d - float(st.u["hp_abs"][0, 0]), hp_a - float(st.u["hp_abs"][0, H])))
        if len(out) > steps:
            break
    return out


class TestMeleeCore2:
    def test_one_swing_of_every_man_in_contact(self):
        # The first strike: F x struck x p x per hit once, i.e. the rate x (p x interval + miss_s) of a formation.
        st = scenario.build([army([(CLANRATS, 0, 0, 90)], [(SWORD, 10, 0, 270)])], P)
        H = st.N // 2
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        st.u["order_kind"][0, H], st.u["order_target"][0, H] = O.ATTACK, 0
        rate, _, _, F, swing = melee.strikes(st.u, pw, contact, P, z, first=True)
        u = P.units[SWORD]["melee"]
        p = (35 + u["attack"] - P.units[CLANRATS]["melee"]["defence"]) / 100
        assert float(F[0, H, 0]) > 1
        assert float(swing[0, H, 0]) == pytest.approx(float(rate[0, H, 0]) * (p * u["attack_interval_s"]
                                                                               + P.sim["melee"]["miss_s"]), rel=1e-4)
        # held (HOLD): its rate is cut by hold_rate, its one swing is not (every man in reach swings; spec R1)
        st.u["order_kind"][0, H], st.u["order_target"][0, H] = O.HOLD, -1
        rate_h, _, _, _, swing_h = melee.strikes(st.u, pw, contact, P, z, first=True)
        assert float(rate_h[0, H, 0]) == pytest.approx(P.sim["contact"]["hold_rate"] * float(rate[0, H, 0]), rel=1e-4)
        assert float(swing_h[0, H, 0]) == pytest.approx(float(swing[0, H, 0]), rel=1e-4)

    def test_a_unit_running_in_strikes_once_at_once_and_a_standing_one_facing_it_too(self):
        steps = lost_per_step(SWORD, CLANRATS)                       # a charge at a run
        first, then = steps[0], steps[2]
        assert first[0] > 4 * then[0]                                # the charger's burst on the contact step
        # the standing clanrats facing it strike once at once too (build/open_melee/spec.md R1)
        assert first[1] > 2 * then[1]
        off = lost_per_step(SWORD, CLANRATS, params=P.with_cal("contact", stand_first_strike=0))
        assert off[0][1] < 1.5 * off[2][1] + 1                       # (without the rule: at their rate)
        walk = lost_per_step(SWORD, CLANRATS, run=False)             # at a walk: sprints in, the same burst
        assert walk[0][0] == pytest.approx(first[0], rel=0.15)
        move = lost_per_step(SWORD, CLANRATS, kind=O.MOVE)           # a move order runs in: a burst, no charge
        assert then[0] < move[0][0] < first[0]

    def test_men_gather_round_a_lord_only_when_he_ran_in(self):
        # build/melee2/spec.md L4: infantry running onto a standing lord strike him in full at once; the formation a
        # lord runs into brings its men to bear over lord_gather_s.
        st = scenario.build([army([(GENERAL, 0, 0, 90, True)], [(SPEAR, 6, 0, 270)])], P)
        H = st.N // 2
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (torch.arange(st.N)[None, None, :] == 0)
        z = torch.zeros_like(st.u["men"])
        st.u["contact_s"] = torch.full_like(st.u["men"], 100.0)
        full = float(melee.strikes(st.u, pw, contact, P, z)[0][0, H, 0])
        st.u["contact_s"] = torch.full_like(st.u["men"], 2.0)
        standing = float(melee.strikes(st.u, pw, contact, P, z)[0][0, H, 0])
        st.u["ran_in"][0, 0] = True
        ran = float(melee.strikes(st.u, pw, contact, P, z)[0][0, H, 0])
        assert full > 0 and standing == pytest.approx(full, rel=1e-5)
        assert ran == pytest.approx(full * 2.0 / P.sim["contact"]["lord_gather_s"], rel=1e-4)

    def test_a_lord_that_charges_ran_in(self):
        st, H = run_in(WARLORD, SPEAR)
        assert bool(st.u["ran_in"][0, H]) and not bool(st.u["ran_in"][0, 0])

    def test_initial_recharge_from_the_passport(self):
        # unit_special_abilities.initial_recharge: the lords' actives ready at once, Strength of the Penitent in 3 s.
        flag = "wh_dlc04_emp_inf_flagellants_0"
        st = scenario.build([army([(GENERAL, 0, 0, 90, True), (flag, 0, 20, 90)], [(CLANRATS, 30, 0, 270)])], P)
        assert all(float(st.u[f"ab{k}_cd"][0, 0]) == 0 for k in range(3))
        cds = [float(st.u[f"fxt{j}_cd"][0, 1]) for j in range(2)]
        assert 3.0 in cds
        o = replay.hold(st)
        battle.step(st, o, P)
        assert max(float(st.u[f"fxt{j}_cd"][0, 1]) for j in range(2)) == pytest.approx(3.0 - P.dt)
