"""tools.nn.sim: the battle simulator (docs/en/training/simulator.md). Needs torch: skipped in the
project's .venv; run in the container (bash tools/nn/dock.sh pytest ... or see the page).

Level 1: small functions. Level 2: a module on made-up cases, checking properties (more armour ->
less damage, a flank -> faster rout...). Level 3: the battle loop wired together."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import abilities, battle, effects, fatigue, geometry, melee, missile, morale  # noqa: E402
from tools.nn.sim import movement, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim import state as S  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR, SLAVE, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0", \
    "wh2_main_skv_inf_clanrat_spearmen_0"
ARCHER, SLINGER = "wh2_dlc13_emp_inf_archers_0", "wh2_main_skv_inf_skavenslave_slingers_0"
GENERAL = "wh_main_emp_cha_general_0"
SHIELD_SPEAR, SWORD = "wh_main_emp_inf_spearmen_1", "wh_main_emp_inf_swordsmen"


def army(side1, side2, attacker=1, factions=("wh_main_emp_empire", "wh2_main_skv_skaven")):
    """side1/side2: lists of (key, x, z, bearing[, general])."""
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": len(r) > 4 and r[4]} for r in rows]
    return {"attacker": attacker, "sides": {1: {"faction": factions[0], "units": units(side1)},
                                            2: {"faction": factions[1], "units": units(side2)}}}


# Formations touch when their edges overlap by -contact.reach_m (the game's formations step into each other).
TOUCH = min(0.0, P.sim["contact"]["reach_m"])


def face_off(key1, key2, gap=0.0, per_side=None):
    """Two units facing each other along x, their fronts `gap` m apart (0: in contact, overlapping by TOUCH)."""
    st = scenario.build([army([(key1, -50, 0, 90)], [(key2, 50, 0, 270)])], P, per_side=per_side)
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    gap = gap + TOUCH
    st.u["x"][0, 0] = -(depth[0, 0] / 2 + gap / 2)
    H = st.N // 2
    st.u["x"][0, H] = depth[0, H] / 2 + gap / 2
    return st


# --- level 1: functions ---

class TestFunctions:
    def test_hit_chance_is_the_database_rule_at_slope_one(self):
        a, d = torch.tensor([20.0, 55.0, 0.0]), torch.tensor([34.0, 0.0, 90.0])
        assert melee.hit_chance(a, d, 1.0).tolist() == pytest.approx([0.21, 0.9, 0.08])
        assert melee.hit_chance(a, d, 0.0).tolist() == pytest.approx([0.35] * 3)

    def test_armour_stops_base_damage_not_armour_piercing(self):
        hp = torch.tensor(100.0)
        lone = torch.tensor(True)
        bare = melee.per_hit(torch.tensor(20.0), torch.tensor(5.0), torch.tensor(0.0), hp, single=lone)
        armoured = melee.per_hit(torch.tensor(20.0), torch.tensor(5.0), torch.tensor(100.0), hp, single=lone)
        assert float(bare) == pytest.approx(25) and float(armoured) == pytest.approx(5 + 20 * 0.25)
        assert float(melee.per_hit(torch.tensor(400.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(60.0))) == 60

    def test_a_killing_blow_kills_a_man_per_man_of_health(self):
        assert float(melee.kill_share(torch.tensor(50.0), torch.tensor(60.0), 0.5)) == 1.0
        assert float(melee.kill_share(torch.tensor(60.0), torch.tensor(15.0), 0.5)) == pytest.approx(0.5)

    def test_morale_tables_are_steps(self):
        points = morale.table(P.morale, "total_casualties_penalty_", (10, 20, 30, 40, 50, 60, 70, 80, 90))
        x = torch.tensor([0.05, 0.1, 0.55, 0.95])
        assert morale.steps(x, points).tolist() == [0.0, -2.0, -16.0, -74.0]

    def test_losing_thresholds_of_their_own_leave_no_even_band(self):
        dealt = torch.tensor([99.0, 59.0, 31.0, 38.0, 100.0, 300.0])     # +1 each: ratios 1, 0.6, 0.32, 0.39, ~1.01, ~3
        taken = torch.full_like(dealt, 99.0)
        on = torch.ones_like(dealt, dtype=torch.bool)
        old = {"combat_ratio": {"slightly": 1.5, "yes": 2.5, "significantly": 4}}
        new = {"combat_ratio": dict(old["combat_ratio"], losing=1.0, losing_significantly=2.5)}
        assert morale.combat_points(dealt, taken, on, old, P.morale).tolist() == [0, -3, -3, -3, 0, 6]
        assert morale.combat_points(dealt, taken, on, new, P.morale).tolist() == [0, -3, -8, -8, 0, 6]
        # on by default (config/nn/sim.json morale.combat_why: Goumin's rule, the probe's 0.32 -> -8)
        assert P.sim["morale"]["combat_ratio"].get("losing") == 1.0
        assert P.sim["morale"]["combat_ratio"].get("losing_significantly") == 2.5

    def test_distance_factor_interpolates_and_holds_at_the_ends(self):
        table = P.sim["missile"]["distance_factor"]
        x = torch.tensor([10.0, table[0][0], (table[0][0] + table[1][0]) / 2, 200.0])
        f = missile.distance_factor(x, table)
        assert f.tolist() == pytest.approx([table[0][1], table[0][1], (table[0][1] + table[1][1]) / 2, table[-1][1]])

    def test_extent_and_sectors(self):
        front, depth = torch.tensor(30.0), torch.tensor(9.0)
        assert float(geometry.half_extent(front, depth, torch.tensor(0.0))) == pytest.approx(4.5)
        assert float(geometry.half_extent(front, depth, torch.tensor(math.pi / 2))) == pytest.approx(15)
        assert float(geometry.face_length(front, depth, torch.tensor(0.0))) == 30
        assert float(geometry.face_length(front, depth, torch.tensor(math.pi / 2))) == 9
        rel = torch.tensor([0.0, math.pi / 2, math.pi])
        assert geometry.sector(rel, 60, 120).tolist() == [0, 1, 2]

    def test_fatigue_states_by_the_database_thresholds(self):
        u = {"fatigue": torch.tensor([0.0, 2799.0, 2800.0, 27000.0]), "fat": torch.zeros(4)}
        idle = {k: torch.zeros(4, dtype=torch.bool) for k in ("melee", "charging", "shooting", "running", "walking")}
        fatigue.step(u, idle, P, 0.0)
        assert u["fat"].tolist() == [0, 0, 1, 5]

    def test_an_exhausted_unit_fights_and_moves_worse(self):
        # unit_fatigue_effects_tables (sim.json fatigue.effects): exhausted attack x0.7, speed x0.85,
        # reload / 0.9; fresh x1; the old values come back.
        st = face_off(SPEAR, SLAVE)
        st.u["fat"][0, 0] = 5.0
        before = {k: st.u[k].clone() for k in ("attack", "run", "reload", "defence")}
        old = fatigue.effects(st.u, P)
        assert float(st.u["attack"][0, 0]) == pytest.approx(0.7 * float(before["attack"][0, 0]))
        assert float(st.u["run"][0, 0]) == pytest.approx(0.85 * float(before["run"][0, 0]))
        assert float(st.u["attack"][0, 1]) == float(before["attack"][0, 1])
        st.u.update(old)
        assert all(torch.equal(st.u[k], v) for k, v in before.items())

    @pytest.mark.parametrize("name,points", [("idle", -18), ("ready", -7), ("walking", -1),
                                              ("running", 4), ("shooting", 7.5), ("melee", 13.7),
                                              ("charging", 34)])
    def test_fatigue_tiring_and_recovery_use_calibrated_ticks(self, name, points):
        # Database points at 10 ticks/s; shooting and a formation's melee are fitted.
        u = {"fatigue": torch.tensor([15000.0]), "fat": torch.zeros(1)}
        activity = {k: torch.tensor([k == name]) for k in
                    ("idle", "walking", "running", "shooting", "melee", "charging")}
        fatigue.step(u, activity, calibrated_fatigue(), 2.0)
        assert float(u["fatigue"][0]) == pytest.approx(15000 + points * 20)

    def test_fatigue_priority_clamping_immunity_and_inactive_slots(self):
        u = {"fatigue": torch.tensor([29900., 100., 15000., 15000.]), "fat": torch.zeros(4),
             "fatigue_immune": torch.tensor([False, False, True, False])}
        activity = {k: torch.tensor([True, False, True, True]) for k in
                    ("walking", "running", "shooting", "melee", "charging")}
        activity["idle"] = torch.ones(4, dtype=torch.bool)
        activity["active"] = torch.tensor([True, True, True, False])
        fatigue.step(u, activity, calibrated_fatigue(), 1.)
        assert u["fatigue"].tolist() == [30000., 0., 15000., 15000.]
        activity["charging"].zero_()
        activity["melee"].zero_()
        activity["shooting"].zero_()
        activity["running"].zero_()
        activity["walking"].zero_()
        fatigue.step(u, activity, calibrated_fatigue(), 1.)
        assert u["fatigue"].tolist() == [29820., 0., 14820., 15000.]

    def test_idle_recovery_crosses_all_state_boundaries(self):
        u = {"fatigue": torch.tensor([27000., 18000., 12600., 6600., 2800.]), "fat": torch.zeros(5)}
        activity = {k: torch.zeros(5, dtype=torch.bool) for k in
                    ("walking", "running", "shooting", "melee", "charging")}
        activity["idle"] = torch.ones(5, dtype=torch.bool)
        fatigue.step(u, activity, calibrated_fatigue(), 1.)
        assert u["fat"].tolist() == [4., 3., 2., 1., 0.]

    def test_state_and_orders_shapes_and_perspective(self):
        st = S.empty(2, 6, history=3)
        assert all(v.shape == (2, 6) for k, v in st.u.items() if k not in S.HISTORY)
        assert st.u["lost_hist"].shape == (2, 18)
        obs = st.observation()
        timers = {f"ab{k}_{t}" for k in range(3) for t in ("on", "cd")}
        order = {"order_kind", "order_target"}         # the order in force (observation ORDER)
        assert set(obs) == set(S.OBSERVED) | {"side", "t", "fx_on", "gone"} | timers | order and obs["t"].shape == (2,)
        x = torch.arange(6).repeat(2, 1)
        assert S.own_first(x, 2)[0].tolist() == [3, 4, 5, 0, 1, 2]
        assert S.slot_from_own_first(torch.tensor([0, 4, -1]), 2, 6).tolist() == [3, 1, -1]
        o = O.hold(2, 6)
        O.check(o, 6)
        o.kind[0, 0] = O.ATTACK
        with pytest.raises(ValueError):
            O.check(o, 6)


class TestScenario:
    def test_an_arena_can_be_given_as_a_dict(self):
        import json
        from tools.nn import scenario as arena_scenario
        entry = json.loads(arena_scenario.ARENAS.read_text(encoding="utf-8"))["arenas"]["pair_spear_v_slave"]
        by_name = scenario.from_arena("pair_spear_v_slave", "attack")
        by_dict = scenario.from_arena(entry, "attack")
        assert by_dict == by_name
        loaded = arena_scenario.load_arena("whole_emp_v_skv")
        assert scenario.from_arena(loaded, "defend") == scenario.from_arena("whole_emp_v_skv", "defend")
        custom = dict(entry, gap_m=300)
        far = scenario.from_arena(custom)
        assert far["sides"][2]["units"][0]["x"] - far["sides"][1]["units"][0]["x"] >             by_name["sides"][2]["units"][0]["x"] - by_name["sides"][1]["units"][0]["x"]
        with pytest.raises(ValueError):
            scenario.from_arena({"gap_m": 100})


# --- level 2: modules on cases ---

class TestMelee:
    def rate(self, st, i=0, j=None):
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        ch = torch.zeros_like(st.u["men"])
        st.u["contact_s"] = torch.full_like(st.u["men"], 100.0)
        rate, _, sector, _ = melee.strikes(st.u, pw, contact, P, ch)
        j = st.N // 2 if j is None else j
        return float(rate[0, i, j]), int(sector[0, i, j])

    def test_two_fronts_that_touch_fight(self):
        st = face_off(SPEAR, SLAVE)
        assert self.rate(st)[0] > 0 and self.rate(st, st.N // 2, 0)[0] > 0

    def test_more_armour_takes_less(self):
        st = face_off(SPEAR, SLAVE)
        base = self.rate(st)[0]
        st.u["armour"][0, st.N // 2] = 80.0
        assert self.rate(st)[0] < base

    def test_a_thinned_formation_keeps_its_width_and_fewer_of_its_men_reach(self):
        """melee.front_fill_ranks (1.5): a formation at half its men keeps its ordered width (its files do not shrink)
        and its men striking are scaled by the fill of that front, (1 - e^(-r/1.5)) / (1 - e^(-r0/1.5)), r = men a
        file; a whole unit is unchanged against the rule off (0), and so is a lone man (the dmgmelee probe: a unit at
        30 % struck 1.6-1.9x the game before)."""
        lam = P.sim["melee"]["front_fill_ranks"]
        assert lam == 1.5
        off = P.with_cal("melee", front_fill_ranks=0)
        st = face_off(SPEAR, SLAVE)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        st.u["contact_s"] = torch.full_like(st.u["men"], 100.0)
        whole_on = float(melee.strikes(st.u, pw, contact, P, z)[3][0, 0, 1])
        whole_off = float(melee.strikes(st.u, pw, contact, off, z)[3][0, 0, 1])
        assert whole_on == pytest.approx(whole_off, rel=1e-4) and whole_on > 0
        files0 = math.floor(float(st.u["width"][0, 0]) / float(st.u["sp_h"][0, 0]) + 1e-4)
        men0 = float(st.u["men0"][0, 0])
        st.u["men"][0, 0] = men0 / 2                         # half the men, the width ordered the same
        half_on = float(melee.strikes(st.u, pw, contact, P, z)[3][0, 0, 1])
        half_off = float(melee.strikes(st.u, pw, contact, off, z)[3][0, 0, 1])
        fill = (1 - math.exp(-(men0 / 2 / files0) / lam)) / (1 - math.exp(-(men0 / files0) / lam))
        assert 0.8 < fill < 0.95
        assert half_on == pytest.approx(half_off * fill, rel=1e-3)
        assert half_off == pytest.approx(whole_off, rel=1e-3)   # the old rule: the half unit struck as the whole one
        # a lone man (the General) strikes his one blow whatever the rule
        st2 = face_off(GENERAL, SLAVE) if "GENERAL" in globals() else None
        if st2 is not None:
            pw2 = geometry.pairwise(st2.u, P.sim["formation"]["spacing_m"])
            c2 = pw2["enemy"] & (pw2["gap"] <= 1.0)
            st2.u["contact_s"] = torch.full_like(st2.u["men"], 100.0)
            assert float(melee.strikes(st2.u, pw2, c2, P, z)[3][0, 0, 1]) == pytest.approx(
                float(melee.strikes(st2.u, pw2, c2, off, z)[3][0, 0, 1]), rel=1e-4)

    def test_a_rear_attack_hits_more(self):
        p = P
        st = face_off(SPEAR, SLAVE)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        front = melee.strikes(st.u, pw, contact, p, z)[0][0, 0, 1]
        st.u["b"][0, 1] = 90.0            # the slaves turn their back to the spearmen
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        rate, _, sector, _ = melee.strikes(st.u, pw, contact, p, z)
        assert int(sector[0, 0, 1]) == 2 and float(rate[0, 0, 1]) > float(front)

    def test_a_flank_attacker_strikes_with_its_own_front(self):
        # melee.flank_face "striker": through the target's flank the striker brings men by its own
        # front, not by the target's short depth ("min": the old rule).
        st = face_off(SPEAR, SLAVE)
        H = st.N // 2
        st.u["b"][0, H] = 0.0             # the slaves turn their flank to the spearmen
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        st.u["x"][0, H] = front[0, H] / 2
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        st.u["contact_s"][:] = 300.0          # a long fight: past the opening wave (melee.wave)
        F = {}
        for mode in ("min", "striker"):
            _, _, sector, f = melee.strikes(st.u, pw, contact, P.with_cal("melee", flank_face=mode), z)
            assert int(sector[0, 0, H]) == 1
            F[mode] = float(f[0, 0, H])
        ff, sp = P.sim["melee"]["fighting_files"], P.spacing_of(SPEAR)[0]
        assert F["min"] == pytest.approx(ff * float(depth[0, H]) / sp, rel=1e-4)
        assert F["striker"] == pytest.approx(ff * float(front[0, 0]) / sp, rel=1e-4)
        assert F["striker"] > F["min"]
        # on by default (config/nn/sim.json melee.flank_face_why: the defender probe - a unit turning its flank to its
        # attacker to leave or to attack another keeps being struck by the attacker's whole front)
        assert P.sim["melee"]["flank_face"] == "striker"

    def test_the_opening_wave_brings_more_men_early_and_fades_with_the_pair_s_time_in_melee(self):
        """melee.wave (config/nn/sim.json wave_why: the men within the database's reach early): men striking
        x (1 + amp exp(-t / tau_s)), t the younger of the two units' contact clocks; off (None): the old rule."""
        st = face_off(SPEAR, SLAVE)
        H = st.N // 2
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        wave = P.sim["melee"]["wave"]
        def F(t_own, t_enemy, params=P):
            st.u["contact_s"][0, 0], st.u["contact_s"][0, H] = t_own, t_enemy
            return float(melee.strikes(st.u, pw, contact, params, z)[3][0, 0, H])
        late = F(300.0, 300.0)
        assert F(0.0, 300.0) == pytest.approx(late * (1 + wave["amp"]), rel=1e-4)        # the younger clock
        assert F(wave["tau_s"], 300.0) == pytest.approx(late * (1 + wave["amp"] / math.e), rel=1e-4)
        assert F(0.0, 0.0, P.with_cal("melee", wave=None)) == pytest.approx(late, rel=1e-3)

    def test_the_men_striking_a_missile_unit_under_an_attack_order_keep_a_floor_of_the_wave(self):
        """melee.wave missile_attack_floor (the wavemiss probes): the enemy's men striking a missile unit that fights
        under an attack order stay at least floor x the steady level; unordered (HOLD) the wave fades as for formations."""
        st = face_off(SLAVE, ARCHER)
        H = st.N // 2
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        floor = P.sim["melee"]["wave"]["missile_attack_floor"]
        assert floor > 1
        st.u["contact_s"][:] = 300.0                                     # past the opening wave
        def F(kind):
            st.u["order_kind"][0, H] = kind
            st.u["order_target"][0, H] = 0 if kind == O.ATTACK else -1
            return float(melee.strikes(st.u, pw, contact, P, z)[3][0, 0, H])
        held, attacking = F(O.HOLD), F(O.ATTACK)
        assert attacking == pytest.approx(floor * held, rel=1e-4)
        st.u["contact_s"][:] = 0.0                                       # at contact the wave is above the floor
        assert F(O.ATTACK) == pytest.approx(F(O.HOLD), rel=1e-4)

    def test_at_most_the_cap_reach_a_lord(self):
        st = face_off(SPEAR, GENERAL)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        st.u["order_kind"][0, 0] = O.ATTACK
        st.u["order_target"][0, 0] = st.N // 2   # attacking the lord: the whole rate
        st.u["contact_s"] = torch.full_like(st.u["men"], 100.0)
        rate, hit, _, _ = melee.strikes(st.u, pw, contact, P, z)
        p = float(melee.hit_chance(torch.tensor(20.0), torch.tensor(45.0)))
        cap = P.sim["contact"]["lord_max_attackers"]
        per_s = p / (p * 5.7 + P.sim["melee"]["miss_s"])
        assert float(rate[0, 0, 1]) == pytest.approx(cap * per_s * float(hit[0, 0, 1]), rel=1e-4)
        # men gather round him over contact.lord_gather_s when he ran into them: half way, half the rate; standing
        # (they ran onto him), the whole rate at once
        st.u["contact_s"] = torch.full_like(st.u["men"], P.sim["contact"]["lord_gather_s"] / 2)
        standing = float(melee.strikes(st.u, pw, contact, P, z)[0][0, 0, 1])
        assert standing == pytest.approx(float(rate[0, 0, 1]), rel=1e-4)
        st.u["ran_in"][0, 1] = True
        half = float(melee.strikes(st.u, pw, contact, P, z)[0][0, 0, 1])
        assert half == pytest.approx(float(rate[0, 0, 1]) / 2, rel=1e-4)

    @staticmethod
    def surrounded(n, rival=False):
        """A General at the centre, n spear units touching him from front, back, left, right
        (and the Warlord on him too); every enemy in contact with him."""
        places = [(0, 6, 180), (0, -6, 0), (-6, 0, 90), (6, 0, 270)][:n]
        enemies = [(SPEAR, x, z, b) for x, z, b in places]
        if rival:
            enemies = [("wh2_main_skv_cha_warlord_0", 0, 1, 180, True)] + enemies
        st = scenario.build([army([(GENERAL, 0, 0, 0, True)], enemies, factions=("wh2_main_skv_skaven",
                                                                                 "wh_main_emp_empire"))], P)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (torch.arange(st.N)[None, None, :] == 0)
        z = torch.zeros_like(st.u["men"])
        st.u["contact_s"] = torch.full_like(st.u["men"], 100.0)     # the steady fight (contact.lord_gather_s)
        return st, pw, contact, z

    def test_more_units_round_a_lord_share_the_same_men(self):
        # The lord swarm probe: 1-4 spear units take from a lord the same HP/s.
        lost = []
        for n in (1, 2, 4):
            st, pw, contact, z = self.surrounded(n)
            rate, _, _, F = melee.strikes(st.u, pw, contact, P, z)
            lost.append(float(rate[0, :, 0].sum()))
            assert float(F[0, :, 0].sum()) == pytest.approx(P.sim["contact"]["lord_max_attackers"], rel=1e-4)
        assert lost[1] == pytest.approx(lost[0], rel=0.01) and lost[2] == pytest.approx(lost[0], rel=0.01)

    def test_a_unit_attacking_another_enemy_barely_strikes_a_lord_it_touches(self):
        st, pw, contact, z = self.surrounded(2)
        H = st.N // 2
        st.u["order_kind"][0, H] = O.ATTACK
        st.u["order_target"][0, H] = 0          # attacking the lord: the whole rate
        full = float(melee.strikes(st.u, pw, contact, P, z)[0][0, H, 0])
        st.u["order_target"][0, H] = 1          # told to attack another slot, touching the lord
        busy = float(melee.strikes(st.u, pw, contact, P, z)[0][0, H, 0])
        assert busy == pytest.approx(full * P.sim["contact"]["lord_incidental"], rel=1e-4)

    @pytest.mark.parametrize("share", [0.3, 0.6, 1.0])
    def test_a_unit_attacking_another_enemy_strikes_a_unit_it_touches_at_its_share(self, share):
        params = P.with_cal("contact", unit_incidental=share)
        st = face_off(SPEAR, SLAVE)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        H = st.N // 2
        st.u["order_kind"][0, 0] = O.ATTACK
        st.u["order_target"][0, 0] = H         # its own target: the whole rate
        full = float(melee.strikes(st.u, pw, contact, params, z)[0][0, 0, H])
        st.u["order_target"][0, 0] = H + 1     # told to attack another slot, touching the slaves
        st.u["order_s"][:] = 100.0             # a settled order (given before the fight or over 24 s ago)
        busy = float(melee.strikes(st.u, pw, contact, params, z)[0][0, 0, H])
        assert busy == pytest.approx(full * share, rel=1e-4)
        st.u["order_s"][:] = 3.0               # given in melee 3 s ago: contact.fresh_incidental within the window
        fresh = float(melee.strikes(st.u, pw, contact, params, z)[0][0, 0, H])
        assert fresh == pytest.approx(full * P.sim["contact"]["fresh_incidental"], abs=1e-6)

    def test_a_formation_or_missile_unit_holding_in_melee_strikes_at_the_hold_rate_a_lord_in_full(self):
        k = P.sim["contact"]["hold_rate"]
        assert 0 < k < 1
        missile = k if P.sim["contact"].get("hold_missile") else 1.0      # contact.hold_missile (the wavemiss probes)
        assert P.sim["contact"].get("hold_missile")
        for key, share in ((SPEAR, k), (ARCHER, missile), ("wh_main_emp_cha_general_0", 1.0)):
            st = face_off(key, SLAVE)
            pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
            contact = pw["enemy"] & (pw["gap"] <= 1.0)
            z = torch.zeros_like(st.u["men"])
            H = st.N // 2
            st.u["order_kind"][0, 0] = O.ATTACK
            st.u["order_target"][0, 0] = H
            attacking = float(melee.strikes(st.u, pw, contact, P, z)[0][0, 0, H])
            st.u["order_kind"][0, 0] = O.HOLD
            st.u["order_target"][0, 0] = -1
            held = float(melee.strikes(st.u, pw, contact, P, z)[0][0, 0, H])
            assert attacking > 0 and held == pytest.approx(share * attacking, rel=1e-5), key

    def test_the_enemy_lord_keeps_his_blow_and_the_infantry_counts_less(self):
        st, pw, contact, z = self.surrounded(0, rival=True)
        alone = float(melee.strikes(st.u, pw, contact, P, z)[0][0, :, 0].sum())
        st, pw, contact, z = self.surrounded(3, rival=True)
        rate = melee.strikes(st.u, pw, contact, P, z)[0][0, :, 0]
        H = st.N // 2
        assert float(rate[H]) == pytest.approx(alone, rel=1e-4)
        st3, pw3, c3, z3 = self.surrounded(3)
        infantry = float(melee.strikes(st3.u, pw3, c3, P, z3)[0][0, :, 0].sum())
        k = P.sim["contact"]["lord_rival_others"]
        # The rival takes one of the cap's places: the infantry share the rest.
        cap = P.sim["contact"]["lord_max_attackers"]
        assert float(rate.sum()) - alone == pytest.approx(k * infantry * (cap - 1) / cap, rel=1e-3)

    def test_a_lord_strikes_a_lord_by_the_database_hit_chance(self):
        # Lord against lord: hit = 35 + attack - defence (the Warlord's 50 against the General's 45: 40 %), a whole
        # blow 120 AP + 280 x (1 - 0.75 x 85 %) every 4 s (no splash share, no overkill: one pool of health).
        # The infantry on him: the same hit chance rule, p / (p x interval + miss_s) blows a second.
        st, pw, contact, z = self.surrounded(1, rival=True)
        H = st.N // 2
        st.u["order_kind"][0, H:H + 2] = O.ATTACK               # attacking him
        st.u["order_target"][0, H:H + 2] = 0
        rate, hit, _, F = melee.strikes(st.u, pw, contact, P, z)
        assert float(rate[0, H, 0]) == pytest.approx(0.40 * (120 + 280 * (1 - 0.75 * 0.85)) / 4.0, rel=1e-4)
        p = 0.35 + (20 - 45) / 100
        per_s = p / (p * 5.7 + P.sim["melee"]["miss_s"])
        k = P.sim["contact"]["lord_rival_others"]
        assert float(rate[0, H + 1, 0]) == pytest.approx(k * float(F[0, H + 1, 0]) * per_s * float(hit[0, H + 1, 0]),
                                                         rel=1e-4)

    def test_swordsmen_out_strike_spearmen_against_clanrats(self):
        # Same men, armour and shield class; the sword: attack 32 (spear 20), 21 + 7 damage, a blow every 4.3 s
        # (the plain spear 5.7 s); no bonus against infantry in the database (bonus_v_infantry 0).
        assert P.units[SWORD]["melee"]["bonus_v_infantry"] == 0
        rates = {}
        for key in (SPEAR, SWORD):
            st = face_off(key, CLANRAT)
            pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
            contact = pw["enemy"] & (pw["gap"] <= 1.0)
            z = torch.zeros_like(st.u["men"])
            rates[key] = float(melee.strikes(st.u, pw, contact, P, z)[0][0, 0, 1])
        assert rates[SWORD] > 1.4 * rates[SPEAR] > 0

    def test_a_charge_adds_only_its_bonus(self):
        # CA: +charge bonus to attack and to damage (split by the weapon's AP share), no impact multiplier.
        st = face_off(SWORD, CLANRAT)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        H = st.N // 2
        steady = float(melee.strikes(st.u, pw, contact, P, z)[0][0, 0, H])
        full = float(melee.strikes(st.u, pw, contact, P, z + 1.0)[0][0, 0, H])
        sw, cr = P.units[SWORD], P.units[CLANRAT]
        cb, md = sw["melee"]["charge_bonus"], cr["melee"]["defence"]

        def rule(bonus):
            p = (35 + sw["melee"]["attack"] + bonus - md) / 100
            d, a = sw["melee"]["damage"], sw["melee"]["ap_damage"]
            hit = melee.per_hit(torch.tensor(d + bonus * d / (d + a)), torch.tensor(a + bonus * a / (d + a)),
                                torch.tensor(float(cr["armour"])), torch.tensor(float(cr["hp_per_man"])),
                                torch.tensor(float(cr["damage_resist"]["physical"]) / 100))
            return p / (p * sw["melee"]["attack_interval_s"] + P.sim["melee"]["miss_s"]) * float(hit)
        assert full / steady == pytest.approx(rule(cb) / rule(0), rel=1e-4)
        assert 1.3 < full / steady < 2.5


class TestMissile:
    OLD = P.with_cal("missile", spill_geometry=None)

    @staticmethod
    def _spill(dx, dz, key=SLAVE, shooter=ARCHER, gap=100.0, params=None):
        """(neighbour's HP / target's HP, target's HP) of one volley: the shooter at x = -gap shoots a formation at
        the origin; another formation of the target's side stands at (dx, dz), both facing the shooter."""
        params = params or P
        st = scenario.build([army([(shooter, -gap, 0, 90)], [(key, 0, 0, 270), (key, dx, dz, 270)])], params)
        H = st.N // 2
        pw = geometry.pairwise(st.u, params.sim["formation"]["spacing_m"])
        target = torch.full((1, st.N), -1)
        target[0, 0] = H
        _, hp, _ = missile.volley(st.u, pw, target, 1.0, params)
        return float(hp[0, 0, H + 1]) / float(hp[0, 0, H]), float(hp[0, 0, H])

    def test_the_spill_comes_from_the_spread_a_pile_shares_the_shots(self):
        """missile.spill_geometry: a neighbour standing in the target's men (4 m behind its centre) takes about as
        much as the target, one 18 m aside a part, one 60 m aside nothing; the target keeps less in a pile; a flat
        shot (handgun) reaches a neighbour behind the target farther than an arrow; off: the old table."""
        pile, hj_pile = self._spill(4.0, 0.0)
        aside, _ = self._spill(0.0, 18.0)
        far, hj_alone = self._spill(0.0, 60.0)
        assert 0.6 < pile < 1.1 and 0.1 < aside < 0.6 and far < 0.01
        assert hj_pile < 0.85 * hj_alone
        arrow_behind, _ = self._spill(30.0, 0.0)
        bullet_behind, _ = self._spill(30.0, 0.0, shooter="wh_main_emp_inf_handgunners")
        assert bullet_behind > arrow_behind
        old, _ = self._spill(0.0, 18.0, params=self.OLD)
        assert 0 < old < aside                               # (the table: a share of the old rate-based hits)

    def test_the_spill_quadrature_is_built_at_import_not_in_the_step(self):
        """spill_geometry's Gauss-Hermite points come from a table built at import (numpy inside the compiled step
        becomes CPU tensors, which inductor cannot compile without a C++ compiler): standard normal moments, the
        state's dtype."""
        x, w = missile._hermite(torch.device("cpu"), torch.float64, 8)
        assert x.dtype == torch.float64 and len(x) == 8
        assert abs(float(w.sum()) - 1) < 1e-12 and abs(float((w * x * x).sum()) - 1) < 1e-9
        assert abs(float((w * x ** 4).sum()) - 3) < 1e-9

    def test_a_shield_blocks_from_the_front_only(self):
        st = face_off(ARCHER, "wh2_main_skv_cha_warlord_0", gap=100)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        target = torch.tensor([[1, -1]])
        _, front, _ = missile.volley(st.u, pw, target, 1.0, P)
        st.u["b"][0, 1] = 90.0
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        _, back, _ = missile.volley(st.u, pw, target, 1.0, P)
        assert float(back.sum()) > float(front.sum()) > 0

    def test_shielded_spearmen_take_fewer_hits_from_the_front_only(self):
        # Skaven slingers shoot Empire spearmen without and with shields (35% block, passport).
        assert P.units[SHIELD_SPEAR]["shield"]["missile_block_chance"] == 35
        assert P.units[SPEAR]["shield"]["missile_block_chance"] == 0
        target = torch.tensor([[-1, 0]])                      # the slingers (slot H) shoot slot 0
        lost = {}
        for key in (SPEAR, SHIELD_SPEAR):
            for facing in (90.0, 270.0):                      # towards the slingers, then away
                st = face_off(key, SLINGER, gap=100)
                st.u["b"][0, 0] = facing
                pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
                lost[key, facing] = float(missile.volley(st.u, pw, target, 1.0, P)[1].sum())
        assert lost[SHIELD_SPEAR, 90.0] == pytest.approx(0.65 * lost[SPEAR, 90.0], rel=1e-4)
        assert lost[SHIELD_SPEAR, 270.0] == pytest.approx(lost[SPEAR, 270.0], rel=1e-4)
        assert lost[SPEAR, 90.0] > 0

    def test_nearer_hits_more_and_a_lone_man_is_a_small_target(self):
        far = face_off(ARCHER, SLAVE, gap=120)
        near = face_off(ARCHER, SLAVE, gap=40)
        target = torch.tensor([[1, -1]])
        hp = [float(missile.volley(s.u, geometry.pairwise(s.u, 1.5), target, 1.0, P)[1].sum()) for s in (far, near)]
        assert hp[1] > hp[0]
        lord = face_off(ARCHER, GENERAL, gap=120)
        lone = float(missile.volley(lord.u, geometry.pairwise(lord.u, 1.5), target, 1.0, P)[1].sum())
        assert lone < 0.2 * hp[0]

    def test_shots_into_a_melee_hit_friends_and_shots_spill_on_neighbours(self):
        # side 1: archers far west, spearmen at x=0; side 2: spearmen touching ours, more spearmen 10 m behind.
        # (the old spill tables: missile.spill_geometry off)
        P = TestMissile.OLD
        st = scenario.build([army([(ARCHER, -100, 0, 90), (SPEAR, 0, 0, 90)],
                                  [(SPEAR, 10, 0, 270), (SPEAR, 30, 0, 270)])], P)
        H = st.N // 2
        pw = geometry.pairwise(st.u, 1.5)
        target = torch.full((1, st.N), -1)
        target[0, 0] = H                                      # the archers shoot the enemy in melee
        contact = torch.zeros((1, st.N, st.N), dtype=torch.bool)
        contact[0, 1, H] = contact[0, H, 1] = True
        _, hp, _ = missile.volley(st.u, pw, target, 1.0, P, contact=contact)
        assert float(hp[0, 0, 1]) > 0                        # friendly fire on our spearmen
        assert float(hp[0, 0, H + 1]) > 0                    # spill on the enemy's neighbour
        _, alone, _ = missile.volley(st.u, pw, target, 1.0, P, contact=torch.zeros_like(contact))
        assert float(alone[0, 0, 1]) == 0 and float(alone[0, 0, H]) > float(hp[0, 0, H])
        share = P.sim["missile"]["friendly_fire"]["arrow"]
        hit = melee.per_hit(torch.tensor(17.0), torch.tensor(2.0), torch.tensor(30.0), torch.tensor(69.0))
        ratio = float(hp[0, 0, 1]) / float(hp[0, 0, H])       # same armour: hits on friends / on target
        assert ratio == pytest.approx(share / (1 - share), rel=1e-3) and float(hit) > 0

    def test_shots_at_a_lord_in_melee_spill_on_his_units_fighting_beside_him(self):
        # side 1: our General and spearmen beside him, both fighting the enemy; side 2: slingers shoot the General.
        # (the old spill tables: missile.spill_geometry off)
        P = TestMissile.OLD
        st = scenario.build([army([(GENERAL, 0, 0, 90), (SPEAR, 0, 8, 90)],
                                  [(SPEAR, 6, 0, 270), (SLINGER, 120, 0, 270)])], P)
        H = st.N // 2
        pw = geometry.pairwise(st.u, 1.5)
        target = torch.full((1, st.N), -1)
        target[0, H + 1] = 0                                  # the slingers shoot our General
        contact = torch.zeros((1, st.N, st.N), dtype=torch.bool)
        for i in (0, 1):
            contact[0, i, H] = contact[0, H, i] = True
        _, hp, _ = missile.volley(st.u, pw, target, 1.0, P, contact=contact)
        assert float(hp[0, H + 1, 1]) > 0 == float(hp[0, H + 1, 0])  # the crowd takes the spill, the lord none
        _, whole, _ = missile.volley(st.u, pw, target, 1.0, P.with_cal("missile", single_entity_in_melee=1.0),
                                     contact=contact)
        assert float(whole[0, H + 1, 1]) > float(whole[0, H + 1, 0]) > 0   # (the old 1: the crowd more than him)
        free = contact.clone()
        free[0, 1, H] = free[0, H, 1] = False                 # our spearmen out of melee: the ordinary spill
        _, out, _ = missile.volley(st.u, pw, target, 1.0, P, contact=free)
        ratio = float(hp[0, H + 1, 1]) / float(out[0, H + 1, 1])
        d = float(pw["dist"][0, 0, 1])
        expect = float(missile.distance_factor(torch.tensor(d), P.sim["missile"]["spill_melee"]))             / float(missile.distance_factor(torch.tensor(d), P.sim["missile"]["spill"]))
        assert ratio == pytest.approx(expect, rel=1e-3)

    def test_a_standing_shooter_turns_to_a_target_behind_before_it_aims(self):
        """missile.stand_fire_arc_deg + turn.formation_deg_s: standing, a shooter facing away from the target of
        its attack order turns at the formation rate and aims only once the target is within the arc."""
        def first_shot(params, steps=24):
            st = scenario.build([army([(ARCHER, -100, 0, 270)], [(SLAVE, 0, 0, 270)])], params)
            b0, bearings = float(st.u["b"][0, 0]), []
            H = st.N // 2
            for k in range(steps):
                o = replay.hold(st)
                o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
                battle.step(st, o, params)
                bearings.append(float(st.u["b"][0, 0]))
                if float(st.u["a"][0, 0]) < float(st.u["ammo0"][0, 0]):
                    return (k + 1) * params.dt, b0, bearings
            return None, b0, bearings
        rate = P.sim["turn"]["formation_deg_s"]
        arc = P.sim["missile"]["stand_fire_arc_deg"]
        aim_s = P.sim["missile"]["aim_s"]["arrow"]
        t, b0, bearings = first_shot(P)
        step = abs(((bearings[0] - b0) + 180) % 360 - 180)
        assert step == pytest.approx(rate * P.dt, abs=0.5)                       # one step of the turn
        assert t is not None and t >= (180 - arc) / rate + aim_s - P.dt         # turned, then aimed
        off = P.with_cal("missile", stand_fire_arc_deg=180, per_man_arc=0, turn_done_deg=None).with_cal("turn", formation_deg_s=0)
        t_off, _, bearings_off = first_shot(off)
        assert t_off is not None and t_off < t and abs(((bearings_off[0] - 90) + 180) % 360 - 180) < 1

    def test_the_turn_limit_takes_a_rate_per_unit(self):
        u = {"b": torch.tensor([[100.0, 100.0]])}
        movement.limit_turn(u, torch.tensor([[0.0, 0.0]]), torch.tensor([[True, True]]), torch.tensor([[40.0, 20.0]]))
        assert u["b"].tolist() == [[40.0, 20.0]]

    def test_out_of_range_nobody_is_chosen(self):
        st = face_off(ARCHER, SLAVE, gap=200)
        pw = geometry.pairwise(st.u, 1.5)
        can = torch.tensor([[True, False]])
        t = missile.choose_target(st.u, pw, can, torch.tensor([[-1, -1]]), torch.tensor([[False, False]]))
        assert t.tolist() == [[-1, -1]]


MILITIA_K, NR = "wh_dlc04_emp_inf_free_company_militia_0", "wh2_main_skv_inf_night_runners_1"


def shooter_and(targets, key=ARCHER, b=0.0, width=None, t_width=None):
    """A shooter at the origin facing bearing b (0 north) and enemy units at (x, z) [, key, bearing]; t_width: the
    targets' front, m (narrow ones: the fire arc reaches their centre)."""
    rows = [(t[2] if len(t) > 2 else SLAVE, t[0], t[1], t[3] if len(t) > 3 else 180.0) for t in targets]
    st = scenario.build([army([(key, 0, 0, b)], rows)], P)
    if width:
        st.u["width"][0, 0] = width
    if t_width:
        H = st.N // 2
        st.u["width"][0, H:H + len(targets)] = t_width
    return st


def at(d, deg):
    a = math.radians(deg)
    return d * math.sin(a), d * math.cos(a)


def shots_over(st, seconds, params=P, every=None):
    """Run HOLD orders; [(t, projectiles fired by slot 0 this step)] for the steps it fired."""
    out = []
    for k in range(int(round(seconds / params.dt))):
        a0 = float(st.u["a"][0, 0])
        battle.step(st, replay.hold(st), params)
        if every:
            every(st, k)
        fired = a0 - float(st.u["a"][0, 0])
        if fired > 1e-6:
            out.append(((k + 1) * params.dt, fired))
    return out


class TestMissileRules:
    """The shooting rules of 06.10.2026 (build/missile2/spec.md, build/accuracy, build/shields/spec.md)."""

    def test_the_arc_share_is_the_men_whose_arc_holds_the_target_centre(self):
        last = 2.0
        for deg in (0, 20, 35, 50, 70):
            for width in (6.0, 60.0):                  # a narrow target, a wide one (its front across the line)
                st = shooter_and([at(60, deg)])
                H = st.N // 2
                st.u["width"][0, H] = width
                st.u["b"][0, H] = 180.0 + deg                                   # facing the shooter
                pw = geometry.pairwise(st.u, 1.5)
                share = float(missile.arc_share(st.u, pw, torch.tensor([[H] + [-1] * (st.N - 1)]))[0, 0])
                W, F = float(pw["front"][0, 0]), float(pw["front"][0, H])
                lat, fwd = 60 * math.sin(math.radians(deg)), 60 * math.cos(math.radians(deg))
                reach = fwd * math.tan(math.radians(30)) + F / 2 * abs(math.cos(math.radians(180 + deg)))
                expect = max(0.0, min(lat + reach, W / 2) - max(lat - reach, -W / 2)) / W
                assert share == pytest.approx(expect, abs=1e-3), (deg, width)
                if deg == 0:
                    assert share == pytest.approx(1.0)
                if width == 6.0:
                    assert share <= last + 1e-6                                  # fewer men the further off
                    last = share
            if deg == 35:
                assert 0.25 < share                                               # the wide one: most of them

    def test_the_militia_arc_is_35_degrees(self):
        st = shooter_and([at(60, 30)], key=MILITIA_K, t_width=6.0)
        assert float(st.u["arc"][0, 0]) == 35.0 and float(shooter_and([at(60, 0)]).u["arc"][0, 0]) == 30.0
        pw = geometry.pairwise(st.u, 1.5)
        H = st.N // 2
        tg = torch.tensor([[H] + [-1] * (st.N - 1)])
        wide = float(missile.arc_share(st.u, pw, tg)[0, 0])
        st.u["arc"][0, 0] = 30.0
        narrow = float(missile.arc_share(st.u, pw, tg)[0, 0])
        assert wide > narrow + 0.1

    def test_a_target_off_to_one_side_is_shot_by_the_near_flank_and_the_unit_keeps_its_facing(self):
        st = shooter_and([at(80, 30)], t_width=6.0)
        fired = shots_over(st, 16)
        assert fired and abs(((float(st.u["b"][0, 0]) - 0) + 180) % 360 - 180) < 1   # bearing unchanged
        straight = shots_over(shooter_and([at(80, 0)]), 16)
        assert fired[0][1] < 0.8 * straight[0][1]                        # fewer men in the volley

    def test_firing_at_will_a_unit_does_not_turn(self):
        st = shooter_and([at(80, 60)])
        fired = shots_over(st, 16)
        assert abs(((float(st.u["b"][0, 0]) - 0) + 180) % 360 - 180) < 1
        assert sum(f for _, f in fired) < 0.2 * float(st.u["men0"][0, 0])

    def test_an_ordered_target_beyond_45_degrees_is_turned_to_then_shot_by_all(self):
        st = shooter_and([at(80, 60)])
        H = st.N // 2
        fired = []
        for k in range(32):
            o = replay.hold(st)
            o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
            a0 = float(st.u["a"][0, 0])
            battle.step(st, o, P)
            if a0 > float(st.u["a"][0, 0]):
                fired.append(((k + 1) * P.dt, a0 - float(st.u["a"][0, 0])))
        off = abs(((float(st.u["b"][0, 0]) - 60) + 180) % 360 - 180)
        assert off <= P.sim["missile"]["turn_done_deg"] + 1                 # turned to within 10 deg
        rate = P.sim["turn"]["formation_deg_s"]
        assert fired[0][0] >= (60 - 10) / rate + P.sim["missile"]["aim_s"]["arrow"] - P.dt
        assert fired[0][1] == pytest.approx(float(st.u["men0"][0, 0]), rel=0.15)

    def test_a_target_straight_ahead_gets_whole_volleys_one_database_reload_apart(self):
        st = shooter_and([(0, 100)])
        fired = shots_over(st, 25)
        assert len(fired) >= 2 and fired[0][1] == pytest.approx(90, abs=0.5)
        assert fired[1][0] - fired[0][0] == pytest.approx(10.0, abs=P.dt + 1e-6)
        assert float(st.u["reload"][0, 0]) == 10.0

    def test_a_new_target_costs_three_seconds_without_fire(self):
        H = None
        kill_at = {}

        def kill(st, k):
            # 9 s after the first volley the nearer target dies: the men are loaded at 10 s
            t = (k + 1) * P.dt
            if "t0" in kill_at and abs(t - kill_at["t0"] - 9.0) < 1e-6:
                st.u["men"][0, H] = 0.0
                st.u["hp_abs"][0, H] = 0.0
                kill_at["t"] = t
        st = shooter_and([(0, 90), (8, 100)])
        H = st.N // 2

        def watch(st, k):
            if "t0" not in kill_at and float(st.u["a"][0, 0]) < float(st.u["ammo0"][0, 0]):
                kill_at["t0"] = (k + 1) * P.dt
            kill(st, k)
        fired = shots_over(st, 30, every=watch)
        assert len(fired) >= 2
        assert fired[1][0] - kill_at["t"] >= P.sim["missile"]["retarget_s"] - P.dt - 1e-6
        none = shots_over(shooter_and([(0, 90)]), 30)
        assert fired[1][0] - fired[0][0] > none[1][0] - none[0][0]          # later than one reload

    def test_a_shooter_keeps_its_target_when_two_are_about_as_near(self):
        def swap(st, k):
            H = st.N // 2
            z0, z1 = float(st.u["z"][0, H]), float(st.u["z"][0, H + 1])
            st.u["z"][0, H], st.u["z"][0, H + 1] = z1, z0                    # the nearest flips every step
        st = shooter_and([(-20, 100), (20, 101)])
        fired = shots_over(st, 25, every=swap)
        assert len(fired) >= 2 and fired[1][0] - fired[0][0] == pytest.approx(10.0, abs=P.dt + 1e-6)

    def test_range_is_centre_to_centre(self):
        for d, ok in ((129.0, True), (131.0, False)):
            st = shooter_and([(0, d, SPEAR)])
            pw = geometry.pairwise(st.u, 1.5)
            t = missile.choose_target(st.u, pw, torch.tensor([[True] + [False] * (st.N - 1)]),
                                      torch.full((1, st.N), -1), torch.zeros((1, st.N), dtype=torch.bool))
            assert (int(t[0, 0]) >= 0) == ok, d
        deep = shooter_and([(0, 129.0, SPEAR)], width=8)                   # a column 40+ m deep
        fired = shots_over(deep, 8)
        assert fired and fired[0][1] == pytest.approx(90, abs=0.5)           # the whole unit fires

    def test_an_attacking_shooter_walks_until_the_target_centre_is_in_range(self):
        st = shooter_and([(0, 200, SPEAR)])
        H = st.N // 2

        def attack(st):
            o = replay.hold(st)
            o.kind[0, 0] = O.ATTACK
            o.target[0, 0] = H
            return o
        first = None
        for k in range(120):
            a0 = float(st.u["a"][0, 0])
            battle.step(st, attack(st), P)
            if first is None and float(st.u["a"][0, 0]) < a0:
                first = float(torch.hypot(st.u["x"][0, 0] - st.u["x"][0, H], st.u["z"][0, 0] - st.u["z"][0, H]))
        assert first is not None and 126.0 <= first <= 130.0

    def test_the_spread_model_hits_less_far_off_loose_and_thinned(self):
        d_near, d_far = shooter_and([(0, 60)]), shooter_and([(0, 120)])
        p = [float(missile.hit_chance(s.u, geometry.pairwise(s.u, 1.5), P)[0, 0, s.N // 2]) for s in (d_near, d_far)]
        assert 1 > p[0] > p[1] > 0.2
        loose = shooter_and([(0, 100, SLINGER)])
        dense = shooter_and([(0, 100, SLAVE)])
        pl, pd = (float(missile.hit_chance(s.u, geometry.pairwise(s.u, 1.5), P)[0, 0, s.N // 2]) for s in (loose, dense))
        assert pd > pl
        thin = shooter_and([(0, 100, SLAVE)])
        thin.u["men"][0, thin.N // 2] = 30.0
        pt = float(missile.hit_chance(thin.u, geometry.pairwise(thin.u, 1.5), P)[0, 0, thin.N // 2])
        assert pt < pd

    def test_a_lone_man_by_the_closed_form(self):
        # the spread in the plane across the line of fire (missile.accuracy.plane; build/open_missile/spec.md S1)
        st = shooter_and([(0, 100, "wh2_main_skv_cha_warlord_0")])
        H = st.N // 2
        p = float(missile.hit_chance(st.u, geometry.pairwise(st.u, 1.5), P)[0, 0, H])
        m = P.units[ARCHER]["missile"]
        lord = P.units["wh2_main_skv_cha_warlord_0"]
        sig = math.sqrt(m["calibration_area_m"]) * 100 / m["calibration_distance_m"] * math.sqrt(1 - (10 + 10) / 100)
        th = 0.5 * math.asin(100 * 9.81 / m["muzzle_velocity"] ** 2)
        L = lord["height_m"] / math.tan(th)
        sa, c, r = sig / math.sin(th), L / 2, lord["radius_m"]

        def phi(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        expect = math.erf(r / (math.sqrt(2) * sig)) * (phi((L + r - c) / sa) - phi((-r - c) / sa))
        assert p == pytest.approx(expect, rel=1e-4) and 0.05 < p < 0.3
        assert "k" not in P.sim["missile"]["accuracy"]                       # no fitted number

    def test_the_spread_is_the_square_root_of_the_calibration_area(self):
        m, unit = P.units[MILITIA_K]["missile"], P.units[MILITIA_K]
        acc = (unit["missile"].get("accuracy", 0) or 0) + (m.get("marksmanship", 0) or 0)
        sig = math.sqrt(m["calibration_area_m"]) * 65 / m["calibration_distance_m"] * math.sqrt(1 - acc / 100)
        assert sig == pytest.approx(1.26, abs=0.02)                          # militia at 65 m (spec S1 d)

    def test_direct_fire_on_a_lone_man_falls_with_distance(self):
        p = []
        for d in (40, 100):
            st = shooter_and([(0, d, "wh2_main_skv_cha_warlord_0")], key=MILITIA_K)
            p.append(float(missile.hit_chance(st.u, geometry.pairwise(st.u, 1.5), P)[0, 0, st.N // 2]))
        assert p[0] > 1.3 * p[1]

    def test_pistols_fire_whole_while_the_front_rank_reaches_the_centre_then_by_ranks(self):
        """missile.per_man_range_direct (build/probes7 P2): militia (range 90, 7 ranks 1.7 m apart, 30 m front) on
        skavenslaves 30 m wide (10 ranks 1.8 m): the whole unit while the target centre is within 90 m of the front
        rank (centre <= 95.1 m), then the ranks within 90 m of the target's front rank (97.6 m: 4 of 7), none at
        110 m; arcing fire (archers) keeps the centre rule."""
        def share(d, key=MILITIA_K):
            st = shooter_and([(0, d, SLAVES)], key=key, width=30.0, t_width=30.0)
            pw = geometry.pairwise(st.u, 1.5)
            return float(missile.rank_share(st.u, pw)[0, 0, st.N // 2]), st, pw
        assert share(88)[0] == 1.0 and share(94.5)[0] == 1.0
        assert share(97.6)[0] == pytest.approx(4 / 7) and share(110)[0] == 0.0
        _, st, pw = share(97.6)
        assert bool(missile.in_range(st.u, pw, per_man=True)[0, 0, st.N // 2])
        assert not bool(missile.in_range(st.u, pw)[0, 0, st.N // 2])            # the centre rule
        _, st, pw = share(132, key=ARCHER)
        assert not bool(missile.in_range(st.u, pw, per_man=True)[0, 0, st.N // 2])  # not direct: centre

    def test_only_units_firing_whilst_moving_count_their_range_per_man(self):
        """missile.per_man_range_fire_move_only (probes newdist, rangenew, 07.10.2026): handgunners (direct fire, no
        fire whilst moving, range 145) take the centre rule - no fire at 153 m centre to centre where the per-rank rule
        would let their front ranks fire; the militia (fire whilst moving) keep the per-man rule."""
        hg = "wh_main_emp_inf_handgunners"
        st = shooter_and([(0, 153, SLAVES)], key=hg, width=30.0, t_width=30.0)
        pw = geometry.pairwise(st.u, 1.5)
        H = st.N // 2
        assert float(missile.rank_share(st.u, pw)[0, 0, H]) > 0                   # the per-rank rule would fire
        mask = missile.per_man_mask(st.u, P)
        assert not bool(mask[0, 0]) and not bool(missile.in_range(st.u, pw, per_man=mask)[0, 0, H])
        assert sum(x for _, x in shots_over(st, 30)) == 0
        st = shooter_and([(0, 97.6, SLAVES)], key=MILITIA_K, width=30.0, t_width=30.0)
        pw = geometry.pairwise(st.u, 1.5)
        mask = missile.per_man_mask(st.u, P)
        assert bool(mask[0, 0]) and bool(missile.in_range(st.u, pw, per_man=mask)[0, 0, st.N // 2])
        off = P.with_cal("missile", per_man_range_fire_move_only=0)
        assert bool(missile.per_man_mask(shooter_and([(0, 153, SLAVES)], key=hg).u, off)[0, 0])

    def test_units_firing_whilst_moving_reach_25_m_past_their_range_at_half_their_men(self):
        """missile.fire_move_reach_m / _rate (probes rangestand, rangenew; the gate sets' records, build/shotgap/reach*):
        the militia (range 90) fire at a standing target 103 / 111 m off centre to centre, the stars (range 70) at
        89.5 m, with at least half the men; nothing past range + 25 m; the handgunners still never at 153 m."""
        reach = missile.reach_of(P)
        assert reach == (25.0, 0.5)
        for key, d, want in ((MILITIA_K, 103, 0.5), (MILITIA_K, 111, 0.5), (MILITIA_K, 116, 0.0),
                             ("wh2_main_skv_inf_night_runners_0", 89.5, 0.5), ("wh2_main_skv_inf_night_runners_0", 96, 0.0)):
            st = shooter_and([(0, d, SLAVES)], key=key, width=30.0, t_width=30.0)
            pw = geometry.pairwise(st.u, 1.5)
            H = st.N // 2
            assert float(missile.rank_share(st.u, pw, *reach)[0, 0, H]) == pytest.approx(want), (key, d)
            assert float(missile.rank_share(st.u, pw)[0, 0, H]) <= 1 / 7 + 1e-6       # the rank rule alone: 0-1 rank
            mask = missile.per_man_mask(st.u, P)
            assert bool(missile.in_range(st.u, pw, per_man=mask, reach=reach)[0, 0, H]) == (want > 0)
        st = shooter_and([(0, 111, SLAVES)], key=MILITIA_K, width=30.0, t_width=30.0)
        assert sum(x for _, x in shots_over(st, 30)) > 0
        st = shooter_and([(0, 111, SLAVES)], key=MILITIA_K, width=30.0, t_width=30.0)
        assert sum(x for _, x in shots_over(st, 30, P.with_cal("missile", fire_move_reach_m=0.0))) == 0
        st = shooter_and([(0, 153, SLAVES)], key="wh_main_emp_inf_handgunners", width=30.0, t_width=30.0)
        assert sum(x for _, x in shots_over(st, 30)) == 0

    def test_bullets_into_a_melee_hit_our_men_by_the_measured_share(self):
        """missile.friendly_fire.musket 0.08 (the probe meleefire: handgunners at 45 / 90 deg to the contact, 0.07 /
        0.08 of the hits on our men); arrows and slings keep their recorded shares."""
        ff = P.sim["missile"]["friendly_fire"]
        assert ff["musket"] == 0.08 and ff["arrow"] == 0.26 and ff["sling"] == 0.56
        assert P.static("wh_main_emp_inf_handgunners")["friendly_fire"] == pytest.approx(0.08)
        assert P.static("wh_dlc04_emp_inf_free_company_militia_0")["friendly_fire"] == pytest.approx(0.08)

    def test_a_dense_block_near_is_hit_almost_every_time_and_less_far(self):
        p = []
        for d in (50, 130):
            st = shooter_and([(0, d, SLAVES)], t_width=30.0)
            p.append(float(missile.hit_chance(st.u, geometry.pairwise(st.u, 1.5), P)[0, 0, st.N // 2]))
        assert p[0] >= 0.85 and 0.6 <= p[1] <= 0.8, p

    def test_a_flat_shot_crossing_many_ranks_meets_more_men_than_one_rank(self):
        deep = shooter_and([(0, 40, SLAVES)], key=MILITIA_K, t_width=30.0)
        line = shooter_and([(0, 40, SLAVES)], key=MILITIA_K, t_width=290.0)
        pd, pl = (float(missile.hit_chance(s.u, geometry.pairwise(s.u, 1.5), P)[0, 0, s.N // 2]) for s in (deep, line))
        assert pd > pl

    def test_a_shooter_aims_again_when_its_target_men_die(self):
        # missile.reaim (spec S3): against a lone lord whole volleys one reload apart; against a formation that loses
        # men the next volley waits aim_s x the share of the men whose target died
        lord = shots_over(shooter_and([(0, 90, "wh2_main_skv_cha_warlord_0")]), 32)
        assert len(lord) >= 3 and all(b[0] - a[0] == pytest.approx(10.0, abs=P.dt + 1e-6) for a, b in zip(lord, lord[1:]))
        st = shooter_and([(0, 100, SLAVES)], t_width=30.0)
        # half its health gone already: its wounded men die under the next volleys (kills.missile_uniform: on fresh
        # men the first volleys mostly wound)
        st.u["hp_abs"][0, st.N // 2] = st.u["hp_abs"][0, st.N // 2] * 0.5
        st.u["leadership"] = st.u["leadership"] + 1e4            # (it does not run off)
        men0 = float(st.u["men"][0, st.N // 2])
        fired = shots_over(st, 62)
        gaps = [b[0] - a[0] for a, b in zip(fired, fired[1:])]
        assert len(gaps) >= 3 and all(g >= 10.0 - 1e-6 for g in gaps) and max(gaps) >= 10.0 + P.dt - 1e-6
        assert float(st.u["men"][0, st.N // 2]) < men0
        # the target's men lost between volleys: the wait is aim_s x their share
        assert max(gaps) <= 10.0 + P.sim["missile"]["aim_s"]["arrow"] + P.dt

    def test_a_target_running_into_range_is_shot_after_the_retarget_pause(self):
        st = shooter_and([(0, 160, SPEAR, 180.0)])
        H = st.N // 2
        entered, first = None, None
        for k in range(80):
            o = replay.hold(st)
            o.kind[0, H], o.x[0, H], o.z[0, H], o.run[0, H] = O.MOVE, 0.0, 40.0, True
            a0 = float(st.u["a"][0, 0])
            battle.step(st, o, P)
            t = (k + 1) * P.dt
            d = float(torch.hypot(st.u["x"][0, H] - st.u["x"][0, 0], st.u["z"][0, H] - st.u["z"][0, 0]))
            if entered is None and d <= float(st.u["range"][0, 0]):
                entered = t
            if first is None and float(st.u["a"][0, 0]) < a0:
                first = t
        assert entered is not None and first is not None
        assert first - entered >= P.sim["missile"]["retarget_s"] - P.dt - 1e-6
        # a halted shooter with a target already in range: aim_s as before
        fired = shots_over(shooter_and([(0, 100)]), 8)
        assert fired[0][0] == pytest.approx(P.sim["missile"]["aim_s"]["arrow"], abs=P.dt + 1e-6)

    def test_the_fire_flag_is_on_while_the_shooter_aims_as_the_games(self):
        """missile.fire_flag_aiming: the observed flag (and the target shown) from the first step a standing shooter
        aims at a target in range, before its first shot; the shot itself still waits aim_s. Off: from the shot."""
        for on in (1, 0):
            params = P.with_cal("missile", fire_flag_aiming=on)
            st = shooter_and([(0, 100)])
            H = st.N // 2
            flag, shot = [], None
            for k in range(int(round(6.0 / params.dt))):
                a0 = float(st.u["a"][0, 0])
                battle.step(st, replay.hold(st), params)
                flag.append(bool(st.u["fire"][0, 0]))
                if shot is None and float(st.u["a"][0, 0]) < a0:
                    shot = k
            assert shot is not None and shot * params.dt == pytest.approx(P.sim["missile"]["aim_s"]["arrow"],
                                                                          abs=params.dt + 1e-6)
            if on:
                assert flag[0] and all(flag[:shot + 1])
                assert int(st.u["target"][0, 0]) == H
            else:
                assert not any(flag[:shot - 1]) and flag[shot]

    def test_units_can_fall_back_to_the_measured_rates(self):
        st = shooter_and([(0, 100, SPEAR)])
        H = st.N // 2
        pw = geometry.pairwise(st.u, 1.5)
        tg = torch.tensor([[H] + [-1] * (st.N - 1)])
        rates = P.with_cal("missile", accuracy=dict(P.sim["missile"]["accuracy"], units="rates"))
        _, hp, hit = missile.volley(st.u, pw, tg, 1.0, rates)
        shots = float(st.u["men"][0, 0]) / float(st.u["reload"][0, 0])
        old = 0.42 * float(missile.distance_factor(pw["dist"][0, 0, H], P.sim["missile"]["distance_factor"]))
        assert float(hp[0, 0, H]) == pytest.approx(shots * old * float(hit[0, 0, H]), rel=1e-4)


class TestShieldRules:
    """build/shields/spec.md: small arms only, a 60 deg cone each side of the facing, no side or rear block,
    in melee too, the General's 55 %."""

    def _lost(self, key, rot, shooter=SLINGER, melee=False, params=P):
        # the target at the origin facing `rot` degrees off the direction to the shooter (100 m east); melee: an
        # enemy spearmen unit (the shooter's side) fights it
        st = scenario.build([army([(key, 0, 0, 90.0 - rot)], [(shooter, 100, 0, 270), (SPEAR, 0, 6, 180)])], params)
        H = st.N // 2
        pw = geometry.pairwise(st.u, 1.5)
        target = torch.full((1, st.N), -1)
        target[0, H] = 0
        contact = torch.zeros((1, st.N, st.N), dtype=torch.bool)
        if melee:
            contact[0, 0, H + 1] = contact[0, H + 1, 0] = True
        return float(missile.volley(st.u, pw, target, 1.0, params, contact=contact)[1][0, H, 0])

    def test_shield_cone_is_60_degrees(self):
        for rot, share in ((59.0, 0.65), (61.0, 1.0)):
            assert self._lost(SHIELD_SPEAR, rot) == pytest.approx(share * self._lost(SPEAR, rot), rel=1e-4), rot

    def test_side_takes_no_block(self):
        assert self._lost(SHIELD_SPEAR, 90.0) == pytest.approx(self._lost(SPEAR, 90.0), rel=1e-4)
        assert self._lost(SHIELD_SPEAR, 180.0) == pytest.approx(self._lost(SPEAR, 180.0), rel=1e-4)

    def test_shield_blocks_in_melee(self):
        fight, plain = self._lost(SHIELD_SPEAR, 0.0, melee=True), self._lost(SPEAR, 0.0, melee=True)
        assert fight == pytest.approx(0.65 * plain, rel=1e-4)
        assert fight < self._lost(SHIELD_SPEAR, 0.0)                       # friendly fire takes a share

    def test_lord_shield_55(self):
        front, back = self._lost(GENERAL, 0.0), self._lost(GENERAL, 180.0)
        assert front == pytest.approx(0.45 * back, rel=1e-4)

    def test_non_small_arms_ignore_shields(self):
        st = scenario.build([army([(SHIELD_SPEAR, 0, 0, 90.0)], [(SLINGER, 100, 0, 270)])], P)
        H = st.N // 2
        pw = geometry.pairwise(st.u, 1.5)
        target = torch.full((1, st.N), -1)
        target[0, H] = 0
        blocked = float(missile.volley(st.u, pw, target, 1.0, P)[1][0, H, 0])
        st.u["small_arms"][0, H] = False                                    # an artillery-like projectile
        through = float(missile.volley(st.u, pw, target, 1.0, P)[1][0, H, 0])
        assert blocked == pytest.approx(0.65 * through, rel=1e-4)
        assert all(P.static(k)["small_arms"] for k in (ARCHER, SLINGER, MILITIA_K, NR))


class TestMorale:
    def ctx(self, st, **over):
        z = torch.zeros_like(st.u["men"], dtype=torch.bool)
        c = {"aura": z, "lord_dead_points": torch.zeros_like(st.u["men"]), "secure": z, "in_melee": z,
             "routing_friends": torch.zeros_like(st.u["men"]), "routing_enemies": torch.zeros_like(st.u["men"]),
             "under_fire": z, "strong_enemy": z, "enemy_near": z | True}
        c.update(over)
        return c

    def test_the_lord_near_raises_and_his_death_lowers(self):
        st = face_off(SPEAR, SLAVE)
        base = morale.target_points(st.u, self.ctx(st), P)
        aura = morale.target_points(st.u, self.ctx(st, aura=torch.ones_like(st.u["r"])), P)
        dead = morale.target_points(st.u, self.ctx(st, lord_dead_points=torch.full_like(st.u["men"], -16.0)), P)
        assert (aura - base)[0, 0] == 4 and (dead - base)[0, 0] == -16

    def test_attacked_in_the_flank_or_rear_holds_while_struck_from_there(self):
        """M3: the database's was_attacked_in_flank / _rear (-6 / -14) in the target every step a blow comes from
        that side (u flank_hit), the rear instead of the flank, 0 once the attacker is gone (the morale probe T-A)."""
        st = face_off(SPEAR, SLAVE)
        base = morale.target_points(st.u, self.ctx(st), P)
        got = []
        for side in (1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0):
            st.u["flank_hit"][0, 0] = side
            got.append(float((morale.target_points(st.u, self.ctx(st), P) - base)[0, 0]))
        assert got == [-6.0] * 5 + [0.0, -14.0, 0.0]

    def test_the_army_beaten_as_a_whole_routs(self):
        # Army destruction: -120 to every unit of the side (battle.py decides when).
        st = face_off(SPEAR, SLAVE)
        hit = torch.ones_like(st.u["r"])
        for _ in range(8):
            morale.step(st.u, self.ctx(st, collapse=hit), P, 0.5)
        assert bool(st.u["r"].all())

    def test_army_losses_use_ammunition_and_count_routers_until_shattered(self):
        st = face_off(ARCHER, SPEAR)
        u = st.u
        # Equal initial value; at 40% HP the missile unit is above the 22% threshold.
        u["cost"][:] = 100
        u["hp"][0, 0] = 0.4
        assert not bool(morale.army_collapse(u, P)[0, 0])
        u["a"][0, 0] = 0
        assert bool(morale.army_collapse(u, P)[0, 0])
        # A routing enemy still contributes strength; a shattered or departed one does not.
        u["r"][0, 1] = True
        assert bool(morale.army_collapse(u, P)[0, 0])
        u["s"][0, 1] = True
        assert not bool(morale.army_collapse(u, P)[0, 0])
        u["s"][0, 1] = False
        u["gone"][0, 1] = True
        assert not bool(morale.army_collapse(u, P)[0, 0])

    def test_army_destruction_requires_both_database_thresholds(self):
        st = face_off(SPEAR, SLAVE)
        u = st.u
        u["cost"][:] = 100
        u["cp_fixed"][:] = 100                  # the strength is the combat potential (morale.collapse strength cp)
        u["hp"][:] = 0.2
        assert not bool(morale.army_collapse(u, P).any())  # both depleted, neither outmatched
        u["hp"][0, 1] = 1
        assert morale.army_collapse(u, P).tolist() == [[True, False]]
        u["hp"][0, 0] = 0.3
        assert not bool(morale.army_collapse(u, P).any())  # outmatched, but not depleted
        off = P.with_cal("morale", collapse={"on": False})
        u["hp"][0, 0] = 0.1
        assert not bool(morale.army_collapse(u, off).any())

    def test_routing_reduces_army_power_without_removing_the_unit(self):
        st = scenario.build([army([(SPEAR, -300, 0, 90), (SPEAR, -300, 100, 90)],
                                  [(SPEAR, 300, 0, 270), (SPEAR, 300, 100, 270)])], P)
        u = st.u
        u["hp"][0, :2] = 0.28
        assert not bool(morale.army_collapse(u, P)[0, :2].any())
        # One friend routes: 0.28 * (1 + 0.5) / 2 = 0.21 of starting army power.
        u["r"][0, 0] = True
        assert bool(morale.army_collapse(u, P)[0, :2].all())
        # Routing the equally strong opposing army halves its strength, but does not zero it.
        u["hp"][0, :2] = 0.4
        u["r"][0, 2:] = True
        assert not bool(morale.army_collapse(u, P).any())

    def test_terminal_routers_lose_morale_at_distance_and_shatter_before_third_rout(self):
        st = face_off(SPEAR, SLAVE)
        st.u["r"][:] = True
        st.u["rout_count"][:] = 1
        st.u["morale"][:] = -49
        hit = torch.ones_like(st.u["r"])
        ctx = self.ctx(st, enemy_near=~hit, collapse=hit)
        for _ in range(8):
            morale.step(st.u, ctx, P, 0.5)
        assert bool(st.u["s"].all()) and bool(st.u["r"].all())
        assert bool((st.u["morale"] < -50).all())
        assert bool((st.u["rout_count"] == 1).all())

    def test_unbreakable_survives_terminal_morale_and_recovery_still_works(self):
        st = face_off(SPEAR, SLAVE)
        st.u["unbreakable"][0, 0] = True
        hit = torch.ones_like(st.u["r"])
        for _ in range(30):
            morale.step(st.u, self.ctx(st, collapse=hit), P, 0.5)
        assert not bool(st.u["r"][0, 0]) and not bool(st.u["s"][0, 0])
        assert bool(st.u["s"][0, 1])
        # Without army losses, a distant router still recovers normally: its morale heads for its target.
        st = face_off(SPEAR, SLAVE)
        st.u["r"][:] = True
        st.u["morale"][:] = -10
        target = morale.target_points(st.u, self.ctx(st, enemy_near=~hit), P)
        morale.step(st.u, self.ctx(st, enemy_near=~hit), P, 0.5)
        assert torch.allclose(st.u["morale"], -10 + 0.15 * (target + 10))

    def test_terminal_shatter_switch_and_post_rally_immunity(self):
        st = face_off(SPEAR, SLAVE)
        hit = torch.ones_like(st.u["r"])
        st.u["rout_count"][:] = 1
        st.u["rally_s"][:] = 0
        st.u["morale"][:] = -60
        # The ordinary ten-second protection after rally does not make a destroyed army immortal.
        morale.step(st.u, self.ctx(st, collapse=hit), P, 0.5)
        assert bool(st.u["s"].all()) and bool(st.u["r"].all())
        st = face_off(SPEAR, SLAVE)
        st.u["morale"][:] = -60
        cal = dict(P.sim["morale"]["collapse"], shatter_below_broken=False)
        morale.step(st.u, self.ctx(st, collapse=hit), P.with_cal("morale", collapse=cal, shatter_rules=False), 0.5)
        assert not bool(st.u["s"].any()) and bool(st.u["r"].all())

    def test_the_third_rout_shatters_and_a_free_unit_rallies(self):
        st = face_off(SPEAR, SLAVE)
        st.u["morale"][0, 0] = -20.0
        free = self.ctx(st, enemy_near=torch.zeros_like(st.u["r"]))
        morale.step(st.u, self.ctx(st), P, 0.5)
        assert st.u["r"][0, 0] and st.u["rout_count"][0, 0] == 1
        for _ in range(200):
            morale.step(st.u, free, P, 0.5)
            if not st.u["r"][0, 0]:
                break
        assert not st.u["r"][0, 0] and st.u["morale"][0, 0] > 0
        assert float(st.u["rally_s"][0, 0]) == 0 and P.sim["morale"]["rally_after_s"] > 0
        st.u["rout_count"][0, 0] = 2
        st.u["rally_s"][0, 0] = 100.0
        st.u["morale"][0, 0] = -20.0
        morale.step(st.u, self.ctx(st), P, 0.5)
        assert st.u["s"][0, 0] and st.u["ms"][0, 0] == 7


# --- level 3: the battle loop ---

class TestReplay:
    def test_a_unit_in_melee_without_a_recorded_target_fights_the_nearest_or_holds(self):
        from tools.nn import gamedata
        import numpy as np
        T, N = 3, 2
        f = {k: np.zeros((T, N)) for k in gamedata.FLOAT_FIELDS}
        f.update({k: np.zeros((T, N), dtype=bool) for k in gamedata.BOOL_FIELDS})
        f["x"][:, 1] = 10.0
        f["ox"][:] = f["x"]
        f["men"][:] = 100
        f["m"][:] = True
        b = gamedata.Battle(run="t", own_ai="net", enemy_role="attack", result={}, t=np.arange(T, dtype=float), f=f,
                            target=np.full((T, N), -1), names=("a", "b"), keys=("", ""), side=np.array([1, 2]))
        rows = replay.recorded_orders(b, [0, 1], N, fight_nearest=[False, True], leave_m=10.0, leavers=[True, False])
        assert (rows["kind"][:, 0] == O.HOLD).all() and (rows["target"][:, 0] == -1).all()
        assert (rows["kind"][:, 1] == O.ATTACK).all() and (rows["target"][:, 1] == 0).all()
        both = replay.recorded_orders(b, [0, 1], N, fight_nearest=True, leave_m=10.0)
        assert (both["kind"] == O.ATTACK).all()


class TestBattle:
    @pytest.mark.parametrize("order,expected", [("hold", True), ("keep", True), ("arrived", True),
                                               ("move", False), ("attack", False), ("aim", False)])
    def test_idle_recovery_tracks_pending_work(self, monkeypatch, order, expected):
        st = face_off(ARCHER if order == "aim" else SPEAR, SLAVE, gap=80 if order == "aim" else 500)
        seen = {}
        def record(u, activity, params, dt):
            seen.update({k: v.clone() for k, v in activity.items()})
        monkeypatch.setattr(fatigue, "step", record)
        cmd = O.hold(st.B, st.N)
        if order == "keep":
            cmd.kind.fill_(O.KEEP)
        elif order in ("arrived", "move"):
            cmd.kind[0, 0] = O.MOVE
            cmd.x[0, 0] = st.u["x"][0, 0] + (30 if order == "move" else 0.4)
            cmd.z[0, 0] = st.u["z"][0, 0]
        elif order == "attack":
            cmd.kind[0, 0] = O.ATTACK
            cmd.target[0, 0] = 1
        battle.step(st, cmd, P)
        assert bool(seen["idle"][0, 0]) is expected
        assert bool(seen["active"][0, 0])

    def test_spearmen_beat_slaves_and_the_battle_ends(self):
        st = scenario.build([scenario.from_arena("pair_spear_v_slave", "attack")], P)
        battle.run(st, replay.nearest_attack, P)
        assert bool(st.done[0]) and int(st.winner[0]) == 1 and float(st.t[0]) < 900
        assert float(st.u["k"][0, 0]) > 0 and float(st.u["fatigue"][0, 0]) > 0

    def test_archers_shoot_a_unit_that_holds_until_it_routs(self):
        st = scenario.build([scenario.from_arena("missile_archers_v_slave", "hold")], P)
        H = st.N // 2

        def policy(s):
            o = replay.nearest_attack(s)
            o.kind[:, :H] = O.HOLD            # our slaves hold
            return o
        battle.run(st, policy, P)
        assert int(st.winner[0]) == 2 and bool(st.u["r"][0, 0])
        assert float(st.u["a"][0, H]) < float(st.u["ammo0"][0, H]) and float(st.u["hp"][0, H]) == 1.0

    def test_on_time_the_defender_wins(self):
        p = P.with_cal("battle_limit_s", value=30)
        st = scenario.build([army([(SPEAR, -300, 0, 90)], [(SPEAR, 300, 0, 270)], attacker=1)], p)
        battle.run(st, replay.hold, p)
        assert bool(st.done[0]) and int(st.winner[0]) == 2 and float(st.t[0]) == pytest.approx(30)

    def test_battles_in_one_batch_do_not_touch_each_other(self):
        a = scenario.from_arena("pair_spear_v_slave", "attack")
        b = scenario.from_arena("pair_general_v_clanrat", "attack")
        both = scenario.build([a, b], P)
        battle.run(both, replay.nearest_attack, P)
        alone = scenario.build([b], P)
        battle.run(alone, replay.nearest_attack, P)
        assert int(both.winner[1]) == int(alone.winner[0])
        assert float(both.t[1]) == pytest.approx(float(alone.t[0]))
        assert float(both.u["hp"][1, 1]) == pytest.approx(float(alone.u["hp"][0, 1]), abs=1e-4)

    def test_a_router_leaves_over_the_edge(self):
        # A second unit far away keeps side 1 in the battle.
        st = scenario.build([army([(SPEAR, 1000, 0, 90), (SPEAR, -500, 0, 90)], [(SPEAR, 900, 0, 90)])], P)
        st.u["r"][0, 0] = True
        st.u["morale"][0, 0] = -20.0
        for _ in range(40):
            battle.step(st, replay.hold(st), P)
        assert bool(st.u["gone"][0, 0]) and not bool(st.done[0])
        assert not bool(st.u["mv"][0, 0]) and int(st.u["target"][0, 0]) == -1

    @staticmethod
    def lord_fall_gap(fall, seconds, factions=("wh_main_emp_empire", "wh2_main_skv_skaven")):
        """Side 1's spearmen's morale, their General fallen (fall(st): battle 0) minus untouched (battle 1),
        at each of `seconds`; the General 600 m off (his aura and his rout touch no one), the enemy 600 m off."""
        sides = army([(GENERAL, -900, 0, 90, True), (SPEAR, -300, 0, 90)], [(SPEAR, 300, 0, 270)], factions=factions)
        st = scenario.build([sides, sides], P)
        for _ in range(40):          # settle first (flanks secure: no enemy within 146 m, +5 from the start)
            battle.step(st, replay.hold(st), P)
        fall(st)
        out, t = [], 0.0
        for s in seconds:
            while t < s - 1e-6:
                battle.step(st, replay.hold(st), P)
                t += 0.5
            out.append(float(st.u["morale"][0, 1] - st.u["morale"][1, 1]))
        return st, out

    def test_a_killed_lord_costs_his_army_16_then_10(self):
        # sim.json morale.lord_fall (in-game, tools/nn/lord_fall.py): units out of the killed lord's aura drop
        # -3 / -7 / -13.5 / -16 at 1 / 2 / 5 / 10 s (the morale update's own ramp to the database's -16),
        # hold -16 to recent_s (45 s), then stand at the database's general_dead -10 to the end.
        def kill(st):
            st.u["men"][0, 0] = 0.0
            st.u["hp_abs"][0, 0] = 0.0
        fall = P.sim["morale"]["lord_fall"]
        assert fall["killed"] and fall["recent_s"] == 45
        st, gap = self.lord_fall_gap(kill, (1, 2, 5, 10, 40, 44, 52, 120))
        assert gap[0] == pytest.approx(-4.4, abs=0.1) and gap[1] == pytest.approx(-7.6, abs=0.1)
        assert gap[2] == pytest.approx(-14.0, abs=1.2) and gap[3] == pytest.approx(-16.0, abs=0.01)
        assert gap[4] == pytest.approx(-16.0, abs=0.01) and gap[5] == pytest.approx(-16.0, abs=0.01)
        assert gap[6] == pytest.approx(-10.0, abs=0.01) and gap[7] == pytest.approx(-10.0, abs=0.01)
        assert float(st.u["dead_s"][0, 0]) == pytest.approx(120.0) and float(st.u["dead_s"][1, 0]) == 0
        assert float(st.lord_dead_s[0, 0]) >= 0 and float(st.lord_dead_s[1, 0]) == -1     # the metrics' "lord lost"

    def test_a_shattered_lord_on_the_field_costs_only_his_aura(self):
        # A shattered (routed) lord is lost as a lord: his fall is timed (lord_dead_s, the metrics) and his aura
        # goes, but while he is on the field his army gets no more (in-game: units out of his aura 0 at every
        # moment; the 75 natural falls of the fair battles were all shatters on the field, about the aura alone).
        def shatter(st):
            st.u["r"][0, 0] = st.u["s"][0, 0] = True
            st.u["morale"][0, 0] = -20.0
        st, gap = self.lord_fall_gap(shatter, (1.5, 10))
        assert float(st.u["men"][0, 0]) > 0 and not bool(st.u["gone"][0, 0])      # lost by shattering alone
        assert float(st.lord_dead_s[0, 0]) == pytest.approx(9.5) and float(st.lord_dead_s[0, 1]) == -1
        assert float(st.lord_dead_s[1, 0]) == -1
        assert gap == pytest.approx([0.0, 0.0]) and float(st.u["dead_s"][0, 0]) == 0
        assert not bool(st.done[0])

    def test_a_lord_routed_off_the_map_costs_16_for_fled_s(self):
        # Routed off the map edge (gone, men left): the database's general_fled_recently -16 for fled_s (120 s,
        # recorded battles), then nothing - no general_dead after it.
        fall = P.sim["morale"]["lord_fall"]
        assert fall["fled"] and fall["fled_s"] == 120
        sides = army([(GENERAL, -900, 0, 90, True), (SPEAR, -300, 0, 90)], [(SPEAR, 300, 0, 270)])
        st = scenario.build([sides, sides], P)
        st.u["r"][0, 0] = st.u["s"][0, 0] = True
        st.u["morale"][0, 0] = -20.0

        def gap():
            return float(st.u["morale"][0, 1] - st.u["morale"][1, 1])
        t = 0.0
        while not bool(st.u["gone"][0, 0]):
            battle.step(st, replay.hold(st), P)
            t += 0.5
            assert t < 120 and gap() == 0.0                    # on the field: nothing beyond the aura
        t_gone = t
        for until, want in ((t_gone + 10, -16.0), (t_gone + 119, -16.0), (t_gone + 140, 0.0)):
            while t < until - 1e-6:
                battle.step(st, replay.hold(st), P)
                t += 0.5
            assert gap() == pytest.approx(want, abs=0.01), (until, gap())
        assert float(st.u["men"][0, 0]) > 0 and float(st.u["dead_s"][0, 0]) == 0

    def test_a_crumbling_factions_routed_lord_counts_as_killed(self):
        # Vampire Counts (sim.json lord_fall.rout_death_s 8): in-game the routed lord crumbles and dies 8-8.5 s
        # into his rout, then his army gets the death's -16 / -10. Other factions: rout_death_s 0 (never).
        vmp = "wh_main_vmp_vampire_counts"
        assert P.static(GENERAL, vmp)["rout_death_s"] == 8 and P.static(GENERAL, "wh_main_emp_empire")["rout_death_s"] == 0

        def rout(st):
            st.u["r"][0, 0] = st.u["s"][0, 0] = True
            st.u["morale"][0, 0] = -20.0
        st, gap = self.lord_fall_gap(rout, (7.5, 9, 20, 60), factions=(vmp, "wh2_main_skv_skaven"))
        assert gap[0] == 0.0 and gap[1] < -3.0
        assert gap[2] == pytest.approx(-16.0, abs=0.01) and gap[3] == pytest.approx(-10.0, abs=0.01)

    def test_orders_move_attack_and_withdraw(self):
        st = scenario.build([army([(SPEAR, -100, 0, 90)], [(SLAVE, 100, 0, 270)])], P)
        H = st.N // 2
        o = replay.hold(st)
        o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, -100.0, 50.0, True
        for _ in range(40):
            battle.step(st, o, P)
        assert float(st.u["z"][0, 0]) == pytest.approx(50.0, abs=1.0) and not bool(st.u["mv"][0, 0])
        o = replay.hold(st)
        o.kind[0, 0], o.target[0, 0], o.run[0, 0] = O.ATTACK, H, True
        for _ in range(200):
            battle.step(st, o, P)
            if st.u["m"][0, 0]:
                break
        assert bool(st.u["m"][0, 0]) and int(st.u["target"][0, 0]) == H
        x0 = float(st.u["x"][0, 0])
        o = replay.hold(st)
        o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.WITHDRAW, -300.0, 0.0, True
        for _ in range(20):                                                   # unchased: it walks out
            battle.step(st, o, P)
        assert not bool(st.u["m"][0, 0]) and float(st.u["x"][0, 0]) < x0 - 15

    @pytest.mark.parametrize("key", [ARCHER, SPEAR])
    def test_a_unit_told_to_move_away_leaves_melee_when_leave_m_is_on(self, key):
        """contact.leave_m: any unit (missile or melee) told to move that far walks out of the fight."""
        moved = {}
        for leave in (0.0, 10.0):
            params = P.with_cal("contact", leave_m=leave, pin_s=0.0, pin_melee_s=0.0)
            st = face_off(key, SPEAR)
            H = st.N // 2
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            battle.step(st, o, params)
            assert bool(st.u["m"][0, 0])
            x0 = float(st.u["x"][0, 0])
            for _ in range(10):
                o = replay.hold(st)
                o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, -200.0, 0.0, True
                battle.step(st, o, params)
            moved[leave] = x0 - float(st.u["x"][0, 0])
        assert moved[0.0] == pytest.approx(0.0, abs=0.5) and moved[10.0] > 10

    def test_a_melee_unit_moving_away_strikes_nobody_and_is_still_struck(self):
        params = P.with_cal("contact", leave_m=10.0)
        st = face_off(SPEAR, SPEAR)
        H = st.N // 2
        o = replay.hold(st)
        o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
        o.kind[0, H], o.target[0, H] = O.ATTACK, 0
        for _ in range(20):
            battle.step(st, o, params)
        assert bool(st.u["m"][0, 0])
        hp0, hpH = float(st.u["hp_abs"][0, 0]), float(st.u["hp_abs"][0, H])
        o = replay.hold(st)
        o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, float(st.u["x"][0, 0]), 200.0, True
        o.kind[0, H], o.target[0, H] = O.ATTACK, 0
        battle.step(st, o, params)
        assert float(st.u["hp_abs"][0, H]) == pytest.approx(hpH) and float(st.u["hp_abs"][0, 0]) < hp0

    def test_a_unit_in_melee_does_not_turn_to_a_flanker(self):
        # A fights B to the east; C comes at A's north flank: A keeps facing B, C hits a flank.
        st = scenario.build([army([(SPEAR, 0, 0, 90)], [(SPEAR, 60, 0, 270), (SPEAR, 0, 80, 180)])], P)
        H = st.N // 2
        for k in range(160):
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H], o.run[0, H] = O.ATTACK, 0, True
            if k > 20:
                o.kind[0, H + 1], o.target[0, H + 1], o.run[0, H + 1] = O.ATTACK, 0, True
            battle.step(st, o, P)
        b = float(st.u["b"][0, 0])
        assert bool(st.u["m"][0, 0]) and abs(((b - 90) + 180) % 360 - 180) < 30
        assert float(st.u["flank_hit"][0, 0]) >= 1


class TestFlanksAndRoles:
    def test_exposed_flanks_lower_morale(self):
        st = face_off(SPEAR, SLAVE)
        z = torch.zeros_like(st.u["r"])
        ctx = {"aura": z, "lord_dead_points": torch.zeros_like(st.u["men"]), "secure": z, "in_melee": z,
               "routing_friends": torch.zeros_like(st.u["men"]), "routing_enemies": torch.zeros_like(st.u["men"]),
               "under_fire": z, "strong_enemy": z, "enemy_near": z | True}
        base = morale.target_points(st.u, ctx, P)
        st.u["lf"][0, 0] = True
        one = morale.target_points(st.u, ctx, P)
        st.u["bf"][0, 0] = True
        two = morale.target_points(st.u, ctx, P)
        assert float((one - base)[0, 0]) == -3 and float((two - base)[0, 0]) == -6

    def test_the_attacker_of_a_recording_comes_from_its_roles(self):
        class Rec:
            own_ai, enemy_role = "net", "attack"
        assert scenario.attacker_of(Rec(), {"own_role": "defend", "enemy_role": "attack"}) == 2
        assert scenario.attacker_of(Rec(), {"own_role": "attack", "enemy_role": "defend"}) == 1
        assert scenario.attacker_of(Rec(), {}) == 2
        Rec.own_ai, Rec.enemy_role = "attack", "?"
        assert scenario.attacker_of(Rec(), {}) == 1


class TestAbilities:
    """Numbers from the ability passports (config/nn/abilities.json); the game's AI fires by its rule,
    the network by order (Orders.ability); passives hold for all."""
    WARLORD = "wh2_main_skv_cha_warlord_0"

    def lords(self):
        # side 1 (the network): our General and spearmen; side 2 (the game's AI): the Warlord touching the General.
        st = scenario.build([army([(GENERAL, 0, 0, 90), (SPEAR, 0, 20, 90)], [(self.WARLORD, 2, 0, 270)])], P)
        u = st.u
        pw = geometry.pairwise(u, 1.5)
        same = u["side"][:, :, None] == u["side"][:, None, :]
        standing = u["men"] > 0
        engaged = torch.zeros_like(standing)
        return st, st.N // 2, pw["dist"], same, standing, engaged

    def test_the_slots_and_the_numbers_are_the_passports(self):
        from tools.nn.sim import abilities
        names = abilities.keys(P)
        slots = lambda key: [names[i] if i >= 0 else "" for i in abilities.slots_of(P, key)]
        assert slots(GENERAL) == ["wh_main_character_abilities_foe_seeker",
                                  "wh_main_character_abilities_stand_your_ground", "wh_main_lord_passive_hold_the_line"]
        assert slots(self.WARLORD) == ["wh2_main_character_abilities_verminous_valour",
                                       "wh_main_character_abilities_deadly_onslaught", "wh_main_character_abilities_rally"]
        assert slots(SPEAR) == ["", "", ""]
        r = dict(zip(abilities.COLS, abilities.row(P, "wh_main_character_abilities_stand_your_ground")))
        assert (r["active_s"], r["recharge_s"], r["range_m"], r["self_cast"], r["modelled"]) == (18, 90, 35, 1, 1)
        assert (r["self_defence"], r["friends_defence"], r["friends_leadership"], r["enemies_defence"]) == (24, 24, 16, 0)
        r = dict(zip(abilities.COLS, abilities.row(P, "wh_main_character_abilities_foe_seeker")))
        assert (r["self_speed"], r["self_charge_speed"], r["friends_speed"]) == (1.25, 1.25, 1.0)
        r = dict(zip(abilities.COLS, abilities.row(P, "wh3_main_unit_passive_single_entity")))
        assert r["passive"] == 1 and r["modelled"] == 0          # innate: tools/nn/sim/effects.py lays it on

    def test_the_game_ai_fires_by_its_rule_and_passives_hold_for_all(self):
        from tools.nn.sim import abilities
        st, H, dist, same, standing, engaged = self.lords()
        u = st.u
        assert not bool(u["ai"][0, 0]) and bool(u["ai"][0, H])
        engaged[0, 0] = engaged[0, H] = True
        dmg, ap, defence, run = (float(u[k][0, i]) for k, i in (("damage", H), ("ap_damage", H), ("defence", 1),
                                                                    ("run", H)))
        engaged[0, 0] = engaged[0, H] = False
        abilities.apply(u, P, 0.5, standing, engaged, dist, same)         # touching but not in melee: nothing
        assert float(u["ab0_on"][0, H]) == 0
        engaged[0, 0] = engaged[0, H] = True
        old = abilities.apply(u, P, 0.5, standing, engaged, dist, same)
        assert float(u["run"][0, H]) == pytest.approx(1.25 * run)         # Verminous Valour (at contact)
        assert float(u["ab0_on"][0, H]) == 17 and float(u["ab0_cd"][0, H]) == 17 + 60
        assert float(u["damage"][0, H]) == pytest.approx(dmg)              # Deadly Onslaught: never by the AI
        assert float(u["ap_damage"][0, H]) == pytest.approx(ap) and float(u["ab1_cd"][0, H]) == 0
        assert float(u["ab2_cd"][0, H]) == 0                               # Rally: no 2 friends within 35 m
        assert float(u["ab0_on"][0, 0]) == 0 and float(u["ab1_on"][0, 0]) == 0   # our General: no order, no use
        assert float(u["defence"][0, 1]) == defence                        # Hold the Line: an innate effect
        abilities.restore(u, old)
        assert float(u["run"][0, H]) == run
        abilities.apply(u, P, 17.0, standing, engaged, dist, same)  # 17 s later: over, recharging
        assert float(u["ab0_on"][0, H]) == 0 and float(u["ab0_cd"][0, H]) == 60

    def test_the_game_ai_stands_its_ground_only_with_friends_near(self):
        # Stand Your Ground: the game's AI fires it in melee with at least one standing friend within 35 m
        # (never in a lone duel); Rally with at least two (config/nn/sim.json abilities.friends_min).
        from tools.nn.sim import abilities
        assert P.sim["abilities"]["friends_min"] == {"wh_main_character_abilities_stand_your_ground": 1,
                                                     "wh_main_character_abilities_rally": 2}
        for friend_z, fired in ((20, True), (50, False)):
            st = scenario.build([army([(self.WARLORD, 2, 0, 270)], [(GENERAL, 0, 0, 90), (SPEAR, 0, friend_z, 90)])], P)
            u = st.u
            H = st.N // 2
            abilities.set_rule(u, torch.tensor([[False, True]]))
            pw = geometry.pairwise(u, 1.5)
            same = u["side"][:, :, None] == u["side"][:, None, :]
            standing = u["men"] > 0
            engaged = torch.zeros_like(standing)
            engaged[0, 0] = engaged[0, H] = True
            defence = float(u["defence"][0, H])
            abilities.apply(u, P, 0.5, standing, engaged, pw["dist"], same)
            assert (float(u["ab1_on"][0, H]) == 18) == fired, friend_z          # slot 1: Stand Your Ground
            assert (float(u["defence"][0, H]) == defence + 24) == fired
            assert float(u["ab0_on"][0, H]) == 25                                # Foe-Seeker: in melee, always

    def test_the_network_fires_a_ready_self_cast_ability_by_order(self):
        from tools.nn.sim import abilities
        st, H, dist, same, standing, engaged = self.lords()
        u = st.u
        defence = float(u["defence"][0, 1])
        use = torch.full_like(u["ab0"], -1)
        use[0, 0] = 1                                                      # Stand Your Ground
        old = abilities.apply(u, P, 0.5, standing, engaged, dist, same, use)
        assert float(u["ab1_on"][0, 0]) == 18 and float(u["ab1_cd"][0, 0]) == 18 + 90
        assert float(u["defence"][0, 1]) == defence + 24                  # (Hold the Line: effects.py)
        assert float(u["defence"][0, H]) == float(old["defence"][0, H])   # friends only, not the enemy
        abilities.restore(u, old)
        abilities.apply(u, P, 0.5, standing, engaged, dist, same, use)     # active: a new order does nothing
        assert float(u["ab1_on"][0, 0]) == pytest.approx(17.5)
        use[0, 0] = 2                                                      # Hold the Line is passive
        cd = [float(u[f"ab{k}_cd"][0, 0]) for k in range(3)]
        abilities.apply(u, P, 0.5, standing, engaged, dist, same, use)
        assert float(u["ab2_on"][0, 0]) == 0 and float(u["ab2_cd"][0, 0]) == 0
        assert [float(u[f"ab{k}_cd"][0, 0]) for k in range(2)] == [cd[0], pytest.approx(cd[1] - 0.5)]

    def _cast_and_move(self, lord, friend, slot, moves, steps):
        """The lord (at 0, 0) casts the ability in `slot` by order at step 0; friends A (30 m) and B (80 m); at the
        steps in moves {step: (role, z)} a friend jumps to z. Returns per step (A has it, B has it, the lord routs)
        by the friends' leadership bonus."""
        from tools.nn.sim import abilities
        emp = lord == GENERAL
        factions = ("wh_main_emp_empire", "wh2_main_skv_skaven") if emp else ("wh2_main_skv_skaven", "wh_main_emp_empire")
        st = scenario.build([army([(lord, 0, 0, 90, True), (friend, 0, 30, 90), (friend, 0, 80, 90)],
                                  [(SLAVE if emp else SPEAR, 300, 0, 270)], factions=factions)], P)
        u = st.u
        H = 0
        same = u["side"][:, :, None] == u["side"][:, None, :]
        standing = u["men"] > 0
        engaged = torch.zeros_like(standing)
        abilities.set_rule(u, torch.tensor([[False, False]]))
        mb = [float(u["morale_bonus"][0, H + i]) for i in (1, 2)]
        out = []
        for k in range(steps):
            for role, z in moves.get(k, ()):
                u["z"][0, H + role] = z
            dist = geometry.pairwise(u, 1.5)["dist"]
            use = torch.full_like(u["ab0"], -1)
            if k == 0:
                use[0, H] = slot
            old = abilities.apply(u, P, 0.5, standing & ~u["r"], engaged, dist, same, use)
            out.append(tuple(float(u["morale_bonus"][0, H + i]) == mb[i - 1] + 16 for i in (1, 2)))
            abilities.restore(u, old)
        return out

    def test_stand_your_ground_is_laid_at_the_cast_and_kept_wherever_they_go(self):
        """The database's update_targets 0 (build/effects/spec.md P2; the recordings: of those that got it, 494 s
        on and 4 off beyond 40 m; of those that came in later, 0 on and 768 off)."""
        assert not P.abilities["wh_main_character_abilities_stand_your_ground"]["update_targets"]
        # A (30 m) leaves to 60 m at 2 s, B (80 m) comes to 20 m at 5 s
        got = self._cast_and_move(GENERAL, SPEAR, 1, {4: [(1, 60.0)], 10: [(2, 20.0)]}, 40)
        assert all(a for a, _ in got[:36]) and not any(a for a, _ in got[36:])     # 18 s: 36 steps of 0.5 s
        assert not any(b for _, b in got)

    def test_rally_is_an_aura_that_follows_the_lord(self):
        """The database's update_targets 1: who is within 35 m each step (the recordings: of those that came in later
        917 s on; of those that left beyond 40 m 784 s off)."""
        assert P.abilities["wh_main_character_abilities_rally"]["update_targets"]
        got = self._cast_and_move(self.WARLORD, "wh2_main_skv_inf_clanrats_1", 2, {4: [(1, 60.0)], 10: [(2, 20.0)]},
                                  28)
        assert all(a for a, _ in got[:4]) and not any(a for a, _ in got[4:])
        assert not any(b for _, b in got[:10]) and all(b for _, b in got[10:28])   # 14 s from the cast

    def test_a_cast_ability_goes_on_when_its_lord_routs(self):
        from tools.nn.sim import abilities
        st, H, dist, same, standing, engaged = self.lords()
        u = st.u
        defence = float(u["defence"][0, 1])
        use = torch.full_like(u["ab0"], -1)
        use[0, 0] = 1                                                      # Stand Your Ground
        abilities.restore(u, abilities.apply(u, P, 0.5, standing, engaged, dist, same, use))
        u["r"][0, 0] = True                                                # the General routs: it goes on
        old = abilities.apply(u, P, 0.5, standing & ~u["r"], engaged, dist, same)
        assert float(u["defence"][0, 0]) == float(old["defence"][0, 0]) + 24
        assert float(u["defence"][0, 1]) == defence + 24
        abilities.restore(u, old)
        u["ab1_cd"][0, 0] = 0.0
        u["ab1_on"][0, 0] = 0.0
        use[0, 0] = 1                                                      # a routing lord casts nothing
        abilities.apply(u, P, 0.5, standing & ~u["r"], engaged, dist, same, use)
        assert float(u["ab1_on"][0, 0]) == 0

    def test_a_network_side_fires_by_order_only_whichever_side_it_plays(self):
        from tools.nn.sim import abilities
        st = scenario.build([army([(GENERAL, 0, 0, 90)], [(self.WARLORD, 2, 0, 270)])] * 2, P)
        H = st.N // 2
        assert st.u["ai"][:, H].tolist() == [True, True]              # the default: side 2 by the rule
        abilities.set_rule(st.u, torch.tensor([[True, False], [False, False]]))
        assert st.u["ai"][:, 0].tolist() == [True, False] and st.u["ai"][:, H].tolist() == [False, False]
        assert not bool(st.u["ai"][st.u["side"] == 0].any())
        o = replay.hold(st)
        for _ in range(4):                                             # in melee, enemy near: no rule fires
            battle.step(st, o, P)
        assert float(st.u["ab0_cd"][1, H]) == 0 and float(st.u["ab1_cd"][1, H]) == 0
        assert float(st.u["ab0_cd"][0, 0]) > 0                         # battle 0: side 1 by the rule

    def test_an_order_reaches_the_step_and_old_orders_hold_no_ability(self):
        st, H, *_ = self.lords()
        o = replay.hold(st)
        assert torch.all(o.ability == -1)
        plain = O.Orders(kind=o.kind, x=o.x, z=o.z, target=o.target, run=o.run)       # the five-field contract
        assert torch.all(plain.ability == -1) and torch.all(O.merge(plain, o, o.kind == 0).ability == -1)
        o.ability[0, 0] = 0                                                # Foe Seeker
        battle.step(st, o, P)
        assert float(st.u["ab0_on"][0, 0]) == 25
        assert float(st.observation()["ab0_on"][0, 0]) == 25
        o.ability[0, 0] = 3
        with pytest.raises(ValueError):
            O.check(o, st.N)


# --- the second Empire wave (02.10.2026): unbreakable, frenzy and the penitent, direct fire ---

FLAG, GREAT, MILITIA = ("wh_dlc04_emp_inf_flagellants_0", "wh_main_emp_inf_greatswords",
                        "wh_dlc04_emp_inf_free_company_militia_0")


class TestSecondWave:
    def test_an_unbreakable_unit_never_routs_nor_wavers(self):
        st = scenario.build([army([(SPEAR, -100, 0, 90), (FLAG, -100, 60, 90)], [(SLAVE, 300, 0, 270)])], P)
        assert bool(st.u["unbreakable"][0, 1]) and not bool(st.u["unbreakable"][0, 0])
        st.u["morale"][0, :2] = -20.0                     # crushed: the spearmen rout, the flagellants not
        for _ in range(4):
            battle.step(st, replay.hold(st), P)
        assert bool(st.u["r"][0, 0])
        assert not bool(st.u["r"][0, 1]) and not bool(st.u["w"][0, 1]) and float(st.u["morale"][0, 1]) >= 100

    def test_flagellants_surrounded_fight_on_and_the_penitent_fires_in_melee(self):
        st = scenario.build([army([(FLAG, 0, 0, 90)], [(CLANRAT, 40, 0, 270), (CLANRAT, 0, 50, 180),
                                                       (CLANRAT, -40, 0, 90)])], P)
        H = st.N // 2
        slot = abilities.slot_keys(FLAG, P.units, P.abilities).index("wh_dlc04_unit_passive_strength_of_the_penitent")
        fired = False
        for _ in range(600):
            o = replay.hold(st)
            for k in range(3):
                o.kind[0, H + k], o.target[0, H + k], o.run[0, H + k] = O.ATTACK, 0, True
            battle.step(st, o, P)
            assert not bool(st.u["r"][0, 0])
            fired = fired or float(st.u[f"ab{slot}_on"][0, 0]) > 0
            if float(st.u["men"][0, 0]) <= 0:
                break
        assert fired and float(st.u["hp"][0, 0]) < 0.3

    def test_frenzy_and_the_penitent_lay_their_effects_by_their_conditions(self):
        st = scenario.build([army([(FLAG, 0, 0, 90)], [(CLANRAT, 12, 0, 270)])], P)
        u = st.u
        pw = geometry.pairwise(u, P.sim["formation"]["spacing_m"])
        same = u["side"][:, :, None] == u["side"][:, None, :]
        standing = u["side"] > 0
        base_att, base_def = float(u["attack"][0, 0]), float(u["defence"][0, 0])
        u["morale"][:] = u["leadership"]
        u["taken"][0, 0], u["dealt"][0, 0] = 100.0, 500.0          # winning: it fires anyway, in melee
        engaged = standing.clone()
        assert float(u["fxt0_cd"][0, 0]) == 3                     # the penitent's initial recharge (the database)
        u["fxt0_cd"][0, 0] = 0.0                                   # 3 s into the battle
        old = effects.apply(u, P, 0.5, standing, engaged, pw["dist"], same)
        assert float(u["attack"][0, 0]) == base_att + 10                          # frenzy
        assert float(u["defence"][0, 0]) == base_def + 14                         # the penitent
        assert float(u["resist_physical"][0, 0]) == pytest.approx(0.15)
        slot = abilities.slot_keys(FLAG, P.units, P.abilities).index("wh_dlc04_unit_passive_strength_of_the_penitent")
        assert float(u[f"ab{slot}_on"][0, 0]) == 20 and float(u["fxt0_on"][0, 0]) == 20   # the bar shows it
        effects.restore(u, old)
        # out of melee the penitent ends at once (and recharges); frenzy stays while morale holds
        old = effects.apply(u, P, 0.5, standing, torch.zeros_like(engaged), pw["dist"], same)
        assert float(u["defence"][0, 0]) == base_def and float(u["attack"][0, 0]) == base_att + 10
        assert float(u["fxt0_on"][0, 0]) == 0 and float(u["fxt0_cd"][0, 0]) == 3
        effects.restore(u, old)
        u["morale"][0, 0] = 0.4 * float(u["leadership"][0, 0])                    # below half: frenzy off
        old = effects.apply(u, P, 0.5, standing, torch.zeros_like(engaged), pw["dist"], same)
        assert float(u["attack"][0, 0]) == base_att
        effects.restore(u, old)
        # the network cannot order them: they are the game's own
        assert not any(P.abilities[k]["self_cast"] for k in P.units[FLAG]["abilities"])

    def test_the_penitent_fires_in_melee_and_recharges_only_while_losing(self):
        """Strength of the Penitent (the database: 20 s, recharge 3 s in the context losing_melee_combat, initial 3 s,
        off out of melee; build/effects/spec.md P1): fired by itself whenever ready in melee; after a fire its 3 s
        stand while the unit wins its melee (the morale's balance, u cmb > 0) and run otherwise - losing, even, or
        out of melee (the probe T-E1 and the recordings); the initial 3 s always."""
        st = scenario.build([army([(FLAG, 0, 0, 90)], [(CLANRAT, 12, 0, 270)])], P)
        u = st.u
        pw = geometry.pairwise(u, P.sim["formation"]["spacing_m"])
        same = u["side"][:, :, None] == u["side"][:, None, :]
        standing = u["side"] > 0
        melee, apart = standing.clone(), torch.zeros_like(standing)
        base = float(u["defence"][0, 0])

        def step(engaged, cmb=0.0):
            u["cmb"][0, 0] = cmb
            old = effects.apply(u, P, 0.5, standing, engaged, pw["dist"], same)
            on = float(u["defence"][0, 0]) == base + 14
            effects.restore(u, old)
            assert on == (float(u["fxt0_on"][0, 0]) > 0)
            return on
        # the initial 3 s run out of melee and without losing; never on out of melee
        assert not any(step(apart) for _ in range(10)) and float(u["fxt0_cd"][0, 0]) == 0
        assert step(melee)                                      # the first step in melee, no losses: on
        assert all(step(melee) for _ in range(39))              # 20 s in all
        assert not step(melee) and float(u["fxt0_cd"][0, 0]) == 3
        assert not any(step(melee, cmb=3.0) for _ in range(20))  # winning: the recharge stands
        assert float(u["fxt0_cd"][0, 0]) == 3
        assert not any(step(melee, cmb=-3.0) for _ in range(5))  # losing: 3 s to run (6 steps of 0.5 s)
        assert step(melee, cmb=-3.0)
        assert all(step(melee, cmb=0.0) for _ in range(39))      # 20 s
        assert not any(step(melee, cmb=0.0) for _ in range(6))   # even: the 3 s run too
        assert step(melee, cmb=0.0)
        assert not step(apart, cmb=3.0)                           # out of melee: off at once
        assert float(u["fxt0_cd"][0, 0]) == 3
        assert not any(step(apart) for _ in range(20))            # never on out of melee, the 3 s run there
        assert float(u["fxt0_cd"][0, 0]) == 0
        assert step(melee, cmb=3.0)                               # on at the next contact

    def _spent(self, shooter, side1_more=(), enemies=((SPEAR, -20, 0, 270),), steps=40, orders=None, params=P):
        st = scenario.build([army([(shooter, -100, 0, 90), *side1_more], list(enemies))], params)
        for _ in range(steps):
            o = replay.hold(st) if orders is None else orders(st)
            battle.step(st, o, params)
        return float(st.u["ammo0"][0, 0] - st.u["a"][0, 0]), st

    def test_direct_fire_is_blocked_by_a_friendly_unit_in_between_arcing_fire_is_not(self):
        """The line's geometry across (a straight line, missile.arc_los off: whether the arc clears the friends is
        test_direct_fire_past_friends_follows_the_bullets_arc)."""
        screen = ((SPEAR, -60, 0, 90),)
        line = P.with_cal("missile", arc_los=None)
        open_m, _ = self._spent(MILITIA, params=line)
        blocked_m, st = self._spent(MILITIA, screen, params=line)
        assert bool(st.u["direct"][0, 0]) and open_m > 0 and blocked_m == 0
        open_a, _ = self._spent(ARCHER)
        over_a, st = self._spent(ARCHER, screen)
        assert not bool(st.u["direct"][0, 0]) and open_a > 0 and over_a == pytest.approx(open_a, rel=0.05)
        # a screen half way, off to one side, blocks about half of the men (their lines converge on the
        # target's centre: half way a line is at half its offset)
        part_m, _ = self._spent(MILITIA, ((SPEAR, -60, 16, 90),), params=line)
        assert 0.2 * open_m < part_m < 0.8 * open_m

    def test_direct_fire_past_friends_follows_the_bullets_arc(self):
        """missile.arc_los (probes lofthresh, lofheight, lofab, hglof; muzzle 1.5 m, aim point 0.8 m measured): handgunners
        with our spearmen 40 m ahead hold fire at skavenslaves ~68 m off (the arc too low over the friends' heads; the
        game 0.04-0.06 of the men) and fire at ~90 m (the game: the full rate, the friends catching 0.22-0.27 hits a
        shot); without arc_los (a straight line) they hold at both; arcing fire (archers) is not checked at all."""
        hg = "wh_main_emp_inf_handgunners"
        screen = ((SPEAR, -60, 0, 90),)

        def fire(d, params=P, key=hg):
            st = scenario.build([army([(key, -100, 0, 90), *screen], [(SLAVES, -100 + d, 0, 270)])], params)
            hp0 = float(st.u["hp_abs"][0, 1])
            for _ in range(80):
                battle.step(st, replay.hold(st), params)
            return float(st.u["ammo0"][0, 0] - st.u["a"][0, 0]), hp0 - float(st.u["hp_abs"][0, 1]), st
        near, f_near, _ = fire(68.0)
        far, f_far, st = fire(90.0)
        men = float(st.u["men0"][0, 0])
        assert near < 0.25 * far and far >= 0.8 * 3 * men                    # ~3 volleys in 40 s at the full rate
        per_hit = 17 + 5 * (1 - 0.75 * 0.30)                                  # the rule's HP a bullet takes from spearmen
        assert 0.1 < f_far / far / per_hit < 0.4                              # the friends catch part of the bullets
        line = P.with_cal("missile", arc_los=None)
        assert fire(90.0, line)[0] == 0
        arch, _, _ = fire(90.0, key=ARCHER)
        arch_line, _, _ = fire(90.0, line, key=ARCHER)
        assert arch == pytest.approx(arch_line) and arch > 0

    def test_a_blocked_direct_fire_unit_shoots_another_target_it_can_see(self):
        _, st = self._spent(MILITIA, ((SPEAR, -60, 0, 90),), enemies=((SPEAR, -20, 0, 270), (SLAVE, -30, 50, 270)),
                            steps=30, params=P.with_cal("missile", arc_los=None))
        H = st.N // 2
        assert int(st.u["target"][0, 0]) == H + 1 and bool(st.u["fire"][0, 0])

    def test_fire_whilst_moving_only_at_targets_ahead(self):
        """mounted_fire_move: shoots on the move, but only within missile.move_fire_arc_deg of the way it walks."""
        def walk(x):
            def orders(st):
                o = replay.hold(st)
                o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, x, 0.0, False
                return o
            return orders
        toward, st = self._spent(MILITIA, steps=30, orders=walk(-40.0))
        assert bool(st.u["fire_move"][0, 0]) and toward > 0 and bool(st.u["mv"][0, 0])
        away, st = self._spent(MILITIA, steps=30, orders=walk(-300.0))
        assert away == 0 and bool(st.u["mv"][0, 0])
        moving_a, _ = self._spent(ARCHER, steps=30, orders=walk(-40.0))
        assert moving_a == 0

    def test_throwing_stars_shoot_behind_on_the_move(self):
        """missile.move_fire_own_arc (the probe starsmove, 07.10.2026): on the move a unit shoots within the wider of
        move_fire_arc_deg and its own fire arc - the Night Runners' throwing stars (360) at a target behind them while
        walking away; the militia (35 each side) still hold fire walking away; off: the stars hold fire too."""
        stars = "wh2_main_skv_inf_night_runners_0"
        near = ((SPEAR, -55, 0, 270),)

        def walk(x):
            def orders(st):
                o = replay.hold(st)
                o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, x, 0.0, False
                return o
            return orders
        away, st = self._spent(stars, enemies=near, steps=30, orders=walk(-300.0))
        assert float(st.u["arc"][0, 0]) == 180 and bool(st.u["mv"][0, 0]) and away > 0
        mil, _ = self._spent(MILITIA, enemies=near, steps=30, orders=walk(-300.0))
        assert mil == 0
        st = scenario.build([army([(stars, -100, 0, 90)], list(near))], P.with_cal("missile", move_fire_own_arc=0))
        params = P.with_cal("missile", move_fire_own_arc=0)
        for _ in range(30):
            battle.step(st, walk(-300.0)(st), params)
        assert float(st.u["ammo0"][0, 0] - st.u["a"][0, 0]) == 0

    def test_a_new_order_makes_a_shooter_aim_again(self):
        """missile.aim_reset_on_order: another kind or another attack target restarts the aim; the same order
        given again does not."""
        st = scenario.build([army([(ARCHER, -100, 0, 90)], [(SLAVE, 0, -20, 270), (SLAVE, 0, 40, 270)])], P)
        H = st.N // 2
        for _ in range(12):
            battle.step(st, replay.hold(st), P)
        aim = float(st.u["aim"][0, 0])
        assert aim > float(st.u["aim_s"][0, 0])
        battle.step(st, replay.hold(st), P)                                        # HOLD again: aims on
        assert float(st.u["aim"][0, 0]) == pytest.approx(aim + P.dt)
        for params, reset in ((P, True), (P.with_cal("missile", aim_reset_on_order=0), False)):
            s = st.clone()
            o = replay.hold(s)
            o.kind[0, 0], o.target[0, 0] = O.ATTACK, H + 1
            battle.step(s, o, params)
            assert (float(s.u["aim"][0, 0]) == pytest.approx(P.dt)) == reset

    def test_greatswords_cut_through_armour(self):
        """Armour-piercing greatswords (10 + 25, bonus 14 v infantry) beat armoured stormvermin faster than
        swordsmen (21 + 7) of about the same attack do."""
        lost = {}
        for key in (GREAT, SWORD):
            st = face_off(key, "wh2_main_skv_inf_stormvermin_0")
            H = st.N // 2
            for _ in range(60):
                o = replay.hold(st)
                o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
                o.kind[0, H], o.target[0, H] = O.ATTACK, 0
                battle.step(st, o, P)
            lost[key] = 1 - float(st.u["hp"][0, H])
        assert lost[GREAT] > 1.5 * lost[SWORD]


# --- innate effects (tools/nn/sim/effects.py, config/nn/effects.json): one mechanism, each effect on and off ---

SLAVES, CLANRAT_SHIELD, RUNNERS = ("wh2_main_skv_inf_skavenslaves_0", "wh2_main_skv_inf_clanrats_1",
                                   "wh2_main_skv_inf_night_runners_1")
WARLORD = "wh2_main_skv_cha_warlord_0"


def innate(st, engaged=None, params=P):
    """effects.apply on a built state as it stands; returns (old, the fields as the step sees them)."""
    u = st.u
    pw = geometry.pairwise(u, params.sim["formation"]["spacing_m"])
    same = u["side"][:, :, None] == u["side"][:, None, :]
    standing = (u["side"] > 0) & ~u["r"]
    engaged = torch.zeros_like(standing) if engaged is None else engaged
    old = effects.apply(u, params, 0.5, standing, engaged, pw["dist"], same)
    after = {k: u[k].clone() for k in old}
    effects.restore(u, old)
    return old, after


def bit(key):
    return 1 << effects.index(P)[key]


class TestInnateEffects:
    SIN, SCURRY = "wh2_main_unit_passive_strength_in_numbers", "wh2_main_unit_passive_scurry_away"

    def test_units_own_the_effects_the_catalogue_links(self):
        st = scenario.build([army([(FLAG, -50, 0, 90), (SPEAR, -50, 40, 90)], [(SLAVES, 50, 0, 270),
                                                                              (RUNNERS, 50, 40, 270)])], P)
        H = st.N // 2
        fx = [int(x) for x in st.u["fx"][0]]
        assert fx[0] & bit("unbreakable") and fx[0] & bit("wh_main_unit_passive_frenzy")
        assert fx[H] & bit("expendable") and fx[H] & bit(self.SIN) and fx[H] & bit(self.SCURRY)
        assert fx[H + 1] & bit("guerrilla_deploy") and not fx[H + 1] & bit("expendable")
        assert bool(st.u["unbreakable"][0, 0]) and bool(st.u["expendable"][0, H]) and bool(st.u["reflect"][0, 1])
        assert int(st.u["fxt0"][0, 0]) == effects.index(P)["wh_dlc04_unit_passive_strength_of_the_penitent"]
        assert int(st.u["fxt0"][0, 1]) == -1

    def test_strength_in_numbers_holds_above_half_health(self):
        st = scenario.build([army([(SPEAR, -100, 0, 90)], [(CLANRAT_SHIELD, 100, 0, 270)])], P)
        H = st.N // 2
        u = st.u
        run, defence, bonus = float(u["run"][0, H]), float(u["defence"][0, H]), float(u["morale_bonus"][0, H])
        _, a = innate(st)
        assert float(a["run"][0, H]) == pytest.approx(0.9 * run) and float(a["defence"][0, H]) == defence + 8
        assert float(a["morale_bonus"][0, H]) == bonus + 6
        assert int(st.u["fx_on"][0, H]) & bit(self.SIN)
        u["hp"][0, H] = 0.45                                       # below half of the start: off
        _, a = innate(st)
        assert float(a["run"][0, H]) == run and float(a["defence"][0, H]) == defence
        assert float(a["morale_bonus"][0, H]) == bonus and not int(st.u["fx_on"][0, H]) & bit(self.SIN)
        assert float(a["run"][0, 0]) == float(u["run"][0, 0])      # the Empire's spearmen own neither

    def test_scurry_away_speeds_wavering_and_routing_units(self):
        st = scenario.build([army([(SPEAR, -100, 0, 90)], [(SLAVES, 100, 0, 270)])], P)
        H = st.N // 2
        u = st.u
        u["hp"][0, H] = 0.3                                        # Strength in Numbers off
        run = float(u["run"][0, H])
        _, a = innate(st)
        assert float(a["run"][0, H]) == run                        # steady: off
        u["w"][0, H] = True
        _, a = innate(st)
        assert float(a["run"][0, H]) == pytest.approx(1.1 * run)   # wavering
        u["w"][0, H], u["r"][0, H] = False, True
        _, a = innate(st)
        assert float(a["run"][0, H]) == pytest.approx(1.1 * run)   # routing (alive): a faster rout
        u["hp"][0, H] = 0.8
        _, a = innate(st)
        assert float(a["run"][0, H]) == pytest.approx(0.99 * run)  # both: x1.1 x0.9

    def test_a_routing_skaven_unit_runs_faster_than_an_empire_one(self):
        # (a standing unit a side far away keeps the battle going)
        st = scenario.build([army([(SPEAR, -300, 0, 90), (SPEAR, -300, 700, 90)],
                                  [(SLAVE, 300, 0, 270), (SLAVE, 300, 700, 270)])], P)
        H = st.N // 2
        for i in (0, H):
            st.u["r"][0, i], st.u["morale"][0, i], st.u["hp"][0, i] = True, -40.0, 0.4
            st.u["hp_abs"][0, i] = 0.4 * st.u["hp0"][0, i]
            st.u["rout_count"][0, i], st.u["rally_s"][0, i] = 1.0, 0.0
        for _ in range(20):
            battle.step(st, replay.hold(st), P)
        share = {}
        for i, key in ((0, SPEAR), (H, SLAVE)):
            assert bool(st.u["r"][0, i])
            v = math.hypot(float(st.u["vx"][0, i]), float(st.u["vz"][0, i]))
            share[i] = v / float(P.units[key]["speed"]["run"])
        rout = P.sim["morale"]["rout_speed"]
        assert share[0] == pytest.approx(rout, rel=0.01) and share[H] == pytest.approx(1.1 * rout, rel=0.01)

    def test_single_entity_weakens_a_lord_below_a_quarter_of_his_health_unless_left_out(self):
        """Single Entity (Wounds, on since 06.10.2026: the recordings show it 5-6 s after a lord falls below 25 %):
        on, the effect's initial cooldown (5 s) after health first falls below a quarter, speed x0.9 and damage x0.8;
        sim.json effects.off can still leave it out."""
        import dataclasses
        import json
        sim = json.loads(json.dumps(P.sim))
        assert "wh3_main_unit_passive_single_entity" not in sim["effects"]["off"]
        sim["effects"]["off"] = ["wh3_main_unit_passive_single_entity"]
        Q = dataclasses.replace(P, sim=sim)
        for params, on in ((P, True), (Q, False)):
            st = scenario.build([army([(GENERAL, -100, 0, 90, True)], [(WARLORD, 100, 0, 270, True)])], params)
            H = st.N // 2
            u = st.u
            dmg, run = float(u["damage"][0, H]), float(u["run"][0, H])
            _, a = innate(st, params=params)
            assert float(a["damage"][0, H]) == dmg
            u["hp"][0, H] = 0.2
            u["low_s"][0, H] = 4.5                       # below a quarter for 4.5 s: not yet
            _, a = innate(st, params=params)
            assert float(a["damage"][0, H]) == dmg
            u["low_s"][0, H] = 5.0
            _, a = innate(st, params=params)
            k = (0.8, 0.9) if on else (1.0, 1.0)
            assert float(a["damage"][0, H]) == pytest.approx(k[0] * dmg) and float(a["run"][0, H]) == pytest.approx(k[1] * run)
            assert int(st.u["fx_on"][0, H]) & bit("wh3_main_unit_passive_single_entity")     # the network sees it

    def test_hold_the_line_reaches_friends_in_range_from_a_living_lord_routing_or_not(self):
        """From a routing General and on routing friends too (the recordings: 1312 of 1312 s and 2017 of 2017 s,
        build/effects/spec.md 3.4); not from a dead one."""
        st = scenario.build([army([(GENERAL, 0, 0, 90, True), (SPEAR, 0, 30, 90), (SPEAR, 0, 120, 90),
                                   (SPEAR, 20, 0, 90)], [(SLAVE, 200, 0, 270)])], P)
        H = st.N // 2
        u = st.u
        d = [float(u["defence"][0, i]) for i in (1, 2, 3, H)]
        _, a = innate(st)
        assert [float(a["defence"][0, i]) for i in (1, 2, 3, H)] == [d[0] + 5, d[1], d[2] + 5, d[3] + 8]  # (SiN)
        assert float(a["morale_bonus"][0, 1]) == float(u["morale_bonus"][0, 1]) + 4
        u["r"][0, 0] = True                                        # a routing General still holds the line
        u["r"][0, 3] = True                                        # and a routing friend gets it
        _, a = innate(st)
        assert float(a["defence"][0, 1]) == d[0] + 5 and float(a["defence"][0, 3]) == d[2] + 5
        u["men"][0, 0] = 0                                         # a dead one does not
        _, a = innate(st)
        assert float(a["defence"][0, 1]) == d[0]

    def test_an_effect_the_simulator_lacks_counts_as_on_but_changes_nothing(self):
        st = scenario.build([army([(SPEAR, -100, 0, 90)], [(RUNNERS, 100, 0, 270)])], P)
        H = st.N // 2
        _, a = innate(st)
        on = int(st.u["fx_on"][0, H])
        assert on & bit("guerrilla_deploy") and on & bit("hide_forest")
        assert not P.effects["effects"]["guerrilla_deploy"]["modelled"]
        assert float(a["attack"][0, H]) == float(st.u["attack"][0, H])

    def test_a_new_effect_in_the_catalogue_works_without_code(self):
        """Perfect Vigour (fatigue_immune) given to the spearmen by the catalogue alone: they never tire."""
        import dataclasses
        import json
        from tools.nn import effects as catalogue
        doc = json.loads(json.dumps(P.effects))
        doc["effects"]["fatigue_immune"] = catalogue.attribute_effect("fatigue_immune")
        doc["order"].append("fatigue_immune")
        doc["units"][SPEAR] = doc["units"][SPEAR] + ["fatigue_immune"]
        Q = dataclasses.replace(P, effects=doc)
        st = scenario.build([army([(SPEAR, -100, 0, 90), (SPEAR, -100, 60, 90)], [(SLAVE, 300, 0, 270)])], Q)
        st.u["fx"][0, 1] = int(st.u["fx"][0, 1]) & ~bit_of(Q, "fatigue_immune")   # the second has it not
        for _ in range(40):
            o = replay.hold(st)
            o.kind[0, :2], o.x[0, :2], o.z[0, :2], o.run[0, :2] = O.MOVE, 200.0, 0.0, True
            battle.step(st, o, Q)
        assert float(st.u["fatigue"][0, 0]) == 0 and float(st.u["fatigue"][0, 1]) > 0


def bit_of(params, key):
    return 1 << effects.index(params)[key]


def calibrated_fatigue():
    import copy
    p = copy.deepcopy(P)
    p.sim["fatigue"]["calibration"]["on"] = True
    return p


def melee_activity(n, **flags):
    activity = {k: torch.zeros(n, dtype=torch.bool) for k in
                ("idle", "walking", "running", "shooting", "charging", "attack", "single")}
    activity.update(melee=torch.ones(n, dtype=torch.bool), active=torch.ones(n, dtype=torch.bool))
    activity.update({k: torch.tensor(v) for k, v in flags.items()})
    return activity


def test_melee_tires_only_under_an_attack_order():
    # single entity the database's +19, formation +13.7 a tick with the order; without it walking -1 or idle -18
    u = {"fatigue": torch.full((4,), 15000.), "fat": torch.zeros(4)}
    activity = melee_activity(4, attack=[True, True, False, False], single=[True, False, True, False],
                              walking=[False, False, True, False])
    fatigue.step(u, activity, calibrated_fatigue(), 1.)
    assert u["fatigue"].tolist() == pytest.approx([15190., 15137., 14990., 14820.])


def test_charging_tires_only_under_an_attack_order():
    u = {"fatigue": torch.full((2,), 15000.), "fat": torch.zeros(2)}
    activity = melee_activity(2, attack=[True, False], charging=[True, True])
    fatigue.step(u, activity, calibrated_fatigue(), 1.)
    assert u["fatigue"].tolist() == [15340., 14820.]


def test_calibrated_melee_without_attack_flags_is_a_formation_attacking():
    u = {"fatigue": torch.full((1,), 15000.), "fat": torch.zeros(1)}
    activity = {k: torch.zeros(1, dtype=torch.bool) for k in ("walking", "running", "shooting", "charging")}
    activity["melee"] = torch.ones(1, dtype=torch.bool)
    fatigue.step(u, activity, calibrated_fatigue(), 1.)
    assert u["fatigue"].tolist() == pytest.approx([15137.])


def test_disabled_fatigue_trial_preserves_legacy_ready_clock():
    import copy
    p = copy.deepcopy(P)
    p.sim["fatigue"]["calibration"]["on"] = False
    u = {"fatigue": torch.full((2,), 15000.), "fat": torch.zeros(2)}
    activity = {k: torch.zeros(2, dtype=torch.bool) for k in
                ("walking", "running", "shooting", "charging", "melee")}
    activity.update(idle=torch.ones(2, dtype=torch.bool), active=torch.zeros(2, dtype=torch.bool))
    fatigue.step(u, activity, p, 1.)
    assert u["fatigue"].tolist() == [14965., 14965.]
    # every engaged unit +19, walking -1, shooting +18 at 5 ticks/s, attack order or not
    activity = melee_activity(3, walking=[False, True, False], shooting=[False, False, True])
    activity["melee"] = torch.tensor([True, False, False])
    u = {"fatigue": torch.full((3,), 15000.), "fat": torch.zeros(3)}
    fatigue.step(u, activity, p, 1.)
    assert u["fatigue"].tolist() == [15095., 14995., 15090.]


def test_a_move_costs_by_its_run_flag_whatever_the_speed():
    # run order +4 even at a walking pace; walk order -1 (database); also while in melee without an attack order
    u = {"fatigue": torch.full((3,), 15000.), "fat": torch.zeros(3)}
    activity = melee_activity(3, melee=[False, False, True], walking=[True, True, True], running=[False, True, False],
                              run_order=[True, False, True])
    fatigue.step(u, activity, calibrated_fatigue(), 1.)
    assert u["fatigue"].tolist() == [15040., 14990., 15040.]


def test_routing_units_tire_at_the_running_rate():
    u = {"fatigue": torch.full((2,), 15000.), "fat": torch.zeros(2)}
    activity = melee_activity(2, melee=[False, False], idle=[False, False], routing=[True, False])
    fatigue.step(u, activity, calibrated_fatigue(), 1.)
    assert u["fatigue"].tolist() == [15040., 14930.]


def test_idle_rest_needs_no_standing_enemy_near():
    u = {"fatigue": torch.full((2,), 15000.), "fat": torch.zeros(2)}
    activity = melee_activity(2, melee=[False, False], idle=[True, True], enemy_near=[True, False])
    fatigue.step(u, activity, calibrated_fatigue(), 1.)
    assert u["fatigue"].tolist() == [14930., 14820.]


@pytest.mark.parametrize("gap,near", [(10, True), (200, False)])
def test_battle_passes_run_flag_routing_and_enemy_near_to_fatigue(monkeypatch, gap, near):
    st = face_off(SPEAR, SLAVE, gap=gap)
    seen = {}
    monkeypatch.setattr(fatigue, "step", lambda u, activity, params, dt: seen.update(activity))
    st.u["r"][0, st.N // 2] = True
    cmd = O.hold(st.B, st.N)
    cmd.kind[0, 0] = O.MOVE
    cmd.x[0, 0], cmd.z[0, 0], cmd.run[0, 0] = st.u["x"][0, 0] - 30, st.u["z"][0, 0], True
    battle.step(st, cmd, P)
    assert bool(seen["run_order"][0, 0]) and not bool(seen["run_order"][0, st.N // 2])
    assert not bool(seen["routing"][0, 0])
    if near:   # far from every standing enemy a router may rally at once
        assert bool(seen["routing"][0, st.N // 2])
    # a routing enemy is not a standing enemy
    assert not bool(seen["enemy_near"][0, 0])
    st.u["r"][0, st.N // 2] = False
    battle.step(st, O.hold(st.B, st.N), P)
    assert bool(seen["enemy_near"][0, 0]) is near


@pytest.mark.parametrize("single", [True, False])
def test_battle_passes_the_attack_order_and_single_entities_to_fatigue(monkeypatch, single):
    st = face_off("wh2_main_skv_cha_warlord_0" if single else SPEAR, SLAVE, gap=0)
    seen = {}
    monkeypatch.setattr(fatigue, "step", lambda u, activity, params, dt: seen.update(activity))
    cmd = O.hold(st.B, st.N)
    cmd.kind[0, 0] = O.ATTACK
    cmd.target[0, 0] = st.N // 2
    battle.step(st, cmd, P)
    assert bool(seen["attack"][0, 0]) and not bool(seen["attack"][0, st.N // 2])
    assert bool(seen["single"][0, 0]) is single and not bool(seen["single"][0, st.N // 2])


@pytest.mark.parametrize("angle,flag", [(-90, "lf"), (90, "rf"), (180, "bf")])
def test_threat_geometry_requires_range_facing_and_a_standing_enemy(angle, flag):
    import copy
    p = copy.deepcopy(P)
    p.sim["threat"]["calibration"]["on"] = True
    x, z = 40 * math.sin(math.radians(angle)), 40 * math.cos(math.radians(angle))
    st = scenario.build([army([(SPEAR, 0, 0, 0)], [(SLAVE, x, z, angle + 180)])], p)
    h = st.N // 2
    def flags():
        pw = geometry.pairwise(st.u, p.sim["formation"]["spacing_m"])
        return battle.threat_flags(st.u, pw, p)
    assert bool(flags()[flag][0, 0])
    assert sum(bool(v[0, 0]) for v in flags().values()) == 1
    st.u["b"][0, h] = angle
    assert not any(bool(v[0, 0]) for v in flags().values())
    st.u["b"][0, h] = angle + 180
    st.u["r"][0, h] = True
    assert not any(bool(v[0, 0]) for v in flags().values())
    st.u["r"][0, h] = False
    st.u["x"][0, h] *= 2
    st.u["z"][0, h] *= 2
    assert not any(bool(v[0, 0]) for v in flags().values())


def test_disabled_threat_trial_keeps_legacy_rear_sector():
    import copy
    p = copy.deepcopy(P)
    p.sim["threat"]["calibration"]["on"] = False
    st = scenario.build([army([(SPEAR, 0, 0, 0)], [(SLAVE, 30, -30, 135)])], p)
    pw = geometry.pairwise(st.u, p.sim["formation"]["spacing_m"])
    assert bool(battle.threat_flags(st.u, pw, p)["bf"][0, 0])


# --- the measured rules of batch 2 (config/nn/sim.json: contact.pursuit_why, pin_why, missile.physical_resist_why,
# morale.windows_why, expendable_why, strong_enemy_why, the morale why on rout_speed) ---

def _router_rate(params, charge=0.0):
    """HP/s the spearmen strike a routing slave unit they touch (melee.strikes)."""
    st = face_off(SPEAR, SLAVE)
    H = st.N // 2
    st.u["r"][0, H] = True
    pw = geometry.pairwise(st.u, params.sim["formation"]["spacing_m"])
    contact = pw["enemy"] & (pw["gap"] <= 1.0)
    rate = melee.strikes(st.u, pw, contact, params, torch.full_like(st.u["men"], charge))[0]
    return float(rate[0, 0, H])


class TestPursuit:
    def test_a_router_is_struck_at_the_pursuit_rate_with_all_its_men_and_no_charge(self):
        full = _router_rate(P.with_cal("contact", pursuit_rate=1.0))
        assert full > 0
        assert _router_rate(P) == pytest.approx(P.sim["contact"]["pursuit_rate"] * full, rel=1e-5)
        # the striker's charge does not count against a router
        assert _router_rate(P, charge=1.0) == pytest.approx(_router_rate(P), rel=1e-5)
        # off (no pursuit_rate): the full rule from the rear (no ramp holds it back any more)
        assert _router_rate(P.with_cal("contact", pursuit_rate=None)) == pytest.approx(full, rel=1e-5)

    def test_a_lone_pursuer_hurts_a_router_and_the_router_strikes_nobody(self):
        st = face_off(SPEAR, SLAVE)
        H = st.N // 2
        st.u["r"][0, H], st.u["morale"][0, H], st.u["rout_count"][0, H] = True, -30.0, 1.0
        hp_spear, hp_slave = float(st.u["hp_abs"][0, 0]), float(st.u["hp_abs"][0, H])
        o = replay.hold(st)
        o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
        battle.step(st, o, P)
        assert float(st.u["hp_abs"][0, H]) < hp_slave and float(st.u["hp_abs"][0, 0]) == hp_spear


class TestRoutSpeed:
    @pytest.mark.parametrize("tired", [False, True])
    def test_a_router_runs_at_rout_speed_of_its_run_and_fatigue_counts_once(self, tired):
        # (a standing unit a side far away keeps the battle going)
        st = scenario.build([army([(SPEAR, -300, 0, 90), (SPEAR, -300, 700, 90)],
                                  [(SPEAR, 300, 0, 270), (SPEAR, 300, 700, 270)])], P)
        st.u["r"][0, 0], st.u["morale"][0, 0], st.u["rout_count"][0, 0], st.u["rally_s"][0, 0] = True, -40.0, 1.0, 0.0
        if tired:
            st.u["fatigue"][0, 0], st.u["fat"][0, 0] = 29000.0, 5.0     # exhausted: speed x0.85 (database)
        for _ in range(6):
            battle.step(st, replay.hold(st), P)
        v = math.hypot(float(st.u["vx"][0, 0]), float(st.u["vz"][0, 0]))
        assert P.sim["morale"]["rout_speed"] == 0.985
        assert v / float(P.units[SPEAR]["speed"]["run"]) == pytest.approx(0.985 * (0.85 if tired else 1.0), rel=0.01)


class TestTargetFace:
    def test_two_units_on_one_face_of_the_target_share_it(self):
        """melee.target_face_cap: two spearmen units standing in each other at the target's front strike it hardly
        harder than one (they share its front's files); off: twice as hard; one on its front and one on its flank: more."""
        def rate(params, second_at):
            st = scenario.build([army([(SPEAR, -50, 0, 90), (SPEAR, -50, 0, 90), (SPEAR, -300, 600, 90)],
                                      [(SLAVE, 50, 0, 270)])], params)
            H = st.N // 2
            front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
            st.u["x"][0, 0] = st.u["x"][0, 1] = -(depth[0, 0] / 2 + TOUCH / 2)
            st.u["x"][0, H] = depth[0, H] / 2 + TOUCH / 2
            if second_at == "flank":                     # beside the slaves' left flank, facing it
                st.u["x"][0, 1] = st.u["x"][0, H]
                st.u["z"][0, 1] = front[0, H] / 2 + depth[0, 1] / 2 + TOUCH / 2
                st.u["b"][0, 1] = 180.0
            if second_at == "none":
                st.u["x"][0, 1], st.u["z"][0, 1] = -300.0, -600.0
            pw = geometry.pairwise(st.u, params.sim["formation"]["spacing_m"])
            contact = pw["enemy"] & (pw["gap"] <= 1.0)
            r = melee.strikes(st.u, pw, contact, params, torch.zeros_like(st.u["men"]))[0]
            return float(r[0, 0, H] + r[0, 1, H])
        one = rate(P, "none")
        assert P.sim["melee"]["target_face_cap"] is True
        assert one < rate(P, "pile") < 1.3 * one          # (the slaves' front is a little longer than the spearmen's)
        assert rate(P.with_cal("melee", target_face_cap=False), "pile") == pytest.approx(2 * one, rel=0.02)
        assert rate(P, "flank") > 1.3 * one


class TestFriendPush:
    def test_own_formations_may_stand_in_each_other_unless_friend_push(self):
        """contact.friend_push false (the game): two own formations sent to the same point end there together; true
        (the old rule): they are pushed apart."""
        def end_gap(params):
            st = scenario.build([army([(SPEAR, -300, -20, 90), (SPEAR, -300, 20, 90), (SPEAR, -300, 600, 90)],
                                      [(SPEAR, 300, 0, 270)])], params)
            for _ in range(int(round(20 / params.dt))):
                o = replay.hold(st)
                o.kind[0, :2] = O.MOVE
                o.x[0, :2], o.z[0, :2] = -280.0, 0.0
                battle.step(st, o, params)
            return math.hypot(float(st.u["x"][0, 0] - st.u["x"][0, 1]), float(st.u["z"][0, 0] - st.u["z"][0, 1]))
        assert P.sim["contact"]["friend_push"] is False
        assert end_gap(P) < 2.0
        assert end_gap(P.with_cal("contact", friend_push=True)) > 10.0


def _rout_from_melee(params, seconds=10.0):
    """Spearmen (HOLD) in contact with skavenslaves for 2 s, then the slaves rout (morale pushed below 0); a standing
    unit far away on each side keeps the battle going. Per step after the rout: (s since the rout, slaves routing,
    in melee, HP, speed, rout-exit clock)."""
    st = scenario.build([army([(SPEAR, -50, 0, 90), (SPEAR, -300, 700, 90)],
                              [(SLAVE, 50, 0, 270), (SLAVE, 300, 700, 270)])], params)
    H = st.N // 2
    front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
    st.u["x"][0, 0] = -(depth[0, 0] / 2 + TOUCH / 2)
    st.u["x"][0, H] = depth[0, H] / 2 + TOUCH / 2
    dt = params.dt
    for _ in range(int(round(2.0 / dt))):
        battle.step(st, replay.hold(st), params)
    assert bool(st.u["m"][0, H]) and not bool(st.u["r"][0, H])
    st.u["morale"][0, H] = -30.0
    rows = []
    for k in range(int(round(seconds / dt))):
        battle.step(st, replay.hold(st), params)
        u = st.u
        rows.append(((k + 1) * dt, bool(u["r"][0, H]), bool(u["m"][0, H]), float(u["hp_abs"][0, H]),
                     math.hypot(float(u["vx"][0, H]), float(u["vz"][0, H])), float(u["rpin_s"][0, H])))
    full = float(st.u["run"][0, H]) * params.sim["morale"]["rout_speed"]
    return rows, full


def _attack_router(st):
    """Orders: the spearmen (unit 0) attack the slaves (unit H), the rest hold."""
    o = O.hold(st.B, st.N, st.device)
    H = st.N // 2
    o.kind[0, 0] = O.ATTACK
    o.target[0, 0] = H
    return o


class TestPursuitSprint:
    def test_a_pursuer_of_a_router_runs_at_its_run_speed_not_the_charge_sprint(self):
        """contact.pursuit_sprint false: spearmen attacking routed skavenslaves beyond the exit run at their run speed
        (3.0) and fall behind the faster router (4.2 x rout_speed); with the sprint (true) they take the charge speed."""
        def chase(params):
            st = scenario.build([army([(SPEAR, -50, 0, 90), (SPEAR, -300, 700, 90)],
                                      [(SLAVE, 50, 0, 270), (SLAVE, 300, 700, 270)])], params)
            H = st.N // 2
            front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
            st.u["x"][0, 0] = -(depth[0, 0] / 2 + TOUCH / 2)
            st.u["x"][0, H] = depth[0, H] / 2 + TOUCH / 2
            for _ in range(int(round(2.0 / params.dt))):
                battle.step(st, _attack_router(st), params)
            st.u["morale"][0, H] = -30.0
            speeds, gaps = [], []
            for k in range(int(round(20.0 / params.dt))):
                battle.step(st, _attack_router(st), params)
                if (k + 1) * params.dt > 9.0:                        # past the exit (rout_pin_s 7) and its wake
                    speeds.append(math.hypot(float(st.u["vx"][0, 0]), float(st.u["vz"][0, 0])))
                    gaps.append(float(st.u["x"][0, H] - st.u["x"][0, 0]))
            return speeds, gaps, float(st.u["run"][0, 0]), float(st.u["charge_speed"][0, 0])
        speeds, gaps, run, charge = chase(P)
        assert max(speeds) <= run + 0.05 and gaps[-1] > gaps[0] + 3.0            # at a run, falling behind
        speeds, gaps, run, charge = chase(P.with_cal("contact", pursuit_sprint=True))
        assert max(speeds) >= charge - 0.05                                      # the sprint: the charge speed


class TestRoutExit:
    def test_a_formation_routing_from_melee_stays_in_contact_and_is_struck_for_rout_pin_s(self):
        pin = P.sim["contact"]["rout_pin_s"]
        assert pin == 7
        on, full = _rout_from_melee(P)
        off, _ = _rout_from_melee(P.with_cal("contact", rout_pin_s=0))
        assert all(r for _, r, *_ in on) and all(r for _, r, *_ in off)
        at = lambda rows, t: next(x for x in rows if x[0] >= t - 1e-6)
        # in melee (the game's flag) through the exit, out of it after; off: out of contact at once
        assert at(on, 3.0)[2] and at(on, 6.0)[2]
        assert not at(on, 9.0)[2] and at(on, 9.0)[5] == 0.0
        assert not at(off, 2.0)[2]
        # struck while it pulls away: more HP lost in the 7 s than without the exit
        lost_on = on[0][3] - at(on, 7.0)[3]
        lost_off = off[0][3] - at(off, 7.0)[3]
        assert lost_on > 1.5 * lost_off and lost_on > 0
        # it gathers speed by rout_pin_speed once it has turned and sped up (0.81 of its free rout speed in s 3-4, 0.97
        # in s 6-7), all of it after the exit; off: all of it from s 3
        assert at(on, 4.0)[4] / full == pytest.approx(0.81, abs=0.04)
        assert at(on, 6.5)[4] / full == pytest.approx(0.95, abs=0.04)
        assert at(on, 9.5)[4] == pytest.approx(full, rel=0.03)
        assert at(off, 4.0)[4] == pytest.approx(full, rel=0.03)

    def test_the_ramp_is_linear_between_its_points(self):
        table = P.sim["contact"]["rout_pin_speed"]
        v = movement.ramp(torch.tensor([0.0, 0.5, 1.0, 7.5, 20.0]), table).tolist()
        assert v == pytest.approx([0.41, 0.41, 0.535, 1.0, 1.0])


class TestRoutDodge:
    def test_a_chased_router_runs_60_deg_off_the_line_from_its_chaser(self):
        """morale.rout_dodge: with a standing enemy formation within 25 m a router's speed away from it is cos 60 deg =
        0.5 of its speed, to a side fixed per unit; with none within 25 m it runs straight away (1.0); off: straight."""
        def away_share(params, enemy_x):
            st = scenario.build([army([(SPEAR, 0, 0, 90), (SPEAR, -300, 700, 90)],
                                      [(SLAVE, enemy_x, 0, 270), (SLAVE, 300, 700, 270)])], params)
            H = st.N // 2
            st.u["r"][0, 0], st.u["rout_count"][0, 0], st.u["morale"][0, 0], st.u["rout_s"][0, 0] = True, 1.0, -20.0, 10.0
            shares = []
            for _ in range(int(round(6.0 / params.dt))):
                lx, lz = float(st.u["x"][0, 0] - st.u["x"][0, H]), float(st.u["z"][0, 0] - st.u["z"][0, H])
                ln = math.hypot(lx, lz)
                battle.step(st, replay.hold(st), params)
                vx, vz = float(st.u["vx"][0, 0]), float(st.u["vz"][0, 0])
                sp = math.hypot(vx, vz)
                if sp > 1.0:
                    shares.append((vx * lx + vz * lz) / (sp * ln))          # speed away from the enemy, of its speed
                # the enemy follows, kept enemy_x m from the router along the line (a chaser behind it)
                st.u["x"][0, H] = st.u["x"][0, 0] - lx / ln * abs(enemy_x)
                st.u["z"][0, H] = st.u["z"][0, 0] - lz / ln * abs(enemy_x)
            return sum(shares[-4:]) / 4
        on = P.with_cal("morale", rout_dodge={"on": True, "m": 25, "deg": 60})
        assert away_share(on, -15.0) == pytest.approx(0.5, abs=0.08)                 # on, chaser 15 m behind: 60 deg off
        assert away_share(on, -100.0) == pytest.approx(1.0, abs=0.05)                # on, none within 25 m: straight
        assert away_share(P, -15.0) == pytest.approx(1.0, abs=0.05)                  # off (the routmob probe): straight

class TestFleeGoal:
    def test_a_router_far_from_every_enemy_keeps_running_from_the_enemy_army(self):
        """morale.flee_near_m 0: a routing formation heads away from every standing enemy weighted 1/distance - with the
        enemy army 300 m north and no enemy within 150 m it runs south, not to its own (west) map edge; with the old
        150 m cutoff (flee_near_m 150) it runs west, to its own edge (the recordings: far from every enemy the game's
        routers keep running from the enemy army, 77-85 % within 30 deg, the own edge 35 %; sim.json flee_near_why)."""
        def heading(params):
            st = scenario.build([army([(SPEAR, 0, 0, 90), (SPEAR, -300, 700, 90)],
                                      [(SLAVE, 0, 300, 180), (SLAVE, 300, 700, 270)])], params)
            st.u["r"][0, 0], st.u["morale"][0, 0], st.u["rout_count"][0, 0], st.u["rout_s"][0, 0] = True, -100.0, 1.0, 20.0
            for _ in range(int(round(6.0 / params.dt))):
                battle.step(st, replay.hold(st), params)
            vx, vz = float(st.u["vx"][0, 0]), float(st.u["vz"][0, 0])
            n = math.hypot(vx, vz)
            assert n > 1.0
            return vx / n, vz / n
        assert P.sim["morale"]["flee_near_m"] == 0
        hx, hz = heading(P)
        assert hz < -0.9                                           # south: away from the enemy army
        hx, hz = heading(P.with_cal("morale", flee_near_m=150))
        assert hx < -0.9                                           # the old rule: west, to its own edge


class TestRoutCrowd:
    def test_a_router_among_other_formations_is_slowed(self):
        """morale.rout_crowd: a router with two own standing formations around it runs at own[2] of its rout speed,
        with one enemy formation at enemy[1]; alone at the full rout speed; off: full."""
        def speed(params, friends=0, enemy=False):
            rows1 = [(SPEAR, -300, 0, 90)] + [(SPEAR, -300, 4.0 * (k + 1), 90) for k in range(friends)] + [(SPEAR, -300, 700, 90)]
            rows2 = [(SPEAR, 300, 0, 270), (SPEAR, 300, 700, 270)]
            if enemy:
                rows2.append((SPEAR, -300, -6.0, 270))
            st = scenario.build([army(rows1, rows2)], params)
            st.u["r"][0, 0], st.u["morale"][0, 0], st.u["rout_count"][0, 0], st.u["rally_s"][0, 0] = True, -40.0, 1.0, 0.0
            vs = []
            for _ in range(int(round(3.0 / params.dt))):
                battle.step(st, replay.hold(st), params)
                vs.append(math.hypot(float(st.u["vx"][0, 0]), float(st.u["vz"][0, 0])))
            return max(vs)
        crowd = P.sim["morale"]["rout_crowd"]
        alone = speed(P)
        assert speed(P, friends=2) == pytest.approx(alone * crowd["own"][2], rel=0.05)
        assert speed(P, enemy=True) == pytest.approx(alone * crowd["enemy"][1], rel=0.05)
        off = P.with_cal("morale", rout_crowd=None)
        assert speed(off, friends=2) == pytest.approx(speed(off), rel=0.02)


def _ctx_of(monkeypatch, st, orders_of, steps):
    """The morale context battle.step hands morale.step, each step (morale itself still steps)."""
    seen = []
    real = morale.step

    def spy(u, ctx, params, dt):
        seen.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in ctx.items()})
        return real(u, ctx, params, dt)
    monkeypatch.setattr(morale, "step", spy)
    for _ in range(steps):
        battle.step(st, orders_of(st), P)
    return seen


class TestMoraleWindows:
    def test_under_fire_holds_under_fire_s_after_the_last_hit(self, monkeypatch):
        st = face_off(SPEAR, SLAVE, gap=400)
        st.u["under_fire_s"][0, 0] = 0.0                       # hit on the step before
        seen = _ctx_of(monkeypatch, st, replay.hold, 32)
        flags = [bool(c["under_fire"][0, 0]) for c in seen]
        held = int(P.sim["morale"]["under_fire_s"] / P.dt)     # 15 s: steps 0.5 ... 14.5 s after the hit
        assert P.sim["morale"]["under_fire_s"] == 15
        assert all(flags[:held - 1]) and not any(flags[held - 1:])

    def test_recent_casualties_are_the_last_4_s_extended_the_last_60_s(self):
        st = face_off(SPEAR, SLAVE)
        u = st.u
        assert u["lost_hist"].shape == (1, morale.history_steps(P, P.dt) * st.N)
        hp0 = float(u["hp0"][0, 0])
        lost = torch.zeros_like(u["men"])
        lost[0, 0] = 0.15 * hp0
        recent, extended = [], []
        for k in range(130):
            morale.casualty_windows(u, lost if k == 0 else torch.zeros_like(lost), P, P.dt)
            recent.append(float(u["recent"][0, 0]))
            extended.append(float(u["extended"][0, 0]))
        assert recent[:8] == pytest.approx([0.15 * hp0] * 8) and recent[8:] == [0.0] * 122
        assert extended[:120] == pytest.approx([0.15 * hp0] * 120) and extended[120:] == [0.0] * 10

    def test_the_windows_give_the_database_points(self):
        st = face_off(SPEAR, SLAVE)
        u = st.u
        ctx = TestMorale().ctx(st)
        base = morale.target_points(u, ctx, P)
        R = P.morale
        u["recent"][0, 0] = 0.15 * u["hp0"][0, 0]
        assert float((morale.target_points(u, ctx, P) - base)[0, 0]) == R["recent_casualties_penalty_15"] == -20
        u["recent"][0, 0] = 0.0
        u["extended"][0, 0] = 0.15 * u["hp0"][0, 0]
        assert float((morale.target_points(u, ctx, P) - base)[0, 0]) == R["extended_casualties_penalty_15"] == -6

    def test_decaying_windows_stay_behind_the_switch(self):
        p = P.with_cal("morale", casualties_window="decay", casualties_s=30, extended_s=0)
        assert morale.history_steps(p, p.dt) == 0
        st = scenario.build([army([(SPEAR, -50, 0, 90)], [(SLAVE, 50, 0, 270)])], p)
        assert st.u["lost_hist"].shape == (1, 0)
        lost = torch.zeros_like(st.u["men"])
        lost[0, 0] = 10.0
        morale.casualty_windows(st.u, lost, p, p.dt)
        morale.casualty_windows(st.u, torch.zeros_like(lost), p, p.dt)
        assert float(st.u["recent"][0, 0]) == pytest.approx(10.0 * math.exp(-0.5 / 30))

    def test_frozen_and_narrowed_battles_carry_the_history(self):
        st = scenario.build([army([(SPEAR, -5, 0, 90)], [(SLAVE, 5, 0, 270)])] * 2, P)
        st.done[1] = True
        o = replay.hold(st)
        o.kind[:, 0], o.target[:, 0] = O.ATTACK, st.N // 2
        for _ in range(10):
            battle.step(st, o, P)
        assert float(st.u["lost_hist"][0].sum()) > 0 and float(st.u["lost_hist"][1].sum()) == 0
        assert st.clone().u["lost_hist"].shape == st.u["lost_hist"].shape


class TestExpendableAndStrongEnemy:
    def test_a_routing_expendable_unit_scares_only_expendable_units(self, monkeypatch):
        st = scenario.build([army([(SPEAR, -600, 0, 90)],
                                  [(SLAVES, 300, 0, 270), (SLAVES, 300, 40, 270), (CLANRAT, 340, 0, 270)])], P)
        H = st.N // 2
        assert bool(st.u["expendable"][0, H]) and not bool(st.u["expendable"][0, H + 2])
        st.u["r"][0, H], st.u["morale"][0, H], st.u["rout_count"][0, H] = True, -30.0, 1.0
        seen = _ctx_of(monkeypatch, st, replay.hold, 1)
        assert float(seen[0]["routing_friends"][0, H + 1]) == 1 and float(seen[0]["routing_friends"][0, H + 2]) == 0
        st = scenario.build([army([(SPEAR, -600, 0, 90)],
                                  [(CLANRAT, 300, 0, 270), (SLAVES, 300, 40, 270), (CLANRAT, 340, 0, 270)])], P)
        st.u["r"][0, H], st.u["morale"][0, H], st.u["rout_count"][0, H] = True, -30.0, 1.0
        seen = _ctx_of(monkeypatch, st, replay.hold, 1)
        assert float(seen[0]["routing_friends"][0, H + 1]) == 1 and float(seen[0]["routing_friends"][0, H + 2]) == 1

    def test_a_stronger_enemy_near_costs_the_database_minimum(self):
        """ctx strong_enemy: the database's enemy_morale_penalty_value_min (-3; config/nn/sim.json strong_enemy_why)."""
        st = face_off(SPEAR, SLAVE)
        ctx = TestMorale().ctx(st)
        strong = TestMorale().ctx(st, strong_enemy=torch.ones_like(st.u["r"]))
        assert float((morale.target_points(st.u, strong, P) - morale.target_points(st.u, ctx, P))[0, 0]) == -3


class TestPhysicalResistanceAgainstMissiles:
    def test_physical_resistance_cuts_missile_damage_capped_at_90_percent(self):
        st = face_off(ARCHER, RUNNERS, gap=100)
        H = st.N // 2
        assert float(st.u["resist_physical"][0, H]) == pytest.approx(0.2)
        pw = geometry.pairwise(st.u, 1.5)
        target = torch.zeros((1, st.N), dtype=torch.long) - 1
        target[0, 0] = H
        hp = float(missile.volley(st.u, pw, target, 1.0, P)[1].sum())
        off = float(missile.volley(st.u, pw, target, 1.0, P.with_cal("missile", physical_resist=False))[1].sum())
        a, r = P.units[ARCHER]["missile"], P.units[RUNNERS]

        def per(resist):
            return float(melee.per_hit(torch.tensor(float(a["damage"])), torch.tensor(float(a["ap_damage"])),
                                       torch.tensor(float(r["armour"])), torch.tensor(float(r["hp_per_man"])),
                                       torch.tensor(resist)))
        assert hp == pytest.approx(per(0.2) / per(0.0) * off, rel=1e-4) and 0 < hp < off
        st.u["resist_missile"][0, H], st.u["resist_physical"][0, H] = 0.6, 0.5
        capped = float(missile.volley(st.u, pw, target, 1.0, P)[1].sum())
        assert capped == pytest.approx(per(0.9) / per(0.0) * off, rel=1e-4)


class TestLeavingMelee:
    def _fight(self, key, n=2):
        """n copies of: key (slot 0) and spearmen (slot H) in melee, both attacking."""
        st = scenario.build([army([(key, -50, 0, 90)], [(SPEAR, 50, 0, 270)])] * n, P)
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        H = st.N // 2
        st.u["x"][:, 0] = -depth[:, 0] / 2 - TOUCH / 2
        st.u["x"][:, H] = depth[:, H] / 2 + TOUCH / 2
        o = replay.hold(st)
        o.kind[:, H], o.target[:, H] = O.ATTACK, 0
        o.kind[:, 0], o.target[:, 0] = O.ATTACK, H
        for _ in range(4):
            battle.step(st, o, P)
        assert bool(st.u["m"][:, 0].all())
        return st, H

    @pytest.mark.parametrize("key,held_s", [(SPEAR, 0.0), (ARCHER, 5.0), (GENERAL, 0.0)])
    def test_a_missile_unit_leaving_melee_is_held_pin_s_a_melee_unit_and_a_lord_not(self, key, held_s):
        # (a melee unit is not held: chased, it walks on with its chaser in contact - contact.chase)
        st, H = self._fight(key, 1)
        x0, hp0 = float(st.u["x"][0, 0]), float(st.u["hp_abs"][0, 0])
        xs = []
        for _ in range(int(held_s / P.dt) + 12):
            st.u["morale"][:] = 1000.0                           # nobody routs in this test
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, -200.0, 0.0, True
            battle.step(st, o, P)
            xs.append(x0 - float(st.u["x"][0, 0]))
        held = int(held_s / P.dt)
        if held:
            assert max(xs[:held]) == pytest.approx(0.0, abs=0.01) and float(st.u["hp_abs"][0, 0]) < hp0
        assert xs[-1] > 3

    @pytest.mark.parametrize("key,mult", [(SPEAR, 1.25), (ARCHER, 0.55)])
    def test_a_leaving_unit_takes_leave_taken_of_the_blows_and_deals_none(self, key, mult):
        st, H = self._fight(key, 2)
        hp0, k0 = st.u["hp_abs"][:, 0].clone(), st.u["k"][:, 0].clone()
        o = replay.hold(st)
        o.kind[:, H], o.target[:, H] = O.ATTACK, 0
        o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, -200.0, 0.0, True   # battle 0 leaves
        battle.step(st, o, P)                                                        # battle 1 holds
        lost = hp0 - st.u["hp_abs"][:, 0]
        assert float(lost[0]) == pytest.approx(mult * float(lost[1]), rel=1e-3) and float(lost[1]) > 0
        assert float(st.u["k"][0, 0]) == float(k0[0])

    def test_off_leaving_melee_units_walk_out_and_take_the_rule(self):
        p = P.with_cal("contact", pin_melee_s=0.0, leave_taken=None)
        st = face_off(SPEAR, SPEAR)
        H = st.N // 2
        o = replay.hold(st)
        o.kind[0, H], o.target[0, H] = O.ATTACK, 0
        battle.step(st, o, p)
        x0 = float(st.u["x"][0, 0])
        for _ in range(6):
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, -200.0, 0.0, True
            battle.step(st, o, p)
        assert x0 - float(st.u["x"][0, 0]) > 1


class TestReplayMeleeGaps:
    def test_fill_gaps(self):
        import numpy as np
        f = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=bool)
        assert replay.fill_gaps(f, 3).tolist() == [True, True, True, True, False, False, False, False, True, False]
        assert replay.fill_gaps(f, 0).tolist() == f.tolist()

    def test_a_flickering_melee_flag_is_melee_for_a_non_leaver_only(self):
        from tools.nn import gamedata
        import numpy as np
        T, N = 6, 2
        f = {k: np.zeros((T, N)) for k in gamedata.FLOAT_FIELDS}
        f.update({k: np.zeros((T, N), dtype=bool) for k in gamedata.BOOL_FIELDS})
        f["x"][:, 1] = 10.0
        f["ox"][:, 0], f["ox"][:, 1] = -30.0, 40.0            # far points (the planner's: the enemy's start)
        f["men"][:] = 100
        f["m"][:] = True
        f["m"][2:4] = False                                   # the flag flickers off 2 s
        b = gamedata.Battle(run="t", own_ai="attack", enemy_role="defend", result={}, t=np.arange(T, dtype=float),
                            f=f, target=np.full((T, N), -1), names=("a", "b"), keys=("", ""), side=np.array([1, 2]))
        rows = replay.recorded_orders(b, [0, 1], N, fight_nearest=True, leave_m=10.0, leavers=[True, False])
        assert (rows["kind"][:, 1] == O.ATTACK).all() and (rows["phase"][:, 1] != 2).all()
        assert (rows["kind"][2:4, 0] == O.MOVE).all()
        gap0 = replay.recorded_orders(b, [0, 1], N, fight_nearest=True, leave_m=10.0, leavers=[True, False],
                                      melee_gap=0)
        assert (gap0["kind"][2:4, 1] == O.MOVE).all()


class TestLordFragility:
    """Batch 3: the lord as fragile as in the game (config/nn/sim.json morale.lord_why, missile.single_in_melee_why,
    fatigue.lord_why)."""

    def test_the_lords_aura_reaches_his_units_not_himself(self, monkeypatch):
        st = scenario.build([army([(GENERAL, 0, 0, 90, True), (SPEAR, 0, 20, 90)], [(SLAVE, 400, 0, 270)])], P)
        seen = _ctx_of(monkeypatch, st, lambda s: O.hold(s.B, s.N), 1)
        assert float(seen[0]["aura"][0, 1]) == 1.0 and float(seen[0]["aura"][0, 0]) == 0.0
        st = scenario.build([army([(GENERAL, 0, 0, 90, True), (SPEAR, 0, 20, 90)], [(SLAVE, 400, 0, 270)])], P)
        own = P.with_cal("morale", lord_own_aura=True)
        seen = []
        real = morale.step
        monkeypatch.setattr(morale, "step", lambda u, ctx, params, dt: (seen.append(ctx["aura"].clone()),
                                                                         real(u, ctx, params, dt))[1])
        battle.step(st, O.hold(st.B, st.N), own)
        assert float(seen[0][0, 0]) == 1.0

    def test_a_single_entity_in_melee_is_always_losing(self):
        st = face_off(GENERAL, SLAVE)
        H = st.N // 2
        st.u["dealt"][0, 0], st.u["taken"][0, 0] = 500.0, 0.0      # the lord wins by HP
        st.u["dealt"][0, H], st.u["taken"][0, H] = 500.0, 0.0      # so do the slaves (a formation)
        melee_ctx = TestMorale().ctx(st, in_melee=torch.ones_like(st.u["r"]))
        calm_ctx = TestMorale().ctx(st)
        now = morale.target_points(st.u, melee_ctx, P) - morale.target_points(st.u, calm_ctx, P)
        old_p = P.with_cal("morale", single_combat=None)
        old = morale.target_points(st.u, melee_ctx, old_p) - morale.target_points(st.u, calm_ctx, old_p)
        assert float(now[0, 0]) == P.morale["losing_combat"] == -3
        assert float(old[0, 0]) == float(old[0, H]) == float(now[0, H]) == P.morale["winning_combat_significantly"]

    def test_a_lord_in_melee_takes_none_of_the_hits_aimed_at_him_and_his_attackers_keep_the_friendly_fire(self):
        st = scenario.build([army([(GENERAL, 0, 0, 90), (SPEAR, 0, 30, 90)],
                                  [(SPEAR, 6, 0, 270), (SLINGER, 120, 0, 270)])], P)
        H = st.N // 2
        pw = geometry.pairwise(st.u, 1.5)
        target = torch.full((1, st.N), -1)
        target[0, H + 1] = 0                                  # the slingers shoot our General
        contact = torch.zeros((1, st.N, st.N), dtype=torch.bool)
        contact[0, 0, H] = contact[0, H, 0] = True            # he fights the enemy spearmen
        _, fight, _ = missile.volley(st.u, pw, target, 1.0, P, contact=contact)
        _, free, _ = missile.volley(st.u, pw, target, 1.0, P, contact=torch.zeros_like(contact))
        _, old, _ = missile.volley(st.u, pw, target, 1.0, P.with_cal("missile", single_entity_in_melee=None),
                                   contact=contact)
        ff = P.sim["missile"]["friendly_fire"]["sling"]
        lone = P.sim["missile"]["single_entity_factor"]
        # out of melee the spread model's lone-man chance; in melee the measured rule (hit_rate x distance factor)
        model = float(missile.hit_chance(st.u, pw, P)[0, H + 1, 0])
        rate = float(st.u["hit_rate"][0, H + 1] * missile.distance_factor(pw["dist"][0, H + 1, 0],
                                                                          P.sim["missile"]["distance_factor"]))
        assert P.sim["missile"]["single_entity_in_melee"] == 0.0         # measured (sim.json single_in_melee_why)
        assert float(fight[0, H + 1, 0]) == 0.0
        assert float(fight[0, H + 1, H]) > 0                          # the slingers' own spearmen in contact: ff
        _, whole, _ = missile.volley(st.u, pw, target, 1.0, P.with_cal("missile", single_entity_in_melee=1.0),
                                     contact=contact)
        assert float(whole[0, H + 1, 0]) / float(free[0, H + 1, 0]) == pytest.approx((1 - ff) * rate / model, rel=1e-3)
        assert float(whole[0, H + 1, H]) == pytest.approx(float(fight[0, H + 1, H]), rel=1e-6)
        assert float(old[0, H + 1, 0]) / float(free[0, H + 1, 0]) == pytest.approx((1 - ff) * lone * rate / model,
                                                                                   rel=1e-3)

    def test_a_lord_tires_slower_in_melee_and_the_charge_costs_while_the_caller_says(self):
        # single entity in melee: the database's 19 a tick, a formation 13.7; the charge (+34) for both while
        # `charging` (battle.py: the first fatigue.calibration.charge_s seconds after the charge's first blow).
        u = {"fatigue": torch.full((4,), 15000.), "fat": torch.zeros(4)}
        activity = melee_activity(4, attack=[True] * 4, single=[True, True, False, False],
                                  charging=[True, False, True, False])
        fatigue.step(u, activity, calibrated_fatigue(), 1.)
        assert u["fatigue"].tolist() == pytest.approx([15340., 15190., 15340., 15137.])


def _keep(st):
    o = O.hold(st.B, st.N)
    o.kind[:] = O.KEEP
    return o


class TestMoraleBatch:
    """The morale batch (06.10.2026, build/morale_spec/spec.md section 4, the morale probe build/morale): flanks secure,
    attacked in the flank / rear while struck, the charge's +15 in blocks, winning the fight by shooting, a router's
    morale following its target and the rally rule, a stronger and faster enemy."""

    def ctx_of(self, st, steps=1, orders=None, monkeypatch=None):
        seen = []
        real = morale.step

        def spy(u, ctx, params, dt):
            seen.append({k: (v.clone() if torch.is_tensor(v) else v) for k, v in ctx.items()})
            return real(u, ctx, params, dt)
        monkeypatch.setattr(morale, "step", spy)
        for _ in range(steps):
            battle.step(st, (orders or replay.hold)(st), P)
        return seen

    def test_flanks_secure_by_enemy_distance_or_friends_at_both_sides(self, monkeypatch):
        """M1, M2: +5 with no standing enemy within morale.secure.enemy_m, or friends (not lords) within side_m at both
        sides; one side, a friend in front or only the lord: no; a threatened flank: no."""
        sc = P.sim["morale"]["secure"]
        far = sc["enemy_m"] + 20
        near = sc["enemy_m"] - 40

        def secure(friends, enemy_z, lf=False):
            # the unit faces +z (bearing 0): its sides are -x / +x
            own = [(SPEAR, 0, 0, 0)] + [(k, x, z, 0) for k, x, z in friends]
            st = scenario.build([army(own, [(SLAVE, 0, enemy_z, 180)])], P)
            st.u["lf"][0, 0] = lf
            return bool(self.ctx_of(st, monkeypatch=monkeypatch)[-1]["secure"][0, 0])
        assert secure([], far) and not secure([], near)
        assert secure([(SPEAR, 34, 0), (SPEAR, -34, 0)], near)
        assert not secure([(SPEAR, 34, 0)], near)
        assert not secure([(SPEAR, 0, 30)], near)
        assert not secure([(GENERAL, 30, 0), (GENERAL, -30, 0)], near)
        assert not secure([(SPEAR, 34, 0), (SPEAR, -34, 0)], near, lf=True)

    def test_the_charge_gives_15_in_blocks_from_the_sprint(self):
        """M4: an attack order at a run from far: +15 from the charge sprint's start (target within the charge
        distance), block_s long; a second block when the first ends at the contact; none on a move order."""
        mc = P.sim["morale"]["charge"]
        st = face_off(SWORD, CLANRAT, gap=120)
        H = st.N // 2
        st.u["leadership"][0, H] += 1e4
        st.u["morale"][0, H] += 1e4
        on, gaps, contact = [], [], None
        for k in range(160):
            o = O.hold(st.B, st.N)
            o.kind[0, 0], o.target[0, 0], o.run[0, 0] = O.ATTACK, H, True
            battle.step(st, o if k == 0 else _keep(st), P)
            on.append(float(st.u["chm_s"][0, 0]) > 0)
            gaps.append(float(geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])["gap"][0, 0, H]))
            if contact is None and bool(st.u["m"][0, 0]):
                contact = k
        first = on.index(True)
        assert gaps[first - 1] > st.u["charge_pose"][0, 0] - 1 and contact is not None
        run = on[first:].index(False) if False in on[first:] else len(on) - first
        assert run * 0.5 == pytest.approx(min(mc["blocks"] * mc["block_s"],
                                              (contact - first) * 0.5 + mc["block_s"] + 0.5), abs=0.51)
        st = face_off(SWORD, CLANRAT, gap=120)
        for k in range(120):
            o = O.hold(st.B, st.N)
            o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, float(st.u["x"][0, H]), float(st.u["z"][0, H]), True
            battle.step(st, o if k == 0 else _keep(st), P)
            assert float(st.u["chm_s"][0, 0]) == 0

    def test_charge_term_and_none_while_routing(self):
        st = face_off(SPEAR, SLAVE)
        ctx = TestMorale().ctx(st)
        base = morale.target_points(st.u, ctx, P)
        st.u["chm_s"][0, 0] = 3.0
        assert float((morale.target_points(st.u, ctx, P) - base)[0, 0]) == 15
        st.u["r"][0, 0] = True
        parts = morale.terms(st.u, ctx, P)
        assert all(float(parts[k][0, 0]) == 0 for k in morale.FIGHT_TERMS)

    def test_a_shooter_wins_the_fight_while_shooting(self):
        """M5: missile HP dealt counts in the fight's balance while the unit shoots (in_combat); out of a fight: 0."""
        st = face_off(SPEAR, SLAVE)
        st.u["shot_dealt"][0, 0] = 50.0
        firing = TestMorale().ctx(st, in_combat=torch.ones_like(st.u["r"]))
        idle = TestMorale().ctx(st, in_combat=torch.zeros_like(st.u["r"]))
        assert float(morale.terms(st.u, firing, P)["combat"][0, 0]) == P.morale["winning_combat_significantly"]
        assert float(morale.terms(st.u, idle, P)["combat"][0, 0]) == 0
        st.u["shot_taken"][0, 0] = 5000.0
        st.u["hp_abs"][0, 0] = 0.85 * st.u["hp0"][0, 0]           # past morale.combat_lost
        assert float(morale.terms(st.u, firing, P)["combat"][0, 0]) == P.morale["losing_combat_significantly"]

    def test_archers_shooting_are_in_a_fight_and_their_target_too(self, monkeypatch):
        st = shooter_and([(0, 100, SPEAR, 180)])
        seen = []
        real = morale.step

        def spy(u, ctx, params, dt):
            seen.append(ctx["in_combat"].clone())
            return real(u, ctx, params, dt)
        monkeypatch.setattr(morale, "step", spy)
        H = st.N // 2
        for k in range(40):
            o = O.hold(st.B, st.N)
            o.kind[0, 0], o.target[0, 0] = O.ATTACK, H
            battle.step(st, o if k == 0 else _keep(st), P)
        assert any(bool(x[0, 0]) and bool(x[0, H]) for x in seen)
        assert float(st.u["shot_dealt"][0, 0]) > 0 and float(st.u["shot_taken"][0, H]) > 0

    def test_a_router_follows_its_target_and_does_not_rally_near_an_enemy(self):
        """M6: a router's morale moves by the usual 15 % (at least 1) a tick towards its target; with an enemy within
        rally_free_m no rally."""
        st = face_off(SPEAR, SLAVE)
        st.u["r"][0, 0], st.u["rout_count"][0, 0], st.u["morale"][0, 0] = True, 1.0, -5.0
        near = TestMorale().ctx(st)
        for _ in range(40):
            target = float(morale.target_points(st.u, near, P)[0, 0])
            before = float(st.u["morale"][0, 0])
            morale.step(st.u, near, P, 0.5)
            step_ = max(1.0, 0.15 * abs(target - before))
            assert float(st.u["morale"][0, 0]) == pytest.approx(before + min(step_, target - before), abs=1e-4)
            assert bool(st.u["r"][0, 0])

    def test_rally_after_rally_after_s_free_and_above_0_and_never_below(self):
        """M7: free of enemies, above 0: the rally comes at rally_after_s into the rout, not before; a router whose
        target is below 0 never rallies."""
        after = P.sim["morale"]["rally_after_s"]
        st = face_off(SPEAR, SLAVE)
        st.u["r"][0, 0], st.u["rout_count"][0, 0], st.u["morale"][0, 0] = True, 1.0, 5.0
        free = TestMorale().ctx(st, enemy_near=torch.zeros_like(st.u["r"]))
        t = 0.0
        while bool(st.u["r"][0, 0]) and t < 100:
            morale.step(st.u, free, P, 0.5)
            t += 0.5
        assert t == pytest.approx(after + P.sim["morale"].get("rally_wait_s", 0.0), abs=1.01)
        st = face_off(SPEAR, SLAVE)
        st.u["hp_abs"][0, 0] = 0.05 * st.u["hp0"][0, 0]
        st.u["r"][0, 0], st.u["rout_count"][0, 0], st.u["morale"][0, 0] = True, 1.0, -5.0
        assert float(morale.target_points(st.u, free, P)[0, 0]) < 0
        for _ in range(400):
            morale.step(st.u, free, P, 0.5)
        assert bool(st.u["r"][0, 0])

    def test_the_rally_is_a_chance_a_second_by_the_nearest_enemys_distance(self):
        """morale.rally_hazard: past rally_after_s a router rallies with the table's chance a second by the nearest
        standing enemy's distance (0 within 95 m), by a fixed draw; two 0.5 s steps give the one-second chance."""
        hz = P.sim["morale"]["rally_hazard"]
        B = 3000
        def share(foe_d, dt, steps):
            st = scenario.build([army([(SPEAR, -50, 0, 90)], [(SLAVE, 50, 0, 270)])] * B, P)
            st.u["r"][:, 0], st.u["rout_count"][:, 0], st.u["morale"][:, 0], st.u["rout_s"][:, 0] = True, 1.0, 5.0, 30.0
            ctx = TestMorale().ctx(st, enemy_near=torch.zeros_like(st.u["r"]), foe_d=torch.full_like(st.u["men"], foe_d),
                                   t=torch.zeros(B))
            for _ in range(steps):
                morale.step(st.u, ctx, P, dt)
                ctx["t"] = ctx["t"] + dt
            return float((~st.u["r"][:, 0]).float().mean())
        assert share(50.0, 1.0, 1) == 0.0
        assert share(130.0, 1.0, 1) == pytest.approx(hz["p_per_s"][2], abs=0.02)         # 110-125 m: 0.095
        assert share(100.0, 1.0, 1) == pytest.approx(hz["p_per_s"][1], abs=0.015)        # 95-110 m: 0.043
        assert share(1000.0, 1.0, 1) == pytest.approx(hz["p_per_s"][4], abs=0.01)        # 150 m and beyond: 0.013
        assert share(130.0, 0.5, 2) == pytest.approx(hz["p_per_s"][2], abs=0.02)         # two half steps = a second
        assert share(130.0, 1.0, 1) == share(130.0, 1.0, 1)                               # the draw is fixed

    def test_a_rallied_unit_with_a_target_below_0_routs_again_after_10_s(self):
        """M8: the database's post_rally_no_rout_timer: no rout in the 10 s after a rally, then at morale <= 0."""
        st = face_off(SPEAR, SLAVE)
        st.u["rout_count"][0, 0], st.u["rally_s"][0, 0], st.u["morale"][0, 0] = 1.0, 0.0, 1.0
        st.u["hp_abs"][0, 0] = 0.05 * st.u["hp0"][0, 0]
        ctx = TestMorale().ctx(st)
        t = 0.0
        while not bool(st.u["r"][0, 0]) and t < 60:
            morale.step(st.u, ctx, P, 0.5)
            t += 0.5
        assert t == pytest.approx(P.morale["post_rally_no_rout_timer"], abs=0.51)

    def test_a_strong_enemy_must_be_worth_strong_ratio_times_as_much(self, monkeypatch):
        """M9: within 70 m, worth 1.5x: 0; worth morale.strong_ratio x (3), faster or slower: -3 (the morale probe)."""
        def strong(ratio, run_enemy=0.5):
            st = face_off(SPEAR, SLAVE, gap=40)
            H = st.N // 2
            st.u["cost"][0, H] = st.u["cost"][0, 0] * ratio
            st.u["run"][0, H] = run_enemy
            return bool(self.ctx_of(st, monkeypatch=monkeypatch)[-1]["strong_enemy"][0, 0])
        assert not strong(1.5, 20.0) and strong(P.sim["morale"]["strong_ratio"]) and strong(4.8, 0.5)



    def test_the_fight_shows_only_once_the_loser_has_lost_a_tenth(self):
        """morale.combat_lost (the morale probe): losing only once the unit has lost 10 % of its health, winning only
        once an enemy it fights has."""
        st = face_off(SPEAR, SLAVE)
        st.u["taken"][0, 0] = 500.0
        fight = TestMorale().ctx(st, in_combat=torch.ones_like(st.u["r"]),
                                 foe_lost=torch.zeros_like(st.u["men"]))
        assert float(morale.terms(st.u, fight, P)["combat"][0, 0]) == 0
        st.u["hp_abs"][0, 0] = 0.89 * st.u["hp0"][0, 0]
        assert float(morale.terms(st.u, fight, P)["combat"][0, 0]) == P.morale["losing_combat_significantly"]
        st = face_off(SPEAR, SLAVE)
        st.u["shot_dealt"][0, 0] = 500.0
        fight = TestMorale().ctx(st, in_combat=torch.ones_like(st.u["r"]), foe_lost=torch.full_like(st.u["men"], 0.05))
        assert float(morale.terms(st.u, fight, P)["combat"][0, 0]) == 0
        fight["foe_lost"] = torch.full_like(st.u["men"], 0.12)
        assert float(morale.terms(st.u, fight, P)["combat"][0, 0]) == P.morale["winning_combat_significantly"]


def test_measured_reloads_and_aim_of_the_third_wave():
    """missile.reload_projectile_s (the probe newdist, 07.10.2026): the handgunners' bullet and the throwing stars take
    their measured cycles (14.8 / 8.1 s against 13 / 7 in the database); the crossbowmen's bolt follows the arrows'
    rule (13 s = its database reload, the arrows' cycle equals theirs); the militia's pistol keeps the category rule
    (9 x 1.2). musket aims 2.0 s (missile.aim_s: militia, handgunners and stars first shot 1.5-2.5 s)."""
    s = {k: P.static(k) for k in ("wh_main_emp_inf_handgunners", "wh2_main_skv_inf_night_runners_0",
                                  "wh_main_emp_inf_crossbowmen", "wh_dlc04_emp_inf_free_company_militia_0")}
    assert s["wh_main_emp_inf_handgunners"]["reload"] == pytest.approx(14.8)
    assert s["wh2_main_skv_inf_night_runners_0"]["reload"] == pytest.approx(8.1)
    assert s["wh_main_emp_inf_crossbowmen"]["reload"] == pytest.approx(13.0)
    assert s["wh_dlc04_emp_inf_free_company_militia_0"]["reload"] == pytest.approx(10.8)
    assert all(s[k]["aim_s"] == pytest.approx(2.0) for k in s if k != "wh_main_emp_inf_crossbowmen")
    assert s["wh_main_emp_inf_crossbowmen"]["aim_s"] == pytest.approx(3.3)
