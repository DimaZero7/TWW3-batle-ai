"""tools.nn.sim: the battle simulator (docs/en/training/simulator.md). Needs torch: skipped in the
project's .venv; run in the container (bash tools/nn/dock.sh pytest ... or see the page).

Level 1: small functions. Level 2: a module on made-up cases, checking properties (more armour ->
less damage, a flank -> faster rout...). Level 3: the battle loop wired together."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, fatigue, geometry, melee, missile, morale, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim import state as S  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR, SLAVE, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0", \
    "wh2_main_skv_inf_clanrat_spearmen_0"
ARCHER, SLINGER = "wh2_dlc13_emp_inf_archers_0", "wh2_main_skv_inf_skavenslave_slingers_0"
GENERAL = "wh_main_emp_cha_general_0"


def army(side1, side2, attacker=1, factions=("wh_main_emp_empire", "wh2_main_skv_skaven")):
    """side1/side2: lists of (key, x, z, bearing[, general])."""
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": len(r) > 4 and r[4]} for r in rows]
    return {"attacker": attacker, "sides": {1: {"faction": factions[0], "units": units(side1)},
                                            2: {"faction": factions[1], "units": units(side2)}}}


def face_off(key1, key2, gap=0.0, per_side=None):
    """Two units facing each other along x, their fronts `gap` m apart."""
    st = scenario.build([army([(key1, -50, 0, 90)], [(key2, 50, 0, 270)])], P, per_side=per_side)
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
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
        bare = melee.per_hit(torch.tensor(20.0), torch.tensor(5.0), torch.tensor(0.0), hp)
        armoured = melee.per_hit(torch.tensor(20.0), torch.tensor(5.0), torch.tensor(100.0), hp)
        assert float(bare) == pytest.approx(25) and float(armoured) == pytest.approx(5 + 20 * 0.25)
        assert float(melee.per_hit(torch.tensor(400.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(60.0))) == 60

    def test_a_killing_blow_kills_a_man_per_man_of_health(self):
        assert float(melee.kill_share(torch.tensor(50.0), torch.tensor(60.0), 0.5)) == 1.0
        assert float(melee.kill_share(torch.tensor(60.0), torch.tensor(15.0), 0.5)) == pytest.approx(0.5)

    def test_morale_tables_are_steps(self):
        points = morale.table(P.morale, "total_casualties_penalty_", (10, 20, 30, 40, 50, 60, 70, 80, 90))
        x = torch.tensor([0.05, 0.1, 0.55, 0.95])
        assert morale.steps(x, points).tolist() == [0.0, -2.0, -16.0, -74.0]

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

    def test_state_and_orders_shapes_and_perspective(self):
        st = S.empty(2, 6)
        assert all(v.shape == (2, 6) for v in st.u.values())
        obs = st.observation()
        assert set(obs) == set(S.OBSERVED) | {"side", "t"} and obs["t"].shape == (2,)
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
        cs = torch.full_like(st.u["men"], 100.0)
        rate, _, sector, _ = melee.strikes(st.u, pw, contact, P, ch, cs)
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

    def test_a_rear_attack_hits_more(self):
        p = P.with_cal("melee", hit_slope=1.0)
        st = face_off(SPEAR, SLAVE)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        front = melee.strikes(st.u, pw, contact, p, z, z + 100)[0][0, 0, 1]
        st.u["b"][0, 1] = 90.0            # the slaves turn their back to the spearmen
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        rate, _, sector, _ = melee.strikes(st.u, pw, contact, p, z, z + 100)
        assert int(sector[0, 0, 1]) == 2 and float(rate[0, 0, 1]) > float(front)

    def test_at_most_eight_reach_a_lord(self):
        st = face_off(SPEAR, GENERAL)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        rate, hit, _, _ = melee.strikes(st.u, pw, contact, P, z, z + 100)
        p = melee.hit_chance(torch.tensor(20.0), torch.tensor(45.0), P.sim["melee"]["hit_slope"])
        eight = 8 * float(p) * float(hit[0, 0, 1]) / 5.7
        assert float(rate[0, 0, 1]) == pytest.approx(eight, rel=1e-4)

    def test_a_charge_hits_harder(self):
        st = face_off(SPEAR, SLAVE)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        steady = melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, 0, 1]
        st.u["charge"][0, 0] = 1.0
        charged = melee.strikes(st.u, pw, contact, P, st.u["charge"], z)[0][0, 0, 1]
        assert float(charged) > 2 * float(steady)


class TestMissile:
    def test_a_shield_blocks_from_the_front_only(self):
        st = face_off(ARCHER, "wh2_main_skv_cha_warlord_0", gap=100)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        target = torch.tensor([[1, -1]])
        _, front, _ = missile.volley(st.u, pw, target, 1.0, P)
        st.u["b"][0, 1] = 90.0
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        _, back, _ = missile.volley(st.u, pw, target, 1.0, P)
        assert float(back.sum()) > float(front.sum()) > 0

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

    def test_out_of_range_nobody_is_chosen(self):
        st = face_off(ARCHER, SLAVE, gap=200)
        pw = geometry.pairwise(st.u, 1.5)
        can = torch.tensor([[True, False]])
        t = missile.choose_target(st.u, pw, can, torch.tensor([[-1, -1]]), torch.tensor([[False, False]]))
        assert t.tolist() == [[-1, -1]]


class TestMorale:
    def ctx(self, st, **over):
        z = torch.zeros_like(st.u["men"], dtype=torch.bool)
        c = {"aura": z, "lord_dead_points": torch.zeros_like(st.u["men"]), "neighbour": z, "in_melee": z,
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

    def test_a_rear_attack_routs_sooner(self):
        def time_to_rout(flank):
            st = face_off(SPEAR, SLAVE)
            st.u["hp_abs"][0, 0] = 0.1 * st.u["hp0"][0, 0]
            st.u["flank_hit"][0, 0] = flank
            for n in range(400):
                morale.step(st.u, self.ctx(st, in_melee=torch.ones_like(st.u["r"])), P, 0.5)
                if st.u["r"][0, 0]:
                    return n
            return 999
        front, flank, rear = time_to_rout(0), time_to_rout(1), time_to_rout(2)
        assert rear < front and rear <= flank <= front

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
        assert not st.u["r"][0, 0] and st.u["mp"][0, 0] >= P.sim["morale"]["rally_mp"]
        st.u["rout_count"][0, 0] = 2
        st.u["rally_s"][0, 0] = 100.0
        st.u["morale"][0, 0] = -20.0
        morale.step(st.u, self.ctx(st), P, 0.5)
        assert st.u["s"][0, 0] and st.u["ms"][0, 0] == 7


# --- level 3: the battle loop ---

class TestBattle:
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
        for _ in range(20):
            battle.step(st, o, P)
        assert not bool(st.u["m"][0, 0]) and float(st.u["x"][0, 0]) < x0 - 15

    def _charge(self, defender_key, run_defender):
        """A spearmen unit charges defender_key head-on; returns the step's state after contact."""
        st = scenario.build([army([(defender_key, 0, 0, 90)], [(SPEAR, 60, 0, 270)])], P)
        H = st.N // 2
        for _ in range(80):
            o = replay.hold(st)
            o.kind[0, H], o.target[0, H], o.run[0, H] = O.ATTACK, 0, True
            if run_defender:
                o.kind[0, 0], o.target[0, 0], o.run[0, 0] = O.ATTACK, H, True
            battle.step(st, o, P)
            if st.u["m"][0, 0]:
                return st, H
        raise AssertionError("no contact")

    def test_braced_spearmen_meet_a_charge_as_a_charge(self):
        braced, H = self._charge(SPEAR, run_defender=False)       # spearmen: charge_reflection
        assert float(braced.u["charge"][0, H]) > 0.5 and float(braced.u["charge"][0, 0]) > 0.5
        caught, H = self._charge(SLAVE, run_defender=False)       # slaves: no charge_reflection
        assert float(caught.u["charge"][0, H]) > 0.5 and float(caught.u["charge"][0, 0]) == 0

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
        ctx = {"aura": z, "lord_dead_points": torch.zeros_like(st.u["men"]), "neighbour": z, "in_melee": z,
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
