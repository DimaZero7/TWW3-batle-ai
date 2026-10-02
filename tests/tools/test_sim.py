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
SHIELD_SPEAR, SWORD = "wh_main_emp_inf_spearmen_1", "wh_main_emp_inf_swordsmen"


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
        timers = {f"ab{k}_{t}" for k in range(3) for t in ("on", "cd")}
        assert set(obs) == set(S.OBSERVED) | {"side", "t"} | timers and obs["t"].shape == (2,)
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

    def test_at_most_the_cap_reach_a_lord(self):
        st = face_off(SPEAR, GENERAL)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        rate, hit, _, _ = melee.strikes(st.u, pw, contact, P, z, z + 100)
        p = melee.hit_chance(torch.tensor(20.0), torch.tensor(45.0), P.sim["melee"]["hit_slope"])
        cap = P.sim["contact"]["lord_max_attackers"]
        assert float(rate[0, 0, 1]) == pytest.approx(cap * float(p) * float(hit[0, 0, 1]) / 5.7, rel=1e-4)

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
        return st, pw, contact, z

    def test_more_units_round_a_lord_share_the_same_men(self):
        # The lord swarm probe: 1-4 spear units take from a lord the same HP/s.
        lost = []
        for n in (1, 2, 4):
            st, pw, contact, z = self.surrounded(n)
            rate, _, _, F = melee.strikes(st.u, pw, contact, P, z, z + 100)
            lost.append(float(rate[0, :, 0].sum()))
            assert float(F[0, :, 0].sum()) == pytest.approx(P.sim["contact"]["lord_max_attackers"], rel=1e-4)
        assert lost[1] == pytest.approx(lost[0], rel=0.01) and lost[2] == pytest.approx(lost[0], rel=0.01)

    def test_a_unit_attacking_another_enemy_barely_strikes_a_lord_it_touches(self):
        st, pw, contact, z = self.surrounded(2)
        H = st.N // 2
        full = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, H, 0])
        st.u["order_kind"][0, H] = O.ATTACK
        st.u["order_target"][0, H] = 1          # told to attack another slot, touching the lord
        busy = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, H, 0])
        assert busy == pytest.approx(full * P.sim["contact"]["lord_incidental"], rel=1e-4)

    def test_the_enemy_lord_keeps_his_blow_and_the_infantry_counts_less(self):
        st, pw, contact, z = self.surrounded(0, rival=True)
        alone = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, :, 0].sum())
        st, pw, contact, z = self.surrounded(3, rival=True)
        rate = melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, :, 0]
        H = st.N // 2
        assert float(rate[H]) == pytest.approx(alone, rel=1e-4)
        st3, pw3, c3, z3 = self.surrounded(3)
        infantry = float(melee.strikes(st3.u, pw3, c3, P, z3, z3 + 100)[0][0, :, 0].sum())
        k = P.sim["contact"]["lord_rival_others"]
        # The rival takes one of the cap's places: the infantry share the rest.
        cap = P.sim["contact"]["lord_max_attackers"]
        assert float(rate.sum()) - alone == pytest.approx(k * infantry * (cap - 1) / cap, rel=1e-3)

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
            rates[key] = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, 0, 1])
        assert rates[SWORD] > 1.5 * rates[SPEAR] > 0

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
        assert float(hp[0, H + 1, 1]) > float(hp[0, H + 1, 0]) > 0   # the crowd takes more than the lord
        free = contact.clone()
        free[0, 1, H] = free[0, H, 1] = False                 # our spearmen out of melee: the ordinary spill
        _, out, _ = missile.volley(st.u, pw, target, 1.0, P, contact=free)
        ratio = float(hp[0, H + 1, 1]) / float(out[0, H + 1, 1])
        d = float(pw["dist"][0, 0, 1])
        expect = float(missile.distance_factor(torch.tensor(d), P.sim["missile"]["spill_melee"]))             / float(missile.distance_factor(torch.tensor(d), P.sim["missile"]["spill"]))
        assert ratio == pytest.approx(expect, rel=1e-3)

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

    def test_a_shattered_lord_counts_as_lost_for_morale(self):
        # Gate runs 02.10.2026: when a lord shatters, his whole army loses 0.5-0.6 of its leadership
        # in that second, as if he had died. Battle 0: side 1's General shattered (far off, so his
        # rout and his aura touch no one); battle 1: the same, the General steady.
        sides = army([(GENERAL, -900, 0, 90, True), (SPEAR, -300, 0, 90)], [(SPEAR, 300, 0, 270)])
        st = scenario.build([sides, sides], P)
        st.u["r"][0, 0] = st.u["s"][0, 0] = True
        st.u["morale"][0, 0] = -20.0
        for _ in range(4):
            battle.step(st, replay.hold(st), P)
        assert float(st.u["men"][0, 0]) > 0 and not bool(st.u["gone"][0, 0])      # lost by shattering alone
        assert float(st.lord_dead_s[0, 0]) == pytest.approx(1.5) and float(st.lord_dead_s[0, 1]) == -1
        assert float(st.lord_dead_s[1, 0]) == -1
        assert float(st.u["morale"][0, 1]) < float(st.u["morale"][1, 1]) - 3
        assert not bool(st.done[0])

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
        assert r["passive"] == 1 and r["modelled"] == 0          # shown to the network, no effect here

    def test_the_game_ai_fires_by_its_rule_and_passives_hold_for_all(self):
        from tools.nn.sim import abilities
        st, H, dist, same, standing, engaged = self.lords()
        u = st.u
        assert not bool(u["ai"][0, 0]) and bool(u["ai"][0, H])
        engaged[0, 0] = engaged[0, H] = True
        dmg, ap, defence, run = (float(u[k][0, i]) for k, i in (("damage", H), ("ap_damage", H), ("defence", 1),
                                                                    ("run", H)))
        old = abilities.apply(u, P, 0.5, standing, engaged, dist, same)
        assert float(u["damage"][0, H]) == pytest.approx(1.25 * dmg)      # Deadly Onslaught (in melee)
        assert float(u["ap_damage"][0, H]) == pytest.approx(1.25 * ap)
        assert float(u["run"][0, H]) == pytest.approx(1.25 * run)         # Verminous Valour (enemy near)
        assert float(u["ab1_on"][0, H]) == 31 and float(u["ab1_cd"][0, H]) == 31 + 90
        assert float(u["ab0_on"][0, 0]) == 0 and float(u["ab1_on"][0, 0]) == 0   # our General: no order, no use
        assert float(u["defence"][0, 1]) == defence + 5                    # Hold the Line reaches our spearmen
        abilities.restore(u, old)
        assert float(u["damage"][0, H]) == dmg
        abilities.apply(u, P, 31.0, standing, engaged, dist, same)  # 31 s later: over, recharging
        assert float(u["ab1_on"][0, H]) == 0 and float(u["ab1_cd"][0, H]) == 90

    def test_the_network_fires_a_ready_self_cast_ability_by_order(self):
        from tools.nn.sim import abilities
        st, H, dist, same, standing, engaged = self.lords()
        u = st.u
        defence = float(u["defence"][0, 1])
        use = torch.full_like(u["ab0"], -1)
        use[0, 0] = 1                                                      # Stand Your Ground
        old = abilities.apply(u, P, 0.5, standing, engaged, dist, same, use)
        assert float(u["ab1_on"][0, 0]) == 18 and float(u["ab1_cd"][0, 0]) == 18 + 90
        assert float(u["defence"][0, 1]) == defence + 24 + 5              # + Hold the Line
        assert float(u["defence"][0, H]) == float(old["defence"][0, H])   # friends only, not the enemy
        abilities.restore(u, old)
        abilities.apply(u, P, 0.5, standing, engaged, dist, same, use)     # active: a new order does nothing
        assert float(u["ab1_on"][0, 0]) == pytest.approx(17.5)
        use[0, 0] = 2                                                      # Hold the Line is passive
        cd = [float(u[f"ab{k}_cd"][0, 0]) for k in range(3)]
        abilities.apply(u, P, 0.5, standing, engaged, dist, same, use)
        assert float(u["ab2_on"][0, 0]) == 0 and float(u["ab2_cd"][0, 0]) == 0
        assert [float(u[f"ab{k}_cd"][0, 0]) for k in range(2)] == [cd[0], pytest.approx(cd[1] - 0.5)]

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
