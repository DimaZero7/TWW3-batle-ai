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

    def test_state_and_orders_shapes_and_perspective(self):
        st = S.empty(2, 6)
        assert all(v.shape == (2, 6) for v in st.u.values())
        obs = st.observation()
        timers = {f"ab{k}_{t}" for k in range(3) for t in ("on", "cd")}
        assert set(obs) == set(S.OBSERVED) | {"side", "t", "fx_on", "gone"} | timers and obs["t"].shape == (2,)
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
        F = {}
        for mode in ("min", "striker"):
            _, _, sector, f = melee.strikes(st.u, pw, contact, P.with_cal("melee", flank_face=mode), z, z + 100)
            assert int(sector[0, 0, H]) == 1
            F[mode] = float(f[0, 0, H])
        ff, sp = P.sim["melee"]["fighting_files"], P.sim["formation"]["spacing_m"]
        assert F["min"] == pytest.approx(ff * float(depth[0, H]) / sp, rel=1e-4)
        assert F["striker"] == pytest.approx(ff * float(front[0, 0]) / sp, rel=1e-4)
        assert F["striker"] > F["min"]

    def test_at_most_the_cap_reach_a_lord(self):
        st = face_off(SPEAR, GENERAL)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        st.u["order_kind"][0, 0] = O.ATTACK
        st.u["order_target"][0, 0] = st.N // 2   # attacking the lord: the whole rate
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
        st.u["order_kind"][0, H] = O.ATTACK
        st.u["order_target"][0, H] = 0          # attacking the lord: the whole rate
        full = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, H, 0])
        st.u["order_target"][0, H] = 1          # told to attack another slot, touching the lord
        busy = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, H, 0])
        assert busy == pytest.approx(full * P.sim["contact"]["lord_incidental"], rel=1e-4)

    def test_a_unit_attacking_another_enemy_strikes_a_unit_it_touches_at_its_share(self):
        st = face_off(SPEAR, SLAVE)
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= 1.0)
        z = torch.zeros_like(st.u["men"])
        H = st.N // 2
        st.u["order_kind"][0, 0] = O.ATTACK
        st.u["order_target"][0, 0] = H         # its own target: the whole rate
        full = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, 0, H])
        st.u["order_target"][0, 0] = H + 1     # told to attack another slot, touching the slaves
        busy = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, 0, H])
        assert 0 < P.sim["contact"]["unit_incidental"] < 1
        assert busy == pytest.approx(full * P.sim["contact"]["unit_incidental"], rel=1e-4)

    def test_a_melee_unit_holding_in_melee_strikes_at_the_hold_rate_a_missile_unit_in_full(self):
        k = P.sim["contact"]["hold_rate"]
        assert 0 < k < 1
        for key, share in ((SPEAR, k), (ARCHER, 1.0)):
            st = face_off(key, SLAVE)
            pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
            contact = pw["enemy"] & (pw["gap"] <= 1.0)
            z = torch.zeros_like(st.u["men"])
            H = st.N // 2
            st.u["order_kind"][0, 0] = O.ATTACK
            st.u["order_target"][0, 0] = H
            attacking = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, 0, H])
            st.u["order_kind"][0, 0] = O.HOLD
            st.u["order_target"][0, 0] = -1
            held = float(melee.strikes(st.u, pw, contact, P, z, z + 100)[0][0, 0, H])
            assert attacking > 0 and held == pytest.approx(share * attacking, rel=1e-5), key

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

    def test_a_lord_strikes_a_lord_at_lord_v_lord_of_the_rule(self):
        st, pw, contact, z = self.surrounded(0, rival=True)
        H = st.N // 2
        full = float(melee.strikes(st.u, pw, contact, P.with_cal("contact", lord_v_lord=1.0), z, z + 100)[0][0, H, 0])
        less = float(melee.strikes(st.u, pw, contact, P.with_cal("contact", lord_v_lord=0.73), z, z + 100)[0][0, H, 0])
        assert full > 0 and less == pytest.approx(0.73 * full, rel=1e-4)

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

    def test_a_standing_shooter_turns_to_a_target_behind_before_it_aims(self):
        """missile.stand_fire_arc_deg + turn.formation_deg_s: standing, a shooter facing away from its only
        target turns at the formation rate and aims only once the target is within the arc."""
        def first_shot(params, steps=24):
            st = scenario.build([army([(ARCHER, -100, 0, 270)], [(SLAVE, 0, 0, 270)])], params)
            b0, bearings = float(st.u["b"][0, 0]), []
            for k in range(steps):
                battle.step(st, replay.hold(st), params)
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
        off = P.with_cal("missile", stand_fire_arc_deg=180).with_cal("turn", formation_deg_s=0)
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

    def test_a_first_strike_in_the_rear_costs_more_than_in_the_flank(self):
        # The database's was_attacked_in_flank / _rear (-6 / -14) for the step a worse side is first
        # struck (sim.json morale.attacked_event; battle.py gives the points).
        after = []
        for pts in (0.0, -6.0, -14.0):
            st = face_off(SPEAR, SLAVE)
            st.u["morale"] = morale.target_points(st.u, self.ctx(st), P)
            morale.step(st.u, self.ctx(st, flank_event=torch.full_like(st.u["men"], pts)), P, 0.5)
            after.append(float(st.u["morale"][0, 0]))
        assert after[2] < after[1] < after[0]

    def test_the_army_beaten_as_a_whole_routs(self):
        # Army destruction: -120 to every unit of the side (battle.py decides when).
        st = face_off(SPEAR, SLAVE)
        hit = torch.ones_like(st.u["r"])
        for _ in range(8):
            morale.step(st.u, self.ctx(st, collapse=hit), P, 0.5)
        assert bool(st.u["r"].all())

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
        # A shattered lord is lost as if he had died: his fall is timed (lord_dead_s) and his army
        # loses sim.json morale.lord_fall (measured: about his aura only, so 0 / 0 on top of it).
        # Battle 0: side 1's General shattered (far off, so his rout and his aura touch no one);
        # battle 1: the same, the General steady.
        sides = army([(GENERAL, -900, 0, 90, True), (SPEAR, -300, 0, 90)], [(SPEAR, 300, 0, 270)])
        st = scenario.build([sides, sides], P)
        st.u["r"][0, 0] = st.u["s"][0, 0] = True
        st.u["morale"][0, 0] = -20.0
        for _ in range(4):
            battle.step(st, replay.hold(st), P)
        assert float(st.u["men"][0, 0]) > 0 and not bool(st.u["gone"][0, 0])      # lost by shattering alone
        assert float(st.lord_dead_s[0, 0]) == pytest.approx(1.5) and float(st.lord_dead_s[0, 1]) == -1
        assert float(st.lord_dead_s[1, 0]) == -1
        fall = P.sim["morale"]["lord_fall"]["recent"]
        assert float(st.u["morale"][0, 1]) <= float(st.u["morale"][1, 1]) + min(fall, 0) * 0.15 + 1e-4
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

    @pytest.mark.parametrize("key", [ARCHER, SPEAR])
    def test_a_unit_told_to_move_away_leaves_melee_when_leave_m_is_on(self, key):
        """contact.leave_m: any unit (missile or melee) told to move that far walks out of the fight."""
        moved = {}
        for leave in (0.0, 10.0):
            params = P.with_cal("contact", leave_m=leave, pin_s=0.0)
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

    @pytest.mark.parametrize("key", [ARCHER, SPEAR])
    def test_a_missile_unit_leaving_melee_is_pinned_for_pin_s_a_melee_unit_is_not(self, key):
        """contact.pin_s: a missile unit told to leave melee stays held (in contact, struck) for pin_s seconds,
        then walks out; a melee unit walks out at once (its enemies follow it, as in the game)."""
        params = P.with_cal("contact", leave_m=10.0, pin_s=5.0)
        st = face_off(key, SPEAR)
        H = st.N // 2
        o = replay.hold(st)
        o.kind[0, H], o.target[0, H] = O.ATTACK, 0
        battle.step(st, o, params)
        x0, hp0 = float(st.u["x"][0, 0]), float(st.u["hp_abs"][0, 0])
        xs = []
        for _ in range(20):
            o = replay.hold(st)
            o.kind[0, 0], o.x[0, 0], o.z[0, 0], o.run[0, 0] = O.MOVE, -200.0, 0.0, True
            battle.step(st, o, params)
            xs.append(x0 - float(st.u["x"][0, 0]))
        held = int(5.0 / params.dt)
        if key == ARCHER:
            assert max(xs[:held]) == pytest.approx(0.0, abs=0.01) and float(st.u["hp_abs"][0, 0]) < hp0
            assert xs[-1] > 10 and float(st.u["leave_s"][0, 0]) == 0      # out: the count is over
        else:
            assert xs[2] > 1 and xs[-1] > 10

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
        assert r["passive"] == 1 and r["modelled"] == 0          # innate: tools/nn/sim/effects.py lays it on

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
        assert float(u["defence"][0, 1]) == defence                        # Hold the Line: an innate effect
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

    def test_flagellants_surrounded_fight_on_and_the_penitent_fires_when_losing(self):
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
        u["taken"][0, 0], u["dealt"][0, 0] = 500.0, 100.0          # losing the melee
        engaged = standing.clone()
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

    def _spent(self, shooter, side1_more=(), enemies=((SPEAR, -20, 0, 270),), steps=40, orders=None):
        st = scenario.build([army([(shooter, -100, 0, 90), *side1_more], list(enemies))], P)
        for _ in range(steps):
            o = replay.hold(st) if orders is None else orders(st)
            battle.step(st, o, P)
        return float(st.u["ammo0"][0, 0] - st.u["a"][0, 0]), st

    def test_direct_fire_is_blocked_by_a_friendly_unit_in_between_arcing_fire_is_not(self):
        screen = ((SPEAR, -60, 0, 90),)
        open_m, _ = self._spent(MILITIA)
        blocked_m, st = self._spent(MILITIA, screen)
        assert bool(st.u["direct"][0, 0]) and open_m > 0 and blocked_m == 0
        open_a, _ = self._spent(ARCHER)
        over_a, st = self._spent(ARCHER, screen)
        assert not bool(st.u["direct"][0, 0]) and open_a > 0 and over_a == pytest.approx(open_a, rel=0.05)
        # a screen half way, off to one side, blocks about half of the men (their lines converge on the
        # target's centre: half way a line is at half its offset)
        part_m, _ = self._spent(MILITIA, ((SPEAR, -60, 16, 90),))
        assert 0.2 * open_m < part_m < 0.8 * open_m

    def test_a_blocked_direct_fire_unit_shoots_another_target_it_can_see(self):
        _, st = self._spent(MILITIA, ((SPEAR, -60, 0, 90),), enemies=((SPEAR, -20, 0, 270), (SLAVE, -30, 60, 270)),
                            steps=30)
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
        """sim.json effects.off leaves Single Entity out (the recordings show no such speed drop); the
        mechanism itself, with the switch empty: below 25 % health, speed x0.9 and damage x0.8."""
        import dataclasses
        import json
        sim = json.loads(json.dumps(P.sim))
        assert "wh3_main_unit_passive_single_entity" in sim["effects"]["off"]
        sim["effects"]["off"] = []
        Q = dataclasses.replace(P, sim=sim)
        for params, on in ((Q, True), (P, False)):
            st = scenario.build([army([(GENERAL, -100, 0, 90, True)], [(WARLORD, 100, 0, 270, True)])], params)
            H = st.N // 2
            u = st.u
            dmg, run = float(u["damage"][0, H]), float(u["run"][0, H])
            _, a = innate(st, params=params)
            assert float(a["damage"][0, H]) == dmg
            u["hp"][0, H] = 0.2
            _, a = innate(st, params=params)
            k = (0.8, 0.9) if on else (1.0, 1.0)
            assert float(a["damage"][0, H]) == pytest.approx(k[0] * dmg) and float(a["run"][0, H]) == pytest.approx(k[1] * run)
            assert int(st.u["fx_on"][0, H]) & bit("wh3_main_unit_passive_single_entity")     # the network sees it

    def test_hold_the_line_reaches_friends_in_range_from_a_standing_lord(self):
        st = scenario.build([army([(GENERAL, 0, 0, 90, True), (SPEAR, 0, 30, 90), (SPEAR, 0, 120, 90)],
                                  [(SLAVE, 20, 0, 270)])], P)
        H = st.N // 2
        u = st.u
        d = [float(u["defence"][0, i]) for i in (1, 2, H)]
        _, a = innate(st)
        assert [float(a["defence"][0, i]) for i in (1, 2, H)] == [d[0] + 5, d[1], d[2] + 8]   # (slaves: SiN)
        assert float(a["morale_bonus"][0, 1]) == float(u["morale_bonus"][0, 1]) + 4
        u["r"][0, 0] = True                                        # a routing lord holds no line
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
