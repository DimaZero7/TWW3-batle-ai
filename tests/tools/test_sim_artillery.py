"""Artillery in the simulator (tools/nn/sim/artillery.py; config/nn/sim.json artillery; step 1, 29.10.2026): the guns
and their crew, the volley of guns x shots_per_volley, the minimum range, the shells' flight, landing and blast, the
fixed trajectory over friends, the guns' share of the damage. Needs torch (the container)."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import artillery, battle, geometry, missile, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
MORTAR, CANNON, VOLLEY, ROCKETS = ("wh_main_emp_art_mortar", "wh_main_emp_art_great_cannon",
                                   "wh_main_emp_art_helblaster_volley_gun", "wh_main_emp_art_helstorm_rocket_battery")
CLAW, WARP = "wh2_main_skv_art_plagueclaw_catapult", "wh2_main_skv_art_warp_lightning_cannon"
SPEAR, CSPEAR, KNIGHTS = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0", "wh_main_emp_cav_empire_knights"
ARCHERS, SWORD = "wh2_dlc13_emp_inf_archers_0", "wh_main_emp_inf_swordsmen"
SP = P.sim["formation"]["spacing_m"]


def army(side1, side2, f1="wh_main_emp_empire", f2="wh2_main_skv_skaven"):
    def units(rows):
        return [{"key": k, "x": x, "z": z, "b": b} for k, x, z, b in rows]
    return {"attacker": 1, "sides": {1: {"faction": f1, "units": units(side1)}, 2: {"faction": f2, "units": units(side2)}}}


def build(side1, side2, params=P, **kw):
    return scenario.build([army(side1, side2, **kw)], params)


def run(st, seconds, params=P, orders=None):
    for _ in range(int(round(seconds / params.dt))):
        battle.step(st, orders(st) if orders else O.hold(st.B, st.N), params, params.dt)


class TestTheUnit:
    def test_crew_guns_and_health(self):
        st = build([(MORTAR, 0, 0, 0)], [(CSPEAR, 0, 300, 180)])
        u = st.u
        assert float(u["hp_abs"][0, 0]) == 4356 and float(u["eng_w"][0, 0]) == 4 * 473 and float(u["eng"][0, 0]) == 4
        assert float(u["men"][0, 0]) == 44 and float(artillery.guns(u)[0, 0]) == 4
        # 3 crew a gun, the crew re-man the guns (artillery.crew "pool")
        for men, guns in ((44, 4), (12, 4), (11, 3), (6, 2), (3, 1), (2, 0)):
            u["men"][0, 0] = men
            assert float(artillery.guns(u)[0, 0]) == guns, men
        assert not bool(artillery.can_fire(u)[0, 0]) and bool(artillery.can_fire(u)[0, 1])
        u["men"][0, 0], u["eng"][0, 0] = 44, 1
        assert float(artillery.guns(u)[0, 0]) == 1
        # a volley: the firing guns x shots_per_volley; the others their men
        st = build([(VOLLEY, 0, 0, 0)], [(CSPEAR, 0, 200, 180)])
        assert float(artillery.shooters(st.u)[0, 0]) == 36 and float(st.u["a"][0, 0]) == 4 * 198
        H = st.N // 2
        assert float(artillery.shooters(st.u)[0, H]) == float(st.u["men"][0, H])

    def test_the_box_is_one_rank_of_guns(self):
        st = build([(MORTAR, 0, 0, 0), (CLAW, 100, 0, 0)], [(CSPEAR, 0, 300, 180)], f1="wh2_main_skv_skaven")
        front, depth = geometry.dims(st.u, SP)
        h, v = P.spacing_of(MORTAR)
        assert float(front[0, 0]) == pytest.approx(4 * h) and float(depth[0, 0]) == pytest.approx(v)
        h, v = P.spacing_of(CLAW)
        assert float(front[0, 1]) == pytest.approx(4 * h) and float(depth[0, 1]) == pytest.approx(v)

    def test_the_guns_take_their_share_and_are_lost_by_their_health(self):
        st = build([(CANNON, 0, 0, 0)], [(CSPEAR, 0, 300, 180)])
        u = st.u
        assert float(artillery.gun_share(u)[0, 0]) == pytest.approx(4 / 48)
        assert float(artillery.gun_share(u)[0, st.N // 2]) == 0.0
        artillery.guns_after(u, torch.tensor([[473.0 + 1.0] + [0.0] * (st.N - 1)]))
        assert float(u["eng_w"][0, 0]) == 4 * 473 - 474 and float(u["eng"][0, 0]) == 3
        u["men"][0, 0] = 0
        artillery.guns_after(u, torch.zeros_like(u["men"]))
        assert float(u["eng"][0, 0]) == 0 and float(u["eng_w"][0, 0]) == 0

    def test_swordsmen_at_the_guns_kill_crew_and_wear_the_guns(self):
        st = build([(MORTAR, 0, 0, 0)], [(SWORD, 0, 15, 180)], f2="wh_main_emp_empire")
        H = st.N // 2

        def attack(s):
            o = O.hold(s.B, s.N)
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            return o
        hp0 = float(st.u["hp_abs"][0, 0])
        run(st, 20, orders=attack)
        u = st.u
        lost = hp0 - float(u["hp_abs"][0, 0])
        gun_lost = 4 * 473 - float(u["eng_w"][0, 0])
        assert lost > 0 and float(u["men"][0, 0]) < 44
        # the guns' share of the HP taken: guns / (crew + guns) each step (4 / 48 at first, more as the crew falls)
        assert 0.08 * lost <= gun_lost <= 0.25 * lost
        # the unit's health is the crew's and the guns' (one number): the crew's within its men's
        crew = float(u["hp_abs"][0, 0]) - float(u["eng_w"][0, 0])
        men = float(u["men"][0, 0])
        assert 0 < crew <= men * 56 + 1e-3 and math.ceil(crew / 56 - 1e-6) <= men


class TestTargets:
    def test_minimum_range_edge_to_edge(self):
        for gap, ok in ((40.0, False), (70.0, True)):
            st = build([(MORTAR, 0, 0, 0)], [(CSPEAR, 0, 100, 180)])
            u, H = st.u, st.N // 2
            front, depth = geometry.dims(u, SP)
            u["z"][0, H] = float(depth[0, 0]) / 2 + float(depth[0, H]) / 2 + gap
            pw = geometry.pairwise(u, SP)
            assert bool(missile.in_range(u, pw)[0, 0, H]) == ok, gap
            assert bool(missile.min_range_ok(u, pw)[0, 0, H]) == ok
        # a unit without one: any gap (the archers 20 m off)
        st = build([(ARCHERS, 0, 0, 0)], [(CSPEAR, 0, 30, 180)])
        pw = geometry.pairwise(st.u, SP)
        assert bool(missile.min_range_ok(st.u, pw).all())

    def test_the_mortar_fires_over_friends_the_cannon_does_not(self):
        # spearmen 15 m ahead of the guns: the Mortar's 50 deg arc passes over them; the cannon's flat arc to a target 75 m
        # off stays below their heads, its line is closed (it would clear them shooting far: its arc rises -
        # missile.friend_los)
        fired = {}
        for gun, d in ((MORTAR, 200), (CANNON, 75)):
            st = build([(gun, 0, 0, 0), (SPEAR, 0, 15, 0)], [(CSPEAR, 0, d, 180)])
            H = st.N // 2
            hp0 = float(st.u["hp_abs"][0, H])
            run(st, 30)
            fired[gun] = (float(st.u["a"][0, 0]), hp0 - float(st.u["hp_abs"][0, H]))
        assert fired[MORTAR][0] < 88 and fired[MORTAR][1] > 0          # over the spearmen 15 m ahead
        assert fired[CANNON][0] == 88                                   # flat: the friends close its line, it holds


class TestShells:
    def test_flight_times(self):
        st = build([(MORTAR, 0, 0, 0), (CANNON, 50, 0, 0)], [(CSPEAR, 0, 300, 180)])
        d = torch.full_like(st.u["men"], 380.0)
        T = artillery.flight(st.u, d)
        assert float(T[0, 0]) == pytest.approx(math.sqrt(2 * 380 * math.tan(math.radians(50)) / 9.81), rel=1e-5)
        assert float(T[0, 0]) == pytest.approx(9.6, abs=0.1)                        # build/diag_arty/empire.md
        th = 0.5 * math.asin(380 * 9.81 / 100 ** 2)
        assert float(T[0, 1]) == pytest.approx(380 / (100 * math.cos(th)), rel=1e-5)

    def test_the_mortar_lands_after_aim_and_flight_and_hits_a_block(self):
        st = build([(MORTAR, 0, 0, 0)], [(CSPEAR, 0, 200, 180)])
        H = st.N // 2
        hp0 = float(st.u["hp_abs"][0, H])
        first = None
        for k in range(int(30 / P.dt)):
            battle.step(st, O.hold(st.B, st.N), P, P.dt)
            if first is None and float(st.u["hp_abs"][0, H]) < hp0 - 1e-3:
                first = float(st.t[0])
        flight = math.sqrt(2 * (200 - 0) * math.tan(math.radians(50)) / 9.81)
        aim = float(st.u["aim_s"][0, 0])
        assert first is not None and aim + flight - 1.0 <= first <= aim + flight + 1.5
        assert float(st.u["a"][0, 0]) == 84 or float(st.u["a"][0, 0]) == 80      # one or two volleys of 4

    def test_a_shell_leads_a_walking_target(self):
        st = build([(MORTAR, 0, 0, 0)], [(CSPEAR, 0, 250, 180)])
        u, H = st.u, st.N // 2
        u["vx"][0, H] = 1.5
        pw = geometry.pairwise(u, SP)
        shots = torch.zeros_like(u["men"])
        shots[0, 0] = 4
        target = torch.full_like(u["side"], -1)
        target[0, 0] = H
        artillery.launch(u, pw, P, shots, target)
        T = float(artillery.flight(u, pw["dist"][:, :, H])[0, 0])
        assert float(u["sh_s"][0, 0]) == pytest.approx(T) and float(u["sh_n"][0, 0]) == 4
        assert float(u["sh_x"][0, 0]) == pytest.approx(float(u["x"][0, H]) + 1.5 * T, rel=1e-5)
        assert float(u["sh_R"][0, 0]) > 5

    def _land(self, gun, target_key, friend=None, gap=0.0, R=None):
        """One volley of 4 shells landing on the target's centre now: (hp [B, i, j], hits, state, H)."""
        side2 = [(target_key, 0, 250, 180)]
        if friend:
            side1 = [(gun, 0, 0, 0), (friend, 0, 250 - gap, 0)]
        else:
            side1 = [(gun, 0, 0, 0)]
        st = build(side1, side2, f1="wh2_main_skv_skaven" if gun in (CLAW, WARP) else "wh_main_emp_empire",
                   f2="wh_main_emp_empire" if gun in (CLAW, WARP) else "wh2_main_skv_skaven")
        u, H = st.u, st.N // 2
        u["sh_s"][0, 0], u["sh_n"][0, 0] = 0.1, 4
        u["sh_x"][0, 0], u["sh_z"][0, 0] = float(u["x"][0, H]), float(u["z"][0, H])
        u["sh_R"][0, 0] = R if R is not None else 10.0
        front, depth = geometry.dims(u, SP)
        hp, kills, hits, knocks = artillery.land(u, P, P.dt, front, depth, torch.zeros_like(u["r"]))
        assert float(u["sh_s"][0, 0]) == -1.0
        return hp, kills, hits, knocks, st, H

    def test_the_blast_strikes_the_target_and_friends_near_it(self):
        hp, kills, hits, knocks, st, H = self._land(MORTAR, CSPEAR, friend=SPEAR, gap=8.0)
        assert float(hp[0, 0, H]) > 0 and float(kills[0, 0, H]) > 0
        assert float(hp[0, 0, 1]) > 0                                   # the friends 8 m off: affects_allies
        assert float(knocks[0, 0, H]) > 0
        far = self._land(MORTAR, CSPEAR, friend=SPEAR, gap=80.0)
        assert float(far[0][0, 0, 1]) == 0.0
        # a tighter landing disc puts more of the shells on the block
        tight, _, _, _, _, H1 = self._land(MORTAR, CSPEAR, R=4.0)
        wide = self._land(MORTAR, CSPEAR, R=30.0)[0]
        assert float(tight[0, 0, H1]) > float(wide[0, 0, H1]) > 0

    def test_a_direct_shells_blast_has_no_own_hit_a_fixed_ones_has(self):
        # the rockets' own hit is the missile's (missile.volley): their landing is the blast alone; the Mortar's own hit
        # lands with its blast - remove the blast and the hit stays
        rockets = self._land(ROCKETS, CSPEAR, R=4.0)[0]
        assert float(rockets.sum()) > 0
        st = build([(MORTAR, 0, 0, 0)], [(CSPEAR, 0, 250, 180)])
        u, H = st.u, st.N // 2
        u["ex_r"][0, 0] = 0.0
        u["sh_s"][0, 0], u["sh_n"][0, 0], u["sh_R"][0, 0] = 0.1, 4, 4.0
        u["sh_x"][0, 0], u["sh_z"][0, 0] = float(u["x"][0, H]), float(u["z"][0, H])
        front, depth = geometry.dims(u, SP)
        hp = artillery.land(u, P, P.dt, front, depth, torch.zeros_like(u["r"]))[0]
        assert float(hp[0, 0, H]) > 0

    def test_the_mortars_volley_shows_no_direct_hit(self):
        st = build([(MORTAR, 0, 0, 0)], [(CSPEAR, 0, 200, 180)])
        u, H = st.u, st.N // 2
        pw = geometry.pairwise(u, SP)
        target = torch.full_like(u["side"], -1)
        target[0, 0] = H
        shots, hp, _ = missile.volley(u, pw, target, P.dt, P, loaded=torch.ones_like(u["men"]))
        assert float(shots[0, 0]) == 4 and float(hp.sum()) == 0.0

    def test_the_warp_cannons_bonus_against_large(self):
        hits = {}
        for tgt in (KNIGHTS, SPEAR):
            st = build([(WARP, 0, 0, 0)], [(tgt, 0, 200, 180)], f1="wh2_main_skv_skaven", f2="wh_main_emp_empire")
            u, H = st.u, st.N // 2
            pw = geometry.pairwise(u, SP)
            target = torch.full_like(u["side"], -1)
            target[0, 0] = H
            u["hp_man"][0, H] = 10000.0          # (no overkill cap on the hit)
            hits[tgt] = missile.volley(u, pw, target, P.dt, P, loaded=torch.ones_like(u["men"]))[2][0, 0, H]
            u["m_bvl"][0, 0] = 0.0
            hits[tgt + "0"] = missile.volley(u, pw, target, P.dt, P, loaded=torch.ones_like(u["men"]))[2][0, 0, H]
        assert float(hits[KNIGHTS]) > float(hits[KNIGHTS + "0"])
        assert float(hits[SPEAR]) == float(hits[SPEAR + "0"])

    def test_the_catapults_blast_lays_its_contact_effect(self):
        st = build([(CLAW, 0, 0, 0)], [(SPEAR, 0, 250, 180)], f1="wh2_main_skv_skaven", f2="wh_main_emp_empire")
        assert int(st.u["ex_fx"][0, 0]) >= 0
        run(st, 40)
        H = st.N // 2
        assert any(float(st.u[f"cfx{j}_s"][0, H]) > 0 for j in range(4)) or float(st.u["hp_abs"][0, H]) < float(
            st.u["hp0"][0, H])


class TestNoArtillery:
    def test_a_battle_without_guns_steps_the_same_with_artillery_off(self):
        """The rules act only on a unit with guns: archers and spearmen step bit for bit the same with sim.json
        artillery on and off (the 36 generated battles: build/diag_arty/identity)."""
        off = P.with_cal("artillery", on=0)
        assert not off.has_art and P.has_art
        side1 = [(ARCHERS, 0, 0, 0), (SPEAR, 30, 20, 0)]
        side2 = [(CSPEAR, 0, 160, 180), (CSPEAR, 40, 170, 180)]
        a, b = build(side1, side2, params=P), build(side1, side2, params=off)

        def attack(s):
            o = O.hold(s.B, s.N)
            H = s.N // 2
            o.kind[0, H], o.target[0, H] = O.ATTACK, 0
            return o
        run(a, 60, params=P, orders=attack)
        run(b, 60, params=off, orders=attack)
        for k, v in b.u.items():
            if torch.is_tensor(v):
                assert torch.equal(a.u[k], v), k
