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
        assert float(u["sh_r"][0, 0]) > 5 and int(u["sh_t"][0, 0]) == H

    @staticmethod
    def _shot(gun, target_key, dist=200.0, friend=None, gap=0.0, swap=False):
        """One volley of 4 (rockets 12) at the target's men dist m off, launched and about to land: (state, H)."""
        side2 = [(target_key, 0, dist, 180)]
        side1 = [(gun, 0, 0, 0)] + ([(friend, 0, dist - gap, 0)] if friend else [])
        skv = gun in (CLAW, WARP)
        st = build(side1, side2, f1="wh2_main_skv_skaven" if skv else "wh_main_emp_empire",
                   f2="wh_main_emp_empire" if skv else "wh2_main_skv_skaven")
        u, H = st.u, st.N // 2
        pw = geometry.pairwise(u, SP)
        shots = torch.zeros_like(u["men"])
        shots[0, 0] = float(artillery.shooters(u)[0, 0])
        target = torch.full_like(u["side"], -1)
        target[0, 0] = H
        artillery.launch(u, pw, P, shots, target)
        u["sh_s"][0, 0] = 0.05
        return st, H

    def _land(self, gun, target_key, friend=None, gap=0.0, r=None, dist=250.0):
        """The volley of _shot landing now: (hp [B, i, j], kills, hits, knocks, state, H); r: its spread's side."""
        st, H = self._shot(gun, target_key, dist=dist, friend=friend, gap=gap)
        u = st.u
        if r is not None:
            u["sh_r"][0, 0] = r
        front, depth = geometry.dims(u, SP)
        hp, kills, hits, knocks = artillery.land(u, P, P.dt, front, depth, torch.zeros_like(u["r"]))
        assert float(u["sh_s"][0, 0]) == -1.0
        return hp, kills, hits, knocks, st, H

    def test_the_blast_strikes_the_target_and_friends_near_it(self):
        hp, kills, hits, knocks, st, H = self._land(MORTAR, CSPEAR, friend=SPEAR, gap=12.0, r=60.0)
        assert float(hp[0, 0, H]) > 0 and float(kills[0, 0, H]) > 0
        assert float(hp[0, 0, 1]) > 0                                   # the friends 12 m nearer: affects_allies
        assert float(knocks[0, 0, H]) > 0
        far = self._land(MORTAR, CSPEAR, friend=SPEAR, gap=80.0)
        assert float(far[0][0, 0, 1]) == 0.0
        # a tighter spread puts more of the shells on the block
        tight, _, _, _, _, H1 = self._land(MORTAR, CSPEAR, r=4.0)
        wide = self._land(MORTAR, CSPEAR, r=120.0)[0]
        assert float(tight[0, 0, H1]) > float(wide[0, 0, H1]) > 0

    def test_a_direct_shells_blast_has_no_own_hit_a_fixed_ones_has(self):
        # the rockets' own hit is the missile's (missile.volley): their landing is the blast alone; the Mortar's own hit
        # lands with its blast - remove the blast and the hit stays
        rockets = self._land(ROCKETS, CSPEAR, r=4.0)[0]
        assert float(rockets.sum()) > 0
        st, H = self._shot(MORTAR, CSPEAR, dist=250.0)
        u = st.u
        u["ex_r"][0, 0] = 0.0
        u["sh_r"][0, 0] = 4.0
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


class TestTheMiss:
    """Where a gun's shells burst (build/diag_arty/mechanic.md; R-work shooting_module.md 2.2-2.3): the aim, the
    barrel's yaw and pitch, the arc's range, the first man, the zeroing-in."""

    @staticmethod
    def _bursts(gun, dist=200.0):
        st, H = TestShells._shot(gun, CSPEAR, dist=dist)
        u = st.u
        front, depth = geometry.dims(u, SP)
        b_, i_ = torch.tensor([0]), torch.tensor([0])
        Pq = artillery.bursts(u, front, depth, b_, i_)[0]                       # [Q, 2]
        # the line of fire runs along +z (the target dead ahead): along = z - the target's centre, across = x
        return Pq[:, 1] - float(u["z"][0, H]), Pq[:, 0] - float(u["x"][0, H]), st, H, float(depth[0, H])

    def test_the_mortar_aims_at_the_centre_the_rockets_at_the_near_edge(self):
        assert bool(P.static(MORTAR)["central"]) and not bool(P.static(ROCKETS)["central"])
        st, H = TestShells._shot(MORTAR, CSPEAR)
        u = st.u
        front, depth = geometry.dims(u, SP)
        beyond = artillery.AIM_H * float(u["height"][0, H]) / math.tan(math.radians(50))
        assert float(u["sh_z"][0, 0]) == pytest.approx(float(u["z"][0, H]) + beyond, abs=1e-3)
        assert float(u["sh_aa"][0, 0]) == 0.0
        st, H = TestShells._shot(ROCKETS, CSPEAR)
        u = st.u
        front, depth = geometry.dims(u, SP)
        el = float(u["sh_el"][0, 0])
        near = float(u["z"][0, H]) - float(depth[0, H]) / 2
        assert float(u["sh_z"][0, 0]) == pytest.approx(near + artillery.AIM_H * float(u["height"][0, H]) / math.tan(el),
                                                       abs=1e-3)
        assert 0 < float(u["sh_aa"][0, 0]) <= float(front[0, H]) / 2

    def test_the_spread_by_accuracy_distance_and_angle(self):
        # the Mortar at 200 m: p = 0.2 x (300 / 200)^2 x (1 - 50 / 180) = 0.325, r = sqrt(380 / p) = 34.2 m
        st, H = TestShells._shot(MORTAR, CSPEAR, dist=200.0)
        u = st.u
        d = float(u["sh_d"][0, 0])
        p = 0.2 * (300 / d) ** 2 * (1 - 50 / 180)
        assert float(u["sh_r"][0, 0]) == pytest.approx(math.sqrt(380 / p), rel=1e-4)
        assert 30 < float(u["sh_r"][0, 0]) < 38

    def test_the_mortars_miss_is_across_the_rockets_along(self):
        # the Mortar's 50 deg arc: the pitch moves its range little (-9..+3 m at 200 m), the yaw +-17 m across; the
        # rockets' 11 deg arc: the pitch +-55 m along
        along, across, st, H, depth = self._bursts(MORTAR)
        assert 14 < float(across.abs().max()) < 18
        assert -11 < float(along.min()) and float(along.max()) < 6
        along_r, across_r, st, H, depth = self._bursts(ROCKETS)
        free = along_r[(along_r > depth / 2 + 10) | (along_r < -depth / 2 - 1)]
        assert float(along_r.max()) > 40 and float(along_r.min()) < -30 and len(free) > 0.5 * len(along_r)

    def test_a_falling_shell_bursts_on_the_first_man(self):
        # a rocket landing past the block's near edge within the shadow h / tan(angle) bursts on the block: none lands
        # in the shadow's reach past its far edge across its span
        along, across, st, H, depth = self._bursts(ROCKETS)
        u = st.u
        el = float(u["sh_el"][0, 0])
        shadow = float(u["height"][0, H]) / math.tan(el)
        front = float(geometry.dims(u, SP)[0][0, H])
        lat = across.abs() <= front / 2 - 0.5
        between = lat & (along > depth / 2 + 0.1) & (along < depth / 2 + shadow - 0.1)
        assert int(between.sum()) == 0
        on = lat & (along.abs() <= depth / 2 + 1e-3)
        assert 0.05 < float(on.float().mean()) < 0.35            # build/diag_arty/mechanic.md: 12-17 %

    def test_the_zeroing_in_narrows_the_spread_and_a_new_target_starts_it_again(self):
        st = build([(MORTAR, 0, 0, 0)], [(CSPEAR, 0, 200, 180), (CSPEAR, 40, 200, 180)])
        u, H = st.u, st.N // 2
        pw = geometry.pairwise(u, SP)
        shots = torch.zeros_like(u["men"])
        shots[0, 0] = 4
        target = torch.full_like(u["side"], -1)
        target[0, 0] = H
        artillery.launch(u, pw, P, shots, target)
        r0 = float(u["sh_r"][0, 0])
        for n in range(1, 8):
            artillery.launch(u, pw, P, shots, target)
            assert float(u["sh_r"][0, 0]) == pytest.approx(r0 * math.sqrt(max(1 - 0.1 * n, 0.5)), rel=1e-5), n
        target[0, 0] = H + 1                     # a new target: the count starts again (its distance a little longer)
        artillery.launch(u, pw, P, shots, target)
        assert float(u["sh_r"][0, 0]) > r0

    def test_the_blast_is_measured_from_the_body_s_edge(self):
        # artillery_rules.md 1.1-1.2: a man whose body reaches the ring (d^2 - r^2 < R^2) takes 1 - (d - r) / R
        Pq = torch.tensor([[[0.0, 0.0]]])
        Cx = torch.tensor([[[[0.5, 0.0], [3.5, 0.0], [6.03, 0.0], [6.1, 0.0]]]])
        rb, re = torch.tensor([[0.65]]), torch.tensor([6.0])
        m1, m2, pk, pc = artillery.exposure(Pq, Cx, rb, re, 1.0)
        assert [round(float(x), 3) for x in m1[0, 0]] == [1.0, round(1 - 2.85 / 6, 3), round(1 - 5.38 / 6, 3), 0.0]
        assert [float(x) for x in pk[0, 0]] == [1.0, 1.0, 0.0, 0.0]


class TestTheCycle:
    def test_a_guns_cycle_is_its_reload_and_its_measured_animations(self):
        # config/nn/sim.json artillery.cycle_add_s (the artillery probe: the Mortar every 25.5 s on reload 19)
        add = P.sim["artillery"]["cycle_add_s"]
        s = P.static(MORTAR)
        assert s["reload"] == pytest.approx(19.0 + add["wh_emp_mortar_shell"]) and s["rl_add"] == add["wh_emp_mortar_shell"]
        assert P.static(ROCKETS)["reload"] == pytest.approx(17.0 + add["wh_emp_rocket"])
        assert P.static(ARCHERS)["rl_add"] == 0.0
