"""The open melee and whole-battle rules by the game (build/open_melee/spec.md R1-R4, R6, R7; build/open_battle/spec.md
1a, 2, 4, 5, 6; docs/en/training/simulator.md): the struck side's first strike, braced reflection and a move order in
contact against hold_rate, the chase out of melee and its 24 s window, the run-up towards the enemy, the wounded pool,
shattering by the database's thresholds, the rally blocked by any enemy and its clock, the charge into a free target
only, the flank and rear sectors. Needs torch (the container)."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, melee, morale, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SPEAR, SLAVE, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_skavenslave_spearmen_0", \
    "wh2_main_skv_inf_clanrat_spearmen_0"
SWORD, CLANRATS = "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1"
TOUCH = min(0.0, P.sim["contact"]["reach_m"])


def army(side1, side2, attacker=1):
    def units(rows):
        return [{"key": r[0], "x": r[1], "z": r[2], "b": r[3], "general": len(r) > 4 and r[4]} for r in rows]
    return {"attacker": attacker, "sides": {1: {"faction": "wh_main_emp_empire", "units": units(side1)},
                                            2: {"faction": "wh2_main_skv_skaven", "units": units(side2)}}}


def face_off(key1, key2, gap=0.0, b1=90.0):
    st = scenario.build([army([(key1, -50, 0, b1)], [(key2, 50, 0, 270)])], P)
    front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    gap = gap + TOUCH
    st.u["x"][0, 0] = -(depth[0, 0] / 2 + gap / 2)
    H = st.N // 2
    st.u["x"][0, H] = depth[0, H] / 2 + gap / 2
    return st


def orders(st, **units):
    """units: slot -> (kind, target or (x, z), run)."""
    o = replay.hold(st)
    for s, (kind, arg, run) in units.items():
        s = int(s)
        o.kind[0, s] = kind
        if kind == O.ATTACK:
            o.target[0, s] = arg
        elif arg is not None:
            o.x[0, s], o.z[0, s] = arg
        o.run[0, s] = run
    return o


# --- R1: a standing unit that is reached strikes at once ---

class TestStandFirstStrike:
    def charge_into(self, b_target):
        """Clanrats charge (14 m run-up) into held swordsmen (no charge reflection) facing them (b 90: the charger
        comes from +x) or facing away (270); HP the charger loses in the contact step."""
        st = face_off(SWORD, CLANRATS, gap=14.0, b1=b_target)
        H = st.N // 2
        lost = None
        for _ in range(20):
            hp0 = float(st.u["hp_abs"][0, H])
            m0 = bool(st.u["m"][0, 0])
            battle.step(st, orders(st, **{str(H): (O.ATTACK, 0, True)}), P)
            if bool(st.u["m"][0, 0]) and not m0:
                lost = hp0 - float(st.u["hp_abs"][0, H])
                break
        return lost

    def test_a_held_unit_charged_from_the_front_strikes_at_once_from_behind_not(self):
        front = self.charge_into(90.0)        # facing the charger (it comes from +x)
        back = self.charge_into(270.0)        # facing away
        assert front is not None and back is not None
        assert front > 30.0 and back < 0.25 * front

    def test_the_first_swing_is_not_cut_by_hold_rate(self):
        on = self.charge_into(90.0)
        old = P.with_cal("contact", first_strike_full=0)
        st = face_off(SWORD, CLANRATS, gap=14.0, b1=90.0)
        H = st.N // 2
        cut = None
        for _ in range(20):
            hp0 = float(st.u["hp_abs"][0, H])
            m0 = bool(st.u["m"][0, 0])
            battle.step(st, orders(st, **{str(H): (O.ATTACK, 0, True)}), old)
            if bool(st.u["m"][0, 0]) and not m0:
                cut = hp0 - float(st.u["hp_abs"][0, H])
                break
        assert on > 1.5 * cut


# --- R2, R3: hold_rate ---

class TestHoldRate:
    def rate_of(self, kind_0, params=P, reflect=False, charge=0.0):
        st = face_off(SPEAR, CLANRAT)
        H = st.N // 2
        st.u["order_kind"][0, 0] = kind_0
        st.u["order_kind"][0, H] = O.ATTACK
        st.u["order_target"][0, H] = 0
        if reflect:
            st.u["reflect"][0, 0] = True
        pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
        contact = pw["enemy"] & (pw["gap"] <= P.sim["contact"]["reach_m"] + 0.01)
        ch = torch.zeros_like(st.u["men"])
        ch[0, H] = charge
        rate, _, _, _ = melee.strikes(st.u, pw, contact, params, ch)
        return float(rate[0, 0, H])

    def test_a_move_order_in_contact_strikes_like_a_held_unit(self):
        attack, held, move = (self.rate_of(k) for k in (O.ATTACK, O.HOLD, O.MOVE))
        hr = P.sim["contact"]["hold_rate"]
        assert held == pytest.approx(hr * attack, rel=1e-4) and move == pytest.approx(held, rel=1e-4)
        off = P.with_cal("contact", hold_move=0)
        assert self.rate_of(O.MOVE, off) == pytest.approx(attack, rel=1e-4)

    def test_braced_reflection_is_not_cut_by_hold_rate(self):
        thr = P.battle["charge_reflect_min_charge_factor_threshold"]
        full = self.rate_of(O.ATTACK, reflect=True, charge=max(thr, 0.9))
        held = self.rate_of(O.HOLD, reflect=True, charge=max(thr, 0.9))
        assert held == pytest.approx(full, rel=1e-4)
        cut = self.rate_of(O.HOLD, P.with_cal("contact", hold_reflect_full=0), reflect=True, charge=max(thr, 0.9))
        assert cut == pytest.approx(P.sim["contact"]["hold_rate"] * full, rel=1e-4)


# --- C2 / R4: leaving melee, the chase and its 24 s window ---

class TestMeleeExit:
    def leave(self, chased, seconds):
        """Swordsmen fight clanrats 10 s, then withdraw west; the clanrats attack them (chased) or hold."""
        st = face_off(SWORD, CLANRATS)
        H = st.N // 2
        for _ in range(20):
            battle.step(st, orders(st, **{"0": (O.ATTACK, H, False), str(H): (O.ATTACK, 0, False)}), P)
        assert bool(st.u["m"][0, 0])
        hp0, hpH = float(st.u["hp_abs"][0, 0]), float(st.u["hp_abs"][0, H])
        out_at, struck = None, False
        for k in range(int(seconds / P.dt)):
            o = orders(st, **{"0": (O.WITHDRAW, (-300.0, 0.0), True)})
            if k > 0:                                                # the withdraw once, then no new orders
                o.kind[:] = O.KEEP
            if chased:
                o.kind[0, H], o.target[0, H], o.run[0, H] = O.ATTACK, 0, True
            hpH_before = float(st.u["hp_abs"][0, H])
            battle.step(st, o, P)
            if out_at is None and not bool(st.u["m"][0, 0]):
                out_at = (k + 1) * P.dt
            if float(st.u["hp_abs"][0, H]) < hpH_before - 1e-6:
                struck = struck or (k + 1) * P.dt
        return st, out_at, hp0 - float(st.u["hp_abs"][0, 0]), struck

    def test_an_unchased_unit_walks_out_within_5_s(self):
        st, out_at, lost, _ = self.leave(chased=False, seconds=10)
        assert out_at is not None and out_at <= 5.0 and lost < 60.0
        assert "pin_melee_s" not in P.sim["contact"]

    def test_a_chased_unit_stays_in_contact_until_the_window_then_fights(self):
        st, out_at, lost, struck = self.leave(chased=True, seconds=30)
        brk = P.battle["melee_breakoff_secs"]
        assert out_at is None or out_at >= brk                      # the chasers keep it in contact
        assert lost > 100.0                                          # struck from behind the whole time
        assert struck and struck >= brk - P.dt                       # it strikes back only after the window


# --- R6: the run-up only towards the enemy ---

def test_running_away_from_the_enemy_is_no_run_up():
    st = face_off(SWORD, CLANRATS, gap=40.0)
    H = st.N // 2
    for _ in range(12):                                              # run 6 s west, away from the clanrats
        battle.step(st, orders(st, **{"0": (O.MOVE, (-300.0, 0.0), True)}), P)
    assert float(st.u["runup"][0, 0]) == pytest.approx(0.0, abs=1e-6)
    st2 = face_off(SWORD, CLANRATS, gap=60.0)
    for _ in range(12):                                              # run towards them
        battle.step(st2, orders(st2, **{"0": (O.ATTACK, H, True)}), P)
    assert float(st2.u["runup"][0, 0]) > 10.0


# --- R7: the wounded pool ---

def test_the_wounded_pool_turns_melee_hp_into_whole_men_beyond_its_cap():
    st = face_off(SWORD, SLAVE)
    H = st.N // 2
    hm = float(st.u["hp_man"][0, H])
    # many wounded already (W = 30 men's worth of HP missing among the living): new blows kill whole men
    st.u["hp_abs"][0, H] = float(st.u["hp_abs"][0, H]) - 30 * hm * 0.6
    men0, hp0 = float(st.u["men"][0, H]), float(st.u["hp_abs"][0, H])
    battle.step(st, orders(st, **{"0": (O.ATTACK, H, False)}), P)
    lost_hp = hp0 - float(st.u["hp_abs"][0, H])
    lost_men = men0 - float(st.u["men"][0, H])
    assert lost_hp > 0 and lost_men == pytest.approx(lost_hp / hm, rel=0.02)
    # with the exponent rule (pool off) fewer men die for the same HP
    st2 = face_off(SWORD, SLAVE)
    st2.u["hp_abs"][0, H] = float(st2.u["hp_abs"][0, H]) - 30 * hm * 0.6
    battle.step(st2, orders(st2, **{"0": (O.ATTACK, H, False)}), P.with_cal("kills", wound_pool=0))
    assert men0 - float(st2.u["men"][0, H]) < lost_men


# --- C1: shattering by the database's thresholds ---

class TestShatter:
    def ctx(self, st):
        z = torch.zeros_like(st.u["men"], dtype=torch.bool)
        return {"aura": torch.zeros_like(st.u["men"]), "lord_dead_points": torch.zeros_like(st.u["men"]), "secure": z,
                "in_melee": z, "in_combat": z, "foe_lost": torch.zeros_like(st.u["men"]), "collapse": z,
                "routing_friends": torch.zeros_like(st.u["men"]), "routing_enemies": torch.zeros_like(st.u["men"]),
                "under_fire": z, "strong_enemy": z, "enemy_near": z | True}

    @pytest.mark.parametrize("routs, hp, shattered", [(1, 0.04, True), (1, 0.06, False), (2, 0.08, True),
                                                     (2, 0.12, False)])
    def test_a_routing_unit_shatters_below_the_database_health(self, routs, hp, shattered):
        st = face_off(SPEAR, SLAVE, gap=200.0)
        st.u["r"][0, 0] = True
        st.u["rout_count"][0, 0] = float(routs)
        st.u["rout_s"][0, 0] = 5.0
        st.u["hp"][0, 0] = hp
        st.u["morale"][0, 0] = -5.0
        morale.step(st.u, self.ctx(st), P, P.dt)
        assert bool(st.u["s"][0, 0]) == shattered

    def test_morale_at_the_floor_shatters_without_army_collapse_unless_unbreakable(self, monkeypatch):
        monkeypatch.setattr(morale, "target_points", lambda u, ctx, params, parts=None: torch.full_like(u["morale"], -90.0))
        for unbreakable, routing in ((False, True), (False, False), (True, False)):
            st = face_off(SPEAR, SLAVE, gap=200.0)
            st.u["morale"][0, 0] = -49.0
            st.u["r"][0, 0] = routing
            st.u["rout_count"][0, 0] = float(routing)
            st.u["hp"][0, 0] = 0.8
            st.u["unbreakable"][0, 0] = unbreakable
            morale.step(st.u, self.ctx(st), P, P.dt)
            assert bool(st.u["s"][0, 0]) == (not unbreakable), (unbreakable, routing)


# --- C5: the rally ---

class TestRally:
    def rally_after(self, enemy_x, enemy_routing, seconds=40):
        # (a second unit of ours far off keeps the army from collapsing)
        st = scenario.build([army([(SPEAR, 0, 0, 90), (SPEAR, -600, 0, 90)],
                                  [(SLAVE, enemy_x, 0, 270), (SLAVE, 700, 400, 270)])], P)   # (the battle goes on)
        H = st.N // 2
        st.u["r"][0, 0] = True
        st.u["rout_count"][0, 0] = 1.0
        st.u["rout_s"][0, 0] = 30.0
        st.u["morale"][0, 0] = 20.0
        if enemy_routing:
            st.u["r"][0, H] = True
        for k in range(int(seconds / P.dt)):
            st.u["x"][0, 0], st.u["z"][0, 0] = 0.0, 0.0                 # kept in place
            st.u["x"][0, H], st.u["z"][0, H] = float(enemy_x), 0.0
            st.u["morale"][0, H] = -5.0 if enemy_routing else st.u["morale"][0, H]
            battle.step(st, replay.hold(st), P)
            if not bool(st.u["r"][0, 0]):
                return (k + 1) * P.dt
        return None

    def test_a_routing_enemy_near_blocks_the_rally(self):
        assert self.rally_after(80.0, enemy_routing=True) is None
        assert self.rally_after(120.0, enemy_routing=True) is not None

    def test_the_rally_comes_after_the_wait(self):
        t = self.rally_after(300.0, enemy_routing=False)
        assert t is not None and t >= P.sim["morale"]["rally_wait_s"] - P.dt - 1e-6


# --- C3: no charge into a target already fighting our units ---

def test_no_sprint_and_no_charge_into_a_target_already_in_melee_with_ours():
    def approach(busy):
        rows = [(SWORD, -40, 0, 90)] + ([(SPEAR, 30, 0, 270)] if busy else [])
        st = scenario.build([army(rows, [(CLANRAT, 0, 0, 270)])], P)
        H = st.N // 2
        front, depth = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
        if busy:   # our spearmen already in contact with the clanrats' rear (the swordsmen come from the front)
            st.u["x"][0, 1] = float(st.u["x"][0, H]) + (float(depth[0, 1]) + float(depth[0, H])) / 2 + TOUCH - 0.5
        speeds, charged = [], False
        for _ in range(40):
            o = orders(st, **{"0": (O.ATTACK, H, True)})
            if busy:
                o.kind[0, 1], o.target[0, 1] = O.ATTACK, H
            battle.step(st, o, P)
            speeds.append(float(torch.hypot(st.u["vx"][0, 0], st.u["vz"][0, 0])))
            charged = charged or float(st.u["charge"][0, 0]) > 0.5
            if bool(st.u["m"][0, 0]):
                break
        return max(speeds), charged
    free_v, free_c = approach(False)
    busy_v, busy_c = approach(True)
    run = P.units[SWORD]["speed"]["run"]
    assert free_v > busy_v + 0.3 and free_c and not busy_c and busy_v <= run + 0.05


# --- C4: the flank and rear sectors ---

def test_the_sectors_are_45_and_135_degrees():
    sector = geometry.sector(torch.tensor([40.0, 50.0, 130.0, 140.0]) * geometry.DEG, P.sim["contact"]["front_deg"],
                             P.sim["contact"]["rear_deg"])
    assert sector.tolist() == [0, 1, 1, 2]
