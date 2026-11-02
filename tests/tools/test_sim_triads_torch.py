"""The Eshin Triads in the simulator (02.11.2026; build/diag_roster/wh2_dlc14_skv_inf_eshin_triads_0/mechanic.md): the numbers of the
older rules on their passport - the reflection on a charger, the charge defence, the thin formation's brace, hidden and braced - and
the probe twin's way to a hidden target. Bombs and blade: test_sim_runners_torch.py runs on both keys."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn import charge_probe as cp  # noqa: E402
from tools.nn.sim import abilities, battle, geometry, impact, melee, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
T3, DR = "wh2_dlc14_skv_inf_eshin_triads_0", "wh2_main_skv_inf_death_runners_0"
KN, SW = "wh_main_emp_cav_empire_knights", "wh_main_emp_inf_swordsmen"
BOMBS = "wh2_main_unit_abilities_concealment_bombs"
T = torch.tensor


def army(foe, x=30.0, key=T3):
    return {"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": False, "units": [{"key": foe, "x": x, "z": 0, "b": 270}]},
        2: {"faction": "wh2_main_skv_skaven", "ai": False, "units": [{"key": key, "x": 0, "z": 0, "b": 90}]}}}


def blows(charge, moving=False, armour=None, key=T3, foe=KN):
    """The Triads (slot H) on the knights in contact, the knights' charge factor `charge`: (the rate, the HP a hit of the man's pool)."""
    st = scenario.build([army(foe, key=key)], P)
    H = st.N // 2
    i, j = H, 0
    u = st.u
    front, depth = geometry.dims(u, P.sim["formation"]["spacing_m"])
    u["x"][0, j] = (float(depth[0, i]) + float(depth[0, j])) / 2
    if moving:
        u["vx"][0, i] = 1.0
    if armour is not None:
        u["armour"][0, j] = armour
    pw = geometry.pairwise(u, P.sim["formation"]["spacing_m"])
    ch = torch.zeros_like(u["men"])
    ch[0, j] = charge
    out = melee.strikes(u, pw, pw["enemy"], P, ch)
    return float(out[0][0, i, j]), float(out[1][0, i, j])


def rule(mult, armour):
    """The mean HP of a hit by the database's numbers: the bonus inside the sum, then x mult, armour U(0.5, 1) on the base part, the
    knights' resistance 0: base 8 + 15 x 8/30, ap 22 + 15 x 22/30."""
    m = P.units[T3]["melee"]
    sh = m["damage"] / (m["damage"] + m["ap_damage"])
    base, ap = (m["damage"] + m["bonus_v_large"] * sh) * mult, (m["ap_damage"] + m["bonus_v_large"] * (1 - sh)) * mult
    a = armour / 100
    return ap + base * max(0.0, 1 - a), ap + base * max(0.0, 1 - a / 2)


def test_the_range_of_a_hit_is_the_mechanics_table():
    # mechanic.md 2.1: x1 33.0-38.4, x2 66.0-76.8; with the blade (armour 55) x1 38.4-41.7, x2 76.8-83.4
    arm = float(P.units[KN]["armour"])
    assert arm == 110
    assert rule(1, arm) == pytest.approx((33.0, 38.4), abs=0.1) and rule(2, arm) == pytest.approx((66.0, 76.8), abs=0.1)
    cut = 55.0
    assert rule(1, cut) == pytest.approx((38.4, 41.7), abs=0.15) and rule(2, cut) == pytest.approx((76.8, 83.4), abs=0.2)


def test_the_reflection_doubles_a_blow_at_07_of_the_charge_and_not_below():
    # (per_hit: the mean of the range above, then a man's overkill: the same call the simulator makes, bonus inside the sum, x2 on it)
    m, k = P.units[T3]["melee"], P.units[KN]
    sh = m["damage"] / (m["damage"] + m["ap_damage"])
    base, ap = m["damage"] + m["bonus_v_large"] * sh, m["ap_damage"] + m["bonus_v_large"] * (1 - sh)

    def want(x):
        return float(melee.per_hit(T(x * base), T(x * ap), T(float(k["armour"])), T(float(k["hp_per_man"]))))
    _, h1 = blows(0.0)
    _, h2 = blows(0.7)
    _, h3 = blows(0.69)
    assert h1 == pytest.approx(h3, rel=1e-5) and h1 == pytest.approx(want(1.0), rel=1e-4)
    assert h2 == pytest.approx(want(2.0), rel=1e-4)


def test_moving_it_does_not_reflect():
    _, h = blows(1.0, moving=True)
    _, h0 = blows(0.0)
    assert h == pytest.approx(h0, rel=1e-5)


def test_the_runners_do_not_reflect():
    _, a = blows(1.0, key=DR)
    _, b = blows(0.0, key=DR)
    assert a == pytest.approx(b, rel=1e-5)


def test_the_flags_of_the_rules():
    st = scenario.build([army(KN)], P)
    H = st.N // 2
    assert bool(st.u["reflect"][0, H]) and bool(st.u["cdl"][0, H]) and bool(st.u["stalk"][0, H])
    assert bool(st.u["hid"][0, H])                # hidden from the start
    assert not bool(st.u["unsp"][0, H])           # the bombs not pressed


def test_the_thin_formation_braces_and_the_knights_knock_it_down():
    # mechanic.md 2.3: B 4.57 at 4 ranks, m x B 411 (under the 600 a small body cannot move, a number the simulator does not read),
    # the speed jump from a knight 1.3 x 1000 / (411 + 1000) x 10.8 = 9.95 over the knock-down threshold 7, under the flight 10.5
    st = scenario.build([army(KN)], P)
    H = st.N // 2
    n = torch.zeros_like(st.u["men"])
    b = impact.bracing(st.u, P, n, n)
    assert float(b[0, H]) == pytest.approx(4.57, abs=0.02)
    mb = float(b[0, H]) * P.units[T3]["mass"]
    assert mb == pytest.approx(411, abs=2) and mb < 600
    dv = 1.3 * P.units[KN]["mass"] / (mb + P.units[KN]["mass"]) * 10.8
    assert 7 < dv < 10.5 and dv == pytest.approx(9.95, abs=0.03)


def test_hidden_it_is_braced_as_well():
    st = scenario.build([army(KN, x=300.0)], P)
    H = st.N // 2
    assert bool(st.u["hid"][0, H])
    n = torch.zeros_like(st.u["men"])
    assert float(impact.bracing(st.u, P, n, n)[0, H]) > 4.0


def fake_lane(kind="knights", gap=150.0, kept=False, foe=KN, casts=None, width=30.0):
    """A recorded lane's skeleton for the twin: only what sim_lanes reads, the target at z = 0 facing +z, the attacker gap_m off."""
    a_key = foe
    a_men = (cp.passport(foe) or P.units[foe])["men"]
    spec = dict(name="L1", kind=kind, attacker="a", target="t", a_key=a_key, t_key=T3, mode="attack_run", target_mode="hold",
                answer=False, a_width=width, t_width=width, a_depth=cp.depth(a_key, a_men, width),
                t_depth=cp.depth(T3, 72, width), x=0.0, z=0.0, move_beyond_m=5.0, a_on_sight=True, a_try_s=3, fight_s=20, max_s=60,
                stealth_rows=True, gap_m=gap)
    if casts:
        spec["t_casts"] = casts
    d = gap + (spec["a_depth"] + spec["t_depth"]) / 2
    s0 = (0.0, {"x": 0.0, "z": d - spec["t_depth"] / 2, "b": 180, "men": a_men}, {"x": 0.0, "z": -spec["t_depth"] / 2, "b": 0, "men": 72},
          None)
    return {"run": "r", "plan": "triads", "spec": spec, "samples": [s0], "contacts": {}, "abilities": [],
            "phases": [{"phase": "a_try_end", "kept": kept, "t": 3000}] + ([] if kept else [{"phase": "a_move", "t": 3000}]),
            "end": {"t": 50000}}


def twin(lane):
    return cp.sim_lanes([lane], params=P, copies=1, jitter_m=0.0)[0]


def test_the_twin_runs_at_a_hidden_target_and_attacks_when_it_sees_it():
    out = twin(fake_lane())
    m = cp.triads_measure(out)
    assert m["unseen_at_start"]
    assert m["seen_gap"] == pytest.approx(64.0, abs=8.0)          # a knight's step at a run is 5 m a tick of the samples
    assert m.get("contact_s") is not None                        # the order came, the knights arrive
    assert any(p["phase"] == "a_seen" for p in out["phases"])


def test_the_twins_target_presses_its_bombs_and_is_seen_at_20_m():
    out = twin(fake_lane("bombs", casts=[{"key": BOMBS, "at_s": 0}]))
    m = cp.triads_measure(out)
    assert m["seen_gap"] == pytest.approx(20.0, abs=8.0)
    assert m.get("contact_s") is not None


def test_a_battle_without_triads_has_no_triads_rule_on():
    st = scenario.build([{"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": False, "units": [{"key": SW, "x": 0, "z": -20, "b": 0}]},
        2: {"faction": "wh2_main_skv_skaven", "ai": False, "units": [{"key": "wh2_main_skv_inf_clanrats_1", "x": 0, "z": 20, "b": 180}]}}}], P)
    assert BOMBS not in abilities.slot_keys(SW, P.units, P.abilities)
    battle.step(st, O.hold(st.B, st.N, st.device), P, P.dt)
    assert not bool(st.u["hid"].any()) and not bool(st.u["unsp"].any())
