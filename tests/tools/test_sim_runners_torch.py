"""The Death Runners' rules in the simulator (02.11.2026; build/diag_roster/wh2_main_skv_inf_death_runners_0/mechanic.md; config/nn/
sim.json abilities.attributes, abilities.uses_limit, stealth.unspottable_m): the attributes of an active ability on its owner, the
limited uses, the spotting distance of an unspottable unit, the armour stat of a contact effect."""
import copy

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import abilities, battle, effects, melee, scenario, stealth  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
DR, SWORD, GS = "wh2_main_skv_inf_death_runners_0", "wh_main_emp_inf_swordsmen", "wh_main_emp_inf_greatswords"
OWNERS = (DR, "wh2_dlc14_skv_inf_eshin_triads_0")          # the same rules on the Eshin Triads: no case of a unit in the code
BOMBS, BLADE = "wh2_main_unit_abilities_concealment_bombs", "wh2_main_unit_contact_weeping_blade"


@pytest.fixture(params=OWNERS, autouse=True)
def owner(request):
    """Every rule here on both owners of the bombs and the blade."""
    global DR
    keep, DR = DR, request.param
    yield DR
    DR = keep


def army(foes, runners, foe=SWORD):
    """side 1: Empire foot at (x, z) facing +z; side 2: Death Runners at (x, z) facing -z."""
    return {"attacker": 2, "sides": {
        1: {"faction": "wh_main_emp_empire", "ai": False, "units": [{"key": foe, "x": x, "z": z, "b": 0} for x, z in foes]},
        2: {"faction": "wh2_main_skv_skaven", "ai": False, "units": [{"key": DR, "x": x, "z": z, "b": 180} for x, z in runners]}}}


def build(*a, params=P, **k):
    return scenario.build([army(*a, **k)], params)


def slot():
    return abilities.slot_keys(DR, P.units, P.abilities).index(BOMBS)


def orders(st, press=(), kind=None):
    o = O.hold(st.B, st.N, st.device)
    o.kind.fill_(O.KEEP)
    for i in press:
        o.ability[0, i] = slot()
    for i, (k, tg, x, z, run) in (kind or {}).items():
        o.kind[0, i], o.target[0, i], o.x[0, i], o.z[0, i], o.run[0, i] = k, tg, x, z, run
    return o


def step(st, o=None, params=P):
    battle.step(st, o if o is not None else orders(st), params, params.dt)


def steps(st, n, params=P):
    for _ in range(n):
        step(st, params=params)


# ------------------------------------------------------------------ uses and attributes of an active ability

def test_the_press_fires_73_s_then_90_s_and_counts_a_use():
    st = build([(0, -300)], [(0, 300)])
    H = st.N // 2
    k = slot()
    assert int(st.u[f"ab{k}_n"][0, H]) == 3
    step(st, orders(st, [H]))
    assert float(st.u[f"ab{k}_on"][0, H]) == pytest.approx(73, abs=0.2)
    assert float(st.u[f"ab{k}_cd"][0, H]) == pytest.approx(163, abs=0.2)
    assert int(st.u[f"ab{k}_n"][0, H]) == 2
    steps(st, int(73.5 / P.dt))
    assert float(st.u[f"ab{k}_on"][0, H]) == 0
    assert 89 < float(st.u[f"ab{k}_cd"][0, H]) <= 90


def test_after_the_third_use_no_recharge_and_no_fourth():
    st = build([(0, -300)], [(0, 300)])
    H, k = st.N // 2, slot()
    st.u[f"ab{k}_n"][0, H] = 1                             # the last use
    step(st, orders(st, [H]))
    assert int(st.u[f"ab{k}_n"][0, H]) == 0 and float(st.u[f"ab{k}_on"][0, H]) > 72
    steps(st, int(170 / P.dt))
    assert float(st.u[f"ab{k}_on"][0, H]) == 0
    assert float(st.u[f"ab{k}_cd"][0, H]) == pytest.approx(90, abs=1e-3)       # frozen: never ready
    step(st, orders(st, [H]))
    assert float(st.u[f"ab{k}_on"][0, H]) == 0 and int(st.u[f"ab{k}_n"][0, H]) == 0


def test_uses_limit_off_all_unlimited():
    p = P.with_cal("abilities", uses_limit={"on": 0})
    st = scenario.build([army([(0, -300)], [(0, 300)])], p)
    assert int(st.u[f"ab{slot()}_n"][0, st.N // 2]) == -1


def test_other_abilities_have_no_limit():
    table = abilities.uses_table(P)
    names = abilities.keys(P)
    assert table[-1] == -1 and table[names.index(BOMBS)] == 3
    assert all(v == -1 for n, v in zip(names, table) if n != BOMBS)


def test_the_flag_lies_only_while_active_and_is_restored():
    st = build([(0, -300)], [(0, 300)])
    H = st.N // 2
    assert not bool(st.u["unsp"][0, H])
    step(st, orders(st, [H]))
    assert not bool(st.u["unsp"][0, H])                     # restored at the step's end (a rule of the step)
    assert abilities.ability_rules(P) == ("stalk", "unspottable")
    # within the step: the reach is 20 m
    seen = {}
    orig = stealth.update

    def spy(u, pw, params, dt, fired, engaged, attack):
        seen["unsp"] = bool(u["unsp"][0, H])
        return orig(u, pw, params, dt, fired, engaged, attack)
    stealth.update = spy
    try:
        step(st)
    finally:
        stealth.update = orig
    assert seen["unsp"]


def test_attributes_off_no_flag():
    p = P.with_cal("abilities", attributes={"on": 0})
    assert abilities.ability_rules(p) == ()


# ------------------------------------------------------------------ seeing

def walk_in(press, start=120.0):
    """The Runners walk on held swordsmen from start m (the fronts' gap about so); the gap (edge to edge) at the first step they are seen."""
    st = build([(0, -20)], [(0, start)])
    H = st.N // 2
    assert bool(st.u["hid"][0, H])
    first = True
    while True:
        o = orders(st, [H] if press and first else (), {H: (O.MOVE, -1, 0.0, -300.0, False)})
        first = False
        step(st, o)
        if not bool(st.u["hid"][0, H]) or float(st.t[0]) > 200:
            break
    return float(st.u["z"][0, H]) - float(st.u["z"][0, 0]), bool(st.u["hid"][0, H])


def test_seen_at_64_m_without_the_bombs_at_20_m_with():
    plain, _ = walk_in(False)
    # (a walk of 1.35 m/s, Strength in Numbers: start near enough that the 73 s do not run out on the way)
    bombs, _ = walk_in(True, start=70.0)
    # centres: edge to edge + half the two depths (about 8 m together)
    assert 70 < plain < 74
    assert plain - bombs == pytest.approx(44, abs=1.0)


def test_when_the_bombs_end_the_old_distance_is_back():
    # from 140 m between the centres the walk does not arrive in 73 s: at their end it is within 64 m and seen at once
    z, _ = walk_in(True, start=120.0)
    assert 35 < z < 45


def test_the_press_does_not_reveal():
    st = build([(0, -300)], [(0, 300)])
    H = st.N // 2
    assert bool(st.u["hid"][0, H])
    for i in range(30):
        step(st, orders(st, [H] if i == 0 else ()))
        assert bool(st.u["hid"][0, H])


def test_melee_reveals_even_under_the_bombs_and_the_timer_runs():
    st = build([(0, -5)], [(0, 5)])
    H, k = st.N // 2, slot()
    step(st, orders(st, [H]))
    on0 = float(st.u[f"ab{k}_on"][0, H])
    for _ in range(100):
        step(st, orders(st, kind={H: (O.ATTACK, 0, 0.0, 0.0, True)}))
        if bool(st.u["m"][0, H]):
            break
    assert bool(st.u["m"][0, H])
    step(st)
    assert not bool(st.u["hid"][0, H])
    assert float(st.u[f"ab{k}_on"][0, H]) < on0 - 0.5


def test_unspottable_off_seen_as_before():
    p = P.with_cal("stealth", unspottable_m=0.0)
    st = scenario.build([army([(0, -20)], [(0, 40)])], p)      # the gap about 52 m: within 64 m
    H = st.N // 2
    step(st, orders(st, [H]), params=p)
    step(st, params=p)
    assert not bool(st.u["hid"][0, H])


# ------------------------------------------------------------------ the blade

def blade_j():
    return next(j for j, _, k in effects.contact_keys(P) if k == BLADE)


def test_five_contact_effects_fit_and_the_table_has_the_armour_column():
    assert len(effects.contact_keys(P)) == 5 and effects.CONTACT_SLOTS >= 5
    row = effects.row(P, BLADE)
    assert row[effects.COL["self_armour"]] == 0.5 and row[effects.COL["duration_s"]] == 35.0
    assert effects.laid_by(P, DR)[0] == effects.index(P)[BLADE]


@pytest.mark.parametrize("base,cut", [(95.0, 48.0), (100.0, 50.0), (25.0, 13.0), (30.0, 15.0), (20.0, 10.0)])
def test_the_armour_halves_rounded_half_up_while_the_blade_runs(base, cut):
    st = build([(0, -300)], [(0, 300)])
    u = st.u
    u["armour"][0, 0] = base
    u[f"cfx{blade_j()}_s"][0, 0] = 5.0
    seen = {}
    orig = melee.per_hit

    def spy(b, a, armour, *x, **y):
        if "armour" not in seen and getattr(armour, "ndim", 0) == 3:
            seen["armour"] = float(armour[0, 0, 0])
        return orig(b, a, armour, *x, **y)
    keep = stealth.update

    def at_end(uu, *a, **k):
        seen["armour_field"] = float(uu["armour"][0, 0])
        return keep(uu, *a, **k)
    stealth.update = at_end
    try:
        step(st)
    finally:
        stealth.update = keep
    # fatigue's armour multiplier x1 fresh: the step's armour is the cut one; restored at the step's end
    assert seen["armour_field"] == cut
    assert float(u["armour"][0, 0]) == base
    # no blade: the same number to the bit
    u[f"cfx{blade_j()}_s"][0, 0] = 0.0
    stealth.update = at_end
    try:
        step(st)
    finally:
        stealth.update = keep
    assert seen["armour_field"] == base


def fight(s, foe=SWORD, params=P):
    st = scenario.build([army([(0, 0)], [(0, 3)], foe=foe)], params)
    u, H = st.u, st.N // 2
    for _ in range(int(s / params.dt)):
        step(st, orders(st, kind={H: (O.ATTACK, 0, 0.0, 0.0, False)}), params=params)
    return st


def test_a_blow_lays_it_for_35_s_refreshed():
    st = fight(6.0)
    j, H = blade_j(), st.N // 2
    assert float(st.u[f"cfx{j}_s"][0, 0]) > 30
    assert float(st.u[f"cfx{j}_s"][0, H]) == 0                # the swordsmen's blows lay none
    # its bit in fx_on (the network's input, the bridge's card)
    i = effects.index(P)[BLADE]
    assert int((st.u["fx_on"][0, 0] >> i) & 1) == 1


def test_the_blade_cuts_the_armour_of_the_greatswords_and_more_health_goes():
    on = fight(25.0, GS)
    off_p = copy.deepcopy(P)
    off_p.sim.setdefault("effects", {})["off"] = list((off_p.sim["effects"].get("off") or ())) + [BLADE]
    off = fight(25.0, GS, params=off_p)
    assert float(on.u["hp_abs"][0, 0]) < float(off.u["hp_abs"][0, 0])


def test_the_timer_ends_after_35_s():
    st = build([(0, -300)], [(0, 300)])
    j = blade_j()
    st.u[f"cfx{j}_s"][0, 0] = 35.0
    steps(st, int(34.5 / P.dt))
    assert float(st.u[f"cfx{j}_s"][0, 0]) > 0
    steps(st, int(1.0 / P.dt))
    assert float(st.u[f"cfx{j}_s"][0, 0]) == 0


# ------------------------------------------------------------------ old armies

def test_a_battle_without_runners_is_the_same_with_the_rules_off():
    def run(params):
        arm = army([(0, -20), (40, -20)], [(0, 20), (40, 20)], foe=SWORD)
        arm["sides"][2]["units"] = [{"key": "wh2_main_skv_inf_clanrats_1", "x": x, "z": 20, "b": 180} for x in (0, 40)]
        st = scenario.build([arm], params)
        H = st.N // 2
        for _ in range(300):
            step(st, orders(st, kind={H: (O.ATTACK, 0, 0.0, 0.0, True)}), params=params)
        return st
    on = run(P)
    off_p = P.with_cal("abilities", attributes={"on": 0}, uses_limit={"on": 0})
    off_p = off_p.with_cal("stealth", unspottable_m=0.0)
    off = run(off_p)
    for k in ("x", "z", "hp_abs", "morale", "men", "fatigue"):
        assert torch.equal(on.u[k], off.u[k]), k


# ------------------------------------------------------------------ the bridge: the same timers (CLAUDE.md rule 13)

def test_the_companions_timers_are_the_simulators():
    from tools.nn.companion import exchange
    st = build([(0, -300)], [(0, 300)])
    H, k = st.N // 2, slot()
    used, counted = None, 0

    def compare(t_ms):
        row = {"n": "enemy_dr", "key": DR, "side": 2, "x": 0, "z": 300,
               "fx": [BOMBS] if float(st.u[f"ab{k}_on"][0, H]) > 0 else []}
        doc = {"t": t_ms, "units": [row], "abilities_used": {"enemy_dr": {BOMBS: used}} if used is not None else {},
               "abilities_count": {"enemy_dr": {BOMBS: counted}} if counted else {}}
        s = exchange.ability_timers(doc, ["enemy_dr"], [[BOMBS]], own=2)
        return float(s["ab0_on"][0, 0]), float(s["ab0_cd"][0, 0]), float(st.u[f"ab{k}_on"][0, H]), float(st.u[f"ab{k}_cd"][0, H])
    t = 0.0
    for press_at in (1.0, 170.0, 340.0):
        while t < press_at:
            step(st)
            t += P.dt
        step(st, orders(st, [H]))
        used, counted = t * 1000.0, counted + 1
        t += P.dt
        for dt_after in (5.0, 80.0, 165.0):
            while t < press_at + dt_after:
                step(st)
                t += P.dt
            a_on, a_cd, s_on, s_cd = compare(t * 1000.0)
            assert a_on == pytest.approx(s_on, abs=0.6) and a_cd == pytest.approx(s_cd, abs=0.6), (press_at, dt_after)


def test_the_games_ai_does_not_press_the_bombs():
    # (config/nn/sim.json abilities.bombs_trigger_why: 1 battle, 3 units; the default rule 'enemy within 60 m' would)
    arm = army([(0, -20)], [(0, 30)])
    arm["sides"][2]["ai"] = True
    st = scenario.build([arm], P)
    H, k = st.N // 2, slot()
    steps(st, 30)
    assert float(st.u[f"ab{k}_on"][0, H]) == 0 and int(st.u[f"ab{k}_n"][0, H]) == 3
