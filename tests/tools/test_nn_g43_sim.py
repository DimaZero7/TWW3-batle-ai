"""Line g43 (25.10.2026): the new units' mechanics in the simulator by the game's rules (docs/ru/training/simulator.md
"Скрытность", "Пуля сквозь нескольких", "Ловчая сеть", "Ломка щитов", "Пороховая выучка"): hiding and spotting
(tools/nn/sim/stealth.py), a projectile through several men (missile.py men_struck / penetration and the fighters lab),
the Snare Net on enemies, the shield breaker, Blackpowder Discipline and the reload skill (effects.py). Torch: the
training container."""
import copy

import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, effects, missile, scenario, stealth  # noqa: E402
from tools.nn.sim import geometry  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import Params, load  # noqa: E402

P = load()
GR, SLINGS = "wh2_main_skv_inf_gutter_runners_0", "wh2_main_skv_inf_gutter_runner_slingers_0"
JEZ, LR, IRON = ("wh2_dlc12_skv_inf_warplock_jezzails_0", "wh3_dlc25_emp_inf_hochland_long_rifles",
                 "wh3_dlc25_emp_inf_nuln_ironsides")
SPEAR, SHIELD_SPEAR, SWORD = "wh_main_emp_inf_spearmen_0", "wh_main_emp_inf_spearmen_1", "wh_main_emp_inf_swordsmen"
HANDGUN, CLANRAT, RATS = "wh_main_emp_inf_handgunners", "wh2_main_skv_inf_clanrats_1", "wh2_dlc16_skv_mon_wolf_rats_0"
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"


def duel(own, enemy, gap, own_b=90.0, enemy_b=270.0, own_side=SKV, extra=()):
    """One battle: side 1 (`own_side`) has unit `own` at x 0, side 2 (the other faction) `enemy` at x = gap (centre
    to centre), facing each other; extra: more (side, key, x, z) units."""
    other = EMP if own_side == SKV else SKV
    s1 = [{"key": own, "x": 0.0, "z": 0.0, "b": own_b}]
    s2 = [{"key": enemy, "x": float(gap), "z": 0.0, "b": enemy_b}]
    for side, key, x, z in extra:
        (s1 if side == 1 else s2).append({"key": key, "x": float(x), "z": float(z), "b": own_b if side == 1 else enemy_b})
    return {"attacker": 1, "sides": {1: {"faction": own_side, "units": s1, "ai": False},
                                     2: {"faction": other, "units": s2, "ai": False}}}


def hold(st):
    return O.hold(st.B, st.N, st.device)


def run(st, steps, orders=None, params=P):
    for _ in range(steps):
        battle.step(st, orders(st) if orders else hold(st), params)
    return st


def gap_of(st, i=0, j=None):
    pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"], rect=bool(P.sim["contact"].get("rect_gap")))
    j = st.N // 2 if j is None else j
    return float(pw["gap"][0, i, j])


def with_sim(**sections):
    sim = copy.deepcopy(P.sim)
    for k, v in sections.items():
        sim[k] = v if not isinstance(v, dict) or k not in sim else {**sim[k], **v}
    return Params(P.units, P.rules, sim, P.abilities, P.effects, P.spells)


# --- hiding and spotting ---------------------------------------------------------------------------------------------

def test_a_stalker_starts_hidden_and_an_enemy_out_of_reach_does_not_see_it():
    st = scenario.build([duel(GR, SPEAR, 300)], P)
    gr, foe = 0, st.N // 2
    assert st.u["hid"][0, gr] and not st.u["vis"][0, gr] and st.u["vis"][0, foe] and not st.u["hid"][0, foe]
    run(st, 10)
    assert st.u["hid"][0, gr] and not st.u["vis"][0, gr]
    # off: nobody hides
    off = with_sim(stealth={"on": 0})
    st2 = scenario.build([duel(GR, SPEAR, 300)], off)
    run(st2, 2, params=off)
    assert bool(st2.u["vis"][0, gr]) and not bool(st2.u["hid"][0, gr])


def test_an_enemy_sees_it_within_its_spotting_x_the_hiding_scalar():
    # the Gutter Runners' 0.8 x the spearmen's 80 m = 64 m, edge to edge (the fresh bonus over at the start)
    for gap_want, seen in ((70.0, False), (58.0, True)):
        st = scenario.build([duel(GR, SPEAR, 100)], P)
        g = gap_of(st)
        st = scenario.build([duel(GR, SPEAR, 100 - (g - gap_want))], P)
        assert gap_of(st) == pytest.approx(gap_want, abs=0.5)
        run(st, 1)
        assert bool(st.u["vis"][0, 0]) == seen, gap_want
    # the Jezzails see within 160 x 0.8 = 128 m
    for gap_want, seen in ((134.0, False), (122.0, True)):
        st = scenario.build([duel(GR, JEZ, 200)], P)
        g = gap_of(st)
        st = scenario.build([duel(GR, JEZ, 200 - (g - gap_want))], P)
        assert float(st.u["spot_m"][0, st.N // 2]) == 160.0
        run(st, 1)
        assert bool(st.u["vis"][0, 0]) == seen, gap_want


def test_firing_reveals_running_does_not_and_it_hides_again_after_5_s():
    st = scenario.build([duel(GR, SPEAR, 150)], P)
    # running across, far from the enemy: still hidden
    def run_north(s):
        o = hold(s)
        o.kind[:, 0], o.x[:, 0], o.z[:, 0], o.run[:, 0] = O.MOVE, 0.0, 300.0, True
        return o
    run(st, 6, run_north)
    assert bool(st.u["hid"][0, 0]) and float(st.u["z"][0, 0]) > 10
    # in range of its stars (70 m) a holding unit shoots at will: firing reveals it
    st = scenario.build([duel(GR, SPEAR, 90)], P)
    seen = []
    for _ in range(30):
        battle.step(st, hold(st), P)
        seen.append(bool(st.u["vis"][0, 0]))
    assert any(seen)
    # revealed, then 5 s with nothing revealing it: hidden again (the hide clock)
    st = scenario.build([duel(GR, SPEAR, 400)], P)
    st.u["hid"][0, 0] = False
    st.u["hid_s"][0, 0] = 0.0
    run(st, 9)
    assert not bool(st.u["hid"][0, 0])                       # 4.5 s
    run(st, 1)
    assert bool(st.u["hid"][0, 0])                           # 5 s


def test_the_fresh_bonus_widens_the_reach_while_the_unit_is_not_yet_hidden():
    # stealth.fresh_before (the program, build/diag_g49/terrain/stealth.md 1): +20 m while its hide clock runs, not
    # after it hid. 72 m: beyond 64, within (80 + 20) x 0.8 = 80
    st = scenario.build([duel(GR, SPEAR, 100)], P)
    g = gap_of(st)
    st = scenario.build([duel(GR, SPEAR, 100 - (g - 72.0))], P)
    st.u["hid_t"][0, 0] = 0.0                                      # it has just hidden: no bonus now
    run(st, 1)
    assert bool(st.u["hid"][0, 0]) and not bool(st.u["vis"][0, 0])
    st = scenario.build([duel(GR, SPEAR, 100 - (g - 72.0))], P)
    st.u["hid"][0, 0], st.u["hid_s"][0, 0], st.u["hid_r"][0, 0] = False, 1.0, 0.0     # its clock running
    run(st, 1)
    assert not bool(st.u["hid"][0, 0]) and float(st.u["hid_s"][0, 0]) == 0.0      # seen: the clock starts over
    old = with_sim(stealth={"fresh_before": 0})
    st = scenario.build([duel(GR, SPEAR, 100 - (g - 72.0))], old)
    st.u["hid_t"][0, 0] = 0.0
    run(st, 1, params=old)
    assert bool(st.u["vis"][0, 0])                                 # the old reading: seen just after it hid


def test_revealed_for_less_than_1_5_s_it_hides_again_at_once():
    st = scenario.build([duel(GR, SPEAR, 400)], P)
    st.u["hid"][0, 0], st.u["hid_s"][0, 0], st.u["hid_r"][0, 0] = False, 0.0, 1.0      # revealed 0.5 s ago
    run(st, 1)
    assert bool(st.u["hid"][0, 0])
    st = scenario.build([duel(GR, SPEAR, 400)], P)
    st.u["hid"][0, 0], st.u["hid_s"][0, 0], st.u["hid_r"][0, 0] = False, 0.0, 0.0      # revealed long ago
    run(st, 1)
    assert not bool(st.u["hid"][0, 0])


def test_a_routing_enemy_sees_nobody():
    st = scenario.build([duel(GR, SPEAR, 100)], P)
    g = gap_of(st)
    st = scenario.build([duel(GR, SPEAR, 100 - (g - 50.0))], P)    # 50 m: within 64
    st.u["r"][0, st.N // 2] = True
    stealth.update(st.u, geometry.pairwise(st.u, P.sim["formation"]["spacing_m"],
                                           rect=bool(P.sim["contact"].get("rect_gap"))), P, 0.5,
                   torch.zeros_like(st.u["hid"]), torch.zeros_like(st.u["hid"]), torch.zeros_like(st.u["hid"]))
    assert bool(st.u["hid"][0, 0])
    st.u["r"][0, st.N // 2] = False
    stealth.update(st.u, geometry.pairwise(st.u, P.sim["formation"]["spacing_m"],
                                           rect=bool(P.sim["contact"].get("rect_gap"))), P, 0.5,
                   torch.zeros_like(st.u["hid"]), torch.zeros_like(st.u["hid"]), torch.zeros_like(st.u["hid"]))
    assert not bool(st.u["hid"][0, 0])


def test_a_hidden_enemy_is_not_shot_not_attacked_and_not_counted_by_morale():
    st = scenario.build([duel(HANDGUN, GR, 100, own_side=EMP)], P)
    gr = st.N // 2
    assert bool(st.u["hid"][0, gr])
    kill_hp = float(st.u["hp_abs"][0, gr])
    def attack(s):
        o = hold(s)
        o.kind[:, 0], o.target[:, 0] = O.ATTACK, gr
        return o
    battle.step(st, attack(st), P)
    assert int(st.u["order_kind"][0, 0]) == O.HOLD and int(st.u["order_target"][0, 0]) == -1
    run(st, 20)                                          # the Gutter Runners hold out of their stars' range (100 m)
    assert float(st.u["hp_abs"][0, gr]) == kill_hp and int(st.u["aim_tgt"][0, 0]) == -1
    assert missile.choose_target(st.u, geometry.pairwise(st.u, 1.5), torch.ones_like(st.u["hid"]),
                                 torch.full_like(st.u["target"], -1), torch.zeros_like(st.u["hid"]))[0, 0] == -1
    # seen, the same handgunners shoot it
    st.u["hid"][0, gr] = False
    st.u["vis"][0, gr] = True
    st.u["hid_s"][0, gr] = 0.0
    off = with_sim(stealth={"on": 0})
    run(st, 20, params=off)
    assert float(st.u["hp_abs"][0, gr]) < kill_hp


def test_no_blows_fall_on_a_hidden_enemy_and_contact_reveals_it():
    st = scenario.build([duel(SWORD, GR, 10, own_side=EMP)], P)
    g = gap_of(st)
    st = scenario.build([duel(SWORD, GR, 10 - (g + 4.0), own_side=EMP)], P)     # the formations overlap by 4 m
    gr = st.N // 2
    hp0 = float(st.u["hp_abs"][0, gr])
    assert bool(st.u["hid"][0, gr])
    battle.step(st, hold(st), P)
    assert float(st.u["hp_abs"][0, gr]) == hp0               # in contact, still hidden at the step's start: no blows
    assert bool(st.u["vis"][0, gr]) and bool(st.u["m"][0, 0])     # melee (and the enemy's eyes) reveal it
    run(st, 4)
    assert float(st.u["hp_abs"][0, gr]) < hp0


# --- a projectile through several men ------------------------------------------------------------------------------

def test_men_struck_by_the_rule():
    st = scenario.build([duel(JEZ, SPEAR, 200, extra=((1, LR, 0, 40), (1, IRON, 0, 80), (1, HANDGUN, 0, 120),
                                                       (2, RATS, 200, 40)))], P)
    n = missile.men_struck(st.u, missile.pen_limit(P))[0]
    H = st.N // 2
    jez, lr, iron, gun = 0, 1, 2, 3
    spear, rats = H, H + 1
    assert missile.pen_limit(P) == 2
    assert n[jez, spear] == 2 and n[lr, spear] == 2 and n[iron, spear] == 1 and n[gun, spear] == 1
    assert n[jez, rats] == 1


def test_the_penetration_factor():
    st = scenario.build([duel(JEZ, SPEAR, 150, extra=((1, HANDGUN, 0, 40),))], P)
    pw = geometry.pairwise(st.u, P.sim["formation"]["spacing_m"])
    zero = torch.zeros_like(pw["dist"])
    f = missile.penetration(st.u, pw, P, zero)[0]
    H = st.N // 2
    assert 1.5 < float(f[0, H]) <= 2.0                       # most bullets have a man behind the first (front_share)
    assert float(f[1, H]) == 1.0                             # handguns: one man
    half = missile.penetration(st.u, pw, P, zero + 0.5)[0]
    assert float(half[0, H]) - 1 == pytest.approx((float(f[0, H]) - 1) * 0.5, rel=1e-5)   # the shield stops it too
    off = with_sim(missile={"penetration": {"on": 0, "max_men": 8}})
    assert float(missile.penetration(st.u, pw, off, zero)[0, 0, H]) == 1.0


def test_jezzails_kill_more_with_penetration():
    def lost(params):
        st = scenario.build([duel(JEZ, SPEAR, 150, own_side=SKV)], params)
        run(st, 60, params=params)
        return float(st.u["hp0"][0, st.N // 2] - st.u["hp_abs"][0, st.N // 2])
    on, off = lost(P), lost(with_sim(missile={"penetration": {"on": 0, "max_men": 8}}))
    assert off > 0 and on > 1.4 * off


# --- the Snare Net, the shield breaker, Blackpowder Discipline -------------------------------------------------------

def speeds_after(st, params=P):
    """(walk, run, charge) of every unit as laid this step (effects.apply)."""
    pw = geometry.pairwise(st.u, 1.5)
    u = st.u
    alive = (u["side"] > 0) & (u["men"] > 0)
    same = u["side"][:, :, None] == u["side"][:, None, :]
    old = effects.apply(u, params, params.dt, alive, torch.zeros_like(alive), pw["dist"], same)
    out = (u["walk"].clone(), u["run"].clone(), u["charge_speed"].clone(), u["shield"].clone(), u["acc"].clone(),
           u["reload"].clone())
    effects.restore(u, old)
    return out


def test_the_snare_net_slows_the_enemies_within_35_m():
    for gap, slowed in ((30.0, True), (40.0, False)):
        st = scenario.build([duel(GR, SWORD, gap)], P)
        H = st.N // 2
        w0, r0, c0 = (float(st.u[k][0, H]) for k in ("walk", "run", "charge_speed"))
        w, r, c, *_ = speeds_after(st)
        if slowed:
            assert float(w[0, H]) == pytest.approx(0.9 * w0) and float(r[0, H]) == pytest.approx(0.9 * r0)
            assert float(c[0, H]) == pytest.approx(0.81 * c0)        # the catalogue's charge 0.81 (= 0.9 x 0.9), alone
        else:
            assert float(r[0, H]) == pytest.approx(r0) and float(c[0, H]) == pytest.approx(c0)
        assert float(r[0, 0]) == pytest.approx(float(st.u["run"][0, 0]) * 0.9)     # its own Strength in Numbers only
    # two nets do not stack (an assumption: not described)
    st = scenario.build([duel(GR, SWORD, 30, extra=((1, SLINGS, 0, 10),))], P)
    H = st.N // 2
    _, r, _, *_ = speeds_after(st)
    assert float(r[0, H]) == pytest.approx(0.9 * float(st.u["run"][0, H]))


def test_the_nets_multiplier_adds_to_poison():
    st = scenario.build([duel(GR, SWORD, 30)], P)
    H = st.N // 2
    j = [j for j, i, k in effects.contact_keys(P) if k == "wh_main_unit_contact_poison"][0]
    st.u[f"cfx{j}_s"][0, H] = 10.0
    _, r, _, *_ = speeds_after(st)
    assert float(r[0, H]) == pytest.approx((1 - 0.1 - 0.15) * float(st.u["run"][0, H]))    # x0.75, not x0.765


def test_the_shield_breaker_takes_24_points_for_10_s():
    st = scenario.build([duel(JEZ, SHIELD_SPEAR, 120)], P)
    H = st.N // 2
    j = [j for j, i, k in effects.contact_keys(P) if k == "wh2_dlc09_unit_contact_shieldbreaker"][0]
    st.u[f"cfx{j}_s"][0, H] = 10.0
    *_, shield, _, _ = speeds_after(st)
    assert float(st.u["shield"][0, H]) == pytest.approx(0.35) and float(shield[0, H]) == pytest.approx(0.11)
    # a hit lays it: the Jezzails shooting the spearmen
    st = scenario.build([duel(JEZ, SHIELD_SPEAR, 120)], P)
    laid = False
    for _ in range(60):
        battle.step(st, hold(st), P)
        laid = laid or float(st.u[f"cfx{j}_s"][0, H]) > 0
    assert laid


def test_blackpowder_discipline_after_10_s_standing():
    st = scenario.build([duel(IRON, SWORD, 400, own_side=EMP)], P)
    acc0, rl0 = float(st.u["acc"][0, 0]), float(st.u["reload"][0, 0])
    run(st, 18)                                             # 9 s standing (+ speeds_after's step: 9.5 s)
    *_, acc, rl = speeds_after(st)
    assert float(acc[0, 0]) == pytest.approx(acc0) and float(rl[0, 0]) == pytest.approx(rl0)
    run(st, 2)                                              # 10 s and more
    *_, acc, rl = speeds_after(st)
    # the reload skill acts, the accuracy does not (sim.json effects.stats_off: the probe of 25.10.2026)
    assert float(acc[0, 0]) == pytest.approx(acc0)
    assert float(rl[0, 0]) == pytest.approx(rl0 * max(4.0, 7.0 * 0.9) / 7.0)    # max(4 s, base x (100 - 10) %)
    full = with_sim(effects={"stats_off": {}})
    *_, acc_db, _ = speeds_after(st, full)
    assert float(acc_db[0, 0]) == pytest.approx(acc0 + 40)                       # the database's +40, switched on
    # moving puts it off and starts its clock again
    def walk(s):
        o = hold(s)
        o.kind[:, 0], o.x[:, 0], o.z[:, 0] = O.MOVE, -100.0, 0.0
        return o
    run(st, 2, walk)
    *_, acc, _ = speeds_after(st)
    assert float(acc[0, 0]) == pytest.approx(acc0) and float(st.u["fxd_s"][0, 0]) == 0.0


def test_the_reload_skill_has_a_4_s_floor():
    st = scenario.build([duel(IRON, SWORD, 400, own_side=EMP)], P)
    u = st.u
    rl0 = float(u["reload"][0, 0])
    T = effects.table(P, u["men"].device)
    i = effects.order(P).index("wh3_dlc25_unit_passive_blackpowder_discipline")
    T2 = T.clone()
    T2[i, effects.COL["self_reloading"]] = 60.0              # a skill of 60: 7 x 0.4 = 2.8 -> 4 s
    orig = effects.table
    effects.table = lambda params, device: T2
    try:
        u["fxd_s"][0, 0] = 20.0
        u["mv"][0, 0] = False
        *_, rl = speeds_after(st)
    finally:
        effects.table = orig
    assert float(rl[0, 0]) == pytest.approx(rl0 * 4.0 / 7.0)


# --- the fighters lab ------------------------------------------------------------------------------------------------

def test_the_lab_traces_a_bullet_on_to_the_man_behind():
    from tools.nn.fighters import kernels as K
    from tools.nn.fighters import cells
    # three men in a file along +x, 1 m apart, radius 0.4, height 1.8; a flat bullet from x = -100 landing at x = 5
    px = torch.tensor([[0.0, 1.0, 2.0]])
    pz = torch.zeros(1, 3)
    uid = torch.zeros(1, 3, dtype=torch.long)
    hittable = torch.ones(1, 3, dtype=torch.bool)
    table = cells.Table(px, pz, hittable, 2.0, 64)
    radius, height = torch.tensor([[0.4]]), torch.tensor([[1.8]])
    lx, lz = torch.tensor([[5.0]]), torch.zeros(1, 1)
    ex, ez = torch.ones(1, 1), torch.zeros(1, 1)
    tan = torch.full((1, 1), 0.05)
    sh = torch.full((1, 1), -1, dtype=torch.long)
    ok = torch.ones(1, 1, dtype=torch.bool)
    first = K.path_first(table, 8, 8, px, pz, hittable, uid, radius, height, lx, lz, ex, ez, tan, sh, ok,
                         use_triton=False)
    assert int(first[0, 0]) == 0
    back = -((px[0, 0] - lx) * ex)
    nxt = K.path_first(table, 8, 8, px, pz, hittable, uid, radius, height, lx, lz, ex, ez, tan, sh, ok,
                       use_triton=False, bmax=back - 1e-4)
    assert int(nxt[0, 0]) == 1


def test_the_nets_charge_is_the_catalogues_081_and_its_speed_09():
    e = P.effects["effects"]["wh2_main_unit_passive_snare_net"]
    st = {x["stat"]: x for x in e["stats"]}
    assert st["scalar_charge_speed"]["value"] == 0.9 and st["scalar_charge_speed"]["sim_value"] == 0.81
    assert st["scalar_speed"]["value"] == 0.9 and "sim_value" not in st["scalar_speed"]
    T = effects.table(P, "cpu")
    i = effects.order(P).index("wh2_main_unit_passive_snare_net")
    assert float(T[i, effects.COL["enemies_charge_speed"]]) == pytest.approx(0.81)
    assert float(T[i, effects.COL["enemies_speed"]]) == pytest.approx(0.9)
    # with poison (x0.85 speed): the charge 1 - 0.19 - 0.15 = 0.66 (the net's own charge multiplier, added)
    st2 = scenario.build([duel(GR, SWORD, 30)], P)
    H = st2.N // 2
    j = [j for j, i_, k in effects.contact_keys(P) if k == "wh_main_unit_contact_poison"][0]
    st2.u[f"cfx{j}_s"][0, H] = 10.0
    _, _, c, *_ = speeds_after(st2)
    assert float(c[0, H]) == pytest.approx((1 - 0.19 - 0.15) * float(st2.u["charge_speed"][0, H]))


def test_the_penetration_limit_and_hiding_are_worked_out_once():
    assert P.pen_men == 2 and missile.pen_limit(P) == 2 and P.can_hide and stealth.on(P)
    old = {k: u for k, u in P.units.items() if "gutter" not in k and "jezzail" not in k and "long_rifles" not in k
           and "huntsmen" not in k}
    q = Params(old, P.rules, P.sim, P.abilities, P.effects, P.spells)
    # (hide_forest units hide in woods: stealth is on while any unit can hide - ground.py's forest)
    assert q.pen_men == 1 and q.can_hide and stealth.on(q)
    bare = {k: {**u, "attributes": [a for a in u.get("attributes") or () if a not in ("stalk", "hide_forest")]}
            for k, u in old.items()}
    q2 = Params(bare, P.rules, P.sim, P.abilities, P.effects, P.spells)
    assert not q2.can_hide and not stealth.on(q2)
