"""The fighters lab (tools/nn/fighters): the men match their units, the formation's places, the file closing up, the
neighbour search, a blow and a projectile land on one man, overkill is lost, the lab runs inside the simulator's step
and leaves the simulator alone outside it. Needs torch: run in the container."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.fighters import cells, hook, melee as FM, missile as FS, state as S  # noqa: E402
from tools.nn.sim import battle, geometry, melee, replay, scenario  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load().with_cal("noise", blows=False)
EMP, SKV = "wh_main_emp_empire", "wh2_main_skv_skaven"
SWORDS, CLANRATS, ARCHERS = "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1", "wh2_dlc13_emp_inf_archers_0"


def army(units1, units2):
    """units: [(key, x, z, bearing, men or None)]."""
    def side(faction, units):
        return {"faction": faction, "units": [{"key": k, "x": x, "z": z, "b": b, "width": 30,
                                               **({"men": m} if m else {})} for k, x, z, b, m in units]}
    return {"attacker": 1, "sides": {1: side(EMP, units1), 2: side(SKV, units2)}}


def build(a, B=1):
    with hook.installed() as h:
        st = scenario.build([a] * B, P)
        return st, h.F


def test_men_match_units_and_form_up():
    st, F = build(army([(SWORDS, 0, 0, 0, None)], [(CLANRATS, 0, 60, 180, 100)]))
    u = st.u
    live = S.alive(F)
    men = S.unit_count(F, live, st.N)
    assert torch.equal(men, u["men"])                          # 120 swordsmen, 100 of 160 clanrats alive
    hp = S.unit_sum(F, torch.where(live, F.hp, torch.zeros_like(F.hp)), st.N)
    assert torch.allclose(hp, u["hp_abs"], atol=0.6)
    assert float(F.hp[0][live[0] & (F.uid[0] == 0)].max()) == float(u["hp_man"][0, 0])
    # the swordsmen's men: 20 files 1.48 m apart (30 m // 1.48), 6 ranks 1.6 m apart, round the unit's centre
    px, pz = S.positions(F, u)
    m = live[0] & (F.uid[0] == 0)
    xs, zs = px[0][m], pz[0][m]
    assert abs(float(xs.mean())) < 0.05 and abs(float(zs.mean())) < 0.05
    assert abs(float(xs.max() - xs.min()) - 19 * 1.48) < 1e-3
    assert abs(float(zs.max() - zs.min()) - 5 * 1.6) < 1e-3
    assert float(zs.max()) > 3.9                                # the front rank ahead (facing +z, bearing 0)


def test_the_man_behind_steps_up():
    st, F = build(army([(SWORDS, 0, 0, 0, None)], [(CLANRATS, 0, 60, 180, None)]))
    live = S.alive(F)
    r0 = S.rank_now(F, live)
    # kill the front man of the first file: the man behind him takes rank 0, the file closes up
    front = live & (F.uid == 0) & (F.file == 0) & (F.rank0 == 0)
    behind = live & (F.uid == 0) & (F.file == 0) & (F.rank0 == 1)
    F.hp = torch.where(front, torch.zeros_like(F.hp), F.hp)
    r1 = S.rank_now(F, S.alive(F))
    assert float(r0[behind]) == 1 and float(r1[behind]) == 0
    other = live & (F.uid == 0) & (F.file == 1) & (F.rank0 == 1)
    assert float(r1[other]) == 1


def test_conform_kills_the_most_wounded_and_sets_health():
    st, F = build(army([(SWORDS, 0, 0, 0, None)], [(CLANRATS, 0, 60, 180, None)]))
    u = st.u
    live = S.alive(F)
    sw = torch.nonzero(live[0] & (F.uid[0] == 0)).squeeze(1)
    F.hp[0, sw[:5]] = 10.0                                      # five badly wounded swordsmen
    u["men"][0, 0] = 115.0
    u["hp_abs"][0, 0] = 115 * 69 - 300.0
    S.conform(F, u)
    live = S.alive(F)
    assert int((live[0] & (F.uid[0] == 0)).sum()) == 115
    assert not bool(live[0, sw[:5]].any())                      # they died
    hp = float(F.hp[0][live[0] & (F.uid[0] == 0)].sum())
    assert abs(hp - (115 * 69 - 300.0)) < 1.0


def test_cells_find_every_neighbour_within_the_cell():
    g = torch.Generator().manual_seed(0)
    x = torch.rand((2, 400), generator=g) * 60
    z = torch.rand((2, 400), generator=g) * 60
    valid = torch.ones((2, 400), dtype=torch.bool)
    t = cells.Table(x, z, valid, 3.6, 4096)
    cand = t.query(x, z, 40)
    d = torch.sqrt((x[:, :, None] - x[:, None, :]) ** 2 + (z[:, :, None] - z[:, None, :]) ** 2)
    want = d <= 3.6
    got = torch.zeros((2, 400, 401), dtype=torch.bool)
    got.scatter_(2, torch.where(cand >= 0, cand, torch.full_like(cand, 400)), torch.ones_like(cand, dtype=torch.bool))
    assert bool((got[..., :400] | ~want).all())                 # nothing within 3.6 m is missed


def _melee_pair(gap_m):
    """Swordsmen and clanrats front to front gap_m apart (their front men), each attacking the other."""
    a = army([(SWORDS, 0, 0, 0, None)], [(CLANRATS, 0, 0, 180, None)])
    st, F = build(a)
    u = st.u
    N = st.N
    # the front ranks gap_m apart along z (the swordsmen face +z, the clanrats -z)
    px, pz = S.positions(F, u)
    live = S.alive(F)[0]
    sw_front = float(pz[0][live & (F.uid[0] == 0)].max())
    cr_front = float(pz[0][live & (F.uid[0] == N // 2)].min()) - float(u["z"][0, N // 2])
    u["z"][0, N // 2] = sw_front + gap_m - cr_front
    return st, F


def test_a_blow_needs_reach_and_lands_on_one_man():
    st, F = _melee_pair(2.0)
    u = st.u
    lab = dict(hook.LAB)
    lab["melee"] = dict(lab["melee"], blow_share=1.0)
    near = hook.Near(F, u, lab)
    N = st.N
    permit = torch.zeros((1, N, N), dtype=torch.bool)
    permit[0, 0, N // 2] = True                                 # only the swordsmen may strike
    pw = geometry.pairwise(u, 1.5)
    hp, men = FM.resolve(F, u, P, lab, permit, pw, torch.zeros_like(u["men"]), 0.5, near)
    # the first rank's 20 men are within reach (2.0 m < 2.5 + 0.65), the second (3.6 m) is not
    assert float(men[0, 0, N // 2]) == 20
    assert float(hp[0, N // 2, 0]) == 0                         # the clanrats were not allowed to strike
    struck = F.pend[0] > 0
    assert bool((F.uid[0][struck] == N // 2).all())             # only clanrats took damage
    assert float(F.pend[0].max()) <= 60.0 + 1e-4                # a man takes at most his health (overkill lost)
    assert abs(float(F.pend[0].sum()) - float(hp[0, 0, N // 2])) < 1e-3


def test_no_blow_out_of_reach():
    st, F = _melee_pair(4.0)
    u = st.u
    near = hook.Near(F, u, hook.LAB)
    N = st.N
    permit = torch.ones((1, N, N), dtype=torch.bool)
    pw = geometry.pairwise(u, 1.5)
    hp, men = FM.resolve(F, u, P, hook.LAB, permit, pw, torch.zeros_like(u["men"]), 0.5, near)
    assert float(hp.sum()) == 0 and float(men.sum()) == 0


def test_a_projectile_hits_one_man_of_the_unit_in_its_path():
    # archers shoot at clanrats 100 m off; swordsmen stand 4 m in front of the clanrats on the line of fire: some of the
    # projectiles aimed at the clanrats fall on the swordsmen's men (friendly fire from the geometry, no rule)
    a = army([(ARCHERS, 0, 0, 0, None), (SWORDS, 0, 86, 0, None)], [(CLANRATS, 0, 100, 180, None)])
    st, F = build(a, B=2)
    u = st.u
    near = hook.Near(F, u, hook.LAB)
    N = st.N
    shots = torch.zeros((2, N))
    shots[:, 0] = 90.0
    target = torch.full((2, N), -1, dtype=torch.long)
    target[:, 0] = N // 2
    hp, traced, wanted = FS.fire(F, u, P, hook.LAB, shots, target, near, 0.5)
    assert traced == wanted and traced > 150
    on_clan, on_friend = float(hp[:, 0, N // 2].sum()), float(hp[:, 0, 1].sum())
    assert on_clan > 0 and on_friend > 0
    # a hit is one man's: never beyond his health (overkill lost)
    hit = F.pend > 0
    assert bool((F.pend[hit] <= F.hp[hit] + 1e-4).all())
    assert int(hit.sum()) < traced                              # misses, shields, two arrows on one man


def test_the_lab_runs_in_the_step_and_the_simulator_is_left_alone():
    a = army([(SWORDS, -15, 0, 90, None), (ARCHERS, -60, 0, 90, None)], [(CLANRATS, 15, 0, 270, None)])
    orig = (battle.step, melee.strikes, geometry.contact_limit, scenario.build)
    with hook.installed() as h:
        st = scenario.build([a] * 2, P)
        F = h.F
        for _ in range(60):
            battle.step(st, replay.nearest_attack(st), P, P.dt)
        live = S.alive(F)
        assert torch.equal(S.unit_count(F, live, st.N), st.u["men"])
        assert float(st.u["hp"][:, st.N // 2].max()) < 1.0      # the clanrats were hurt
        assert bool(st.u["m"][:, 0].all())                       # the men's contact lit the melee flag
        # a copy (the step replay's probe step) runs the plain simulator
        probe = st.clone()
        battle.step(probe, replay.nearest_attack(probe), P, P.dt)
    assert (battle.step, melee.strikes, geometry.contact_limit, scenario.build) == orig
    assert not math.isnan(float(st.u["hp_abs"].sum()))


def _doom(st, n=12.0):
    """n Searing Doom projectiles this step on the clanrats' centre, 'cast' by the swordsmen's slot (knock.bombard)."""
    from tools.nn.sim import magic
    u = st.u
    N = st.N
    u["sp0"][0, 0] = magic.keys(P).index("wh_main_spell_metal_searing_doom")
    u["bm0_x"][0, 0], u["bm0_z"][0, 0] = u["x"][0, N // 2], u["z"][0, N // 2]
    return [(0, torch.tensor([[0, 0]]), torch.tensor([n]))]


def test_a_spell_knocks_men_down_and_the_lying_take_nothing():
    from tools.nn.fighters import knock as FK
    st, F = build(army([(SWORDS, 0, 0, 0, None)], [(CLANRATS, 0, 60, 180, None)]))
    u = st.u
    N = st.N
    hp, hits = FK.bombard(F, u, P, hook.LAB, _doom(st), 0.5)
    assert float(hp[0, 0, N // 2]) > 0 and float(hp[0, 0, 0]) == 0           # the clanrats, not the 'caster'
    lying = F.down > 0
    assert int(lying.sum()) > 0 and bool((F.uid[lying] == N // 2).all())
    lo, hi = P.battle["knockdown_getting_up_delay_min"], P.battle["knockdown_getting_up_delay_max"]
    assert float(F.down[lying].min()) >= lo and float(F.down[lying].max()) <= hi
    # lying, they take nothing from the next projectiles
    before = F.pend[lying].clone()
    FK.bombard(F, u, P, hook.LAB, _doom(st, 40.0), 0.5)
    assert torch.equal(F.pend[lying], before)
    # ... strike nobody in melee and nobody strikes them
    u["z"][0, N // 2] = 12.0                                    # the clanrats on the swordsmen
    near = hook.Near(F, u, hook.LAB)
    F.tgt = torch.full_like(F.tgt, -1)
    permit = torch.ones((1, N, N), dtype=torch.bool)
    FM.resolve(F, u, P, hook.LAB, permit, geometry.pairwise(u, 1.5), torch.zeros_like(u["men"]), 0.5, near)
    assert int((F.tgt >= 0).sum()) > 0
    assert not bool((F.tgt[lying] >= 0).any())
    assert not bool(lying[0, F.tgt[F.tgt >= 0]].any())
    # they get up after their time
    for _ in range(6):
        FK.tick(F, 0.5)
    assert not bool((F.down > 0).any())


def test_places_from_the_recording_put_the_men_and_hold_the_shape():
    # tools/nn/fighters/places.py: the living men onto the recorded places (least way), held as the unit's places
    from tools.nn.fighters import places
    st, F = build(army([(SWORDS, 0, 0, 0, 10)], [(CLANRATS, 0, 60, 180, None)]))
    u = st.u
    F.fixed = torch.zeros_like(F.valid)
    F.fa, F.ff = torch.zeros_like(F.dx), torch.zeros_like(F.dx)
    pts = torch.stack([torch.arange(10.0) * 0.8 - 3.6, torch.full((10,), 2.0)], 1).numpy()   # one line 2 m ahead
    assert places.put(F, u, 0, 0, pts) == 10
    px, pz = S.positions(F, u)
    m = S.alive(F)[0] & (F.uid[0] == 0)
    got = sorted(zip(px[0][m].tolist(), pz[0][m].tolist()))
    assert all(abs(a - b) < 1e-4 and abs(c - 2.0) < 1e-4 for (a, c), b in zip(got, sorted(pts[:, 0].tolist())))
    with places.installed():
        sa, sf = S.slots(F, u)                      # the places are the recorded ones in the unit's frame
        assert torch.allclose(sf[0][m], torch.full_like(sf[0][m], 2.0), atol=1e-4)
        # turned with the unit: a quarter turn carries the recorded shape
        u["b"] = torch.full_like(u["b"], 90.0)
        gx, gz = S.to_world(u, F, *S.slots(F, u))
        assert torch.allclose(gx[0][m], torch.full_like(gx[0][m], 2.0), atol=1e-4)
    assert not torch.allclose(S.slots(F, u)[1][0][m], torch.full_like(sf[0][m], 2.0))   # outside: the formation


def test_unit_features_of_a_block_and_its_enemy():
    # shape.unit_features: a 10 x 6 block 1.5 m apart facing +z, an enemy line 2 m before its front
    import numpy as np
    from tools.nn.fighters import shape
    xs = (np.arange(10) - 4.5) * 1.5
    block = np.array([(x, (r - 2.5) * 1.5) for r in range(6) for x in xs])
    line = np.array([(x, 3.75 + 2.0) for x in xs])
    f = shape.features_from_rows({"a": block, "e": line}, ["a", "e"], [1, 2], [0.0, 180.0], [60, 10], [0.9, 1.0])
    F = dict(zip(shape.FEATURES, f[0, :len(shape.FEATURES)].tolist()))
    assert abs(F["width_m"] - 1.5 * math.sqrt(99)) < 0.05 and abs(F["depth_m"] - 1.5 * math.sqrt(35)) < 0.05
    assert abs(F["men_m2"] - 60 / (F["width_m"] * F["depth_m"])) < 1e-4
    assert abs(F["skew_fwd_m"]) < 1e-4 and abs(F["skew_side_m"]) < 1e-4
    assert abs(F["contact"] - 1 / 6) < 1e-6                    # the front rank (10 of 60) within 3.15 m
    assert F["holes"] == 0 and F["front_holes"] == 0
    assert abs(F["wounded"] - 0.1) < 1e-5                      # 90 % of the start's health on all 60 men
    grid = f[0, len(shape.FEATURES):].reshape(4, 4)
    assert abs(float(grid.sum()) - 1) < 1e-5 and float(grid[3].sum()) > 0   # the front row (index 3) holds men
    # the rear half gone: the mass at the front, holes behind
    f2 = shape.features_from_rows({"a": block[30:], "e": line}, ["a", "e"], [1, 2], [0.0, 180.0], [60, 10],
                                  [0.6, 1.0])
    assert float(f2[0, 2]) < F["depth_m"] and abs(float(f2[0, 3])) < 1e-4
    # turned: the same numbers in its own frame
    rot = lambda p: np.stack([p[:, 1], -p[:, 0]], 1)            # a quarter turn: facing +x is bearing 90
    f3 = shape.features_from_rows({"a": rot(block), "e": rot(line)}, ["a", "e"], [1, 2], [90.0, 270.0], [60, 10],
                                  [0.9, 1.0])
    assert torch.allclose(f3, f, atol=1e-4)


def test_the_lab_gives_the_network_its_units_shape():
    # hook.installed(shape=True): after every step the state carries shape.unit_features and shape_ok (observation
    # SHAPE); without it the state has none (the rectangle: zeros, shape_known 0)
    from tools.nn.fighters import shape
    from tools.nn.model import observation as ob
    a = army([(SWORDS, -15, 0, 90, None)], [(CLANRATS, 15, 0, 270, None)])
    with hook.installed(shape=True):
        st = scenario.build([a], P)
        for _ in range(6):
            battle.step(st, replay.nearest_attack(st), P, P.dt)
        o = st.observation()
    N = st.N
    assert tuple(o["shape"].shape) == (1, N, len(shape.FEATURES) + shape.GRID ** 2)
    assert o["shape_ok"][0].tolist() == [n in (0, N // 2) for n in range(N)]      # the two units with men
    assert float(o["shape"][0, 0, 1]) > 20 and 0 < float(o["shape"][0, 0, 5]) <= 1   # width; within reach
    plain = scenario.build([a], P)
    assert "shape" not in plain.observation()
    assert ob.NAMES[-len(ob.SHAPE):] == ob.SHAPE


def _rats_in_reach(free_on, steps=24):
    """Wolf rats attacking swordsmen (nearest_attack), the lab's move.free on or off: (rats within reach of a
    swordsman - 2.5 m + 0.65 -, rats alive) after steps."""
    import copy
    lab = copy.deepcopy(hook.LAB)
    lab["move"]["free"]["on"] = free_on
    a = army([(SWORDS, 0, 0, 0, None)], [("wh2_dlc16_skv_mon_wolf_rats_0", 0, 25, 180, None)])
    with hook.installed(lab=lab) as h:
        st = scenario.build([a], P)
        F = h.F
        for _ in range(steps):
            battle.step(st, replay.nearest_attack(st), P, P.dt)
        px, pz = S.positions(F, st.u)
        live = S.alive(F)[0]
        N = st.N
        r = live & (F.uid[0] == N // 2)
        s = live & (F.uid[0] == 0)
        d = torch.sqrt((px[0][r][:, None] - px[0][s][None, :]) ** 2 + (pz[0][r][:, None] - pz[0][s][None, :]) ** 2)
        return int((d.min(1).values <= 3.15).sum()), int(r.sum())


def test_beasts_fight_free_round_their_target():
    # lab.json move.free: a beast's men (an ellipse body, no missile weapon: the wolf rats) leave their places once
    # their unit is in the fight and each closes on an enemy man - the game's probe lanes: 46-77 of 80 rats within
    # reach against the formed lab's 15-20
    near_free, alive = _rats_in_reach(True)
    near_formed, _ = _rats_in_reach(False)
    assert near_free > 1.5 * near_formed and near_free > 0.4 * alive
    from tools.nn.fighters import move as MV
    u = {"body_ratio": torch.tensor([[1.0, 0.4, 0.6, 0.4]]), "range": torch.tensor([[0.0, 0.0, 120.0, 0.0]]),
         "men0": torch.tensor([[120.0, 80.0, 30.0, 1.0]])}
    assert MV.free_units(u, hook.LAB)[0].tolist() == [False, True, False, False]


def test_a_man_takes_at_most_k_attackers_blows():
    # lab.json melee.max_attackers (off by default): with K, the blows landing on a man fought by n men in reach
    # are cut to K / n of them
    import copy
    got = []
    for K in (None, 1):
        # 120 swordsmen on 6 clanrats spread along their front: several swordsmen in reach of each clanrat
        a = army([(SWORDS, 0, 0, 0, None)], [(CLANRATS, 0, 6.0, 180, 6)])
        st, F = build(a)
        u = st.u
        N = st.N
        lab = copy.deepcopy(hook.LAB)
        lab["melee"]["max_attackers"] = K
        permit = torch.zeros((1, N, N), dtype=torch.bool)
        permit[0, 0, N // 2] = True
        pw = geometry.pairwise(u, 1.5)
        near = hook.Near(F, u, lab)
        hp, men = FM.resolve(F, u, P, lab, permit, pw, torch.zeros_like(u["men"]), 0.5, near)
        got.append((float(F.pend[0].sum()), float(men[0, 0, N // 2])))
    assert got[0][1] == got[1][1] and got[0][1] > 6               # the same swordsmen in reach, more than the rats
    assert 0 < got[1][0] < 0.6 * got[0][0]


def _resolve_once(**melee_lab):
    """One melee step of 120 swordsmen on 6 clanrats 6 m ahead (the swordsmen's unit permitted on them): (HP the
    clanrats take, swordsmen with an enemy in reach), the lab's melee section changed by melee_lab."""
    import copy
    a = army([(SWORDS, 0, 0, 0, None)], [(CLANRATS, 0, 6.0, 180, 6)])
    st, F = build(a)
    u = st.u
    N = st.N
    lab = copy.deepcopy(hook.LAB)
    lab["melee"].update(melee_lab)
    permit = torch.zeros((1, N, N), dtype=torch.bool)
    permit[0, 0, N // 2] = True
    pw = geometry.pairwise(u, 1.5)
    near = hook.Near(F, u, lab)
    _, men = FM.resolve(F, u, P, lab, permit, pw, torch.zeros_like(u["men"]), 0.5, near)
    return float(F.pend[0].sum()), float(men[0, 0, N // 2])


def test_the_reach_is_the_combat_distance_plus_the_struck_mans_radius_or_alone():
    # lab.json melee.reach_radius (on): 2.5 m + the struck man's radius; off: 2.5 m centre to centre - fewer in reach
    _, n_on = _resolve_once()
    _, n_off = _resolve_once(reach_radius=False)
    assert n_on > n_off > 0


def test_without_the_target_preference_a_man_strikes_the_nearest():
    # lab.json melee.prefer_target (on): a man of his unit's attack target counts 2 m nearer; off: the nearest. With one
    # enemy unit in reach both pick the same men
    assert _resolve_once()[1] == _resolve_once(prefer_target=False)[1]


def test_an_armour_piercing_shot_leaves_the_shield_part_of_its_chance():
    # the simulator's missile.shield_ap_pierce (the lab's shield follows it): 1 - 0.5 x the AP share of the damage;
    # the Ratling Gun (2 + 6 AP) keeps 0.625 of a shield's chance, off 1
    from tools.nn.sim import missile as SM
    u = {"m_damage": torch.tensor([[2.0, 17.0]]), "m_ap": torch.tensor([[6.0, 2.0]])}
    on = P.with_cal("missile", shield_ap_pierce=True)
    assert SM.shield_pierce(u, P.with_cal("missile", shield_ap_pierce=False)).tolist() == [[1.0, 1.0]]
    got = SM.shield_pierce(u, on)[0].tolist()
    assert got[0] == pytest.approx(0.625) and got[1] == pytest.approx(1 - 0.5 * 2 / 19)
    assert SM.shield_pierce(u, on, torch.tensor([[1, 1, 0]]))[0].tolist() == pytest.approx([got[1], got[1], got[0]])


def test_a_man_with_a_friend_in_the_way_is_blocked():
    # lab.json melee.clear_line (off): melee.blocked_line - a living man of his own side whose body crosses the segment
    # to the man he fights blocks him; a friend beside the line (farther than his radius) does not
    from types import SimpleNamespace
    px = torch.tensor([[0.0, 1.2, 2.5, 0.0, 1.2, 2.5, -1.0]])
    pz = torch.tensor([[0.0, 0.3, 0.0, 5.0, 5.9, 5.0, 0.0]])
    uid = torch.tensor([[0, 0, 1, 0, 0, 1, 0]])                     # unit 0 side 1, unit 1 side 2
    M = px.shape[1]
    cand = torch.tensor([[[j for j in range(M) if j != i] for i in range(M)]])
    near = SimpleNamespace(px=px, pz=pz, cand=cand)
    F = SimpleNamespace(uid=uid, B=1)
    u = {"side": torch.tensor([[1, 2]]), "radius": torch.tensor([[0.65, 0.65]])}
    tgt = torch.tensor([[2, -1, -1, 5, -1, -1, 2]])
    live = torch.ones((1, M), dtype=torch.bool)
    got = FM.blocked_line(F, u, near, tgt, live)[0].tolist()
    assert got == [True, False, False, False, False, False, True]   # 0 behind 1; 3's friend 0.9 m aside; 6 behind 0, 1


def test_a_man_holding_his_enemy_is_not_carried_by_his_unit():
    # lab.json move.hold (off): a man with an enemy man in reach (F.tgt) of a standing unit of 2+ is held - unless his
    # unit routs or its centre jumped (the step replay's reset); off: nobody
    import copy
    from tools.nn.fighters import move as MV
    st, F = build(army([(SWORDS, 0, 0, 0, None)], [(CLANRATS, 0, 60, 180, None)]))
    u = st.u
    lab = copy.deepcopy(hook.LAB)
    assert MV.held(F, u, lab) is None
    lab["move"]["hold"] = {"on": True, "release_m": None}
    F.x_last, F.z_last = u["x"].clone(), u["z"].clone()
    first = S.alive(F) & (F.uid == 0)
    F.tgt = torch.where(first, torch.zeros_like(F.tgt), torch.full_like(F.tgt, -1))
    assert torch.equal(MV.held(F, u, lab), first)
    u["r"][0, 0] = True
    assert not bool(MV.held(F, u, lab).any())
    u["r"][0, 0] = False
    F.x_last = F.x_last - 10.0
    assert not bool(MV.held(F, u, lab).any())


def test_a_fighting_man_behind_the_front_stays_within_the_tether():
    # lab.json move.tether_all (off): a man behind the front with an enemy man in reach stands where he is while within
    # the database's tether (2 m) of his place; off he walks back to his place
    import copy
    from tools.nn.fighters import move as MV
    got = []
    for on in (False, True):
        st, F = build(army([(SWORDS, 0, 0, 0, None)], [(CLANRATS, 0, 60, 180, None)]))
        u = st.u
        lab = copy.deepcopy(hook.LAB)
        lab["move"]["tether_all"] = on
        live = S.alive(F)
        # the outermost man of a middle rank, 1 m out of his place (no neighbour's body there), fighting someone (tgt
        # only marks the fight here)
        m = live & (F.uid == 0) & (F.rank0 == 3)
        k = int(torch.where(m, F.dx, torch.full_like(F.dx, 1e9))[0].argmin())
        F.dx[0, k] = F.dx[0, k] - 1.0
        x0 = float(F.dx[0, k])
        F.tgt[0, k] = 0
        MV.step(F, u, P, lab, P.dt, hook.Near(F, u, lab))
        got.append(abs(float(F.dx[0, k]) - x0))
    assert got[0] > 0.5 and got[1] < 0.2
