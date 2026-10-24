"""tools/nn/general/mission.py and teacher.py: what carrying out each mission kind is, on hand-made states, and the
teacher's rule (docs/en/training/general.md "Missions"). torch: the training container, on the CPU."""
import math

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import mission, teacher                  # noqa: E402

CFG = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2)
G, C = CFG.places, CFG.target_grid ** 2
K = {k.name: k.index for k in gconfig.MISSION_KINDS}
HALF = 200.0                                                     # a coarse tile's half side (800 m / 4)


def state(pos, side, **kw):
    """One battle: units at pos [(x, z)], sides; kw: per-unit fields (b, m, target, fire, range, a, r, men, gone)."""
    n = len(pos)
    u = {"x": torch.tensor([[p[0] for p in pos]], dtype=torch.float32),
         "z": torch.tensor([[p[1] for p in pos]], dtype=torch.float32),
         "side": torch.tensor([side]), "b": torch.zeros(1, n), "men": torch.full((1, n), 100.0),
         "gone": torch.zeros(1, n, dtype=torch.bool), "r": torch.zeros(1, n, dtype=torch.bool),
         "m": torch.zeros(1, n, dtype=torch.bool), "target": torch.full((1, n), -1), "fire": torch.zeros(1, n,
                                                                                                         dtype=torch.bool),
         "range": torch.zeros(1, n), "a": torch.zeros(1, n)}
    for k, v in kw.items():
        u[k] = torch.tensor([v], dtype=u[k].dtype)
    return u


def run(u, main, missions, tiles=None):
    """The measures of side 1's units: main [n] the plan's groups, missions {group: (kind name, target)}, tiles
    {group: (x, z)} of a tile target. -> (m [n], ok [n])."""
    n = len(main)
    kind = torch.zeros(1, G, dtype=torch.long)
    target = torch.full((1, G), 2 * G, dtype=torch.long)          # (no mission: attack a tile - not counted)
    tgt = torch.full((1, G, 2), float("nan"))
    for g, (k, t) in missions.items():
        kind[0, g], target[0, g] = K[k], t
    for g, p in (tiles or {}).items():
        tgt[0, g] = torch.tensor(p)
    m, ok, _ = mission.measure(u, torch.tensor([0]), torch.tensor([1]), torch.tensor([main]), kind, target, tgt,
                               torch.ones(1, n, dtype=torch.bool), G, C, HALF, CFG)
    return m[0].tolist(), ok[0].tolist()


SIDE = [1, 1, 1, 2, 2, 2]
MAIN = [0, 0, 1, G, G, G + 1]                    # ours: g0 (units 0, 1), g1 (unit 2); theirs: e0 (3, 4), e1 (5)


def test_attack_is_melee_with_the_target_group_else_half_its_closeness_to_its_centre():
    u = state([(0, 0), (0, -150), (500, 0), (0, 0), (30, 0), (900, 900)], SIDE,
              m=[True, False, False, True, False, False], target=[3, -1, -1, 0, -1, -1])
    m, ok = run(u, MAIN, {0: ("attack", G)})
    assert m[0] == pytest.approx(1.0)
    assert m[1] == pytest.approx(0.5 * (1 - math.hypot(15, 150) / 300))     # to e0's centre (15, 0), not unit 3
    u2 = state([(0, 0), (0, -150), (500, 0), (0, 0), (30, 0), (900, 900)], SIDE,
               m=[True, False, False, True, False, False], target=[5, -1, -1, 0, -1, -1])
    m, _ = run(u2, MAIN, {0: ("attack", G)})
    assert m[0] < 1.0                                                       # fighting another group: not the mission
    assert ok[:2] == [True, True] and ok[2:] == [False] * 4                 # g1 has no mission here; theirs never


def test_flank_counts_only_off_the_targets_front():
    # unit 3 faces north (b 0: facing (sin, cos) = (0, 1)); ours fights it from its front, then from its side
    front = state([(0, 20), (0, 100), (500, 0), (0, 0), (300, 300), (900, 900)], SIDE,
                  m=[True, False, False, True, False, False], target=[3, -1, -1, 0, -1, -1])
    m, _ = run(front, MAIN, {0: ("flank", G)})
    assert m[0] == pytest.approx(0.0)
    side = state([(20, 0), (0, 100), (500, 0), (0, 0), (300, 300), (900, 900)], SIDE,
                 m=[True, False, False, True, False, False], target=[3, -1, -1, 0, -1, -1])
    m, _ = run(side, MAIN, {0: ("flank", G)})
    assert m[0] == pytest.approx(1.0)
    # not in melee, behind the nearest target unit, 150 m from the target group's centre: half its closeness
    rear = state([(0, -150), (0, -150), (500, 0), (0, 0), (0, 0), (900, 900)], SIDE)
    m, _ = run(rear, MAIN, {0: ("flank", G)})
    assert m[0] == pytest.approx(0.5 * 0.5)
    assert 45.0 == mission.FLANK_DEG


def test_a_unit_far_from_its_own_group_carries_out_less():
    """Every measure x the cohesion: 1 within 150 m of its group's centre, down to 0 at 300 m."""
    u = state([(0, 0), (0, -450), (500, 0), (0, 0), (30, 0), (900, 900)], SIDE,
              m=[True, False, False, True, False, False], target=[3, -1, -1, 0, -1, -1])
    m, _ = run(u, MAIN, {0: ("attack", G)})                                # g0's centre (0, -225): both 225 m off
    assert m[0] == pytest.approx(0.5)                                       # in melee with e0: 1 x 0.5
    assert m[1] == pytest.approx(0.0)                                       # 450 m from e0 - and 0.5 of it
    far = state([(0, 0), (0, -700), (500, 0), (0, 0), (30, 0), (900, 900)], SIDE,
                m=[True, False, False, True, False, False], target=[3, -1, -1, 0, -1, -1])
    m, _ = run(far, MAIN, {0: ("attack", G)})
    assert m[0] == 0.0                                                      # 350 m from its group's centre
    assert mission.COHESION_M == (150.0, 300.0)


def test_hold_is_within_reach_of_the_tile_or_the_group():
    u = state([(250, 0), (500, 0), (0, 0), (0, 900), (0, 950), (900, 900)], SIDE)
    m, ok = run(u, MAIN, {0: ("hold", 2 * G + 5)}, tiles={0: (100.0, 0.0)})
    assert m[0] == pytest.approx(1.0)                                       # 150 m from the tile's centre (< 200)
    assert m[1] == pytest.approx(1 - (400 - HALF) / (2 * HALF))             # 400 m: down to 0 at 600
    assert ok[0] and ok[1]
    g = state([(0, 100), (0, 200), (0, 0), (0, 900), (0, 950), (900, 900)], SIDE)
    m, _ = run(g, MAIN, {0: ("hold", 1)})
    assert m[0] == pytest.approx(1.0) and m[1] == pytest.approx(0.5)       # our g1 at (0, 0): 100 m reach
    m, ok = run(u, MAIN, {0: ("hold", 2 * G + 5)})                          # a tile outside the map (NaN): not counted
    assert not ok[0] and m[0] == 0.0


def test_screen_is_between_the_group_and_its_nearest_enemy():
    u = state([(10, 100), (0, -50), (0, 0), (0, 200), (500, 500), (900, 900)], SIDE)
    m, ok = run(u, MAIN, {0: ("screen", 1)})
    assert m[0] == pytest.approx(1.0)                                       # half way, 10 m off the line
    assert m[1] == pytest.approx(0.0)                                       # behind the group
    far = state([(130, 100), (0, -50), (0, 0), (0, 200), (500, 500), (900, 900)], SIDE)
    m, _ = run(far, MAIN, {0: ("screen", 1)})
    assert m[0] == pytest.approx(1 - (130 - 50) / 100)                      # 130 m off: down to 0 at 150


def test_shoot_keeps_the_target_in_range_out_of_melee():
    u = state([(0, 0), (0, 0), (500, 0), (0, 100), (0, 100), (900, 900)], SIDE,
              fire=[True, False, False, False, False, False], target=[3, -1, -1, -1, -1, -1],
              range=[150.0, 150.0, 0.0, 0, 0, 0], a=[10.0, 10.0, 0.0, 0, 0, 0])
    m, _ = run(u, MAIN, {0: ("shoot", G)})
    assert m[0] == pytest.approx(1.0) and m[1] == pytest.approx(0.5)       # shooting at it; in range of it
    melee = state([(0, 0), (0, 0), (500, 0), (0, 100), (0, 100), (900, 900)], SIDE,
                  m=[True, False, False, False, False, False], range=[150.0, 0.0, 0.0, 0, 0, 0],
                  a=[10.0, 0.0, 0.0, 0, 0, 0])
    m, _ = run(melee, MAIN, {0: ("shoot", G)})
    assert m[0] == 0.0                                                      # in melee
    assert m[1] == pytest.approx(1.0)                                       # no missiles: at 100 m (50-250) of it


def test_withdraw_goes_to_the_point_away_from_the_enemy_out_of_melee():
    u = state([(0, -300), (0, -300), (0, -300), (0, 100), (0, 120), (900, 900)], SIDE,
              m=[False, True, False, False, False, False])
    m, _ = run(u, MAIN, {0: ("withdraw", 1)})
    assert m[0] == pytest.approx(1.0)                                       # at our g1, 400 m from the enemy
    assert m[1] == 0.0                                                      # in melee


def test_a_target_group_that_is_gone_is_not_counted():
    u = state([(0, 0), (0, -150), (500, 0), (0, 0), (30, 0), (900, 900)], SIDE, r=[False, False, False, True, True,
                                                                                    False])
    m, ok = run(u, MAIN, {0: ("attack", G)})                                # e0 all routing
    assert ok[:2] == [False, False] and m[:2] == [0.0, 0.0]


def test_the_teacher_gives_meaningful_missions():
    """teacher.choose: every group attacks the nearest enemy group; shooters shoot; one group screens the shooters;
    a threatened lord withdraws to the largest other group."""
    N = 8
    # ours: 0 lord (g0), 1-2 spears (g1), 3-4 archers (g2), 5 spear (g3); theirs: 6 (e0 near g1), 7 (e1 far)
    x = torch.tensor([[0.0, 0, 10, 100, 110, 250, 0, 900]])
    z = torch.tensor([[-100.0, 0, 0, -50, -50, 0, 100, 900]])
    main = torch.tensor([[0, 1, 1, 2, 2, 3, G, G + 1]])
    active = torch.zeros(1, 2 * G, dtype=torch.bool)
    active[0, [0, 1, 2, 3, G, G + 1]] = True
    own = torch.tensor([[True] * 6 + [False] * 2])
    standing = torch.ones(1, N, dtype=torch.bool)
    shooter = torch.tensor([[False, False, False, True, True, False, False, False]])
    lord = torch.tensor([[True] + [False] * 7])
    kind, target, use = teacher.choose(main, active, x, z, own, standing, shooter, lord, G, CFG.kinds)
    assert use[0, :4].all() and not use[0, 4:].any()
    assert int(kind[0, 1]) == K["screen"] and int(target[0, 1]) == 2     # the nearest non-shooters screen the archers
    assert int(kind[0, 2]) == K["shoot"] and int(target[0, 2]) == G      # archers shoot the nearest enemy group
    # g0 and g3 both attack e0: g0 (200 m from e0's centre) pins it, g3 (269 m, within 300) flanks it (the flank rule)
    assert int(kind[0, 3]) == K["flank"] and int(target[0, 3]) == G
    assert int(kind[0, 0]) == K["attack"]                                # the lord is 200 m from e0: not threatened
    kind_nf, _, _ = teacher.choose(main, active, x, z, own, standing, shooter, lord, G, CFG.kinds, flank=False)
    assert int(kind_nf[0, 3]) == K["attack"]                             # the rule off: both attack
    kind1, target1 = kind, target
    z2 = z.clone()
    z2[0, 6] = -50.0                                                     # e0 within 150 m of the lord now
    kind, target, _ = teacher.choose(main, active, x, z2, own, standing, shooter, lord, G, CFG.kinds)
    assert int(kind[0, 0]) == K["withdraw"] and int(target[0, 0]) in (1, 2)   # to a largest other group (2 units)
    assert int(kind[0, 3]) == K["attack"]                                # the only attacker of e0 now
    # every pair the teacher gives is one the general may choose (the kind takes the target's type)
    for kd, tg in ((kind, target), (kind1, target1)):
        for g in range(4):
            t = int(tg[0, g])
            tt = "own" if t < G else "enemy" if t < 2 * G else "tile"
            assert tt in gconfig.MISSION_KINDS[int(kd[0, g])].targets
    assert math.isfinite(teacher.THREAT_M)


def test_the_teacher_waits_on_its_tiles_until_the_enemy_is_near_or_our_shooters_have_fired():
    """Before the clash: the groups hold the tile they stand on (shooters shoot) while the enemy is farther than
    HOLD_M and our shooters have not fired FIRE_S; then they attack. Every pair is one the general may choose."""
    N = 6
    # ours: 0 spear (g0), 1-2 archers (g1), 3 spear (g2); theirs: 4 (e0, 600 m ahead), 5 (e1)
    x = torch.tensor([[0.0, 50, 60, -100, 600, 700]])
    z = torch.zeros(1, N)
    main = torch.tensor([[0, 1, 1, 2, G, G + 1]])
    active = torch.zeros(1, 2 * G, dtype=torch.bool)
    active[0, [0, 1, 2, G, G + 1]] = True
    own = torch.tensor([[True] * 4 + [False] * 2])
    standing = torch.ones(1, N, dtype=torch.bool)
    shooter = torch.tensor([[False, True, True, False, False, False]])
    lord = torch.zeros(1, N, dtype=torch.bool)
    tiles = torch.tensor([[[0.0, 0.0], [-100.0, 0.0], [600.0, 0.0]] + [[5000.0, 5000.0]] * (C - 3)])
    tiles_ok = torch.tensor([[True, True, True] + [False] * (C - 3)])
    args = (main, active, x, z, own, standing, shooter, lord, G, CFG.kinds)
    kind, target, use = teacher.choose(*args, tiles=tiles, tiles_ok=tiles_ok, engage=torch.tensor([False]))
    assert int(kind[0, 0]) == K["screen"] and int(target[0, 0]) == 1                # the nearest screens the archers
    assert int(kind[0, 1]) == K["shoot"] and int(target[0, 1]) == G
    assert int(kind[0, 2]) == K["hold"] and int(target[0, 2]) == 2 * G + 1          # its own tile
    assert bool(use[0, :3].all())
    # the shooters have fired long enough: attack
    kind, target, _ = teacher.choose(*args, tiles=tiles, tiles_ok=tiles_ok, engage=torch.tensor([True]))
    assert int(kind[0, 2]) == K["attack"] and int(target[0, 2]) == G
    # the enemy near (within HOLD_M of g2's centre): attack at once
    x2 = x.clone()
    x2[0, 4] = 100.0
    kind, target, _ = teacher.choose(main, active, x2, z, own, standing, shooter, lord, G, CFG.kinds, tiles=tiles,
                                     tiles_ok=tiles_ok, engage=torch.tensor([False]))
    assert int(kind[0, 2]) == K["attack"]
    for g in range(3):
        t = int(target[0, g])
        tt = "own" if t < G else "enemy" if t < 2 * G else "tile"
        assert tt in gconfig.MISSION_KINDS[int(kind[0, g])].targets
    assert teacher.HOLD_M == 250.0 and teacher.FIRE_S == 30.0


def test_the_teacher_flanks_with_the_group_off_the_targets_front():
    """Two groups attack one enemy group: the one nearest to its front point (FRONT_M ahead of its centre along its
    facing) pins it, the other flanks; a group attacking another enemy group alone keeps attacking."""
    N = 6
    # ours: g0 (unit 0) 150 m north of e0, g1 (unit 1) 120 m east of e0, g2 (unit 2) near e1; theirs: 3 e0, 4-5 e1
    x = torch.tensor([[0.0, 120, 1000, 0, 1000, 1010]])
    z = torch.tensor([[150.0, 0, 150, 0, 0, 0]])
    main = torch.tensor([[0, 1, 2, G, G + 1, G + 1]])
    active = torch.zeros(1, 2 * G, dtype=torch.bool)
    active[0, [0, 1, 2, G, G + 1]] = True
    own = torch.tensor([[True] * 3 + [False] * 3])
    standing = torch.ones(1, N, dtype=torch.bool)
    no = torch.zeros(1, N, dtype=torch.bool)
    args = (main, active, x, z, own, standing, no, no, G, CFG.kinds)
    # e0 faces north (b 0): g0 stands at its front and pins it, g1 (nearer to its centre) flanks
    kind, target, use = teacher.choose(*args, facing=torch.zeros(1, N))
    assert int(kind[0, 0]) == K["attack"] and int(kind[0, 1]) == K["flank"]
    assert int(target[0, 0]) == G and int(target[0, 1]) == G and bool(use[0, :3].all())
    assert int(kind[0, 2]) == K["attack"] and int(target[0, 2]) == G + 1
    # e0 faces east (b 90): g1 is at its front now - g1 pins, g0 flanks
    kind, _, _ = teacher.choose(*args, facing=torch.tensor([[0.0, 0, 0, 90, 0, 0]]))
    assert int(kind[0, 1]) == K["attack"] and int(kind[0, 0]) == K["flank"]
    # without facings: by the centres - g1 (120 m) pins, g0 (150 m) flanks
    kind, _, _ = teacher.choose(*args)
    assert int(kind[0, 1]) == K["attack"] and int(kind[0, 0]) == K["flank"]
    assert teacher.FRONT_M == 100.0
    # the second attacker farther than NEAR_FLANK_M from the target: it attacks
    x_far = x.clone()
    x_far[0, 1] = 400.0                                              # g1 400 m east of e0
    kind, _, _ = teacher.choose(main, active, x_far, z, own, standing, no, no, G, CFG.kinds, facing=torch.zeros(1, N))
    assert int(kind[0, 0]) == K["attack"] and int(kind[0, 1]) == K["attack"]
    assert teacher.NEAR_FLANK_M == 300.0


def test_a_fast_group_flanks_the_enemy_group_our_melee_pins():
    """A fast group (role "fast") told attack flanks the enemy group nearest to it that one of our non-fast groups
    attacks; with no enemy group pinned (or no fast units known) it attacks as before."""
    N = 4
    # ours: g0 melee (unit 0) 100 m north of e0, g1 fast (unit 1) 200 m from e1, 500 m from e0; theirs: 2 e0, 3 e1
    x = torch.tensor([[0.0, 400, 0, 600]])
    z = torch.tensor([[100.0, 300, 0, 300]])
    main = torch.tensor([[0, 1, G, G + 1]])
    active = torch.zeros(1, 2 * G, dtype=torch.bool)
    active[0, [0, 1, G, G + 1]] = True
    own = torch.tensor([[True, True, False, False]])
    standing = torch.ones(1, N, dtype=torch.bool)
    no = torch.zeros(1, N, dtype=torch.bool)
    fast = torch.tensor([[False, True, False, False]])
    args = (main, active, x, z, own, standing, no, no, G, CFG.kinds)
    kind, target, use = teacher.choose(*args, fast=fast)
    assert int(kind[0, 0]) == K["attack"] and int(target[0, 0]) == G
    assert int(kind[0, 1]) == K["flank"] and int(target[0, 1]) == G and bool(use[0, 1])
    kind, target, _ = teacher.choose(*args)                          # no fast units known
    assert int(kind[0, 1]) == K["attack"] and int(target[0, 1]) == G + 1
    kind, target, _ = teacher.choose(*args, fast=fast, flank=False)
    assert int(kind[0, 1]) == K["attack"]
    act2 = active.clone()
    act2[0, 0] = False                                               # no melee group: nothing pinned
    kind, target, _ = teacher.choose(main, act2, x, z, own, standing, no, no, G, CFG.kinds, fast=fast)
    assert int(kind[0, 1]) == K["attack"] and int(target[0, 1]) == G + 1
    # the fast group is the only one attacking e0: it does not pin for itself
    x3 = x.clone()
    x3[0, 1] = 50.0
    kind, _, _ = teacher.choose(main, act2, x3, z, own, standing, no, no, G, CFG.kinds, fast=fast)
    assert int(kind[0, 1]) == K["attack"]
    assert teacher.PIN_M == 150.0


def _flank_m(u, flank_by="contact"):
    kind = torch.zeros(1, G, dtype=torch.long)
    target = torch.full((1, G), 2 * G, dtype=torch.long)
    kind[0, 0], target[0, 0] = K["flank"], G
    m, _, _ = mission.measure(u, torch.tensor([0]), torch.tensor([1]), torch.tensor([MAIN]), kind, target,
                              torch.full((1, G, 2), float("nan")), torch.ones(1, 6, dtype=torch.bool), G, C, HALF, CFG,
                              flank_by=flank_by)
    return float(m[0, 0])


def test_flank_in_melee_counts_by_the_side_the_contact_began_on():
    """flank_by "contact": in melee the side of the target the contact began on (hit_sec: row the attacker, column the
    struck unit; 2 flank, 3 rear -> 1, 1 front -> 0), whatever the angle now; no side held: the angle; "angle": the
    angle always."""
    # unit 0 fights unit 3 from its front now (unit 3 turned to face it)
    u = state([(0, 20), (0, 100), (500, 0), (0, 0), (300, 300), (900, 900)], SIDE,
              m=[True, False, False, True, False, False], target=[3, -1, -1, 0, -1, -1])
    for began, want in ((2.0, 1.0), (3.0, 1.0), (1.0, 0.0)):        # flank, rear: a flank; front: not
        hs = torch.zeros(1, 36)
        hs[0, 0 * 6 + 3] = began
        u["hit_sec"] = hs
        assert _flank_m(u) == pytest.approx(want)
        assert _flank_m(u, "angle") == pytest.approx(0.0)            # the angle now: its front
    # a flank contact with a unit of another group, or 3 striking 0's rear: no side held with e0 - the angle (front)
    hs = torch.zeros(1, 36)
    hs[0, 0 * 6 + 5] = 2.0
    hs[0, 3 * 6 + 0] = 3.0
    u["hit_sec"] = hs
    assert _flank_m(u) == pytest.approx(0.0)
    # without the field (the game's state): the angle, no error
    u.pop("hit_sec")
    assert _flank_m(u) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        _flank_m(u, "side")


def test_the_teachers_attacking_groups_wait_for_the_farthest_and_attack_together():
    """The wait by the farthest attacking group: g0 200 m from the enemy, g1 400 m - both hold (by each group alone g0
    would attack now); the enemy within 250 m of both: both attack at once; the shooters' clock: both attack."""
    N = 4
    # ours: 0 spear (g0, 200 m from e0), 1 spear (g1, 400 m); theirs: 2-3 (e0)
    x = torch.tensor([[0.0, -200, 200, 210]])
    z = torch.tensor([[0.0, 300, 0, 0]])
    main = torch.tensor([[0, 1, G, G]])
    active = torch.zeros(1, 2 * G, dtype=torch.bool)
    active[0, [0, 1, G]] = True
    own = torch.tensor([[True, True, False, False]])
    standing = torch.ones(1, N, dtype=torch.bool)
    no = torch.zeros(1, N, dtype=torch.bool)
    tiles = torch.tensor([[[0.0, 0.0], [-200.0, 300.0]] + [[5000.0, 5000.0]] * (C - 2)])
    tiles_ok = torch.tensor([[True, True] + [False] * (C - 2)])
    kw = dict(tiles=tiles, tiles_ok=tiles_ok, engage=torch.tensor([False]), flank=False)
    kind, target, _ = teacher.choose(main, active, x, z, own, standing, no, no, G, CFG.kinds, **kw)
    assert int(kind[0, 0]) == K["hold"] and int(kind[0, 1]) == K["hold"]          # g0 waits for g1
    assert int(target[0, 0]) == 2 * G and int(target[0, 1]) == 2 * G + 1           # each its own tile
    x2, z2 = x.clone(), z.clone()
    x2[0, 2:] = torch.tensor([-50.0, -40.0])
    z2[0, 2:] = torch.tensor([150.0, 150.0])                                       # within 250 m of both
    kind, target, _ = teacher.choose(main, active, x2, z2, own, standing, no, no, G, CFG.kinds, **kw)
    assert int(kind[0, 0]) == K["attack"] and int(kind[0, 1]) == K["attack"]
    assert int(target[0, 0]) == G and int(target[0, 1]) == G
    kind, _, _ = teacher.choose(main, active, x, z, own, standing, no, no, G, CFG.kinds,
                                **dict(kw, engage=torch.tensor([True])))
    assert int(kind[0, 0]) == K["attack"] and int(kind[0, 1]) == K["attack"]


def test_the_teacher_gives_no_withdraw_to_the_lords_group_in_melee():
    """The lord threatened: his group withdraws only out of melee; fighting in melee and holding (none wavering:
    taskmask.fighting) it holds the tile it stands on; a unit wavering lets it withdraw. melee None: as before."""
    N = 5
    # ours: 0 lord + 1 spear (g0), 2 spear (g1); theirs: 3 (e0, 40 m from the lord), 4 (e1)
    x = torch.tensor([[0.0, 20, 300, 40, 900]])
    z = torch.zeros(1, N)
    main = torch.tensor([[0, 0, 1, G, G + 1]])
    active = torch.zeros(1, 2 * G, dtype=torch.bool)
    active[0, [0, 1, G, G + 1]] = True
    own = torch.tensor([[True] * 3 + [False] * 2])
    standing = torch.ones(1, N, dtype=torch.bool)
    no = torch.zeros(1, N, dtype=torch.bool)
    lord = torch.tensor([[True, False, False, False, False]])
    tiles = torch.tensor([[[0.0, 0.0], [300.0, 0.0]] + [[5000.0, 5000.0]] * (C - 2)])
    tiles_ok = torch.tensor([[True, True] + [False] * (C - 2)])
    args = (main, active, x, z, own, standing, no, lord, G, CFG.kinds)
    kw = dict(tiles=tiles, tiles_ok=tiles_ok)
    kind, target, _ = teacher.choose(*args, **kw)                               # no melee known: withdraw
    assert int(kind[0, 0]) == K["withdraw"] and int(target[0, 0]) == 1
    out = torch.tensor([[False] * N])
    kind, _, _ = teacher.choose(*args, **kw, melee=out, waver=out)              # out of melee: withdraw
    assert int(kind[0, 0]) == K["withdraw"]
    melee = torch.tensor([[False, True, False, True, False]])                   # the spear of g0 fights
    kind, target, use = teacher.choose(*args, **kw, melee=melee, waver=out)
    assert int(kind[0, 0]) == K["hold"] and int(target[0, 0]) == 2 * G + 0 and bool(use[0, 0])   # its own tile
    waver = torch.tensor([[False, True, False, False, False]])
    kind, _, _ = teacher.choose(*args, **kw, melee=melee, waver=waver)          # the spear wavers: withdraw
    assert int(kind[0, 0]) == K["withdraw"]
    kind, _, _ = teacher.choose(*args, melee=melee, waver=out)                   # no tiles: no withdraw either
    assert int(kind[0, 0]) != K["withdraw"]


def test_the_teachers_shooters_pick_the_target_their_shots_hurt_most():
    """teacher.shot_hp + shoot_target: slingers (base 7, armour-piercing 1) between greatswords of armour 95 (nearer)
    and militia of armour 25 shoot the militia; a target beyond range + walk x REACH_S counts 0; equal values: the
    nearest; nothing in reach: the nearest; a shield blocks small arms (the simulator's per_hit, sim/melee.py)."""
    from tools.nn.sim import melee as smelee
    N = 4
    # ours: 0-1 slingers (g0); theirs: 2 greatswords (e0, 120 m), 3 militia (e1, 160 m)
    x = torch.tensor([[0.0, 10, 120, 160]])
    z = torch.zeros(1, N)
    main = torch.tensor([[0, 0, G, G + 1]])
    active = torch.zeros(1, 2 * G, dtype=torch.bool)
    active[0, [0, G, G + 1]] = True
    own = torch.tensor([[True, True, False, False]])
    standing = torch.ones(1, N, dtype=torch.bool)
    shooter = torch.tensor([[True, True, False, False]])
    no = torch.zeros(1, N, dtype=torch.bool)
    f = (lambda *v: torch.tensor([list(v)], dtype=torch.float32))
    rng, dmg, ap = f(150, 150, 0, 0), f(7, 7, 0, 0), f(1, 1, 0, 0)
    small = torch.tensor([[True, True, False, False]])
    armour, hp_man, shield, resist, men0 = f(0, 0, 95, 25), f(60, 60, 80, 60), f(0, 0, 0, 0), f(0, 0, 0, 0), f(80,
                                                                                                              80, 60, 90)
    walk = f(4, 4, 0, 0)
    shot = teacher.shot_hp(x, z, rng, dmg, ap, small, armour, hp_man, shield, resist, men0, walk=walk)
    want = smelee.per_hit(torch.tensor(7.0), torch.tensor(1.0), torch.tensor(25.0), torch.tensor(60.0))
    assert abs(float(shot[0, 0, 3]) - float(want)) < 1e-5 and float(shot[0, 0, 2]) < float(shot[0, 0, 3])
    args = (main, active, x, z, own, standing, shooter, no, G, CFG.kinds)
    kind, target, _ = teacher.choose(*args)                                     # without shot: the nearest
    assert int(kind[0, 0]) == K["shoot"] and int(target[0, 0]) == G
    kind, target, _ = teacher.choose(*args, shot=shot)
    assert int(kind[0, 0]) == K["shoot"] and int(target[0, 0]) == G + 1         # the militia
    # the militia out of reach (range 150 + 4 m/s x 10 s): the greatswords
    x_far = x.clone()
    x_far[0, 3] = 400.0
    shot_far = teacher.shot_hp(x_far, z, rng, dmg, ap, small, armour, hp_man, shield, resist, men0, walk=walk)
    assert float(shot_far[0, 0, 3]) == 0 and float(shot_far[0, 0, 2]) > 0
    _, target, _ = teacher.choose(main, active, x_far, z, own, standing, shooter, no, G, CFG.kinds, shot=shot_far)
    assert int(target[0, 0]) == G
    # both alike: the nearest; nothing in reach: the nearest
    same = teacher.shot_hp(x, z, rng, dmg, ap, small, f(0, 0, 25, 25), f(60, 60, 60, 60), shield, resist, men0,
                           walk=walk)
    _, target, _ = teacher.choose(*args, shot=same)
    assert int(target[0, 0]) == G
    _, target, _ = teacher.choose(*args, shot=torch.zeros(1, N, N))
    assert int(target[0, 0]) == G
    # a shield of 0.9 on the militia blocks the slings: the greatswords again
    shielded = teacher.shot_hp(x, z, rng, dmg, ap, small, armour, hp_man, f(0, 0, 0, 0.9), resist, men0, walk=walk)
    _, target, _ = teacher.choose(*args, shot=shielded)
    assert int(target[0, 0]) == G
    assert teacher.REACH_S == 10.0
    # without the walk: the range alone (the militia at 160 m out of 150)
    assert float(teacher.shot_hp(x, z, rng, dmg, ap, small, armour, hp_man, shield, resist, men0)[0, 0, 3]) == 0


def test_the_teachers_shooters_pick_a_target_in_range_or_within_ten_seconds_of_a_walk():
    """teacher reach (the human's battle against g34: range + 100 m kept his shooters the target while they stood out
    of range): the enemy's soft shooters at 200 m (range 150, a walk of 3 m/s: 150 + 30 = 180 m) are not chosen before
    the enemy's armoured infantry at 140 m in range; at 175 m they are; nothing in reach - the group the shooters
    bring into range soonest (approach_s), not the nearest by the centres."""
    N = 4
    f = (lambda *v: torch.tensor([list(v)], dtype=torch.float32))
    main = torch.tensor([[0, 0, G, G + 1]])
    active = torch.zeros(1, 2 * G, dtype=torch.bool)
    active[0, [0, G, G + 1]] = True
    own = torch.tensor([[True, True, False, False]])
    standing = torch.ones(1, N, dtype=torch.bool)
    shooter = torch.tensor([[True, True, False, False]])
    no = torch.zeros(1, N, dtype=torch.bool)
    rng, dmg, ap, walk = f(150, 150, 0, 0), f(7, 7, 0, 0), f(1, 1, 0, 0), f(3, 3, 0, 0)
    small = torch.tensor([[True, True, False, False]])
    # theirs: 2 infantry of armour 95 (e0), 3 shooters of armour 0 (e1)
    armour, hp_man, shield, resist, men0 = f(0, 0, 95, 0), f(60, 60, 80, 60), f(0, 0, 0, 0), f(0, 0, 0, 0), f(80, 80, 60,
                                                                                                         90)

    def pick(xs, zs):
        x, z = f(*xs), f(*zs)
        shot = teacher.shot_hp(x, z, rng, dmg, ap, small, armour, hp_man, shield, resist, men0, walk=walk)
        app = teacher.approach_s(x, z, rng, walk)
        kind, target, _ = teacher.choose(main, active, x, z, own, standing, shooter, no, G, CFG.kinds, shot=shot,
                                         approach=app)
        assert int(kind[0, 0]) == K["shoot"]
        return int(target[0, 0])

    assert pick((0, 0, 140, 0), (0, 0, 0, 200)) == G                  # the shooters at 200 m: the infantry in range
    assert pick((0, 0, 140, 0), (0, 0, 0, 175)) == G + 1              # at 175 m (within 180): the soft shooters
    # nothing in reach: the infantry nearer by the centres (300 against 320 m), but our shooter 1 has the enemy's
    # shooters in range after (220 - 150) / 3 s - sooner than any of ours the infantry
    xs, zs = (0, 0, 300, 0), (0, 200, 100, 420)
    t = teacher.approach_s(f(*xs), f(*zs), rng, walk)
    assert float(t[0, 1, 3]) == pytest.approx(70 / 3, rel=1e-4) and float(t[0, 0, 2]) > 50
    assert pick(xs, zs) == G + 1
    assert torch.isinf(t[0, 2, 0])                                     # no range: never
