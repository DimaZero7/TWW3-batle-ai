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


def test_attack_is_melee_with_the_target_group_else_half_its_closeness():
    u = state([(0, 0), (0, -150), (500, 0), (0, 0), (30, 0), (900, 900)], SIDE,
              m=[True, False, False, True, False, False], target=[3, -1, -1, 0, -1, -1])
    m, ok = run(u, MAIN, {0: ("attack", G)})
    assert m[0] == pytest.approx(1.0)
    assert m[1] == pytest.approx(0.5 * (1 - 150 / 300))                     # 150 m to unit 3
    u2 = state([(0, 0), (0, -150), (500, 0), (0, 0), (30, 0), (900, 900)], SIDE,
               m=[True, False, False, True, False, False], target=[5, -1, -1, 0, -1, -1])
    m, _ = run(u2, MAIN, {0: ("attack", G)})
    assert m[0] < 1.0                                                       # fighting another group: not the mission
    assert ok[:2] == [True, True] and ok[2:] == [False] * 4                 # g1 has no mission here; theirs never


def test_flank_counts_only_off_the_targets_front():
    # unit 3 faces north (b 0: facing (sin, cos) = (0, 1)); ours fights it from its front, then from its side
    front = state([(0, 20), (0, 400), (500, 0), (0, 0), (300, 300), (900, 900)], SIDE,
                  m=[True, False, False, True, False, False], target=[3, -1, -1, 0, -1, -1])
    m, _ = run(front, MAIN, {0: ("flank", G)})
    assert m[0] == pytest.approx(0.0)
    side = state([(20, 0), (0, 400), (500, 0), (0, 0), (300, 300), (900, 900)], SIDE,
                 m=[True, False, False, True, False, False], target=[3, -1, -1, 0, -1, -1])
    m, _ = run(side, MAIN, {0: ("flank", G)})
    assert m[0] == pytest.approx(1.0)
    # not in melee, behind the nearest target unit, 150 m: half its closeness
    rear = state([(0, -150), (0, 400), (500, 0), (0, 0), (300, 300), (900, 900)], SIDE)
    m, _ = run(rear, MAIN, {0: ("flank", G)})
    assert m[0] == pytest.approx(0.5 * 0.5)
    assert 45.0 == mission.FLANK_DEG


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
    x = torch.tensor([[0.0, 0, 10, 100, 110, 300, 0, 900]])
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
    assert int(kind[0, 3]) == K["attack"] and int(target[0, 3]) == G
    assert int(kind[0, 0]) == K["attack"]                                # the lord is 200 m from e0: not threatened
    z2 = z.clone()
    z2[0, 6] = -50.0                                                     # e0 within 150 m of the lord now
    kind, target, _ = teacher.choose(main, active, x, z2, own, standing, shooter, lord, G, CFG.kinds)
    assert int(kind[0, 0]) == K["withdraw"] and int(target[0, 0]) in (1, 2)   # to a largest other group (2 units)
    # every pair the teacher gives is one the general may choose (the kind takes the target's type)
    for g in range(4):
        t = int(target[0, g])
        tt = "own" if t < G else "enemy" if t < 2 * G else "tile"
        assert tt in gconfig.MISSION_KINDS[int(kind[0, g])].targets
    assert math.isfinite(teacher.THREAT_M)
