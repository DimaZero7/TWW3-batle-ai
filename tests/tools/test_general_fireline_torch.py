"""tools/nn/general/fireline.py and mission.py's blocker / clear screen (run.py --general-exec-blocker,
--general-screen-clear, --general-exec-shoot-strict): who stands in our shooters' lines of fire, on hand-made states. torch: the training
container, on the CPU."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import fireline, mission                 # noqa: E402

CFG = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2)
G, C = CFG.places, CFG.target_grid ** 2
K = {k.name: k.index for k in gconfig.MISSION_KINDS}


def state(pos, side, **kw):
    """One battle: units at pos [(x, z)] facing +x (b 90), 100 men 30 m wide (front 30 m, depth 7.5 m), sides; kw:
    per-unit fields."""
    n = len(pos)
    f = (lambda v, dt=torch.float32: torch.full((1, n), v, dtype=dt))
    u = {"x": torch.tensor([[p[0] for p in pos]], dtype=torch.float32),
         "z": torch.tensor([[p[1] for p in pos]], dtype=torch.float32),
         "side": torch.tensor([side]), "b": f(90.0), "men": f(100.0), "men0": f(100.0), "width": f(30.0),
         "sp_h": f(1.5), "sp_v": f(1.5), "radius": f(0.4),
         "gone": f(False, torch.bool), "r": f(False, torch.bool), "m": f(False, torch.bool),
         "direct": f(False, torch.bool), "fire": f(False, torch.bool), "target": f(-1, torch.long),
         "aim_tgt": f(-1, torch.long), "range": f(0.0), "a": f(0.0), "fc_skip": f(0.0),
         "vis": f(True, torch.bool)}
    for k, v in kw.items():
        u[k] = torch.tensor([v], dtype=u[k].dtype)
    return u


B0, S1 = torch.tensor([0]), torch.tensor([1])


def test_a_friend_ahead_in_the_band_closes_the_line_not_beside_behind_or_beyond():
    # shooter 0 at (0, 0) aiming at the enemy 5 at (100, 0); friends: 1 ahead in the line, 2 beside (40 m off: its
    # half front 15 + the shooter's 15 = 30), 3 behind, 4 beyond the target
    u = state([(0, 0), (50, 0), (50, 40), (-20, 0), (150, 0), (100, 0)], [1, 1, 1, 1, 1, 2],
              direct=[True] + [False] * 5, aim_tgt=[5, -1, -1, -1, -1, -1])
    sh = torch.tensor([[True] + [False] * 5])
    L = fireline.lines(u, B0, S1, sh, u["aim_tgt"])
    assert L[0, 0].tolist() == [False, True, False, False, False, False]
    # turned side-on (b 0: its depth across the line) the friend at 25 m off is clear, front-on (b 90) it is not
    u2 = state([(0, 0), (50, 25), (100, 0)], [1, 1, 2], aim_tgt=[2, -1, -1], b=[90.0, 0.0, 90.0])
    assert not bool(fireline.lines(u2, B0, S1, torch.tensor([[True, False, False]]), u2["aim_tgt"])[0, 0, 1])
    u3 = state([(0, 0), (50, 25), (100, 0)], [1, 1, 2], aim_tgt=[2, -1, -1])
    assert bool(fireline.lines(u3, B0, S1, torch.tensor([[True, False, False]]), u3["aim_tgt"])[0, 0, 1])


def test_blocking_needs_a_direct_shooter_that_lost_fire():
    pos, side = [(0, 0), (50, 0), (100, 0)], [1, 1, 2]
    lost = state(pos, side, direct=[True, False, False], aim_tgt=[2, -1, -1], fc_skip=[0.3, 0.0, 0.0])
    assert fireline.blocking(lost, B0, S1)[0].tolist() == [False, True, False]
    clear = state(pos, side, direct=[True, False, False], aim_tgt=[2, -1, -1], fc_skip=[0.0, 0.0, 0.0])
    assert not bool(fireline.blocking(clear, B0, S1).any())
    lob = state(pos, side, direct=[False, False, False], aim_tgt=[2, -1, -1], fc_skip=[0.3, 0.0, 0.0])
    assert not bool(fireline.blocking(lob, B0, S1).any())             # arcing fire: not a direct line


def _measure(u, main, missions, tiles=None, **kw):
    n = len(main)
    kind = torch.zeros(1, G, dtype=torch.long)
    target = torch.full((1, G), 2 * G, dtype=torch.long)
    tgt = torch.full((1, G, 2), float("nan"))
    for g, (k, t) in missions.items():
        kind[0, g], target[0, g] = K[k], t
    for g, p in (tiles or {}).items():
        tgt[0, g] = torch.tensor(p)
    m, ok, _ = mission.measure(u, B0, S1, torch.tensor([main]), kind, target, tgt, torch.ones(1, n, dtype=torch.bool),
                               G, C, 200.0, CFG, **kw)
    return m[0].tolist()


def test_the_blocker_lowers_any_measure_out_of_melee_by_its_time_in_a_lost_line():
    # two hold units on their tile (measure 1), the second in melee; both stood half the decision in a lost line
    u = state([(0, 0), (0, 0), (900, 900)], [1, 1, 2], m=[False, True, False])
    main = [0, 0, G]
    blk = torch.tensor([[0.5, 0.5, 0.0]])
    m0 = _measure(u, main, {0: ("hold", 2 * G)}, tiles={0: (0.0, 0.0)})
    m = _measure(u, main, {0: ("hold", 2 * G)}, tiles={0: (0.0, 0.0)}, blocking=blk, blocker=1.0)
    assert m0[0] == pytest.approx(1.0) and m[0] == pytest.approx(0.5)
    assert m[1] == pytest.approx(m0[1])                                  # in melee: untouched
    m = _measure(u, main, {0: ("hold", 2 * G)}, tiles={0: (0.0, 0.0)}, blocking=blk, blocker=0.0)
    assert m == pytest.approx(m0)                                        # B 0: as before


def test_a_clear_screen_stands_off_our_shooters_lines():
    # ours: the screen 0 (group 0) of group 1 - the shooter 1 at (0, 0), range 250; the enemy 2 at (200, 0)
    def at(p):
        return state([p, (0, 0), (200, 0)], [1, 1, 2], direct=[False, True, False], range=[0.0, 250.0, 0.0],
                     a=[0.0, 10.0, 0.0])
    main, ms = [0, 1, G], {0: ("screen", 1)}
    on_line, beside = at((100, 0)), at((100, 40))
    assert _measure(on_line, main, ms)[0] == pytest.approx(1.0)                         # as before: on the segment
    assert _measure(on_line, main, ms, screen_clear=True)[0] == 0.0                     # in the shooter's line
    assert _measure(beside, main, ms, screen_clear=True)[0] == pytest.approx(1.0)       # beside it: clear
    far = state([(100, 0), (0, 0), (400, 0)], [1, 1, 2], direct=[False, True, False], range=[0.0, 250.0, 0.0],
                a=[0.0, 10.0, 0.0])
    # the enemy out of range and no aim: no line to keep clear
    assert _measure(far, main, ms, screen_clear=True)[0] == pytest.approx(_measure(far, main, ms)[0])


def test_strict_shoot_pays_for_the_missions_target_in_range_and_seen_only():
    # our shooter 0 (group 0, range 150) under "shoot" at the enemy group G (unit 1 at 100 m); unit 2 another group
    def at(x1, target, vis=True):
        return state([(0, 0), (x1, 0), (50, 0)], [1, 2, 2], fire=[True, False, False], target=[target, -1, -1],
                     range=[150.0, 0.0, 0.0], a=[10.0, 0.0, 0.0], vis=[True, vis, True])
    main, ms = [0, G, G + 1], {0: ("shoot", G)}
    other = at(100, 2)
    assert _measure(other, main, ms)[0] == pytest.approx(0.5)                        # as before: in range of it
    assert _measure(other, main, ms, shoot_strict=1.0)[0] == pytest.approx(0.0)      # firing at another group
    assert _measure(other, main, ms, shoot_strict=0.5)[0] == pytest.approx(0.25)
    assert _measure(at(100, 1), main, ms, shoot_strict=1.0)[0] == pytest.approx(1.0)  # at the target
    assert _measure(at(100, 2, vis=False), main, ms, shoot_strict=1.0)[0] == pytest.approx(0.5)   # unseen: as before
    assert _measure(at(300, 2), main, ms, shoot_strict=1.0)[0] == pytest.approx(0.0)  # out of range: as before (0)
