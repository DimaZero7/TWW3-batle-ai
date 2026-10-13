"""A move order through the enemy's edge (contact.move_push; the walkpass2 probe, tools/nn/charge_probe.py): the walker is
not held where it touches the enemy, it pushes on to its point at a share of its speed by the share of its front the
enemy covers across its way (geometry.blocked_share). Needs torch (the container)."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.sim import battle, geometry, replay, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402

P = load()
SWORD, CLANRATS = "wh_main_emp_inf_swordsmen", "wh2_main_skv_inf_clanrats_1"


def test_blocked_share_is_the_overlap_of_the_front_across_the_way():
    # i at x 0 walking -z (30 m front); j a 30 x 6 m box facing +z, its centre `dx` across; a second j beside it
    def share(dx, bj=0.0, mask=True):
        u = {"x": torch.tensor([[0.0, dx]]), "z": torch.tensor([[0.0, -20.0]]), "b": torch.tensor([[180.0, bj]])}
        pw = {"front": torch.tensor([[30.0, 30.0]]), "depth": torch.tensor([[5.0, 6.0]])}
        m = torch.tensor([[[False, mask], [False, False]]])
        return float(geometry.blocked_share(u, pw, m, torch.tensor([[0.0, 0.0]]), torch.tensor([[-100.0, 0.0]]))[0, 0])
    assert share(28.0) == pytest.approx(2 / 30, abs=1e-4)
    assert share(-20.0) == pytest.approx(10 / 30, abs=1e-4)
    assert share(0.0) == pytest.approx(1.0) and share(31.0) == 0.0
    assert share(20.0, mask=False) == 0.0
    # turned side-on (its depth across the way): 3 m half extent
    assert share(16.0, bj=90.0) == pytest.approx(2 / 30, abs=1e-4)


def lane(dx):
    """Swordsmen 30 m north of held clanrats (centres dx apart across the way), the clanrats' slot."""
    st = scenario.build([{"attacker": 1, "sides": {
        1: {"faction": "wh_main_emp_empire", "units": [{"key": SWORD, "x": dx, "z": 30.0, "b": 180.0}]},
        2: {"faction": "wh2_main_skv_skaven", "units": [{"key": CLANRATS, "x": 0.0, "z": 0.0, "b": 0.0}]}}}], P)
    return st, st.N // 2


def walk_past(dx, params=P, seconds=60):
    """The swordsmen told to walk to 60 m south of the clanrats, past or through them; -> (their mean speed over the
    steps they were in melee, those steps, their walk speed)."""
    st, _ = lane(dx)
    o = replay.hold(st)
    o.kind[0, 0] = O.MOVE
    o.x[0, 0], o.z[0, 0] = dx, -60.0
    o.run[0, 0] = False
    v, n = 0.0, 0
    for _ in range(int(seconds / params.dt)):
        battle.step(st, o, params)
        if bool(st.u["m"][0, 0]):
            v += float(torch.hypot(st.u["vx"][0, 0], st.u["vz"][0, 0]))
            n += 1
    return v / max(n, 1), n, float(st.u["walk"][0, 0])


def test_a_walker_clipping_the_enemy_edge_pushes_on_fighting_and_one_straight_through_barely_moves():
    cal = P.sim["contact"]["move_push"]
    assert cal["on"] == 1
    st, H = lane(0.0)
    front, _ = geometry.dims(st.u, P.sim["formation"]["spacing_m"])
    edge_dx = float(front[0, 0] / 2 + front[0, H] / 2)                 # side by side, edges touching
    clip, n_clip, walk = walk_past(edge_dx - 4.0)                       # the way overlaps the clanrats by 4 m
    through, n_through, _ = walk_past(0.0)
    assert n_clip > 0 and n_through > 0                                 # both fight (the melee flag)
    assert clip > 0.5 * walk                                            # the game: 0.65-0.89 of its speed
    assert 0 < through < 0.2 * walk                                     # the game: 0.08
    off = P.with_cal("contact", move_push=dict(cal, on=0))
    held, n_held, _ = walk_past(edge_dx - 4.0, params=off)              # the old rule: held where it touches
    assert n_held > n_clip and held < 0.3 * clip
