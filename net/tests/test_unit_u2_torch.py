"""Drill U2 by the method (net/unit/u2.py): the execution measure counts the seconds the formation stands in the frame
facing by it, the judge 'met head-on' is the enemy's contact on our front, the order gate counts a new order to the game
only when it changes (net/common/reissue.py), and the smoke - the generator's skirmishes played by each script. Needs
torch (the container, CPU)."""
import pytest

torch = pytest.importorskip("torch")

from sim import orders as O  # noqa: E402
from sim.params import load  # noqa: E402
from net.common import reissue  # noqa: E402
from net.unit import scripts as SC  # noqa: E402
from net.unit import u2  # noqa: E402

P = load()
SPEAR, CLANRAT = "wh_main_emp_inf_spearmen_0", "wh2_main_skv_inf_clanrat_spearmen_0"


def spec(x, z, b, course=0.0, limit=20.0, enemy=False, ez=2000.0):
    """Our spearmen at (x, z) facing b, the frame 64 x 48 m at the origin with the course; the enemy ez ahead (+z),
    holding (enemy False) or walking to the frame."""
    return {"me": SPEAR, "foe": CLANRAT, "ours": "emp", "theirs": "skv", "far": False, "enemy": enemy, "pace": "any",
            "map": "flat", "x": x, "z": z, "b": b, "edge": 0.0, "frame": (0.0, 0.0, course, 64.0, 48.0), "ex": 0.0,
            "ez": ez, "eb": 180.0, "ready_s": 0.0, "lead_s": 0.0, "limit": limit, "gold": (300.0, 325.0)}


def test_exec_counts_the_seconds_in_the_frame_facing_its_course():
    sp = [spec(0.0, 0.0, 0.0), spec(0.0, 0.0, 180.0), spec(0.0, -200.0, 0.0), spec(0.0, 0.0, 90.0, course=90.0)]
    m = u2.play(sp, SC.Stand(), P, "cpu")
    assert m["exec"].tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])
    # the needed move turns the one facing away: in the frame, facing it after the re-form (3.9 s) and the turn
    m = u2.play(sp[1:2], SC.Straight(), P, "cpu")
    assert 0.5 < float(m["exec"][0]) < 0.8
    m = u2.play(sp[1:2], SC.Random(), P, "cpu")            # a new course every second: it never turns
    assert float(m["exec"][0]) < 0.1


def test_met_head_on_is_the_enemy_s_contact_on_our_front():
    sp = [spec(0.0, 0.0, 0.0, limit=90.0, enemy=True, ez=70.0), spec(0.0, 0.0, 180.0, limit=90.0, enemy=True, ez=70.0),
          spec(0.0, 0.0, 0.0, limit=20.0)]
    m = u2.play(sp, SC.Stand(), P, "cpu")
    assert m["contact"][:2].tolist() == [1.0, 1.0] and m["head_on"][:2].tolist() == [1.0, 0.0]
    assert torch.isnan(m["head_on"][2])                                # no enemy: no judge


def test_the_gate_sends_an_order_only_when_it_changes():
    g = reissue.Gate(1, 1, "cpu")
    live = torch.ones((1, 1), dtype=torch.bool)

    def give(kind, x=0.0, z=0.0, face=-1.0, target=-1):
        o = O.hold(1, 1)
        o.kind[:], o.x[:], o.z[:], o.face[:], o.target[:] = kind, x, z, face, target
        return g.apply(o, live)
    sent = []
    for args in ((O.MOVE, 0, 0, 90), (O.MOVE, 0, 0, 90), (O.MOVE, 5, 0, 90), (O.MOVE, 9, 0, 90), (O.MOVE, 9, 0, 100),
                 (O.MOVE, 9, 0, 125), (O.HOLD,), (O.HOLD, 50, 50), (O.ATTACK, 0, 0, -1, 0), (O.ATTACK, 0, 0, -1, 0),
                 (O.KEEP,)):
        before = float(g.sent.sum())
        out = give(*args)
        sent.append(float(g.sent.sum()) > before)
        if not sent[-1] and args[0] != O.KEEP:
            assert int(out.kind) == int(g.last.kind) and float(out.x) == float(g.last.x)     # the last one re-given
    assert sent == [True, False, False, True, False, True, True, False, True, False, False]
    assert float(g.per_minute()) == pytest.approx(60.0 * 5 / 11)


def test_smoke_the_generator_s_skirmishes_played_by_each_script():
    sp = u2.specs(6, torch.Generator().manual_seed(1))
    for s in sp:
        s["map"], s["limit"] = "flat", 30.0
    many = u2.specs(64, torch.Generator().manual_seed(2))
    assert {s["far"] for s in many} == {True, False} and {s["enemy"] for s in many} == {True, False}
    assert all(48.0 <= v <= 80.0 for s in many for v in s["frame"][3:])
    for name in SC.PLAYERS:
        m = u2.play(sp, SC.player(name), P, "cpu")
        assert set(("exec", "head_on", "reached", "orders_min")) <= set(m) and m["exec_all"].shape == (6,)
        assert torch.isfinite(m["exec_all"]).all() and (m["orders_min"] >= 0).all()


def test_charged_counts_a_charge_from_the_frame_on_the_front():
    """The counter's measure: the script under 'counter' stands in its frame, then charges the enemy coming at its
    front - charged 1; 'stand' never charges - 0."""
    from net.unit import skills as SK
    sp = SK.specs("counter", 1, torch.Generator().manual_seed(3))[0]
    sp.update(me=SPEAR, foe=CLANRAT, ours="emp", theirs="skv", cause=True, pace="any", map="flat", b=0.0,
              frame=(0.0, 0.0, 0.0, 64.0, 48.0), A=(0.0, 160.0, 180.0, SK.WALK_E), limit=200.0)
    assert float(SK.play([sp], SC.Unit(), P, "cpu")["charged"][0]) == 1.0
    assert float(SK.play([sp], SC.Stand(), P, "cpu")["charged"][0]) == 0.0


def test_smoke_each_skill_drill_played_by_the_script():
    from net.unit import skills as SK
    for skill in SK.SKILL_DRILLS:
        sp = SK.specs(skill, 4, torch.Generator().manual_seed(1))
        for s in sp:
            s["map"], s["limit"] = "flat", 30.0
        m = SK.play(sp, SC.Unit(), P, "cpu")
        assert m["orders_min"].shape == (4,) and (m["orders_min"] >= 0).all()
