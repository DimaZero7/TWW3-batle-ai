"""apps.approach: stop line, 50 m steps, one manoeuvre at a time, alignment first."""
import pytest

from tests.lua_runtime import load, new_runtime


@pytest.fixture(scope="module")
def ap():
    lua = new_runtime()
    m = load(lua, "apps.approach.services")
    m.t = lambda x: lua.table_from(x, recursive=True)
    return m


def test_reach_past_front(ap):
    # Our archers 20 m behind our front (front at along 5), range 130: reach 110 past it.
    ours = ap.reach_past_front(ap.t([{"along": -15, "range_m": 130}, {"along": -40, "range_m": 130}]), 5, 1)
    # Their archers 25 m behind their front (front at 300, they face -along): 105.
    theirs = ap.reach_past_front(ap.t([{"along": 325, "range_m": 130}]), 300, -1)
    assert ours == pytest.approx(110) and theirs == pytest.approx(105)
    assert ap.reach_past_front(ap.t([]), 0, 1) == 0


def test_stop_gap_is_20_m_short_of_the_first_reach(ap):
    assert ap.stop_gap(82, 95) == pytest.approx(115)


def test_steps_of_50_m_then_the_short_last_one(ap):
    first = ap.next_step(290, 115)
    last = ap.next_step(140, 115)
    done = ap.next_step(116, 115)
    assert first.action == "step" and first.advance_m == 50
    assert last.action == "step" and last.advance_m == pytest.approx(25)
    assert done.action == "arrived"


def test_one_manoeuvre_at_a_time(ap):
    c = ap.new_commander()
    step = ap.t({"action": "step", "advance_m": 50})
    ok = ap.t({"ok": True})
    assert c.decide(ap.t({"step": dict(step), "path": dict(ok)}))[0] == "approach"
    c.start("approach", 0, None)
    assert c.decide(ap.t({"step": dict(step), "path": dict(ok)})) == ("wait", "manoeuvre_under_way")
    with pytest.raises(Exception, match="already under way"):
        c.start("align", 1000, None)
    c.finish(30000, "stopped")
    assert c.decide(ap.t({"step": dict(step), "path": dict(ok)}))[0] == "approach"
    assert c.log[1].kind == "approach" and c.log[1].reason == "stopped"


def test_alignment_comes_first(ap):
    c = ap.new_commander()
    step = {"action": "step", "advance_m": 50}
    assert c.decide(ap.t({"align": {"needed": True}, "governor": "align", "step": step, "path": {"ok": True}}))[0] == "align"
    assert c.decide(ap.t({"align": {"needed": True}, "governor": "cooldown", "step": step,
                          "path": {"ok": True}})) == ("wait", "alignment_cooldown")
    # Given up on aligning: the approach may go on.
    assert c.decide(ap.t({"align": {"needed": True}, "governor": "given_up", "step": step,
                          "path": {"ok": True}}))[0] == "approach"


def test_hold_at_the_stop_line_and_blocked_by_the_mask(ap):
    c = ap.new_commander()
    assert c.decide(ap.t({"step": {"action": "arrived"}})) == ("hold", "at_stop_line")
    assert c.decide(ap.t({"step": {"action": "step", "advance_m": 50},
                          "path": {"ok": False, "reason": "lane_blocked"}})) == ("blocked", "lane_blocked")


def test_choose_advance_goes_past_the_obstacle(ap):
    lua_fits = lambda rock_from, rock_to: (lambda a: not (rock_from <= a <= rock_to))
    # The rock covers advances 40..95 m: 50 does not fit, the first place after it is 96+.
    r = ap.choose_advance(50, 170, lua_fits(40, 95), lambda a: False)
    assert r.ok and 95 < r.advance_m <= 98 and r.detour and r.searched_m > 0
    # Fits at once but the straight lane is blocked: the engine walks around it.
    r2 = ap.choose_advance(50, 170, lambda a: True, lambda a: False)
    assert r2.ok and r2.advance_m == 50 and r2.detour and r2.searched_m == 0
    # Nothing fits before the stop line and the search may not go further.
    r3 = ap.choose_advance(50, 80, lua_fits(40, 95), lambda a: True)
    assert not r3.ok and r3.reason == "no_place"
    # Without the enemy army the place past the obstacle may lie beyond the stop line (flagged).
    r4 = ap.choose_advance(50, 80, lua_fits(40, 95), lambda a: False, 3, 150)
    assert r4.ok and r4.advance_m > 95 and r4.beyond_stop_line
