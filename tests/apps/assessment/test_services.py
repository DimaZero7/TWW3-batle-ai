"""apps.assessment.services: unit categories and force features (level 1: functions)."""
import pytest

from tests.lua_runtime import load, new_runtime

LORD = {"id": "l", "men": 1, "commanding": True, "class": "com", "range_m": 0}
SPEARS = {"id": "s", "men": 120, "class": "inf_mel", "range_m": 0, "health": 8280, "armour": 30,
          "melee_attack": 20, "melee_defence": 34}
ARCHERS = {"id": "a", "men": 90, "class": "inf_mis", "range_m": 130, "fire": "arc", "health": 6210,
           "armour": 20, "melee_attack": 14, "melee_defence": 17, "missile_damage": 19}
GUNS = dict(ARCHERS, id="g", fire="direct")
HORSE = {"id": "h", "men": 60, "class": "cav_mel", "range_m": 0}


@pytest.fixture
def a():
    lua = new_runtime()
    m = load(lua, "apps.assessment.services")
    m.t = lambda x: lua.table_from(x, recursive=True)
    return m


def test_categories(a):
    assert [a.category(a.t(u)) for u in (LORD, SPEARS, ARCHERS, GUNS, HORSE)] == \
        ["lord", "infantry", "shooter", "shooter", "other"]


def test_powers(a):
    assert a.melee_power(a.t(SPEARS)) == pytest.approx(8280 * 54 / 100)
    assert a.ranged_power(a.t(ARCHERS)) == pytest.approx(90 * 19)
    # Unknown card values: men instead of health, 1 instead of damage.
    assert a.ranged_power(a.t({"men": 50})) == 50


def test_side_counts_arc_share_and_armour(a):
    s = a.side(a.t([LORD, SPEARS, ARCHERS, GUNS]))
    assert dict(s.count) == {"lord": 1, "shooter": 2, "infantry": 1, "other": 0}
    assert s.arc_share == pytest.approx(0.5) and s.infantry_armour == pytest.approx(30)
    assert a.side(a.t([LORD])).infantry_armour is None


def test_assess_ratios(a):
    # Copies: lupa converts a repeated Python object only once.
    f = a.assess(a.t({"role": "attack", "own": [dict(SPEARS), dict(ARCHERS)], "enemy": [dict(SPEARS), dict(SPEARS)]}))
    assert f.role == "attack" and f.melee_ratio == pytest.approx(0.5) and f.ranged_ratio is None
