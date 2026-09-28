"""apps.missile: damage of archers on infantry, calibrated on the game (archer-range --range-mode damage)."""
import pytest

from tests.lua_runtime import load, new_runtime

ARCHERS, SPEARMEN = "wh2_dlc13_emp_inf_archers_0", "wh_main_emp_inf_spearmen_0"


@pytest.fixture(scope="module")
def mis():
    lua = new_runtime()
    m = load(lua, "apps.missile.services")
    m.t = lambda x: lua.table_from(x, recursive=True)
    return m


def archers(uid, z, side="own"):
    # 20 m wide, 18 m deep, facing +Z; front rank at z.
    return {"id": uid, "side": side, "key": ARCHERS, "x": 0, "z": z, "bearing": 0, "front_m": 20.2, "depth_m": 18,
            "range_m": 130, "men": 90, "men_max": 90, "hp": 6210, "hp_max": 6210, "ammo": 1800, "missile_damage": 19}


def spearmen(uid, z, side="enemy"):
    # 30 m wide, facing -Z (towards the archers); front rank at z.
    return {"id": uid, "side": side, "key": SPEARMEN, "x": 0, "z": z, "bearing": 180, "front_m": 28.4,
            "depth_m": 9.3, "range_m": 0, "men": 120, "men_max": 120, "hp": 8280, "hp_max": 8280, "ammo": 0}


def exchange(mis, units, seconds):
    lua_units = mis.t(units)
    timeline = []
    for t in range(1, seconds + 1):
        mis.step(lua_units, 1)
        timeline.append([dict(hp=u.hp, men=u.men, ammo=u.ammo) for u in lua_units.values()])
    return timeline


def test_hp_per_arrow_follows_the_measurement(mis):
    assert mis.hp_per_arrow(70, 19) == pytest.approx(13.4)
    assert mis.hp_per_arrow(100, 19) == pytest.approx(10.8)
    assert mis.hp_per_arrow(125, 19) == pytest.approx(10.0)
    assert mis.hp_per_arrow(125, 38) == pytest.approx(20.0)   # other shooters: scaled by the card


@pytest.mark.parametrize("mid_m, half_s, dead_s", [(70, (33, 42), (108, 124)), (105, (43, 67), (135, 187)),
                                                    (120, (43, 57), (128, 175))])
def test_one_archer_unit_on_spearmen_as_in_the_game(mis, mid_m, half_s, dead_s):
    # Target's front rank mid_m ahead of the archers' middle (9 m behind their front).
    tl = exchange(mis, [archers("a", 0), spearmen("s", mid_m - 9)], 240)
    half = next(t for t, row in enumerate(tl, 1) if row[1]["hp"] <= 8280 / 2)
    dead = next(t for t, row in enumerate(tl, 1) if row[1]["men"] == 0)
    # Measured spans (three runs) widened by 20% for the model's smoothness.
    assert half_s[0] * 0.8 <= half <= half_s[1] * 1.2, half
    assert dead_s[0] * 0.8 <= dead <= dead_s[1] * 1.2, dead
    assert tl[-1][0]["ammo"] < 1800


def test_out_of_reach_nobody_shoots(mis):
    tl = exchange(mis, [archers("a", 0), spearmen("s", 128 - 9)], 10)
    assert tl[-1][1]["hp"] == 8280 and tl[-1][0]["ammo"] == 1800


def test_both_sides_shoot_the_nearest_they_reach(mis):
    ours = archers("a", 0)
    theirs = dict(archers("b", 110, side="enemy"), bearing=180)   # facing us, front rank 110 m ahead
    near = spearmen("s", 60)
    tl = exchange(mis, [ours, theirs, near, dict(spearmen("w", -1, side="own"), bearing=0, z=12)], 5)
    hp = {u: tl[-1][i]["hp"] for i, u in enumerate(["a", "b", "s", "w"])}
    assert hp["s"] < 8280 and hp["b"] == 6210     # ours shoot the nearer spearmen
    assert hp["w"] < 8280 and hp["a"] == 6210     # theirs shoot our wall in front of us
