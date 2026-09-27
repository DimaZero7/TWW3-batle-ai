"""tools.analysis.formation_probe: the stand-still verdict."""
from tools.analysis import formation_probe as fp


def rows(orders=0, shift_dm=0, moving=False):
    men = [0, 0, 10, 0, 0, -10, 10, -10]
    moved = [v + (shift_dm if i % 2 == 0 else 0) for i, v in enumerate(men)]
    unit = lambda xz: {"script_name": "own_1", "soldiers_dm": xz, "motion": {"x": 0.5, "z": -0.5}}
    sample = lambda t, x: {"event": "hold_sample", "t_ms": t, "orders_after_placed": orders,
                           "units": [{"script_name": "own_1", "motion": {"x": x, "z": 0, "ordered_x": 0, "ordered_z": 0,
                                                                         "is_moving": moving}}]}
    return [{"event": "stage_snapshot", "stage": "placed", "units": [unit(men)]},
            sample(0, 0), sample(1000, 2.0),  # the native position may jump without anyone moving
            {"event": "stage_snapshot", "stage": "hold", "units": [unit(moved)]}]


def test_still_army_passes_even_if_the_native_position_jumps():
    s = fp.stability(rows())
    assert s["verdict"] == "PASS" and s["units"]["own_1"]["native_position_jump_m"] == 2.0


def test_orders_movement_or_walking_fail():
    assert fp.stability(rows(orders=1))["verdict"] == "FAIL"
    assert fp.stability(rows(shift_dm=20))["verdict"] == "FAIL"
    assert fp.stability(rows(moving=True))["verdict"] == "FAIL"
