"""tools/nn/fighters/soldiersdata.py: the recorded soldiers' places and the men-in-reach geometry (numpy)."""
import json

import numpy as np

from tools.nn.fighters import soldiersdata as SD


def test_places_are_read_in_metres_and_found_by_time(tmp_path):
    rows = [{"event": "nn_sample", "t": 0, "units": []},
            {"event": "nn_soldiers", "t": 1000, "units": [{"n": "a", "side": 1, "xz_dm": [10, 20, 30, 40]},
                                                           {"n": "b", "side": 2, "unavailable": "call_failed"}]},
            {"event": "nn_soldiers", "t": 2000, "units": [{"n": "a", "side": 1, "xz_dm": [11, 21]}]}]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pl = SD.load(tmp_path)
    assert len(pl) == 2
    assert np.allclose(pl.at(1.1)["a"], [[1.0, 2.0], [3.0, 4.0]]) and "b" not in pl.at(1.0)
    assert np.allclose(pl.at(2.0)["a"], [[1.1, 2.1]])
    assert pl.at(5.0) is None                       # no record near enough


def _block(cx, cz, files, ranks, h=1.5, v=1.5):
    """A formation facing +z (bearing 0) centred at (cx, cz)."""
    xs = (np.arange(files) - (files - 1) / 2) * h + cx
    zs = (np.arange(ranks) - (ranks - 1) / 2) * v + cz
    return np.array([(x, z) for z in zs for x in xs])


def test_two_formations_face_to_face_reach_across_the_front_only():
    # a facing +z, its front rank at z = +3; b facing -z, its front rank 2 m beyond a's: only front ranks in reach,
    # nobody's nearest enemy inside his own formation
    a = _block(0, 0, 10, 5)
    b = _block(0, 3 + 2 + 3, 10, 5)
    near, inside = SD.geometry({"a": a, "b": b}, {"a": 1, "b": 2}, {"a": (0, 0), "b": (0, 8)}, {"a": 0.0, "b": 180.0})
    assert near > 0 and inside == 0


def test_formations_standing_in_each_other_strike_inside():
    a = _block(0, 0, 10, 5)
    b = _block(0, 1.0, 10, 5) + np.array([0.75, 0.0])     # b stands in a, shifted half a file
    near, inside = SD.geometry({"a": a, "b": b}, {"a": 1, "b": 2}, {"a": (0, 0), "b": (0.75, 1.0)},
                               {"a": 0.0, "b": 180.0})
    assert near == 100 and inside > 30          # all but the edge men (1 m in from the box)
