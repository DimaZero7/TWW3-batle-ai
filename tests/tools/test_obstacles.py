"""tools.analysis.obstacles on a synthetic map: clusters, gaps, corners, fence lines."""
import numpy as np

from tools.analysis import obstacles


def synthetic():
    n = 40
    clear = np.ones((n, n), dtype=np.int8)
    clear[10:30, 5:8] = 0        # wall A
    clear[10:30, 10:13] = 0      # wall B: 2 free cells (6 m) from A
    clear[5:8, 25:35] = 0        # U-shape: bottom
    clear[5:15, 25:28] = 0       # U-shape: left arm
    clear[5:15, 32:35] = 0       # U-shape: right arm
    return {"clear": clear, "inside": np.ones((n, n), dtype=bool),
            "height": np.zeros((n, n)), "forest": np.zeros((n, n), dtype=np.int8)}


def test_clusters_gap_and_corner():
    summary, gaps, _, _, _, _ = obstacles.analyse(synthetic(), 0.0, 3.0, [], 30.0)
    assert summary["obstacles"]["clusters"] == 3
    assert any(g["width_m"] == 6.0 for g in gaps), gaps
    corner = summary["candidates"]["corner"][0]
    assert corner["center"][0] > 70           # the U-shape (x ~ 90 m), not a straight wall
    assert corner["concavity"] > 0.2
    walls = [c for c in summary["largest_clusters"] if c["center"][0] < 40]
    assert all(c["concavity"] < 0.05 for c in walls)


def test_fence_lines_and_gap():
    fences = [{"name": "glb_fence_01", "x": 5.8 * i, "z": 0.0} for i in range(5)]
    fences += [{"name": "glb_fence_02", "x": 5.8 * 4 + 10.0 + 5.8 * i, "z": 0.0} for i in range(5)]
    info, gaps = obstacles.line_objects(fences)
    assert info["count"] == 10 and len(info["lines"]) == 2
    assert abs(gaps[0]["width_m"] - (10.0 - 5.8)) < 0.2
