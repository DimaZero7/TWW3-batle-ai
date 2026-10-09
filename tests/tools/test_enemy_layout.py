"""tools.analysis.enemy_layout: group links from soldier positions."""
import numpy as np

import pytest

pytest.importorskip("matplotlib", reason="matplotlib: on the host, not in the training container")

from tools.analysis import enemy_layout as el  # noqa: E402


def block(name, key, x, z):
    """A 10 x 2 m block of soldiers with its left-front corner at (x, z), in decimetres."""
    pts = [(x + i, z - j) for i in range(0, 11, 2) for j in (0, 2)]
    return {"name": name, "key": key, "soldiers_dm": [int(v * 10) for p in pts for v in p]}


def test_spanning_tree_takes_shortest_links():
    dist = [[0, 1, 5], [1, 0, 2], [5, 2, 0]]
    assert sorted(e[2] for e in el.spanning_tree(3, dist)) == [1, 2]


def test_group_link_is_the_longest_needed_gap():
    units = [block("a", "spearmen", 0, 0), block("b", "spearmen", 14, 0), block("c", "archers", 0, -30),
             block("l", "wh_main_emp_cha_general_0", 60, 0)]
    whole, _ = el.group(units, True)
    no_lord, _ = el.group(units, False)
    # a-b 4 m apart edge to edge, c 28 m behind a, the lord 36 m right of b.
    assert no_lord["group_link_gap_m"] == 28.0 and whole["group_link_gap_m"] == 36.0
    assert el.kind("wh_main_emp_cha_general_0") == "general" and el.kind("x_archers_0") == "archers"
    assert np.isclose(el.extent(units[:2], 0)["front_m"], 24.0)
