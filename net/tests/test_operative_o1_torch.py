"""Drill O0-1 'advance on the holder' (net/operative: scenes_o1, advance, lines, sally, exam_o1) - the area's smoke test:
the twins share the cause scene's draws, the teacher's rule holds on its hand-made states, and a 45 s play of a cause
scene and its role twin under TEACH moves the zone once at 30 s in the cause scene only. Needs torch (the container,
CPU)."""
import pytest

torch = pytest.importorskip("torch")


def test_the_drill_plays_its_cause_and_keeps_in_its_twin():
    from net.operative import advance as AD
    from net.operative import exam_o1 as EX
    from net.operative import scenes_o1 as SO
    AD.table()                                          # (asserts each hand-made state)
    a, come, role = SO.scene(5), SO.scene(5, twin="come"), SO.scene(5, twin="role")
    assert a["places_mine"] == come["places_mine"] == role["places_mine"] and a["mine"] == come["mine"]
    assert [p[:2] for p in a["places_theirs"]] == [tuple(p) for p in come["o1"]["pose"]]
    assert (a["o1"]["att"], role["o1"]["att"], come["o1"]["enemy"]) == (1, 0, "walkin")
    for s in (a, role):
        s["limit"] = 45
    res = EX.play("TEACH", [a, role], [0], 2, "cpu", compile=False)
    assert res["per_battle"]["changes"] == [1.0, 0.0]
    t30 = {r[0]: r for r in res["ticks"] if r[1] == 30}
    assert t30[0][3] == 2 and t30[1][3] == -1           # (L60 in the cause scene, KEEP in the role twin)
