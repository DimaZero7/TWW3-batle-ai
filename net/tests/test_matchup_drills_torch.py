"""The operative's drills out of the user's matchups (net/matchup/drills.py: threat, guns, skirm, still): each start has
the drill's enemy where the drill says, and a short batch under the teacher (v2) gives the drill's measure in its
range, one a battle. Needs torch (the container, CPU)."""
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from net.matchup import drills as MD  # noqa: E402
from sim import maps as arena_map  # noqa: E402
from sim.params import load  # noqa: E402

pytestmark = pytest.mark.skipif(not (arena_map.MAPS / "ksl_plains_01" / "meta.json").exists(),
                                reason="sim/maps/ksl_plains_01 (the captured map) is not here")
P = load()


@pytest.fixture(scope="module")
def unit_net():
    from net.tactic import ckpt as TK
    return TK.unit_network("cpu")


@pytest.mark.parametrize("drill", sorted(MD.DRILLS))
def test_the_start_has_the_drills_enemy(drill):
    d = MD.DRILLS[drill]
    for seed in range(4):
        b = MD.start(drill, random.Random(seed))
        assert sum(u["general"] for u in b[1]) == 1 and sum(u["general"] for u in b[2]) == 1
        roles2 = MD.roles_of([u["key"] for u in b[2]])
        if d["enemy_roles"] is None:
            assert len(b[2]) == 20                                     # the whole army, on its places
            continue
        rest = [u for u, r in zip(b[2], roles2) if not u["general"]]
        assert rest and all(r in d["enemy_roles"] for u, r in zip(b[2], roles2) if not u["general"])
        if d["gap"] is not None:
            roles1 = MD.roles_of([u["key"] for u in b[1]])
            tx = max(u["x"] for u, r in zip(b[1], roles1) if r in ("shooter", "sniper"))
            gaps = np.array([u["x"] - tx for u in rest])
            assert gaps.min() >= d["gap"][0] - 10 and gaps.max() <= d["gap"][1] + 10, (drill, gaps)


@pytest.mark.parametrize("drill", sorted(MD.DRILLS))
def test_a_short_batch_reads_the_measure(drill, unit_net, monkeypatch):
    monkeypatch.setitem(MD.DRILLS[drill], "limit", 20)
    rd, stats = MD.play(drill, 2, "v2", unit_net, P, "cpu", seed=0)
    m = rd["measure"]
    assert m.shape == (2,) and bool(torch.isfinite(m).all()) and bool((m >= 0).all())
    if drill != "still":                                               # (still: plan changes a battle, a count)
        assert bool((m <= 1).all())
    if drill == "threat":                                              # 20 s: the fast ones are still 200 m off
        assert bool((m > 0.9).all())
    if drill == "still":                                               # the windows 45 - 90 s: none in 20 s
        assert float(m.sum()) == 0.0
