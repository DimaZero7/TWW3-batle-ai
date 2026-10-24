"""tools/nn/train/exam_sim.py --general-set (the checked network's general rules overridden: set_rules) and its own lord
per battle (evaluate.play's lord_dead, lord_dead_at, lord_melee_s; exam_sim.lord_means). torch: the container."""
import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import config as gconfig, executor   # noqa: E402
from tools.nn.model import config, policy                  # noqa: E402
from tools.nn.train import evaluate, exam_sim              # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)


def test_general_set_overrides_the_rules_and_says_which():
    actor = executor.wrap(policy.Actor(TINY).eval(), gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2))
    assert actor.gcfg.melee_stay and actor.gcfg.replan_near_m == 250.0
    over = exam_sim.general_set(actor, ["melee_stay=0", "replan-near-m=0", "waver_share=0.7"])
    assert over == {"melee_stay": False, "replan_near_m": 0.0, "waver_share": 0.7}
    assert not actor.gcfg.melee_stay and actor.gcfg.replan_near_m == 0.0 and actor.heads.waver_share == 0.7
    assert "melee_stay" not in actor.heads.parts                       # set_rules: the masks follow
    assert exam_sim.general_set(actor, []) == {}
    with pytest.raises(ValueError):
        exam_sim.general_set(actor, ["no_such=1"])
    with pytest.raises(ValueError):
        exam_sim.general_set(policy.Actor(TINY).eval(), ["melee_stay=0"])     # no general


def test_every_battle_carries_its_own_lord():
    actor = policy.Actor(config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1,
                                       critic_heads=2)).eval()
    res = evaluate.play(actor, opponents=("nearest",), generated=4, device="cpu", limit_s=60.0, measure=(),
                        compile=False, cuda_graph=False)
    b = res["battles"]["nearest"]
    n = len(b["won"])
    assert len(b["lord_dead"]) == len(b["lord_dead_at"]) == len(b["lord_melee_s"]) == n
    for dead, at, melee in zip(b["lord_dead"], b["lord_dead_at"], b["lord_melee_s"]):
        assert (at is not None) == dead and 0.0 <= melee <= 60.0 + 1.0
        assert at is None or 0.0 <= at <= 61.0
    m = exam_sim.lord_means(b)
    assert set(m) == {"dead", "dead_at_s", "melee_s"} and 0.0 <= m["dead"] <= 1.0
    assert exam_sim.lord_means({"lord_dead": [True, False], "lord_dead_at": [30.0, None], "lord_melee_s": [10.0, 20.0]}) \
        == {"dead": 0.5, "dead_at_s": 30.0, "melee_s": 15.0}
