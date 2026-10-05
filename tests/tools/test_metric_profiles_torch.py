"""Metric profiles with torch: the evaluation computes only the chosen blocks and the same mandatory
numbers (evaluate.play measure), the fatigue tracker (behaviour.Fatigue), test5's evaluation wiring
(spies) and the gap card's simulator replay of a recorded start (tools/nn/train/gapsim.py)."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, policy  # noqa: E402
from tools.nn.sim import scenario  # noqa: E402
from tools.nn.train import behaviour, evaluate, gapsim, rollout, test5  # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)


def actor():
    torch.manual_seed(0)
    return policy.Actor(CFG).eval()


class TestFatigueTracker:
    def test_standing_unit_seconds_by_state(self):
        st = scenario.build([scenario.from_arena("arena")], rollout.params_with_limit(60.0))
        params = rollout.params_with_limit(60.0)
        mine = st.u["side"] == 1
        fat = behaviour.Fatigue(st, params, mine)
        st.u["fat"] = torch.where(mine, torch.full_like(st.u["fat"], 5.0), st.u["fat"])
        live = torch.ones(st.B, dtype=torch.bool)
        fat.update(st, live)
        fat.update(st, ~live)                                  # an ended battle adds nothing
        s = fat.summary(np.ones(st.B, dtype=bool))
        assert s["fatigue_exhausted_share"] == 1.0 and s["fatigue_tired_share"] == 1.0
        assert s["fatigue_shares"][5] == 1.0
        n = int((mine & behaviour.standing_mask(st.u)).sum())
        assert float(fat.sums["fat_s"].sum()) == pytest.approx(n * params.dt)

    def test_the_run_share_far_from_the_fight(self):
        params = rollout.params_with_limit(60.0)
        st = scenario.build([scenario.from_arena("arena")], params)
        u = st.u
        mine = u["side"] == 1
        u["x"] = torch.where(mine, u["x"] - 1000.0, u["x"])                 # far apart
        own = mine & behaviour.standing_mask(u)
        first = own & (torch.cumsum(own.long(), 1) == 1)                   # one own unit runs, the rest walk
        u["order_kind"] = torch.where(own, torch.full_like(u["order_kind"], 1), u["order_kind"])
        u["order_run"] = first.to(u["order_run"].dtype)
        fat = behaviour.Fatigue(st, params, mine)
        live = torch.ones(st.B, dtype=torch.bool)
        fat.update(st, live)
        n = int(own.sum())
        assert fat.summary(np.ones(1, dtype=bool))["run_far_share"] == pytest.approx(1 / n)
        u["m"] = u["m"] | first                                            # first contact: nothing counts after
        fat.update(st, live)
        u["m"] = u["m"] & ~first
        fat.update(st, live)
        assert float(fat.sums["far_move_s"].sum()) == pytest.approx(2 * n * params.dt)


class TestEvaluationProfiles:
    def test_mandatory_skips_the_blocks_and_keeps_the_numbers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluate, "BASELINES", tmp_path)
        kw = dict(opponents=("nearest",), generated=4, limit_s=6.0, together=True)
        full = evaluate.play(actor(), **kw)
        bare = evaluate.play(actor(), measure=set(), **kw)
        a, b = full["by_opponent"]["nearest"], bare["by_opponent"]["nearest"]
        assert "missile_melee_share" in a["behaviour"] and "order_changes_per_min" in a["behaviour"]
        assert "fatigue_exhausted_share" in a["behaviour"]
        assert b["behaviour"] == {} and bare["transfer"] == {}
        assert a["win_rate"] == b["win_rate"] and a["roles"]["attack"]["lord_dead_own"] == b["roles"]["attack"]["lord_dead_own"]
        assert full["battles"]["nearest"]["won"] == bare["battles"]["nearest"]["won"]
        some = evaluate.play(actor(), measure={"behaviour"}, **kw)["by_opponent"]["nearest"]["behaviour"]
        assert "missile_melee_share" in some and "order_changes_per_min" not in some and "fatigue_shares" not in some

    def test_test5_evaluates_the_chosen_profiles(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(evaluate, "play", lambda *a, **k: seen.setdefault("measure", k["measure"]) and {} or {})
        monkeypatch.setattr(evaluate, "play_drills", lambda *a, **k: seen.setdefault("drills", True))
        args = SimpleNamespace(eval=8, drill_eval=16, profiles=("fatigue",))
        res = test5.evaluation(None, args, "cpu", SimpleNamespace(meta=lambda: {}))
        assert seen["measure"] == {"fatigue"} and "drills" not in seen and res["profile"] == "mandatory+fatigue"
        seen.clear()
        res = test5.evaluation(None, SimpleNamespace(eval=8, drill_eval=16), "cpu", SimpleNamespace(meta=lambda: {}))
        assert seen["drills"] and res["profile"] == "full"
        assert test5.metrics({"by_opponent": {}, "profile": "mandatory"})["profile"] == "mandatory"


class TestGapSim:
    def test_a_recorded_start_plays_copies_in_the_recording_s_unit_order(self, monkeypatch):
        army = scenario.from_arena("arena")
        for s in (1, 2):
            for k, u in enumerate(army["sides"][s]["units"]):
                u["name"] = f"{s}_{k}"
        listed = [(s, u) for s in (1, 2) for u in army["sides"][s]["units"]]
        g = SimpleNamespace(names=tuple(u["name"] for _, u in listed), keys=tuple(u["key"] for _, u in listed),
                            side=np.array([s for s, _ in listed]), own_ai="attack", arena="arena")
        monkeypatch.setattr(gapsim.checkpoint, "load_policy", lambda path, device: actor())
        out = gapsim.play([(g, army)], "x.pt", copies=2, limit_s=4.0, log=lambda s: None)
        assert len(out) == 1 and len(out[0]) == 2
        b = out[0][0]["battle"]
        assert b.names == g.names and b.f["x"].shape[1] == len(g.names) and b.t[-1] >= 3.0
        assert np.allclose(b.f["x"][0], [u["x"] for _, u in listed], atol=3.0)      # the start, 2 m jitter
        assert out[0][0]["winner"] in (0, 1, 2) and isinstance(out[0][0]["abilities"], list)
        o = out[0][0]["orders"]
        assert o["move"].shape == b.f["x"].shape and o["run"].dtype == bool
