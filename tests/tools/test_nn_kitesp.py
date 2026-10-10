"""tools/nn/train/kitesp.py: kite battles (the shooters outrun every infantry unit of the other side by the speed
rule, equal gold, normal deployment), the training source's draw (only the self / past rows, the share, `want`),
and the kiting meter on simulator battles."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.armies import generate
from tools.nn.sim import battle, scenario
from tools.nn.sim.params import load
from tools.nn.train import drills as D
from tools.nn.train import kitesp as K
from tools.nn.train import league, reward, scenes
from tools.nn.train.drills import kiting

UNITS = json.loads((Path(__file__).resolve().parents[2] / "config/nn/units.json").read_text(encoding="utf-8"))["units"]
SEEDS = range(48)


def _run(key):
    return UNITS[key]["speed"]["run"]


class TestArmies:
    def test_classes_keep_the_speed_rule(self):
        classes = K.pairs(generate.default().pools)
        assert classes
        for fa, fb, thr, shooters, infantry, extra in classes:
            assert all(u.missile and _run(u.key) >= thr for u in shooters)
            for u in infantry:
                sp = UNITS[u.key]["speed"]
                assert not u.missile and thr >= (1 + K.SPEED_MARGIN) * sp["run"] - 1e-9 and thr > sp["charge"]
            assert all(u.missile and _run(u.key) < thr for u in extra)
        # Night Runners against Empire infantry; no Empire missile unit outruns anything by the margin
        assert any(fa == "wh2_main_skv_skaven" and fb == "wh_main_emp_empire" for fa, fb, *_ in classes)
        assert not any(fa == "wh_main_emp_empire" for fa, *_ in classes)

    def test_a_battle_is_shooters_against_slower_infantry_with_even_gold(self):
        for (d, name) in K.armies(SEEDS):
            ks = d["kite_side"]
            mine, theirs = d["sides"][ks]["units"], d["sides"][3 - ks]["units"]
            assert sum(u["general"] for u in mine) == 1 and sum(u["general"] for u in theirs) == 1
            shooters = [u for u in mine if not u["general"]]
            assert shooters and all(UNITS[u["key"]]["missile"] for u in shooters)
            slowest = min(_run(u["key"]) for u in shooters)
            body = [u for u in theirs if not u["general"]]
            melee = [u for u in body if not UNITS[u["key"]]["missile"]]
            assert melee
            for u in melee:
                assert K.outruns(slowest, u["key"]), (name, u["key"])
            missile = [u for u in body if UNITS[u["key"]]["missile"]]
            assert all(_run(u["key"]) < slowest for u in missile)
            assert len(missile) <= max(1, int(K.EXTRA_MISSILE * (len(body) + 1)))
            assert len(mine) <= 20 and len(theirs) <= 20
            g = D.gold(d)
            assert abs(g[1] - g[2]) <= 0.06 * max(g.values()), (name, g)
            assert d["attacker"] in (1, 2)
        sides = [d["kite_side"] for d, _ in K.armies(SEEDS)]
        assert 1 in sides and 2 in sides

    def test_same_seed_same_battle_and_not_the_normal_one(self):
        a, b = K.armies([5])[0][0], K.armies([5])[0][0]
        assert a["kite"] == b["kite"]


class _Inner:
    """A source picking bank row 0 for every battle (the kite draw is what is tested)."""

    def __init__(self, bank):
        self.bank = bank

    def pick(self, want=None):
        return torch.zeros(len(want), dtype=torch.long)


class TestSource:
    def test_only_the_rows_take_kite_battles_at_the_share(self):
        params = load()
        gen = scenes.Generated(range(4), 19, params, "cpu", seed=0)
        kite = K.armies(range(6))
        B = 4000
        rows = np.arange(B) % 2 == 0
        src = K.Source(_Inner(gen.bank), kite, 0.25, rows, params, "cpu", seed=1)
        assert src.bank.M == gen.bank.M + 6 and src.bank.N == gen.bank.N
        assert src.kite_side[:gen.bank.M].eq(0).all()
        assert [int(x) for x in src.kite_side[gen.bank.M:]] == [d["kite_side"] for d, _ in kite]
        want = torch.zeros(B, dtype=torch.long)
        want[1::4] = 1                     # an attack-only row: never a kite battle (its side must attack)
        idx = src.pick(want)
        is_kite = idx >= gen.bank.M
        assert not is_kite[torch.as_tensor(~rows)].any()
        share = float(is_kite[torch.as_tensor(rows)].float().mean())
        assert 0.21 < share < 0.29
        # the joined bank: a kite row is the kite battle's start
        j = gen.bank.M + 3
        ref = scenario.build([kite[3][0]], params, per_side=gen.bank.N // 2)
        assert torch.equal(src.bank.state.u["x"][j], ref.u["x"][0])
        assert src.bank.setup.factions[j] == (kite[3][0]["sides"][1]["faction"], kite[3][0]["sides"][2]["faction"])

    def test_rows_of_the_layout(self):
        lay = league.layout(100, 6, {"self": 0.2, "past": 0.3, "ai_like": 0.5})
        r = K.rows_of(lay)
        assert r.sum() == 50 and set(lay.opponent[r]) == {league.CODE["self"], league.CODE["past"]}


class TestMeter:
    def _play(self, script, enemy, n=4):
        descs = [d for d, _ in K.armies(range(100, 100 + n))]
        st = scenario.build(descs)
        ks = torch.tensor([d["kite_side"] for d in descs])
        params = load()
        m = K.Meter(st.B, st.N, "cpu")
        m.restarted(st, torch.ones(st.B, dtype=torch.bool))
        mine = st.u["side"] == ks[:, None]
        steps = 0
        while not bool(st.done.all()) and steps < 2400:
            o = D.merged(st, ks, script(st), enemy(st))
            battle.step(st, o, params, params.dt)
            st.u["lost_worst"] = reward.track(st.u)
            steps += 1
            if steps % 2 == 0:              # a decision every second
                m.update(st, mine, st.done.clone() if bool(st.done.all()) else torch.zeros(st.B, dtype=torch.bool))
        m.battles(st, ks, torch.ones(st.B, dtype=torch.bool))
        return m.stats()

    def test_kiter_kites_and_naive_does_not(self):
        k = self._play(K.kiter, kiting.chase)
        n = self._play(K.naive, kiting.chase)
        assert k["episodes"] > 0 and n["episodes"] > 0
        assert k["kited"] > 0.3 and k["kited"] > n["kited"] + 0.2
        assert n["ran"] < 0.2
        assert k["battles"] == 4 and -1.0 <= k["trade"] <= 1.0
