"""Past versions as OpenAI Five's (docs/en/training/training.md "Past versions"): the pool's quality scores
(league.Pool), a battle keeping its past version from its start (rollout.Battles' slots), Adam's state in the
checkpoint (run.resume), the entropy decay over a chain (run.decay_share)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, critic, policy  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.train import checkpoint, league, randomise, rollout  # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
MIRROR = [("arena", "attack")]


def nets(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(CFG).eval(), critic.Critic(CFG).eval()


def files(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"v{i}.pt"
        p.write_bytes(b"")
        out.append(p)
    return out


class TestPool:
    def test_a_version_is_drawn_with_probability_proportional_to_exp_q(self, tmp_path):
        pool = league.Pool(tmp_path / "pool", size=8, seed=1)
        for p in files(tmp_path, 3):
            pool.add(p)
        for e, q in zip(pool.entries, (0.0, -1.0, 1.0)):
            e["q"] = q
        want = np.exp([0.0, -1.0, 1.0])
        want /= want.sum()
        assert np.allclose(pool.probs(), want)
        n = 20000
        names = [pool.sample()[0].name for _ in range(n)]
        got = np.array([names.count(f"v{i}.pt") / n for i in range(3)])
        assert np.abs(got - want).max() < 0.015
        path, untrained, p = pool.sample()
        assert not untrained and p == pytest.approx(want[int(path.stem[1:])])

    def test_a_new_version_comes_in_with_the_highest_q_and_one_already_in_keeps_its_own(self, tmp_path):
        pool = league.Pool(tmp_path / "pool", size=8)
        a, b, c = files(tmp_path, 3)
        assert pool.add(a)["q"] == 0.0                     # an empty pool: 0
        pool.add(b)
        pool.entries[0]["q"], pool.entries[1]["q"] = -2.0, -0.5
        assert pool.add(c)["q"] == -0.5
        pool.entries[0]["q"] = -3.0
        assert pool.add(a)["q"] == -3.0 and len(pool) == 3

    def test_a_win_lowers_q_by_eta_over_n_p_and_a_loss_changes_nothing(self, tmp_path):
        pool = league.Pool(tmp_path / "pool", size=8, eta=0.01)
        vs = files(tmp_path, 4)
        for p in vs:
            pool.add(p)
        pool.result(vs[1], games=10, wins=7, p=0.2)
        assert pool.find(vs[1])["q"] == pytest.approx(-0.01 * 7 / (4 * 0.2))
        assert pool.find(vs[1])["games"] == 10 and pool.find(vs[1])["wins"] == 7
        pool.result(vs[2], games=10, wins=0, p=0.2)
        assert pool.find(vs[2])["q"] == 0.0
        pool.result(tmp_path / "gone.pt", games=5, wins=5, p=0.5)        # not in the pool: left alone
        assert [e["q"] for e in pool.entries if e["path"] != pool.key(vs[1])] == [0.0, 0.0, 0.0]

    def test_over_the_size_the_lowest_q_leaves_and_the_untrained_has_no_lasting_place(self, tmp_path):
        pool = league.Pool(tmp_path / "pool", size=3)
        r, a, b, c = files(tmp_path, 4)
        pool.add(r, untrained=True)
        pool.add(a)
        pool.add(b)
        pool.find(r)["q"] = -1.0
        pool.find(a)["q"] = -0.5
        pool.add(c)                                       # comes in with max q (0, b's), the untrained leaves
        assert [e["path"] for e in pool.entries] == [pool.key(p) for p in (a, b, c)]

    def test_the_pool_carries_on_from_its_file_and_stores_project_paths_relative(self, tmp_path, monkeypatch):
        monkeypatch.setattr(league.project, "ROOT", tmp_path)
        root = tmp_path / "pool"
        pool = league.Pool(root, size=8)
        vs = files(tmp_path, 2)
        pool.add(vs[0], untrained=True)
        pool.add(vs[1])
        pool.result(vs[1], 4, 4, 0.5)
        pool.tick()
        pool.tick()
        pool.write()
        data = json.loads((root / "q.json").read_text(encoding="utf-8"))
        assert [e["path"] for e in data["entries"]] == ["v0.pt", "v1.pt"] and data["clock"] == 2
        again = league.Pool(root, size=8)
        assert again.entries == pool.entries and again.clock == 2
        assert again.paths == [tmp_path / "v0.pt", tmp_path / "v1.pt"]
        assert again.sample()[0] in again.paths
        again.add(tmp_path / "v2.pt")
        assert again.top(1)[0][0] == "v2.pt" or again.find(tmp_path / "v2.pt")["added"] == 2
        assert "pool 3 versions (clock 2)" in again.text()


class TestSlots:
    """rollout.Battles' past slots: a battle keeps its slot's version from its start."""

    @staticmethod
    def forced(kind, seed):
        actor = nets(seed)[0]
        with torch.no_grad():
            actor.heads.kind.weight.zero_()
            actor.heads.kind.bias.fill_(-30.0)
            actor.heads.kind.bias[kind] = 30.0
        return actor

    def test_running_battles_keep_their_version_and_new_ones_take_the_active_slot(self):
        lay = league.layout(4, 1, {"past": 1.0})
        env = rollout.Battles(lay, MIRROR, params=rollout.params_with_limit(4.0), spread=randomise.NONE, past_slots=2,
                              compile=False)
        hold, move = self.forced(O.HOLD, 1), self.forced(O.MOVE, 2)
        env.set_past(hold, slot=0)
        learner, crit = nets()
        env.step(learner, crit)
        h0 = env.past_h[0].clone()
        assert env.slot_counts() == [4, 0]
        env.set_past(move, untrained=True, slot=1)
        env.activate(1)
        assert env.battle_slot.tolist() == [0, 0, 0, 0] and torch.equal(env.past_h[0], h0)   # memory kept
        a, frame, _ = env.cur
        orders, hs = env._act_past(a, frame)
        ctrl = rollout.rows_of(a, env.rows_past)["ctrl"]
        assert set(hs) == {0, 1} and bool((orders.kind[ctrl] == O.HOLD).all())       # all still slot 0's version
        for _ in range(12):                                   # the 4 s limit: every battle ends and starts again
            env.step(learner, crit)
        assert env.battle_slot.tolist() == [1, 1, 1, 1] and env.slot_counts() == [0, 4]
        assert env.slot_used == {1}
        a, frame, _ = env.cur
        orders, hs = env._act_past(a, frame)
        assert set(hs) == {1} and bool((orders.kind[rollout.rows_of(a, env.rows_past)["ctrl"]] == O.MOVE).all())
        g = env.take_slot_stats()
        assert g[0, 0] == 4 and g[1, 0] >= 4 and (g[:, 1] <= g[:, 0]).all()
        stats = env.take_stats()                              # slot 1's version marked untrained: its battles apart
        assert stats.get("past/attack", (0,))[0] + stats.get("past/defend", (0,))[0] == 4

    def test_a_battle_mixing_two_slots_takes_each_rows_own_version(self):
        lay = league.layout(4, 1, {"past": 1.0})
        env = rollout.Battles(lay, MIRROR, params=rollout.params_with_limit(60.0), spread=randomise.NONE,
                              past_slots=2, compile=False)
        env.set_past(self.forced(O.HOLD, 1), slot=0)
        env.set_past(self.forced(O.MOVE, 2), slot=1)
        env.battle_slot = torch.tensor([0, 1, 0, 1])
        env.slot_used = {0, 1}
        learner, crit = nets()
        env.step(learner, crit)
        a, frame, _ = env.cur
        orders, _ = env._act_past(a, frame)
        ctrl = rollout.rows_of(a, env.rows_past)["ctrl"]
        slot = env.battle_slot[env.rows_past % env.B]
        want = torch.where(slot == 0, O.HOLD, O.MOVE)[:, None].expand_as(orders.kind)
        assert bool((orders.kind[ctrl] == want[ctrl]).all())

    def test_one_slot_is_the_old_single_version(self):
        env = rollout.Battles(league.layout(4, 1, {"past": 0.5, "nearest": 0.5}), MIRROR, compile=False)
        env.set_past(nets(1)[0], untrained=True)
        assert env.past_slots == 1 and env.past_actor is env.past[0] and bool(env.slot_untrained[0])


class TestResume:
    def test_adams_state_and_the_counters_go_into_the_checkpoint_and_back(self, tmp_path):
        from tools.nn.train import run
        actor, crit = nets()
        opt = run.optimizer(actor, crit, 1e-3)
        for p in list(actor.parameters()) + list(crit.parameters()):
            p.grad = torch.ones_like(p)
        opt.step()
        state = {"floor_w": 0.02, "entropy_decay_s": 600.0, "chain_s": 1800.0, "chain_updates": 61}
        path = checkpoint.save(tmp_path / "latest.pt", actor, crit, "small", {}, {"optim": opt.state_dict(),
                                                                                 "state": state})
        a2, c2 = nets(5)
        a2.load_state_dict(checkpoint.read(path)["actor"])
        c2.load(checkpoint.read(path)["critic"])
        opt2 = run.optimizer(a2, c2, 5e-5)
        assert run.resume(opt2, path) == state
        assert opt2.param_groups[0]["lr"] == 5e-5                      # the options' learning rate stays
        s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
        assert s1.keys() == s2.keys() and all(torch.equal(s1[k]["exp_avg_sq"], s2[k]["exp_avg_sq"]) for k in s1)
        bare = checkpoint.save(tmp_path / "bare.pt", actor, crit)
        assert run.resume(run.optimizer(*nets(), 1e-3), bare) is None    # nothing to resume
        other = config.preset("small", d=48, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
        wide = run.optimizer(policy.Actor(other), critic.Critic(other), 1e-3)
        assert run.resume(wide, path) is None                          # other shapes: afresh

    def test_a_run_goes_on_from_its_latest_with_adam_the_counters_and_the_shared_pool(self, tmp_path, monkeypatch):
        from tools.nn.train import run
        actor, crit = nets()
        init = checkpoint.save(tmp_path / "init.pt", actor, crit)
        monkeypatch.setattr(checkpoint, "DIR", tmp_path)
        monkeypatch.setattr(checkpoint, "RANDOM", checkpoint.save(tmp_path / "random.pt", *nets(3)))
        pool_dir = tmp_path / "night_pool"
        base = ["--battles", "8", "--steps", "4", "--updates", "4", "--minutes", "3", "--device", "cpu", "--no-eval",
                "--mix", '{"past": 0.5, "nearest": 0.5}', "--limit", "6", "--bank", "8", "--max-units", "4", "--snapshot-every", "2",
                "--past-every", "1", "--past-slots", "2", "--pool-dir", str(pool_dir), "--critic-warmup", "2",
                "--entropy", "0.01", "--entropy-end", "0.003", "--entropy-decay", "0.05"]
        _, _, out1 = run.train(run.parser().parse_args(["--name", "p1", "--init", str(init)] + base))
        q1 = json.loads((pool_dir / "q.json").read_text(encoding="utf-8"))
        names = [e["path"].rsplit("/", 1)[-1] for e in q1["entries"]]
        assert q1["clock"] == 4 and names == ["random.pt", "init.pt", "p1_v00002.pt", "p1_v00004.pt"]
        assert [e["untrained"] for e in q1["entries"]] == [True, False, False, False]
        assert sum(e["games"] for e in q1["entries"]) > 0                 # battles against past versions counted
        assert all(e["q"] <= 0 for e in q1["entries"]) and any(e["q"] < 0 for e in q1["entries"] if e["wins"])
        rows = [json.loads(x) for x in (out1 / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(2 <= r["pool"]["n"] <= len(q1["entries"]) and r["pool"]["top"] for r in rows)
        assert rows[0]["grad_norm"] == 0.0 and rows[2]["grad_norm"] > 0     # the warm-up: the critic only
        tr = checkpoint.train_state(out1 / "latest.pt")
        assert tr["state"]["chain_updates"] == 4 and tr["state"]["entropy_decay_s"] > 0 and tr["optim"]["state"]
        _, _, out2 = run.train(run.parser().parse_args(["--name", "p2", "--init", str(out1 / "latest.pt"),
                                                        "--critic-init", str(out1 / "latest.pt")] + base))
        q2 = json.loads((pool_dir / "q.json").read_text(encoding="utf-8"))
        assert q2["clock"] == 8 and [e["path"] for e in q2["entries"]][:len(q1["entries"])] == \
            [e["path"] for e in q1["entries"]]                         # the same pool goes on (no new init entry)
        rows2 = [json.loads(x) for x in (out2 / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert 0.003 <= rows2[0]["entropy_weight"] < rows[-1]["entropy_weight"] < 0.01   # the decay went on
        assert rows2[0]["grad_norm"] > 0                                # no warm-up: Adam and the critic go on
        assert checkpoint.train_state(out2 / "latest.pt")["state"]["chain_updates"] == 8


class TestEntropyDecay:
    def test_the_decay_counts_its_own_minutes_over_the_chain_else_the_run(self):
        from tools.nn.train import run
        assert run.decay_share(0.0, 999.0, 0.25) == 0.25
        assert run.decay_share(60.0, 1800.0, 0.9) == 0.5
        assert run.schedule(0.01, 0.003, run.decay_share(60.0, 7200.0, 0.0)) == pytest.approx(0.003)
        args = run.parser().parse_args([])
        assert args.entropy_decay == 0.0 and args.past_slots == 3 and args.past_every == 6 and args.resume
