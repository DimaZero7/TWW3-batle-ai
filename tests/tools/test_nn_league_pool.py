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

    def test_new_pools_in_other_folders_draw_other_versions_and_one_folder_the_same(self, tmp_path):
        vs = files(tmp_path, 4)

        def draws(folder):
            pool = league.Pool(folder, size=8, seed=0)          # a new pool (no q.json): the clock 0, no entries
            for p in vs:
                pool.add(p)
            return [pool.sample()[0].name for _ in range(12)]
        a = draws(tmp_path / "runs" / "test5_a" / "pool")
        assert draws(tmp_path / "runs" / "test5_b" / "pool") != a     # every step of a chain: its own draws
        assert draws(tmp_path / "runs" / "test5_a" / "pool") == a     # the same folder: the same draws


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

    @pytest.mark.parametrize("share", [0.0, 1.0])
    def test_a_pinned_slot_takes_its_share_of_the_starting_past_battles(self, share):
        lay = league.layout(4, 1, {"past": 1.0})
        env = rollout.Battles(lay, MIRROR, params=rollout.params_with_limit(4.0), spread=randomise.NONE, past_slots=2,
                              compile=False)
        env.set_past(self.forced(O.HOLD, 1), slot=0)
        env.pin_slot(1, self.forced(O.MOVE, 2), share, seed=3)
        want = 4 if share else 0
        assert int((env.battle_slot == 1).sum()) == want            # the battles that start now draw too
        learner, crit = nets()
        for _ in range(14):                                          # the 4 s limit: every battle ends and starts again
            env.step(learner, crit)
        assert int((env.battle_slot == 1).sum()) == want
        counts = env.slot_counts()
        assert counts[1] == want and 1 in env.slot_used                 # the pinned slot always computes
        a, frame, _ = env.cur
        orders, _ = env._act_past(a, frame)
        ctrl = rollout.rows_of(a, env.rows_past)["ctrl"]
        assert bool((orders.kind[ctrl] == (O.MOVE if share else O.HOLD)).all())
        stats = env.take_stats()
        n = {who: sum(stats.get(f"{who}/{r}", (0,))[0] for r in ("attack", "defend")) for who in ("past", "pinned")}
        assert n["pinned" if share else "past"] >= 4 and n["past" if share else "pinned"] == 0
        g = env.take_slot_stats()
        assert g[1, 0] == n["pinned"] and g[0, 0] == n["past"]
        # ... and by matchup (take_pin_stats): the learner's role and factions, the same games and wins
        by = env.take_pin_stats()
        assert all(k.split("/")[0] == "pinned" and k.split("/")[1] in ("attack", "defend")
                   and k.split("/")[2] in ("EMP-EMP", "EMP-SKV", "SKV-EMP", "SKV-SKV") for k in by)
        assert sum(gm for gm, _, _ in by.values()) == n["pinned"]
        for role in ("attack", "defend"):
            gm, wn, _ = stats.get(f"pinned/{role}", (0, 0, 0.0))
            assert sum(v[0] for k, v in by.items() if k.split("/")[1] == role) == gm
            assert sum(v[1] for k, v in by.items() if k.split("/")[1] == role) == wn
        assert all(-1.5 <= t <= 1.5 for _, _, t in by.values()) and env.take_pin_stats() == {}
        with pytest.raises(ValueError):
            env.activate(1)                                          # the pool never takes the pinned slot
        with pytest.raises(ValueError):
            env.pin_slot(0, self.forced(O.MOVE, 2), 0.5)             # the active slot cannot be pinned

    def test_the_pinned_share_of_many_starting_battles(self):
        env = rollout.Battles(league.layout(200, 1, {"past": 0.5, "nearest": 0.5}), MIRROR, past_slots=2,
                              compile=False)
        env.set_past(nets(1)[0], slot=0)
        env.pin_slot(1, nets(2)[0], 0.65, seed=1)
        share = float((env.battle_slot[env.is_past] == 1).float().mean())
        assert 0.5 < share < 0.8 and bool(env.slot_pinned[1]) and not bool(env.slot_pinned[0])

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
        assert run.resume(wide, path) == state                         # other shapes: Adam afresh, the counters go on
        assert not wide.state_dict()["state"]

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


    def test_a_pinned_past_version_plays_outside_the_pool(self, tmp_path, monkeypatch):
        from tools.nn.train import run
        actor, crit = nets()
        init = checkpoint.save(tmp_path / "init.pt", actor, crit)
        pin = checkpoint.save(tmp_path / "pin.pt", *nets(7))
        monkeypatch.setattr(checkpoint, "DIR", tmp_path)
        monkeypatch.setattr(checkpoint, "RANDOM", checkpoint.save(tmp_path / "random.pt", *nets(3)))
        args = ["--name", "pin", "--init", str(init), "--battles", "8", "--steps", "4", "--updates", "4", "--minutes",
                "3", "--device", "cpu", "--no-eval", "--mix", '{"past": 0.5, "nearest": 0.5}', "--limit", "6",
                "--bank", "8", "--max-units", "4", "--past-pin", str(pin), "--past-pin-share", "1.0", "--critic-warmup", "0"]
        _, trained, out = run.train(run.parser().parse_args(args))
        q = json.loads((out / "pool" / "q.json").read_text(encoding="utf-8"))
        assert not any(e["path"].endswith("pin.pt") for e in q["entries"])          # not in the pool
        assert sum(e["games"] for e in q["entries"]) == 0      # share 1: every past battle the pinned one's, no q changes
        games = {}
        for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines():
            for k, g in json.loads(line)["games"].items():
                games[k] = games.get(k, 0) + g
        assert games.get("pinned/attack", 0) + games.get("pinned/defend", 0) > 0
        # the pinned battles by matchup, per update and in the summary (report.json "training")
        rows = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        by = {}
        for r in rows:
            for k, c in (r.get("pinned_matchups") or {}).items():
                by[k] = by.get(k, 0) + c["games"]
                assert set(c) == {"games", "wins", "win_rate", "gold_trade"}
        pinned = games.get("pinned/attack", 0) + games.get("pinned/defend", 0)
        assert sum(by.values()) == pinned and all(k.startswith(("pinned/attack/", "pinned/defend/")) for k in by)
        assert {k: c["games"] for k, c in trained["pinned_matchups"].items()} == by
        assert not any(k.startswith("past/") for k in games) and any(k.startswith("nearest/") for k in games)
        with pytest.raises(SystemExit):                                              # no past battles: nothing to pin
            run.train(run.parser().parse_args(["pin2" if a == "pin" else '{"nearest": 1.0}' if a.startswith('{"past"')
                                               else a for a in args]))


class TestEntropyDecay:
    def test_the_decay_counts_its_own_minutes_over_the_chain_else_the_run(self):
        from tools.nn.train import run
        assert run.decay_share(0.0, 999.0, 0.25) == 0.25
        assert run.decay_share(60.0, 1800.0, 0.9) == 0.5
        assert run.schedule(0.01, 0.003, run.decay_share(60.0, 7200.0, 0.0)) == pytest.approx(0.003)
        args = run.parser().parse_args([])
        assert args.entropy_decay == 0.0 and args.past_slots == 3 and args.past_every == 6 and args.resume


class TestChainPool:
    """A chain step's pool (league.open_pool, run.py): it carries on from the previous step's pool/q.json beside
    --init or --critic-init (--pool-fresh: not); --pool-no-untrained keeps the untrained network out."""

    def prev_run(self, tmp_path):
        run = tmp_path / "runs" / "test5_prev"
        (run / "pool").mkdir(parents=True)
        (run / "latest.pt").write_bytes(b"")
        a, b, rnd = files(tmp_path, 3)
        state = {"clock": 33, "eta": 0.01, "entries": [
            {"path": str(rnd), "q": -0.9, "added": 0, "games": 40, "wins": 38, "untrained": True},
            {"path": str(a), "q": -0.3, "added": 10, "games": 20, "wins": 12, "untrained": False},
            {"path": str(b), "q": 0.0, "added": 30, "games": 2, "wins": 0, "untrained": False}]}
        (run / "pool" / "q.json").write_text(json.dumps(state), encoding="utf-8")
        return run, state, (a, b, rnd)

    def test_a_new_step_carries_on_from_the_previous_steps_q(self, tmp_path):
        run, state, (a, b, rnd) = self.prev_run(tmp_path)
        made = []
        init = tmp_path / "test5" / "prev" / "m40.pt"                     # (a network folder without a pool)
        for i, src in enumerate(({"init": run / "latest.pt"}, {"init": init, "critic": run / "latest.pt"})):
            root = tmp_path / f"new{i}" / "pool"
            pool, notes = league.open_pool(root, size=12, untrained=lambda: made.append(1) or tmp_path / "rnd.pt", **src)
            assert [e["q"] for e in pool.entries] == [-0.9, -0.3, 0.0] and pool.clock == 33
            assert pool.probs()[2] > pool.probs()[0] and notes and not made     # not all equal; no new untrained
            assert json.loads((root / "q.json").read_text(encoding="utf-8"))["clock"] == 33
        assert json.loads((run / "pool" / "q.json").read_text(encoding="utf-8")) == state    # the old one untouched

    def test_no_untrained_drops_it_and_a_fresh_or_own_pool_is_left_as_it_was(self, tmp_path):
        run, _, (a, b, rnd) = self.prev_run(tmp_path)
        pool, _ = league.open_pool(tmp_path / "n1" / "pool", init=run / "latest.pt", untrained=lambda: rnd,
                                   no_untrained=True)
        assert [e["q"] for e in pool.entries] == [-0.3, 0.0] and not any(e["untrained"] for e in pool.entries)
        # --pool-fresh: a new pool - the untrained network (or, with no_untrained, the init alone) and the init
        init = tmp_path / "init.pt"
        init.write_bytes(b"")
        pool, _ = league.open_pool(tmp_path / "n2" / "pool", init=run / "latest.pt", untrained=lambda: rnd,
                                   inherit=False)
        assert [bool(e["untrained"]) for e in pool.entries] == [True, False]
        pool, _ = league.open_pool(tmp_path / "n3" / "pool", init=init, untrained=lambda: rnd, no_untrained=True)
        assert len(pool) == 1 and pool.entries[0]["path"] == league.Pool.key(init)
        pool, _ = league.open_pool(tmp_path / "n4" / "pool", untrained=lambda: rnd, no_untrained=True)
        assert len(pool) == 1 and pool.entries[0]["untrained"]                 # nothing else to play: kept
        # a folder with its own q.json (--pool-dir, a resumed run) keeps it
        own = tmp_path / "n3" / "pool"
        pool, notes = league.open_pool(own, init=run / "latest.pt", untrained=lambda: rnd, extra=[a])
        assert not notes and [e["path"] for e in pool.entries] == [league.Pool.key(init), league.Pool.key(a)]

    def test_the_flags_and_the_general_steps_of_the_chain(self):
        from pathlib import Path
        from tools.nn.train import run
        ns = run.parser().parse_args(["--pool-no-untrained", "--pool-fresh"])
        assert ns.pool_no_untrained and ns.pool_fresh
        ns = run.parser().parse_args([])
        assert not ns.pool_no_untrained and not ns.pool_fresh
        chain = json.loads((Path(__file__).resolve().parents[2] / "config" / "train-chain.json").read_text(
            encoding="utf-8"))
        assert chain["steps"]["general"]["options"]["pool-no-untrained"] is True


class TestPinCells:
    """run.py --past-pin-cells (rollout.pin_cells, Battles.pin_slot cells): the pinned version only in past battles of
    its cells - the cell in ITS view (its role, its own faction first) - with --past-pin-share there; elsewhere the
    pool's version."""

    def env(self, B=400):
        env = rollout.Battles(league.layout(B, 1, {"past": 1.0}), MIRROR, past_slots=2, compile=False)
        env.set_past(nets(1)[0], slot=0)
        return env

    def set_cells(self, env, gen):
        """Random factions and attackers for every battle (a new setup.factions list: the codes made again)."""
        F = ["wh_main_emp_empire", "wh2_main_skv_skaven"]
        fac = [[F[int(torch.randint(2, (1,), generator=gen))], F[int(torch.randint(2, (1,), generator=gen))]]
               for _ in range(env.B)]
        env.setup.factions = fac
        env._fac_key = None
        env.st.attacker = torch.randint(1, 3, (env.B,), generator=gen)
        ps = (env.ctrl[:, 0] != league.CODE["past"]).long()
        names = []
        for b in range(env.B):
            p = int(ps[b])
            role = "attack" if int(env.st.attacker[b]) == p + 1 else "defend"
            own, enemy = rollout.matchups.SHORT[fac[b][p]], rollout.matchups.SHORT[fac[b][1 - p]]
            names.append(f"{role}/{own}-{enemy}")
        return names

    def test_names_and_the_auto_cells_of_a_report(self, tmp_path):
        assert rollout.pin_cells("defend/EMP-EMP, attack/SKV-EMP") == ["defend/EMP-EMP", "attack/SKV-EMP"]
        assert rollout.pin_cell_code("pinned/defend/EMP-EMP") == rollout.pin_cell_code("defend/EMP-EMP")
        for bad in ("hold/EMP-EMP", "defend/EMP", "defend/EMP-XYZ", ""):
            with pytest.raises(ValueError):
                rollout.pin_cells(bad)
        rep = {"training": {"pinned_matchups": {
            "pinned/defend/EMP-EMP": {"games": 311, "wins": 200, "win_rate": 0.643, "gold_trade": 0.07},
            "pinned/defend/SKV-EMP": {"games": 676, "wins": 369, "win_rate": 0.546, "gold_trade": 0.08},
            "pinned/attack/SKV-SKV": {"games": 511, "wins": 281, "win_rate": 0.55, "gold_trade": 0.03},
            "pinned/attack/EMP-SKV": {"games": 60, "wins": 50, "win_rate": 0.83, "gold_trade": 0.2}}}}
        path = tmp_path / "x2" / "report.json"
        path.parent.mkdir()
        path.write_text(json.dumps(rep), encoding="utf-8")
        assert rollout.pin_cells(f"auto:{path}:0.55") == ["attack/SKV-SKV", "defend/EMP-EMP"]   # (60 games: too few)
        assert rollout.pin_cells(f"auto:{path}:0.5:50") == ["attack/EMP-SKV", "attack/SKV-SKV", "defend/EMP-EMP",
                                                            "defend/SKV-EMP"]
        assert rollout.pin_cells(f"auto:{path}") == rollout.pin_cells(f"auto:{path}:0.55")
        with pytest.raises(ValueError):
            rollout.pin_cells(f"auto:{path}:0.9")

    def test_the_pinned_version_plays_only_its_cells_at_its_share_there(self):
        gen = torch.Generator().manual_seed(5)
        env = self.env()
        names = self.set_cells(env, gen)
        cells = ["defend/EMP-EMP", "attack/SKV-EMP"]
        env.pin_slot(1, nets(2)[0], 0.65, seed=1, cells=cells)
        for _ in range(3):                                          # new battles: new cells, new draws
            slot = env._restart_slot(torch.ones(env.B, dtype=torch.bool))
            mine = torch.tensor([n in cells for n in names])
            assert int(mine.sum()) > 40
            assert not bool((slot[~mine] == 1).any())                # never outside its cells
            share = float((slot[mine] == 1).float().mean())
            assert 0.5 < share < 0.8                                  # its share among its cells' battles
            names = self.set_cells(env, gen)
        # without cells: any cell
        env2 = self.env()
        self.set_cells(env2, gen)
        env2.pin_slot(1, nets(2)[0], 1.0, seed=1)
        assert bool((env2.battle_slot == 1).all())

    def test_the_run_refuses_cells_without_a_pin(self):
        from tools.nn.train import run
        with pytest.raises(SystemExit):
            run.train(run.parser().parse_args(["--past-pin-cells", "defend/EMP-EMP", "--device", "cpu"]))
        ns = run.parser().parse_args(["--past-pin", "x.pt", "--past-pin-cells", "auto:r.json:0.55"])
        assert ns.past_pin_cells == "auto:r.json:0.55"
