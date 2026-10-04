"""tools/nn/train/drills/teach.py: the drills' teacher (an annealed imitation term, run.py --drill-teach).

torch is not in the project's .venv: skipped there, run in the training container. Level 1: the
weight's schedule. Level 2: the mapping - on kiting drill states the skilled script's labels decode
(decide.to_orders) back to the script's orders: the kind, run, the move point to the nearest bin's,
the attack target to the same slot; the rollout labels only the taught drill's rows. Level 3: pure
imitation on drill battles (a short CPU run) raises the agreement; PPO's update and run.py log the
term and the agreement per drill, test5 reads them.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, critic, heads, policy  # noqa: E402
from tools.nn.model import observation as ob  # noqa: E402
from tools.nn.model.decide import to_orders  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.train import checkpoint, league, ppo, randomise, rollout  # noqa: E402
from tools.nn.train import drills as D  # noqa: E402
from tools.nn.train.drills import kiting as K  # noqa: E402
from tools.nn.train.drills import source as ds  # noqa: E402
from tools.nn.train.drills import teach  # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
KITING = league.CODE[D.opponent("kiting")]


def nets(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(CFG).eval(), critic.Critic(CFG).eval()


def kiting_env(B, other=0, taught=True):
    """B battles of the kiting drill (our side alternating) and `other` against nearest (generated)."""
    lay = league.Layout(np.zeros(B + other, dtype=int), 1 + np.arange(B + other) % 2,
                        np.array([KITING] * B + [league.CODE["nearest"]] * other))
    broad, embed, D.BROAD, D.EMBED = D.BROAD, D.EMBED, 0.0, 0.0     # the clean frame: its only active label is the run-back
    try:
        src = ds.Mixed(np.arange(other), 3, None, "cpu", lay, per_drill=max(B, 2), drills={"kiting": K.DRILL}, seed=1)
    finally:
        D.BROAD, D.EMBED = broad, embed
    return rollout.Battles(lay, device="cpu", source=src, compile=False, spread=randomise.NONE,
                           teach={"kiting": K.skilled} if taught else None)


def chasers_near(env, battles, m=30.0):
    """Every enemy of the given battles 30 m from our first unit (our side: the learner's)."""
    u = env.st.u
    for b in battles:
        ours = int(env.layout.learner[b])
        i0 = int((u["side"][b] == ours).nonzero()[0])
        for j in (u["side"][b] == 3 - ours).nonzero().flatten().tolist():
            u["x"][b, j], u["z"][b, j] = u["x"][b, i0] + m, u["z"][b, i0]


class TestSchedule:
    def test_the_weight_goes_linearly_to_zero(self):
        assert teach.weights({"kiting": 0.5}, 10, 0) == {"kiting": 0.5}
        assert teach.weights({"kiting": 0.5}, 10, 5)["kiting"] == pytest.approx(0.25)
        assert teach.weights({"kiting": 0.5}, 10, 12) == {"kiting": 0.0}
        assert teach.weights({}, 10, 1) == {}


class TestMapping:
    def test_kiting_labels_decode_back_to_the_scripts_orders(self):
        B = 8
        env = kiting_env(B)
        chasers_near(env, range(0, B, 2))                              # half run back, half hold
        a, frame, _ = env.observe(critic=False)
        rows = env.rows_learn
        obs_r, frame_r = rollout.rows_of(a, rows), rollout.frame_rows(frame, rows)
        o = K.skilled(env.st)
        lab, valid = teach.label(CFG, o, obs_r, frame_r, rows, B)
        dec = to_orders(CFG, lab, obs_r, frame_r, env.bounds2[rows])
        want = O.Orders(*(getattr(o, k)[rows % B] for k in O.FIELDS))
        assert bool(valid.any()) and bool((valid == obs_r["ctrl"]).all())   # every unit that takes orders
        assert bool((dec.kind[valid] == want.kind[valid]).all())
        moving = valid & (want.kind == O.MOVE)
        assert int(moving.sum()) >= 3 and int((valid & (want.kind == O.HOLD)).sum()) >= 3
        assert bool((dec.run[moving] == want.run[moving]).all()) and bool(want.run[moving].all())
        # the point: the bin nearest the script's (decoded and clipped no farther than any bin's point)
        sx, sz = want.x, want.z
        err = torch.hypot(dec.x - sx, dec.z - sz)
        pos = obs_r["pos"] * ob.POS                                              # [R, N, 2]
        off = heads.point_offsets(CFG)                                           # [P, 2]
        bx, bz = frame_r.world(pos[..., None, 0] + off[:, 0], pos[..., None, 1] + off[:, 1])
        best = torch.hypot(bx - sx[..., None], bz - sz[..., None]).min(-1).values
        assert bool((err[moving] <= best[moving] + 1e-2).all())
        hx, hz = frame_r.world(pos[..., 0], pos[..., 1])                         # the unit's place
        reach = torch.hypot(sx - hx, sz - hz)
        assert bool((err[moving] <= 0.35 * reach[moving] + 1.0).all())         # 16 directions x geometric distances

    def test_attack_labels_point_at_the_same_slot(self):
        """The pointer: the enemy side's chase orders (ATTACK a slot), labelled on its rows, decode to the slot."""
        B = 6
        env = kiting_env(B)
        a, frame, _ = env.observe(critic=False)
        enemy_side = 3 - torch.as_tensor(env.layout.learner)
        rows = (enemy_side - 1) * B + torch.arange(B)
        obs_r, frame_r = rollout.rows_of(a, rows), rollout.frame_rows(frame, rows)
        o = K.enemy(env.st)
        lab, valid = teach.label(CFG, o, obs_r, frame_r, rows, B)
        dec = to_orders(CFG, lab, obs_r, frame_r, env.bounds2[rows])
        want_t, want_k = o.target[rows % B], o.kind[rows % B]
        att = valid & (want_k == O.ATTACK)
        assert int(att.sum()) >= B
        assert bool((dec.kind[att] == O.ATTACK).all()) and bool((dec.target[att] == want_t[att]).all())
        assert bool((dec.run[att] == o.run[rows % B][att]).all())
        # an attack on a target that may not be attacked now is not a label
        hidden = obs_r["ctrl"] & (want_k == O.ATTACK) & ~obs_r["target_ok"].gather(-1, want_t.clamp(min=0))
        assert not bool((valid & hidden).any())

    def test_the_rollout_labels_only_the_taught_drills_rows(self):
        env = kiting_env(4, other=4)
        actor, crit = nets()
        batch = rollout.collect(env, actor, crit, 3)
        t = batch["teach"]
        T, R = batch["reward"].shape
        assert t["names"] == ("kiting",) and t["valid"].shape == (T, R, env.N) and t["drill"].shape == (T, R, env.N)
        drill_rows = env.row_opp == KITING
        assert bool((t["drill"][:, drill_rows] == 0).all()) and bool((t["drill"][:, ~drill_rows] == -1).all())
        assert bool(t["valid"][:, drill_rows].any()) and not bool(t["valid"][:, ~drill_rows].any())
        assert not bool((t["valid"] & ~batch["obs"]["ctrl"]).any())
        assert "teach" not in rollout.collect(kiting_env(2, taught=False), actor, crit, 1)


def _flat(x):
    return x.reshape(-1, *x.shape[2:])


def _imitate(actor, batch):
    """(logits flattened [T * R, ...], labels, valid, drill) of a collected batch."""
    obs = rollout.full_obs(batch["obs"], batch["abil_static"])
    logits, _ = actor.sequence(obs, batch["h0"], batch["reset"])
    t = batch["teach"]
    lab = heads.Action(*(_flat(getattr(t["action"], f)) for f in ("kind", "point", "target", "run")))
    return {k: _flat(v) for k, v in logits.items()}, lab, _flat(t["valid"]), _flat(t["drill"]), t["names"]


class TestImitation:
    def test_pure_imitation_on_drill_battles_raises_the_agreement(self, monkeypatch):
        """A short CPU run of the imitation term alone (no PPO): rounds of a rollout played by the network,
        then gradient steps on the labels; the agreement on each new rollout (states the steps did not
        see) is measured before training on it."""
        monkeypatch.setattr(K, "GAP_M", (70.0, 150.0))                 # the chasers come near soon: run-back labels
        torch.manual_seed(0)
        env = kiting_env(16)
        actor, crit = nets()
        opt = torch.optim.Adam(actor.parameters(), lr=2e-3)
        fresh, moves = [], []
        for _ in range(6):
            batch = rollout.collect(env, actor, crit, 16)
            with torch.no_grad():
                logits, lab, valid, drill, names = _imitate(actor, batch)
                _, sums = teach.terms(logits, lab, valid, drill, names)
                fresh.append(teach.summary(sums)["kiting"])
                mv = valid & (lab.kind == O.MOVE)
                moves.append(float((teach.agree(logits, lab) & mv).float().sum() / mv.float().sum().clamp(min=1)))
                assert fresh[-1]["agree_active"] == pytest.approx(moves[-1], abs=1e-3)   # kiting: active = run back
            actor.train()
            for _ in range(40):
                logits, lab, valid, drill, names = _imitate(actor, batch)
                loss, _ = teach.terms(logits, lab, valid, drill, names, {"kiting": 1.0})
                opt.zero_grad()
                loss.backward()
                opt.step()
            actor.eval()
        print("fresh rollouts:", fresh, "move-label agreement:", moves)
        assert fresh[0]["units"] > 100 and all(f["units"] > 0 for f in fresh)
        # measured (CPU, seed 0): agreement 0.00 -> 0.54, on the kind 0.19 -> 0.87, on the run-back labels
        # 0.00 -> 0.45, cross-entropy 2.41 -> 1.14
        assert fresh[-1]["agree"] >= fresh[0]["agree"] + 0.3
        assert fresh[-1]["agree_kind"] >= max(0.75, fresh[0]["agree_kind"] + 0.3)
        assert moves[-1] >= moves[0] + 0.25
        assert fresh[-1]["ce"] < 0.6 * fresh[0]["ce"]

    def test_ppo_update_adds_the_term_and_logs_it_per_drill(self):
        env = kiting_env(4, other=2)
        actor, crit = nets()
        batch = rollout.collect(env, actor, crit, 4)
        opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=1e-3)
        cfg = ppo.PPOConfig(minibatch=16, target_kl=0.0)               # every minibatch (no early stop)
        st = ppo.update(actor, crit, opt, batch, cfg, teach={"kiting": 0.5})
        t = st["teach"]["kiting"]
        assert t["weight"] == 0.5 and t["units"] > 0 and t["ce"] > 0 and 0 <= t["agree"] <= t["agree_kind"] <= 1
        assert 0 <= t["active"] <= t["units"] and (t["agree_active"] is None) == (t["active"] == 0)
        st0 = ppo.update(actor, crit, opt, batch, cfg)                  # no weights: logged, no term
        assert st0["teach"]["kiting"]["weight"] == 0.0 and st0["teach"]["kiting"]["units"] == t["units"]
        del batch["teach"]
        assert "teach" not in ppo.update(actor, crit, opt, batch, cfg)

    def test_a_run_with_the_teacher_logs_it_and_test5_reports_it(self, tmp_path, monkeypatch):
        from tools.nn.train import run, test5
        actor, crit = nets()
        init = checkpoint.save(tmp_path / "init.pt", actor, crit)
        monkeypatch.setattr(checkpoint, "DIR", tmp_path)
        monkeypatch.setattr(checkpoint, "RANDOM", init)
        base = ["--init", str(init), "--battles", "8", "--steps", "4", "--updates", "3", "--minutes", "3",
                "--device", "cpu", "--no-eval", "--mix", '{"nearest": 1.0}', "--drills", "0.5",
                "--drill-weights", '{"kiting": 1}', "--drill-bank", "8", "--bank", "8", "--max-units", "4",
                "--print-every", "1", "--drill-embed", "0"]
        args = run.parser().parse_args(["--name", "teach"] + base + ["--drill-teach", '{"kiting": 0.5}',
                                                                     "--drill-teach-minutes", "100"])
        _, summary, out = run.train(args)
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3 and all("kiting" in r["teach"] for r in rows)
        w = [r["teach"]["kiting"]["weight"] for r in rows]
        assert 0.49 < w[0] <= 0.5 and w[0] >= w[1] >= w[2] and sum(r["teach"]["kiting"]["units"] for r in rows) > 0
        rep = test5.teacher(out, 0, 3)
        assert rep["kiting"]["updates"] >= 1 and 0.49 < rep["kiting"]["weight"] <= 0.5
        lines = test5.teach_block([{}, {"teach": rep}], ["before → after"], " → ")
        assert any(line.startswith("| teacher kiting: agreement") for line in lines)
        assert test5.teacher(out, 3, 9) is None
        bad = run.parser().parse_args(["--name", "bad"] + base + ["--drill-teach", '{"counter": 0.5}'])
        with pytest.raises(SystemExit):
            run.train(bad)


class TestAdaptiveShare:
    """The adaptive teacher (run.py --drill-teach auto, teach_auto.py): only a share of a drill's battles is
    labelled, whole battles, never another opponent's; the term over all the drill's units."""

    def test_only_a_share_of_the_drills_battles_is_labelled(self):
        env = kiting_env(16, other=4)
        actor, crit = nets()
        drill_rows = env.row_opp == KITING
        for share in (0.0, 0.5, 1.0):
            env.set_teach_shares({"kiting": share})
            batch = rollout.collect(env, actor, crit, 3)
            t = batch["teach"]
            assert t["shares"] == {"kiting": share} and t["picked"].shape == t["drill"].shape
            assert not bool(t["picked"][:, ~drill_rows].any())                  # never another opponent's battle
            draw = env.teach_draw[env.rows_learn % env.B]
            want = drill_rows & (draw < share)                                   # the draw of the battle: whole
            assert bool((t["picked"] == want[:, None]).all())
            assert bool(t["valid"][:, drill_rows].any())                         # labels on every drill row still
        assert 0 < int((drill_rows & (draw < 0.5)).sum()) < int(drill_rows.sum())

    def test_a_new_battle_draws_again(self):
        env = kiting_env(8)
        before = env.teach_draw.clone()
        finished = torch.zeros(env.B, dtype=torch.bool)
        finished[:3] = True
        env._reset(finished)
        assert bool((env.teach_draw[:3] != before[:3]).all()) and bool((env.teach_draw[3:] == before[3:]).all())

    def test_the_term_sums_the_labelled_units_over_all_the_drills(self):
        env = kiting_env(8)
        actor, crit = nets()
        batch = rollout.collect(env, actor, crit, 4)
        logits, lab, valid, drill, names = _imitate(actor, batch)
        logits = {k: v.detach() for k, v in logits.items()}
        w = {"kiting": 0.15}
        full, s_full = teach.terms(logits, lab, valid, drill, names, w)
        every = torch.ones_like(drill, dtype=torch.bool)
        assert float(teach.terms(logits, lab, valid, drill, names, w, every)[0]) == pytest.approx(float(full))
        half = torch.arange(len(drill)) % 2 == 0
        a, s_a = teach.terms(logits, lab, valid, drill, names, w, half)
        b, _ = teach.terms(logits, lab, valid, drill, names, w, ~half)
        assert 0 < float(a) < float(full) and float(a + b) == pytest.approx(float(full), rel=1e-5)
        none, s_none = teach.terms(logits, lab, valid, drill, names, w, torch.zeros_like(every))
        assert float(none) == 0.0
        # the log: agreement over all the drill's units, labelled or not
        assert s_none["kiting"][:6] == s_full["kiting"][:6] and s_none["kiting"][6] == 0.0
        assert s_full["kiting"][6] == s_full["kiting"][3] and 0 < s_a["kiting"][6] < s_full["kiting"][3]
        out = teach.summary(s_a, w, {"kiting": 0.5})["kiting"]
        assert out["share"] == 0.5 and 0 < out["labelled"] < 1 and out["units"] == int(s_full["kiting"][3])

    def test_a_run_with_the_auto_teacher_follows_the_evaluations(self, tmp_path, monkeypatch):
        """run.train with --drill-teach auto: the first shares from the prior, a hook's observe() changes
        them for the next updates, the log has the share and the labelled share; test5 reports the table."""
        from tools.nn.train import run, teach_auto, test5
        actor, crit = nets()
        init = checkpoint.save(tmp_path / "init.pt", actor, crit)
        monkeypatch.setattr(checkpoint, "DIR", tmp_path)
        monkeypatch.setattr(checkpoint, "RANDOM", init)
        base = ["--init", str(init), "--battles", "16", "--steps", "4", "--updates", "4", "--minutes", "3",
                "--device", "cpu", "--no-eval", "--mix", '{"nearest": 1.0}', "--drills", "0.5",
                "--drill-weights", '{"kiting": 1}', "--drill-bank", "16", "--bank", "8", "--max-units", "4",
                "--print-every", "1", "--drill-embed", "0", "--drill-teach", "auto", "--drill-teach-cap", "1.0", "--drill-teach-k", "1.0"]
        args = run.parser().parse_args(["--name", "auto"] + base)
        prior = {"kiting": {"win_rate": 0.0, "scripts": {"skilled": {"win_rate": 1.0}}},
                 "counter": {"win_rate": 0.0, "scripts": {"skilled": {"win_rate": 1.0}}}}
        auto = teach_auto.Auto(args.drill_teach_k, args.drill_teach_cap, args.drill_teach_weight, prior, "test")
        seen = []

        def hook(actor_, critic_, minute, update):
            seen.append(update)
            if update == 2:                                       # the network matches the script now
                auto.observe({"kiting": {"win_rate": 1.0, "scripts": {"skilled": {"win_rate": 1.0}}}})
        _, summary, out = run.train(args, (1e-6, hook), teacher=auto)
        assert auto.names == ("kiting",)                           # the READY drills the run plays only
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        k = [r["teach"]["kiting"] for r in rows]
        assert [x["weight"] for x in k] == [0.15] * 4 and seen[:2] == [1, 2]
        assert k[0]["share"] == 1.0 and k[0]["labelled"] == 1.0 and k[0]["units"] > 0
        assert k[-1]["share"] == 0.0 and k[-1]["labelled"] == 0.0 and k[-1]["units"] > 0
        # the run alone: the prior from the init's files (none here: the default)
        _, _, out2 = run.train(run.parser().parse_args(["--name", "auto2"] + base))
        first = json.loads((out2 / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])["teach"]["kiting"]
        assert first["share"] == 0.5                               # half the cap (1.0)
        # test5's blocks
        rep = test5.teacher(out, 0, 4)
        assert 0 <= rep["kiting"]["labelled"] <= 1
        point = {"teach_auto": auto.history[-1], "teach": rep}
        lines = test5.auto_block([{"teach_auto": auto.history[0]}, point], ["min 0", "min 5"])
        assert "min 0:" in lines and "min 5:" in lines
        assert any(x.startswith("| kiting | 1.000 | 1.000 | 0.000 | 1.00 → 0.00 |") for x in lines)
        assert test5.auto_block([{}, {}], ["a", "b"]) == []
        # a run with no READY drill
        bad = run.parser().parse_args(["--name", "bad"] + [x for x in base] + ["--drills", "0"])
        with pytest.raises(SystemExit):
            run.train(bad)
