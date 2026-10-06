"""Embedded drill frames (drills.EMBED, Drill.embedded) and the teacher in normal battles (run.py --teach-normal;
rollout.Battles teach_normal; docs/en/training/training.md "Drills", "Embedded frames").

torch: the training container. Level 1: the frame draw (EMBED 0 leaves every battle as it was; the default
embeds kiting and hold_fire); the embedded battle is a normal generated one (both lords, the arena's
deployment, our side 2 half-turned onto side 2's zone, even gold) with the drill's tagged units placed as the
situation needs. Level 2: the scripts in an embedded battle (ai_like for untagged units, the drill's rule for
the tagged ones); the rollout labels an embedded drill battle only at the drill's moments and a normal
battle's units only at the moments, under "<drill>@normal". Level 3: a CPU run with --teach-normal logs the
normal teacher (fixed and adaptive shares).
"""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, critic, policy  # noqa: E402
from tools.nn.sim import battle, scenario  # noqa: E402
from tools.nn.sim import orders as O  # noqa: E402
from tools.nn.sim.params import load  # noqa: E402
from tools.nn.train import checkpoint, league, opponents, randomise, rollout, teach_auto  # noqa: E402
from tools.nn.train import drills as D  # noqa: E402
from tools.nn.train.drills import hold_fire as H  # noqa: E402
from tools.nn.train.drills import kiting as K  # noqa: E402
from tools.nn.train.drills import source as ds  # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
SEEDS = range(24)


@pytest.fixture(autouse=True)
def default_shares(monkeypatch):
    """The drills' frame shares at their defaults (run.train sets them from its arguments: another test's run
    may have left them changed)."""
    monkeypatch.setattr(D, "EMBED", D.EMBED_DEFAULT)
    monkeypatch.setattr(D, "BROAD", D.BROAD_DEFAULT)


def _side_x(desc, side):
    return np.mean([u["x"] for u in desc["sides"][side]["units"]])


class TestFrame:
    def test_embed_zero_leaves_every_battle_as_it_was_and_the_default_embeds(self):
        from tools.nn.train import run
        assert run.parser().parse_args([]).drill_embed == 1.0 and run.parser().parse_args([]).teach_normal == ""
        assert D.EMBED == 1.0 and set(D.TRAIN) == {"kiting", "hold_fire"} and "counter" in D.READY
        for drill in (K.DRILL, H.DRILL):
            old = D.battles(drill, range(12), broad=0.5, embed=0.0)
            assert all(d["frame"] in ("clean", "broad") for d, _ in old)
            assert all(d["frame"] == "embedded" for d, _ in D.battles(drill, range(6)))
            half = D.battles(drill, range(40), broad=0.5, embed=0.5)
            n = sum(d["frame"] == "embedded" for d, _ in half)
            assert 8 < n < 32
            for (a, sa), (b, sb) in zip(old, half[:12]):
                if b["frame"] != "embedded":
                    assert a == b and sa == sb
        from tools.nn.train.drills import counter
        assert all(d["frame"] != "embedded" for d, _ in D.battles(counter.DRILL, range(6)))

    @pytest.mark.parametrize("drill", [K.DRILL, H.DRILL], ids=["kiting", "hold_fire"])
    def test_an_embedded_battle_is_a_normal_one_with_the_situation_inserted(self, drill):
        for i, (desc, ours) in enumerate(D.battles(drill, SEEDS)):
            assert desc["frame"] == "embedded" and not desc["broad"]
            for s in (1, 2):
                units = desc["sides"][s]["units"]
                assert sum(bool(u.get("general")) for u in units) == 1 and len(units) <= D.MAX_UNITS + 1
            # the arena's deployment: our army where a normal battle's side `ours` stands (side 1 west, 2 east)
            ours_x = np.mean([u["x"] for u in desc["sides"][ours]["units"] if not u.get("tag")])
            assert (ours_x < 0) == (ours == 1)
            assert D.imbalance(desc) <= D.EMBED_TOLERANCE + 1e-9
            mine = [u for u in desc["sides"][ours]["units"] if u.get("tag") == D.TAG_OURS]
            theirs = [u for u in desc["sides"][3 - ours]["units"] if u.get("tag") == D.TAG_ENEMY]
            assert mine and theirs
            assert not any(u.get("tag") == D.TAG_ENEMY for u in desc["sides"][ours]["units"])
        if drill is K.DRILL:
            for desc, ours in D.battles(drill, SEEDS):
                mine = [u for u in desc["sides"][ours]["units"] if u.get("tag")]
                chasers = [u for u in desc["sides"][3 - ours]["units"] if u.get("tag")]
                assert {u["key"] for u in mine} == {K.OURS[0]} and desc["sides"][ours]["faction"] == K.SKAVEN
                assert all(K.OURS[1] - r >= K.MIN_GAP_MS for k, r in K.CHASERS if k in {u["key"] for u in chasers})
                # the kiters and the chasers on the same flank (the same side of the battle's axis)
                lat = lambda us: np.mean([u["z"] for u in us])
                assert np.sign(lat(mine)) == np.sign(lat(chasers))
                d = math.hypot(np.mean([u["x"] for u in mine]) - np.mean([u["x"] for u in chasers]),
                               lat(mine) - lat(chasers))
                assert 250.0 < d < 500.0
        else:
            for desc, ours in D.battles(drill, SEEDS):
                lord = next(u for u in desc["sides"][3 - ours]["units"] if u.get("general"))
                inf = next(u for u in desc["sides"][ours]["units"] if u.get("tag") and u["key"] in H.OURS_MELEE[desc["sides"][ours]["faction"]])
                assert lord.get("tag") == D.TAG_ENEMY and math.hypot(lord["x"] - inf["x"], lord["z"] - inf["z"]) < 15.0

    def test_balance_drops_untagged_units_of_the_richer_side(self):
        desc = D.army([D.unit(K.OURS[0], 0, 0, 0, 30, tag=D.TAG_OURS), D.unit("wh2_main_skv_inf_skavenslaves_0", 0, 0, 0, 30),
                       D.unit("wh2_main_skv_inf_clanrats_1", 0, 0, 0, 30)],
                      [D.unit(K.SWORDS[0], 0, 0, 0, 30, tag=D.TAG_ENEMY)], 1)
        # 450 + 125 + 350 v 375: the clanrats go (diff 550 -> 200), then the slaves (-> 75, within 20 % of 450)
        assert D.balance(desc, 0.2) is True
        assert [u["key"] for u in desc["sides"][1]["units"]] == [K.OURS[0]]
        assert D.balance(desc, 0.1) is False                  # only the tagged unit is left: nothing to drop


def _state(pairs):
    st = scenario.build([d for d, _ in pairs], per_side=D.MAX_UNITS + 1)
    return st, torch.tensor([o for _, o in pairs])


class TestScripts:
    def test_embedded_scripts(self):
        pairs = D.battles(K.DRILL, range(4))
        st, ours = _state(pairs)
        assert bool(D.embedded_rows(st).all())
        tag = st.u["tag"]
        ai = opponents.ai_like(st)
        en = K.enemy(st)
        rest = (tag == 0) & (st.u["side"] > 0)
        assert bool((en.kind[rest] == ai.kind[rest]).all())
        chasers = tag == D.TAG_ENEMY
        # far from our kiters at the start: the chasers march in ai_like's line
        assert bool((en.kind[chasers] == ai.kind[chasers]).all())
        # a kiter within EMB_CHASE_M of them: they chase it
        u = st.u
        for b in range(st.B):
            k = int((tag[b] == D.TAG_OURS).nonzero()[0])
            for j in (tag[b] == D.TAG_ENEMY).nonzero().flatten().tolist():
                u["x"][b, j], u["z"][b, j] = u["x"][b, k] + 100.0, u["z"][b, k]
        en = K.enemy(st)
        assert bool((en.kind[chasers] == O.ATTACK).all())
        assert bool((tag.gather(1, en.target.clamp(min=0))[chasers] == D.TAG_OURS).all())
        st, ours = _state(pairs)
        nv = K.naive(st)
        assert bool((nv.kind[tag == D.TAG_OURS] == O.HOLD).all())
        assert bool((nv.kind[rest] == ai.kind[rest]).all())
        params = load()
        for script in (K.naive, K.skilled):
            st, ours = _state(pairs)
            for _ in range(20):
                orders = D.merged(st, ours, script(st), K.enemy(st))
                O.check(orders, st.N)
                battle.step(st, orders, params, params.dt)
            assert bool(torch.isfinite(st.u["x"]).all())
        hp = D.battles(H.DRILL, range(2))
        st, _ = _state(hp)
        o = H.enemy(st)
        lord = st.u["lord"] & (st.u["tag"] == D.TAG_ENEMY)
        assert bool((o.kind[lord] == O.ATTACK).all())

    def test_a_clean_battle_plays_as_before(self):
        pairs = D.battles(K.DRILL, range(4), broad=0.0, embed=0.0)
        st, _ = _state(pairs)
        assert not bool(D.embedded_rows(st).any())
        for a, b in ((K.enemy(st), K.chase(st)), (K.skilled(st), K.kite(st)), (H.skilled(st), H.spare(st))):
            assert all(bool(torch.equal(getattr(a, f), getattr(b, f))) for f in O.FIELDS)


KITING = league.CODE[D.opponent("kiting")]


def nets(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(CFG).eval(), critic.Critic(CFG).eval()


def env_of(B, other, embed, teach=True, normal=False):
    lay = league.Layout(np.zeros(B + other, dtype=int), 1 + np.arange(B + other) % 2,
                        np.array([KITING] * B + [league.CODE["nearest"]] * other))
    was = D.EMBED
    D.EMBED = embed
    try:
        src = ds.Mixed(np.arange(other), 6, None, "cpu", lay, per_drill=max(B, 2), drills={"kiting": K.DRILL}, seed=1)
    finally:
        D.EMBED = was
    return rollout.Battles(lay, device="cpu", source=src, compile=False, spread=randomise.NONE,
                           teach={"kiting": K.DRILL} if teach else None,
                           teach_normal={"kiting": K.DRILL} if normal else None)


class TestTeacher:
    def test_an_embedded_drill_battle_is_labelled_at_the_moments_only(self, monkeypatch):
        env = env_of(4, 0, 1.0)
        seen = []
        real = K.moments

        def spy(st, o):
            m = real(st, o)
            seen.append(m)
            return m
        monkeypatch.setattr(env, "teach", {k: (i, f, spy) for k, (i, f, _) in env.teach.items()})
        actor, crit = nets()
        a, frame, _ = env.observe(False)
        obs_r = rollout.rows_of(a, env.rows_learn)
        lab, valid, drill, picked = env._teach_labels(actor.cfg, obs_r, frame)
        at = seen[-1][env.rows_learn % env.B]
        assert drill.shape == valid.shape and bool((drill == 0).all())
        assert not bool((valid & ~at).any())                     # labels only at the moments in an embedded battle

    def test_the_teacher_in_normal_battles_labels_normal_rows_at_the_moments(self):
        env = env_of(2, 6, 1.0, teach=False, normal=True)
        assert env.teach_names == ("kiting@normal",) and bool(env.row_normal.sum() == 6 * 1)
        # a normal battle: put an enemy melee unit near our first missile unit, closing in
        u = env.st.u
        hit = False
        for b in range(2, env.B):
            ours = int(env.layout.learner[b])
            mine = ((u["side"][b] == ours) & (u["range"][b] > 0)).nonzero().flatten()
            foes = ((u["side"][b] == 3 - ours) & (u["range"][b] <= 0) & ~u["lord"][b]).nonzero().flatten()
            if len(mine) and len(foes):
                i, j = int(mine[0]), int(foes[0])
                u["run"][b, i] = u["run"][b, j] + 2.0                  # ours the faster
                u["x"][b, j], u["z"][b, j] = u["x"][b, i] + 30.0, u["z"][b, i]
                u["vx"][b, j] = -3.0
                hit = True
        assert hit
        actor, crit = nets()
        a, frame, _ = env.observe(False)
        obs_r = rollout.rows_of(a, env.rows_learn)
        lab, valid, drill, picked = env._teach_labels(actor.cfg, obs_r, frame)
        sit = K.moments(env.st, K.kite(env.st))[env.rows_learn % env.B]
        assert bool(valid.any()) and not bool((valid & ~sit).any()) and not bool((valid & ~env.row_normal[:, None]).any())
        assert bool((drill[valid] == 0).all()) and bool((drill[~valid] == -1).all())
        env.set_teach_shares({"kiting@normal": 0.0})
        assert not bool(env._teach_labels(actor.cfg, obs_r, frame)[3].any())
        env.decisions = 1                                            # not a probe decision: the script is not run
        assert not bool(env._teach_labels(actor.cfg, obs_r, frame)[1].any())
        batch = rollout.collect(env, actor, crit, 2)
        assert batch["teach"]["names"] == ("kiting@normal",) and batch["teach"]["drill"].shape == batch["teach"]["valid"].shape

    def test_runs_with_the_teacher_in_normal_battles(self, tmp_path, monkeypatch):
        from tools.nn.train import run
        actor, crit = nets()
        init = checkpoint.save(tmp_path / "init.pt", actor, crit)
        monkeypatch.setattr(checkpoint, "DIR", tmp_path)
        monkeypatch.setattr(checkpoint, "RANDOM", init)
        base = ["--init", str(init), "--battles", "8", "--steps", "4", "--updates", "2", "--minutes", "3",
                "--device", "cpu", "--no-eval", "--mix", '{"nearest": 1.0}', "--bank", "8", "--max-units", "6",
                "--print-every", "1"]
        _, _, out = run.train(run.parser().parse_args(["--name", "nfix"] + base + ["--teach-normal", '{"kiting": 1.0}']))
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(r["teach"]["kiting@normal"]["weight"] == teach_auto.NORMAL_WEIGHT for r in rows)
        assert all(r["teach"]["kiting@normal"]["share"] == 1.0 for r in rows)
        prior = {"kiting": {"network": {"share": 0.0}, "ai_like": {"share": 0.7}},
                 "hold_fire": {"network": {"share": 0.84}, "ai_like": {"share": 0.39}}}
        normal = teach_auto.Transfer(prior=prior, source="test")
        _, _, out = run.train(run.parser().parse_args(["--name", "nauto"] + base + ["--teach-normal", "auto"]),
                              normal=normal)
        rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows[0]["teach"]["kiting@normal"]["share"] == teach_auto.NORMAL_CAP
        assert rows[0]["teach"]["hold_fire@normal"]["share"] == 0.0
        with pytest.raises(SystemExit):
            run.train(run.parser().parse_args(["--name", "bad"] + base + ["--teach-normal", "counter"]))
