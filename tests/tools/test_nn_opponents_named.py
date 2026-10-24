"""run.py --opponents (league.parse_opponents, named_layout; rollout.Battles.fix_slots, take_named): every opponent by
name at a hard share of the batch's battles - scripts, self and checkpoints, each checkpoint in a past slot of its own
for the whole run; the log by opponent and role, the big checkpoints by matchup. torch: the training container."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.model import config, critic, policy             # noqa: E402
from tools.nn.train import checkpoint, league, randomise, rollout   # noqa: E402

CFG = config.preset("small", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2)
SHARES = {"self": 0.35, "build/v2/itS27_4.pt": 0.25, "build/nn-train/test5/x2_exploit_general/m40.pt": 0.23,
          "build/v2/itS26_15.pt": 0.04, "build/v2/itS25_1.pt": 0.04, "build/v2/itS26_5.pt": 0.04, "ai_like": 0.02,
          "nearest": 0.01, "human": 0.02}


def nets(seed=0):
    torch.manual_seed(seed)
    return policy.Actor(CFG).eval(), critic.Critic(CFG).eval()


def test_the_names_the_shares_and_their_checks():
    assert league.parse_opponents(json.dumps(SHARES)) == SHARES
    assert league.parse_opponents("self:0.5,ai_like:0.5") == {"self": 0.5, "ai_like": 0.5}
    assert league.opponent_key("build/v2/itS27_4.pt") == "itS27_4"
    assert league.opponent_key("build/nn-train/test5/x2_exploit_general/m40.pt") == "x2_exploit_general_m40"
    assert league.opponent_key("ai_like") == "ai_like" and league.is_checkpoint("a/b.pt")
    for bad in ('{"self": 0.5, "ai_like": 0.4}', '{"self": 0.5, "past": 0.5}', '{"self": 1.0, "ai_like": 0}',
                '{"a/m40.pt": 0.5, "b/x/a_m40.pt": 0.5}', ""):
        with pytest.raises(ValueError):
            league.parse_opponents(bad)


@pytest.mark.parametrize("B", [100, 256, 1000, 37])
def test_the_hard_shares_hold_within_one_battle(B):
    nl = league.named_layout(B, 6, SHARES)
    assert nl.layout.B == B and sum(nl.rows.values()) == B
    for name, share in SHARES.items():
        assert abs(nl.rows[name] - share * B) <= 1.0                        # +-1 battle (per 100: +-1)
    past = nl.layout.opponent == league.CODE["past"]
    assert (past == (nl.row_slot >= 0)).all()
    ck = [n for n in SHARES if league.is_checkpoint(n)]
    assert nl.checkpoints == ck
    for k, n in enumerate(ck):
        assert int((nl.row_slot == k).sum()) == nl.rows[n]
    assert int((nl.layout.opponent == league.CODE["self"]).sum()) == nl.rows["self"]
    assert int((nl.layout.opponent == league.CODE["human"]).sum()) == nl.rows["human"]
    # the learner's side cycles within each opponent's rows
    rows = np.nonzero(nl.row_slot == 0)[0]
    if len(rows) >= 12:
        assert set(nl.layout.learner[rows].tolist()) == {1, 2}


def test_each_checkpoints_rows_keep_its_slot_and_are_counted_by_name():
    nl = league.named_layout(8, 1, {"a/one.pt": 0.5, "b/two.pt": 0.25, "nearest": 0.25})
    env = rollout.Battles(nl.layout, [("arena", "attack")], params=rollout.params_with_limit(4.0),
                          spread=randomise.NONE, past_slots=2, compile=False)
    env.set_past(nets(1)[0], slot=0)
    env.set_past(nets(2)[0], slot=1)
    env.fix_slots(nl.row_slot)
    want = torch.as_tensor(np.where(nl.row_slot >= 0, nl.row_slot, 0))
    assert torch.equal(env.battle_slot.cpu(), want) and env.slot_used == {0, 1}
    learner, crit = nets()
    for _ in range(14):                                        # 4 s: every battle ends and starts again
        env.step(learner, crit)
    assert torch.equal(env.battle_slot.cpu(), want)            # the same slot after the restarts
    stats = env.take_stats()
    named, cells = env.take_named(["one", "two"])
    g = {n: sum(named.get(f"{n}/{r}", (0,))[0] for r in ("attack", "defend")) for n in ("one", "two")}
    past = sum(stats.get(f"past/{r}", (0,))[0] for r in ("attack", "defend"))
    assert g["one"] >= 4 and g["two"] >= 2 and g["one"] + g["two"] == past
    assert sum(v[0] for v in cells["one"].values()) == g["one"]
    assert all(k.split("/")[0] in ("attack", "defend") for k in cells["one"])
    assert env.take_named(["one", "two"]) == ({}, {})


def test_a_run_with_named_opponents_logs_them_by_name(tmp_path, monkeypatch):
    from tools.nn.train import run
    actor, crit = nets()
    init = checkpoint.save(tmp_path / "init.pt", actor, crit)
    (tmp_path / "x2").mkdir()
    a = checkpoint.save(tmp_path / "x2" / "m40.pt", *nets(7))
    b = checkpoint.save(tmp_path / "old.pt", *nets(8))
    monkeypatch.setattr(checkpoint, "DIR", tmp_path)
    monkeypatch.setattr(checkpoint, "RANDOM", checkpoint.save(tmp_path / "random.pt", *nets(3)))
    opp = json.dumps({"self": 0.25, str(a): 0.25, str(b): 0.25, "nearest": 0.25})
    base = ["--init", str(init), "--battles", "8", "--steps", "4", "--updates", "3", "--minutes", "3", "--device",
            "cpu", "--no-eval", "--limit", "6", "--bank", "8", "--max-units", "4", "--critic-warmup", "0"]
    _, trained, out = run.train(run.parser().parse_args(["--name", "named", "--opponents", opp] + base))
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    games = {}
    for r in rows:
        assert set(r["opponents"]) == {"self", "x2_m40", "old", "nearest"}
        for k, g in r["games"].items():
            games[k] = games.get(k, 0) + g
    assert not any(k.startswith(("past/", "pinned/")) for k in games)
    assert any(k.startswith("x2_m40/") for k in games) and any(k.startswith("old/") for k in games)
    assert trained["opponent_shares"] == {"self": 0.25, "x2_m40": 0.25, "old": 0.25, "nearest": 0.25}
    assert set(trained["opponents"]) == {"self", "x2_m40", "old", "nearest"}
    assert trained["opponents"]["x2_m40"]["all"]["games"] == sum(v for k, v in games.items() if k.startswith("x2_m40/"))
    assert set(trained["opponent_matchups"]) <= {"x2_m40", "old"} and trained["opponent_matchups"]
    q = json.loads((out / "pool" / "q.json").read_text(encoding="utf-8"))
    assert sum(e["games"] for e in q["entries"]) == 0                  # the pool draws nothing
    with pytest.raises(SystemExit):                                    # not with --mix
        run.train(run.parser().parse_args(["--name", "n2", "--opponents", opp, "--mix", '{"self": 1.0}'] + base))
    with pytest.raises(SystemExit):                                    # a checkpoint that is not there
        run.train(run.parser().parse_args(["--name", "n3", "--opponents", '{"self": 0.5, "no/such.pt": 0.5}'] + base))
