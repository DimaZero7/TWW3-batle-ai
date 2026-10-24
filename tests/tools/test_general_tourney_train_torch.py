"""The learned group collection's training branch (tools/nn/general/tourney_train.py) and its graft into another
general (graft.py). torch: the training container, on the CPU."""
import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import executor                          # noqa: E402
from tools.nn.model import policy                              # noqa: E402

from tests.tools.test_general_tourney_torch import GCFG, TINY, nets   # noqa: E402


def test_the_branch_trains_only_the_collection(tmp_path):
    """tourney_train: two rounds on tiny networks - only general.assign. and general.k_head. move, the log and the
    checkpoints are written, --resume goes on from the round saved."""
    from tools.nn.general import tourney_train
    from tools.nn.model import critic as mcritic
    from tools.nn.train import checkpoint
    actor, enemy = nets(2)
    torch.manual_seed(3)
    critic = mcritic.Critic(TINY)
    init, foe = tmp_path / "init.pt", tmp_path / "foe.pt"
    checkpoint.save(init, actor, critic)
    checkpoint.save(foe, enemy)
    out = tmp_path / "t"
    argv = ["--init", str(init), "--enemy", str(foe), "--out", str(out), "--rounds", "2", "--states", "2", "--ks", "2,3",
            "--copies", "2", "--limit-s", "40", "--armies", "v2", "--device", "cpu", "--minibatch", "16",
            "--save-every", "1"]
    tourney_train.main(argv)
    before = checkpoint.read(init)["actor"]
    after = checkpoint.read(out / "latest.pt")["actor"]
    moved = {k.split(".")[1] for k in before if not torch.equal(before[k], after[k])}
    assert moved == {"assign", "k_head"}, moved
    assert checkpoint.general_of(checkpoint.read(out / "latest.pt")).groups == "learned"
    assert (out / "r2.pt").exists()
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows if "round" in r] == [0, 1]
    assert all(r["exact_k"] == 1.0 for r in rows if "round" in r)
    tourney_train.main(["--init", str(out / "latest.pt"), "--resume"] + argv[2:6] + ["--rounds", "1"] + argv[8:])
    rows = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows if "round" in r] == [0, 1, 2]


def test_graft_puts_the_collection_into_an_older_general(tmp_path):
    """graft.py: the branch's collection into a general without one (an older checkpoint: no assign., k_head.) - its
    weights are the branch's, every other the target's, groups "learned" with the branch's bounds; it loads."""
    from tools.nn.general import graft
    from tools.nn.train import checkpoint
    src_actor, _ = nets(4)
    with torch.no_grad():
        for n, p in src_actor.named_parameters():
            if n.startswith("general.assign."):
                p.add_(1.0)
    src_actor.set_rules(dataclasses.replace(src_actor.gcfg, assign_hi=3.0))
    torch.manual_seed(5)
    dst_actor = executor.wrap(policy.Actor(TINY).eval(), dataclasses.replace(GCFG, groups="rule")).eval()
    s, d = tmp_path / "s.pt", tmp_path / "d.pt"
    checkpoint.save(s, src_actor)
    checkpoint.save(d, dst_actor)
    old = torch.load(d, weights_only=True)
    old["actor"] = {k: v for k, v in old["actor"].items() if not k.startswith(("general.assign.", "general.k_head."))}
    torch.save(old, d)
    graft.main(["--from", str(s), "--into", str(d), "--out", str(tmp_path / "g.pt")])
    g = checkpoint.load_policy(tmp_path / "g.pt")
    assert g.gcfg.groups == "learned" and g.gcfg.assign_hi == 3.0
    sd, ss, sdst = g.state_dict(), src_actor.state_dict(), dst_actor.state_dict()
    for k in sd:
        ref = ss if k.startswith(("general.assign.", "general.k_head.")) else sdst
        assert torch.equal(sd[k], ref[k]), k
