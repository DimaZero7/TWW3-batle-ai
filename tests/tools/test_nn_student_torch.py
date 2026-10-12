"""The game AI's student (tools/nn/student) with torch: the commitment replayed, the streams, the loss, a tiny run,
the evaluation and the pool taking the student's checkpoint (run in the training container)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.tools.test_nn_observe_torch import TINY, demo_arrays  # noqa: E402
from tools.nn.model import policy  # noqa: E402
from tools.nn.sim.orders import ATTACK, KEEP, MOVE  # noqa: E402
from tools.nn.student import dataset as ds, eval as sev, objective as obj, train as strain  # noqa: E402
from tools.nn.train import checkpoint, league  # noqa: E402


U = 3      # the synthetic side's unit 0 takes no orders (its lord): observe's demo labels on it go to unit 3


def student_demo(T=8):
    """observe's demo side with the labels of unit 0 on unit U: attacks of U at 1 and 5, a running move of unit 1 at
    2, a hold of unit 2 at 3."""
    a = demo_arrays(T)
    for k in ("kind", "target", "point", "run", "run_known", "alt"):
        a[k][:, U] = a[k][:, 0]
        a[k][:, 0] = -1 if a[k].dtype.kind == "i" else False
    return a


def write_battle(path, T=8, sides=(1, 2)):
    arrays = student_demo(T)
    meta = {"run": path.stem, "sides": list(sides), "T": T, "N": int(arrays["own"].shape[1]), "path": "",
            "cfg": {"sectors": TINY.sectors, "fine": TINY.fine, "grid_half": TINY.grid_half}, "per_side": {}}
    np.savez_compressed(path, meta=np.asarray(json.dumps(meta)),
                        **{f"s{s}_{k}": v for s in sides for k, v in arrays.items()})
    return arrays


def test_the_replay_holds_a_unit_after_its_order_and_teaches_keep_to_the_free_ones(tmp_path):
    a = student_demo(8)
    lab = ds.student_labels(a)
    free, cin, kind, dropped = ds.replay(a, lab)
    assert dropped == 0 and lab["commit"][1, U] == 1                    # attacks at 1 and 5: kept 4 s
    assert kind[1, U] == ATTACK and kind[5, U] == ATTACK and kind[2, 1] == MOVE
    assert (kind[2:5, U] == -1).all() and cin[3, U, 1] == 1             # held between them: nothing taught
    ctrl = a["ctrl"]
    assert ((kind == KEEP) <= (free & ctrl)).all() and (kind[~ctrl] == -1).all()
    assert kind[0, 1] == KEEP and kind[3, 2] == 0 and kind[4, 2] == -1   # no order: keep; the hold, then held
    a["kind"][2, U], a["point"][2, U] = MOVE, a["point"][2, 1]          # one second after the attack: inside 2 s
    lab = ds.student_labels(a)
    free, cin, kind, dropped = ds.replay(a, lab)
    assert lab["commit"][1, U] == 0 and dropped == 1 and kind[2, U] == -1 and not free[2, U]


def test_the_streams_give_every_second_once_and_reset_the_memory_at_a_new_battle(tmp_path):
    for i, T in enumerate((8, 9, 10)):
        write_battle(tmp_path / f"b{i}.npz", T, sides=(1,))
    sides = ds.discover([tmp_path])
    st = ds.Streams(sides, B=2, T=4, shuffle=True, seed=3)
    seen, resets, refs = [], 0, []
    for segs in st:
        b = ds.to_batch(segs)
        assert b["obs"]["tokens"].shape[:2] == (4, 2) and b["kind"].shape == (4, 2, st.N)
        assert b["obs"]["geo"].shape == (4, 2, 8) and b["obs"]["ctx"].shape[:2] == (4, 2)
        resets += int(b["reset"].sum())
        refs += [a for row in segs for (a, _, _, _) in row]           # (alive: their ids stay theirs)
        seen += [(id(a), int(k)) for row in segs for (a, rows, _, v) in row if v for k in rows]
        assert bool((b["kind"][~b["valid"]] == -1).all())
        # a stream's seconds in order; the memory starts afresh exactly where a side starts
        for j, row in enumerate(segs):
            t = np.concatenate([a["t"][rows] for (a, rows, _, _) in row])
            starts = np.concatenate([(rows == 0) & v for (_, rows, _, v) in row])
            assert np.array_equal(b["reset"][:, j].numpy(), starts) and len(t) == 4
    assert len(seen) == 27 == len(set(seen)) and resets == 3


def test_the_loss_trains_and_the_accuracies_are_shares(tmp_path):
    write_battle(tmp_path / "b0.npz", 8)
    sides = ds.discover([tmp_path])
    torch.manual_seed(0)
    actor = policy.Actor(TINY).train()
    st = ds.Streams(sides, B=2, T=8, shuffle=False)
    b = ds.to_batch(next(iter(st)))
    h = actor.initial({k: v[0] for k, v in b["obs"].items()})
    loss, s, h2, aux = obj.run(actor, b, h)
    assert torch.isfinite(loss) and h2.shape == h.shape and not h2.requires_grad
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in actor.parameters())
    sm = obj.summary(s)
    assert sm["new"] == 2 * 4 and 0 <= sm["kind_acc"] <= 1 and 0 <= sm["keep_acc"] <= 1   # 4 orders a side
    assert sm["labels"] > sm["new"] and sm["place"] == 2 and sm["attack"] == 4
    opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    first = float(loss.detach())
    for _ in range(30):
        loss, _, _, _ = obj.run(actor, b, h, stats=False)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first * 0.5


def test_a_tiny_run_writes_a_checkpoint_the_pool_takes_and_eval_reads(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    for i in range(6):
        write_battle(data / f"b{i}.npz", 8)
    torch.manual_seed(0)
    init = checkpoint.save(tmp_path / "tiny.pt", policy.Actor(TINY), None, "v2")
    out = tmp_path / "out"
    assert strain.main(["--name", "t", "--scratch", str(init), "--data", str(data), "--out", str(out), "--epochs", "2",
                        "--B", "2", "--T", "4", "--val-share", "0.4", "--device", "cpu"]) == 0
    best = out / "best.pt"
    assert best.exists() and (out / "latest.pt").exists() and len((out / "log.jsonl").read_text().splitlines()) == 2
    # the pool of past versions takes it as it is (run.py --pool-extra)
    pool = league.Pool(tmp_path / "pool")
    pool.add(best)
    path, _, _ = pool.sample()
    data_ = checkpoint.read(path)
    past = policy.Actor(checkpoint.config_of(data_))
    past.load_state_dict(data_["actor"])
    # the evaluation and the baselines on the held-out sides
    actor = checkpoint.load_policy(best)
    sides = ds.split(ds.discover([data]), 0.4)["val"]
    sm, err = sev.accuracy(actor, sides, B=2, T=4)
    assert sm["new"] > 0 and len(err) == sm["place"]
    base = sev.baselines(sides, actor.cfg)
    assert set(base) == {"repeat", "nearest", "majority"} and base["majority"][0]["kind_acc"] > 0


def test_the_cell_of_a_point_and_its_centre_agree():
    cfg = TINY
    f, l = np.array([-799.0, 0.0, 12.0, 799.0]), np.array([0.0, -400.0, 13.0, 799.0])
    p = sev.cell_at(cfg, f, l)
    cf, cl = sev.cell_centre(cfg, p)
    assert np.all(np.abs(cf - f) <= 12.5 + 1e-6) and np.all(np.abs(cl - l) <= 12.5 + 1e-6)
    assert sev.point_error(cfg, p[:1], p[:1])[0] == 0
