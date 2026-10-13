"""The ability price's data (tools/nn/abilities_price/collect.py): the open-loop pair with the recorder, the trajectory,
and layer 2's masks against the simulator's own.

torch is not in the project's .venv: skipped there, run in the training container.
"""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.nn.abilities_price import ability as abl  # noqa: E402
from tools.nn.abilities_price import bonuses as bz  # noqa: E402
from tools.nn.abilities_price import collect as co  # noqa: E402
from tools.nn.sim import abilities as sab  # noqa: E402
from tools.nn.train import ability_branch as ab  # noqa: E402

from tests.tools.test_nn_ability_branch import grabbed, pick  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PASSPORTS = json.loads((ROOT / "config/nn/abilities.json").read_text(encoding="utf-8"))["abilities"]


def test_one_slot_is_ability_branch_play_pair_exactly_and_each_changes_nothing():
    actor, env, cand = grabbed()
    snap = env.brancher.got[0]
    b, side, unit, slot = pick(cand)
    want = ab.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 6, seed=5, openloop=True)
    rec = co.Recorder(ab.select(snap, b), unit, side, slot, 2, env.params)
    d, bg, other, at = co.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 6, seed=5, each=rec.each,
                                    marks=(3, 6))
    assert torch.equal(d, want[0]) and torch.equal(bg, want[1])
    assert at.shape == (len(b), 2, 2) and torch.equal(at[:, :, -1], d)      # the price at the marks, the last = d
    d0, _, other0, _ = co.play_pair(env, actor, ab.select(snap, b), unit, side, slot, 2, 6, seed=5, one_slot=False)
    # all slots (the training's old way): differs from one slot only where the pressed branch ordered another slot
    same = (other0 == 0).all(1)
    assert torch.equal(d0[same], d[same])
    r = rec.result(len(b))
    assert r["x"].shape == (len(b), 2, 7, env.N // 2) and r["on"].shape == (len(b), 2, 7)
    assert (r["on"][:, :, 0] == 0).all() and (r["on"][:, :, 1] > 0).any()        # the press: its timer runs


def test_the_layer2_masks_are_the_simulators():
    """Who gets the bonus by ability.py (positions, timers, the passport) against the simulator's own record
    (abilities.RECEIVED), on the pressed branch's every decision."""
    actor, env, cand = grabbed()
    snap = env.brancher.got[0]
    b, side, unit, slot = pick(cand)
    K, D = 1, 20
    rec = co.Recorder(ab.select(snap, b), unit, side, slot, K, env.params)
    sab.RECEIVED = []
    try:
        co.play_pair(env, actor, ab.select(snap, b), unit, side, slot, K, D, seed=3, each=rec.each)
        got = list(sab.RECEIVED)
    finally:
        sab.RECEIVED = None
    names = sab.keys(env.params)
    r = rec.result(len(b))
    bricks = bz.bricks()
    abilities = abl.abilities(PASSPORTS, bricks)
    H = env.N // 2
    agree = total = 0
    for p in range(len(b)):
        key = names[int(r["key"][p])]
        a = abilities[key]
        tr = {k: r[k][p, 0].astype(bool if k in bz.FLAGS else float) for k in co.FLAGS + co.FLOATS + ("on",)}
        tr.update(owner=int(r["owner"][p]) if "owner" in r else int(unit[p]) - (int(side[p]) - 1) * H, dt_s=1.0,
                  mmax=r["mmax"][p, 0])
        m = a.masks(tr)
        mine = m["self"] | m["friends"]
        own = slice((int(side[p]) - 1) * H, int(side[p]) * H)
        for i in range(D):
            # the pressed branch's last simulator step of decision i (pressed, then not: 2 x k steps a decision)
            sim = got[(2 * i) * env.k + env.k - 1].get(key)
            sim = sim[p, own].cpu().numpy() if sim is not None else np.zeros(H, dtype=bool)
            agree += int((sim == mine[i + 1]).sum())
            total += H
    assert agree / total >= 0.97, agree / total
