"""The game AI's student (tools/nn/student): the battles found and split by start, the commitment read from the
gaps between the AI's orders (numpy; the torch parts: test_nn_student_torch.py)."""
import json

import numpy as np

from tools.nn.student import dataset as ds


def write_meta(path, run, sides, human=False, T=10, N=6):
    meta = {"run": run, "sides": sides, "T": T, "N": N, "path": f"build/nowhere/{run}",
            "cfg": {"sectors": 16, "fine": 4, "grid_half": 800.0},
            "per_side": {str(s): {"human": human} for s in sides}}
    np.savez_compressed(path, meta=np.asarray(json.dumps(meta)))


def test_the_commitment_is_the_longest_duration_within_the_gap_to_the_next_order():
    T, N = 40, 3
    kind = np.full((T, N), -1)
    t = np.arange(T, dtype=float)
    kind[0, 0], kind[5, 0], kind[6, 0], kind[30, 0] = 1, 2, 2, 0      # gaps 5, 1, 24, none
    kind[3, 1] = 2                                                     # the only order: the longest
    c = ds.commit_labels(kind, t)
    assert c[0, 0] == 1 and c[5, 0] == 0 and c[6, 0] == 3 and c[30, 0] == 3     # 4 s, 2 s (at least), 16, 16
    assert c[3, 1] == 3 and (c[:, 2] == -1).all() and (c[kind < 0] == -1).all()
    t2 = t * 2                                                         # seconds of the record, not rows
    assert ds.commit_labels(kind, t2)[0, 0] == 2                       # gap 10 s -> 8 s


def test_an_order_repeating_the_last_one_is_keep_and_the_commitment_counts_from_the_real_changes():
    T, N = 30, 3
    kind, target, point = np.full((T, N), -1), np.full((T, N), -1), np.full((T, N), -1)
    run, known = np.zeros((T, N), bool), np.zeros((T, N), bool)
    # unit 0: move to cell 7 at 0, again at 1 and 2 (re-clicks), to cell 8 at 3, back to 7 at 4
    for k, c in ((0, 7), (1, 7), (2, 7), (3, 8), (4, 7)):
        kind[k, 0], point[k, 0], known[k, 0] = ds.MOVE, c, True
    run[2, 0] = True                                                    # ... the one at 2 runs: a new order
    # unit 1: attack 5, attack 5 again, attack 6; then a hold, a hold again
    kind[0, 1], target[0, 1] = ds.ATTACK, 5
    kind[1, 1], target[1, 1] = ds.ATTACK, 5
    kind[2, 1], target[2, 1] = ds.ATTACK, 6
    kind[5, 1] = kind[6, 1] = ds.HOLD
    rep = ds.repeats(kind, target, point, run, known)
    assert rep[:, 0].tolist()[:5] == [False, True, False, False, False]
    assert rep[:, 1].tolist()[:7] == [False, True, False, False, False, False, True]
    assert not rep[:, 2].any()
    # the attack's run unknown: the same target is a repeat whatever the flag says
    a = {"kind": kind, "target": target, "point": point, "run": run, "run_known": known, "alt": np.zeros_like(run),
         "t": np.arange(T, dtype=float)}
    lab = ds.student_labels(a)
    assert lab["repeats"] == 3 and lab["order"][1, 0] == -1 and lab["order"][6, 1] == -1
    assert lab["commit"][5, 1] == 3                                     # the hold at 5: no real change after it


def test_battles_are_found_and_split_by_start_both_sides_together_the_exam_apart(tmp_path):
    for i in range(40):
        write_meta(tmp_path / f"a{i}.npz", f"a{i}", [1, 2])
    write_meta(tmp_path / "e0.npz", "e0", [2])
    write_meta(tmp_path / "h0.npz", "h0", [1], human=True)
    sides = ds.discover([tmp_path, tmp_path])                          # a run in two folders: once
    assert len(sides) == 81 and {s.kind for s in sides} == {"aivai", "exam"}
    p = ds.split(sides, 0.25)
    assert len(p["exam"]) == 1 and p["exam"][0].run == "e0"
    tr, va = {s.group for s in p["train"]}, {s.group for s in p["val"]}
    assert not tr & va and 3 <= len(va) <= 20
    for s in p["val"]:                                                 # both sides of a held-out battle
        assert sum(x.run == s.run for x in p["val"]) == 2
    assert p == ds.split(sides, 0.25)                                  # the same every time


def test_a_side_recorded_with_the_old_token_loads_in_todays_width(tmp_path, monkeypatch):
    """Data recorded before the token grew (11.10.2026, 155 columns): the loader puts its columns at their new places
    (observe.tokens.widen, observation.old_columns) and zeros in the new ones; the network's width loads as it is."""
    from tools.nn.model import observation as ob
    from tools.nn.observe.tokens import widen
    old = 155
    cols = ob.old_columns(old)
    assert cols is not None and ob.TOKEN > old
    L, N = 3, 2
    tok = np.random.default_rng(0).random((L, N, old)).astype(np.float32)
    path = tmp_path / "b.npz"
    np.savez_compressed(path, s0_tokens=tok, s0_ctx=np.zeros((L, 4), np.float32), s0_geo=np.zeros(3, np.float32),
                        s0_t=np.arange(L, dtype=np.float32))
    lab = {k: np.full((L, N), -1, np.int64) for k in ("kind", "target", "point", "commit", "order")}
    lab.update({k: np.zeros((L, N), bool) for k in ("run", "run_known", "alt")})
    seen = {}
    monkeypatch.setattr(ds, "student_labels", lambda a: seen.setdefault("w", a["tokens"].shape[-1]) and dict(lab))
    monkeypatch.setattr(ds, "replay", lambda a, l: (np.zeros((L, N), bool), np.zeros((L, N)), l["kind"], 0))
    side = ds.Side(path, 0, "r", "g", "exam", L, N)
    out = ds.Loader(N).load(side)
    assert seen["w"] == ob.TOKEN                       # the labels and the replay see today's token too
    assert out["tokens"].shape == (L, N, ob.TOKEN)
    assert np.array_equal(out["tokens"][..., cols], tok)
    rest = np.setdiff1d(np.arange(ob.TOKEN), cols)
    assert not out["tokens"][..., rest].any()
    now = np.ones((L, N, ob.TOKEN), np.float32)
    assert widen(now) is now
