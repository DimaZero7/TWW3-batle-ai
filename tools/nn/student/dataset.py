"""The student's data: the converted battles (tools/nn/observe/convert.py, one .npz a battle), split by start, with
the student's labels, in batches of streams.

* Which battles: the game AI against the game AI (both sides observed: build/observe/aivai_ok.txt) - "aivai"; the
  game AI against the network (only the AI's side: the exam, 3 starts) - "exam"; a human's battles are left out.
* The split goes by start (the army seed of the run's manifest, both sides of a battle together): the held-out starts
  are battles the student never saw, not other seconds of the same ones. The exam starts are their own check (their
  ~400 battles are 3 starts: training on them would teach those 3 by heart); --exam-train puts some in training.
* The labels (student_labels): every recorded second of every unit that takes orders, the AI's new order (convert.py
  read it: kind, attack target, move cell, run; a running move is the set {move + run, withdraw}) or, when it gave
  none, keep - the AI gives an order now and then and lets it run, as the network's keep does. An order that repeats
  the unit's last one (the same kind, target, cell and run: the AI re-clicks, its point moving a few metres) is keep
  too (repeats()). The commitment (v2:
  with a new order the unit keeps it 2 / 4 / 8 / 16 s, commit.py) is read from the gap to the unit's next new order:
  the longest duration within it. The commitment is then replayed over the battle (commit.inputs / apply, as the
  simulator keeps it), so the inputs "free" and "commit" are what the student sees when it plays, and a held unit's
  second teaches nothing (its only choice is keep). A new order inside the 2-s minimum (the AI re-ordered the unit
  after 1 s) falls on a held unit: dropped, counted ("held_drop").
* The batches (Streams): B streams, each a queue of battle sides played from their start in chunks of T seconds; the
  memory (GRU) is carried from chunk to chunk (no gradient across them) and starts empty where a new battle side
  begins inside a chunk (reset). The units of every battle are padded to the largest battle's.
"""
import json
import math
import random
import zlib
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import config as project
from tools.nn.observe.tokens import widen

DIRS = (project.BUILD / "observe" / "data", project.BUILD / "student" / "data")
OBS = ("tokens", "ctx", "own", "attend", "ctrl", "target_ok", "pos", "abil", "abil_ok")
F32 = ("tokens", "ctx", "pos", "abil")
DURATIONS = (2.0, 4.0, 8.0, 16.0)        # tools/nn/model/commit.py DURATIONS (torch-free here)
HOLD, MOVE, ATTACK, WITHDRAW, KEEP = 0, 1, 2, 3, 4   # tools/nn/sim/orders.py


@dataclass
class Side:
    path: Path
    side: int
    run: str
    group: str        # the start: "seed<army seed>" (else the run's name)
    kind: str         # "aivai", "exam"
    T: int
    N: int

    @property
    def name(self):
        return f"{self.run}/s{self.side}"


def _manifest(meta):
    """The run's manifest (meta["path"] as convert.py stored it: inside the container /repo/build/..., or build/...)."""
    p = str(meta.get("path", "")).replace("\\", "/")
    rest = p.split("build/", 1)[1] if "build/" in p else None
    f = project.BUILD / rest / "manifest.json" if rest else None
    if f is None or not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def discover(dirs=DIRS, cfg=None):
    """[Side] of the converted battles in the folders (a run in two folders: the first). cfg: refuse a battle
    converted for another grid."""
    out, seen = [], set()
    for d in dirs:
        for p in sorted(Path(d).glob("*.npz")):
            with np.load(p) as z:
                meta = json.loads(str(z["meta"]))
            if meta["run"] in seen:
                continue
            seen.add(meta["run"])
            if cfg is not None and (meta["cfg"]["sectors"], meta["cfg"]["fine"], meta["cfg"]["grid_half"]) != \
                    (cfg.sectors, cfg.fine, cfg.grid_half):
                raise SystemExit(f"{p}: converted for another grid {meta['cfg']}")
            per = meta.get("per_side", {})
            if any(v.get("human") for v in per.values()):
                continue
            army = (_manifest(meta).get("config") or {}).get("army") or {}
            group = f"seed{army['seed']}" if army.get("seed") is not None else meta["run"]
            kind = "aivai" if len(meta["sides"]) == 2 else "exam"
            for s in meta["sides"]:
                out.append(Side(p, int(s), meta["run"], group, kind, int(meta["T"]), int(meta["N"])))
    return out


def held_out(group, share, salt="student"):
    """A start held out (deterministic: the same in every run and in eval)."""
    return zlib.crc32(f"{salt}:{group}".encode()) % 10000 < share * 10000


def split(sides, val_share=0.15, exam_train=(), salt="student"):
    """{"train", "val", "exam"}: [Side]. The AI-vs-AI starts by held_out; the exam's in "exam" unless their seed is in
    exam_train (then in training)."""
    out = {"train": [], "val": [], "exam": []}
    keep = {f"seed{s}" for s in exam_train}
    for s in sides:
        if s.kind == "exam":
            out["train" if s.group in keep else "exam"].append(s)
        else:
            out["val" if held_out(s.group, val_share, salt) else "train"].append(s)
    return out


def commit_labels(kind, t, durations=DURATIONS):
    """[T, N] the commitment's index of every new order (kind >= 0; else -1): the longest duration within the gap to
    the unit's next new order (none later: the longest), at least the shortest."""
    T, N = kind.shape
    out = np.full((T, N), -1, np.int64)
    d = np.asarray(durations)
    for i in range(N):
        rows = np.nonzero(kind[:, i] >= 0)[0]
        for a, k in enumerate(rows):
            gap = (t[rows[a + 1]] - t[k]) if a + 1 < len(rows) else math.inf
            out[k, i] = max(0, int((d <= gap + 1e-6).sum()) - 1)
    return out


def repeats(kind, target, point, run, run_known):
    """[T, N] the AI's new orders that repeat the unit's last one: the same kind and, for an attack, the same target,
    for a move the same cell, run the same where known (a hold after a hold). The AI re-clicks often (its order point
    moves by a few metres in the same cell); in the network's language that is keep - the order goes on."""
    T, N = kind.shape
    out = np.zeros((T, N), bool)
    for i in range(N):
        last = None
        for k in np.nonzero(kind[:, i] >= 0)[0]:
            o = (int(kind[k, i]), int(target[k, i]) if kind[k, i] == ATTACK else -1,
                 int(point[k, i]) if kind[k, i] in (MOVE, WITHDRAW) else -1)
            r = bool(run[k, i]) if run_known[k, i] else None
            if last is not None and o == last[0] and (r is None or last[1] is None or r == last[1]):
                out[k, i] = True
                last = (o, last[1] if r is None else r)
            else:
                last = (o, r)
    return out


def student_labels(arrays):
    """The student's labels of a side's arrays (the npz keys without s<side>_) -> dict of [T, N]: order (the AI's
    new orders without the repeats: repeats()), target, point, run, run_known, alt, commit (from the gaps between the
    orders left); "repeats": how many were taken as keep; numpy. The kind (KEEP where none) comes from replay()."""
    kind = arrays["kind"].astype(np.int64)
    target, point = arrays["target"].astype(np.int64), arrays["point"].astype(np.int64)
    run, known = arrays["run"].astype(bool), arrays["run_known"].astype(bool)
    rep = repeats(kind, target, point, run, known)
    kind = np.where(rep, -1, kind)
    return {"order": kind, "target": target, "point": point, "run": run, "run_known": known,
            "alt": arrays["alt"].astype(bool), "commit": commit_labels(kind, arrays["t"]), "repeats": int(rep.sum())}


def replay(arrays, lab):
    """The commitment over the battle (module doc) -> (free [T, N], commit inputs [T, N, 2], kind [T, N]: the AI's
    order on a free unit, KEEP on a free unit without one, -1 elsewhere; held_drop: new orders on a held unit)."""
    import torch

    from tools.nn.model import commit as cm
    T, N = lab["order"].shape
    st = cm.start(1, N)
    free = np.zeros((T, N), bool)
    cin = np.zeros((T, N, 2), np.float32)
    kind = np.full((T, N), -1, np.int64)
    dropped = 0
    keys = ("tokens", "ctx", "own", "attend", "ctrl", "target_ok", "pos")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)          # ~40 tiny operations a second: a thread pool only waits on itself
    try:
        return _replay(arrays, lab, st, free, cin, kind, dropped, keys, T)
    finally:
        torch.set_num_threads(threads)


def _replay(arrays, lab, st, free, cin, kind, dropped, keys, T):
    import torch

    from tools.nn.model import commit as cm
    with torch.no_grad():
        for k in range(T):
            o = {key: torch.as_tensor(np.asarray(arrays[key][k:k + 1])) for key in keys}
            for key in ("tokens", "ctx", "pos"):
                o[key] = o[key].float()
            tk = float(arrays["t"][k])
            o = cm.inputs(st, o, tk)
            f = o["free"][0].numpy()
            order = lab["order"][k]
            new = f & (order >= 0)
            dropped += int(((order >= 0) & ~f).sum())
            kd = np.where(new, order, KEEP)
            kind[k] = np.where(f, kd, -1)
            free[k], cin[k] = f, o["commit"][0].numpy()
            act = SimpleNamespace()
            act.kind = torch.as_tensor(kd)[None]
            act.target = torch.as_tensor(np.where(new & (order == ATTACK), lab["target"][k], -1))[None]
            act.commit = torch.as_tensor(np.maximum(lab["commit"][k], 0))[None]
            st = cm.apply(st, o, tk, act)
    return free, cin, kind, dropped


def _pad(a, N, value=0):
    if a.ndim < 2 or a.shape[1] == N:
        return a
    w = [(0, 0)] * a.ndim
    w[1] = (0, N - a.shape[1])
    return np.pad(a, w, constant_values=value)


class Loader:
    """A side's arrays padded to N, with the labels and the replayed commitment (cached: the replay costs ~1 ms a
    second; the observations are read from the file again every time, the cache holds only the small parts)."""

    def __init__(self, N):
        self.N = N
        self.cache = {}

    def load(self, side):
        with np.load(side.path) as z:
            pre = f"s{side.side}_"
            a = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
        if "tokens" in a:
            a["tokens"] = widen(a["tokens"])        # recorded before the token grew (11.10.2026): today's width
        key = side.name
        if key not in self.cache:
            lab = student_labels(a)
            free, cin, kind, dropped = replay(a, lab)
            lab.update(kind=kind, free=free, commit_in=cin)
            lab["held_drop"] = dropped
            lab["orders"] = int((lab["order"] >= 0).sum())
            self.cache[key] = lab
        lab = self.cache[key]
        N = self.N
        out = {k: a[k] if k == "ctx" else _pad(a[k], N) for k in OBS if k in a}      # (ctx: no units axis)
        out["geo"] = a["geo"]
        out["t"] = a["t"]
        for k in ("kind", "target", "point", "commit"):
            out[k] = _pad(lab[k], N, -1)
        for k in ("run", "run_known", "alt", "free"):
            out[k] = _pad(lab[k], N, False)
        out["commit_in"] = _pad(lab["commit_in"], N)
        out["held_drop"] = lab["held_drop"]
        return out


LABELS = ("kind", "target", "point", "commit", "run", "run_known", "alt")


def to_batch(segs, device="cpu"):
    """B streams' segments (Streams: segs[b] = [(arrays, rows, reset, valid)], rows the seconds of that side, T a
    stream) -> the batch dict of tensors [T, B, ...]: obs (with free, commit, geo), the labels, reset [T, B] (on the
    segment's first second), valid [T, B]."""
    import torch

    def stack(fn, dtype=None):
        x = np.stack([np.concatenate([fn(a, rows) for (a, rows, _, _) in row]) for row in segs], 1)
        t = torch.as_tensor(x)
        return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device)
    first = segs[0][0][0]
    obs = {}
    for k in OBS:
        if k in first:
            obs[k] = stack(lambda a, r, k=k: a[k][r], torch.float32 if k in F32 else None)
    obs["free"] = stack(lambda a, r: a["free"][r])
    obs["commit"] = stack(lambda a, r: a["commit_in"][r], torch.float32)
    obs["geo"] = stack(lambda a, r: np.broadcast_to(a["geo"], (len(r),) + a["geo"].shape), torch.float32)
    out = {"obs": obs}
    for k in LABELS:
        out[k] = stack(lambda a, r, k=k: a[k][r])
    flags = lambda f: torch.as_tensor(np.stack([np.concatenate([f(rows, rs, v) for (_, rows, rs, v) in row])
                                                for row in segs], 1), device=device)
    valid = flags(lambda rows, rs, v: np.full(len(rows), v))
    out["kind"] = torch.where(valid[..., None], out["kind"], torch.full_like(out["kind"], -1))
    out["reset"] = flags(lambda rows, rs, v: np.arange(len(rows)) == 0 if rs else np.zeros(len(rows), bool))
    out["valid"] = valid
    return out


class Streams:
    """B streams over the sides (module doc): iter() gives the batches of one pass over all of them; the memory is the
    caller's (reset tells where it starts empty)."""

    def __init__(self, sides, B=8, T=32, N=None, seed=0, shuffle=True):
        self.sides = list(sides)
        self.B, self.T = max(1, min(B, len(self.sides))), T
        self.N = N or max(s.N for s in self.sides)
        self.loader = Loader(self.N)
        self.rng = random.Random(seed)
        self.shuffle = shuffle

    def queues(self):
        """The sides dealt to the streams, the longest first to the shortest stream (lengths even)."""
        order = list(self.sides)
        if self.shuffle:
            self.rng.shuffle(order)
        q = [[] for _ in range(self.B)]
        length = [0] * self.B
        for s in order:
            j = int(np.argmin(length))
            q[j].append(s)
            length[j] += s.T
        return q, max(length)

    def __iter__(self):
        """Batches of segments (to_batch): every stream T seconds, from its side now and the next ones in its queue;
        a stream whose queue has run out repeats its last second, not taught (valid False)."""
        q, L = self.queues()
        cur = [None] * self.B          # [arrays, position] of each stream's side now
        last = [None] * self.B         # (arrays, its last second given)
        for _ in range(0, L, self.T):
            segs, any_valid = [], False
            for b in range(self.B):
                row, need = [], self.T
                while need:
                    if cur[b] is not None and cur[b][1] < len(cur[b][0]["t"]):
                        a, k = cur[b]
                        n = min(need, len(a["t"]) - k)
                        row.append((a, np.arange(k, k + n), k == 0, True))
                        cur[b][1] += n
                        last[b] = (a, k + n - 1)
                        need -= n
                        any_valid = True
                    elif q[b]:
                        cur[b] = [self.loader.load(q[b].pop(0)), 0]
                    else:                    # this stream is done: its last second again, nothing taught
                        row.append((last[b][0], np.full(need, last[b][1]), False, False))
                        need = 0
                segs.append(row)
            if not any_valid:
                break
            yield segs

    def held_drop(self):
        return sum(v["held_drop"] for v in self.loader.cache.values())

    def counts(self):
        """{"read": the AI's new orders as read, "repeats": taken as keep, "orders": the rest, "held_drop": of those
        on a held unit} over the sides loaded so far."""
        c = self.loader.cache.values()
        o, r, h = (sum(v[k] for v in c) for k in ("orders", "repeats", "held_drop"))
        return {"read": o + r, "repeats": r, "orders": o, "held_drop": h}

    def counts_text(self):
        c = self.counts()
        return (f"the AI's orders read {c['read']}: repeats taken as keep {c['repeats']} "
                f"({c['repeats'] / max(1, c['read']):.2f}), the rest {c['orders']}, of them on a held unit (dropped) "
                f"{c['held_drop']} ({c['held_drop'] / max(1, c['orders']):.2f})")
