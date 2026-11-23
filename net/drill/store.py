"""Shards of one chunk + a manifest (tmp/engine/design_merged.md section 4); the mixture lives in RAM (no sharded
shuffle in v1: above ram_gb the fit halts before it starts).

    Store(dir)            a collection's folder: shard_<part>_<i>.pt (rows: tensors of one chunk, battle ids local to
                          the collection; cards [B, ...] per battle; meta per battle) and manifest.json (shard name,
                          sha256, bytes, rows, battles, part), written atomically after every shard
    Store(dir, run_key, round)   the manifest carries the run key and the round: a folder whose manifest has another
                          run key is refused (StoreRefused); store.digest() - the manifest's sha256 (the done-record
                          keeps it; resume verifies it)
    store.put(part, rows, meta, cards=None, stats=None)   one shard
    store.mixture(ram_gb=None)   (rows of every shard concatenated, battle ids offset to the mixture's, the cards
                          stacked, meta) - halts (MemoryError) when twice the manifest's bytes (the parts and the
                          concatenation live together) exceed ram_gb, before anything is loaded
    mixture_of(stores, ram_gb, remap, pad)   the same over several collections; pad(parts) - the level's padding of
                          rows of different battle sizes (zeros to the max N, n_slots recorded), cards too
    Lock(path, run_key, log)  a writer's lock file (O_EXCL; {pid, host, run_key}, a heartbeat while held): a second
                          writer refused; a lock of the same run key whose holder is dead (this host) or silent for
                          STALE_S is taken over and said in the log
"""
import hashlib
import json
import os
from pathlib import Path

import torch


def atomic_save(obj, path):
    """torch.save to <path>.part, fsync of the written handle, os.replace (net/tactic/ckpt.py:38-47)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_text(text, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nbytes(rows):
    return sum(t.numel() * t.element_size() for t in rows.values() if torch.is_tensor(t))


class StoreRefused(RuntimeError):
    pass


HEARTBEAT_S = 60
STALE_S = 600
_LOCK_WHY = ("brief_A3 R4b: every docker run is a new container (ops/dock.sh: --rm, a new hostname), so a crashed "
             "writer's pid cannot be asked from the next one; the holder touches its lock every HEARTBEAT_S (60 s) "
             "and a lock of the SAME run key is taken over when its holder is dead on this host or its heartbeat is "
             "older than STALE_S (10 heartbeats); another run key's lock is always refused")


class Lock:
    """(module doc) - a context manager. path: a folder (its writer.lock) or a lock file (*.lock). The lock file holds
    {pid, host, run_key, t}; a heartbeat thread touches it while held."""

    def __init__(self, path, run_key=None, log=None):
        p = Path(path)
        self.path = p if p.suffix == ".lock" else p / "writer.lock"
        self.run_key, self.log = run_key, log or (lambda x: None)
        self._stop = None

    def _body(self):
        import socket
        import time
        return json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "run_key": self.run_key,
                           "t": time.time()})

    def _stale(self):
        """(takeable, why) of the lock file on disk."""
        import socket
        import time
        try:
            x = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, "an unreadable lock"
        if self.run_key is None or x.get("run_key") != self.run_key:
            return False, f"run key {x.get('run_key')} (this run {self.run_key})"
        age = time.time() - self.path.stat().st_mtime
        # (os.kill(pid, 0) only on POSIX: on Windows it would terminate the process - there only the heartbeat)
        if x.get("host") == socket.gethostname() and os.name == "posix":
            try:
                os.kill(int(x["pid"]), 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except (PermissionError, OSError, ValueError, KeyError):
                alive = True
            if not alive:
                return True, f"its pid {x.get('pid')} is dead on this host"
        if age > STALE_S:
            return True, f"its heartbeat is {age:.0f} s old (> {STALE_S} s)"
        return False, f"its holder pid {x.get('pid')} on {x.get('host')} is live (heartbeat {age:.0f} s ago)"

    def __enter__(self):
        import threading
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            ok, why = self._stale()
            if not ok:
                raise StoreRefused(f"{self.path}: another writer holds this shard folder ({why}; its lock: "
                                   f"{self.path.read_text(encoding='utf-8', errors='replace')!r}) - refused")
            self.log({"lock_taken_over": str(self.path), "why": why,
                      "old": self.path.read_text(encoding="utf-8", errors="replace")})
            os.remove(self.path)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, self._body().encode())
        os.close(fd)
        self._stop = threading.Event()

        def beat(stop=self._stop, path=self.path):
            while not stop.wait(HEARTBEAT_S):
                try:
                    os.utime(path)
                except OSError:
                    return
        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *exc):
        if self._stop is not None:
            self._stop.set()
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        return False


class Store:
    def __init__(self, path, run_key=None, round=None):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        mf = self.dir / "manifest.json"
        self.manifest = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"shards": []}
        if run_key is not None:
            have = self.manifest.get("run_key")
            if mf.exists() and (have != run_key or self.manifest.get("round") != round):
                raise StoreRefused(f"{mf}: the manifest of run key {have} round {self.manifest.get('round')} - this "
                                   f"run is {run_key} round {round} (refused)")
            self.manifest.update(run_key=run_key, round=round)

    def digest(self):
        mf = self.dir / "manifest.json"
        return sha256(mf) if mf.exists() else None

    def put(self, part, rows, meta, cards=None, stats=None, maps=None):
        """maps: the level's table of what the rows' ids point to in this process (tactic: the map registry's names
        and grid sha, levels/tactic.map_table) - mixture(remap=) rewrites the ids in another process."""
        i = len(self.manifest["shards"])
        name = f"shard_{part}_{i:03d}.pt"
        atomic_save({"rows": rows, "meta": meta, "cards": cards, "stats": stats, "maps": maps}, self.dir / name)
        n = int(next(iter(rows.values())).shape[0]) if rows else 0
        cb = sum(c.numel() * c.element_size() for c in (cards or []) if c is not None)
        self.manifest["shards"].append({"name": name, "part": part, "rows": n, "battles": len(meta),
                                        "bytes": nbytes(rows) + cb, "sha256": sha256(self.dir / name)})
        atomic_text(json.dumps(self.manifest, indent=1), self.dir / "manifest.json")
        return name

    def bytes(self):
        return sum(s["bytes"] for s in self.manifest["shards"])

    def shards(self, verify=True):
        for s in self.manifest["shards"]:
            p = self.dir / s["name"]
            if verify and sha256(p) != s["sha256"]:
                raise RuntimeError(f"{p}: sha256 differs from the manifest")
            yield s, torch.load(p, map_location="cpu", weights_only=False)

    def mixture(self, ram_gb=None, remap=None):
        return mixture_of([self], ram_gb, remap)


def ram_check(nbytes, ram_gb, what="the mixture"):
    """(module doc) twice nbytes against ram_gb (None: no limit) - MemoryError before anything is loaded."""
    if ram_gb is not None and 2 * nbytes > float(ram_gb) * (1 << 30):
        raise MemoryError(f"{what}: 2 x {nbytes / (1 << 30):.2f} GB (the parts and their concatenation) exceed ram_gb "
                          f"{ram_gb}: halts before it starts")


def mixture_of(stores, ram_gb=None, remap=None, pad=None):
    """(rows, cards [B, ...] or None, meta) of the stores' shards in order; the battle ids 'b' offset; remap(rows,
    maps) - the level's id rewrite of each shard (levels/tactic.remap); pad (module doc)."""
    ram_check(sum(s.bytes() for s in stores), ram_gb)
    parts, cards, meta = [], [], []
    for st in stores:
        for _, sh in st.shards():
            rows = dict(sh["rows"])
            if remap is not None and rows:
                rows = remap(rows, sh.get("maps"))
            if rows and int(next(iter(rows.values())).shape[0]) > 0:
                rows["b"] = rows["b"].long() + len(meta)
                parts.append(rows)
            meta += sh["meta"]
            cards += list(sh["cards"]) if sh["cards"] is not None else [None] * len(sh["meta"])
    if pad is not None:
        parts, cards = pad(parts, cards)
    rows = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]} if parts else {}
    some = next((c for c in cards if c is not None), None)
    # (a battle without a stored tick: zeros - the legacy NetBuffer.data()'s `some * 0`)
    cards = torch.stack([some * 0 if c is None else c for c in cards], 0) if some is not None else None
    return rows, cards, meta


def pad_parts(parts, cards, keys, n_of=None):
    """A level's pad for mixture_of (NEEDS_A 3): the tensors `keys` of every part (and the per-battle cards) zero-padded
    to the largest size in every trailing dim; n_of(part) -> [rows] the part's own battle size, recorded as the rows'
    'n_slots' (long) - the padded slots stay distinguishable. Parts whose shapes agree are returned unchanged (but
    'n_slots' is still added when n_of is given)."""
    import torch.nn.functional as F

    def padded(t, shape):
        if tuple(t.shape[1:]) == tuple(shape):
            return t
        pad = []
        for have, want in zip(reversed(t.shape[1:]), reversed(shape)):
            pad += [0, int(want) - int(have)]
        return F.pad(t, pad) if t.dtype != torch.bool else F.pad(t.to(torch.uint8), pad).bool()
    out = []
    for p in parts:
        q = dict(p)
        if n_of is not None:
            q["n_slots"] = torch.as_tensor(n_of(p), dtype=torch.long)
        out.append(q)
    for k in keys:
        have = [p[k] for p in out if k in p]
        if not have:
            continue
        shape = [max(int(t.shape[d]) for t in have) for d in range(1, have[0].dim())]
        for q in out:
            if k in q:
                q[k] = padded(q[k], shape)
    cs = [c for c in cards if c is not None]
    if cs:
        shape = [max(int(c.shape[d]) for c in cs) for d in range(cs[0].dim())]
        cards = [None if c is None else padded(c[None], shape)[0] for c in cards]
    return out, cards
