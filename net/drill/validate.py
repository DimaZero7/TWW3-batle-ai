"""Validation V (tmp/engine/design_merged.md section 4; brief_A2 Parts 2-3), level-blind: the teacher-path cache and the
relative stop. The level plays and reads (levels/<level>.v_read); the card names the marks.

    play_key(device, level_parts)   the cache key: the runtime (torch, CUDA, cuDNN, deterministic flags, the compiler's
                environment), device kind, dtype + the level's parts (simulator key, files, configs, weights, specs with
                their annotations, chunk, seeds)
    TeacherCache(dir).ensure(key, play, replay, remap)   path T of a set under dir/<key>/: played when absent (store
                shards + key.json written last); an interrupted build (shards, no key.json) is moved away to
                <key>.staleN and played anew; on reuse the first chunk is replayed in memory and compared to the
                stored first shard (every tensor equal, the cards, the meta less its local index) - a difference
                halts (StaleCache). read(key, remap, ram_gb) checks twice the stored bytes against ram_gb first
    relative_stop(card, tv, tv0, tv_ref, report)   design section 0's stop -> (stop, reasons, judged): without a
                card.reference - card.absolute's lines and each mark of card.marks_lines on path T passes its line;
                with one (brief_A3 R1) - each mark on path T (a mark with path N only: N) not worse than the
                reference's V read by more than two errors, the absolute lines printed in report; always path N no
                worse than path T by more than two errors (both paths' errors), every old card's mark
                (card.retention) within two errors of the parent's V0. Every cell read has n >= its declared minimum
                (mark_min: card.mark_min, default cards.MARK_MIN_N; printed beside the n in report), else the round
                is not judged (no stop). Errors by battle (brief 22.11 (c)): a level's cell may carry bb {keys, x,
                n} - its numerator and denominator a battle in the canonical battle order (keys: their sha) - and
                two_errors by battle (cluster_two_err; the binomial one kept as two_errors_tick) and its support
                (contributing battles, the top battle's share). The stop compares (compare) the exact ratios of the
                counts; two such cells need MIN_BATTLES contributing battles each and the same roster (paired by
                battle) - else the mark is not judged (no stop); unpaired by battle only in reports; two legacy
                cells by tick
"""
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from net.drill.store import Store, atomic_text, mixture_of, ram_check

ROOT = Path(__file__).resolve().parents[2]
WATCH_SCHEMA = 1


class StaleCache(RuntimeError):
    pass


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def files_sha(paths):
    h = hashlib.sha256()
    fs = []
    for p in paths:
        q = ROOT / p
        if q.is_file():
            fs.append(q)
        elif q.is_dir():
            fs += [x for x in q.rglob("*.py") if "__pycache__" not in x.parts]
    for f in sorted(set(fs), key=lambda x: x.relative_to(ROOT).as_posix()):
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def ram_gb():
    return json.loads((ROOT / "net" / "config" / "drill.json").read_text(encoding="utf-8")).get("ram_gb")


def play_key(device, level_parts):
    x = {"torch": torch.__version__, "cuda": torch.version.cuda,
         "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
         "deterministic": torch.are_deterministic_algorithms_enabled(), "device": str(device).split(":")[0],
         "dtype": str(torch.get_default_dtype()),
         "compiler_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("TORCH_COMPILE", "TORCHDYNAMO"))},
         "level": level_parts}
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ the teacher path's cache
class MemStore:
    """A Store's put() kept in memory (the replay)."""

    def __init__(self):
        self.manifest = {"shards": []}
        self.items = []

    def put(self, part, rows, meta, cards=None, stats=None, maps=None):
        self.items.append({"rows": rows, "meta": meta, "cards": cards, "maps": maps})
        self.manifest["shards"].append({"part": part})


def _same(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b) or (
            a.is_floating_point() and torch.equal(torch.nan_to_num(a, 1e30), torch.nan_to_num(b, 1e30))))
    return a == b


class TeacherCache:
    """(module doc) dir / <key> / {shard_*.pt, manifest.json, key.json}."""

    def __init__(self, root):
        self.root = Path(root)

    def path(self, key):
        return self.root / key

    def have(self, key):
        return (self.path(key) / "key.json").exists()

    def ensure(self, key, play, replay=None, remap=None, log=print):
        """-> how: 'played' | 'played (an interrupted build moved away to ...)' | 'read (first chunk replayed: equal)'
        | 'read (no replay)'. play(store) -> info dict; replay(MemStore) -> info."""
        d = self.path(key)
        note = ""
        if not self.have(key) and d.exists() and any(d.glob("shard_*.pt")):
            n = 1
            while (self.root / f"{key}.stale{n}").exists():
                n += 1
            d.rename(self.root / f"{key}.stale{n}")
            note = f" (an interrupted build moved away to {key}.stale{n})"
            log({"V_cache": "interrupted build", "moved": f"{key}.stale{n}"})
        if not self.have(key):
            st = Store(d)
            info = play(st)
            atomic_text(json.dumps({"key": key, **info, "shards": len(st.manifest["shards"])}), d / "key.json")
            return "played" + note
        if replay is None:
            return "read (no replay)"
        mem = MemStore()
        replay(mem)
        st = Store(d)
        first = next(iter(st.shards()), None)
        if first is None or not mem.items:
            raise StaleCache(f"{d}: nothing to compare (stored shards {len(st.manifest['shards'])}, replayed "
                             f"{len(mem.items)})")
        sh = first[1]
        rows = dict(sh["rows"])
        if remap is not None and rows:
            rows = remap(rows, sh.get("maps"))
        new = dict(mem.items[0]["rows"])
        bad = sorted(set(rows) ^ set(new)) + [k for k in rows if k in new and not _same(rows[k], new[k])]
        cs = sh["cards"] or []
        nc = mem.items[0]["cards"] or []
        if len(cs) != len(nc) or any(not _same(x, y) for x, y in zip(cs, nc)):
            bad.append("cards")
        strip = lambda m: [{k: v for k, v in x.items() if k != "spec"} for x in m]       # noqa: E731
        if strip(sh["meta"]) != strip(mem.items[0]["meta"]):
            bad.append("meta")
        if bad:
            raise StaleCache(f"{d}: the replayed first chunk differs from the stored one in {bad[:8]} - a stale cache "
                             "(give a fresh --v-cache or remove this key)")
        log({"V_cache": "first chunk replayed", "rows": int(next(iter(new.values())).shape[0]) if new else 0,
             "equal": True})
        return "read (first chunk replayed: equal)"

    def read(self, key, remap=None, ram_gb=None):
        st = Store(self.path(key))
        ram_check(st.bytes(), ram_gb, f"the teacher-path cache {self.path(key)}")
        rows, cards, meta = mixture_of([st], None, remap)
        info = json.loads((self.path(key) / "key.json").read_text(encoding="utf-8"))
        return rows, cards, meta, info


# ------------------------------------------------------------------ errors by battle
MIN_BATTLES = 8
_MIN_BATTLES_WHY = ("the coordinator's ruling 22.11 (Astra engine_code3 finding 2): a cluster variance from fewer "
                    "contributing battles is not estimable - the mark is not judged, never a silent tick error")


def cluster_two_err(x, n):
    """Two errors of the ratio sum(x) / sum(n) with the battles as the clusters (the rows of one battle are not
    independent): var = k / (k - 1) * sum_b (x_b - p n_b)^2 / N^2 over the k battles with n_b > 0; None when k < 2
    (estimable only from MIN_BATTLES - support())."""
    N = float(sum(n))
    k = sum(1 for v in n if v > 0)
    if N <= 0 or k < 2:
        return None
    p = sum(x) / N
    s = sum((a - p * b) ** 2 for a, b in zip(x, n) if b > 0)
    return round(2 * math.sqrt(k / (k - 1) * s) / N, 6)


def support(n):
    """{battles: contributing (n_b > 0), top_share: the largest battle's share of the labels, estimable: battles >=
    MIN_BATTLES}."""
    N = sum(n)
    k = sum(1 for v in n if v > 0)
    return {"battles": k, "top_share": round(max(n) / N, 4) if N else None, "estimable": k >= MIN_BATTLES}


def paired_two_err(bc, b0):
    """Two errors of p_c - p_0 paired by battle (the same roster, cells bb of equal length): d_b = (x_cb - p_c n_cb) /
    N_c - (x_0b - p_0 n_0b) / N_0, var = k / (k - 1) * sum d_b^2 over the k battles with a row in either; None when
    either is empty or k < 2."""
    xc, nc, x0, n0 = bc["x"], bc["n"], b0["x"], b0["n"]
    Nc, N0 = float(sum(nc)), float(sum(n0))
    if Nc <= 0 or N0 <= 0:
        return None
    pc, p0 = sum(xc) / Nc, sum(x0) / N0
    d = [(a - pc * b) / Nc - (c - p0 * e) / N0 for a, b, c, e in zip(xc, nc, x0, n0) if b > 0 or e > 0]
    if len(d) < 2:
        return None
    return round(2 * math.sqrt(len(d) / (len(d) - 1) * sum(v * v for v in d)), 6)


def _roster(b):
    return b.get("ids") if b.get("ids") is not None else None


def same_roster(b, b0):
    """(same, missing): the two cells' battle rosters - the ids when both carry them, else their keys digest and
    length; missing names up to 5 battles of either roster absent from the other."""
    if b.get("keys") != b0.get("keys") or len(b["x"]) != len(b0["x"]):
        r, r0 = _roster(b), _roster(b0)
        if r is not None and r0 is not None:
            s, s0 = set(r), set(r0)
            miss = sorted(s0 - s)[:5] + sorted(s - s0)[:5]
            return False, miss or [f"the same ids in another play (digest {b.get('keys')} vs {b0.get('keys')})"]
        return False, [f"digest {b.get('keys')} ({len(b['x'])} battles) vs {b0.get('keys')} ({len(b0['x'])})"]
    return True, []


def pair_of(c, c0):
    """How two cells are compared: 'paired by battle' (both carry bb of the same roster), 'by battle, unpaired' (both
    carry bb, rosters differ - a report only, never a stop), 'mixed' (one carries bb), else 'by tick' (two legacy
    cells: the binomial errors)."""
    b, b0 = c.get("bb"), c0.get("bb")
    if b and b0:
        return "paired by battle" if same_roster(b, b0)[0] else "by battle, unpaired"
    return "mixed" if (b or b0) else "by tick"


def ratio(c):
    """The exact share from the stored counts (bb), else the cell's share (a legacy cell)."""
    b = c.get("bb")
    if b and sum(b["n"]):
        return sum(b["x"]) / sum(b["n"])
    return c["share"]


def compare(op, c, c0, stop=True):
    """c against c0 (module doc): {worse, e, how, unjudged} - a recall (>=) lower, a false rate (<=) higher beyond two
    errors, on the exact ratios. Two cells with battle counts: each needs MIN_BATTLES contributing battles, and for a
    stop the same roster (paired by battle); unpaired (stop=False) only for a report. One cell with them and one
    without: not judged for a stop. Two legacy cells: the binomial errors ('by tick')."""
    how = pair_of(c, c0)
    out = {"worse": False, "e": None, "how": how, "unjudged": None}
    b, b0 = c.get("bb"), c0.get("bb")
    if how == "mixed" and stop:
        out["unjudged"] = "one cell carries battle counts, the other not - not judged"
        return out
    if b and b0:
        s, s0 = support(b["n"]), support(b0["n"])
        out["support"] = [s, s0]
        if not (s["estimable"] and s0["estimable"]):
            out["unjudged"] = (f"by battle not estimable: contributing battles {s['battles']} / {s0['battles']} "
                               f"(< {MIN_BATTLES}), top battle's share {s['top_share']} / {s0['top_share']}")
            return out
        ok, miss = same_roster(b, b0)
        if not ok and stop:
            out["unjudged"] = f"battle rosters differ (missing {miss}) - not judged"
            return out
        e = paired_two_err(b, b0) if ok else math.sqrt(two_err(c) ** 2 + two_err(c0) ** 2)
    else:
        e = math.sqrt(two_err(c) ** 2 + two_err(c0) ** 2)
    p, p0 = ratio(c), ratio(c0)
    out["e"] = e
    out["diff"] = p - p0
    out["worse"] = (p < p0 - e) if op == ">=" else (p > p0 + e)
    return out


# ------------------------------------------------------------------ the stop
def two_err(c):
    if c.get("two_errors") is not None:
        return float(c["two_errors"])
    p, n = c["share"], c["n"]
    return 2 * math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def _judged(c, mn):
    return bool(c) and c.get("share") is not None and int(c.get("n") or 0) >= mn


def mark_min(card, m, path):
    """A mark's declared minimum on a path (card.mark_min {mark: {path: n}}), default cards.MARK_MIN_N."""
    from net.drill.cards import MARK_MIN_N as MN
    return int(((getattr(card, "mark_min", None) or {}).get(m) or {}).get(path, MN))


def _worse(op, c, c0):
    """(worse, e) of compare(stop=True) - a cell pair compare() cannot judge raises Unjudged (callers that gate catch
    it and say NOT JUDGED)."""
    r = compare(op, c, c0)
    if r["unjudged"]:
        raise Unjudged(r["unjudged"])
    return r["worse"], r["e"]


class Unjudged(ValueError):
    pass


def _gate(op, c, c0, label, why, rep):
    """One compared pair of the stop: its line in rep; -> 'worse' | 'ok' | 'unjudged' (the reason in why)."""
    r = compare(op, c, c0)
    sup = r.get("support")
    rep.append(f"{label}: {r['how']}, two errors {r['e'] if r['e'] is None else round(r['e'], 6)}"
               + (f", battles {sup[0]['battles']} / {sup[1]['battles']}, top share {sup[0]['top_share']} / "
                  f"{sup[1]['top_share']}" if sup else ""))
    if r["unjudged"]:
        why.append(f"{label}: {r['unjudged']}")
        return "unjudged"
    return "worse" if r["worse"] else "ok"


def relative_stop(card, tv, tv0, tv_ref=None, report=None):
    """(module doc) -> (stop, reasons, judged); report (a list) gets the lines printed and not gated: each cell's n
    beside its minimum, and with a reference the absolute lines (card.absolute, card.marks_lines on path T)."""
    from net.drill.cards import MARK_MIN_N as MN
    why, judged = [], True
    rep = report if report is not None else []
    ref = getattr(card, "reference", None)
    if ref is not None and tv_ref is None:
        judged = False
        why.append(f"reference {ref.get('path')}: its V read missing (v_round_ref.json)")
    if getattr(card, "absolute", None) is not None:
        ok, f, jd = card.absolute(tv)
        if ref is None:
            why += f
            judged &= jd
        else:
            rep.append(f"absolute (printed, not gated - the card's reference): ok {ok and jd}, {f}")
    marks = card.marks(tv) if card.marks is not None else {}
    mref = card.marks(tv_ref) if (card.marks is not None and tv_ref is not None) else {}
    for m, (op, line) in (card.marks_lines or {}).items():
        cells = marks.get(m) or {}
        T, N = cells.get("T"), cells.get("N")
        R = {p: (mref.get(m) or {}).get(p) for p in ("T", "N")} if tv_ref is not None else {}
        need = [("T", T), ("N", N)] if "T" in cells else [("N", N)]
        if ref is not None and tv_ref is not None:
            need += [(f"reference {p}", R.get(p)) for p, _ in list(need)]
        ok_n = True
        for path, c in need:
            mn = mark_min(card, m, path.split()[-1])
            rep.append(f"{m} {path}: n {(c or {}).get('n')} (minimum {mn})")
            if not _judged(c, mn):
                ok_n = judged = False
                why.append(f"{m} path {path}: n {(c or {}).get('n')} < {mn} - not judged")
        if not ok_n:
            continue
        if ref is None:
            p0 = "T" if "T" in cells else "N"
            c0 = cells[p0]
            if (c0["share"] < line) if op == ">=" else (c0["share"] > line):
                why.append(f"{m} path {p0} {c0['share']} {'<' if op == '>=' else '>'} {line} (n {c0['n']})")
        elif tv_ref is not None:
            if "T" in cells:
                rep.append(f"{m} path T {T['share']} against the absolute line {op} {line} (printed)")
            p0 = "T" if "T" in cells else "N"
            g = _gate(op, cells[p0], R[p0], f"{m} path {p0} against the reference", why, rep)
            if g == "unjudged":
                judged = False
            elif g == "worse":
                why.append(f"{m} path {p0} {ratio(cells[p0]):.6f} worse than the reference's {ratio(R[p0]):.6f} "
                           f"beyond two errors ({rep[-1]})")
        if "T" in cells:
            g = _gate(op, N, T, f"{m} path N against T", why, rep)
            if g == "unjudged":
                judged = False
            elif g == "worse":
                why.append(f"{m} path N {ratio(N):.6f} {'<' if op == '>=' else '>'} path T {ratio(T):.6f} beyond two "
                           f"errors ({rep[-1]})")
    if getattr(card, "retention", None) is not None and card.parent is not None:
        if tv0 is None:
            judged = False
            why.append("retention: no V0 of the parent")
        else:
            r, r0 = card.retention(tv), card.retention(tv0)
            for m, (op, c) in r.items():
                c0 = r0[m][1]
                if not (_judged(c, MN) and _judged(c0, MN)):
                    judged = False
                    why.append(f"retention {m}: n {(c or {}).get('n')} / parent {(c0 or {}).get('n')} < {MN} - "
                               "not judged")
                    continue
                g = _gate(op, c, c0, f"retention {m} against the parent", why, rep)
                if g == "unjudged":
                    judged = False
                elif g == "worse":
                    why.append(f"retention {m} {ratio(c):.6f} vs the parent's {ratio(c0):.6f} beyond two errors "
                               f"({rep[-1]})")
    return (not why) and judged, why, judged
