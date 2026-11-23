"""The tactic level's validation V (moved from net/drill/validate.py; tmp/engine/design_merged.md section 4): the
teacher's path played once and stored with the teacher-forced targets and approaches of earlier slots (the walk fed with
them, imitate.py:408-416), then asked offline; the network's own path played every round (its follow windows and
standing side live, imitate.py:419-441).

    VBuffer(side)        path T's recorder: imitate.NetBuffer (no thinning) + the label-only columns the marks need
    ask_T(net, rows, cards, meta)   path T's imitate.Marks rows offline (the pointer's greedy pick on the stored walk,
                the side head's at the label; margins kept per row)
    MarksN(net)          path N's imitate.Marks with the engine's own post-finish guard
    v_parts(smoke, side, chunk), v_key(...)   V's sets and its cache key
    v_read(net, a, P, chunk, smoke, side, cache, log)   imitate.v_read's result (marks, done_rows, parts) with path T
                from validate.TeacherCache (played once a key; on reuse its first chunk replayed and compared); the
                engine's side marks side_marks_engine beside the legacy ones
    side_marks_engine(m)   the side marks with path N's wrong-target and KEEP picks IN the denominator (counted as not
                followed; imitate.side_marks drops them, imitate.py:470,620) and their share printed as "missing"
"""
import hashlib
import json
from pathlib import Path

import torch

from net.drill import masks as MK
from net.drill import record as RC
from net.drill import validate as VA
from net.drill.levels import tactic as LT

ROOT = Path(__file__).resolve().parents[3]
EXEC_FILES = ("net/unit", "net/operative/lower.py", "net/stage0/battle.py", "net/matchup/duel.py", "net/tactic",
              "bridge/companion/exchange.py")
RECORDER_FILES = ("net/drill/levels/tactic_v.py", "net/drill/levels/tactic_collect.py", "net/drill/record.py",
                  "net/drill/masks.py")


# ------------------------------------------------------------------ path T stored
def VBuffer(side):
    from net.tactic import heads as HD
    from net.tactic import imitate as IM
    from net.tactic import learn as LN
    from net.tactic import pin as PN

    class _VB(IM.NetBuffer):
        def __init__(self):
            super().__init__(check=0, every=1, side=side)
            self.done_cnt = [0, 0]
            self.vmeta = []

        def __call__(self, v, T, net_driven, lab=None):
            fx = v.fx
            dn = IM.done_rows(v)
            h0 = T["head_free"].any(1)
            self.done_cnt[0] += int(h0.sum())
            self.done_cnt[1] += int((h0 & dn).sum())
            n0 = len(self.pending)
            super().__call__(v, T, net_driven, lab)
            if len(self.pending) == n0:
                return
            out = self.pending[-1]
            sel = (T["head_free"] & ~dn[:, None]).any(1)           # (every = 1: no tick thinned)
            assert int(sel.sum()) == int(out["b"].shape[0])
            b = IM._bands()
            label = torch.where(T["attack"], T["k"], torch.full_like(T["k"], -1))
            lc = label.clamp(min=0)
            pin = PN.pinned(fx, touch=float(b["touch_m"][1]))
            far = fx.g_dist > float(b["reach_m"][1])
            ex = {"e": fx.slot_of_row(label), "es": fx.g_side.gather(2, lc[..., None])[..., 0],
                  "nhead": HD.head_free(fx), "far": far}
            for k, t in ex.items():
                out["v_" + k] = LN._pad_g(t[sel], 0).cpu()
            out["v_pin"] = pin[sel].cpu()
            out["v_sec32"] = torch.full((int(sel.sum()),), int(round(float(v.now))), dtype=torch.int32)
            from net.drill.levels.tactic_collect import own_done
            out["done_own"] = own_done(v)[sel].cpu()

        def flush(self, ii, specs, kind, net_driven=False, disturbed=()):
            for i in ii:
                s = specs[i]
                self.vmeta.append({"kind": kind, "recipe": s.get("recipe", "G"), "twin_of": s.get("twin_of"),
                                   "held": s.get("held"), "n": len(s["mine"]), "spec": i})
            super().flush(ii, specs, kind, net_driven, disturbed)
    return _VB()


def ask_T(net, rows, cards, meta, mb=1024, dev="cpu"):
    """(module doc) -> (imitate.Marks of path T, margins [rows of Marks] float)."""
    from net.tactic import heads as HD
    from net.tactic import imitate as IM
    from net.tactic import learn as LN
    mT = IM.Marks(net, "T")
    mT.meta = list(meta)
    side = isinstance(net, HD.TacticNet3) and "ap" in rows
    M = int(rows["b"].shape[0])
    out, margins = [], []
    net.eval()
    for i in range(0, M, mb):
        idx = torch.arange(i, min(M, i + mb))
        batch = RC.from_netbuffer({**rows, "cards": cards}, idx).to(dev)
        with torch.no_grad():
            lo = LT.logits(net, batch)
        lg = lo["pointer"].cpu()
        t = {k: v[idx] for k, v in rows.items()}
        nhead = t["v_nhead"].bool()
        c = lg.argmax(-1)
        k = torch.where(nhead & (c > 0), c - 1, torch.full_like(c, -1))
        top = lg.topk(2, -1).values
        mg = top[..., 0] - top[..., 1]
        head = t["head"].bool()
        if "done_own" in t:                            # (the engine's own post-finish guard)
            head = head & ~t["done_own"].bool()[:, None]
        label = t["label"].long()
        k = torch.where(head, k, torch.full_like(k, -1))
        kc = k.clamp(min=0)
        dcp = t["dc"].bool().gather(2, kc[..., None])[..., 0]
        code = torch.where(k < 0, 0, torch.where(k == label, 1, torch.where(dcp, 2, 3)))
        pin = t["v_pin"].bool()
        far = t["v_far"].bool().gather(2, kc[..., None])[..., 0]
        unc = (k >= 0) & (~pin.gather(1, kc) | far)
        r, g = head.nonzero(as_tuple=True)
        row = {"b": t["b"][r].long(), "sec": t["v_sec32"][r], "att": (label >= 0)[r, g], "drop": t["drop"].bool()[r, g],
               "kinds": t["kinds"][r, g], "near2": t["near2"].bool()[r, g], "code": code[r, g].to(torch.int8),
               "unc": unc[r, g], "e": t["v_e"][r, g], "es": t["v_es"][r, g]}
        if side:
            lap = t["ap"].long()
            nap = lo["side"].cpu().argmax(-1)
            ok_n = label >= 0
            # (masks.of - the one definition: the band and the closed row of the approach head; on the rows the side
            # marks read - decided, an attack label with an approach - equal to learn.side_band / imitate.open_at)
            ma = MK.of("tactic_approach", batch)
            sb = ((ma.why & MK.R_SIDE_BAND) != 0).cpu()
            sc = ((ma.why & MK.R_SIDE_CLOSED) != 0).cpu()
            row.update(lap=lap[r, g], nap=torch.where(ok_n, nap, torch.full_like(nap, -1))[r, g], sband=sb[r, g],
                       sclosed=sc[r, g])
        out.append(row)
        margins.append(mg[r, g])
    mT.rows = [{k: torch.cat([x[k] for x in out], 0) for k in out[0]}] if out else []
    return mT, (torch.cat(margins) if margins else torch.zeros(0))


def MarksN(net):
    """imitate.Marks for path N with the engine's own post-finish guard: the battles over (st.done, collect.own_done)
    taken out of the head-free units before the legacy hook sees them."""
    from net.tactic import imitate as IM
    from net.drill.levels.tactic_collect import own_done

    class _M(IM.Marks):
        own_cnt = None

        def __call__(self, v, T, drv, lab=None):
            dn = own_done(v).to(T["head_free"].device)
            # (counted BEFORE the removal: head-free group-ticks and those of battles over - the legacy done_cnt
            # sees none of them once they are taken out, validate.py:224-227 of round 1)
            h0 = T["head_free"].any(1)
            if self.own_cnt is None:
                self.own_cnt = [0, 0]
            self.own_cnt[0] += int(h0.sum())
            self.own_cnt[1] += int((h0 & dn).sum())
            if bool(dn.any()):
                T = {**T, "head_free": T["head_free"] & ~dn[:, None]}
            n0 = len(self.pending)
            out = super().__call__(v, T, drv, lab)
            last = getattr(drv, "last", None) or {}
            if len(self.pending) > n0 and "lap" in self.pending[-1] and "approach" in last:
                # (brief_A3 R3: the network's approach at ITS pick, whatever the target - a flank order on a wrong
                # target is a false flank on a front label; the legacy row's nap is -1 there)
                head = T["head_free"] & ~IM.done_rows(v)[:, None]
                r, g = head.nonzero(as_tuple=True)
                k = last["k"]
                self.pending[-1]["nap_any"] = torch.where(k >= 0, last["approach"], torch.full_like(k, -1))[r, g].cpu()
            return out
    return _M(net, "N")


def v_parts(smoke, side, chunk):
    from net.tactic import imitate as IM
    from net.tactic import learn as LN
    sc, gen = IM.v_specs(smoke, side)
    t0 = int(IM.L()["seeds"]["V_tseed"])
    ch, chg = min(chunk, len(sc)), min(chunk, len(gen))
    return [("scene", sc, True, LN._seeds(t0, len(sc), ch), ch), ("general", gen, False, LN._seeds(t0 + 50, len(gen), chg),
                                                                   chg)]


def v_key(side, smoke, chunk, device, parts, teacher):
    """V's cache key: validate.play_key's runtime + the level's: the simulator key, the executor / teacher / unit-script
    files (EXEC_FILES, the bridge's RUN_MARGIN included - net/unit/view.py:72), the recorder's files, the configs (the
    tactic's whole, stage0's whole - its lord too), the loaded weights (card8, the unit network), the ordered specs WITH
    their annotations (twin_of, held, recipe, family: the cached meta depends on them), the chunk and the seeds."""
    from net.stage0 import cfg as scfg
    from net.tactic import cfg as tcfg
    from net.tactic import exam_pin as EP
    from net.tactic import ckpt as TK
    from net.drill import seeds as SD
    tc = tcfg()
    specs = [s for _, sp, _, _, _ in parts for s in sp]
    unit = ROOT / str(tc["train"].get("unit_ckpt", ""))
    ann = [hashlib.sha256(json.dumps({k: s.get(k) for k in SD.ANNOTATIONS}, sort_keys=True,
                                     default=str).encode()).hexdigest()[:12] for s in specs]
    x = {"sim_key": EP.sim_key(), "files": VA.files_sha(EXEC_FILES), "recorder": VA.files_sha(RECORDER_FILES),
         "watch": VA.WATCH_SCHEMA, "teacher": teacher, "configs": {"tactic": tc, "stage0": scfg()},
         "card8": TK.card8_sha(), "unit": VA.sha16(unit) if unit.is_file() else tc["train"].get("unit_ckpt"),
         "specs": [SD.spec_sha(s) + ":" + an for s, an in zip(specs, ann)],
         "seeds": [s for p in parts for s in p[3]], "chunk": int(chunk), "smoke": smoke, "side": side}
    return VA.play_key(device, x)


def _play_T(parts, a, P, teacher, side, st, log, first_chunk=False):
    """Path T into the store st (a shard a chunk); first_chunk: only the first part's first chunk (the replay)."""
    from net.tactic import exam_pin as EP
    from net.tactic import imitate as IM
    buf = VBuffer(side)
    for kind, specs, scene, seeds, chunk in parts:
        if first_chunk:
            order = sorted(range(len(specs)), key=lambda i: int(specs[i]["limit"]))[:chunk]
            specs, seeds = [specs[i] for i in order], seeds[:1]

        def flush(ii, specs=specs, kind=kind):
            b0, t0 = len(buf.meta), len(buf.ticks)
            m0 = len(buf.vmeta)
            buf.flush(ii, specs, kind)
            new = buf.ticks[t0:]
            rows = {k: torch.cat([t[k] for t in new], 0) for k in new[0]} if new else {}
            if rows:
                rows["b"] = rows["b"] - b0
            st.put(kind, rows, buf.vmeta[m0:], buf.cards[b0:], maps=LT.map_table(rows))
            del buf.ticks[t0:]
        IM.play(teacher, specs, scene, seeds, chunk, a, P, lambda ii: EP.build_under(teacher), lambda d_: buf.hook(False),
                flush, needs_input=True, labeller=teacher)
        log({"V_teacher_played": kind, "battles": len(specs), "shards": len(st.manifest["shards"]),
             "replay": first_chunk})
        if first_chunk:
            break
    return {"done_cnt": buf.done_cnt}


def teacher_path(cache, key, parts, a, P, teacher, side, log=print, replay=True):
    """(rows, cards, meta, info, how) of path T from the cache (module doc)."""
    tc = VA.TeacherCache(cache)
    how = tc.ensure(key, lambda st: _play_T(parts, a, P, teacher, side, st, log),
                    (lambda st: _play_T(parts, a, P, teacher, side, st, log, first_chunk=True)) if replay else None,
                    remap=LT.remap, log=log)              # (finding 7: the stored shard remapped before the compare)
    rows, cards, meta, info = tc.read(key, remap=LT.remap, ram_gb=VA.ram_gb())
    return rows, cards, meta, info, how


def v_read(net, a, P, chunk, smoke=0, side=False, cache=None, log=print, margins_out=None, replay=True, teacher=None):
    """(module doc); teacher - the card's (card.teacher), default TEACH2 for the side drill, else TEACH."""
    from net.tactic import imitate as IM
    teacher = teacher or ("TEACH2" if side else "TEACH")
    parts = v_parts(smoke, side, chunk)
    key = v_key(side, smoke, chunk, a.device, parts, teacher)
    want = getattr(a, "v_key", None)
    if want is not None and want != key:
        # (finding 1: the run pinned its V play key at the start - a file or runtime under V changed since)
        raise SystemExit(f"V's play key {key} != the run's pinned {want}: the executor / teacher / configs / runtime "
                         "under V changed during the run - V0, the reference's marks and the stop would mix two plays "
                         "(refused; restore the files or start a fresh --out)")
    chk = getattr(a, "check_starts_v", None)
    if chk is not None:                                  # (finding 6: V's realised starts in the registry's V split)
        from net.drill.levels.tactic_collect import starts_of
        n_chk = sum(chk([("torch.chunk", None, int(x)) for x in seeds] + [t for s_ in sp for t in starts_of(s_)])
                    for _, sp, _, seeds, _ in parts)
        log({"V_starts_checked": n_chk})
    mN = MarksN(net)
    # path N first, as the legacy read (IM.read: N then T, per part)
    for kind, sp, scene, seeds, ch in parts:
        IM.read(net, sp, scene, seeds, ch, a, P, kind, mN, IM.Marks(net, "T"), paths="N", teacher=teacher)
    rows, cards, meta, info, how = teacher_path(cache, key, parts, a, P, teacher, side, log, replay)
    mT, mg = ask_T(net, rows, cards, meta, dev=a.device)
    mT.done_cnt = list(info["done_cnt"])
    if margins_out is not None:
        margins_out.append(mg)
    out = IM.marks(mN, mT, "V")
    out["done_rows"] = {"N": list(mN.done_cnt), "T": list(mT.done_cnt), "N_engine": list(mN.own_cnt or [0, 0])}
    keep = {}
    if side:
        from net.tactic import learn as LN
        keep = {"retention": lambda m: m["recipe"] not in LN.SIDE_RECIPES, "S2": lambda m: m["recipe"] in LN.SIDE_RECIPES}
        out["parts"] = {nm: IM.marks(IM.subset(mN, f), IM.subset(mT, f), "V") for nm, f in keep.items()}
    # (every clustered mark's error by battle - validate module doc; brief 22.11 (c))
    out["battle_keys"], bbs = by_battle(out, mN, mT, parts, keep, teacher)
    if side:
        out["side_engine"] = {"N": side_marks_engine(mN, bbs.get("N")), "T": side_marks_engine(mT, bbs.get("T"))}
    out["cache"] = {"key": key, "teacher_path": how}
    log({"V": how, "key": key})
    return out, mN, mT


def side_marks_engine(m, bb=None):
    """(module doc) {group: {front, left, right, false_flank, missing}} - groups 'all', 'S2', 'S2N'; bb (ByBattle of
    this read): each cell's errors by battle."""
    from net.tactic import learn as LN
    from net.tactic import orders as OR
    if not m.rows or "lap" not in m.rows[0]:
        return None
    d = m.data()
    gr = [LN.side_group(x) for x in m.meta]
    gb = [gr[int(b)].split(" held")[0] for b in d["b"].tolist()]
    sb = d["sband"] if "sband" in d else torch.zeros_like(d["drop"])
    scl = d["sclosed"] if "sclosed" in d else torch.zeros_like(d["drop"])
    ok = ~d["drop"] & (d["lap"] >= 0) & ~sb & ~scl     # (no condition on nap: a wrong-target / KEEP pick stays in)
    lap, nap = d["lap"].long(), d["nap"].long()
    nany = d["nap_any"].long() if "nap_any" in d else torch.full_like(nap, -1)
    flank_any = (nap > 0) | ((nap < 0) & (nany > 0))
    out = {}
    for name in ("all", "S2", "S2N"):
        grp = torch.tensor([name == "all" or x == name for x in gb], dtype=torch.bool)
        sel = ok & grp
        cs = {c: (sel & (lap == i) & (nap == i), sel & (lap == i)) for i, c in enumerate(OR.APPROACHES)}
        # (a flank order on a wrong target counted as a false flank on a front label - brief_A3 R3)
        cs["false_flank"] = (sel & (lap == 0) & flank_any, sel & (lap == 0))
        cs["missing"] = (sel & (nap < 0), sel)
        res = {c: LN.share(int(x.sum()), int(n.sum())) for c, (x, n) in cs.items()}
        if bb is not None:
            for c, (x, n) in cs.items():
                bb.put(res[c], x, n)
        out[name] = res
    return out


# ------------------------------------------------------------------ the marks' errors by battle
def battle_order(meta, parts, teacher=None, with_ids=False):
    """(order, roster sha[, ids]) of a read's battles: its meta sorted by (kind, spec index); a battle's id is the
    sha of (kind, index, spec_sha, its annotations - recipe, seed, twin identity: seeds.ANNOTATIONS); the roster
    sha covers the ids in order, the teacher and each part's chunk and chunk seeds (Astra engine_code3 finding 1:
    a changed chunk seed or recipe on the same geometry is another roster) - two reads of the same roster pair battle
    by battle, whatever their play order."""
    from net.drill import seeds as SD
    sp = {kind: specs for kind, specs, _, _, _ in parts}

    def bid(m):
        x = sp[m["kind"]][int(m["spec"])]
        ann = {k: x.get(k) for k in SD.ANNOTATIONS}
        return hashlib.sha256(json.dumps([m["kind"], int(m["spec"]), SD.spec_sha(x), ann], sort_keys=True,
                                         default=str).encode()).hexdigest()[:12]
    order = sorted(range(len(meta)), key=lambda i: (meta[i]["kind"], int(meta[i]["spec"])))
    ids = [bid(meta[i]) for i in order]
    head = {"teacher": teacher, "parts": [[kind, int(ch), [int(x) for x in seeds]] for kind, _, _, seeds, ch in parts]}
    sha = hashlib.sha256(json.dumps([head, ids], sort_keys=True).encode()).hexdigest()[:16]
    return (order, sha, ids) if with_ids else (order, sha)


class ByBattle:
    """A read's per-battle sums (rows' battle index b, B battles, battle_order's order): put(cell, x, n, keep) - x, n
    row masks (or battle vectors, battle=True), keep a battle mask (a part) - checks the sums against the cell's own
    share and n (a drifted definition raises), keeps the binomial error as two_errors_tick and sets two_errors to the
    error by battle (validate.cluster_two_err) and bb {keys, x, n}."""

    def __init__(self, b, B, order, sha, ids=None):
        self.b, self.B, self.order, self.sha, self.ids = b.long(), B, torch.tensor(order, dtype=torch.long), sha, ids

    def per(self, m):
        c = torch.zeros(self.B, dtype=torch.long)
        c.index_add_(0, self.b[m], torch.ones(int(m.sum()), dtype=torch.long))
        return c

    def put(self, cell, x, n, keep=None, battle=False):
        if not cell or not cell.get("n"):
            return cell
        xb, nb = (x.long(), n.long()) if battle else (self.per(x), self.per(n))
        if keep is not None:
            xb, nb = xb * keep.long(), nb * keep.long()
        X, N = int(xb.sum()), int(nb.sum())
        if N != int(cell["n"]) or abs(X / N - float(cell["share"])) > 6e-5:
            raise AssertionError(f"by battle: x {X} / n {N} against the cell's share {cell['share']} / n {cell['n']} "
                                 "(the definitions drifted)")
        xl, nl = xb[self.order].tolist(), nb[self.order].tolist()
        cell["two_errors_tick"] = cell.get("two_errors")
        sup = VA.support(nl)
        e = VA.cluster_two_err(xl, nl) if sup["estimable"] else None
        if e is not None:
            cell["two_errors"] = e
        # (finding 2: the support said beside the error; not estimable - the tick error stays, named so, and the
        # stop does not judge the mark - validate.compare)
        cell.update(battles=sup["battles"], top_share=sup["top_share"],
                    error="by battle" if e is not None else f"by tick (by battle not estimable: {sup['battles']} "
                                                            f"battles < {VA.MIN_BATTLES})")
        cell["bb"] = {"keys": self.sha, "x": xl, "n": nl, **({"ids": self.ids} if self.ids is not None else {})}
        return cell


def _followed_N(m, d, lab):
    """imitate.marks' followed rows of path N (the label's target and side held by the network within -1..+2 ticks)."""
    from net.tactic import cfg as tcfg
    step = int(tcfg()["decide_s"])
    fol = torch.zeros_like(lab)
    bl, sl, el, sd = d["b"].tolist(), d["sec"].tolist(), d["e"].tolist(), d["es"].tolist()
    for i in lab.nonzero()[:, 0].tolist():
        c = el[i] * 4 + sd[i]
        fol[i] = any(c in m.holds.get((bl[i], sl[i] + k * step), ()) for k in (-1, 0, 1, 2))
    return fol


def by_battle(out, mN, mT, parts, keep_parts=None, teacher=None):
    """The errors by battle of V's clustered marks (validate module doc), in place on out (imitate.marks' table):
    followed pooled and false a tick on both paths, uncaused and no-cause on N, the same in out['parts'] (keep_parts
    {part: battle filter on the meta}) -> ({path: keys sha}, {path: ByBattle}); the side engine's cells get theirs in
    side_marks_engine(m, bb)."""
    res, bbs = {}, {}
    for path, m in (("N", mN), ("T", mT)):
        if m is None or not m.rows:
            continue
        d = m.data()
        order, sha, ids = battle_order(m.meta, parts, teacher, with_ids=True)
        bb = ByBattle(d["b"], len(m.meta), order, sha, ids)
        bbs[path], res[path] = bb, sha
        ok = ~d["drop"]
        code = d["code"].long()
        lab = d["att"] & ok
        fol = _followed_N(m, d, lab) if path == "N" else (code == 1)
        fal = ok & (code == 3)
        tabs = [(out, None)] + [((out.get("parts") or {}).get(nm), torch.tensor([bool(f(x)) for x in m.meta]))
                                for nm, f in (keep_parts or {}).items()]
        for t, keep in tabs:
            if not t:
                continue
            bb.put((t.get(f"followed_{path}") or {}).get("pooled"), fol & lab, lab, keep)
            bb.put((t.get(f"false_{path}") or {}).get("tick"), fal, ok, keep)
            if path == "N":
                pick = code > 0
                bb.put(t.get("uncaused"), d["unc"] & pick, pick, keep)
                nc = torch.tensor([x["kind"] == "scene" for x in m.meta]) & (bb.per(lab) == 0)
                bb.put(t.get("nocause_false"), nc & (bb.per(fal) > 0), nc, keep, battle=True)
    return res, bbs
