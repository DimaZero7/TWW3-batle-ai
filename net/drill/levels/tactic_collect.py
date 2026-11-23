"""The tactic level's collection (moved from net/drill/collect.py; tmp/engine/design_merged.md section 4): the live
runner exam_pin.run_player under tags.Record (its hook = imitate.NetBuffer, the recorder); the teacher drives
(disturbed) or the network drives net_share; every chunk's rows go into a store shard as they are flushed.

    ShardBuffer(store, side, check)   imitate.NetBuffer whose flush writes the chunk's rows / meta / cards as one shard;
                the trivial-thinning counter continues across shards (one buffer a collection, imitate.py:177-184)
    play_chunks(...)   imitate.play with the level's own input built for the first needs_input_first chunks; on_chunk_start
                (ii, torch seed) is called before each chunk plays (the start check)
    starts_of(spec)    the realised starts of a built battle: (generator, key, seed) - a twin's base at its realised
                (retry) seed too
    tactic_round(r, net, a, P, chunk, no_general, smoke, log, side, store_dir, check_teacher, check_starts)   a
                training round's collection, the draw of imitate.collect (the same specs, Random streams, disturbance
                plans and chunk seeds: a same-tree legacy run gives the same rows), plus the rebuild check on
                teacher-driven rows; check_starts([(gen, key, value)]) once a chunk (seeds.assert_starts)
"""
import copy
import random
import time

import torch

from net.drill.levels import tactic as LT
from net.drill.store import Store


def _tac():
    from net.stage0 import cfg as scfg
    from net.tactic import exam_pin as EP
    from net.tactic import imitate as IM
    from net.tactic import learn as LN
    from net.tactic import pin as PN
    from net.tactic import tags as TG
    return scfg, EP, IM, LN, PN, TG


def own_done(v):
    """[R] bool: the view's battles over, read from the simulator's own st.done (not imitate.done_rows): the engine's
    second guard against post-finish rows (a finished battle's frozen state seen again, REVIEW2 finding 3)."""
    st = v.sim
    dn = getattr(st, "done", None) if st is not None else None
    if dn is None:
        return torch.zeros(v.idx.shape, dtype=torch.bool)
    return dn.to(v.idx.device)[v.idx].bool()


class Seen:
    """The driver with its last output kept (Record calls main(v); every attribute forwarded)."""

    def __init__(self, drv):
        self.__dict__["_drv"], self.__dict__["out"] = drv, None

    def __call__(self, v):
        self.__dict__["out"] = self._drv(v)
        return self.out

    def __getattr__(self, k):
        return getattr(self._drv, k)

    def __setattr__(self, k, x):
        setattr(self._drv, k, x)


def picks(v, seen):
    """([R, G], [R, G]) the driver's candidate this tick (KEEP 0, an attack row + 1, NONE -1: a target with no row) -
    proposed: its pick; executed: the pick when it changes the order in force (tags.tick's 'att' rule), else KEEP;
    (None, None) without the driver's output."""
    from net.unit import scripts as SC
    out = getattr(seen, "out", None) if seen is not None else None
    if not isinstance(out, dict) or "target" not in out or "skill" not in out:
        return None, None
    fx, o = v.fx, v.order
    t, sk = out["target"], out["skill"]
    att = fx.able & (sk == SC.ATTACK) & (t >= 0)
    k = fx.row_of_slot(torch.where(att, t, torch.full_like(t, -1)))
    cand = torch.where(att, torch.where(k >= 0, k + 1, torch.full_like(k, -1)), torch.zeros_like(k))
    change = att & ((t != o["tgt"]) | (o["skill"] != SC.ATTACK))
    return cand, torch.where(change, cand, torch.zeros_like(cand))


def ShardBuffer(store, side=False, check=0):
    """(module doc) a NetBuffer subclass instance (built lazily: imitate imports the simulator)."""
    _, _, IM, LN, _, _ = _tac()

    class _SB(IM.NetBuffer):
        def __init__(self):
            super().__init__(check=check, side=side)
            self.store, self.part, self.n_shards = store, None, 0

        def __call__(self, v, T, net_driven, lab=None):
            # (additive: the label drop's T0-1 part, dropped | dc_band, kept beside the legacy 'drop' so masks.tactic_drop
            # recomputes imitate.drop_of from its parts)
            n0, nt = len(self.pending), self.n_triv
            super().__call__(v, T, net_driven, lab)
            if len(self.pending) > n0:
                out = self.pending[-1]
                # (the rows NetBuffer kept, recomputed with its own thinning from the counter before the call)
                head = T["head_free"] & ~IM.done_rows(v)[:, None]
                sel = head.any(1)
                triv = sel & ~(head & ~T["trivial"]).any(1)
                for r in triv.nonzero()[:, 0].tolist():
                    nt += 1
                    if nt % self.every:
                        sel[r] = False
                assert nt == self.n_triv and int(sel.sum()) == int(out["b"].shape[0])
                base = (T["dropped"] | T["dc_band"])[sel]
                out["drop_base"] = LN._pad_g(base, False).cpu()
                out["done_own"] = own_done(v)[sel].cpu()
                if "dc_thr" not in out:                     # (drill T0-1: the through-own part kept too)
                    out["dc_thr"] = LN._pad_g(T["dc_thr"][sel], False).cpu()
                pr, ex = picks(v, getattr(self, "seen", None))
                if pr is not None:
                    out["proposed"] = LN._pad_g(pr[sel], -1).cpu()
                    out["executed"] = LN._pad_g(ex[sel], -1).cpu()

        def flush(self, ii, specs, kind, net_driven, disturbed=()):
            t0, b0 = len(self.ticks), len(self.meta)
            super().flush(ii, specs, kind, net_driven, disturbed)
            new = self.ticks[t0:]
            rows = {k: torch.cat([t[k] for t in new], 0) for k in new[0]} if new else {}
            if rows:
                rows["b"] = rows["b"] - b0
            self.store.put(self.part or kind, rows, self.meta[b0:], self.cards[b0:],
                           stats={"done": {k: list(v) for k, v in self.done_stats.items()},
                                  "checked": self.checked, "check_max": self.check_max},
                           maps=LT.map_table(rows))
            del self.ticks[t0:]                     # (the rows live in the store; the mixture is read back from it)
            self.n_shards += 1
    return _SB()


def play_chunks(name, specs, scene, seeds, chunk, a, P, make_driver, hook_of, on_flush, labeller, needs_input_first=0,
                watch=None, on_chunk_start=None, on_driver=None):
    """imitate.play with the level's own input built for the first needs_input_first chunks (the rebuild check on
    teacher-driven rows); otherwise the same call (order by limit, chunks, Record under the labeller's own name)."""
    _, EP, IM, _, _, TG = _tac()
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["limit"]))
    chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
    state = [0]
    drivers = []

    def make():
        ii = chunks[state[0]]
        if on_chunk_start is not None:
            on_chunk_start(ii, seeds[state[0]])
        state[0] += 1
        drv = make_driver(ii)
        drivers.append(drv)
        main = drv
        if on_driver is not None:                      # (the driver's output seen by the recorder: proposed / executed)
            main = Seen(drv)
            on_driver(main)
        rec = TG.Record(main, EP.build_under(labeller), hook=hook_of(drv))
        if state[0] <= needs_input_first:
            rec.needs_input = True
        return rec
    res = EP.run_player(name, specs, seeds, chunk, a, P, make=make, on_chunk=lambda ii, w, s: on_flush(ii),
                        watch_cls=watch, full=True, by_limit=True, **IM._kw(scene))
    return res, drivers


def starts_of(sp):
    """(module doc)."""
    rec = sp.get("recipe")
    if rec in ("S2", "S2N"):
        return [("tactic.side", None, int(sp["seed"]))]
    if rec is None or "seed" not in sp:
        return []                                      # (a general: its Random stream's seed is the round's)
    out = [("tactic.make", rec, int(sp["seed"]))]
    tw = sp.get("twin_of")
    if tw:
        out.append(("tactic.make", tw[0], int(tw[1])))
    return out


def tactic_round(r, net, a, P, chunk, no_general, smoke, log, side, store_dir, check_teacher=True, check_starts=None,
                 teacher=None):
    """(module doc) -> (Store, stats)."""
    scfg, EP, IM, LN, PN, TG = _tac()
    c = IM.L()
    sd = c["seeds"]
    n_sc = float(scfg()["drills"]["attack"]["sets"]["round"]["scenes"])
    n_g = float(scfg()["drills"]["attack"]["sets"]["round"]["general"])
    scale, gscale = (1.0, 1.0) if not smoke else (smoke / n_sc, smoke / n_g)
    scenes, gen = LN.round_specs(r, scale, gscale, seeds=sd)
    if smoke:
        scenes, gen = IM._limit(scenes[:smoke], 200), IM._limit(gen[:smoke], 200)
    if side:
        scenes = scenes + (IM._limit(IM.side_round(r, smoke), 200) if smoke else IM.side_round(r))
    teacher = teacher or ("TEACH2" if side else "TEACH")          # (the card names its teacher: card.teacher)
    store = store_dir if isinstance(store_dir, Store) else Store(store_dir)
    budget = 64 if smoke else 32                     # (the legacy check's rows; split teacher / network below)
    buf = ShardBuffer(store, side=side, check=0)
    rnd = random.Random(int(sd["tseed"]) * 1000 + r)
    dc = c["disturb"]
    stats = {"battles_teacher": 0, "battles_net": 0, "disturbed": 0, "send": 0, "hold": 0, "held_ticks": 0,
             "rebuild_checked_teacher": 0, "rebuild_checked_net": 0}
    parts = [("scene", scenes, True, 0)] + ([] if no_general else [("general", gen, False, 50)])
    n_checked = [0]
    if check_starts is not None and not no_general:
        n_checked[0] += check_starts([("random.general", None, int(sd["general"]) + r)])

    def on_start(sp_):
        def f(ii, tseed):
            if check_starts is not None:
                n_checked[0] += check_starts([("torch.chunk", None, int(tseed))]
                                             + [s for i in ii for s in starts_of(sp_[i])])
        return f
    for kind, sp, scene, off in parts:
        if r == 0 and not smoke:
            sp_t, sp_n = sp, []
        else:
            cut = int(round(len(sp) * (1.0 - float(c["net_share"]))))
            perm = list(range(len(sp)))
            rnd.shuffle(perm)
            sp_t, sp_n = [sp[i] for i in sorted(perm[:cut])], [sp[i] for i in sorted(perm[cut:])]
        dis = set(rnd.sample(range(len(sp_t)), int(round(float(dc["share"]) * len(sp_t)))))
        plans = {i: {"kind": rnd.choice(("send", "hold")), "left": rnd.randint(*dc["events"])} for i in dis}
        dseed = rnd.randrange(1 << 30)
        stats["battles_teacher"] += len(sp_t)
        stats["battles_net"] += len(sp_n)
        stats["disturbed"] += len(dis)
        t0 = time.time()
        if sp_t:
            ch = min(chunk, len(sp_t))
            buf.part, n0 = f"{kind}_teacher", buf.checked
            buf.check = budget // 4 if check_teacher else 0
            _, drv = play_chunks(teacher, sp_t, scene, LN._seeds(int(sd["tseed"]) + 100 * r + off, len(sp_t), ch), ch,
                                 a, P,
                                 lambda ii: IM.Disturb(EP.build_under(teacher),
                                                       {j: copy.deepcopy(plans[i]) for j, i in enumerate(ii) if i in plans},
                                                       dseed + ii[0]),
                                 lambda d_: buf.hook(False), lambda ii: buf.flush(ii, sp_t, kind, False, dis),
                                 labeller=teacher, needs_input_first=1 if check_teacher else 0,
                                 on_chunk_start=on_start(sp_t), on_driver=lambda s: setattr(buf, "seen", s))
            stats["rebuild_checked_teacher"] += buf.checked - n0
            for d_ in drv:
                for k in ("send", "hold", "held_ticks"):
                    stats[k] += d_.n[k]
        if sp_n:
            net.eval()
            ch = min(chunk, len(sp_n))
            buf.part, n0 = f"{kind}_net", buf.checked
            buf.check = budget // 4
            play_chunks("N", sp_n, scene, LN._seeds(int(sd["tseed"]) + 100 * r + off + 25, len(sp_n), ch), ch, a, P,
                        lambda ii: PN.NetPlayer(net, greedy=True), lambda d_: buf.hook(True),
                        lambda ii: buf.flush(ii, sp_n, kind, True), labeller=teacher, on_chunk_start=on_start(sp_n),
                        on_driver=lambda s: setattr(buf, "seen", s))
            stats["rebuild_checked_net"] += buf.checked - n0
        log({"collect": kind, "round": r + 1, "teacher": len(sp_t), "net": len(sp_n), "disturbed": len(dis),
             "group_ticks_total": sum(s["rows"] for s in store.manifest["shards"]),
             "min": round((time.time() - t0) / 60, 1)})
    stats.update(starts_checked=n_checked[0], rebuild_checked=buf.checked, rebuild_max_diff=buf.check_max,
                 done_rows={k: list(v) for k, v in buf.done_stats.items()}, NK=list(buf.NK) if buf.NK else None)
    if buf.checked == 0:
        raise RuntimeError("the rebuild check made 0 comparisons (design section 7: it must make non-zero ones)")
    if buf.check_max > 1e-3:                          # (imitate.py:1015)
        raise RuntimeError(f"the rebuilt input differs from the level's: {buf.check_max}")
    return store, stats
