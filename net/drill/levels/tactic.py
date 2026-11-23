"""The tactic level's adapter (tmp/engine/design_merged.md sections 2-3; the interface of net/drill/levels/__init__.py):
tick 5 s, unit = slot g of a group-tick, candidates KEEP + the walk-open target rows, then the conditional approach
(heads.py:186-200). Execution stays in the live runner (exam_pin.run_player under tags.Record, its hook = the recorder:
levels/tactic_collect.py; V: levels/tactic_v.py; census classes: levels/tactic_census.py). A card's level_args:
{"drill": "attack" | "side", "tags": checkpoint drill tags} - 'side' adds the side scenes, TEACH2, the side head's pre-
pass, the side census classes and V's side parts.

    rebuild(batch)          inputs.build of stored rows (the map windows rebuilt; imitate.rebuild)
    losses(net, batch, w)   {"pointer": heads.loss on the recorded legal / dont_care, "side": the side head's loss on
                            masks.of('tactic_approach') - a TacticNet3 only, else None}
    logits(net, batch)      the pointer's logits [M, G, 1 + K] (and the side logits) of stored rows
"""
import torch

from net.drill import masks as MK
from net.drill import record as RC

tick_s = 5.0


# ------------------------------------------------------------------ the maps of stored rows
# A stored row's raw carries its battle's map as terrain.register's id (raw column 'terr'), an index into a registry
# filled per process in the order battles were set up (net/unit/realmap._entry, net/matchup/duel.map_id): the same id in
# another process is another map or none (a fresh process rebuilt the smoke's rows with h0 233 m off). A shard keeps its
# maps' names and grid sha; reading it in another process registers them again and rewrites the ids.
def _grid_sha(mid):
    import hashlib
    from sim import terrain
    g = terrain._MAPS[mid][0]
    return hashlib.sha256(g.tobytes() + repr(terrain._MAPS[mid][1:]).encode()).hexdigest()[:16]


def _names():
    """{registry id: map folder name} of this process's loaders."""
    from net.unit import realmap as RM
    out = {e["mid"]: e["name"] for e in RM._LOADED.values()}
    try:
        from net.matchup import duel as DU
        out.update({mid: n for n, mid in DU._MAPS.items()})
    except ImportError:
        pass
    return out


def map_table(rows):
    """{str(id): {name, sha}} of the maps the rows' raw uses (none: {})."""
    from net.tactic import raw as TR
    if not rows or "raw" not in rows:
        return {}
    ids = torch.unique(rows["raw"][..., TR.I["terr"]].round().long())
    nm = _names()
    out = {}
    for i in ids.tolist():
        if i < 0:
            continue
        if i not in nm:
            raise RuntimeError(f"map id {i} of the stored rows has no loader name in this process")
        out[str(i)] = {"name": nm[i], "sha": _grid_sha(i)}
    return out


def remap(rows, table):
    """The rows' raw map ids rewritten to this process's registry (registering a map by name when absent); a grid whose
    sha differs halts."""
    from sim import terrain
    from sim.maps import MAPS
    from net.tactic import raw as TR
    from net.unit import realmap as RM
    if not table:
        return rows
    sub = {}
    for s, e in table.items():
        i = int(s)
        if i < len(terrain._MAPS) and _grid_sha(i) == e["sha"]:
            sub[i] = i
            continue
        mid = next((j for j in range(len(terrain._MAPS)) if _grid_sha(j) == e["sha"]), None)
        if mid is None:
            mid = RM._entry(MAPS / e["name"])["mid"]
        if _grid_sha(mid) != e["sha"]:
            raise RuntimeError(f"map {e['name']}: its grid differs from the stored rows' (sha {e['sha']})")
        sub[i] = mid
    if all(k == v for k, v in sub.items()):
        return rows
    raw = rows["raw"].clone()
    col = raw[..., TR.I["terr"]]
    new = col.clone()
    for i, j in sub.items():
        new[col.round().long() == i] = float(j)
    raw[..., TR.I["terr"]] = new
    return {**rows, "raw": raw}


def K_of(batch):
    return int(batch.legal.shape[-1]) - 1


def rebuild(batch):
    from net.tactic import imitate as IM
    t = RC.to_netbuffer(batch)
    with torch.no_grad():
        return IM.rebuild(t, batch.obs["card"], K_of(batch))


def _enc(net, batch, inp):
    from net.tactic import heads as HD
    o = batch.obs
    g_row = o["g_row"].long()
    if isinstance(net, HD.TacticNet3):
        return net.enc(inp, batch.order["command"], o["cols"], o["own"], g_row, o["g_on"], o["s2"])
    return net.enc(inp, batch.order["command"], o["cols"], o["own"], g_row, o["g_on"])


def logits(net, batch, inp=None):
    """{pointer [M, G, 1 + K], side [M, G, 3] at the teacher's label (TacticNet3), enc}."""
    from net.tactic import heads as HD
    from net.tactic import learn as LN
    inp = rebuild(batch) if inp is None else inp
    enc = _enc(net, batch, inp)
    o = batch.obs
    g_row = o["g_row"].long()
    lg = net.ptr.logits_all(enc["cards"], o["own"], g_row, o["pair"], batch.legal[..., 1:])
    out = {"pointer": lg, "enc": enc}
    if isinstance(net, HD.TacticNet3):
        cards = enc["cards"]
        W = cards.shape[-1]
        lab = RC.label_of(batch)
        cg = cards.gather(1, g_row[..., None].expand(-1, -1, W))
        ck = cards.gather(1, lab.clamp(min=0)[..., None].expand(-1, -1, W))
        out["side"] = net.side(cg, o["own"], ck, LN.at_label(o["pair"], lab), batch.aux["x_k"].float())
    return out


def losses(net, batch, w, inp=None):
    """(module doc) w [M, G]."""
    from net.tactic import heads as HD
    lo = logits(net, batch, inp)
    # (masks.of - the one definition of "this label trains this head": heads.loss's own logit legality then agrees)
    wp = w * MK.of("tactic_pointer", batch).eligible.to(w.dtype)
    lp, _ = HD.loss(lo["pointer"], RC.label_of(batch), wp, batch.dont_care[..., 1:])
    if "side" not in lo:
        return {"pointer": lp, "side": None}
    m = MK.of("tactic_approach", batch, w).eligible
    ls, _ = HD.side_loss(lo["side"][m], batch.aux["ap"].long()[m], w[m])
    return {"pointer": lp, "side": ls}


# ------------------------------------------------------------------ the level interface (levels/__init__.py)
NAME = "tactic"
IMPL_FILES = ("net/tactic/*.py", "net/unit/*.py", "net/stage0/*.py")
pad = None


def side_of(card):
    return card.level_args.get("drill") == "side"


def conf(card):
    from net.stage0 import cfg as scfg
    from net.tactic import cfg as tcfg
    from net.tactic import ckpt as TK
    from net.tactic import imitate as IM
    import copy
    import hashlib
    from pathlib import Path
    c = IM.L()
    tc = copy.deepcopy(tcfg())
    tc["train"]["learn"].pop("rounds", None)            # (the cap out of the hashed config)
    unit = Path(__file__).resolve().parents[3] / str(tc["train"].get("unit_ckpt", ""))
    return {"lr": float(c["lr"]), "minibatch": int(c["minibatch"]), "max_grad": float(c["max_grad"]),
            "tseed": int(c["seeds"]["tseed"]), "chunk": int(c["chunk"]), "rounds_registered": int(c["rounds"]),
            "pre_pass": side_of(card),
            "key": {"tactic": tc, "stage0": scfg(), "card8": TK.card8_sha(),
                    "unit": hashlib.sha256(unit.read_bytes()).hexdigest()[:16] if unit.is_file() else None}}


def sim_key():
    from net.tactic import exam_pin as EP
    return EP.sim_key()


def net_new(card, device):
    from net.tactic import heads as HD
    return HD.TacticNet2().to(device)


def net_parent(card, path, device):
    from net.tactic import ckpt as TK
    net, _ = TK.from_v2(path, device)
    return net


def net_reference(card, path, device):
    """card.reference's network in the card's own form: the side card's as its parent (v3 from v2), else as saved."""
    from net.tactic import ckpt as TK
    if side_of(card):
        return net_parent(card, path, device)
    return TK.restore(path, device)[0]


def seed_check(card, log):
    from net.tactic import imitate as IM
    if side_of(card):
        IM.seed_check(int(IM.L()["rounds"]), log)


def collect_round(card, r, net, ctx, store_dir):
    from net.drill.levels import tactic_collect as TC
    from net.drill.store import Store
    st = Store(store_dir, ctx.run_key, r + 1)
    return TC.tactic_round(r, net, ctx, ctx.params, int(ctx.chunk), ctx.no_general, ctx.smoke, ctx.log,
                           side_of(card), st, check_starts=ctx.check_starts, teacher=card.teacher)


def weights(card, rows, meta):
    from net.drill.levels import tactic_census as TS
    return TS.weights(rows, meta)


def batch_of(rows, cards, idx):
    return RC.from_netbuffer({**rows, "cards": cards}, idx)


def census_batch(rows):
    from net.drill.levels import tactic_census as TS
    return TS.batch_of(rows)


def census_classes(card, batch, w, meta):
    from net.drill.levels import tactic_census as TS
    return TS.classes(batch, w, side_of(card))


def census_extra(card, batch, rows, meta, w):
    from net.drill.levels import tactic_census as TS
    return TS.extra(batch, rows, meta, w, side_of(card))


def parent_masks(card):
    """(parent_fn, child_fn, classes_fn) on the parent drill's rows (the T0-1 scenes of the child's mixture): the
    parent's pointer eligibility (T0-1: through own dropped, lift=False) and the child's (masks.of as fitted);
    classes_fn(batch) -> {attack, keep} of the old-drill labels."""
    from net.tactic import learn as LN

    def old(batch, meta):
        rec = torch.tensor([m.get("recipe") not in LN.SIDE_RECIPES for m in meta], dtype=torch.bool)
        return rec[batch.b][:, None] & (batch.status == RC.DECIDED)

    def parent_fn(batch):
        return MK.of("tactic_pointer", batch, lift=False).eligible

    def child_fn(batch):
        return MK.of("tactic_pointer", batch).eligible

    def classes_fn(batch, meta):
        o = old(batch, meta)
        lab = RC.label_of(batch)
        return {"attack": o & (lab >= 0), "keep": o & (lab < 0)}
    return parent_fn, child_fn, classes_fn


def v_key(card, a, chunk):
    """V's play key of this card's run (levels/tactic_v.v_key over its V parts and teacher) - pinned by run.py."""
    from net.drill.levels import tactic_v as TV
    side = side_of(card)
    return TV.v_key(side, a.smoke, chunk, a.device, TV.v_parts(a.smoke, side, chunk), card.teacher)


def v_read(card, net, ctx, cache):
    from net.drill.levels import tactic_v as TV
    out, _, _ = TV.v_read(net, ctx, ctx.params, int(ctx.chunk), ctx.smoke, side_of(card), cache, ctx.log,
                           teacher=card.teacher)
    return out


def brief(v):
    from net.tactic import imitate as IM
    return IM.brief(v)


def save(path, net, opt, card, state):
    from net.tactic import ckpt as TK
    tags = list(card.level_args.get("tags", ()))
    if side_of(card):
        TK.save_v3(path, net, opt, tags, **state)
    else:
        TK.save_v2(path, net, opt, tags, **state)


def load(path, device):
    return torch.load(path, map_location=device, weights_only=False)
