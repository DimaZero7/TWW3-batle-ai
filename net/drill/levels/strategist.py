"""The drill engine's strategist level (net/drill/levels/__init__.py's interface; drill S0-1 'the line on the crest',
tmp/drill_s1/design_merged.md, design_opus 3 and 10). One decision a battle, at tick 0, blind: a collection is scenes +
the teacher's labels - no play for labels (the decision changes nothing before itself: the network driving a battle
would give the same row). One decision unit (G = 1); candidates C = 1 + 3: KEEP (the centre line) and the finder's <= 3
crest lines (net/strategist/crest.py), legal when valid.

    rows_of(specs, obs, labels, net_set)   the stored rows {b, sec 0, tw, net, status DECIDED, why 0, teacher, proposed,
                 executed, legal [M, 1, C], dont_care, kind, obs_<key>} - obs: net/strategist/s1_net.observe
    mask_of(legal)   the head's masks (KEEP always, a candidate when valid; never delegated) - NEEDS_A: masks.of
                 ('strategist') to hold this one definition
    balanced(rows)   CREST half the mass, KEEP's half split equally over the KEEP kinds present (KINDS) - the FORCED
                 rows (KEEP the only legal choice: no gradient) weigh 0 and stay out of every normalisation (Astra 10)
    marks(net, rows) on the UNFORCED rows {followed (CREST labels: the network's choice is the teacher's candidate),
                 false (KEEP labels: the network chose a crest), pointer}; the forced rows apart ("forced")
    the interface: NAME, tick_s, IMPL_FILES, conf, sim_key, net_new, net_parent, collect_round, remap, pad, weights,
    batch_of, losses, logits, census_batch, census_classes, census_extra, v_key, v_read, brief, save, load
"""
import torch
import torch.nn.functional as F

from net.drill import record as RC

NAME = "strategist"
tick_s = 0.0
IMPL_FILES = ("net/strategist/*.py", "net/operative/*.py", "net/tactic/*.py", "net/unit/*.py", "net/stage0/*.py",
              "net/matchup/duel.py")
KEEP = 0
KINDS = ("cause", "flat", "none", "shift", "low", "role", "general", "general_crest", "kept")


def kind_of(spec, k):
    """The row's kind for the census and the balance: a CREST label 'cause' / 'general_crest'; a KEEP label its twin's
    made kind, 'role', 'general' or 'kept' (a cause scene the teacher keeps - not drawn so: the generator asserts)."""
    s1 = spec["s1"]
    if s1.get("general"):
        return "general_crest" if k else "general"
    if spec["twin"] is None:
        return "cause" if k else "kept"
    if spec["twin"] == "role":
        return "role"
    return s1.get("twin_made") or spec["twin"]


def mask_of(legal):
    """(legal [M, 1, C], dont_care [M, 1, C]) - module doc."""
    lg = torch.as_tensor(legal, dtype=torch.bool)[:, None, :]
    return lg, torch.zeros_like(lg)


def rows_of(specs, obs, labels, net_set=frozenset(), chosen=None):
    from net.strategist import s1_net as SN
    M = len(specs)
    legal, dc = mask_of(torch.stack([o["legal"] for o in obs]))
    te = torch.tensor(labels, dtype=torch.long)
    ex = te if chosen is None else torch.tensor(chosen, dtype=torch.long)
    out = {"b": torch.arange(M), "sec": torch.zeros(M, dtype=torch.long), "tw": torch.ones(M),
           "net": torch.tensor([i in net_set for i in range(M)]), "status": torch.full((M,), RC.DECIDED, dtype=torch.int8),
           "why": torch.zeros(M, dtype=torch.int32), "teacher": te, "proposed": ex.clone(), "executed": ex.clone(),
           "legal": legal, "dont_care": dc,
           "kind": torch.tensor([KINDS.index(kind_of(s, k)) for s, k in zip(specs, labels)], dtype=torch.long)}
    for k in SN.NET_KEYS:
        if k != "legal":
            out["obs_" + k] = torch.stack([o[k] for o in obs])
    return out


def to_batch(rows, idx=None):
    from net.strategist import s1_net as SN
    if idx is None:
        idx = torch.arange(rows["b"].shape[0])
    g = lambda k: rows[k][idx]                                       # noqa: E731
    obs = {k: g("obs_" + k) for k in SN.NET_KEYS if k != "legal"}
    return RC.Batch(b=g("b"), sec=g("sec"), tw=g("tw"), net=g("net"), obs=obs, order={}, legal=g("legal"),
                    dont_care=g("dont_care"), status=g("status")[:, None], why=g("why")[:, None],
                    proposed=g("proposed")[:, None], executed=g("executed")[:, None], teacher=g("teacher")[:, None],
                    aux={"kind": g("kind")})


def _obs(batch):
    return {**batch.obs, "legal": batch.legal[:, 0]}


def unforced(rows):
    """[M] bool: >= 1 legal candidate besides KEEP (a forced row - KEEP alone - moves no logit)."""
    return rows["legal"][:, 0, 1:].any(1)


def balanced(rows):
    te, kd = rows["teacher"], rows["kind"]
    uf = unforced(rows)
    w = torch.zeros(len(te))
    m = (te > 0) & uf
    if bool(m.any()):
        w[m] = 0.5 / float(m.sum())
    ks = [k for k in range(len(KINDS)) if bool(((te == 0) & uf & (kd == k)).any())]
    for k in ks:
        mk = (te == 0) & uf & (kd == k)
        w[mk] = (0.5 if bool(m.any()) else 1.0) / len(ks) / float(mk.sum())
    return (w * float(int(uf.sum())))[:, None]


def _sh(m, hit):
    n = int(m.sum())
    if not n:
        return None
    k = float((hit & m).sum())
    q = (k + 1) / (n + 2)
    return {"p": round(k / n, 4), "two_errors": round(2 * (q * (1 - q) / n) ** 0.5, 4), "n": n}


@torch.no_grad()
def choices(net, rows, device="cpu"):
    out = []
    net.eval()
    for i in range(0, rows["b"].shape[0], 512):
        idx = torch.arange(i, min(i + 512, rows["b"].shape[0]))
        out.append(net(_obs(to_batch(rows, idx).to(device))).argmax(1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def marks(net, rows, device="cpu"):
    k = choices(net, rows, device)
    te = rows["teacher"]
    uf = unforced(rows)
    two = rows["legal"][:, 0, 1:].sum(1) >= 2
    return {"followed": _sh((te > 0) & uf, k == te), "false": _sh((te == 0) & uf, k > 0),
            "pointer": _sh((te > 0) & uf & two, k == te),
            "forced": {"n": int((~uf).sum()), "kept": _sh(~uf, k == 0)}}


# ------------------------------------------------------------------ the interface
def conf(card):
    a = dict(card.level_args or {})
    from net.strategist import crest as CR
    from net.strategist import s1_scenes as SS
    return {"lr": float(a.get("lr", 1e-3)), "minibatch": int(a.get("minibatch", 256)),
            "max_grad": float(a.get("max_grad", 1.0)), "tseed": int(a.get("tseed", 11)), "chunk": int(a.get("chunk", 128)),
            "pre_pass": False, "rounds_registered": int(a.get("rounds_registered", 4)),
            "key": {"crest": CR.cfg(), "axes": SS.AXES, "twins": SS.TWINS, "general": SS.GENERAL}}


def sim_key():
    from net.tactic import exam_pin as EP
    return EP.sim_key()


def net_new(card, device):
    from net.strategist import s1_net as SN
    return SN.StratNet1().to(device)


def net_parent(card, path, device):
    from net.strategist import s1_net as SN
    return SN.restore(path, device)[0]


def _identities(s0, n, gs0, n_general, split):
    from net.strategist import s1_scenes as SS
    return ([("s1.scene", None, s0 + i) for i in range(n)] + [("s1.scene", SS.twin_kind(s0 + i), s0 + i) for i in range(n)]
            + [("s1.general", None, gs0 + i) for i in range(n_general)])


def spec_cache_file(s0, n, gs0, n_general, split):
    """build/s1_specs/<split>_<s0>_<n>_<gs0>_<ng>__<code>.pkl: a round's (or V's) specs built once - by the CPU preflight
    before the GPU lock (tmp/drill_s1/c2/pre_train_s1.py) - keyed by the generator's code (exam_s1 KEY_TREES)."""
    from net import ROOT
    from net.strategist import exam_s1 as EX
    return ROOT / "build" / "s1_specs" / f"{split}_{s0}_{n}_{gs0}_{n_general}__{EX._tree_sha(EX.KEY_TREES)}.pkl"


def specs_of_round(card, s0, n, gs0, n_general, split="train", cache=True):
    """n cause scenes + n twins from s0 and n_general general battles from gs0 (module doc); with cache, read from
    spec_cache_file when its identities and digest hold (else built and written there)."""
    import pickle
    from net.strategist import exam_s1 as EX
    from net.strategist import s1_scenes as SS
    f = spec_cache_file(s0, n, gs0, n_general, split) if cache else None
    want = _identities(s0, n, gs0, n_general, split)
    if f is not None and f.exists():
        d = pickle.loads(f.read_bytes())
        specs = d["specs"]
        if [SS.identity(s) for s in specs] != [tuple(x) for x in want] or EX.spec_sha(specs) != d["sha"]:
            raise SystemExit(f"{f}: a specs cache whose identities or digest differ - refused (remove it by hand)")
        return specs
    specs = SS.scenes_at(s0, n, split) + [SS.general(gs0 + i, split) for i in range(n_general)]
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        part = f.with_suffix(".part")
        part.write_bytes(pickle.dumps({"sha": EX.spec_sha(specs), "specs": specs}))
        part.replace(f)
    return specs


def collect_round(card, r, net, ctx, store_dir):
    """Round r (0-based) of card.stages {n, general_share, seed0, gseed0, stride}: n cause scenes + n twins from seed0 +
    stride x r, and general battles from gseed0 + stride x r making general_share of the round; their inputs at tick 0
    (s1_net.observe) and the teacher's labels. No battle is played (module doc); net_driven 0 by construction."""
    from net.drill import store as ST
    from net.drill.cards import s0_1 as CARD
    from net.strategist import s1_net as SN
    from net.strategist import s1_scenes as SS
    stg = dict(card.stages or {})
    n = int(stg.get("n", 512))
    if int(getattr(ctx, "smoke", 0) or 0) > 0:
        n = int(ctx.smoke)
    share = float(stg.get("general_share", 0.25))
    ng = int(round(share / (1 - share) * 2 * n))
    stride = int(stg.get("stride", CARD.ROUND_STRIDE))
    s0 = int(stg.get("seed0", CARD.SETS["train"])) + stride * int(r)
    gs0 = int(stg.get("gseed0", CARD.SETS["train_general"])) + stride * int(r)
    specs = specs_of_round(card, s0, n, gs0, ng)
    chk = getattr(ctx, "check_starts", None)
    n_starts = chk([CARD.start_of(s) for s in specs]) if chk is not None else 0
    obs = SN.observe(specs, getattr(ctx, "device", "cpu"), getattr(ctx, "params", None),
                     chunk=int(getattr(ctx, "chunk", None) or conf(card)["chunk"]))
    labels = [SS.label(s) for s in specs]
    rows = rows_of(specs, obs, labels)
    meta = [{"seed": s["seed"], "start": kind_of(s, k), "twin": s["twin"], "map": s["map"], "n": len(s["mine"])}
            for s, k in zip(specs, labels)]
    st = ST.Store(store_dir, getattr(ctx, "run_key", None), int(r) + 1)
    stats = {"done_rows": {"strat": (len(specs), 0)}, "starts_checked": n_starts,
             "NK": (int(rows["obs_rows"].shape[0]), int(rows["obs_rows"].shape[1])), "rebuild_compared": 0,
             "rebuild_max_diff": 0.0, "battles": len(specs), "net_driven": 0, "census": census(rows)}
    st.put("train", rows, meta, cards=None, stats=stats)
    return st, stats


remap = None             # (no map handle in the stored rows: the obs are numbers)
pad = None               # (fixed-shape obs: K rows of the operative's input, C = 4 candidates)


def census(rows):
    te, kd = rows["teacher"], rows["kind"]
    return {"rows": int(len(te)), "CREST": int((te > 0).sum()), "KEEP": int((te == 0).sum()),
            "by_kind": {KINDS[k]: int((kd == k).sum()) for k in range(len(KINDS)) if int((kd == k).sum())},
            "pointer": {str(j): int((te == j).sum()) for j in range(1, 4)}}


def weights(card, rows, meta):
    w = balanced(rows)
    return w, {**census(rows), "mass": "CREST half; KEEP's half split equally over the KEEP kinds present (balanced)"}


def batch_of(rows, cards, idx):
    return to_batch(rows, idx)


def losses(net, batch, w):
    lg = net(_obs(batch))
    y = batch.teacher[:, 0].clamp(min=0)
    ce = F.cross_entropy(lg, y, reduction="none")
    ww = w[:, 0] * (batch.status[:, 0] == RC.DECIDED).float()
    return {"ce": (ce * ww).sum() / ww.sum().clamp(min=1e-9)}


def logits(net, batch):
    return {"strat": net(_obs(batch))}


def census_batch(rows):
    return to_batch(rows)


def census_classes(card, batch, w, meta):
    """{"strat": {"KEEP" | "CREST" | "POINTER": census.Cls}}; grad = weighted and >= 2 legal choices (a KEEP label
    with no valid crest moves no logit); POINTER - the CREST labels with >= 2 valid crests (the pointer's own gradient;
    the card's minimum, Astra 10)."""
    from net.drill import census as CN
    t = batch.teacher
    dec = batch.status == RC.DECIDED
    two = batch.legal.sum(-1) >= 2
    ptr = batch.legal[..., 1:].sum(-1) >= 2
    ww = w > 0
    out = {}
    for name, sel in (("KEEP", t == 0), ("CREST", t > 0), ("POINTER", (t > 0) & ptr)):
        lab = dec & sel
        # (a row with ONE legal choice - KEEP alone: flat twins, flat general battles - is out of the loss by design
        # (balanced weighs it 0, no logit moves): don't-care, not a lost label - else the census' kept share counts
        # it and halts round 1 at 422 / 777 = 0.54 < keep_min; review of the S0-1 chain, 23.11)
        out[name] = CN.Cls(labels=lab, dont_care=lab & ~two, weighted=lab & ww & two, grad=lab & ww & two,
                           removed={"one_legal": lab & ~two, "no_weight": lab & two & ~ww})
    return {"strat": out}


def census_extra(card, batch, rows, meta, w):
    return {"census": census(rows)}


def v_key(card, ctx, chunk=None):
    """V's play key - the engine calls it as v_key(card, args, chunk) (net/drill/run.py), v_read as v_key(card, ctx);
    the key reads neither (V is no play: the inputs at tick 0), so both give the same dict."""
    from net.drill import store as ST
    from net.drill.cards import s0_1 as CARD
    from net.strategist import exam_s1 as EX
    from net import ROOT
    v = (card.sets or {}).get("V")
    return {"files": {f: ST.sha256(ROOT / f)[:16] for f in CARD.FILES}, "code": EX._tree_sha(EX.KEY_TREES),
            "sim": sim_key(), "V": None if v is None else [v.seed0, v.n, CARD.SETS["V_general"]],
            "device": str(getattr(ctx, "device", "cpu")).split(":")[0]}


def v_read(card, net, ctx, cache):
    """{T, N}: the marks on V's scenes (n cause + n twins + the general share); T and N are the same rows - the decision
    is the battle's first, so the network's path reaches the same state (module doc)."""
    import hashlib
    import json
    from pathlib import Path
    from net.drill import store as ST
    from net.drill.cards import s0_1 as CARD
    from net.strategist import s1_net as SN
    from net.strategist import s1_scenes as SS
    v = card.sets["V"]
    dev = getattr(ctx, "device", "cpu")
    nv = min(int(v.n), int(getattr(ctx, "smoke", 0) or 0)) if int(getattr(ctx, "smoke", 0) or 0) > 0 else int(v.n)
    root = Path(cache) if isinstance(cache, (str, Path)) else Path(cache.root)   # (a Path's .root is its drive: Astra 10)
    key = hashlib.sha256(json.dumps(v_key(card, ctx), sort_keys=True, default=str).encode()).hexdigest()[:16]
    d = root / (key + ("" if nv == int(v.n) else f"_smoke{nv}"))
    if not (d / "key.json").exists():
        if d.exists():                         # (an unfinished cache - shards without key.json: moved aside whole,
            k = 0                              # never appended to; Astra o1_train_review3 7)
            while d.with_name(f"{d.name}.stale{k}").exists():
                k += 1
            d.rename(d.with_name(f"{d.name}.stale{k}"))
        specs = specs_of_round(card, v.seed0, nv, CARD.SETS["V_general"], nv // 2, "V")
        chk = getattr(ctx, "check_starts_v", None)
        if chk is not None:                    # (V's realised starts against the registry's V ranges: o1 review 3, visible)
            chk([CARD.start_of(s) for s in specs])
        obs = SN.observe(specs, dev)
        labels = [SS.label(s) for s in specs]
        ST.Store(d).put("V", rows_of(specs, obs, labels), [{"seed": s["seed"]} for s in specs], cards=None)
        ST.atomic_text(json.dumps({"key": v_key(card, ctx)}, default=str), d / "key.json")
    rows, _, _ = ST.Store(d).mixture()
    m = marks(net, rows, dev)
    return {"T": m, "N": m, "cache": {"teacher_path": str(d)}}


def brief(v):
    return {p: {k: (v.get(p) or {}).get(k) for k in ("followed", "false")} for p in ("T", "N")}


def save(path, net, opt, card, state):
    from net.strategist import s1_net as SN
    SN.save(path, net, {"opt": None if opt is None else opt.state_dict(), "rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "gen_state": (state or {}).get("gen_state"), "run_key": (state or {}).get("run_key"),
                        "round": (state or {}).get("round"), "stop": (state or {}).get("stop")})


def load(path, device="cpu"):
    from net.strategist import s1_net as SN
    net, ck = SN.restore(path, device)
    return {"net": net.state_dict(), "opt": ck.get("opt"), "gen_state": ck.get("gen_state"), "rng": ck.get("rng"),
            "cuda_rng": ck.get("cuda_rng"), "run_key": ck.get("run_key"), "round": ck.get("round"),
            "stop": ck.get("stop")}
