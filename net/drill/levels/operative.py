"""The drill engine's operative level (tmp/engine/design_merged.md 2-5): O0-1's decision record on the live runner
exam_o1.play (O1Side's hook), its rows for store.Store, the record.Batch view, the rebuild, the census, the losses and
the greedy replay. One decision unit a battle (G = 1, the own group); candidates heads.CANDIDATES (0 KEEP, 1 LS).

    Recorder(net=False)    O1Side's recorder: a row a decision of a battle with facts, BEFORE the zone reaches the
                 level below; masks.of('operative') on its facts gives legal / why / delegated (a member in melee);
                 teacher / proposed / executed are candidate indexes (-1: none - a delegated unit's teacher, a pointer
                 outside the candidates such as DUMB's LM); act - the executed lines.POINTERS index (-1 KEEP); obs -
                 heads.observe (the network's stored input); for the rebuild the battle's raw, axes, the group (members,
                 E), the order in force, C and the stand-off zone
    rec.rows() -> {key: tensor [M, ...]}, rec.cards() -> [B, N, 69]    one chunk's shard for store.Store.put
    to_batch(rows, cards, idx) -> record.Batch
    rec.maps(n) -> [map name | None] per battle (the raw's 'terr' is a handle into this process's sim.terrain registry)
    rebuild(rows, cards, idx, maps) -> (comparisons, max |diff|)    obs recomputed from the stored raw state, the raw's
                 map handle re-resolved by the battle's map name (duel.map_id) - another process registers maps in
                 another order (a fresh process gave a max diff 5.07 without it, tmp/engine/b/milestone2.log)
    census(rows)           eligible decided KEEP / LS labels, delegated, battles, distinct battles with a first LS
    balanced(rows) -> [M, 1]  LS half the mass, KEEP's half split equally over the teacher's KEEP reasons
    LEVEL                  the adapter (record.Level): tick_s, play, rebuild, losses, ask, save, load
    marks(net, rows)       V's followed / false (the stop: validate.relative_stop on the card's marks)
    the engine's level interface (net/drill/levels/__init__.py): NAME, tick_s, IMPL_FILES, conf, sim_key, net_new,
    net_parent, collect_round, remap, pad, weights, batch_of, losses, logits, census_batch, census_classes,
    census_extra, v_read, v_key, brief, save, load - O0-1 trains only through `python -m net.drill.run train --card o0_1`
    (fit.passes); the label probe keeps its own small loop (net/drill/probe.py passes)
"""
import torch
import torch.nn.functional as F

from net.drill import masks as MK
from net.drill import record as RC
from net.operative import advance as AD
from net.operative import heads as HD
from net.operative import lines as LN
from net.tactic import raw as TR

KEEP, LS = 0, 1
OBS = tuple(k for k in HD.NET_KEYS if k != "legal")
FIELD = ("x", "z", "b", "front", "depth", "on")


def cand(ptr):
    if ptr is None:
        return KEEP
    return HD.CANDIDATES.index(ptr) if ptr in HD.CANDIDATES else RC.NONE


def mask_of(facts):
    """masks.of('operative') with this card's candidates (the stand-off rename)."""
    return MK.of("operative", facts, cands=HD.CANDIDATES, open_=("LS",), same_key="ls_same", same_ptr="LS")


class Recorder:
    def __init__(self, net=False, keep_raw=True, keep_delegated=True, check_every=0):
        self.net = bool(net)
        self.check_every, self.n_dec, self.checked, self.check_max = int(check_every), 0, 0, 0.0
        self.keep_raw, self.keep_delegated = bool(keep_raw), bool(keep_delegated)   # (the probe: neither - its rows
        # are obs and labels only: a 384-battle collection with raw ~2 GB)
        self.r = []
        self.ii = None
        self.card = {}
        self.map = {}

    def chunk(self, ci, ii):
        self.ii = list(ii)

    net_set = frozenset()                                      # (the set's battles the network drove: train rounds)

    def decision(self, b, sec, f, order, obs, teacher, why, proposed, executed, raw, card, axes):
        if self.ii is None or b >= len(self.ii):               # (a padded copy: not the set's battle)
            return
        fc = f["fc"]
        m = mask_of([f])
        dec = bool(m.eligible[0])
        if not dec and not self.keep_delegated:
            return
        if dec and self.check_every:
            if self.n_dec % self.check_every == 0:
                self._check(raw, card, axes, f, order, sec, obs)
            self.n_dec += 1
        self.card.setdefault(self.ii[b], card.detach().cpu().clone())
        if self.ii[b] not in self.map:
            from net.matchup import duel as DU
            h = int(round(float(raw[0, TR.I["terr"]])))
            self.map[self.ii[b]] = next((k for k, v in DU._MAPS.items() if v == h), None) if h >= 0 else None
        o = order
        if not self.keep_raw:
            raw = raw[:0]
        self.r.append({
            "b": self.ii[b], "sec": int(sec), "tw": 1.0, "net": self.net or self.ii[b] in self.net_set, "status": RC.DECIDED if dec else RC.DELEGATED,
            "why": int(m.why[0]), "teacher": cand(teacher) if dec else RC.NONE, "proposed": cand(proposed),
            "executed": cand(executed), "act": -1 if executed is None else LN.POINTERS.index(executed),
            "reason": AD.REASONS.index(why), "legal": m.legal[0], "obs": {k: obs[k].detach().cpu() for k in OBS},
            "obs_legal": obs["legal"].cpu(), "raw": raw.detach().cpu().clone(),
            "ax": [float(axes.cx[0]), float(axes.cz[0]), float(axes.heading[0])] + [float(axes.field[k][0]) for k in FIELD],
            "mem": fc["mem"].cpu().clone(), "E": fc["E"].cpu().clone(),
            "o_zone": list(o["zone"]), "o_goal": list(o["goal"]), "o_born": float(o["born"]),
            "o_ptr": LN.POINTERS.index(o["ptr"]), "C": [float(fc["C"][0]), float(fc["C"][1])],
            "ls": [float(v) for v in f["ls"]], "att": int(f["att"]), "ls_same": bool(f["ls_same"])})

    def _check(self, raw, card, axes, f, order, sec, obs):
        """The rebuild at the decision: obs again from this row's raw state through the stored-row path."""
        fc = f["fc"]
        C = (float(fc["C"][0]), float(fc["C"][1]))
        ls = tuple(float(v) for v in f["ls"])
        o = HD.inputs_of(raw[None], card[None], fc["mem"], fc["E"], order["zone"], int(sec), axes)
        o["order"] = torch.tensor(HD.order_feats(order, C, ls, int(sec)), device=raw.device)
        o["mission"] = torch.tensor([float(f["att"]), min(float(sec) / 600.0, 2.0)], device=raw.device)
        o["cand"] = torch.tensor(HD.cand_feats(order, C, ls), device=raw.device)
        o["legal"] = torch.tensor(HD.legal_of({"ls_same": bool(f["ls_same"])}), device=raw.device)
        for k in HD.NET_KEYS:
            x, y = o[k], obs[k]
            self.checked += x.numel()
            if x.numel():
                self.check_max = max(self.check_max, float((x.double() - y.double()).abs().max()))

    def rows(self):
        R = self.r
        t = lambda k, dt: torch.tensor([x[k] for x in R], dtype=dt)          # noqa: E731
        out = {"b": t("b", torch.long), "sec": t("sec", torch.long), "tw": t("tw", torch.float32),
               "net": t("net", torch.bool), "status": t("status", torch.int8), "why": t("why", torch.int32),
               "teacher": t("teacher", torch.long), "proposed": t("proposed", torch.long),
               "executed": t("executed", torch.long), "act": t("act", torch.long), "reason": t("reason", torch.long),
               "legal": torch.stack([x["legal"] for x in R])[:, None], "obs_legal": torch.stack([x["obs_legal"] for x in R]),
               "raw": torch.stack([x["raw"] for x in R]) if self.keep_raw else torch.zeros((len(R), 0)), "ax": t("ax", torch.float32), "mem": torch.stack([x["mem"] for x in R]) if self.keep_raw else torch.zeros((len(R), 0), dtype=torch.bool),
               "E": torch.stack([x["E"] for x in R]) if self.keep_raw else torch.zeros((len(R), 0), dtype=torch.bool), "o_zone": t("o_zone", torch.float64), "o_goal": t("o_goal", torch.float64),
               "o_born": t("o_born", torch.float64), "o_ptr": t("o_ptr", torch.long), "C": t("C", torch.float64),
               "ls": t("ls", torch.float64), "att": t("att", torch.long), "ls_same": t("ls_same", torch.bool)}
        out["dont_care"] = torch.zeros_like(out["legal"])
        for k in OBS:
            out["obs_" + k] = torch.stack([x["obs"][k] for x in R])
        return out

    def maps(self, n):
        return [self.map.get(b) for b in range(n)]

    def cards(self, n=None):
        """[card [N, 69] | None] per battle 0 .. n - 1 (None: a battle without a recorded decision)."""
        n = (max(self.card) + 1 if self.card else 0) if n is None else n
        return [self.card.get(b) for b in range(n)]


def to_batch(rows, idx=None):
    if idx is None:
        idx = torch.arange(rows["b"].shape[0])
    g = lambda k: rows[k][idx]                                       # noqa: E731
    obs = {k: g("obs_" + k) for k in OBS}
    return RC.Batch(b=g("b"), sec=g("sec"), tw=g("tw"), net=g("net"), obs=obs, order={"feats": obs["order"]},
                    legal=g("legal"), dont_care=g("dont_care"), status=g("status")[:, None], why=g("why")[:, None],
                    proposed=g("proposed")[:, None], executed=g("executed")[:, None], teacher=g("teacher")[:, None])


def _order(rows, i):
    return {"ptr": LN.POINTERS[int(rows["o_ptr"][i])], "zone": tuple(float(v) for v in rows["o_zone"][i]),
            "goal": tuple(float(v) for v in rows["o_goal"][i]), "born": float(rows["o_born"][i])}


def rebuild(rows, cards, idx, maps):
    """(comparisons, max |diff|): obs of rows idx recomputed from the stored raw state, the card of its battle; maps
    [name | None] per battle (module doc)."""
    from net.matchup import duel as DU
    from net.operative import frame as FR
    n, worst = 0, 0.0
    for i in [int(x) for x in idx]:
        raw = rows["raw"][i]
        if float(raw[0, TR.I["terr"]]) >= 0:
            mp = maps[int(rows["b"][i])]
            assert mp is not None, f"row {i}: a map handle without the battle's map name"
            raw = raw.clone()
            raw[:, TR.I["terr"]] = float(DU.map_id(mp))
        a = rows["ax"][i]
        axes = FR.Axes(a[0:1], a[1:2], a[2:3], {k: (a[3 + j:4 + j] > 0.5 if k == "on" else a[3 + j:4 + j])
                                                for j, k in enumerate(FIELD)})
        od, sec = _order(rows, i), int(rows["sec"][i])
        C = (float(rows["C"][i][0]), float(rows["C"][i][1]))
        ls = tuple(float(v) for v in rows["ls"][i])
        o = HD.inputs_of(raw[None], cards[int(rows["b"][i])][None], rows["mem"][i], rows["E"][i],
                         od["zone"], sec, axes)
        o["order"] = torch.tensor(HD.order_feats(od, C, ls, sec))
        o["mission"] = torch.tensor([float(rows["att"][i]), min(float(sec) / 600.0, 2.0)])
        o["cand"] = torch.tensor(HD.cand_feats(od, C, ls))
        o["legal"] = torch.tensor(HD.legal_of({"ls_same": bool(rows["ls_same"][i])}))
        for k in HD.NET_KEYS:
            x, y = o[k], rows["obs_legal" if k == "legal" else "obs_" + k][i]
            assert x.shape == y.shape, (k, tuple(x.shape), tuple(y.shape))
            n += x.numel()
            if x.numel():
                worst = max(worst, float((x.double() - y.double()).abs().max()))
    return n, worst


def eligible(rows):
    te = rows["teacher"]
    dec = rows["status"] == RC.DECIDED
    leg = rows["legal"][:, 0]
    return dec & (te >= 0) & leg.gather(1, te.clamp(min=0)[:, None])[:, 0]


def census(rows):
    te, b, dec = rows["teacher"], rows["b"], rows["status"] == RC.DECIDED
    el = eligible(rows)
    first = {}
    for i in range(len(b)):
        if bool(el[i]) and int(te[i]) == LS and int(b[i]) not in first:
            first[int(b[i])] = int(rows["sec"][i])
    return {"rows": int(len(b)), "delegated": int((~dec).sum()), "eligible_KEEP": int((el & (te == KEEP)).sum()),
            "eligible_LS": int((el & (te == LS)).sum()), "ineligible_decided": int((dec & ~el).sum()),
            "battles": int(len(set(b.tolist()))), "distinct_first_LS": len(first),
            "first_LS_sec": sorted(first.values())}


def teaches(rows):
    """[M] bool: eligible rows with at least two legal candidates - a label that moves the head (a row with only KEEP
    legal - the LS line already in force, 'same' - gives no gradient: Astra's o1 review 6, 15.9 % of probe_train's
    weight sat there)."""
    return eligible(rows) & (rows["legal"][:, 0].sum(-1) >= 2)


def balanced(rows, forced=False):
    """[M, 1]: LS half the mass; KEEP's half split equally over the teacher's KEEP reasons present (advance.REASONS:
    role, early, melee, comer, comes, near, same) - mean 1 over the eligible rows. _why: with KEEP / LS halves alone
    (5,826 KEEP rows against 269 LS on probe_train) the motion negatives - the come twins' first decisions, reason
    'comes' / 'comer' - weigh 1/20 of a positive each and drown among the easy 'role' / 'same' rows: OpNet1 stalled at
    loss 0.63 with every come twin's first decision LS (cause / come pairs at chance) at lr 1e-3 and 3e-3; split by
    reason the same network reached pairs 64 / 64 followed, 0 / 64 false in 10 passes (tmp/engine/b/diag4.log), while a
    plain MLP on the pooled real input separated them already (diag.log) - the cause was visible, the weights hid it."""
    from net.operative import advance as AD
    el = eligible(rows) if forced else teaches(rows)      # (forced=True: the 22.11 weights, forced-KEEP rows in)
    te, rs = rows["teacher"], rows["reason"]
    w = torch.zeros(len(te))
    m = el & (te == LS)
    if bool(m.any()):
        w[m] = 0.5 / float(m.sum())
    ks = [k for k in range(len(AD.REASONS)) if bool((el & (te == KEEP) & (rs == k)).any())]
    for k in ks:
        m = el & (te == KEEP) & (rs == k)
        w[m] = 0.5 / len(ks) / float(m.sum())
    return (w * float(el.sum()))[:, None]

class OperativeLevel:
    tick_s = 15.0

    @staticmethod
    def obs(batch):
        return {**batch.obs, "legal": batch.legal[:, 0]}

    def losses(self, net, batch, w):
        lg = net(self.obs(batch))
        y = batch.teacher[:, 0].clamp(min=0)
        ce = F.cross_entropy(lg, y, reduction="none")
        ww = w[:, 0] * (batch.status[:, 0] == RC.DECIDED).float() * (batch.teacher[:, 0] >= 0).float()
        return {"ce": (ce * ww).sum() / ww.sum().clamp(min=1e-9)}

    def ask(self, net, batch):
        with torch.no_grad():
            return net(self.obs(batch))

    def rebuild(self, rows, cards, idx, maps):
        return rebuild(rows, cards, idx, maps)

    VISIBLE = True        # (Astra's o1 review 1: the teacher that labels and drives reads the SEEN foes only - the facts
    #                       the network's input is built from; every engine path - collection, V, the probe - plays so)

    def play(self, name, specs, seeds, chunk, device="cpu", make_player=None, recorder=None, compile=False):
        from net.operative import exam_o1 as EX
        return EX.play(name, specs, seeds, chunk, device, make_player=make_player, recorder=recorder, compile=compile,
                       visible=self.VISIBLE)

    def tick0(self):
        return int(LN.conf()["start_s"])

    def judge(self, d):
        from net.operative import exam_o1 as EX
        return EX.judge(d)

    def save(self, path, net, **more):
        from net.operative import ckpt as CK
        CK.save_o1(path, net, more)

    def load(self, path, device="cpu"):
        from net.operative import ckpt as CK
        return CK.restore_o1(path, device)


LEVEL = OperativeLevel()


# ------------------------------------------------------------------ O0-1's training through the engine
class Mixed:
    """A chunk's player: the network (heads.NetPlayer) on the battles of net_local (chunk-local indexes), else the
    card's disturbed teacher (cards/o0_1.Disturbed with the chunk's delays)."""
    wants_view = True

    def __init__(self, net, net_local, teacher):
        self.n, self.local, self.t = HD.NetPlayer(net), set(net_local), teacher

    def __call__(self, f, order, c=None, obs=None, b=None):
        return self.n(f, order, c, obs=obs, b=b) if b in self.local else self.t(f, order, c, obs=obs, b=b)


def _sh(m, hit):
    n = int(m.sum())
    if not n:
        return None
    k = float((hit & m).sum())
    q = (k + 1) / (n + 2)
    return {"p": round(k / n, 4), "two_errors": round(2 * (q * (1 - q) / n) ** 0.5, 4), "n": n}


def _pred(net, rows, idx, device):
    out = []
    for i in range(0, idx.numel(), 1024):
        b = to_batch(rows, idx[i:i + 1024]).to(device)
        out.append(LEVEL.ask(net.eval(), b).argmax(1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def marks(net, rows, device="cpu", kinds=None):
    """V's marks of the network's greedy choices on rows (one path), Astra's o1 review 6 / the orchestrator F3 - the
    decisions that matter, per battle (kinds[b]: cause | come | role):
        cause_first_followed   B1 on this path (Astra o1_train_review3 5; tmp/engine/b/LINES_o1.md): every cause
                               battle of kinds (one with no decided row too - a miss); t0 = the teacher's first eligible
                               LS label on this path's rows, tN = the network's first LS pick with LS legal; a hit iff
                               both exist and |tN - t0| <= decide_s - no credit for an earlier lead; beside: 'no_label'
                               (no teacher LS on the path), 'no_move', 'early' (tN < t0 - decide_s), 'late'
        come_first_false       a come twin's first decided row: the network LS
        role_false             a role twin: the network LS on any decided row
    beside (printed, not gating): followed / false on every eligible tick."""
    el = eligible(rows)
    idx = el.nonzero()[:, 0]
    if not idx.numel():
        return {"followed": None, "false": None, "cause_first_followed": None, "come_first_false": None,
                "role_false": None}
    k = _pred(net, rows, idx, device)
    te = rows["teacher"][idx]
    out = {"followed": _sh(te == LS, k == LS), "false": _sh(te == KEEP, k == LS)}
    if kinds is None:
        return out
    dec = (rows["status"] == RC.DECIDED).nonzero()[:, 0]
    kd = _pred(net, rows, dec, device)
    leg_ls = rows["legal"][dec, 0, LS]
    by = {}
    for j, i in enumerate(dec.tolist()):
        by.setdefault(int(rows["b"][i]), []).append((int(rows["sec"][i]), i, int(kd[j]), bool(leg_ls[j])))
    cf, cfh, mf, mfh, rf, rfh = [], [], [], [], [], []
    dt = int(LN.conf()["decide_s"])                      # (one operative tick, seconds)
    extra = {"no_label": 0, "no_move": 0, "early": 0, "late": 0, "no_rows": 0}
    for b, kind in enumerate(kinds):                     # (every battle of the path's list, rows or not)
        if kind == "cause" and b not in by:
            extra["no_rows"] += 1
            cf.append(True)
            cfh.append(False)
    for b, rs in by.items():
        rs.sort()
        kind = kinds[b] if b < len(kinds) else None
        if kind == "cause":
            tT = next((sec for sec, i, _, _ in rs if int(rows["teacher"][i]) == LS and bool(el[i])), None)
            tN = next((sec for sec, _, k_, leg in rs if k_ == LS and leg), None)
            hit = tT is not None and tN is not None and abs(tN - tT) <= dt
            if tT is None:
                extra["no_label"] += 1
            elif tN is None:
                extra["no_move"] += 1
            elif not hit:
                extra["early" if tN < tT else "late"] += 1
            cf.append(True)
            cfh.append(hit)
        elif kind == "come":
            mf.append(True)
            mfh.append(rs[0][2] == LS)
        elif kind == "role":
            rf.append(True)
            rfh.append(any(r[2] == LS for r in rs))
    T = lambda x: torch.tensor(x, dtype=torch.bool)                     # noqa: E731
    out.update(cause_first_followed=_sh(T(cf), T(cfh)) if cf else None,
               come_first_false=_sh(T(mf), T(mfh)) if mf else None, role_false=_sh(T(rf), T(rfh)) if rf else None,
               cause_first_extra=dict(extra, battles=len(cf)))
    return out


# ------------------------------------------------------------------ the engine's level interface (levels/__init__.py)
NAME = "operative"
IMPL_FILES = ("net/operative/*.py", "net/tactic/*.py", "net/unit/*.py", "net/stage0/*.py", "net/matchup/duel.py")


def conf(card):
    """The fit's knobs and the level's configs for the run key (card.level_args may override lr / minibatch).
    _why lr 3e-4: the label probe on three seeds (tmp/engine/b/probe/probe.json = probe_lr3e4_b): pairs 64 / 64
    followed, 0 / 64 false on every seed at the last and the lowest-loss pass, the ablated arm at chance; at 1e-3 one
    seed of two oscillated (probe_fit3 vs diag4) and minibatch 64 failed seed 5 (probe_mb64)."""
    a = dict(card.level_args or {})
    return {"pick": bool(a.get("pick", False)), "lr": float(a.get("lr", 3e-4)), "minibatch": int(a.get("minibatch", 256)),
            "max_grad": float(a.get("max_grad", 1.0)), "tseed": int(a.get("tseed", 11)),
            "chunk": int(a.get("chunk", 256)), "pre_pass": False,
            "key": {"o1": LN.conf(), "council_extra": cfg_extra(), "pick": bool(a.get("pick", False))}}


def cfg_extra():
    from net.operative import cfg
    return cfg()["council_extra"]


def sim_key():
    from net.tactic import exam_pin as EP
    return EP.sim_key()


def net_new(card, device):
    return HD.OpNet1(pick=conf(card)["pick"]).to(device)


def net_parent(card, path, device):
    from net.operative import ckpt as CK
    return CK.restore_o1(path, device)[0]


GENERAL_SEED = 46_000_000     # (the training general battles: SP.seeded(GENERAL_SEED + 10,000 r) for mission attack,
#                               + 5,000 for mission hold; torch chunk seeds s0 + 5,000 + k / s0 + 7,000 + k inside the
#                               registered train chunk range - tmp/engine/b/NEEDS_A.md 4)


def _parts(card, r, n, s0, smoke):
    """[(name, specs, chunk seed base)] of round r: the scenes (n cause + n twins, cards/o0_1.scenes_at - scenes_fair)
    and stage general_share of all battles as the stage's general battles, half mission attack (G's recipe), half
    mission hold (G_def's) - net/operative/attack.generals."""
    from net.drill.cards import o0_1 as CARD
    from net.operative import attack as AT
    stg = dict(card.stages or {})
    gs = float(stg.get("general_share", 0.0))
    clear_zone_cache()
    out = [("scenes", CARD.scenes_at(s0, n), s0)]
    if gs > 0:
        ng = int(round(gs / (1.0 - gs) * 2 * n))        # (generals / all battles = general_share)
        if smoke:
            ng = max(2, min(ng, 2 * int(smoke) // 2))
        na, nh = (ng + 1) // 2, ng // 2
        g0 = GENERAL_SEED + 10_000 * int(r)
        clear_zone_cache()
        ga = AT.generals(na, g0, "attack")
        clear_zone_cache()
        out += [("G", ga, s0 + 5_000), ("G_def", AT.generals(nh, g0 + 5_000, "hold"), s0 + 7_000)]
    return out


def clear_zone_cache():
    """net/tactic/zfit._CACHE emptied before every set build (Astra o1_train_review3 6): its key rounds the local
    places to 0.1 m and stores the FIRST caller's unrounded result, so a set built after another (a later round, a
    resume, a preflight of several sets) got other zones (cz -9.797174393178818e-16 vs ...824e-16) - a build-order leak.
    Cleared per set, a set's starts depend on its own seeds only (the exam built each set in a fresh process)."""
    from net.tactic import zfit
    zfit._CACHE.clear()


def gate_key(card):
    """The key a gate file must carry to let card's training start (Astra's second o1 review 4): sha256[:16] of the
    card's files (teacher, lines, enemy, scenes, the fair wrapper, head, level, card), the o1 config, the council_extra
    config, the level's config and the facts' visibility - the probe and the teacher's verdict bound to THESE scenes and
    this teacher, not two booleans."""
    import hashlib
    import json
    from net.drill import store as ST
    from net.drill.cards import o0_1 as CARD
    x = {"files": {f: ST.sha256(f)[:16] for f in CARD.FILES}, "o1": LN.conf(), "council_extra": cfg_extra(),
         "level": {k: v for k, v in conf(card).items() if k != "key"}, "visible": LEVEL.VISIBLE}
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def gates_open(card, smoke=0):
    """(ok, reasons) of the training's start: the card's probe file (CARD.PROBE_FILE) and the teacher's verdict
    (CARD.TEACHER_FILE) both passed AND carry gate_key(card); a smoke run prints them and goes on."""
    import json
    from pathlib import Path
    from net.drill.cards import o0_1 as CARD
    want, why = gate_key(card), []
    for nm, f in (("probe", CARD.PROBE_FILE), ("teacher verdict", CARD.TEACHER_FILE)):
        p = Path(f)
        if not p.exists():
            why.append(f"{nm}: {f} missing")
            continue
        v = json.loads(p.read_text(encoding="utf-8"))
        ok = bool((v.get("verdict") or {}).get("pass")) if nm == "probe" else v.get("verdict") == "PASSED"
        if not ok:
            why.append(f"{nm}: {f} not passed")
        if v.get("gate_key") != want:
            why.append(f"{nm}: {f} gate_key {v.get('gate_key')} != this card's {want}")
    return not why, why


def collect_round(card, r, net, ctx, store_dir):
    """Round r (0-based) of card.stages {n, net_share, seed0 (default the card's train range), stride, general_share}:
    n cause scenes + n twins from seed0 + stride x r (cards/o0_1.scenes_at, through scenes_fair) and the general
    battles of _parts; round 0 the card's disturbed teacher drives every battle, later rounds the network drives
    net_share of the seed families (a cause scene and its twin together; a general battle alone); the level's facts on
    the seen foes only (LEVEL.VISIBLE); decided rows only, no raw state (fixed-shape obs: pad is None); every
    REBUILD_EVERY-th decided row rebuilt at the decision from its own raw state. One shard. -> (Store, stats)."""
    import random
    from net.drill import store as ST
    from net.drill.cards import o0_1 as CARD
    stg = dict(card.stages or {})
    n, share = int(stg.get("n", 512)), float(stg.get("net_share", 0.5))
    smoke = int(getattr(ctx, "smoke", 0) or 0)
    if r == 0:                  # (no round before the gates: the probe and the teacher's verdict of these scenes)
        ok, why = gates_open(card)
        log = getattr(ctx, "log", print)
        log({"gates": "open" if ok else "CLOSED", "why": why, "smoke": smoke})
        if not ok and not smoke:
            raise SystemExit(f"O0-1: the training's gates are closed - {why}")
    if smoke > 0:                                       # (a smoke run: n cause scenes + n twins, n = --smoke)
        n = smoke
    s0 = int(stg.get("seed0", CARD.SETS["train"])) + int(stg.get("stride", CARD.ROUND_STRIDE)) * int(r)
    ch = int(getattr(ctx, "chunk", None) or conf(card)["chunk"])
    dev = getattr(ctx, "device", "cpu")
    chk = getattr(ctx, "check_starts", None)
    if net is not None:
        net.eval()
    parts, metas, n_starts, nets, checked, cmax, off = [], [], 0, 0, 0, 0.0, 0
    for name, specs, c0 in _parts(card, r, n, s0, smoke):
        mk_t, delays = CARD.make_disturbed(specs, c0)
        fam = sorted({s["seed"] for s in specs})
        net_seeds = set(random.Random(c0 + 7919).sample(fam, int(round(share * len(fam)))))             if r > 0 and net is not None else set()
        net_set = frozenset(i for i, s in enumerate(specs) if s["seed"] in net_seeds)
        rec = Recorder(keep_raw=False, keep_delegated=False, check_every=REBUILD_EVERY)
        rec.net_set = net_set

        def mk(ci, ii, net_set=net_set, mk_t=mk_t):
            return Mixed(net, [k for k, i in enumerate(ii) if i in net_set], mk_t(ci, ii))
        seeds = [c0 + k for k in range((len(specs) + ch - 1) // ch)]
        if chk is not None:      # (the realised starts: every battle's identity and the torch chunk seeds - seeds.py)
            from net.drill import seeds as SD
            n_starts += chk([SD.identity(s) for s in specs] + [("torch.chunk", None, int(x)) for x in seeds])
        LEVEL.play("TEACH~", specs, seeds, ch, dev, make_player=mk, recorder=rec,
                   compile=torch.device(dev).type == "cuda")
        if rec.r:
            parts.append((rec.rows(), off))
        metas += [{"seed": s["seed"], "kind": name if name != "scenes" else (s["o1"]["twin"] or "cause"),
                   "net": i in net_set, "delay": delays.get(i, 0),
                   "start": name if name != "scenes" else ("cause" if s["o1"]["cause"] else s["o1"]["twin"]),
                   "chunk_seed": seeds[i // ch]} for i, s in enumerate(specs)]
        nets += len(net_set)
        checked, cmax = checked + rec.checked, max(cmax, rec.check_max)
        off += len(specs)
    rows = _cat(parts)
    st = ST.Store(store_dir, getattr(ctx, "run_key", None), int(r) + 1)
    NK = (int(rows["obs_rows"].shape[0]), int(rows["obs_rows"].shape[1]))      # (the recorder asks only live battles:
    # no row of a finished battle is kept - done_rows' left-out count is 0)
    stats = {"done_rows": {"op": (int(len(rows["b"])), 0)}, "starts_checked": n_starts, "NK": NK,
             "rebuild_compared": checked, "rebuild_max_diff": cmax, "battles": len(metas), "net_driven": nets,
             "by_kind": {k: sum(1 for m in metas if m["start"] == k) for k in sorted({m["start"] for m in metas})},
             "census": census(rows)}
    st.put("train", rows, metas, cards=None, stats=stats)
    return st, stats


def _cat(parts):
    """Rows of several plays concatenated, the battle ids offset by each play's first battle."""
    out = {}
    for rows, off in parts:
        for k, v in rows.items():
            out.setdefault(k, []).append(v + off if k == "b" else v)
    return {k: torch.cat(v, 0) for k, v in out.items()}


REBUILD_EVERY = 25       # (a decided row in 25 rebuilt at the decision: ~600 rows a 1,024-battle round)


remap = None             # (no per-process handle in the stored rows: no raw state kept)
pad = None               # (fixed-shape obs [K = 40]: shards of different battle sizes stack as they are)


def weights(card, rows, meta):
    w = balanced(rows)
    el = eligible(rows)
    te = rows["teacher"]
    forced = el & ~teaches(rows)
    w0 = balanced(rows, forced=True)[:, 0]
    return w, {"eligible_KEEP": int((el & (te == KEEP)).sum()), "eligible_LS": int((el & (te == LS)).sum()),
               "forced_KEEP_rows": int(forced.sum()),
               "forced_weight_share_before": round(float(w0[forced].sum() / w0.sum().clamp(min=1e-9)), 4),
               "forced_weight_share_now": round(float(w[:, 0][forced].sum() / w.sum().clamp(min=1e-9)), 4),
               "mass": "LS half; KEEP half split equally over the teacher's KEEP reasons; rows with only KEEP legal "
                       "out (teaches)"}


def batch_of(rows, cards, idx):
    return to_batch(rows, idx)


def losses(net, batch, w):
    return LEVEL.losses(net, batch, w)


def logits(net, batch):
    return {"op": net(LEVEL.obs(batch))}


def census_batch(rows):
    b = to_batch(rows)
    b.reason = rows["reason"][:, None]
    return b


def census_classes(card, batch, w, meta):
    """{"op": {"KEEP" | "LS": census.Cls}} - masks [M, G]; weighted = weight > 0; grad = weighted, decided and >= 2
    legal candidates (a KEEP label with LS closed moves no logit: no gradient)."""
    from net.drill import census as CN
    t = batch.teacher
    dec = batch.status == RC.DECIDED
    two = batch.legal.sum(-1) >= 2
    ww = w > 0
    out = {}

    def cls(lab):
        return CN.Cls(labels=lab, dont_care=lab & ~two, weighted=lab & ww, grad=lab & ww & two,
                      removed={"one_legal": lab & ~two, "no_weight": lab & ~ww & two})
    # (a row with only KEEP legal is out of the loss by design - teaches(): don't-care, not a lost label)
    for name, k in (("KEEP", KEEP), ("LS", LS)):
        out[name] = cls(dec & (t == k))
    reason = getattr(batch, "reason", None)
    if reason is not None:
        from net.operative import advance as AD
        for nm in CENSUS_REASONS:                  # (the KEEP reasons whose labels carry a gradient: a minimum each)
            out["KEEP:" + nm] = cls(dec & (t == KEEP) & (reason == AD.REASONS.index(nm)))
    first = torch.zeros_like(dec)                  # (a battle's first decided LS label: the distinct first advances)
    seen = set()
    bb = batch.b.tolist()
    for i in range(len(bb)):
        if bool(dec[i, 0]) and int(t[i, 0]) == LS and bb[i] not in seen:
            seen.add(bb[i])
            first[i, 0] = True
    out["LS:first"] = cls(first)
    return {"op": out}


CENSUS_REASONS = ("role", "comer", "comes", "near")
_CENSUS_WHY = ("Astra's o1 review 6 / the orchestrator F3: a minimum of gradient-bearing labels per KEEP reason that "
               "teaches the head something different - role (our mission hold), comer / comes (the enemy comes: the "
               "cause's negatives), near (already close) - and of distinct first advances a round; 'same' (the line in "
               "force, only KEEP legal), 'early', 'melee' carry no gradient by construction")


def census_extra(card, batch, rows, meta, w):
    return {"census": census(rows)}


def v_key(card, ctx, chunk=None):
    """V's teacher-path cache key (Astra's o1 review 5; o1_train_review3 1 and 7) - run.py calls v_key(card, a, chunk)
    at the start (a: its argparse namespace, .device / .smoke), v_read the same through ctx.v_key: the simulator's key,
    the level's / tactic's / unit script's implementation files (IMPL_FILES), the card's files, the configs, the o1
    config, the card's level config, the V set (seed, n, chunk), the device type and the compile mode (CUDA pads a
    chunk and compiles; the CPU does neither), both paths' torch seeds, the specs' digest, the visibility, the player."""
    import glob
    import hashlib
    import json
    from net import ROOT
    from net.drill import store as ST
    from net.drill.cards import o0_1 as CARD
    v = (card.sets or {}).get("V")
    impl = sorted({f for g in IMPL_FILES for f in glob.glob(str(ROOT / g))})
    dev = torch.device(getattr(ctx, "device", "cpu")).type
    smoke = int(getattr(ctx, "smoke", 0) or 0)
    nv = None if v is None else (min(int(v.n), smoke) if smoke > 0 else int(v.n))
    specs = None if v is None else CARD.scenes_at(v.seed0, nv)
    nch = None if v is None else (2 * nv + v.chunk - 1) // v.chunk
    run = {"device": dev, "compile": dev == "cuda", "chunk": None if v is None else int(v.chunk),
           "engine_chunk": None if chunk is None else int(chunk),
           "seeds_T": None if v is None else [v.seed0 + k for k in range(nch)],
           "seeds_N": None if v is None else [v.seed0 + V_N_OFFSET + k for k in range(nch)],
           "specs": None if specs is None else hashlib.sha256(json.dumps(specs, sort_keys=True, default=str)
                                                              .encode()).hexdigest()[:16]}
    return {"sim_key": sim_key(), "impl": {Path_rel(f): ST.sha256(f)[:16] for f in impl}, "run": run,
            "files": {f: ST.sha256(f)[:16] for f in CARD.FILES}, "o1": LN.conf(), "level": conf(card),
            "configs": {Path_rel(f): ST.sha256(f)[:16] for f in sorted(glob.glob(str(ROOT / "net/config/*.json")))
                        + sorted(glob.glob(str(ROOT / "sim/config/*.json")))},
            "V": None if v is None else [v.seed0, v.n, v.chunk], "visible": LEVEL.VISIBLE, "tactic": "TEACH",
            "smoke": smoke}


def Path_rel(f):
    from pathlib import Path
    from net import ROOT
    return Path(f).resolve().relative_to(Path(ROOT).resolve()).as_posix()


V_N_OFFSET = 500         # (path N's torch chunk seeds V + 500 + k: apart from path T's V + k at any chunk count - the
#                          former + 1 + k shared V + 1 with T's second chunk once V took two chunks)


def v_read(card, net, ctx, cache):
    """{T: marks(net, teacher-path rows), N: marks on the network's own path}: path T played once (the plain teacher)
    into <cache>/<key> as a store shard, read back on reuse (key: v_key); path N played with the network. cache: the
    run's V cache folder (run.py: --v-cache or DIR/v_cache) - a str / Path, or an object with a .dir / .path folder."""
    import hashlib
    import json
    from pathlib import Path
    from net.drill import store as ST
    from net.drill.cards import o0_1 as CARD
    v = card.sets["V"]
    dev = getattr(ctx, "device", "cpu")
    comp = torch.device(dev).type == "cuda"
    if isinstance(cache, (str, Path)):
        root = Path(cache)
    else:
        root = Path(getattr(cache, "dir", None) or getattr(cache, "path"))
    smoke = int(getattr(ctx, "smoke", 0) or 0)
    nv = min(int(v.n), smoke) if smoke > 0 else int(v.n)
    vk = getattr(ctx, "v_key", None) or v_key(card, ctx, getattr(ctx, "chunk", None))
    key = hashlib.sha256(json.dumps(vk, sort_keys=True, default=str).encode()).hexdigest()[:16]
    d = root / key
    clear_zone_cache()
    specs = CARD.scenes_at(v.seed0, nv)
    kinds = ["cause" if s["o1"]["cause"] else s["o1"]["twin"] for s in specs]
    nch = (len(specs) + v.chunk - 1) // v.chunk
    sT = [v.seed0 + k for k in range(nch)]
    sN = [v.seed0 + V_N_OFFSET + k for k in range(nch)]
    chk = getattr(ctx, "check_starts_v", None)        # (o1_train_review3 visible: V's starts and both paths' seeds)
    if chk is not None:
        from net.drill import seeds as SD
        chk([SD.identity(s) for s in specs] + [("torch.chunk", None, int(x)) for x in sT + sN])
    if d.exists() and not (d / "key.json").exists():  # (an interrupted cache: moved aside whole, never appended to)
        k = 0
        while (root / f"{key}.stale{k}").exists():
            k += 1
        d.rename(root / f"{key}.stale{k}")
    if d.exists():
        got = json.loads((d / "key.json").read_text(encoding="utf-8")).get("key")
        if json.loads(json.dumps(got, default=str)) != json.loads(json.dumps(vk, default=str)):
            raise SystemExit(f"{d}: V cache key.json differs from this run's V key - refused")
    if not (d / "key.json").exists():
        rec = Recorder(keep_raw=False, keep_delegated=False)
        LEVEL.play("TEACH", specs, sT, v.chunk, dev, recorder=rec, compile=comp)
        ST.Store(d).put("V", rec.rows(), [{"seed": s["seed"]} for s in specs], cards=None)
        ST.atomic_text(json.dumps({"key": vk}, default=str), d / "key.json")
    vT, _, vmeta = ST.Store(d).mixture()
    if len(vmeta) != len(specs) or [m.get("seed") for m in vmeta] != [s["seed"] for s in specs]:
        raise SystemExit(f"{d}: V cache holds {len(vmeta)} battles, not this V's {len(specs)} in order - refused")
    nrec = Recorder(net=True, keep_raw=False, keep_delegated=False)
    LEVEL.play("NET", specs, sN, v.chunk, dev, make_player=lambda ci, ii: HD.NetPlayer(net), recorder=nrec,
               compile=comp)
    return {"T": marks(net, vT, dev, kinds), "N": marks(net, nrec.rows(), dev, kinds),
            "cache": {"teacher_path": str(d)}}


def brief(v):
    """V's numbers in one log row: the gating marks and the per-tick rates, both paths."""
    ks = ("cause_first_followed", "come_first_false", "role_false", "followed", "false")
    return {p: {k: (v.get(p) or {}).get(k) for k in ks} for p in ("T", "N")}


def save(path, net, opt, card, state):
    from net.operative import ckpt as CK
    CK.save_o1(path, net, {"opt": None if opt is None else opt.state_dict(), "rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                           "gen_state": (state or {}).get("gen_state"), "run_key": (state or {}).get("run_key"),
                           "round": (state or {}).get("round"), "stop": (state or {}).get("stop")})


def load(path, device="cpu"):
    from net.operative import ckpt as CK
    net, ck = CK.restore_o1(path, device)
    return {"net": net.state_dict(), "opt": ck.get("opt"), "gen_state": ck.get("gen_state"), "rng": ck.get("rng"),
            "cuda_rng": ck.get("cuda_rng"), "run_key": ck.get("run_key"), "round": ck.get("round"),
            "stop": ck.get("stop")}
