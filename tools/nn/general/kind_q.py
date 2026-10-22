"""The outcome Q of the mission kinds: the kind head's dense own signal (run.py --general-kind-q outcome|both, the
default both; docs/en/training/general.md "The kind's own signal").

Why: the branches (branch.py) price one group of ~5 % of the plans (~124 rows an update), with a noise of 0.022 of the
budget a copy and a steady bias of the partial open loop (build/diag_kind_critic); the kind critic on those prices
(kind_critic.py) learns nothing of the kinds (ev <= 0.015). Here every own group of every ended plan (~2 500 plans x ~3
groups an update) is a sample of what its kind led to in the battle itself, with no replay:

    outcome   the planning side's trade over `horizon` seconds of battle since the plan began (ability_branch.trade:
              (the enemy's gold lost - its own) / budget, as the branches' "trade" price), from the recorded battle:
              Runtime tracks it (q_h > 0: the trade at the plan's start, each ended plan waits under its id "qid" until
              horizon seconds passed or the battle ended - then its trade is final, "ended" marks it; the next plans of
              the battle are in it: Q of the policy); end_chunk gives the outcomes known in the chunk ("q_out"), each
              with its battle's number "bid" (both sides of a battle the same)
    features  (X [M, G, WIDTH], logp [M, G, K]) of every own group of the plans' records: everything the kind head reads
              when it chooses - the group's query card (net.General.decide's "query": the card after the group council
              and the threat, fight and tile layers, the very vector kind_q reads), detached; the kind critic's 20
              fixed numbers (kind_critic.features: threat, fight, tiles under the group, sizes, deployment); the
              inertia (net.General.prev_kind_hot: the last kind of a group that kept its units, what kind_stick adds
              to); the kinds the group may take (decide's kind_ok: Q(attack) of a group that may shoot is not learned
              together with one of a group that may not) - so the choice depends on nothing Q does not see (else
              Q(chosen) would carry what made the choice). logp: the general's kind log-probabilities now (the
              baseline's pi).
    Model     [.., K] the outcome of every kind (budget share): inputs standardised (fixed at the first fit: the cards
              drift while the general learns, the standardisation does not follow - a known limit), two hidden SiLU
              layers with dropout, in `unit` (the first fit's root mean square of the outcomes)
    OutcomeQ  waits for the outcomes (the features of a chunk's ended plans kept under their qid until Runtime's
              outcome comes, at most `wait` chunks: the horizon's chunks + a margin; the features that never got one
              are the log's "lost"), keeps a buffer of the last `buffer` groups (saved with the checkpoint's "train"),
              fits a few Huber steps an update on Q(chosen) ~ outcome, holds every 10th battle out whole (held_out:
              all its plans and both its sides - a plan re-made every 30 s shares half its 60 s window and often its
              kind with the next one of the group, and in a battle of the network with itself the other side has the
              same outcome with the other sign: held out by plan, ev and the shrink came out too high), and gives every
              own group of the update's ended plans Q [M, G, K] (learn.critic_advantage: Q(chosen) - sum_k pi0(k) Q(k),
              COMA's baseline) once ready.

Ready: at least `warmup` groups and the held-out explained variance of the outcome (ev, Q(chosen) against the outcome)
at least `ev_min`. The shrink s (Q given = s x Q): the held-out least-squares coefficient c of the advantage A =
Q(chosen) - sum pi Q in outcome ~ 1 + V + A (V = sum pi Q) less two of its standard errors (by battle, kind_coef), within
[0, 1] - how much of Q's kind part the held-out outcomes bear out (most of ev is the state's part V, which the head never
sees); the point coefficient alone jumps +-0.3 when that part is small and gave noise a shrink near 1 at times. The scale (learn.critic_advantage divides by it):
each kind's running root mean square of the unshrunk A of its groups (signal.KindNorm over the advantages,
--general-kind-norm per-kind; chunk: the update's over all groups), at least FLOOR - the head gets about s in units.

Agreement with the branches (the log's kind_q_agree): for the update's branch rows, the correlation of Q(chosen) -
Q(other) (the mean over the copies' other kinds) with the branch's centred price.

Its own parameters and optimiser: not a head of the general, nothing of it reaches the general's layers; training only.
"""
import math
import time

import torch
from torch import nn

from tools.nn.general import kind_critic as kc
from tools.nn.general import signal as gsignal
from tools.nn.general.net import CHOICES

SD_MIN = 0.1                 # the standardisation's std at least this, the inputs within +-CLIP after it
CLIP = 10.0
FIRST = 1024                 # training groups before the first fit (it fixes the standardisation and the unit)
VAL_EVERY = 10               # every 10th plan (by its qid) is held out, all its groups
WAIT_CHUNKS = 4              # a chunk's features wait this many chunks for their outcomes at most (the default;
#                              run.py: wait_chunks of the horizon)
SE_K = 2.0                   # the shrink: the kind part's coefficient less this many standard errors
FLOOR = 1e-4                 # the advantages' scale at least this (budget share)
MODES = ("outcome", "branch", "both")


def width(card, kinds):
    """The input's width: the query card, the kind critic's numbers, the inertia's one-hot, the kinds allowed."""
    return int(card) + kc.WIDTH + 2 * int(kinds)


def wait_chunks(horizon_s, decision_s, steps):
    """How many chunks a chunk's features wait for their outcomes: an outcome comes at most the horizon after the plan's
    start, so after its end too - the horizon's chunks of `steps` decisions + 2 (a margin)."""
    return int(math.ceil(float(horizon_s) / max(1e-9, float(decision_s) * float(steps)))) + 2


def held_out(bid):
    """[n] bool: the held-out groups by their battle's number (every VAL_EVERY-th battle, whole)."""
    return (bid % VAL_EVERY) == 0


@torch.no_grad()
def features(gen, rec, minibatch=512):
    """(X [M, G, width], logp [M, G, K]) of the plans' records rec (Runtime.record's; module doc); zeros for a place
    that is not our group."""
    M = rec["kind"].shape[0]
    G, K = gen.cfg.places, gen.cfg.kinds
    xs, ps = [], []
    for mb in torch.arange(M, device=rec["kind"].device).split(minibatch):
        r = {k: v[mb] for k, v in rec.items() if torch.is_tensor(v)}
        c, d = gen(r, act={k: r[k] for k in CHOICES})
        hot = gen.prev_kind_hot(r, c, K).float()
        x = torch.cat([d["query"].detach().float(), kc.features(gen, r), hot, d["kind_ok"].float()], -1)
        xs.append(x * r["active"][:, :G, None].float())
        ps.append(d["kind_logp"].float())
    if not xs:
        dev = rec["kind"].device
        return torch.zeros(0, G, width(gen.cfg.card, K), device=dev), torch.zeros(0, G, K, device=dev)
    return torch.cat(xs), torch.cat(ps)


class Model(nn.Module):
    """Q of every kind (module doc)."""

    def __init__(self, width_, kinds, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width_, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, kinds))
        self.register_buffer("mu", torch.zeros(width_))
        self.register_buffer("sd", torch.ones(width_))
        self.register_buffer("unit", torch.ones(()))
        self.register_buffer("fixed", torch.zeros((), dtype=torch.bool))
        self.register_buffer("shrink", torch.zeros(()))

    def raw(self, x):
        """[.., K] Q in units (the regression's scale), not shrunk."""
        return self.net(((x - self.mu) / self.sd).clamp(-CLIP, CLIP))


def huber(a, b):
    return nn.functional.huber_loss(a, b, delta=1.0)


def advantage(q, kind, logp):
    """[..] A = Q(chosen) - sum_k pi(k) Q(k) (kind_critic.baseline_advantage on any leading shape)."""
    return kc.baseline_advantage(q, kind, logp)


def ev(y, p):
    """The explained variance of y by p (0 when y does not vary)."""
    v = y.var()
    return float(1 - (y - p).var() / v) if len(y) > 1 and float(v) > 0 else 0.0


def kind_coef(y, v, a, cluster=None):
    """(c, se): the least-squares coefficient of a in y ~ 1 + v + a (all [n]) and its standard error, robust to the
    rows' dependence within a cluster (cluster [n]: the battle's number - its plans, groups and sides share outcomes;
    None: each row its own; the sandwich with C / (C - 1)); (0, inf) when the system is singular or with < 3 clusters."""
    X = torch.stack([torch.ones_like(y), v, a], 1).double()
    xtx = X.T @ X
    if float(torch.linalg.det(xtx)) <= 1e-18:
        return 0.0, float("inf")
    bread = torch.linalg.inv(xtx)
    beta = bread @ (X.T @ y.double())
    e = y.double() - X @ beta
    cl = torch.arange(len(y), device=y.device) if cluster is None else cluster
    _, inv = torch.unique(cl, return_inverse=True)
    C = int(inv.max()) + 1 if len(inv) else 0
    if C < 3:
        return float(beta[2]), float("inf")
    sc = torch.zeros(C, 3, dtype=torch.float64, device=y.device).index_add_(0, inv, X * e[:, None])
    var = (bread @ (sc.T @ sc) @ bread) * (C / (C - 1))
    return float(beta[2]), float(var[2, 2].clamp(min=0).sqrt())


def pearson(x, y):
    """The correlation of x and y [n] (nan with fewer than 3 or no spread)."""
    if len(x) < 3:
        return float("nan")
    x, y = x.double() - x.double().mean(), y.double() - y.double().mean()
    d = float((x.pow(2).sum() * y.pow(2).sum()).sqrt())
    return float((x * y).sum()) / d if d > 0 else float("nan")


class OutcomeQ:
    """The model, its AdamW, the waiting features and the buffer (module doc)."""

    def __init__(self, kinds, card, device="cpu", buffer=30_000, warmup=10_000, ev_min=0.05, steps=64, minibatch=512,
                 lr=3e-4, hidden=128, dropout=0.1, decay=1e-2, norm="per-kind", shrink=True, seed=0, wait=WAIT_CHUNKS):
        self.K, self.card = int(kinds), int(card)
        self.W = width(card, kinds)
        self.device = torch.device(device)
        self.cap, self.warmup, self.ev_min = int(buffer), int(warmup), float(ev_min)
        self.steps, self.minibatch = int(steps), int(minibatch)
        self.first = min(FIRST, max(16, self.warmup // 2))
        self.use_shrink = bool(shrink)
        self.wait_n = int(wait)
        self.lost = 0                             # waiting features dropped without their outcome (since the stats)
        self.norm_kind = norm
        self.args = dict(hidden=hidden, dropout=dropout, lr=lr, decay=decay)
        self.model = Model(self.W, self.K, hidden, dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=decay)
        self.norm = gsignal.KindNorm(self.K, FLOOR) if norm == "per-kind" else None
        z = (lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=self.device))
        self.X, self.a, self.y, self.pid = z(0, self.W), z(0, dt=torch.long), z(0), z(0, dt=torch.long)
        self.lp, self.bid = z(0, self.K), z(0, dt=torch.long)
        self.wait = None                          # {"X" [w, G, W], "kind", "act" [w, G], "lp" [w, G, K], "qid", "age" [w]}
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7741)
        self.last = {}
        self.stats = {}

    def __len__(self):
        return len(self.a)

    # --- the samples --------------------------------------------------------------------------------------------
    def hold(self, X, lp, ready):
        """The features X, logp lp of the update's ended plans ready (with "qid") wait for their outcomes."""
        if ready is None or "qid" not in ready or not len(ready["qid"]):
            return
        G = X.shape[1]
        new = {"X": X.to(self.device), "kind": ready["kind"].to(self.device),
               "act": ready["active"][:, :G].to(self.device), "lp": lp.to(self.device),
               "qid": ready["qid"].to(self.device), "age": torch.zeros(len(X), dtype=torch.long, device=self.device)}
        self.wait = new if self.wait is None else {k: torch.cat([self.wait[k], new[k]]) for k in new}

    def join(self, out):
        """The outcomes known (Runtime's q_out: "qid", "y", "span", "ended", "bid" [n]) into the buffer, each plan's own
        groups; -> (plans joined, ended share, mean span)."""
        if self.wait is None or not len(self.wait["qid"]) or out is None or not len(out["qid"]):
            self._age()
            return 0, 0.0, 0.0
        w = self.wait
        qid = out["qid"].to(self.device)
        order = torch.argsort(w["qid"])
        pos = torch.searchsorted(w["qid"][order], qid).clamp(max=len(order) - 1)
        hit = w["qid"][order][pos] == qid
        idx = order[pos[hit]]                                   # the waiting plans of the outcomes known
        y = out["y"].to(self.device).float()[hit]
        bid = out.get("bid", out["qid"]).to(self.device).long()[hit]
        n = int(hit.sum())
        if n:
            act = w["act"][idx]                                  # [n, G]
            p, g = act.nonzero(as_tuple=True)
            self._add(w["X"][idx][p, g], w["kind"][idx][p, g], y[p], w["qid"][idx][p], w["lp"][idx][p, g], bid[p])
            keep = torch.ones(len(w["qid"]), dtype=torch.bool, device=self.device)
            keep[idx] = False
            self.wait = {k: v[keep] for k, v in w.items()}
        self._age()
        ended = out["ended"].to(self.device)[hit]
        span = out["span"].to(self.device).float()[hit]
        return n, (float(ended.float().mean()) if n else 0.0), (float(span.mean()) if n else 0.0)

    def _age(self):
        if self.wait is None:
            return
        self.wait["age"] += 1
        keep = self.wait["age"] <= self.wait_n
        if not bool(keep.all()):
            self.lost += int((~keep).sum())
            self.wait = {k: v[keep] for k, v in self.wait.items()}

    def _add(self, X, a, y, pid, lp, bid=None):
        cat = (lambda old, new: torch.cat([old, new])[-self.cap:])
        self.X, self.a, self.y = cat(self.X, X.float()), cat(self.a, a.long()), cat(self.y, y.float())
        self.pid, self.lp = cat(self.pid, pid.long()), cat(self.lp, lp.float())
        self.bid = cat(self.bid, (pid if bid is None else bid).long())

    # --- the fit ------------------------------------------------------------------------------------------------
    def _fix_scale(self, X, y):
        m = self.model
        if bool(m.fixed):
            return
        m.mu.copy_(X.mean(0))
        m.sd.copy_(X.std(0).clamp(min=SD_MIN) if len(X) > 1 else torch.ones_like(m.sd))
        m.unit.copy_(y.pow(2).mean().sqrt().clamp(min=1e-5))
        m.fixed.fill_(True)

    @torch.enable_grad()
    def fit(self):
        """(Gradients on: called under no_grad.) A few Huber steps of Q(chosen) on the outcomes, then on the held-out
        battles: ev, the kind part's coefficient and the shrink (module doc) -> stats (self.last): samples, held (the
        held-out groups), train, val, base (the held-out loss of the outcomes' mean), ev, ev_kind (the share of what V
        leaves that A explains), coef, coef_se (by battle), shrink."""
        st = {"samples": len(self)}
        val = held_out(self.bid)
        st["held"] = int(val.sum())
        tr = ~val
        if int(tr.sum()) < self.first:
            self.last = st
            return st
        Xt, at, yt = self.X[tr], self.a[tr], self.y[tr]
        self._fix_scale(Xt, yt)
        m, u = self.model, float(self.model.unit)
        m.train()
        tot = 0.0
        for _ in range(self.steps):
            i = torch.randint(len(at), (min(self.minibatch, len(at)),), generator=self.gen).to(self.device)
            loss = huber(m.raw(Xt[i]).gather(1, at[i, None])[:, 0], yt[i] / u)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            self.opt.step()
            tot += float(loss.detach())
        m.eval()
        st["train"] = tot / self.steps
        with torch.no_grad():
            Xv, av, yv, lv, bv = self.X[val], self.a[val], self.y[val] / u, self.lp[val], self.bid[val]
            if len(av) >= 32:
                q = m.raw(Xv)
                p = q.gather(1, av[:, None])[:, 0]
                pi = lv.exp()
                pi = pi / pi.sum(-1, keepdim=True).clamp(min=1e-12)
                v = (pi * q).sum(-1)
                a = p - v
                st["val"] = float(huber(p, yv))
                st["base"] = float(huber(torch.full_like(yv, float(yt.mean() / u)), yv))
                st["ev"] = ev(yv, p)
                st["ev_kind"] = ev(yv - v, a)
                coef, se = kind_coef(yv, v, a, bv)
                st["coef"], st["coef_se"] = coef, se
                m.shrink.fill_(min(1.0, max(0.0, coef - SE_K * se)) if self.use_shrink and math.isfinite(se)
                               else (1.0 if not self.use_shrink else 0.0))
            else:
                m.shrink.fill_(0.0)
            st["shrink"] = float(m.shrink)
        self.last = st
        return st

    def ready(self):
        """Enough groups, and the held-out ev at least ev_min."""
        s = self.last
        return len(self) >= self.warmup and "ev" in s and s["ev"] >= self.ev_min

    @torch.no_grad()
    def predict(self, X, shrunk=True):
        """[.., K] Q (budget share) of the features X [.., W]; shrunk: x the shrink (module doc)."""
        self.model.eval()
        q = self.model.raw(X.float().to(self.device)) * self.model.unit
        return (q * self.model.shrink if shrunk else q).to(X.device)

    # --- the update's estimate ---------------------------------------------------------------------------------
    @torch.no_grad()
    def attach(self, batch, actor):
        """batch (rollout.collect's): its ended plans' features wait for their outcomes, the outcomes known join the
        buffer, a fit, the agreement with the branches, and - once ready - batch["general"]["kind_q"] = {"q" [M, G, K]
        (shrunk, budget share), "scale" [K]} (learn.update's kind term). The log's numbers in self.stats."""
        g = batch.get("general")
        if g is None:
            return batch
        t0 = time.time()
        gen = actor.general
        ready = g.get("ready")
        X = lp = None
        if ready is not None and len(ready["kind"]):
            X, lp = features(gen, ready)
            self.hold(X, lp, ready)
        q_out = g.get("q_out")
        n, ended, span = self.join(q_out)
        st = {"matured": n, "ended_share": round(ended, 4), "span_s": round(span, 2),
              "waiting": 0 if self.wait is None else int(len(self.wait["qid"])), "lost": self.lost,
              "orphans": (0 if q_out is None else len(q_out["qid"])) - n}
        self.lost = 0
        if len(self):
            st["y_mean"] = round(float(self.y.mean()), 5)
            st["y_std"] = round(float(self.y.std()), 5) if len(self) > 1 else 0.0
        st.update(rounded(self.fit()))
        st["on"] = int(self.ready())
        br = g.get("branch")
        if br is not None and "alt" in br and bool(self.model.fixed) and len(br["kind"]):
            # Q(chosen) - Q(other) of the branch rows against their centred price
            Xb, _ = features(gen, br["rec"])
            E = len(br["kind"])
            qb = self.predict(Xb[torch.arange(E, device=Xb.device), br["group"]], shrunk=False)     # [E, K]
            diff = qb.gather(1, br["kind"][:, None]) - qb.gather(1, br["alt"].clamp(min=0))
            st["agree"] = rounded({"x": pearson(diff.mean(1), br["price"].float())})["x"]
            st["agree_n"] = E
        if X is not None and self.ready():
            q = self.predict(X, shrunk=False)                                              # [M, G, K]
            mine = ready["active"][:, :X.shape[1]]
            A = advantage(q, ready["kind"], lp)                                            # [M, G]
            if self.norm is not None:
                sc = self.norm.update(ready["kind"][mine].tolist(), A[mine].tolist())
            else:
                sc = [max(float(A[mine].pow(2).mean().sqrt()) if bool(mine.any()) else 0.0, FLOOR)] * self.K
            g["kind_q"] = {"q": q * self.model.shrink, "scale": torch.tensor(sc, device=q.device)}
            st["adv_rms"] = round(float(A[mine].pow(2).mean().sqrt()), 6) if bool(mine.any()) else 0.0
            st.update({f"scale_{i}": round(float(v), 6) for i, v in enumerate(sc)})
        st["seconds"] = round(time.time() - t0, 2)
        self.stats = st
        return batch

    # --- the checkpoint ----------------------------------------------------------------------------------------
    def state(self):
        """What a resumed run needs (the checkpoint's "train"): the model, AdamW, the norm, the buffer (features in
        float16)."""
        return {"width": self.W, "kinds": self.K, "model": self.model.state_dict(), "optim": self.opt.state_dict(),
                "norm": None if self.norm is None else self.norm.state(),
                "X": self.X.half(), "a": self.a, "y": self.y, "pid": self.pid, "lp": self.lp.half(), "bid": self.bid}

    def load(self, d):
        """A saved state(); one of another input width or kind count is dropped. -> loaded or not."""
        if not d or int(d.get("width", -1)) != self.W or int(d.get("kinds", -1)) != self.K:
            return False
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["optim"])
        if self.norm is not None:
            self.norm.load(d.get("norm"))
        dev = self.device
        self.X = d["X"].float().to(dev)[-self.cap:]
        self.a = d["a"].long().to(dev)[-self.cap:]
        self.y = d["y"].float().to(dev)[-self.cap:]
        self.pid = d["pid"].long().to(dev)[-self.cap:]
        self.lp = d["lp"].float().to(dev)[-self.cap:]
        self.bid = (d["bid"] if d.get("bid") is not None else d["pid"]).long().to(dev)[-self.cap:]
        return True


def rounded(st):
    return {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for k, v in st.items()}


def text(st):
    """One log line of the outcome Q's numbers (OutcomeQ.stats)."""
    if not st:
        return ""
    out = (f"kind Q {st.get('samples', 0)} groups (+{st.get('matured', 0)} plans, waiting {st.get('waiting', 0)}, "
           f"lost {st.get('lost', 0)}, ended {st.get('ended_share', 0.0):.2f}), on {st.get('on', 0)}")
    if "ev" in st:
        out += (f", ev {st['ev']:.3f} kind part {st.get('ev_kind', float('nan')):.3f} coef "
                f"{st.get('coef', float('nan')):+.2f} se {st.get('coef_se', float('nan')):.2f} shrink "
                f"{st.get('shrink', float('nan')):.2f}")
    if "agree" in st:
        out += f", agrees with the branches {st['agree']:+.3f} ({st.get('agree_n', 0)})"
    return out
