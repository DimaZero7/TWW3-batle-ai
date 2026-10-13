"""The ability critic: a small model of the ability branches' price (docs/en/training/training.md "Abilities by their
effect"; run.py --ability-critic, off by default).

The branches (ability_branch.py) price only the replayed moments (~105 of ~5000 an update) and the price is noisy
(the trade of 60 s of battle, 4 copies: S25 standard error 0.007 at a spread of 0.012 of the budget). The critic learns
the price from every replayed moment and predicts it at every moment an ability is ready, so the ability head gets a
signal at all of them, smoothed over many battles:

    features(obs, r, n, slot)  -> [M, WIDTH] what the policy itself sees at the decision: the unit's token without its
                                  card (health, men, morale, fatigue, melee, threats, recent melee / rout), the slot's
                                  ability input (owned, ready, recharge, active, its passport: which ability), the side's
                                  context (factions, the clock since the start, the damage timers; never the time to the
                                  end) and the neighbourhood from the observation's positions (nearest seen enemy; own
                                  units, in melee, wavering, routing, and seen enemies, in melee, within NEAR metres)
    Critic                     -> the predicted price of pressing the slot now (a share of the budget, as the branches')
    AbilityCritic              -> the model, its own optimiser (AdamW), the replay buffer (the last `buffer` replayed
                                  moments, kept across updates and saved with the checkpoint's "train"); fit(): a few
                                  steps of Huber regression an update, then the shrink calibrated on the held-out tenth;
                                  ready() once it has `warmup` samples and beats the mean there by MARGIN;
                                  reliability(): the real share of the buffer's price variance from the two halves of
                                  each price's copies (kept with it) - the ceiling of the critic's explained variance

Why it stays off on the S25 lord abilities (docs/en/training/training.md "Abilities by their effect"): the 60 s price is
mostly the copies' noise (its real share ~0.15-0.4) and the real part is not a function of what the critic sees - no
learner (ridge, boosted trees, MLPs of 32-256, cross-validated) explained more than 0 of 2 596 buffer prices or 447
probe prices. A critic that knew only the mean (+0.001 of the budget) would push the head to press at once whenever
ready, so the gate keeping it off is right.

The shrink: a noisy target and a few thousand samples, a small network still fits some noise (on a synthetic price with
noise 1.5x the signal, unshrunk predictions of a wider network without dropout explained less than nothing of the
noiseless price: -0.7). The prediction is
mean + shrink x (network - mean), the shrink in [0, 1] the least-squares one on the held-out tenth: where the network
knows little its prediction falls back to the mean price instead of passing its noise to the head.

Its own parameters and optimiser: not a head of the policy, so nothing of it reaches the shared layers. It is used only
in training (as the side's critic), so it adds nothing to what the game must provide.
"""
import math

import torch
from torch import nn

from tools.nn.model import abilities as mab
from tools.nn.model import observation as ob

NEAR = (35.0, 80.0)          # m: the auras' range (rally, stand your ground: 35 m), and a wider ring
FAR = 500.0                  # m: the nearest enemy's distance / FAR, capped at 1 (1 also when none is seen)
LOCAL = 2 + 6 * len(NEAR)
OWN = ob.INDEX["passport_0"]  # the token's dynamic columns and flags (before the unit's card)
WIDTH = OWN + mab.SIZE + ob.CONTEXT + LOCAL
SD_MIN = 0.1                 # the standardisation's std at least this (a column constant in the first samples, e.g. an
#                              ability not seen yet, is not blown up when it changes later) ...
CLIP = 10.0                  # ... and the standardised inputs within +-this
FIRST = 256                  # training samples before the first fit (it fixes the standardisation and the unit)
VAL_EVERY = 10               # every 10th sample (by its number) is held out: the validation tenth
MARGIN = 0.98                # ready: the held-out loss below this x the mean's (a chance fit of the shrink gains ~1/n)
_I = ob.INDEX


def features(obs, r, n, slot):
    """[M, WIDTH] the critic's input of the moments (r, n, slot: [M] long: the learner row, the unit, the ability slot)
    from the decision's observation of the learner rows (obs: "tokens", "ctx", "own", "pos", "abil", "abil_idx")."""
    tok = obs["tokens"][r].float()                                   # [M, N, TOKEN]
    me = tok.gather(1, n[:, None, None].expand(-1, 1, tok.shape[-1]))[:, 0, :OWN]
    k = slot.clamp(min=0)
    ab = mab.full(obs["abil"][r, n, k].float(), obs["abil_idx"][r, n, k])     # [M, SIZE]
    ctx = obs["ctx"][r].float()[:, :ob.CONTEXT]
    pos = obs["pos"][r].float() * ob.POS                             # [M, N, 2] m
    here = pos.gather(1, n[:, None, None].expand(-1, 1, 2))
    d = (pos - here).norm(dim=-1)                                    # [M, N]
    alive = tok[..., _I["men"]] > 0
    own = obs["own"][r] & alive
    own = own & (torch.arange(own.shape[1], device=own.device)[None] != n[:, None])
    foe = ~obs["own"][r] & alive & (tok[..., _I["visible"]] > 0)
    nearest = torch.where(foe, d, torch.full_like(d, 1e9)).min(1).values
    cols = [(nearest / FAR).clamp(max=1.0), (~foe.any(1)).float()]
    melee, wav, rout = (tok[..., _I[k]] > 0 for k in ("melee", "state_wavering", "state_routing"))
    for radius in NEAR:
        o, f = own & (d <= radius), foe & (d <= radius)
        cols += [x.float().sum(1) / 4.0 for x in (o, o & melee, o & wav, o & rout, f, f & melee)]
    return torch.cat([me, ab, ctx, torch.stack(cols, 1)], 1)


class Critic(nn.Module):
    """The price of pressing: the inputs standardised (the buffer's mean / std, fixed at the first fit), two hidden
    layers with dropout, in `unit` (the first fit's root mean square of the prices); the output shrunk to the training
    mean (`center`) by `shrink` (module doc)."""

    def __init__(self, width=WIDTH, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(width))
        self.register_buffer("sd", torch.ones(width))
        self.register_buffer("unit", torch.ones(()))
        self.register_buffer("fixed", torch.zeros((), dtype=torch.bool))
        self.register_buffer("center", torch.zeros(()))
        self.register_buffer("shrink", torch.zeros(()))

    def raw(self, x):
        """The network's output in units (the regression's scale), not shrunk."""
        return self.net(((x - self.mu) / self.sd).clamp(-CLIP, CLIP))[:, 0]

    def forward(self, x):
        """The predicted price (budget share)."""
        return (self.center + self.shrink * (self.raw(x) - self.center)) * self.unit


def huber(a, b):
    return nn.functional.huber_loss(a, b, delta=1.0)


class AbilityCritic:
    """The critic, its optimiser and its replay buffer (module doc)."""

    def __init__(self, device="cpu", buffer=25_000, warmup=2_000, steps=32, minibatch=256, lr=3e-4, seed=0,
                 hidden=64, dropout=0.2, decay=1e-2):
        self.device = torch.device(device)
        self.cap, self.warmup, self.steps, self.minibatch = int(buffer), int(warmup), int(steps), int(minibatch)
        self.model = Critic(hidden=hidden, dropout=dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=decay)
        self.X = torch.zeros(0, WIDTH, device=self.device)
        self.y = torch.zeros(0, device=self.device)
        self.ids = torch.zeros(0, dtype=torch.long, device=self.device)
        self.halves = torch.zeros(0, 2, device=self.device)  # each price's two halves of the copies (NaN: unknown)
        self.seen = 0                            # samples added ever (their numbers: the validation split)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7717)
        self.last = {}

    def __len__(self):
        return len(self.y)

    def add(self, X, y, halves=None):
        """New replayed moments: X [k, WIDTH], y [k] their measured price (budget share), halves [k, 2] the means of
        the two halves of its copies (reliability(): what of the price is real). The oldest go past `buffer`."""
        k = len(y)
        if not k:
            return
        h = (torch.full((k, 2), float("nan")) if halves is None else halves.detach().float()).to(self.device)
        self.halves = torch.cat([self.halves, h])[-self.cap:]
        ids = torch.arange(self.seen, self.seen + k, device=self.device)
        self.seen += k
        self.X = torch.cat([self.X, X.detach().float().to(self.device)])[-self.cap:]
        self.y = torch.cat([self.y, y.detach().float().to(self.device)])[-self.cap:]
        self.ids = torch.cat([self.ids, ids])[-self.cap:]

    def reliability(self):
        """The share of the buffer's price variance that is real, from the halves of the copies (split-half, Spearman-
        Brown: r = corr(half 1, half 2), 2r / (1 + r)); None with fewer than 50 prices that have them. It bounds the
        critic's explained variance."""
        h = self.halves[~self.halves.isnan().any(1)]
        if len(h) < 50:
            return None
        r = float(torch.corrcoef(h.T)[0, 1])
        return 2 * r / (1 + r) if r > -1 else -1.0

    def rms(self):
        """The buffer's root mean square of the prices (0 when empty)."""
        return float(self.y.pow(2).mean().sqrt()) if len(self.y) else 0.0

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
        """(Gradients on: it is called from Brancher.finish, which runs under no_grad.) A few steps of Huber regression on the buffer's training part, the shrink on the held-out tenth, and its
        loss there against "predict the training mean" -> stats (self.last): samples, train (the steps' loss), val, base
        (the mean's), ev (explained variance there), shrink."""
        val = (self.ids % VAL_EVERY) == 0
        Xt, yt, Xv, yv = self.X[~val], self.y[~val], self.X[val], self.y[val]
        st = {"samples": len(self.y)}
        if len(yt) < FIRST:
            self.last = st
            return st
        self._fix_scale(Xt, yt)
        m, u = self.model, float(self.model.unit)
        m.train()
        tot = 0.0
        for _ in range(self.steps):
            i = torch.randint(len(yt), (min(self.minibatch, len(yt)),), generator=self.gen).to(self.device)
            loss = huber(m.raw(Xt[i]), yt[i] / u)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            self.opt.step()
            tot += float(loss.detach())
        m.eval()
        st["train"] = tot / self.steps
        with torch.no_grad():
            m.center.fill_(float(yt.mean()) / u)
            if len(yv) >= 16:
                d, e = m.raw(Xv) - m.center, yv / u - m.center
                m.shrink.fill_(min(1.0, max(0.0, float((d * e).sum() / (d * d).sum().clamp(min=1e-12)))))
                pv = m(Xv)
                st["val"] = float(huber(pv / u, yv / u))
                st["base"] = float(huber(m.center.expand_as(yv), yv / u))
                v = yv.var()
                st["ev"] = float(1 - (yv - pv).var() / v) if float(v) > 0 else 0.0
            else:
                m.shrink.fill_(0.0)
            st["shrink"] = float(m.shrink)
        self.last = st
        return st

    def ready(self):
        """Enough samples, and the held-out tenth's error below the mean's by MARGIN."""
        s = self.last
        return len(self.y) >= self.warmup and "val" in s and s["val"] < MARGIN * s["base"]

    @torch.no_grad()
    def predict(self, X):
        """[k] the predicted price (budget share) of the moments X [k, WIDTH]."""
        if not len(X):
            return torch.zeros(0, device=X.device)
        self.model.eval()
        return self.model(X.float().to(self.device)).to(X.device)

    def state(self):
        """What a resumed run needs (checkpoint.py "train"): the model, AdamW, the buffer (features in float16)."""
        return {"width": WIDTH, "model": self.model.state_dict(), "optim": self.opt.state_dict(),
                "X": self.X.half(), "y": self.y, "ids": self.ids, "seen": self.seen, "halves": self.halves}

    def load(self, d):
        """A saved state(); one of another input width (the observation changed) is dropped. -> loaded or not."""
        if not d or int(d.get("width", -1)) != WIDTH:
            return False
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["optim"])
        self.X = d["X"].float().to(self.device)[-self.cap:]
        self.y = d["y"].float().to(self.device)[-self.cap:]
        self.ids = d["ids"].long().to(self.device)[-self.cap:]
        h = d.get("halves")                      # (older states: none kept)
        self.halves = (h.float().to(self.device) if h is not None
                       else torch.full((len(d["y"]), 2), float("nan"), device=self.device))[-self.cap:]
        self.seen = int(d["seen"])
        return True


def rounded(st):
    return {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for k, v in st.items()}
