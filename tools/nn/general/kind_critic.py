"""The mission kind critic: a small model of the mission branches' price (docs/en/training/general.md "The kind critic";
run.py --general-kind-critic 1, the default; the ability critic's way, tools/nn/train/ability_critic.py).

The branches (branch.py) price one group of ~2-5 % of the plans (64-128 rows an update of ~2 500 plans x ~3 groups),
and a price is noisy (the trade of 60 s of battle, 4 copies). The critic learns from every price and gives every group
of every ended plan an estimate, smoothed over many battles:

    features(gen, rec)  -> [M, G, WIDTH] what the general itself reads of each own group at the plan, as fixed numbers
                           (not its learned card: the buffer's rows stay comparable while the general learns): the
                           group's threat and fight numbers (net.THREAT, net.FIGHT: the nearest enemy group's distance,
                           the strength ratio, the shooters on both sides, contact, under fire, in melee), the tiles
                           under its centre (net.group_tiles), its size, the plan's own and enemy groups, deployment
    Critic              -> Q [.., K]: a number per mission kind (budget share); the price of a branch is
                           Q(chosen) - Q(other), so the regression is on that difference, copy by copy (each copy has its
                           own other kind)
    KindCritic          -> the model, its AdamW, the replay buffer (the last `buffer` prices, kept across updates and
                           saved with the checkpoint's "train"); fit(): a few steps of Huber regression an update, then
                           the shrink calibrated on the held-out tenth; ready() once it has `warmup` prices and beats
                           "no difference" there by MARGIN; advantage(): Q(chosen) - sum_k pi(k) Q(k) of each group (the
                           policy's baseline over the kinds it may take: an advantage with an action-independent
                           baseline, COMA's).

The shrink: prediction = shrink x Q, the shrink in [0, 1] the least-squares one on the held-out tenth (the ability
critic shrinks to the mean price; here the null is "no kind better than another", 0): where the network knows little its
estimate falls to 0 instead of passing its noise to the head.

Its own parameters and optimiser: not a head of the general, nothing of it reaches the general's layers; used only in
training.
"""
import math

import torch
from torch import nn

from tools.nn.general import net as gnet
from tools.nn.general import tiles as gtiles

EXTRA = ("units", "own_groups", "enemy_groups", "deploy")
WIDTH = len(gnet.THREAT) + len(gnet.FIGHT) + len(gtiles.FEATURES) + len(EXTRA)
SD_MIN = 0.1                 # the standardisation's std at least this, the inputs within +-CLIP after it
CLIP = 10.0
FIRST = 256                  # training prices before the first fit (it fixes the standardisation and the unit)
VAL_EVERY = 10               # every 10th price (by its number) is held out
MARGIN = 0.98                # ready: the held-out loss below this x the loss of "no difference" (0)


@torch.no_grad()
def features(gen, rec):
    """[M, G, WIDTH] the critic's input of our groups of the plans' records rec (Runtime.record's: the inputs and the
    plan's "main", "active"); zeros for a place that is not a group."""
    G = gen.cfg.places
    c = {"main": rec["main"]}
    inp = {k: rec[k] for k in ("num", "pos", "on", "own", "tiles", "tile_ok")}
    inp["fight"] = rec.get("fight")
    th = gen.threat(inp, c).float()                                        # [M, G, 6]
    fi = gen.fight(inp, c).float()                                         # [M, G, 2]
    gt = gen.group_tiles(inp, c)[:, :G].float()                            # [M, G, F]
    _, n, _ = gen.centres(inp, c)                                          # [M, 2G]
    act = rec["active"]
    own = act[:, :G].float().sum(1, keepdim=True).expand(-1, G)
    foe = act[:, G:].float().sum(1, keepdim=True).expand(-1, G)
    dep = rec["deploy"].float().reshape(-1, 1).expand(-1, G)
    extra = torch.stack([n[:, :G].float() / 8.0, own / G, foe / G, dep], -1)
    x = torch.cat([th, fi, gt, extra], -1)
    return x * act[:, :G, None].float()


class Critic(nn.Module):
    """Q of every kind: the inputs standardised (fixed at the first fit), two hidden layers with dropout, in `unit`
    (the first fit's root mean square of the prices), shrunk to 0 by `shrink` (module doc)."""

    def __init__(self, kinds, width=WIDTH, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, kinds))
        self.register_buffer("mu", torch.zeros(width))
        self.register_buffer("sd", torch.ones(width))
        self.register_buffer("unit", torch.ones(()))
        self.register_buffer("fixed", torch.zeros((), dtype=torch.bool))
        self.register_buffer("shrink", torch.zeros(()))

    def raw(self, x):
        """[.., K] the network's Q in units (the regression's scale), not shrunk."""
        return self.net(((x - self.mu) / self.sd).clamp(-CLIP, CLIP))

    def forward(self, x):
        """[.., K] the predicted Q (budget share), shrunk."""
        return self.shrink * self.raw(x) * self.unit


def huber(a, b):
    return nn.functional.huber_loss(a, b, delta=1.0)


def pair_diff(q, a, b):
    """q [k, K], a [k] the chosen kind, b [k, C] the copies' other kinds -> [k, C] Q(a) - Q(b_c)."""
    return q.gather(1, a[:, None]) - q.gather(1, b)


def baseline_advantage(q, kind, logp):
    """[.., G] Q(chosen) - sum_k pi(k) Q(k): q [.., G, K], kind [.., G] the chosen kind, logp [.., G, K] the policy's
    log-probabilities (a masked kind: ~-inf, weight 0)."""
    p = logp.float().exp()
    p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
    return q.gather(-1, kind[..., None].clamp(min=0))[..., 0] - (p * q).sum(-1)


class KindCritic:
    """The critic, its optimiser and its replay buffer (module doc)."""

    def __init__(self, kinds, device="cpu", buffer=25_000, warmup=1_000, steps=32, minibatch=256, lr=3e-4, seed=0,
                 hidden=64, dropout=0.2, decay=1e-2):
        self.K = int(kinds)
        self.device = torch.device(device)
        self.cap, self.warmup, self.steps, self.minibatch = int(buffer), int(warmup), int(steps), int(minibatch)
        self.first = min(FIRST, max(16, self.warmup // 2))
        self.model = Critic(self.K, hidden=hidden, dropout=dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=decay)
        self.X = torch.zeros(0, WIDTH, device=self.device)
        self.a = torch.zeros(0, dtype=torch.long, device=self.device)
        self.b = None                              # [n, C] the copies' other kinds
        self.d = None                              # [n, C] the copies' prices
        self.ids = torch.zeros(0, dtype=torch.long, device=self.device)
        self.seen = 0
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7727)
        self.last = {}

    def __len__(self):
        return len(self.a)

    def add(self, X, a, b, d):
        """New prices: X [k, WIDTH] the priced group's features, a [k] its chosen kind, b [k, C] each copy's other kind,
        d [k, C] each copy's price (chosen - other, budget share). The oldest go past `buffer`; a buffer of another
        copy count starts afresh."""
        k = len(a)
        if not k:
            return
        b, d = b.long().to(self.device), d.detach().float().to(self.device)
        if self.b is not None and self.b.shape[1] != b.shape[1]:
            self.__init__(self.K, self.device, self.cap, self.warmup, self.steps, self.minibatch)
        ids = torch.arange(self.seen, self.seen + k, device=self.device)
        self.seen += k
        cat = (lambda old, new: new if old is None else torch.cat([old, new])[-self.cap:])
        self.X = cat(self.X, X.detach().float().to(self.device))
        self.a = cat(self.a, a.long().to(self.device))
        self.b, self.d = cat(self.b, b), cat(self.d, d)
        self.ids = cat(self.ids, ids)

    def rms(self):
        """The buffer's root mean square of the prices (mean over the copies; 0 when empty)."""
        return float(self.d.mean(1).pow(2).mean().sqrt()) if self.d is not None and len(self.d) else 0.0

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
        """(Gradients on: called under no_grad.) A few steps of Huber regression of Q(a) - Q(b_c) on the copies'
        prices, the shrink on the held-out tenth and its loss there against "no difference" -> stats (self.last):
        samples, train, val, base, ev (the explained variance of the held-out prices, the copies' mean), shrink."""
        st = {"samples": len(self)}
        if self.d is None:
            self.last = st
            return st
        val = (self.ids % VAL_EVERY) == 0
        tr = ~val
        if int(tr.sum()) < self.first:
            self.last = st
            return st
        Xt, at, bt, dt = self.X[tr], self.a[tr], self.b[tr], self.d[tr]
        self._fix_scale(Xt, dt.mean(1))
        m, u = self.model, float(self.model.unit)
        m.train()
        tot = 0.0
        for _ in range(self.steps):
            i = torch.randint(len(at), (min(self.minibatch, len(at)),), generator=self.gen).to(self.device)
            loss = huber(pair_diff(m.raw(Xt[i]), at[i], bt[i]), dt[i] / u)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            self.opt.step()
            tot += float(loss.detach())
        m.eval()
        st["train"] = tot / self.steps
        with torch.no_grad():
            Xv, av, bv, dv = self.X[val], self.a[val], self.b[val], self.d[val]
            if len(av) >= 16:
                p = pair_diff(m.raw(Xv), av, bv).mean(1)                       # [v] units
                y = dv.mean(1) / u
                m.shrink.fill_(min(1.0, max(0.0, float((p * y).sum() / (p * p).sum().clamp(min=1e-12)))))
                pv = float(m.shrink) * p
                st["val"] = float(huber(pv, y))
                st["base"] = float(huber(torch.zeros_like(y), y))
                v = y.var()
                st["ev"] = float(1 - (y - pv).var() / v) if float(v) > 0 else 0.0
            else:
                m.shrink.fill_(0.0)
            st["shrink"] = float(m.shrink)
        self.last = st
        return st

    def ready(self):
        """Enough prices, and the held-out error below "no difference"'s by MARGIN."""
        s = self.last
        return len(self) >= self.warmup and "val" in s and s["val"] < MARGIN * s["base"]

    @torch.no_grad()
    def predict(self, X):
        """[.., K] the predicted Q (budget share) of the features X [.., WIDTH]."""
        self.model.eval()
        return self.model(X.float().to(self.device)).to(X.device)

    def state(self):
        """What a resumed run needs (the checkpoint's "train"): the model, AdamW, the buffer (features in float16)."""
        return {"width": WIDTH, "kinds": self.K, "model": self.model.state_dict(), "optim": self.opt.state_dict(),
                "X": self.X.half(), "a": self.a, "b": self.b, "d": self.d, "ids": self.ids, "seen": self.seen}

    def load(self, d):
        """A saved state(); one of another input width or kind count is dropped. -> loaded or not."""
        if not d or int(d.get("width", -1)) != WIDTH or int(d.get("kinds", -1)) != self.K:
            return False
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["optim"])
        dev = self.device
        self.X = d["X"].float().to(dev)[-self.cap:]
        self.a = d["a"].long().to(dev)[-self.cap:]
        self.b = None if d["b"] is None else d["b"].long().to(dev)[-self.cap:]
        self.d = None if d["d"] is None else d["d"].float().to(dev)[-self.cap:]
        self.ids = d["ids"].long().to(dev)[-self.cap:]
        self.seen = int(d["seen"])
        return True


def rounded(st):
    return {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for k, v in st.items()}
