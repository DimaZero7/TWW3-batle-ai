"""The operative's memory (prototype, not in the network yet; tmp/memory/design.md): what the side really saw, kept for
the whole battle - one function on the simulator's raw state and on the bridge's (net/tactic/raw.py RAW, or its prefix
net/unit/view.py RAW: TR.Seer.raw / OperativeSeer.raw; contract/test_memory_bridge_parity_torch.py).

Honest: an enemy's columns enter the store only in a second it is seen - raw `vis` (the simulator's u["vis"] = present
and not hidden, sim/stealth.py; the game's row `v` = is_visible_to_alliance, bridge/companion/exchange.arrays). Its
death is public (the game shows a destroyed unit): only the flag, the place stays where it was last seen. Own units are
always seen. The store reads nothing else of the raw state, so an unseen enemy's true columns cannot change it. No
column of the known divergences (contract/parity.md: vx, vz, f - the game's shift a second against the simulator's
speed) is read: the motion is the shift of the last seen places between ticks.

Whole: the store is updated every second (see), the records are taken every decision (tick, 15 s) from the window's
flags - a rout, a rally, a vanishing between two decisions is kept; the counters and the last seen rows last the battle.

    Store(B, N, side, T)    the side's memory of B battles of N slots; T ticks the blocks read (4: 60 s, 40: 10 min;
                            the ring keeps at least LOSS_LAG + 1 for the summary's losses)
      .see(raw, t)          once a second: the last seen raw row, the counters (routs, rallies seen), the window's flags
      .tick(t, axes)        once a decision: the units' records [B, N, F] (REC) into the ring (newest first), the track,
                            the window reset; returns the records
      .summary(t, axes)     [B, N, S] (SUM): the code's compressed features - counters, last seen, the track of its last
                            seen ticks, losses over 15 / 60 / 300 s, the share seen in the last minute
      .ghosts(t)            (rows [B, N, RAW] the last seen raw rows, mask [B, N] the known alive enemies unseen now,
                            age [B, N] s) - for the input's rows at their last seen place
      .sequence(members)    (seq [B, G, T, F + 1], mask [B, G, T], age [B, G, T] s): a group's members (now) back
                            through their records - the centre, the shares; the group is today's, nothing to match
    Orders(B, G)            the operative's own decisions per group: the skill, since when, the changes in 60 / 300 s
    Summary, TimeAttention, SeqGRU    the blocks compared (tmp/memory/bench.py, learn.py): [R, ...] -> [R, W]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.operative import cfg
from net.unit import view as V

REC = ("known", "seen", "seen_share", "age", "right", "forward", "cos", "sin", "men", "routing", "routed", "rallied",
       "melee", "firing", "dead", "enemy", "vanished", "appeared")
RI = {k: i for i, k in enumerate(REC)}
SUM = ("routs", "rallies", "since_rout", "has_routed", "since_rally", "has_rallied", "age", "tr1_s", "tr1_f",
       "tr2_s", "tr2_f", "tr3_s", "tr3_f", "lost_15", "lost_60", "lost_300", "seen_60")
SI = {k: i for i, k in enumerate(SUM)}
ORD = ("on", "since", "changes_60", "changes_300")
AGE_S = 600.0          # a record's age / 10 min (within 3: 30 min)
SINCE_S = 300.0        # the counters' times / 5 min (within 2; never: 2)
TRACK = 4              # the last seen positions at the last 4 ticks it was seen (newest first: [0] is the last tick)
TRACK_M = 100.0        # the track's offsets from the last seen place / 100 m
LOSS_LAGS = (1, 4, 20)  # the summary's losses: men now less men 1 / 4 / 20 ticks ago (15 s / 1 min / 5 min)
I = V.I


class Store:
    def __init__(self, B, N, side, T=40, device="cpu"):
        """side [B] long (the side whose memory it is) or an int."""
        side = torch.as_tensor(side, device=device)
        self.side = side.expand(B).clone() if side.dim() == 0 else side.clone()
        self.B, self.N, self.T, self.dev = B, N, T, device
        L = max(T, LOSS_LAGS[-1] + 1)                                # the ring's length
        z = lambda dt=torch.float32: torch.zeros((B, N), dtype=dt, device=device)        # noqa: E731
        self.last = None                       # [B, N, RAW] the last seen raw row (zeros: never)
        self.known, self.dead, self.seen, self.prev_seen = z(torch.bool), z(torch.bool), z(torch.bool), z(torch.bool)
        self.enemy = z(torch.bool)
        self.t_seen, self.t_rout, self.t_rally = z() - 1, z() - 1, z() - 1
        self.routs, self.rallies = z(), z()
        self.w_seen, self.w_rout, self.w_rally, self.w_melee = z(), z(torch.bool), z(torch.bool), z(torch.bool)
        self.w_fire = z(torch.bool)
        self.w_vanish, self.w_appear = z(torch.bool), z(torch.bool)
        self.w_s = 0
        self.track = torch.zeros((B, N, TRACK, 2), device=device)       # world x, z
        self.track_n = z(torch.long)
        self.rec = torch.zeros((L, B, N, len(REC)), device=device)     # newest first
        self.rec_t = torch.full((L,), -1.0, device=device)
        self.t = 0.0

    def see(self, raw, t):
        """One second of raw [B, N, RAW] at battle time t (s)."""
        if self.last is None:
            self.last = torch.zeros_like(raw)
        present = raw[..., I["side"]] > 0
        alive = present & (raw[..., I["alive"]] > 0)
        own = present & (raw[..., I["side"]] == self.side[:, None].to(raw.dtype))
        self.enemy = present & ~own
        vis = raw[..., I["vis"]] > 0
        see = alive & (own | vis)
        was_r = self.known & (self.last[..., I["r"]] > 0)
        r_now = raw[..., I["r"]] > 0
        rout = see & r_now & ~was_r                      # a rout seen starting (first seen routing counts)
        rally = see & ~r_now & was_r                     # seen routing last time, seen standing now
        self.routs = self.routs + rout.float()
        self.rallies = self.rallies + rally.float()
        self.t_rout = torch.where(rout, torch.full_like(self.t_rout, t), self.t_rout)
        self.t_rally = torch.where(rally, torch.full_like(self.t_rally, t), self.t_rally)
        died = present & ~alive & self.known & ~self.dead            # public: the game shows a destroyed unit
        self.dead = self.dead | died
        self.w_seen = self.w_seen + see.float()
        self.w_rout = self.w_rout | (see & r_now)
        self.w_rally = self.w_rally | rally
        self.w_melee = self.w_melee | (see & (raw[..., I["m"]] > 0))
        self.w_fire = self.w_fire | (see & (raw[..., I["fire"]] > 0))
        self.w_vanish = self.w_vanish | (self.prev_seen & ~see & alive)
        self.w_appear = self.w_appear | (see & ~self.prev_seen)
        self.last = torch.where(see[..., None], raw, self.last)
        self.t_seen = torch.where(see, torch.full_like(self.t_seen, t), self.t_seen)
        self.known = self.known | see
        self.seen, self.prev_seen = see, see
        self.w_s += 1
        self.t = t

    def _local(self, axes, x, z):
        return axes.local(x, z)

    def tick(self, t, axes):
        """The decision's records [B, N, F] (module doc), pushed into the ring; the window reset."""
        k = float(cfg()["place_m"])
        L = self.last
        known = self.known
        kf = known.float()
        age = torch.where(known, (t - self.t_seen) / AGE_S, torch.zeros_like(self.t_seen)).clamp(0, 3)
        s, f = self._local(axes, L[..., I["x"]], L[..., I["z"]])
        rel = (L[..., I["b"]] - axes.heading[:, None]) * (torch.pi / 180)
        cols = {"known": kf, "seen": self.seen.float(), "seen_share": self.w_seen / max(self.w_s, 1), "age": age,
                "right": s / k, "forward": f / k, "cos": torch.cos(rel), "sin": torch.sin(rel),
                "men": L[..., I["men"]], "routing": (L[..., I["r"]] > 0).float(), "routed": self.w_rout.float(),
                "rallied": self.w_rally.float(), "melee": self.w_melee.float(), "firing": self.w_fire.float(),
                "dead": self.dead.float(), "enemy": self.enemy.float(), "vanished": self.w_vanish.float(),
                "appeared": self.w_appear.float()}
        rec = torch.stack([cols[n] for n in REC], -1) * kf[..., None]
        self.rec = torch.cat([rec[None], self.rec[:-1]])
        self.rec_t = torch.cat([torch.tensor([float(t)], device=self.dev), self.rec_t[:-1]])
        hit = self.w_seen > 0                                       # seen in this window: its place joins the track
        pos = torch.stack([L[..., I["x"]], L[..., I["z"]]], -1)
        pushed = torch.cat([pos[:, :, None], self.track[:, :, :-1]], 2)
        self.track = torch.where(hit[..., None, None], pushed, self.track)
        self.track_n = torch.where(hit, (self.track_n + 1).clamp(max=TRACK), self.track_n)
        for n in ("w_rout", "w_rally", "w_melee", "w_fire", "w_vanish", "w_appear"):
            setattr(self, n, torch.zeros_like(getattr(self, n)))
        self.w_seen = torch.zeros_like(self.w_seen)
        self.w_s = 0
        return rec

    def summary(self, t, axes):
        """[B, N, S] (SUM) after the decision's tick."""
        known = self.known
        def since(tt):
            return torch.where(tt < 0, torch.full_like(tt, 2.0), ((t - tt) / SINCE_S).clamp(0, 2))
        L = self.last
        ls, lf = self._local(axes, L[..., I["x"]], L[..., I["z"]])
        tr = []
        for j in (1, 2, 3):
            ts, tf = self._local(axes, self.track[..., j, 0], self.track[..., j, 1])
            ok = (self.track_n > j).float()
            tr += [(ts - ls) / TRACK_M * ok, (tf - lf) / TRACK_M * ok]
        men = L[..., I["men"]]
        lost = []
        for lag in LOSS_LAGS:
            was = self.rec[lag, ..., RI["men"]]
            ok = (self.rec_t[lag] >= 0) & (self.rec[lag, ..., RI["known"]] > 0)
            lost.append(torch.where(ok, (was - men).clamp(min=0), torch.zeros_like(men)))
        n = 4
        have = (self.rec_t[:n] >= 0).float()
        seen60 = (self.rec[:n, ..., RI["seen_share"]] * have[:, None, None]).sum(0) / have.sum().clamp(min=1)
        age = torch.where(known, ((t - self.t_seen) / SINCE_S).clamp(0, 6), torch.zeros_like(men))
        cols = [(self.routs / 3).clamp(max=1), (self.rallies / 3).clamp(max=1), since(self.t_rout),
                (self.t_rout >= 0).float(), since(self.t_rally), (self.t_rally >= 0).float(), age]
        cols += tr + lost + [seen60]
        return torch.stack(cols, -1) * known.float()[..., None]

    def ghosts(self, t):
        mask = self.known & self.enemy & ~self.dead & ~self.seen
        return self.last, mask, torch.where(mask, t - self.t_seen, torch.zeros_like(self.t_seen))

    def sequence(self, members, t=None, T=None):
        """members [B, G, N] bool (the groups now: own - groups.members(), the enemy's - the code's cut) -> (seq
        [B, G, T, F + 1]: the members' mean record by tick (position: the centre of the known ones) and the share of
        the members known; mask [B, G, T]; age [B, G, T] s of each tick). T: the last T ticks (None: the store's)."""
        t = self.t if t is None else t
        T = T or self.T
        return sequence_of(self.rec[:T], self.rec_t[:T], members, t)

    def units(self, t=None, T=None):
        """(seq [B, N, T, F + 1], mask [B, N, T], age [B, N, T]): each unit's own records (the per-unit variant)."""
        t = self.t if t is None else t
        T = T or self.T
        rec = self.rec[:T].permute(1, 2, 0, 3)                        # [B, N, T, F]
        have = self.rec_t[:T] >= 0
        mask = (rec[..., RI["known"]] > 0) & have[None, None]
        seq = torch.cat([rec, mask[..., None].float()], -1)
        age = torch.where(have, t - self.rec_t[:T], torch.zeros_like(self.rec_t[:T]))[None, None].expand(mask.shape)
        return seq * mask[..., None], mask, age

    def state(self):
        """The store's tensors (the parity test compares them)."""
        return {n: getattr(self, n) for n in ("last", "known", "dead", "seen", "t_seen", "t_rout", "t_rally", "routs",
                                               "rallies", "track", "track_n", "rec", "rec_t")}


def sequence_of(rec_ring, rec_t, members, t):
    """Store.sequence on a stored ring: rec_ring [T, B, N, F], rec_t [T] (or [B, T]: the training's slices)."""
    rec = rec_ring.permute(1, 0, 2, 3)                           # [B, T, N, F]
    w = members.float()[:, None] * rec[..., RI["known"]][:, :, None]       # [B, T, G, N]
    k = w.sum(-1)
    mean = torch.einsum("btgn,btnf->btgf", w, rec) / k.clamp(min=1)[..., None]
    share = k / members.float().sum(-1).clamp(min=1)[:, None]
    seq = torch.cat([mean, share[..., None]], -1).permute(0, 2, 1, 3)      # [B, G, T, F + 1]
    rt = rec_t if rec_t.dim() == 2 else rec_t[None].expand(rec.shape[0], -1)   # [B, T]
    have = rt >= 0
    mask = (k > 0).permute(0, 2, 1) & have[:, None]
    tt = torch.as_tensor(t, dtype=rt.dtype, device=rt.device)
    tt = tt[:, None] if tt.dim() == 1 else tt
    age = torch.where(have, tt - rt, torch.zeros_like(rt))[:, None].expand(mask.shape)
    return seq * mask[..., None], mask, age


class Orders:
    """The operative's own decisions per group: note(t, skill [B, G] long, on [B, G] bool) every decision ->
    [B, G, len(ORD)]: on, the seconds since the skill changed / 300 (within 2), the changes in the last 60 / 300 s
    / 3 (within 2)."""

    def __init__(self, B, G, device="cpu", keep=8):
        self.skill = torch.full((B, G), -1, dtype=torch.long, device=device)
        self.since = torch.full((B, G), -1.0, device=device)
        self.changes = torch.full((B, G, keep), -1e9, device=device)

    def note(self, t, skill, on):
        ch = on & (skill != self.skill)
        self.since = torch.where(ch, torch.full_like(self.since, t), self.since)
        pushed = torch.cat([torch.full_like(self.changes[..., :1], t), self.changes[..., :-1]], -1)
        self.changes = torch.where(ch[..., None], pushed, self.changes)
        self.skill = torch.where(on, skill, torch.full_like(skill, -1))
        age = t - self.changes
        n60 = ((age <= 60).float().sum(-1) / 3).clamp(max=2)
        n300 = ((age <= 300).float().sum(-1) / 3).clamp(max=2)
        x = torch.stack([on.float(), ((t - self.since) / SINCE_S).clamp(0, 2), n60, n300], -1)
        return x * on.float()[..., None]


# the blocks: R entities (a group or a unit of a battle) -> [R, W]

class Summary(nn.Module):
    """C: the newest record + the code's summary -> W (no time axis)."""

    def __init__(self, rec=len(REC) + 1, summ=len(SUM), width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(rec + summ, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())

    def forward(self, now, summ):
        return self.net(torch.cat([now, summ], -1))


def _time(age):
    a = age / AGE_S
    return torch.stack([a, torch.log1p(age / 15.0) / 5.0], -1)


class TimeAttention(nn.Module):
    """A: the newest record asks the T records (each with its age) - one attention layer + a feed-forward."""

    def __init__(self, rec=len(REC) + 1, width=64, heads=4):
        super().__init__()
        self.q = nn.Linear(rec, width)
        self.kv = nn.Linear(rec + 2, width)
        self.att = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n1 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))
        self.n2 = nn.LayerNorm(width)

    def forward(self, seq, mask, age):
        """seq [R, T, F], mask [R, T] (newest first), age [R, T] s."""
        q = self.q(seq[:, :1])
        kv = self.kv(torch.cat([seq, _time(age)], -1))
        m = mask.clone()
        m[:, 0] = True                                   # (an entity with no record asks its empty newest: no NaN)
        a, _ = self.att(q, kv, kv, key_padding_mask=~m, need_weights=False)
        h = self.n1(q + a)
        return self.n2(h + self.ff(h))[:, 0]


class SeqGRU(nn.Module):
    """G: the T records oldest to newest through a GRU (as the unit level's MemoryGRU), the last state."""

    def __init__(self, rec=len(REC) + 1, width=64):
        super().__init__()
        self.gru = nn.GRU(rec + 3, width, batch_first=True)

    def forward(self, seq, mask, age):
        x = torch.cat([seq, mask[..., None].float(), _time(age)], -1).flip(1)
        _, h = self.gru(x)
        return h[0]
