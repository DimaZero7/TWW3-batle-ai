"""The tactician's first drill (design/tactic "Первое упражнение"; the user's decisions 01.11). The stupidity guarded:
the shooters take the first blow while the infantry idles behind them.

    specs()     n random battles: 2 infantry + 1 lobbing, front-firing shooters against 3 infantry of the other faction;
                half the starts the shooters in front, half the infantry; the group's zone a 2 x 2 square shifted 50-120 m any
                way from the group; the enemy's centre ~300 m straight ahead of the zone's centre (so the zone is turned
                to it and the time to the contact hardly varies); flat
    build()     the state, the parcels (one a battle: my 3 slots, the target = the 3 enemies - the page names no target,
                with one the near window and the numbers show them), the order (verb 'hold', pace 'any')
    Script      the enemy: walks at the nearest standing enemy, the target looked at again once in 5 s
    Ender       the end: one side has nothing standing, fight_s after the first contact, or the limit
    Watch       the contact: the first moment one of mine and an enemy touch; 'between' - the shooters untouched and every
                touching enemy's straight line to the shooters' centre crosses an infantry formation; 'hit' - the
                shooters touched at that moment or within soon_s of it

The numbers: net/config/tactic.json drill1, reward.
"""
import math

import torch

from sim import geometry, scenario
from sim import replay
from net.tactic import cfg
from net.tactic import parcel as PC
from net.tactic import raw as TR
from net.tactic.decide import order_vec
from net.unit import drill_hold as DH
from net.unit import zmap as Z

STARTS = ("shooters_front", "infantry_front")
H = 3                 # slots a side: my 3 first, the enemy's 3 after


def specs(n, gen, fixed=None):
    """n random battles (dicts); fixed: a list of starts to cycle (the exam)."""
    c = cfg()["drill1"]
    out = []
    for k in range(n):
        u = lambda a, b: a + (b - a) * float(torch.rand(1, generator=gen))       # noqa: E731
        ri = lambda m: int(torch.randint(m, (1,), generator=gen))                  # noqa: E731
        ours = "emp" if float(torch.rand(1, generator=gen)) < 0.5 else "skv"
        theirs = "skv" if ours == "emp" else "emp"
        pick = lambda names: names[ri(len(names))]                                 # noqa: E731
        infantry = [pick(c["roster"][ours]["infantry"]), pick(c["roster"][ours]["infantry"])]
        shooters = pick(c["roster"][ours]["shooters"])
        out.append({"ours": ours, "theirs": theirs, "infantry": infantry, "shooters": shooters,
                    "foes": [pick(c["roster"][theirs]["infantry"]) for _ in range(H)],
                    "start": fixed[k % len(fixed)] if fixed else STARTS[ri(2)],
                    "b": u(0.0, 360.0), "dist": u(*c["distance_m"]), "shift": u(*c["shift_m"]), "shift_b": u(0.0, 360.0),
                    "shooter_slot": ri(H)})
    return out


def exam_specs(seed=None, n=None):
    """The fixed exam: the two starts in turn."""
    c = cfg()["exam"]
    return specs(int(c["battles"] if n is None else n), torch.Generator().manual_seed(int(c["seed"] if seed is None else seed)),
                 fixed=list(STARTS))


def zone_of(s):
    """(x, z, heading) of the battle's zone: the group's centre (0, 0) shifted; the heading: at the enemy."""
    a = math.radians(s["shift_b"])
    return s["shift"] * math.sin(a), s["shift"] * math.cos(a), s["b"]


def enemy_centre(s):
    x, z, b = zone_of(s)
    r = math.radians(b)
    return x + s["dist"] * math.sin(r), z + s["dist"] * math.cos(r)


def facing(s):
    """The group's start: its bearing at the enemy's centre."""
    ex, ez = enemy_centre(s)
    return math.degrees(math.atan2(ex, ez)) % 360


def layout(s):
    """[(key, right, forward)] of my three by slot, in the group's axes (its centre at 0, 'forward' at the enemy)."""
    c = cfg()["drill1"]
    half, gap = float(c["rows_m"]) / 2, float(c["infantry_gap_m"]) / 2
    front = s["start"] == "shooters_front"
    sh = (s["shooters"], 0.0, half if front else -half)
    inf = [(s["infantry"][0], -gap, -half if front else half), (s["infantry"][1], gap, -half if front else half)]
    slots = inf[:]
    slots.insert(s["shooter_slot"], sh)
    return slots


def army(s):
    c = cfg()["drill1"]
    b0 = facing(s)
    mine = []
    for key, r, f in layout(s):
        x, z = Z.world(torch.tensor(r), torch.tensor(f), 0.0, 0.0, b0)
        mine.append({"key": key, "x": float(x), "z": float(z), "b": b0})
    zx, zz, zb = zone_of(s)
    foes = []
    for i, key in enumerate(s["foes"]):
        x, z = Z.world(torch.tensor((i - 1) * float(c["foe_gap_m"])), torch.tensor(s["dist"]), zx, zz, zb)
        foes.append({"key": key, "x": float(x), "z": float(z), "b": (zb + 180.0) % 360})
    return {"attacker": 2, "sides": {1: {"faction": DH.FACTIONS[s["ours"]], "units": mine},
                                     2: {"faction": DH.FACTIONS[s["theirs"]], "units": foes}}}


def build(sp, P, device, check=True):
    """(st, parcel [B groups], order [B, 7], own [B, N], shooter [B, N], foe [B, N])."""
    st = scenario.build([army(s) for s in sp], P, device=device)
    B, N = st.B, st.N
    assert N == 2 * H, N
    parcels = []
    for b, s in enumerate(sp):
        x, z, h = zone_of(s)
        parcels.append(PC.Parcel.make(b, list(range(H)), list(range(H, 2 * H)), x, z, h, line=False, group=0, n=N,
                                      device=device))
    parcel = PC.Parcel.cat(parcels)
    own = torch.zeros((B, N), dtype=torch.bool, device=device)
    own[:, :H] = True
    shooter = st.u["range"] > 0
    if check:
        raw = TR.Seer(P).raw(st)
        PC.ensure(raw, parcel)
    return st, parcel, order_vec(B, "hold", "any", device), own, shooter & own, ~own


class Script:
    """The enemy's orders: replay.nearest_attack at a walk, looked at again every look_s; between looks the orders stand."""

    def __init__(self, look_s=None):
        self.look = float(cfg()["drill1"]["look_s"] if look_s is None else look_s)
        self.next = -1.0
        self.o = None

    def __call__(self, st):
        t = float(st.t.max())
        if self.o is None or t >= self.next - 1e-6:
            self.o = replay.nearest_attack(st, run=False)
            self.next = t + self.look
        return self.o


class Ender:
    """The drill's end, after every simulator step: one side has nothing standing, fight_s after the first contact (the
    trade is read by then; a fight's end gives no more to this drill), or the limit."""

    def __init__(self, st, own, limit_s=None):
        self.over = torch.zeros(st.B, dtype=torch.bool, device=st.device)
        self.why = torch.zeros(st.B, dtype=torch.long, device=st.device)      # 1 we are out, 2 the enemy is out, 3 time
        self.limit = float(cfg()["drill1"]["limit_s"] if limit_s is None else limit_s)
        self.fight = float(cfg()["drill1"]["fight_s"])
        self.own = own

    def after_step(self, st, t_contact=None):
        u = st.u
        out = u["r"] | (u["men"] <= 0) | u["gone"]
        we = (out | ~self.own).all(1)
        they = (out | self.own).all(1)
        late = st.t >= self.limit
        if t_contact is not None:
            late = late | ((t_contact >= 0) & (st.t >= t_contact + self.fight))
        new = ~self.over & (we | they | late)
        self.why = torch.where(new, torch.where(we, 1, torch.where(they, 2, 3)), self.why)
        self.over = self.over | new
        st.winner = torch.where(new, torch.where(we, 2, torch.where(they, 1, 0)), st.winner)
        st.done = self.over.clone()
        return new


def crosses(ax, az, bx, bz, x, z, b, front, depth):
    """[...] bool: the segment a -> b crosses the turned rectangle (slabs in its axes; all broadcast)."""
    sa, fa = Z.local(ax, az, x, z, b)
    sb, fb = Z.local(bx, bz, x, z, b)
    t0, t1 = torch.zeros_like(sa), torch.ones_like(sa)
    for p0, p1, half in ((sa, sb, front / 2), (fa, fb, depth / 2)):
        d = p1 - p0
        par = d.abs() < 1e-6
        lo = (-half - p0) / torch.where(par, torch.ones_like(d), d)
        hi = (half - p0) / torch.where(par, torch.ones_like(d), d)
        a, c = torch.minimum(lo, hi), torch.maximum(lo, hi)
        t0 = torch.where(par, t0, torch.maximum(t0, a))
        t1 = torch.where(par, torch.where(p0.abs() <= half, t1, torch.full_like(t1, -1.0)), torch.minimum(t1, c))
    return t0 <= t1


class Watch:
    """The contact's readings of a batch: call step() after every simulator step, with the battles that were over."""

    def __init__(self, st, own, shooter, spacing, device):
        B = st.B
        self.own, self.shooter, self.foe = own, shooter, ~own
        self.spacing = spacing
        z = lambda: torch.zeros(B, dtype=torch.bool, device=device)                # noqa: E731
        self.contact, self.between, self.hit, self.melee = z(), z(), z(), z()
        self.fired_s = torch.zeros(B, device=device)
        self.t_contact = torch.full((B,), -1.0, device=device)
        self.fill = torch.full((B,), float("nan"), device=device)
        self.soon = float(cfg()["reward"]["soon_s"])
        self.margin = float(cfg()["reward"]["between_margin_m"])
        self.dt = None

    def _between(self, u, touch, idx):
        """[n] bool for the battles idx [n]: every touching enemy's line to the shooters crosses an infantry formation."""
        front, depth = geometry.dims(u, self.spacing)
        N = u["x"].shape[1]
        bb, sh = idx, self.shooter[idx].float().argmax(1)
        sx, sz = u["x"][bb, sh], u["z"][bb, sh]
        cov = torch.ones_like(idx, dtype=torch.bool)
        for j in range(N):
            eng = (touch[bb, :, j] & self.own[bb]).any(1) & self.foe[bb, j]
            ok = torch.zeros_like(eng)
            for i in range(N):
                inf = self.own[bb, i] & ~self.shooter[bb, i] & (u["men"][bb, i] > 0) & ~u["r"][bb, i]
                c = crosses(u["x"][bb, j], u["z"][bb, j], sx, sz, u["x"][bb, i], u["z"][bb, i], u["b"][bb, i],
                            front[bb, i] + 2 * self.margin, depth[bb, i] + 2 * self.margin)
                ok = ok | (inf & c)
            cov = cov & (~eng | ok)
        return cov

    def step(self, st, was_done, dt, area=None):
        """(new_contact [B] bool, new between [B] bool, new hit [B] bool) this step."""
        u = st.u
        live = ~was_done
        B, N = st.B, st.N
        hs = u["hit_sec"].view(B, N, N)
        touch = ((hs > 0) | (hs.transpose(1, 2) > 0)) & self.own[:, :, None] & self.foe[:, None, :]
        any_touch = touch.flatten(1).any(1) & live
        on_shooter = (touch & self.shooter[:, :, None]).flatten(1).any(1) & live
        new = any_touch & ~self.contact
        self.t_contact = torch.where(new, st.t.float(), self.t_contact)
        self.contact = self.contact | new
        got = torch.zeros_like(new)
        if bool(new.any()):
            idx = new.nonzero()[:, 0]
            got[idx] = self._between(u, touch, idx) & ~on_shooter[idx] & (u["men"][idx] * self.shooter[idx]).sum(1).gt(0)
            self.between = self.between | got
            if area is not None:
                # the infantry's front to its frame's width at the contact (information: does a unit fill a wide zone)
                front, _ = geometry.dims(u, self.spacing)
                inf = self.own & ~self.shooter
                r = (front / area["front"].clamp(min=1.0) * inf).sum(1) / inf.sum(1).clamp(min=1)
                self.fill = torch.where(new, r, self.fill)
        soon = self.contact & (st.t.float() <= self.t_contact + self.soon)
        hit = on_shooter & soon & ~self.hit
        self.hit = self.hit | hit
        self.melee = self.melee | on_shooter
        self.fired_s = self.fired_s + (u["fire"] & self.shooter).any(1).float() * dt * live.float()
        return new, got, hit
