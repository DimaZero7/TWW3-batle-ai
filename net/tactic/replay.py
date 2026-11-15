"""The branch test by whole-batch replays (METHOD 'парная игра'; tmp/tactic_slice/review_attack_v22.md C.9): a wrapper
player in the place of a player of tactic/pin.py. It plays the base player unchanged until a battle's fork tick, then
holds a branch for one named unit i of that battle (and its enemy e) for window_s, while every other unit goes on under
the base. Blows are deterministic (sim.json noise.blows off) and battles of a batch independent, so a replay under the
same torch seed is the same battle up to its first differing order: a battle without a state, and every battle before
its fork tick, plays as under the base (the runner asserts it).

    Branch(base, plan, window_s, masks=True)   plan {b (the Tacticians' row = the batch's battle): state}; a state:
        t      the fork tick, s (a tick second)
        i, e   my unit's and the enemy's battle slots
        u      cause (b): the slot of my unit in melee e came at (-1 none)
        kind   keep - no order to i; now - 'attack e' at t; later<k> - at t + k ticks if the cause still stands then
               (cond); at - at the tick 'at' (s), unconditioned; pin - once e is pinned (pin.pinned) on two ticks after t;
               other - at t the nearest open valid enemy within orders.reach_m that is not e ('nearest'), or the nearest
               to i of those pinned by u (other='by_u'; none: no order)
        cond   later<k>'s cause: 'pinned' (pin.pinned) or 'second' (pin.second of the base's trigger)
        cause  the tag written on the order (pin.Teacher.last_tags: 1 cause (a), 2 cause (b), 3 other)
    With masks, from t to t + window_s: unit i is closed to every order of the base and e to every other striker
    (pin.Teacher.shut); without (the near-miss branches), only the order. self.log {b: {sec, target}} the orders put in.
    No plan: the base as it is.
"""
import torch

from net.tactic import orders as OR
from net.tactic import pin as PN
from net.unit import scripts as SC


class Branch:
    def __init__(self, base, plan, window_s=300.0, masks=True, other="nearest"):
        self.base = PN.player(base) if isinstance(base, str) else base
        self.name = getattr(self.base, "name", "branch")
        self.allow = getattr(self.base, "allow", (SC.HOLD, SC.ATTACK))
        self.plan, self.window, self.masks, self.other = dict(plan), float(window_s), masks, other
        self.log = {}
        self.prev_pin = {}                 # (pin: e pinned at the battle's last tick after t)
        self.done = set()
        self._cache = {}

    @property
    def last_tags(self):
        return self.base.last_tags

    def _c(self, k, fx):
        """pin.pinned / the base's pin.second of this tick's facts, computed once a tick."""
        if self._cache.get("fx") is not fx:
            self._cache = {"fx": fx}
        if k not in self._cache:
            sb = self.base
            self._cache[k] = (PN.pinned(fx, touch=getattr(sb, "touch", None)) if k == "pin" else
                              PN.second(fx, sb.tight, float(sb.second_m) if sb.nearest else None)[0])
        return self._cache[k]

    def _due(self, b, st, now, fx, r, krow):
        """The target slot to give i at this tick (-1 none)."""
        kind, t = st["kind"], int(st["t"])
        if b in self.done or now < t:
            return -1
        e = int(st["e"])
        if kind == "keep":
            return -1
        if kind == "now":
            return e if now == t else -1
        if kind == "at":
            return e if now == int(st["at"]) else -1
        if kind.startswith("later"):
            if now != t + 5 * int(kind[5:]):
                return -1
            if krow < 0:
                return -1
            if st["cond"] == "pinned":
                ok = bool(self._c("pin", fx)[r, krow])
            else:
                ok = bool(self._c("second", fx)[r, krow])
            return e if ok else -1
        if kind == "pin":
            if now <= t or now >= t + self.window:
                return -1
            p = krow >= 0 and bool(self._c("pin", fx)[r, krow])
            before = self.prev_pin.get(b, False)
            self.prev_pin[b] = p
            return e if (p and before) else -1
        if kind == "other":
            if now != t:
                return -1
            return -2                       # (chosen in __call__: needs i's row and the walk)
        raise ValueError(kind)

    def _other(self, fx, r, g, st):
        """The 'other target' slot for my unit g of row r (-1 none)."""
        e = int(st["e"])
        reach = float(OR.c_orders()["reach_m"])
        ok = fx.valid[r] & OR.Walk(fx).open(g)[r] & (fx.slot[r] != e) & (fx.g_dist[r, g] <= reach)
        if self.other == "by_u":
            ku = fx.row_of_slot(torch.full((fx.R, 1), int(st["u"]), dtype=torch.long, device=fx.s.device))[r, 0]
            if ku < 0:
                return -1
            ok = ok & self._c("pin", fx)[r] & (fx.gap[r, ku] <= SC.TOUCH_M)
        if not bool(ok.any()):
            return -1
        d = torch.where(ok, fx.g_dist[r, g], torch.full_like(fx.g_dist[r, g], OR.INF))
        return int(fx.slot[r, int(d.argmin())])

    def __call__(self, v):
        fx = v.fx
        now = int(round(float(v.now)))
        rows = v.idx.tolist()
        S = int(max(int(fx.slot.max()), int(fx.my.max()), 0)) + 1
        for st in self.plan.values():
            S = max(S, int(st["i"]) + 1, int(st["e"]) + 1)
        su = torch.zeros((len(rows), S), dtype=torch.bool, device=fx.s.device)
        se = torch.zeros_like(su)
        give = []
        es = torch.tensor([int(self.plan[b]["e"]) if b in self.plan else -1 for b in rows], dtype=torch.long,
                          device=fx.s.device)
        erow = fx.row_of_slot(es[:, None])[:, 0].tolist()
        for r, b in enumerate(rows):
            st = self.plan.get(b)
            if st is None:
                continue
            t = int(st["t"])
            if self.masks and t <= now < t + self.window:
                su[r, int(st["i"])] = True
                se[r, int(st["e"])] = True
            krow = int(erow[r])
            tg = self._due(b, st, now, fx, r, krow)
            if tg != -1:
                give.append((r, b, st, tg))
        self.base.shut = {"units": su, "targets": se} if self.masks else None
        out = self.base(v)
        self.base.shut = None
        for r, b, st, tg in give:
            hit = (fx.my[r] == int(st["i"])).nonzero()
            if hit.numel() == 0:
                self.done.add(b)
                self.log[b] = {"sec": now, "target": -1, "why": "no unit"}
                continue
            g = int(hit[0, 0])
            if tg == -2:
                tg = self._other(fx, r, g, st)
                if tg < 0:
                    self.done.add(b)
                    self.log[b] = {"sec": now, "target": -1, "why": "no other target"}
                    continue
            out["skill"][r, g] = SC.ATTACK
            out["target"][r, g] = tg
            tags = getattr(self.base, "last_tags", None)          # (a base without tags: tactic/lanes' C)
            if tags is not None:
                tags["cause"][r, g] = int(st.get("cause", 3))
                tags["u"][r, g] = int(st.get("u", -1)) if int(st.get("cause", 3)) == 2 else -1
            self.done.add(b)
            self.log[b] = {"sec": now, "target": tg, "melee": bool(v.melee[r, g])}
        return out
