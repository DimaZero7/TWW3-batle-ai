"""Drill T0-2 'pin and strike' of the tactic level (tmp/tactic_slice/design_flank_v2.md; stage0_results.md binding):
the facts of the teacher's rule and the players - each a callable in the place of teacher.drive in lower.Tacticians,
returning the level's output for a tick (tactic/orders.py: module doc). They read the level's input only (orders.Facts:
the rows and the orders in force); the code's masks (orders.masks) stand over all of them.

    pinned(fx)   [R, K] valid(e), in melee, its speed under scripts.APPROACH_MS, and a standing unit of mine in melee
                 whose formation is within scripts.TOUCH_M of e's
    K            keep everything; a unit that must change (no frame, or the code's causes) holds at the open cell nearest it
    T            the teacher (design section 2): in slot order - in melee: keep; forced: attack the nearest e of cand(i)
                 if any, else hold at the open cell nearest it; free(i) with cand(i): attack the nearest e of cand(i);
                 else keep. cand(i) = {e pinned, open for i, centre within orders.reach_m, i the nearest free unit of its
                 side class to e not yet given an order in this walk}. No memory
    D            dumb move 2: a free unit attacks the nearest valid open enemy within reach, pinned or not (D35: an enemy
                 within 35 m edge to edge - the counter's reach, scripts.CHARGE_M + run x LEAD_S)
    A1           the pilot's player (tmp/tactic_slice/pilot/attack_players.py) as a level output: an enemy touched in
                 melee by a standing unit of mine at this tick and the one before (its memory) gets up to two strikers,
                 one a side (the side by where the unit stands across the enemy's front), each the nearest free unit of
                 that side within reach (free: standing, not in melee, no task); the task ends when the target routs,
                 dies or is unseen, or the unit routs: hold in the frame in force (its deployment frame)
    P4           everyone under 'counter' from the first tick (a reference)
    T_recall     T, and a striker out of melee farther than 30 m from its target (edge to edge) whose target has no other
                 unit of mine within scripts.STICK_M gets 'hold' (a reference; design section 6)
    T_nohyg      T without its melee row; run with lower.Tacticians(hygiene=False) (a reference)
    TC           T, and my units under 'hold' (no target) get 'counter' (the unit script's 'meet': a charge on the enemy
                 coming at the unit within ~35 m) - a reference; P4 is the same without T's attacks (also named C)
    T_wait2, T_nonear, T_nospeed, T_sign   T with one of its four differences from A1 turned to A1's rule (Teacher
                 doc: e pinned on two ticks; free(i) without 'no enemy within STICK_M'; pinned without the speed rule;
                 the nearest free unit by the side's sign); T_a1rules - all four
"""
import torch

from net.tactic import orders as OR
from net.unit import scripts as SC

D35_M = SC.CHARGE_M + 5.0            # D35: 'meet at ~35 m' (scripts.py: the counter's reach, ~35 m edge to edge)
RECALL_M = 30.0                      # T_recall: farther than this from its target (design section 6, a setting)


def pinned(fx, speed=True):
    """[R, K] (module doc); speed False: without the speed rule on e (A1's)."""
    hold = fx.mine & ~fx.r & fx.m                                                   # [R, K] my units in melee
    touch = hold[:, :, None] & (fx.gap <= SC.TOUCH_M)                               # [R, a, e]
    slow = (fx.speed < SC.APPROACH_MS) if speed else torch.ones_like(fx.valid)
    return fx.valid & fx.m & slow & touch.any(1)


def _out(v, fr, skill=None, tgt=None, side=None):
    o = v.order
    return {"skill": o["skill"].clone() if skill is None else skill, "target": o["tgt"].clone() if tgt is None else tgt,
            "side": o["side"].clone() if side is None else side, **fr}


def _nearest(score, ok):
    """([R] row, [R] any): the row of the least score [R, K] where ok."""
    s = torch.where(ok, score, torch.full_like(score, OR.INF))
    return s.argmin(1), ok.any(1)


class Keep:
    name = "K"

    def __call__(self, v):
        return _out(v, v.frames())


class Teacher:
    name = "T"

    def __init__(self, melee_row=True, recall=False, dumb=None, counter=False, wait2=False, near=True, speed=True,
                 sign=False):
        # dumb: None (T), "reach" (D), "d35" (D35); counter: my units under 'hold' get 'counter' (TC). The four rules
        # where T differs from A1 (the ablation; T's own rule first): wait2 - e pinned at this tick and the one before
        # (A1's memory: by the Tacticians' row and the battle's slot); near False - free(i) without 'no standing enemy
        # within STICK_M'; speed False - pinned without the speed rule; sign - the nearest free unit by the sign of its
        # offset across e's facing line (A1's), not by the side class (axis / left / right)
        self.melee_row, self.recall, self.dumb, self.counter = melee_row, recall, dumb, counter
        self.wait2, self.near, self.speed, self.sign = wait2, near, speed, sign
        self.prev_pin = None
        if counter:
            self.allow = (SC.HOLD, SC.ATTACK, SC.COUNTER)

    def _pinned(self, v):
        fx = v.fx
        pin = pinned(fx, self.speed)
        if not self.wait2:
            return pin
        N = int(fx.slot.max()) + 1 if fx.slot.numel() else 1
        n = int(v.idx.max()) + 1
        if self.prev_pin is None or self.prev_pin.shape[0] < n or self.prev_pin.shape[1] < N:
            t = torch.zeros((max(n, 1), max(N, 64)), dtype=torch.bool, device=d_dev(fx))
            if self.prev_pin is not None:
                t[:self.prev_pin.shape[0], :self.prev_pin.shape[1]] = self.prev_pin
            self.prev_pin = t
        now_s = torch.zeros((fx.R, self.prev_pin.shape[1]), dtype=torch.bool, device=d_dev(fx))
        now_s.scatter_(1, fx.slot.clamp(min=0), pin & (fx.slot >= 0))
        before = self.prev_pin[v.idx].gather(1, fx.slot.clamp(min=0)) & (fx.slot >= 0)
        self.prev_pin[v.idx] = now_s
        return pin & before

    def _free(self, fx):
        if self.near:
            return fx.free
        idle = (fx.order["skill"] == SC.HOLD) | (fx.order["skill"] == SC.COUNTER)
        return fx.able & ~fx.g_m & (idle | (fx.attack & ~fx.t_valid))

    def _side(self, fx):
        """[R, G, K] the side my units are grouped by for 'the nearest free unit of its side'."""
        if not self.sign:
            return fx.g_side
        th = torch.deg2rad(fx.b)[:, None, :]
        gs, gf = fx.s.gather(1, fx.g_row), fx.f.gather(1, fx.g_row)
        off = (gs[:, :, None] - fx.s[:, None, :]) * torch.cos(th) - (gf[:, :, None] - fx.f[:, None, :]) * torch.sin(th)
        return torch.where(off >= 0, 1, -1)

    def __call__(self, v):
        fx = v.fx
        fr = v.frames()
        G = fx.G
        reach = float(OR.c_orders()["reach_m"])
        pin = self._pinned(v)
        free = self._free(fx)
        w = OR.Walk(fx)
        skill, tgt, side = v.order["skill"].clone(), v.order["tgt"].clone(), v.order["side"].clone()
        melee = v.melee if self.melee_row else torch.zeros_like(v.melee)
        elig = fx.able & ~melee & (free | v.must)                       # may be given an order in this walk
        if self.recall:
            k = fx.t_row.clamp(min=0)
            gap_t = fx.g_gap.gather(2, k[..., None])[..., 0]
            others = fx.mine[:, None, :] & ~fx.r[:, None, :] & (fx.gap.gather(1, k[..., None].expand(-1, -1, fx.K))
                                                              <= SC.STICK_M)       # [R, G, a]
            mine_g = torch.zeros_like(others)
            mine_g.scatter_(2, fx.g_row[..., None], True)
            lone = ~(others & ~mine_g).any(2)
            rec = fx.striking & ~fx.g_m & (gap_t > RECALL_M) & lone
            self.n_recall = getattr(self, "n_recall", 0) + int(rec.sum())
        for g in range(G):
            a = fx.able[:, g] & ~melee[:, g]
            if not bool(a.any()):
                continue
            if self.recall:
                w.release(g, rec[:, g])
                skill[:, g] = torch.where(rec[:, g], torch.full_like(skill[:, g], SC.HOLD), skill[:, g])
                tgt[:, g] = torch.where(rec[:, g], torch.full_like(tgt[:, g], -1), tgt[:, g])
            go = a & (v.must[:, g] | free[:, g] | (self.recall and rec[:, g]))
            op = w.open(g)
            if self.dumb is None:
                sd = self._side(fx)                                                  # [R, G, K]
                same = sd == sd[:, g:g + 1, :]
                d = fx.g_dist + torch.arange(G, device=d_dev(fx))[None, :, None] * 1e-4      # (ties by slot)
                best = torch.where(same & elig[:, :, None], d, torch.full_like(d, OR.INF)).amin(1)
                cand = pin & op & (fx.g_dist[:, g] <= reach) & (d[:, g] <= best) & elig[:, g:g + 1]
            elif self.dumb == "reach":
                cand = op & (fx.g_dist[:, g] <= reach)
            else:
                cand = op & (fx.g_gap[:, g] <= D35_M)
            k, has = _nearest(fx.g_dist[:, g], cand)
            att = go & has
            w.give(g, k, att)
            elig[:, g] = elig[:, g] & ~att
            skill[:, g] = torch.where(att, torch.full_like(skill[:, g], SC.ATTACK), skill[:, g])
            tgt[:, g] = torch.where(att, fx.slot.gather(1, k[:, None])[:, 0], tgt[:, g])
            side[:, g] = torch.where(att, w.side[:, g], side[:, g])
            hold = a & v.must[:, g] & ~has                                         # forced, no candidate: hold
            skill[:, g] = torch.where(hold, torch.full_like(skill[:, g], SC.HOLD), skill[:, g])
            tgt[:, g] = torch.where(hold, torch.full_like(tgt[:, g], -1), tgt[:, g])
        if self.counter:
            idle = fx.able & (skill == SC.HOLD) & (tgt < 0)
            skill = torch.where(idle, torch.full_like(skill, SC.COUNTER), skill)
        return _out(v, fr, skill, tgt, side)


def d_dev(fx):
    return fx.s.device


class Attack1:
    """A1 through the level (module doc); its memory by the Tacticians' row: the enemies touched at the last tick (by
    the battle's slot) and the sign of each of its tasks (by my slot index)."""
    name = "A1"

    def __init__(self):
        self.touched, self.sign = None, None

    def _mem(self, v, N):
        n = int(v.idx.max()) + 1
        dev = d_dev(v.fx)
        if self.touched is None or self.touched.shape[0] < n or self.touched.shape[1] < N:
            t = torch.zeros((max(n, 1), max(N, 1)), dtype=torch.bool, device=dev)
            sg = torch.zeros((max(n, 1), 64), dtype=torch.long, device=dev)
            if self.touched is not None:
                t[:self.touched.shape[0], :self.touched.shape[1]] = self.touched
                sg[:self.sign.shape[0]] = self.sign
            self.touched, self.sign = t, sg

    def __call__(self, v):
        fx = v.fx
        fr = v.frames()
        R, G = fx.R, fx.G
        dev = d_dev(fx)
        reach = float(OR.c_orders()["reach_m"])
        hold = fx.mine & ~fx.r & fx.m
        foe = fx.enemy & ~fx.r & ~fx.lord                                           # A1's foe (no speed rule)
        touch = foe & fx.m & (hold[:, :, None] & (fx.gap <= SC.TOUCH_M)).any(1)      # [R, K]
        N = int(fx.slot.max()) + 1 if fx.slot.numel() else 1
        self._mem(v, N)
        now_s = torch.zeros((R, N), dtype=torch.bool, device=dev)
        now_s.scatter_(1, fx.slot.clamp(min=0), touch & (fx.slot >= 0))
        ready_s = now_s & self.touched[v.idx, :N]
        self.touched[v.idx, :N] = now_s
        sign_mem = self.sign[v.idx, :G]
        skill, tgt, side = v.order["skill"].clone(), v.order["tgt"].clone(), v.order["side"].clone()
        task = (skill == SC.ATTACK) & (tgt >= 0)
        # tasks end: the target gone (no standing seen row) or the unit routing - hold in its frame
        k = fx.t_row.clamp(min=0)
        t_ok = (fx.t_row >= 0) & foe.gather(1, k)
        end = task & (fx.my >= 0) & (~t_ok | ~fx.able)
        skill = torch.where(end, torch.full_like(skill, SC.HOLD), skill)
        tgt = torch.where(end, torch.full_like(tgt, -1), tgt)
        task = task & ~end
        free = fx.able & ~fx.g_m & ~task
        w = OR.Walk(fx, skill=skill.clone(), t_row=torch.where(task, fx.t_row, torch.full_like(tgt, -1)),
                    side=side.clone())
        ar = torch.arange(R, device=dev)
        tie = torch.arange(G, device=dev)[None] * 1e-4
        gs, gf = fx.s.gather(1, fx.g_row), fx.f.gather(1, fx.g_row)
        for e in range(N):
            ke = fx.row_of_slot(torch.full((R, 1), e, device=dev, dtype=torch.long))[:, 0]
            ready = ready_s[:, e] & (ke >= 0)
            if not bool(ready.any()):
                continue
            kc = ke.clamp(min=0)
            th = torch.deg2rad(fx.b.gather(1, kc[:, None]))                                  # [R, 1]
            off = (gs - fx.s.gather(1, kc[:, None])) * torch.cos(th) - (gf - fx.f.gather(1, kc[:, None])) * torch.sin(th)
            sign = torch.where(off >= 0, 1, -1)
            dist = fx.g_dist.gather(2, kc[:, None, None].expand(R, G, 1))[..., 0]          # [R, G]
            for sg in (1, -1):
                taken = (task & (tgt == e) & (sign_mem == sg)).any(1)
                cand = free & (sign == sg) & (ready & ~taken)[:, None]
                d = torch.where(cand, dist + tie, torch.full_like(dist, OR.INF))
                g = d.argmin(1)
                ok = cand.any(1) & (dist[ar, g] <= reach)
                for gg in g[ok].unique().tolist():
                    sel = ok & (g == gg) & w.open(gg).gather(1, kc[:, None])[:, 0]
                    if not bool(sel.any()):
                        continue
                    w.give(gg, kc, sel)
                    skill[:, gg] = torch.where(sel, torch.full_like(skill[:, gg], SC.ATTACK), skill[:, gg])
                    tgt[:, gg] = torch.where(sel, torch.full_like(tgt[:, gg], e), tgt[:, gg])
                    side[:, gg] = torch.where(sel, w.side[:, gg], side[:, gg])
                    free[:, gg] = free[:, gg] & ~sel
                    task[:, gg] = task[:, gg] | sel
                    sign_mem[:, gg] = torch.where(sel, torch.full_like(sign_mem[:, gg], sg), sign_mem[:, gg])
        self.sign[v.idx, :G] = sign_mem
        return _out(v, fr, skill, tgt, side)


class Counter:
    name = "P4"
    allow = (SC.HOLD, SC.ATTACK, SC.COUNTER)

    def __call__(self, v):
        fr = v.frames()
        skill = torch.where(v.fx.able & (v.order["skill"] == SC.HOLD) & (v.order["tgt"] < 0),
                            torch.full_like(v.order["skill"], SC.COUNTER), v.order["skill"])
        return _out(v, fr, skill)


PLAYERS = {"K": Keep, "T": Teacher, "D": lambda: Teacher(dumb="reach"), "D35": lambda: Teacher(dumb="d35"),
           "A1": Attack1, "P4": Counter, "T_recall": lambda: Teacher(recall=True),
           "T_nohyg": lambda: Teacher(melee_row=False), "TC": lambda: Teacher(counter=True), "C": Counter,
           "T_wait2": lambda: Teacher(wait2=True), "T_nonear": lambda: Teacher(near=False),
           "T_nospeed": lambda: Teacher(speed=False), "T_sign": lambda: Teacher(sign=True),
           "T_a1rules": lambda: Teacher(wait2=True, near=False, speed=False, sign=True)}
HYGIENE_OFF = ("T_nohyg",)


def player(name):
    p = PLAYERS[name]()
    p.name = name
    return p
