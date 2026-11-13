"""The readings of a stage 0 battle, every second (matchup/duel.play's watch), and the measures out of them.

The stage's measures (each battle; my side is side 1, the lords left out of everything):
    trade         the enemy's gold lost less mine, over the mean of the two armies' gold (reward.gold_lost: the worst
                  share lost so far, kept here every second; the lords out of both)
    win           1 my side won, -1 lost, 0 none by the limit
    t_contact     the first second of melee (the limit: none)
    flank_hits    my contacts that began on an enemy's flank or rear (sim hit_sec 2 / 3: the side of the touched
                  unit the contact began on), per battle; flank_taken - the enemy's on mine
    flank_dmg     the share of the HP my units took off the enemy in melee from the flank / rear: each second each
                  enemy's HP lost x the share of my units touching it from its flank / rear
    gap_m         the widest gap between the edges of my line's formations (the units within line_depth_m of the
                  frontmost, towards the enemy's centre; the formation's front by geometry.dims) at the battle's first
                  contact second (NaN: fewer than 2 in the line)
    dh            my unit's height less the touched enemy's at my unit's first contact, the mean of my units (NaN: none)
    head_on       my units' first contacts with the touched enemy's centre in their front sector (contact.front_deg),
                  a share of my units with a contact (U2's judge, net/unit/u2.py)
    reserve_s     from my first rout (not the lord) to the first melee of the reserve (the named one - a drill's, a
                  behaviour's - else my units free at that second: not in melee, not routing), capped at
                  reserve_cap_s; NaN: no rout of mine
    orders_min    my units' order changes (the chain's intended order kind other than the last, not KEEP) per unit a
                  minute alive
    game_orders_min   with the order gate (battle.play gate): my units' new orders to the game per unit a minute
                  (net/common/reissue.py; raw_orders_min - the intended order's changes by the same rule, no gate)
    frame_exec    my units' seconds standing out of melee (not routing) with the formation in their frame and facing by
                  it, over all such seconds (U2's execution: net/unit/u2.in_frame)
    enemy_rout_s  the first rout of an enemy unit since the first contact (NaN: none)
The drills' own:
    pin_flank     T0-2: a contact on an enemy's flank / rear while another of my units already touches it, within
                  flank_window_s of the first contact
    trade_after   T0-2: the trade trade_after_s after the first contact (the end's when the battle ended sooner)
    reserve_early O0-3: the reserve in melee before early_s after the first contact and before my first rout
"""
import math

import torch

from sim import geometry, terrain
from sim import reward


def _gold_trade(u, lord):
    g = reward.gold_lost(u)
    side = u["side"]
    keep = ~lord & (side > 0)
    c = u["cost"] * keep
    lost = [((g * keep) * (side == s)).sum(1) for s in (1, 2)]
    budget = torch.stack([(c * (side == s)).sum(1) for s in (1, 2)], 1).mean(1).clamp(min=1.0)
    return (lost[1] - lost[0]) / budget


class Watch:
    """watch(sec, st) of matchup/duel.play (the state is made there: the readings start at the first call)."""

    def __init__(self, P, cfg_general, chain=None, reserve=None, window_s=30.0, after_s=90.0, early_s=60.0):
        self.P, self.cfg_general, self.chain, self.reserve0 = P, cfg_general, chain, reserve
        self.window_s, self.after_s, self.early_s = window_s, after_s, early_s
        self.st = None

    def _init(self, st):
        P, cfg_general, chain, reserve = self.P, self.cfg_general, self.chain, self.reserve0
        window_s, after_s, early_s = self.window_s, self.after_s, self.early_s
        self.st = st
        u = st.u
        B, N = st.B, st.N
        dev = st.device
        self.B, self.N, self.dev = B, N, dev
        self.chain = chain
        self.spacing = P.sim["formation"]["spacing_m"]
        self.line_depth = float(cfg_general["line_depth_m"])
        self.cap = float(cfg_general["reserve_cap_s"])
        self.window, self.after, self.early = float(window_s), float(after_s), float(early_s)
        side = u["side"]
        self.lord = u["lord"].clone()
        self.x0, self.z0 = u["x"].clone(), u["z"].clone()
        self.mine = (side == 1) & ~self.lord
        self.foe = (side == 2) & ~self.lord
        self.named = reserve if reserve is not None else torch.zeros((B, N), dtype=torch.bool, device=dev)
        self.has_named = self.named.any(1)
        f = lambda v=float("nan"): torch.full((B,), v, device=dev)              # noqa: E731
        self.t_contact = f()
        self.flank_hits, self.flank_taken = f(0.0), f(0.0)
        self.dmg, self.dmg_flank = f(0.0), f(0.0)
        self.gap = f()
        self.dh_sum, self.dh_n = f(0.0), f(0.0)
        self.ho_sum, self.ho_n = f(0.0), f(0.0)
        self.t_rout_own, self.t_rout_foe = f(), f()
        self.reserve = torch.zeros((B, N), dtype=torch.bool, device=dev)
        self.t_reserve = f()
        self.t_named = f()                    # the named reserve's first melee (any time)
        self.pin_flank = f(0.0)
        self.trade_after = f()
        self.changes, self.alive_s = f(0.0), f(0.0)
        self.exec_ok, self.exec_n = f(0.0), f(0.0)
        self.touch = torch.zeros((B, N, N), dtype=torch.bool, device=dev)
        self.first = torch.zeros((B, N), dtype=torch.bool, device=dev)
        self.hp = u["hp_abs"].clone()
        self.last_kind = None
        self.t_end = f(-1.0)

    def __call__(self, sec, st):
        if self.st is None:
            self._init(st)
        u = st.u
        B, N = self.B, self.N
        t = torch.full((B,), float(sec + 1), device=self.dev)
        live = ~st.done
        just = st.done & (self.t_end < 0)
        self.t_end = torch.where(just, st.t.float(), self.t_end)
        if "lost_worst" in u:
            u["lost_worst"] = reward.track(u)
        out = u["r"] | (u["men"] <= 0) | u["gone"]
        hs = u["hit_sec"].view(B, N, N)
        touch = hs > 0
        new = touch & ~self.touch
        mine, foe = self.mine, self.foe
        fl = (hs >= 2) & new
        on_foe = new & foe[:, :, None] & mine[:, None, :]           # [b, i enemy, j mine]
        self.flank_hits += (fl & on_foe).flatten(1).sum(1).float() * live
        self.flank_taken += (fl & mine[:, :, None] & foe[:, None, :]).flatten(1).sum(1).float() * live
        # pinned: the enemy i already touched by another of mine before this second
        before = (self.touch & foe[:, :, None] & mine[:, None, :]).any(2)
        tc = self.t_contact
        in_win = torch.isfinite(tc) & (t - tc <= self.window)
        pin = (fl & on_foe).any(2) & before
        self.pin_flank = torch.where(live & in_win & pin.any(1), torch.ones_like(self.pin_flank), self.pin_flank)
        # the first contact of the battle: the gap in my line
        any_touch = (touch & (mine[:, :, None] | foe[:, :, None])).flatten(1).any(1)
        first_c = any_touch & torch.isnan(self.t_contact) & live
        self.t_contact = torch.where(first_c, t, self.t_contact)
        if bool(first_c.any()):
            self.gap = torch.where(first_c, self._gap(st, out), self.gap)
        # my units' first contact: the height against the enemy touched
        my_touch = (touch & mine[:, :, None] & foe[:, None, :])
        now_first = my_touch.any(2) & ~self.first & live[:, None]
        if bool(now_first.any()):
            j = my_touch.float().argmax(2)
            h = terrain.height(u, u["x"], u["z"])
            dh = h - h.gather(1, j)
            self.dh_sum += (dh * now_first).sum(1)
            self.dh_n += now_first.float().sum(1)
            dx, dz = u["x"].gather(1, j) - u["x"], u["z"].gather(1, j) - u["z"]
            rel = torch.remainder(torch.atan2(dx, dz) - torch.deg2rad(u["b"]) + math.pi, 2 * math.pi) - math.pi
            front = rel.abs() <= math.radians(self.P.sim["contact"]["front_deg"])
            self.ho_sum += (front & now_first).float().sum(1)
            self.ho_n += now_first.float().sum(1)
            self.first = self.first | now_first
        # the melee damage from the flank / rear
        d_hp = (self.hp - u["hp_abs"]).clamp(min=0)
        tf = touch & foe[:, :, None] & mine[:, None, :]
        n_t = tf.float().sum(2)
        n_f = (tf & (hs >= 2)).float().sum(2)
        hit = (n_t > 0) & foe
        self.dmg += (d_hp * hit).sum(1) * live
        self.dmg_flank += (d_hp * torch.where(hit, n_f / n_t.clamp(min=1), torch.zeros_like(n_t))).sum(1) * live
        self.hp = u["hp_abs"].clone()
        # the first routs
        r_own = (out & mine).any(1) & torch.isnan(self.t_rout_own) & live
        if bool(r_own.any()):
            free = mine & ~out & ~u["m"]
            res = torch.where(self.has_named[:, None], self.named, free)
            self.reserve = torch.where(r_own[:, None], res, self.reserve)
            self.t_rout_own = torch.where(r_own, t, self.t_rout_own)
        r_foe = (out & foe).any(1) & torch.isnan(self.t_rout_foe) & live
        self.t_rout_foe = torch.where(r_foe, t, self.t_rout_foe)
        # the reserve's melee after the rout
        rm = (self.reserve & u["m"]).any(1) & torch.isfinite(self.t_rout_own) & torch.isnan(self.t_reserve) & live
        self.t_reserve = torch.where(rm, t, self.t_reserve)
        nm = (self.named & u["m"]).any(1) & torch.isnan(self.t_named) & live
        self.t_named = torch.where(nm, t, self.t_named)
        # the trade after the first contact
        ta = torch.isfinite(self.t_contact) & (t - self.t_contact >= self.after) & torch.isnan(self.trade_after) & live
        if bool(ta.any()):
            self.trade_after = torch.where(ta, _gold_trade(u, self.lord), self.trade_after)
        # my order changes
        if self.chain is not None and hasattr(self.chain, "units"):
            from sim import orders as O
            k = self.chain.units.intended.kind
            alive = mine & ~out & live[:, None]
            if self.last_kind is not None:
                self.changes += ((k != self.last_kind) & (k != O.KEEP) & alive).float().sum(1)
            self.last_kind = k.clone()
            self.alive_s += alive.float().sum(1)
            from net.unit import u2
            ins, face = u2.in_frame(u, self.chain.units.area, self.spacing)
            scored = alive & ~u["m"]
            self.exec_ok += (scored & ins & face).float().sum(1)
            self.exec_n += scored.float().sum(1)
        self.touch = touch

    def _gap(self, st, out):
        u = st.u
        front, _ = geometry.dims(u, self.spacing)
        ok = self.mine & ~out
        okf = self.foe & ~out
        w = ok.float()
        k = w.sum(1).clamp(min=1)
        mx, mz = (u["x"] * w).sum(1) / k, (u["z"] * w).sum(1) / k
        we = okf.float()
        ke = we.sum(1).clamp(min=1)
        ex, ez = (u["x"] * we).sum(1) / ke, (u["z"] * we).sum(1) / ke
        vx, vz = ex - mx, ez - mz
        L = torch.sqrt(vx * vx + vz * vz).clamp(min=1e-6)
        fx, fz = vx / L, vz / L
        f = (u["x"] - mx[:, None]) * fx[:, None] + (u["z"] - mz[:, None]) * fz[:, None]
        s = (u["x"] - mx[:, None]) * fz[:, None] - (u["z"] - mz[:, None]) * fx[:, None]
        fmax = torch.where(ok, f, torch.full_like(f, -1e9)).amax(1, keepdim=True)
        line = ok & (f >= fmax - self.line_depth)
        # the widest gap between the line's formations along the front: by their left edges, each one's left edge less
        # the farthest right edge of those before it
        big = 1e9
        lo = torch.where(line, s - front / 2, torch.full_like(s, big))
        hi = torch.where(line, s + front / 2, torch.full_like(s, -big))
        lo, order = lo.sort(1)
        reach = hi.gather(1, order).cummax(1).values
        gaps = torch.where(lo[:, 1:] < big / 2, lo[:, 1:] - reach[:, :-1], torch.full_like(lo[:, 1:], -big))
        g = gaps.amax(1).clamp(min=0.0) if gaps.shape[1] else torch.zeros(st.B, device=st.device)
        return torch.where(line.sum(1) >= 2, g, torch.full_like(g, float("nan")))

    def measures(self, limit, winner=None):
        st = self.st
        u = st.u
        trade = _gold_trade(u, self.lord)
        w = winner
        dh = torch.where(self.dh_n > 0, self.dh_sum / self.dh_n.clamp(min=1), torch.full_like(self.dh_sum, float("nan")))
        res = torch.where(torch.isfinite(self.t_rout_own),
                          torch.where(torch.isfinite(self.t_reserve), (self.t_reserve - self.t_rout_own).clamp(max=self.cap),
                                      torch.full_like(self.t_rout_own, self.cap)),
                          torch.full_like(self.t_rout_own, float("nan")))
        # (a reserve never in melee within cap after the rout: cap; one in melee at the rout's second: 0)
        early = (torch.isfinite(self.t_named) & torch.isfinite(self.t_contact)
                 & (self.t_named < self.t_contact + self.early)
                 & (torch.isnan(self.t_rout_own) | (self.t_named < self.t_rout_own))).float()
        early = torch.where(self.has_named, early, torch.full_like(early, float("nan")))
        tc = torch.where(torch.isfinite(self.t_contact), self.t_contact, torch.full_like(self.t_contact, float(limit)))
        ta = torch.where(torch.isnan(self.trade_after), trade, self.trade_after)
        m = {"trade": trade, "t_contact": tc, "flank_hits": self.flank_hits, "flank_taken": self.flank_taken,
             "flank_dmg": torch.where(self.dmg > 0, self.dmg_flank / self.dmg.clamp(min=1e-6),
                                      torch.full_like(self.dmg, float("nan"))),
             "gap_m": self.gap, "dh": dh, "reserve_s": res,
             "orders_min": self.changes / (self.alive_s / 60.0).clamp(min=1e-6),
             "enemy_rout_s": self.t_rout_foe - self.t_contact,
             "pin_flank": torch.where(torch.isfinite(self.t_contact), self.pin_flank,
                                      torch.full_like(self.pin_flank, float("nan"))),
             "trade_after": ta, "reserve_early": early,
             "t_end": torch.where(self.t_end > 0, self.t_end, torch.full_like(self.t_end, float(limit)))}
        if self.chain is not None and getattr(getattr(self.chain, "units", None), "gate", None) is not None:
            gt = self.chain.units.gate
            m["game_orders_min"] = gt.per_minute(self.mine)
            m["raw_orders_min"] = gt.per_minute(self.mine, raw=True)
        m["head_on"] = torch.where(self.ho_n > 0, self.ho_sum / self.ho_n.clamp(min=1),
                                   torch.full_like(self.ho_n, float("nan")))
        m["frame_exec"] = torch.where(self.exec_n > 0, self.exec_ok / self.exec_n.clamp(min=1),
                                      torch.full_like(self.exec_n, float("nan")))
        my_lord = (self.lord & (u["side"] == 1)).float()
        moved = torch.sqrt((u["x"] - self.x0) ** 2 + (u["z"] - self.z0) ** 2)
        m["lord_moved_m"] = (moved * my_lord).sum(1)          # (my lord holds 150 m behind: a check of the rule)
        if w is not None:
            m["win"] = torch.where(w == 1, 1.0, torch.where(w == 2, -1.0, 0.0)).float()
        return m


def mean_ci(x):
    """(mean, 1.96 x the standard error) over the finite values; (nan, nan) when none."""
    x = x.float()
    x = x[torch.isfinite(x)]
    n = x.numel()
    if n == 0:
        return float("nan"), float("nan")
    sd = float(x.std()) if n > 1 else 0.0
    return round(float(x.mean()), 4), round(1.96 * sd / math.sqrt(n), 4)


def summary(m, sel=None):
    out = {}
    for k, v in m.items():
        v = v if sel is None else v[sel]
        out[k] = mean_ci(v)
        out[k + "_n"] = int(torch.isfinite(v.float()).sum())
    return out
