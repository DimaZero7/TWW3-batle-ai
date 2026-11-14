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


def hp_trade(u, lord):
    """[B] the trade by plain health (the gold trade's form without the rout surcharge): each unit's health lost
    (1 - hp_abs / hp0) x its cost, the enemy's less mine, over the mean of the two armies' cost; the lords out."""
    keep = ~lord & (u["side"] > 0)
    c = u["cost"] * keep
    lost_hp = c * (1 - u["hp_abs"] / u["hp0"].clamp(min=1e-6)).clamp(0, 1)
    lost = [(lost_hp * (u["side"] == s)).sum(1) for s in (1, 2)]
    budget = torch.stack([(c * (u["side"] == s)).sum(1) for s in (1, 2)], 1).mean(1).clamp(min=1.0)
    return (lost[1] - lost[0]) / budget


class PinWatch(Watch):
    """Watch + the measures of drill T0-2 'pin and strike' (tmp/tactic_slice/design_flank_v2.md section 6), my side a
    battle.TacticSide (its level's orders in force: lower.Tacticians.orders_of). Each counts what its name says; from
    the simulator's state (a measure, not an input), contact = either unit's hit_sec on the other:
    hp_trade        hp_trade() at the end of the play; hp_series [B, T] the same every SERIES_S s, kept from the second
                    the battle ended (the judge's window is read off it: the anchor + 300 s)
    routs_own, routs_enemy   rout events (a unit's r turning on), the lords out
    attack_orders   attack orders given (an order = its target and its second)
    intercepted     attack orders whose striker, before its first contact with its target, was in contact with another
                    enemy while its target's formation was INTERCEPT_M or more away (edge to edge)
    pin_lost        attack orders whose target, standing, before the striker's first contact, had no other standing unit
                    of mine in contact with it
    gone_before     attack orders whose target routed, died or left before the striker's first contact
    resumed_on_rallied   attack orders whose target routed while the order stood and rallied, and the unit's script
                    then took the attack on it up again; resumed_way_m - the striker's distance to it then (summed)
    reached         attack orders whose striker came into contact with its target
    inter_nearer    intercepted orders whose catching enemy was nearer the striker than its target at the order
    order60_{inter,clean}_{n,sum}   attack orders, intercepted or not: the striker's and its target's trade from the
                    order to 60 s on (or to the next order): (target's health lost x its cost - the striker's) / the mean
                    of the two costs
    fc_{cat}_{n,n30,s30,n60,s60}   first contacts of my units with enemy units by who charged (FC_CATS: _contacts) and
                    the pair's trade summed 30 / 60 s on
    cause_{c}_{orders,reached,intercepted,first,o60_n,o60_sum,loss60_sum}   the attack orders by the cause the player
                    tagged them with (CAUSES; its last_tags {cause, u, cell} [r, G] of the tick's new orders, tactic/pin.py
                    Teacher; untagged: none): given, reached, intercepted, reached before the target touched the unit u
                    it came at (the tag's u), and from the order to 60 s on (or to the next order; the play's end for
                    the last ones) the pair's trade and the striker's own health lost (share)
    order_log       a row a closed order (ORDER_LOG; the measures' caller reads it)"""
    SERIES_S = 5
    CAUSES = ("none", "pin", "second", "other")
    ORDER_LOG = ("b", "slot", "sec", "target", "cause", "u", "cell", "trade60", "loss60", "reached", "first",
                 "intercepted", "final")
    INTERCEPT_M = 10.0
    FC_CATS = ("mine_charging", "enemy_onto_standing", "both_moving", "neither")
    MOVING_MS = 1.0          # (the first contacts' split: 'charging' = moving over this the second before, m/s)

    def _init(self, st):
        super()._init(st)
        B, N, dev = self.B, self.N, self.dev
        z = torch.zeros(B, device=dev)
        self.routs = {1: z.clone(), 2: z.clone()}
        self.r_prev = st.u["r"].clone()
        self.series = []
        self.hp_last = None
        self.ev = {k: z.clone() for k in ("attack_orders", "intercepted", "pin_lost", "gone_before",
                                          "resumed_on_rallied", "resumed_way_m", "reached")}
        zb = lambda: torch.zeros((B, N), dtype=torch.bool, device=dev)          # noqa: E731
        self.cur_t = torch.full((B, N), -1, dtype=torch.long, device=dev)
        self.cur_born = torch.full((B, N), -1.0, device=dev)
        self.f_reach, self.f_inter, self.f_lost, self.f_gone, self.f_routed, self.f_res = (zb() for _ in range(6))
        # (the orders' 60 s trade by interception; the first contacts' pair trade by who charged)
        zf = lambda: torch.zeros((B, N), device=dev)                            # noqa: E731
        self.o_t0, self.o_hi, self.o_ht, self.o_ci, self.o_ct = (zf() for _ in range(5))
        self.o_pend, self.o_int = zb(), zb()
        self.o_d0 = torch.zeros((B, N, N), device=dev)
        self.o_tg = torch.zeros((B, N), dtype=torch.long, device=dev)
        for k in ("order60_inter_n", "order60_inter_sum", "order60_clean_n", "order60_clean_sum", "inter_nearer"):
            self.ev[k] = z.clone()
        zz = lambda: torch.zeros((B, N, N), device=dev)                         # noqa: E731
        self.fc_seen = torch.zeros((B, N, N), dtype=torch.bool, device=dev)
        self.fc_t0, self.fc_hi, self.fc_hj = zz(), zz(), zz()
        self.fc_cat = torch.full((B, N, N), -1, dtype=torch.long, device=dev)
        for c in self.FC_CATS:
            for w in ("n", "n30", "s30", "n60", "s60"):
                self.ev[f"fc_{c}_{w}"] = z.clone()
        self.v_prev = torch.zeros((B, N), device=dev)
        # (the orders by the player's tags: cause, the unit u its second enemy came at, the target's cell)
        zl = lambda v: torch.full((B, N), v, dtype=torch.long, device=dev)      # noqa: E731
        self.tag_now = {"cause": zl(0), "u": zl(-1), "cell": zl(-1)}
        self.o_tag = {k: v.clone() for k, v in self.tag_now.items()}
        self.f_utouch, self.f_first = zb(), zb()
        self.tick_seen = -1
        self.order_log = []
        for c in self.CAUSES:
            for k in ("orders", "reached", "first", "intercepted", "o60_n", "o60_sum", "loss60_sum"):
                self.ev[f"cause_{c}_{k}"] = z.clone()

    def __call__(self, sec, st):
        super().__call__(sec, st)
        u = st.u
        live = ~st.done
        new = u["r"] & ~self.r_prev & ~self.lord & live[:, None]
        for s in (1, 2):
            self.routs[s] += (new & (u["side"] == s)).float().sum(1)
        self.r_prev = u["r"].clone()
        hp = hp_trade(u, self.lord)
        self.hp_last = hp if self.hp_last is None else torch.where(live, hp, self.hp_last)
        if sec % self.SERIES_S == 0:
            self.series.append(self.hp_last.clone())
        tac = getattr(self.chain, "tac", None)
        if tac is None or getattr(tac, "player", None) is None:
            return
        self._events(st, tac, live, sec)
        self._contacts(st, live, sec)
        self.v_prev = torch.sqrt(u["vx"] ** 2 + u["vz"] ** 2)

    def _hpf(self, u):
        return u["hp_abs"] / u["hp0"].clamp(min=1e-6)

    def _contacts(self, st, live, sec):
        """The first contact of each pair (my unit i, an enemy j): who charged - my unit moving over MOVING_MS the
        second before under the game order ATTACK on j (mine_charging), else my unit under MOVING_MS and j over it
        (enemy_onto_standing), else both over it (both_moving), else neither - and the pair's trade 30 and 60 s on:
        (j's health lost x its cost - i's x its cost) / the mean of the two costs (from all sources)."""
        from sim import orders as O
        u = st.u
        B, N = self.B, self.N
        hs = u["hit_sec"].view(B, N, N)
        touch = (hs > 0) | (hs.transpose(1, 2) > 0)
        pair = self.mine[:, :, None] & self.foe[:, None, :] & live[:, None, None]
        new = touch & pair & ~self.fc_seen
        hp = self._hpf(u)
        if bool(new.any()):
            it = self.chain.units.intended
            j = torch.arange(N, device=self.dev)[None, None, :]
            mine_att = (it.kind == O.ATTACK)[:, :, None] & (it.target[:, :, None] == j)
            vi, vj = self.v_prev[:, :, None], self.v_prev[:, None, :]
            fast_i, fast_j = vi > self.MOVING_MS, vj > self.MOVING_MS
            cat = torch.where(mine_att & fast_i, 0, torch.where(~fast_i & fast_j, 1, torch.where(fast_i & fast_j, 2, 3)))
            self.fc_cat = torch.where(new, cat, self.fc_cat)
            self.fc_t0 = torch.where(new, torch.full_like(self.fc_t0, float(sec)), self.fc_t0)
            self.fc_hi = torch.where(new, hp[:, :, None].expand(B, N, N), self.fc_hi)
            self.fc_hj = torch.where(new, hp[:, None, :].expand(B, N, N), self.fc_hj)
            self.fc_seen |= new
        c = u["cost"]
        ci, cj = c[:, :, None], c[:, None, :]
        tr = ((self.fc_hj - hp[:, None, :]) * cj - (self.fc_hi - hp[:, :, None]) * ci) / ((ci + cj) / 2).clamp(min=1e-6)
        for w in (30, 60):
            due = self.fc_seen & (float(sec) - self.fc_t0 == w)
            if not bool(due.any()):
                continue
            for k, name in enumerate(self.FC_CATS):
                d = due & (self.fc_cat == k)
                self.ev[f"fc_{name}_n{w}"] += d.flatten(1).float().sum(1)
                self.ev[f"fc_{name}_s{w}"] += (tr * d).flatten(1).sum(1)
        for k, name in enumerate(self.FC_CATS):
            self.ev[f"fc_{name}_n"] += (new & (self.fc_cat == k)).flatten(1).float().sum(1)

    def _order_close(self, st, which, final=False):
        """The pending orders which [B, N] evaluated: the striker and its target's trade since the order (as the first
        contacts'), summed by intercepted or not (final: the orders still pending at the play's end - only the cause
        sums and the log), and by the player's cause; the striker's own health lost since the order; a row of
        order_log each (ORDER_LOG)."""
        u = st.u
        hp = self._hpf(u)
        ht = hp.gather(1, self.o_tg)
        tr = ((self.o_ht - ht) * self.o_ct - (self.o_hi - hp) * self.o_ci) / ((self.o_ci + self.o_ct) / 2).clamp(min=1e-6)
        if not final:
            for flag, k in ((self.o_int, "inter"), (~self.o_int, "clean")):
                d = which & flag
                self.ev[f"order60_{k}_n"] += d.float().sum(1)
                self.ev[f"order60_{k}_sum"] += (tr * d).sum(1)
        loss = self.o_hi - hp
        for c, name in enumerate(self.CAUSES):
            d = which & (self.o_tag["cause"] == c)
            self.ev[f"cause_{name}_o60_n"] += d.float().sum(1)
            self.ev[f"cause_{name}_o60_sum"] += (tr * d).sum(1)
            self.ev[f"cause_{name}_loss60_sum"] += (loss * d).sum(1)
        b, n = which.nonzero(as_tuple=True)
        if b.numel():
            sec = self.o_t0[b, n]
            cols = [b.float(), n.float(), sec, self.o_tg[b, n].float()] + [self.o_tag[k][b, n].float() for k in
                                                                            ("cause", "u", "cell")]
            cols += [tr[b, n], loss[b, n], self.f_reach[b, n].float(), self.f_first[b, n].float(),
                     self.o_int[b, n].float(), torch.full_like(sec, float(final))]
            self.order_log += torch.stack(cols, 1).cpu().tolist()
        self.o_pend &= ~which

    def _events(self, st, tac, live, sec=0):
        from sim import orders as O
        u = st.u
        B, N = self.B, self.N
        o = tac.orders_of()
        att = o["on"] & (o["skill"] == 2) & (o["tgt"] >= 0) & self.mine & live[:, None]      # (scripts.ATTACK)
        t = o["tgt"].clamp(min=0)
        fresh = att & ((o["tgt"] != self.cur_t) | (o["born"] != self.cur_born))
        # the player's tags of the attack orders its last tick gave (Teacher.last_tags [r, G] in my-slot order)
        tags = getattr(tac.player, "last_tags", None)
        if tags is not None and tac.view is not None and tac.ticks != self.tick_seen:
            self.tick_seen = tac.ticks
            v = tac.view
            na = v["new_attack"] & (v["my"] >= 0)
            rr, gg = na.nonzero(as_tuple=True)
            if rr.numel():
                bb, nn = v["bi"][rr], v["my"][rr, gg]
                for k in self.tag_now:
                    self.tag_now[k][bb, nn] = tags[k][rr, gg].long()
        # the orders' trade: 60 s after the order, or when another order replaces it
        self._order_close(st, self.o_pend & (fresh | (float(sec) - self.o_t0 >= 60.0)))
        for k in self.tag_now:
            self.o_tag[k] = torch.where(fresh, self.tag_now[k], self.o_tag[k])
            self.tag_now[k] = torch.where(fresh, torch.full_like(self.tag_now[k], -1 if k != "cause" else 0),
                                          self.tag_now[k])
        for c, name in enumerate(self.CAUSES):
            self.ev[f"cause_{name}_orders"] += (fresh & (self.o_tag["cause"] == c)).float().sum(1)
        if bool(fresh.any()):
            hp = self._hpf(u)
            dd = torch.sqrt((u["x"][:, :, None] - u["x"][:, None, :]) ** 2 + (u["z"][:, :, None] - u["z"][:, None, :]) ** 2)
            self.o_t0 = torch.where(fresh, torch.full_like(self.o_t0, float(sec)), self.o_t0)
            self.o_tg = torch.where(fresh, t, self.o_tg)
            self.o_hi = torch.where(fresh, hp, self.o_hi)
            self.o_ht = torch.where(fresh, hp.gather(1, t), self.o_ht)
            self.o_ci = torch.where(fresh, u["cost"], self.o_ci)
            self.o_ct = torch.where(fresh, u["cost"].gather(1, t), self.o_ct)
            self.o_d0 = torch.where(fresh[..., None], dd, self.o_d0)
            self.o_pend |= fresh
            self.o_int &= ~fresh
        for f in (self.f_reach, self.f_inter, self.f_lost, self.f_gone, self.f_routed, self.f_res, self.f_utouch,
                  self.f_first):
            f &= ~fresh
        self.ev["attack_orders"] += fresh.float().sum(1)
        self.cur_t = torch.where(att, o["tgt"], torch.full_like(self.cur_t, -1))
        self.cur_born = torch.where(att, o["born"], torch.full_like(self.cur_born, -1.0))
        hs = u["hit_sec"].view(B, N, N)
        touch = (hs > 0) | (hs.transpose(1, 2) > 0)
        standing = (u["men"] > 0) & ~u["gone"] & ~u["r"]
        g = lambda v: v.gather(1, t)                                                 # noqa: E731
        t_touch = touch.gather(2, t[..., None])[..., 0]
        front, depth = geometry.dims(u, self.spacing)
        gap = geometry.rect_gap(u["x"], u["z"], u["b"], front, depth).gather(2, t[..., None])[..., 0]
        foe_touch = (touch & self.foe[:, None, :] & (torch.arange(N, device=self.dev)[None, None, :] != t[..., None])).any(2)
        mine_on = (touch & (self.mine & standing)[:, :, None]).float().sum(1)               # [B, e] my touchers
        others = g(mine_on) - t_touch.float()
        before = att & ~self.f_reach
        icp = before & foe_touch & (gap >= self.INTERCEPT_M) & ~t_touch
        nw_i = icp & ~self.f_inter
        if bool(nw_i.any()):
            jn = torch.arange(N, device=self.dev)[None, None, :]
            catch = touch & self.foe[:, None, :] & (jn != t[..., None])
            dj = torch.where(catch, self.o_d0, torch.full_like(self.o_d0, float("inf"))).amin(2)
            near = nw_i & (dj < self.o_d0.gather(2, t[..., None])[..., 0])
            self.ev["inter_nearer"] += near.float().sum(1)
            self.o_int |= nw_i & self.o_pend
        cz = self.o_tag["cause"]
        for c, name in enumerate(self.CAUSES):
            self.ev[f"cause_{name}_intercepted"] += (icp & ~self.f_inter & (cz == c)).float().sum(1)
        self._flag(self.f_inter, icp, "intercepted")
        self._flag(self.f_lost, before & g(standing) & (others <= 0) & ~t_touch, "pin_lost")
        self._flag(self.f_gone, before & ~g(standing), "gone_before")
        # the target touched the unit u it came at (the tag's u) before the striker reached it
        uu = self.o_tag["u"]
        has_u = uu >= 0
        tu =touch[torch.arange(B, device=self.dev)[:, None], t, uu.clamp(min=0)] & has_u
        self.f_utouch |= before & tu
        nw_r = att & t_touch & ~self.f_reach
        first = nw_r & has_u & ~self.f_utouch
        self.f_first |= first
        for c, name in enumerate(self.CAUSES):
            self.ev[f"cause_{name}_reached"] += (nw_r & (cz == c)).float().sum(1)
            self.ev[f"cause_{name}_first"] += (first & (cz == c)).float().sum(1)
        self._flag(self.f_reach, att & t_touch, "reached")
        self.f_routed |= att & g(u["r"])
        it = self.chain.units.intended
        back = att & self.f_routed & g(standing) & (it.kind == O.ATTACK) & (it.target == o["tgt"])
        nw = back & ~self.f_res
        d = torch.sqrt((u["x"] - g(u["x"])) ** 2 + (u["z"] - g(u["z"])) ** 2)
        self.ev["resumed_way_m"] += (d * nw).sum(1)
        self._flag(self.f_res, back, "resumed_on_rallied")

    def _flag(self, f, cond, name):
        nw = cond & ~f
        self.ev[name] += nw.float().sum(1)
        f |= nw

    def measures(self, limit, winner=None):
        m = super().measures(limit, winner)
        u = self.st.u
        if hasattr(self, "o_pend") and bool(self.o_pend.any()):
            self._order_close(self.st, self.o_pend.clone(), final=True)
        m["hp_trade"] = hp_trade(u, self.lord)             # (at the end of the play, as the pilot's: tmp/tactic_slice/pilot)
        m["routs_own"], m["routs_enemy"] = self.routs[1], self.routs[2]
        ser = torch.stack(self.series, 1)
        T = int(limit) // self.SERIES_S + 1                 # (a play that ended early: its last value kept to the limit)
        if ser.shape[1] < T:
            ser = torch.cat([ser, ser[:, -1:].expand(-1, T - ser.shape[1])], 1)
        m["hp_series"] = ser[:, :T]
        m.update(self.ev)
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
