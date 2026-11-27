"""The lord's executor at the unit level (no network): the T2 lord script (bridge/companion/lord_script.py - its rules
and its constants, imported from there: one source) as the lord's DEFAULT order, batched in torch for the simulator.
It reads only the input's columns (the unit level's raw row net/unit/view.RAW, the tactic row's button timers a_on /
a_cd of net/tactic/raw.py, and the game's current target of each unit - the bridge's row `t`, the simulator's
u["target"]) - the same numbers the bridge has on its state document (lord_inputs_doc), so the decision is the
bridge script's (contract/test_lord_script_parity.py).

    post      LS.POST_BACK_M behind the centre pair of the front cluster of our standing units (the line's heading: the
              normal of their centres' principal axis toward the seen enemy), never less than LS.POST_MIN_M behind
              the front cluster's rearmost centre; a MOVE when the post moved more than LS.REORDER_M from the point sent
    run-in    the pair advances (a pair unit's order in force ATTACK, or the pair's midpoint forward faster than
              LS.ADV_MPS in LS.ADV_TICKS decisions running): the post leads by LS.LEAD_M, re-sent past LS.ADV_REORDER_M,
              a near point nudged LS.FACE_NUDGE_DEG (the bridge's same-order filter)
    fight     from the first contact of our line: ATTACK the nearest enemy in witnessed melee with our units (LS.witnessed:
              a target link within LS.LINK_M, or no link and no other opponent within LS.NEAR_M); the target dropped
              unseen / down, or LS.GRACE_S after neither our units nor he fight it
    hygiene   no new order while he is in melee, except the step-out
    step-out  health below LS.STEP_OUT_HP: WITHDRAW LS.OUT_BACK_M behind the nearest standing own unit (re-chosen when it
              goes or with no progress, LS.RESENDS times), 'out' within LS.ARRIVED_M; no return
    buttons   one press a decision by priority: Verminous Valour due at the step-out; Stand Your Ground - >= LS.SYG_MIN own
              units in melee < LS.AURA_M; Rally - >= 1 wavering < LS.AURA_M; Foe-Seeker / Deadly Onslaught - he is in
              melee. A button is pressable when its state is ready and its phase is not on (a_on 0). Requested ->
              took when its timer shows (a_cd > 0) within LS.CONFIRM_S: cooldown to the timer's end + LS.READY_PAD_S
              (= call + active + recharge + 1 s: the game refused every press at exactly call + active + recharge,
              tmp/gate_press/evidence_draft.md); not took -> a miss, again after LS.RETRY_S; the LS.TRIES-th miss stops
              that button. What the bridge reads from the card (the phase seen) is the timer here: on the bridge's
              documents the two agree (exchange.ability_timers zeroes a use the card does not show)

T2(lord, keys, rank, device, own=1, buttons=True)   lord [B] the lord's slot, keys [B] his unit key, rank [B, N] the
              slots' order numbers (the bridge's slot_no of the names; the simulator's slot index)
T2.decide(raw [B, N, len(V.RAW)], tgt [B, N] long (-1 none, -2 a unit not in the battle's rows), a_on / a_cd [B, N, A]
              (net/tactic/raw.ARSENAL), kinds [B, N] long (this second's orders of our other units, KEEP none), t [B] s,
              live [B]) -> the lord's decision {kind [B] (O.KEEP: none new), x, z, face, run, target [B], press [B] (the
              simulator's ability slot, -1 none), key [B] (index of PRIORITY, -1)}; self.ph [B] the branch (PHASES)
lord_inputs_sim(st, P) / lord_inputs_doc(doc, names, seer_raw)   (raw, tgt, a_on, a_cd) the same columns of the two
"""
import torch

from bridge.companion import lord_script as LS
from net.tactic import raw as TR
from net.unit import view as V
from sim import orders as O

PRIORITY = (LS.VALOUR, LS.SYG, LS.RALLY, LS.FOE_SEEKER, LS.ONSLAUGHT)
P_VAL, P_SYG, P_RALLY, P_FS, P_DO = range(len(PRIORITY))
READY, PENDING, COOL, UNKNOWN, WALL = range(5)
NONE = -1                                    # (kind: no order given yet - the bridge script's None)
PHASES = ("down", "out", "stepping", "step_out", "melee", "fight", "alone", "place")
F64 = torch.float64


def _fwd(h):
    a = torch.deg2rad(h)
    return torch.sin(a), torch.cos(a)


def _bearing(dx, dz):
    return torch.remainder(torch.rad2deg(torch.atan2(dx, dz)), 360.0)


def _angle(a, b):
    return (torch.remainder(a - b + 180.0, 360.0) - 180.0).abs()


def _w(m, a, b):
    return torch.where(m, a, b)


class T2:
    def __init__(self, lord, keys, rank, device, own=1, buttons=True):
        from net.cards import abilities as MA
        from net.cards import passport
        from sim import abilities as SA
        B = int(lord.shape[0])
        self.B, self.own, self.dev = B, own, device
        self.li = lord.to(device).long()
        self.rank = rank.to(device).to(F64)
        self.buttons = buttons
        units, pp = passport.load(), {**MA.load_spells(), **MA.load()}
        K = len(PRIORITY)
        self.owns = torch.zeros((B, K), dtype=torch.bool, device=device)
        self.slot = torch.full((B, K), -1, dtype=torch.long, device=device)
        cache = {}
        for b, key in enumerate(keys):
            if key not in cache:
                cache[key] = SA.press_map(key, units, pp)          # (fails closed on an unmapped active)
            for s, k in cache[key].items():
                if k in PRIORITY:
                    self.owns[b, PRIORITY.index(k)] = True
                    self.slot[b, PRIORITY.index(k)] = int(s)
        self.ai = torch.tensor([TR.ARSENAL.index(k) for k in PRIORITY], device=device)
        z = lambda dt=F64, v=0.0: torch.full((B,), v, dtype=dt, device=device)        # noqa: E731
        zb = lambda: torch.zeros(B, dtype=torch.bool, device=device)                 # noqa: E731
        # the line and the post
        self.has_h, self.h = zb(), z()
        self.pair = torch.full((B, 2), -1, dtype=torch.long, device=device)
        self.pp_ok, self.pp_pair, self.pp_t = zb(), self.pair.clone(), z()
        self.pp_x, self.pp_z = z(), z()
        self.fast = z(torch.long, 0)
        self.adv = zb()
        self.last_kind = None                                       # [B, N] the order in force of our units
        self.has_lpost, self.lpx, self.lpz = zb(), z(), z()
        self.contact = zb()
        # the order in force (the script's memory)
        self.kind = z(torch.long, NONE)
        self.attacking = z(torch.long, -1)
        self.has_lp, self.lpt_x, self.lpt_z, self.move_t = zb(), z(), z(), z()
        self.target, self.has_lost, self.lost_t = z(torch.long, -1), zb(), z()
        self.led, self.nudge = zb(), z(v=1.0)
        # the step-out
        self.step, self.s_fail, self.out, self.val_due = zb(), zb(), zb(), zb()
        self.sx, self.sz, self.sface, self.s_t, self.s_d = z(), z(), z(), z(), z()
        self.s_ally, self.s_res = z(torch.long, -1), z(torch.long, 0)
        self.s_tried = None                                         # [B, N]
        # the buttons
        self.bst = torch.full((B, K), READY, dtype=torch.long, device=device)
        self.treq, self.cdend, self.retry = (torch.zeros((B, K), dtype=F64, device=device) for _ in range(3))
        self.miss = torch.zeros((B, K), dtype=torch.long, device=device)
        self.ph = z(torch.long, 0)
        self.presses = torch.zeros(B, device=device)
        self.n_orders, self.n_melee_orders, self.n_melee_ma, self.melee_s = (torch.zeros(B, device=device)
                                                                             for _ in range(4))

    # ------------------------------------------------------------------ the buttons' bookkeeping
    def _follow(self, live, t, on, cd):
        tk = t[:, None]
        cool_end = live[:, None] & (self.bst == COOL) & (tk >= self.cdend)
        self.miss = _w(cool_end, torch.zeros_like(self.miss), self.miss)
        self.bst = _w(cool_end, torch.full_like(self.bst, READY), self.bst)
        unk_end = live[:, None] & (self.bst == UNKNOWN) & (tk >= self.retry)
        self.bst = _w(unk_end, torch.full_like(self.bst, READY), self.bst)
        pend = live[:, None] & (self.bst == PENDING)
        late = pend & (tk - self.treq > LS.CONFIRM_S + 1e-6)        # (the deadline first, as the bridge)
        took = pend & ~late & (cd > 0) & (tk > self.treq)
        self.miss = self.miss + late.long()
        self.bst = _w(late, _w(self.miss >= LS.TRIES, torch.full_like(self.bst, WALL),
                               torch.full_like(self.bst, UNKNOWN)), self.bst)
        self.retry = _w(late, tk + LS.RETRY_S, self.retry)
        self.cdend = _w(took, tk + cd + LS.READY_PAD_S, self.cdend)
        self.bst = _w(took, torch.full_like(self.bst, COOL), self.bst)
        self.val_due = self.val_due & ~(took | (late & (self.bst == WALL)))[:, P_VAL]

    # ------------------------------------------------------------------ one decision
    @torch.no_grad()
    def decide(self, raw, tgt, a_on, a_cd, kinds, t, live):
        B, N = raw.shape[:2]
        dev = self.dev
        I = V.I
        c = lambda k: raw[..., I[k]].to(F64)                                             # noqa: E731
        x, z, bb, hp = c("x"), c("z"), c("b"), c("hp")
        side = raw[..., I["side"]].round().long()
        alive = raw[..., I["alive"]] > 0
        standing = alive & ~(raw[..., I["r"]] > 0) & ~(raw[..., I["s"]] > 0)
        m, w = raw[..., I["m"]] > 0, raw[..., I["w"]] > 0
        vis = raw[..., I["vis"]] > 0
        t = t.to(F64)
        ar = torch.arange(N, device=dev)
        bi = torch.arange(B, device=dev)
        li = self.li
        lmask = ar[None, :] == li[:, None]
        own = side == self.own
        if self.last_kind is None:
            self.last_kind = torch.full((B, N), NONE, dtype=torch.long, device=dev)
            self.s_tried = torch.zeros((B, N), dtype=torch.bool, device=dev)
        # the order in force of our units: this second's lines (alive, not routing; a KEEP keeps it)
        ctrl = own & alive & ~(raw[..., I["r"]] > 0) & ~lmask & live[:, None]
        self.last_kind = _w(ctrl & (kinds != O.KEEP), kinds, self.last_kind)
        # the buttons' timers on the lord
        on = a_on[bi, li][:, self.ai].to(F64)                                            # [B, K]
        cd = a_cd[bi, li][:, self.ai].to(F64)
        self._follow(live, t, on, cd)
        g = lambda v: v[bi, li]                                                          # noqa: E731
        lx, lz = g(x), g(z)
        up = live & g(standing)
        ph = torch.zeros(B, dtype=torch.long, device=dev)
        inf = own & standing & ~lmask
        foe = (side != self.own) & (side > 0) & vis & standing
        lord_m = g(m)
        self.contact = self.contact | (up & (inf & m).any(1))
        # the line's heading, the post
        self._heading(up, inf, foe, x, z, bb)
        post_ok, px, pz = self._post(up, inf, x, z, t)
        self.has_lpost = self.has_lpost | (up & post_ok)
        self.lpx, self.lpz = _w(up & post_ok, px, self.lpx), _w(up & post_ok, pz, self.lpz)
        dl = torch.sqrt((x - lx[:, None]) ** 2 + (z - lz[:, None]) ** 2)
        near = inf & (dl < LS.AURA_M)
        n_melee = (near & m).sum(1)
        n_waver = (near & w).sum(1)
        # the branches (the bridge's order): out, stepping, step-out, melee (no new order), else move or fight
        kind = torch.full((B,), O.KEEP, dtype=torch.long, device=dev)
        ox, oz, oface = torch.zeros(B, dtype=F64, device=dev), torch.zeros(B, dtype=F64, device=dev), \
            torch.full((B,), -1.0, dtype=F64, device=dev)
        orun = torch.zeros(B, dtype=torch.bool, device=dev)
        otg = torch.full((B,), -1, dtype=torch.long, device=dev)
        o = [kind, ox, oz, oface, orun, otg]
        b_out = up & self.out
        b_step = up & ~self.out & self.step
        b_so = up & ~self.out & ~self.step & (g(hp) < LS.STEP_OUT_HP)
        b_mel = up & ~self.out & ~self.step & ~(g(hp) < LS.STEP_OUT_HP) & lord_m
        b_mf = up & ~self.out & ~self.step & ~(g(hp) < LS.STEP_OUT_HP) & ~lord_m
        ph = _w(~up, torch.full_like(ph, PHASES.index("down")), ph)
        ph = _w(b_out, torch.full_like(ph, PHASES.index("out")), ph)
        ph = _w(b_mel, torch.full_like(ph, PHASES.index("melee")), ph)
        held = self._witnessed(inf, foe, m, tgt, x, z)                                   # [B, N] enemies held
        self._stepping(b_step, inf, x, z, bb, lx, lz, dl, t, o, ph)
        self._step_out(b_so, inf, x, z, bb, lx, lz, dl, t, o, ph)
        lt = g(tgt)
        self._keep_target(b_mel, held, foe, lord_m, lt, tgt, t)
        self._move_or_fight(b_mf, held, foe, lord_m, lt, tgt, t, x, z, lx, lz, post_ok, px, pz, o, ph)
        self.ph = ph
        # the button (one a decision, by priority)
        press = torch.full((B,), -1, dtype=torch.long, device=dev)
        keyi = torch.full((B,), -1, dtype=torch.long, device=dev)
        if self.buttons:
            want = torch.zeros((B, len(PRIORITY)), dtype=torch.bool, device=dev)
            want[:, P_VAL] = self.val_due & ~self.out
            free = ~self.step
            want[:, P_SYG] = free & (n_melee >= LS.SYG_MIN)
            want[:, P_RALLY] = free & (n_waver >= 1)
            want[:, P_FS] = free & lord_m
            want[:, P_DO] = free & lord_m
            can = up[:, None] & want & self.owns & (self.bst == READY) & ~(on > 0)
            any_ = can.any(1)
            first = torch.where(can, torch.arange(len(PRIORITY), device=dev)[None, :].expand(B, -1),
                                torch.full_like(self.slot, 99)).min(1).values
            k = first.clamp(max=len(PRIORITY) - 1)
            hit = any_[:, None] & (torch.arange(len(PRIORITY), device=dev)[None, :] == k[:, None])
            self.bst = _w(hit, torch.full_like(self.bst, PENDING), self.bst)
            self.treq = _w(hit, t[:, None].expand(-1, len(PRIORITY)), self.treq)
            press = _w(any_, self.slot[bi, k], press)
            keyi = _w(any_, k, keyi)
            self.presses += any_.float()
        kind, ox, oz, oface, orun, otg = o
        # the counts: his new orders, those given while he is in melee (MOVE / ATTACK apart), his melee seconds
        new = up & (kind != O.KEEP)
        self.n_orders += new.float()
        self.n_melee_orders += (new & lord_m).float()
        self.n_melee_ma += (new & lord_m & ((kind == O.MOVE) | (kind == O.ATTACK))).float()
        self.melee_s += (up & lord_m).float()
        return {"kind": kind, "x": ox, "z": oz, "face": oface, "run": orun, "target": otg, "press": press, "key": keyi}

    # ------------------------------------------------------------------ the line
    def _heading(self, up, inf, foe, x, z, bb):
        n = inf.sum(1)
        nf = foe.sum(1)
        fi = inf.to(F64)
        ff = foe.to(F64)
        mx = (x * fi).sum(1) / n.clamp(min=1)
        mz = (z * fi).sum(1) / n.clamp(min=1)
        ex = (x * ff).sum(1) / nf.clamp(min=1)
        ez = (z * ff).sum(1) / nf.clamp(min=1)
        last, has = self.h, self.has_h
        # one unit: the bearing to the seen enemies' centroid
        cx1, cz1 = ex - mx, ez - mz
        one = (n == 1) & (nf > 0) & ((cx1 != 0) | (cz1 != 0))
        h1 = _bearing(cx1, cz1)
        # two or more: the normal of the principal axis
        dx, dz = (x - mx[:, None]) * fi, (z - mz[:, None]) * fi
        sxx, szz, sxz = (dx * dx).sum(1), (dz * dz).sum(1), (dx * dz).sum(1)
        tr, det = sxx + szz, sxx * szz - sxz * sxz
        disc = torch.sqrt((tr * tr / 4 - det).clamp(min=0.0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc
        bunched = (l1 <= 1e-6) | ((l2 > 1e-6) & (l1 / l2.clamp(min=1e-300) < 2.0))
        th = 0.5 * torch.atan2(2 * sxz, sxx - szz)
        h = _bearing(-torch.sin(th), torch.cos(th))
        fx, fz = _fwd(h)
        flip_f = (nf > 0) & (fx * cx1 + fz * cz1 < 0)
        flip_l = (nf == 0) & has & (_angle(h, last) > 90)
        sb = (torch.sin(torch.deg2rad(bb)) * fi).sum(1)
        cb = (torch.cos(torch.deg2rad(bb)) * fi).sum(1)
        mb = torch.remainder(torch.rad2deg(torch.atan2(sb, cb)), 360.0)
        flip_b = (nf == 0) & ~has & (_angle(h, mb) > 90)
        h = _w(flip_f | flip_l | flip_b, torch.remainder(h + 180.0, 360.0), h)
        multi = (n >= 2) & ~bunched
        newh = _w(multi, h, _w((n < 2) & one, h1, last))
        newhas = has | multi | ((n < 2) & one)
        self.h = _w(up, newh, self.h)
        self.has_h = _w(up, newhas, self.has_h)

    def _post(self, up, inf, x, z, t):
        """(post_ok [B], px, pz) and the pair, the run-in (module doc; lord_script._post)."""
        B, N = inf.shape
        dev = self.dev
        ok = up & inf.any(1) & self.has_h
        fx, fz = _fwd(self.h)
        fi = inf.to(F64)
        n = inf.sum(1).clamp(min=1)
        mx, mz = (x * fi).sum(1) / n, (z * fi).sum(1) / n
        ff = (x - mx[:, None]) * fx[:, None] + (z - mz[:, None]) * fz[:, None]
        fl = (x - mx[:, None]) * fz[:, None] - (z - mz[:, None]) * fx[:, None]
        INF = torch.tensor(float("inf"), dtype=F64, device=dev)
        # the front cluster: sorted by forward (stable: the rows' order), split where the gap exceeds RANK_M
        order = torch.sort(_w(inf, -ff, INF), dim=1, stable=True).indices
        fs = ff.gather(1, order)
        vs = inf.gather(1, order)
        split = torch.zeros((B, N), dtype=torch.long, device=dev)
        split[:, 1:] = ((fs[:, :-1] - fs[:, 1:]) > LS.RANK_M).long()
        gid = torch.cumsum(split, 1)
        gid = _w(vs, gid, torch.full_like(gid, N + 1))
        size = torch.zeros((B, N + 2), dtype=torch.long, device=dev).scatter_add_(1, gid, vs.long())[:, :N]
        two = size >= 2
        last_g = _w(vs, gid, torch.full_like(gid, -1)).max(1).values
        kfirst = torch.where(two, torch.arange(N, device=dev)[None, :].expand(B, -1),
                             torch.full_like(size, N + 5)).min(1).values
        k = _w(two.any(1), kfirst, last_g)
        front_s = vs & (gid <= k[:, None])
        front = torch.zeros_like(inf).scatter_(1, order, front_s)
        nfr = front.sum(1)
        mean_f = (ff * front.to(F64)).sum(1) / nfr.clamp(min=1)
        rear = _w(front, ff, INF).min(1).values
        # the pair kept by name while both stand within KEEP_M of the front cluster's mean
        p0, p1 = self.pair[:, 0], self.pair[:, 1]
        has_p = p0 >= 0
        two_p = p1 >= 0
        bi = torch.arange(B, device=dev)
        in0 = inf[bi, p0.clamp(min=0)] & (ff[bi, p0.clamp(min=0)] - mean_f).abs().le(LS.KEEP_M)
        in1 = inf[bi, p1.clamp(min=0)] & (ff[bi, p1.clamp(min=0)] - mean_f).abs().le(LS.KEEP_M)
        keep = has_p & in0 & (~two_p | in1) & ~(~two_p & (nfr >= 2))
        # the middle pair of the front cluster by lateral order (ties: the lower slot number)
        o1 = torch.sort(_w(front, self.rank, INF), dim=1, stable=True).indices
        lat1 = fl.gather(1, o1)
        o2 = torch.sort(_w(front.gather(1, o1), lat1, INF), dim=1, stable=True).indices
        lat = o1.gather(1, o2)                                     # the front's slots by (lateral, slot_no)
        nf = nfr
        at = lambda i: lat[bi, i.clamp(min=0, max=N - 1)]          # noqa: E731
        mid = nf // 2
        q0 = torch.where(nf <= 2, at(torch.zeros_like(nf)), torch.where(nf % 2 == 0, at(mid - 1), at(mid)))
        q1 = torch.where(nf <= 1, torch.full_like(nf, -1), torch.where(nf == 2, at(torch.ones_like(nf)),
                                                                        torch.where(nf % 2 == 0, at(mid), at(mid))))
        odd = (nf >= 3) & (nf % 2 == 1)
        a_, c_ = at(mid - 1), at(mid + 1)
        lower = self.rank[bi, a_] <= self.rank[bi, c_]
        q0 = _w(odd, _w(lower, a_, at(mid)), q0)
        q1 = _w(odd, _w(lower, at(mid), c_), q1)
        newp = torch.stack([q0, q1], 1)
        pair = _w((ok & ~keep)[:, None], newp, self.pair)
        pair = _w(up[:, None] & ~ok[:, None], torch.full_like(pair, -1), pair)
        self.pair = _w(up[:, None], pair, self.pair)
        P0, P1 = self.pair[:, 0].clamp(min=0), self.pair[:, 1]
        two_p = P1 >= 0
        P1c = P1.clamp(min=0)
        npr = 1 + two_p.to(F64)
        zero = torch.zeros(B, dtype=F64, device=dev)
        px = (x[bi, P0] + _w(two_p, x[bi, P1c], zero)) / npr
        pz = (z[bi, P0] + _w(two_p, z[bi, P1c], zero)) / npr
        pf = (ff[bi, P0] + _w(two_p, ff[bi, P1c], zero)) / npr
        # the run-in: the pair's forward speed since the last decision (the same pair, in order) or its ATTACK
        same = self.pp_ok & (self.pp_pair == self.pair).all(1) & (t > self.pp_t)
        v = ((px - self.pp_x) * fx + (pz - self.pp_z) * fz) / (t - self.pp_t).clamp(min=1e-9)
        ordered = (self.last_kind[bi, P0] == O.ATTACK) | (two_p & (self.last_kind[bi, P1c] == O.ATTACK))
        fast = _w(same & (v > LS.ADV_MPS), self.fast + 1, torch.zeros_like(self.fast))
        adv = ordered | (fast >= LS.ADV_TICKS)
        lead = _w(adv, torch.full_like(px, LS.LEAD_M), torch.zeros_like(px))
        back = torch.maximum(LS.POST_BACK_M - lead, pf - rear + LS.POST_MIN_M)
        qx, qz = px - back * fx, pz - back * fz
        upd = up & ok
        self.pp_ok = _w(up, ok, self.pp_ok)
        self.pp_pair = _w(upd[:, None], self.pair, self.pp_pair)
        self.pp_t = _w(upd, t, self.pp_t)
        self.pp_x, self.pp_z = _w(upd, px, self.pp_x), _w(upd, pz, self.pp_z)
        self.fast = _w(up, _w(ok, fast, torch.zeros_like(fast)), self.fast)
        self.adv = _w(up, ok & adv, self.adv)
        return ok, qx, qz

    # ------------------------------------------------------------------ the fight
    def _witnessed(self, inf, foe, m, tgt, x, z):
        """[B, N] the enemies in witnessed melee with our standing units (lord_script.witnessed)."""
        B, N = inf.shape
        ar = torch.arange(N, device=self.dev)
        ti, tj = tgt[:, :, None], tgt[:, None, :]                  # [b, i (ours), j (theirs)]
        link = (ti == ar[None, None, :]) | (tj == ar[None, :, None])
        other = ((ti != -1) & (ti != ar[None, None, :])) | ((tj != -1) & (tj != ar[None, :, None]))
        both = m[:, :, None] & m[:, None, :]
        d = torch.sqrt((x[:, :, None] - x[:, None, :]) ** 2 + (z[:, :, None] - z[:, None, :]) ** 2)
        ok = inf[:, :, None] & foe[:, None, :] & both & ((link & (d < LS.LINK_M)) | (~link & ~other & (d < LS.NEAR_M)))
        return ok.any(1)

    def _keep_target(self, on, held, foe, lord_m, lt, tgt, t):
        """[B] the latched target stays (lord_script._keep_target) for the battles on; drops it otherwise."""
        B = on.shape[0]
        bi = torch.arange(B, device=self.dev)
        tg = self.target
        has = on & (tg >= 0)
        tc = tg.clamp(min=0)
        gone = has & ~foe[bi, tc]
        live = has & ~gone
        lord_on = lord_m & ((lt == tg) | (tgt[bi, tc] == self.li))
        hold = live & (held[bi, tc] | lord_on)
        self.has_lost = _w(hold, torch.zeros_like(self.has_lost), self.has_lost)
        loose = live & ~hold
        self.lost_t = _w(loose & ~self.has_lost, t, self.lost_t)
        self.has_lost = self.has_lost | loose
        drop = gone | (loose & (t - self.lost_t >= LS.GRACE_S))
        self.target = _w(drop, torch.full_like(tg, -1), self.target)
        self.has_lost = _w(drop, torch.zeros_like(self.has_lost), self.has_lost)
        return hold | (loose & ~drop)

    def _move(self, on, x, z, face, t, lx, lz, o, kind=O.MOVE, run_in=None):
        """lord_script._move for the battles on: o's fields and the memory."""
        if run_in is None:
            run_in = torch.zeros_like(on)
        run = (kind == O.WITHDRAW) | run_in | (torch.sqrt((lx - x) ** 2 + (lz - z) ** 2) > LS.RUN_FAR_M)
        nud = on & (run_in | self.led) & (self.kind == O.MOVE) & self.has_lp & \
            (torch.sqrt((x - self.lpt_x) ** 2 + (z - self.lpt_z) ** 2) <= LS.REORDER_M)
        face = _w(nud, face + LS.FACE_NUDGE_DEG * self.nudge, face)
        self.nudge = _w(nud, -self.nudge, self.nudge)
        self.kind = _w(on, torch.full_like(self.kind, kind), self.kind)
        self.has_lp = self.has_lp | on
        self.lpt_x, self.lpt_z, self.move_t = _w(on, x, self.lpt_x), _w(on, z, self.lpt_z), _w(on, t, self.move_t)
        self.attacking = _w(on, torch.full_like(self.attacking, -1), self.attacking)
        self.led = _w(on, run_in, self.led)
        o[0] = _w(on, torch.full_like(o[0], kind), o[0])
        o[1], o[2] = _w(on, x, o[1]), _w(on, z, o[2])
        o[3] = _w(on, torch.remainder(face, 360.0), o[3])
        o[4] = _w(on, run, o[4])
        o[5] = _w(on, torch.full_like(o[5], -1), o[5])

    def _move_or_fight(self, on, held, foe, lord_m, lt, tgt, t, x, z, lx, lz, post_ok, px, pz, o, ph):
        B = on.shape[0]
        dev = self.dev
        has = self._keep_target(on, held, foe, lord_m, lt, tgt, t)
        pick = on & self.contact & ~has & (held & foe).any(1)
        d = torch.sqrt((x - lx[:, None]) ** 2 + (z - lz[:, None]) ** 2)
        INF = torch.tensor(float("inf"), dtype=F64, device=dev)
        j = _w(held & foe, d, INF).argmin(1)
        self.target = _w(pick, j, self.target)
        self.has_lost = _w(pick, torch.zeros_like(self.has_lost), self.has_lost)
        has = has | pick
        fight = on & has
        ph[:] = _w(fight, torch.full_like(ph, PHASES.index("fight")), ph)
        new_att = fight & ((self.kind != O.ATTACK) | (self.attacking != self.target))
        self.kind = _w(new_att, torch.full_like(self.kind, O.ATTACK), self.kind)
        self.attacking = _w(new_att, self.target, self.attacking)
        self.has_lp = self.has_lp & ~new_att
        o[0] = _w(new_att, torch.full_like(o[0], O.ATTACK), o[0])
        o[4] = _w(new_att, torch.ones_like(o[4]), o[4])
        o[5] = _w(new_att, self.target, o[5])
        # alone: no post (no own unit standing, or no heading)
        alone = on & ~has & ~post_ok
        ph[:] = _w(alone, torch.full_like(ph, PHASES.index("alone")), ph)
        back = alone & (self.kind == O.ATTACK)
        to_lp = back & self.has_lpost
        h0 = _w(self.has_h, self.h, torch.zeros_like(self.h))
        hold = back & ~self.has_lpost
        self._move(to_lp, self.lpx, self.lpz, h0, t, lx, lz, o)
        self.kind = _w(hold, torch.full_like(self.kind, O.HOLD), self.kind)
        self.attacking = _w(hold, torch.full_like(self.attacking, -1), self.attacking)
        self.has_lp = self.has_lp & ~hold
        o[0] = _w(hold, torch.full_like(o[0], O.HOLD), o[0])
        o[1], o[2] = _w(hold, lx, o[1]), _w(hold, lz, o[2])
        # the post
        place = on & ~has & post_ok
        ph[:] = _w(place, torch.full_like(ph, PHASES.index("place")), ph)
        d_lord = torch.sqrt((lx - px) ** 2 + (lz - pz) ** 2)
        moved = torch.sqrt((px - self.lpt_x) ** 2 + (pz - self.lpt_z) ** 2)
        was_move = place & (self.kind == O.MOVE) & self.has_lp
        go1 = was_move & ((moved > LS.REORDER_M) | ((self.adv | self.led) & (moved > LS.ADV_REORDER_M)))
        go2 = place & ~was_move & ((self.kind == O.ATTACK) | (d_lord > LS.REORDER_M))
        self._move(go1 | go2, px, pz, self.h, t, lx, lz, o, run_in=self.adv)

    # ------------------------------------------------------------------ the step-out
    def _out_point(self, inf, x, z, bb, lx, lz, dl, avoid=None):
        B, N = inf.shape
        bi = torch.arange(B, device=self.dev)
        h = _w(self.has_h, self.h, bb[bi, self.li])
        fx, fz = _fwd(h)
        cand = inf if avoid is None else inf & ~avoid
        cand = _w(cand.any(1)[:, None], cand, inf)
        INF = torch.tensor(float("inf"), dtype=F64, device=self.dev)
        j = _w(cand, dl, INF).argmin(1)
        has = cand.any(1)
        x0 = _w(has, x[bi, j] - LS.OUT_BACK_M * fx, lx - 2 * LS.OUT_BACK_M * fx)
        z0 = _w(has, z[bi, j] - LS.OUT_BACK_M * fz, lz - 2 * LS.OUT_BACK_M * fz)
        return x0, z0, h, _w(has, j, torch.full_like(j, -1))

    def _step_out(self, on, inf, x, z, bb, lx, lz, dl, t, o, ph):
        ph[:] = _w(on, torch.full_like(ph, PHASES.index("step_out")), ph)
        x0, z0, h, ally = self._out_point(inf, x, z, bb, lx, lz, dl)
        self.step = self.step | on
        self.sx, self.sz, self.sface = _w(on, x0, self.sx), _w(on, z0, self.sz), _w(on, h, self.sface)
        self.s_ally = _w(on, ally, self.s_ally)
        self.s_t = _w(on, t, self.s_t)
        self.s_d = _w(on, torch.sqrt((lx - x0) ** 2 + (lz - z0) ** 2), self.s_d)
        self.s_res = _w(on, torch.zeros_like(self.s_res), self.s_res)
        hot = torch.nn.functional.one_hot(ally.clamp(min=0), inf.shape[1]).bool() & (ally >= 0)[:, None]
        self.s_tried = _w(on[:, None], hot, self.s_tried)
        self.val_due = self.val_due | (on & self.owns[:, P_VAL])
        self._move(on, x0, z0, h, t, lx, lz, o, kind=O.WITHDRAW)

    def _stepping(self, on, inf, x, z, bb, lx, lz, dl, t, o, ph):
        B, N = inf.shape
        bi = torch.arange(B, device=self.dev)
        ph[:] = _w(on, torch.full_like(ph, PHASES.index("stepping")), ph)
        live = on & ~self.s_fail
        d = torch.sqrt((lx - self.sx) ** 2 + (lz - self.sz) ** 2)
        arrived = live & (d <= LS.ARRIVED_M)
        self.out = self.out | arrived
        go = live & ~arrived
        gone = go & (self.s_ally >= 0) & ~inf[bi, self.s_ally.clamp(min=0)]
        due = go & (t - self.s_t >= LS.RESEND_S)
        stalled = due & (self.s_d - d < LS.PROGRESS_M)
        prog = due & ~stalled
        self.s_t, self.s_d = _w(prog, t, self.s_t), _w(prog, d, self.s_d)
        again = go & (gone | stalled)
        fail = again & (self.s_res >= LS.RESENDS)
        self.s_fail = self.s_fail | fail
        re = again & ~fail
        x0, z0, h, ally = self._out_point(inf, x, z, bb, lx, lz, dl, avoid=_w(stalled[:, None], self.s_tried,
                                                                              torch.zeros_like(self.s_tried)))
        same = torch.sqrt((x0 - self.sx) ** 2 + (z0 - self.sz) ** 2) <= LS.REORDER_M
        fx, fz = _fwd(h)
        sg = _w(self.s_res % 2 == 0, torch.ones_like(x0), -torch.ones_like(x0))
        x1 = _w(same, x0 - 2 * LS.REORDER_M * fz * sg, x0)
        z1 = _w(same, z0 + 2 * LS.REORDER_M * fx * sg, z0)
        self.sx, self.sz, self.sface = _w(re, x1, self.sx), _w(re, z1, self.sz), _w(re, h, self.sface)
        self.s_ally, self.s_t = _w(re, ally, self.s_ally), _w(re, t, self.s_t)
        self.s_d = _w(re, torch.sqrt((lx - x1) ** 2 + (lz - z1) ** 2), self.s_d)
        self.s_res = _w(re, self.s_res + 1, self.s_res)
        hot = torch.nn.functional.one_hot(ally.clamp(min=0), N).bool() & (ally >= 0)[:, None]
        self.s_tried = self.s_tried | (re[:, None] & hot)
        self._move(re, x1, z1, h, t, lx, lz, o, kind=O.WITHDRAW)


# ---------------------------------------------------------------------- the inputs, the same columns on both sides
def lord_inputs_sim(st, P, raw_u=None):
    """(raw [B, N, V.RAW], tgt [B, N], a_on [B, N, A], a_cd [B, N, A]) of a simulator state: view.raw (or the unit
    level's raw of this second), the game's current target (u["target"]), the button timers (tactic/raw.observe_sim)."""
    if raw_u is None:
        raw_u = V.raw(st, P.sim["formation"]["spacing_m"], missile=P.sim["missile"])
    obs, _ = TR.observe_sim(st, P)
    alive = (st.u["side"] > 0) & (st.u["men"] > 0) & ~st.u["gone"]
    tgt = torch.where(alive, st.u["target"].long(), torch.full_like(st.u["target"].long(), -1))
    a = alive[..., None].float()
    return raw_u, tgt, obs["a_on"] * a, obs["a_cd"] * a


def lord_inputs_doc(doc, names, raw):
    """The same of one bridge state document: raw = the companion's tactic raw [1, N, TR.RAW] of doc (TacticSeer.raw,
    names its rows), tgt from the rows' `t` (a name not among names: -2, none: -1)."""
    idx = {n: i for i, n in enumerate(names)}
    rows = {u["n"]: u for u in doc["units"]}
    t = []
    for n in names:
        v = (rows.get(n) or {}).get("t")
        t.append(-1 if v in (None, "") else idx.get(v, -2))
    tgt = torch.tensor([t], dtype=torch.long, device=raw.device)
    A = TR.A
    return (raw[..., :TR.BASE], tgt, raw[..., TR.I["a_on0"]:TR.I["a_on0"] + A],
            raw[..., TR.I["a_cd0"]:TR.I["a_cd0"] + A])


def slot_no(name):
    return LS.slot_no(name)


def rank_of_names(names):
    return torch.tensor([[float(min(slot_no(n), 10 ** 6)) for n in names]], dtype=F64)