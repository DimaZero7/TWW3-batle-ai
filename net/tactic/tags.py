"""The tags of drill T0-2's labels and the kinds of its head-free unit-ticks (tmp/tactic_slice/design_attack_v23.md
section 4 'Unit-ticks'; review_attack_v23 C binding: n1 don't-care; n2 as tested taught keep; n2 widened, n4, n5
don't-care; n3 don't-care; n6 the coming enemy's own key don't-care, the pinned target of a unit with an own enemy coming within
come_reach_m (150 m, review J) taught keep (n7); n7 taught
keep; n9 taught keep; n9L don't-care; bands of config stage0.json drills.attack.bands). The labeller reads the level's
input only; the tagger may read the simulator's state too (View.sim: n9L, my lord's fight).

    head_free(g)   alive, not routing, not in melee, no attack order on a valid target, at least one open row
    tick(v, lab, teacher)  {per unit-tick [r, G]: head_free, attack (the label: a new attack order), k (its row), trivial,
                   dropped (the tick's own label or state inside a band), dc_label (an attack label of weight 0:
                   through_own, own enemy beyond the reach, its key in a band), kinds {name: [r, G]} (KINDS), tags of
                   the attack labels {near2, near2_at_i, mt, fresh, late, through_own, dist, side}; dc_keys [r, G, K];
                   kind_keys {n2w, n4, n5: [r, G, K]} - the enemies that make the kind (branches' kind picks)}
KINDS     taught keep: n2 (an unheld enemy within 150 m comes at another unit of mine not in melee, and a nearer enemy
          also comes at it), n7 (a cause target in reach, but a standing enemy within STICK_M of g or one of its own
          coming within come_own's reach), n9 (an enemy within reach two ticks in melee but not pinned); don't-care keys:
          n1 (the cause but mt(e) = 1), n2w (the enemy is the nearest coming at my idle unit), n3 (a cause target the
          teacher gave another unit or held back by the walk), n4 (a cause target 170-240 m away), n5 (an unheld enemy
          within 150 m that stands, goes elsewhere or chases a router), n6 (an unheld enemy comes at g within 150 m:
          its own key), n9L (the enemy's only opponent is my lord); taught keep, its key 'not the
          target' (review B): nb - the old cause (b), an unheld enemy within reach coming at a unit of mine in melee
Label tags beside: hp_e / hp_pin - the health share of the target and of the unit of mine pinning it (nearest in melee),
nearest_e - the target is the valid enemy nearest the striker; for taught keeps of n7 / n9 the same two health shares of
their key enemy (design section 4 check 8)
"""
import torch

from net.stage0 import cfg as scfg
from net.tactic import orders as OR
from net.tactic import pin as PN
from net.tactic import rows as RW
from net.unit import scripts as SC

KINDS = ("n1", "n2", "n2w", "n3", "n4", "n5", "n6", "n7", "n9", "n9L", "nb")
TAUGHT = ("n2", "n7", "n9", "nb")


def _c():
    return scfg()["drills"]["attack"]


def _within(x, band):
    return (x >= float(band[0])) & (x <= float(band[1]))


def tick(v, lab, teacher):
    """(module doc) of a View v, the labeller's output lab on it and the labeller (its touch, come_own options)."""
    fx = v.fx
    o = v.order
    R, G, K = fx.R, fx.G, fx.K
    dev = fx.s.device
    oc = OR.c_orders()
    bands, kc = _c()["bands"], _c()["kinds"]
    reach = float(oc["reach_m"])
    w = OR.Walk(fx)
    opn = torch.stack([w.open(g) for g in range(G)], 1)                             # [R, G, K]
    able, melee = fx.able, v.melee
    head = able & ~melee & ~fx.striking & opn.any(2)
    att = able & (lab["skill"] == SC.ATTACK) & (lab["target"] >= 0) & (
        (lab["target"] != o["tgt"]) | (o["skill"] != SC.ATTACK))
    k = fx.row_of_slot(torch.where(att, lab["target"], torch.full_like(lab["target"], -1)))
    kc_ = k.clamp(min=0)
    onehot = torch.nn.functional.one_hot(kc_, K).bool() & att[..., None]             # [R, G, K]
    touch = getattr(teacher, "touch", None)
    pin = PN.pinned(fx, touch=touch)
    mt = fx.mt
    dist, gap = fx.g_dist, fx.g_gap
    valid = fx.valid
    E = valid[:, None, :].expand(R, G, K)
    cause = E & (pin & (mt >= 2))[:, None, :] & opn & (dist <= reach)
    # my unit's own facts
    sten = fx.enemy & ~fx.r
    g_near = torch.where(sten[:, None, :], gap, torch.full_like(gap, OR.INF)).amin(2)       # [R, G]
    come = PN.come_own(fx, getattr(teacher, "come_free", False), getattr(teacher, "come_reach", None))
    near_e = g_near <= SC.STICK_M
    free = fx.free & able & ~melee
    cm = OR.comes(fx)                                                              # [R, i, j] j comes at i
    cmg = cm.gather(1, fx.g_row[:, :, None].expand(-1, -1, K))                     # [R, G, K] k comes at my g
    unheld = valid & ~fx.m & (w.count == 0)
    own_reach = SC.CHARGE_M + fx.run * SC.LEAD_S                                   # [R, K] by the enemy's run
    cr = getattr(teacher, "come_reach", None)
    if cr is not None:
        own_reach = own_reach.clamp(min=float(cr))
    kinds = {}
    dck = torch.zeros((R, G, K), dtype=torch.bool, device=dev)
    # n1 / n4 / n3 / n7
    n1k = E & (pin & (mt == 1))[:, None, :] & opn & (dist <= reach)
    n4k = E & (pin & (mt >= 2))[:, None, :] & opn & (dist >= kc["n4_m"][0]) & (dist <= kc["n4_m"][1])
    n7 = cause.any(2) & (near_e | come)
    n3k = cause & ~onehot & (free & ~come & ~near_e)[..., None]
    # n6: an unheld enemy coming at g beyond the own reach, within 150 m (edge gap)
    n6k = unheld[:, None, :] & cmg & (gap <= reach)                               # (its own key: don't-care)
    # n2 / n2w: an unheld enemy within reach of g comes at another unit u of mine not in melee
    mine_idle = fx.mine & ~fx.r & ~fx.m                                            # [R, A]
    hold_m0 = fx.mine & ~fx.r & fx.m
    at_u = cm & mine_idle[:, :, None] & fx.enemy[:, None, :] & ~fx.r[:, None, :]  # [R, A, K]
    gq = torch.where(at_u, fx.gap, torch.full_like(fx.gap, OR.INF))
    nearest_u = at_u & (gq <= gq.amin(2, keepdim=True))
    other = torch.ones((R, G, K), dtype=torch.bool, device=dev)                    # [R, G, A] a is not g
    other.scatter_(2, fx.g_row[..., None], False)
    far_ok = unheld[:, None, :] & (dist <= reach)                                  # [R, G, K]
    n2k = far_ok & ((at_u & ~nearest_u)[:, None, :, :] & other[..., None]).any(2)
    n2wk = far_ok & ((nearest_u)[:, None, :, :] & other[..., None]).any(2) & ~n2k
    # n5: an unheld enemy within 150 m that stands, goes elsewhere (comes at nobody of mine) or chases a router
    comes_any = (cm & (fx.mine & ~fx.r)[:, :, None]).any(1)                        # [R, K]
    mine_any = fx.mine[:, None, :]
    gap_m = torch.where(mine_any, fx.gap, torch.full_like(fx.gap, OR.INF))
    router = fx.r.gather(1, gap_m.argmin(2))
    stands = fx.speed < SC.APPROACH_MS
    n5e = unheld & (stands | ~comes_any | ((fx.speed >= SC.APPROACH_MS) & router))
    n5k = far_ok & n5e[:, None, :] & ~n2k & ~n2wk
    # n9: within reach, two ticks in melee, not pinned (nobody of mine standing in melee within the touch, or moving)
    n9k = E & (fx.m & (mt >= 2) & ~pin)[:, None, :] & (dist <= reach)
    n9 = n9k.any(2)
    # nb: the old cause (b) - an unheld enemy within reach of g comes at a unit of mine in melee
    at_m = (cm & hold_m0[:, :, None]).any(1)                                       # [R, K]
    nbk = far_ok & at_m[:, None, :]
    # n9L: the enemy's only opponent is my lord (the simulator's contacts)
    n9Lk = torch.zeros_like(n9k)
    st = v.sim
    if st is not None:
        bi = v.idx                                                                 # (one group a battle: row = battle)
        hs = st.u["hit_sec"].view(st.B, st.N, st.N)[bi]
        tch = (hs > 0) | (hs.transpose(1, 2) > 0)                                  # [R, N, N]
        lord = (st.u["lord"] & (st.u["side"] == 1))[bi]                            # [R, N]
        slot = fx.slot.clamp(min=0)
        t_rows = tch.gather(1, slot[:, :, None].expand(-1, -1, st.N))              # [R, K, N]
        by_lord = (t_rows & lord[:, None, :]).any(2)
        mine_s = torch.zeros((R, st.N), dtype=torch.bool, device=dev)
        mine_s.scatter_(1, slot, fx.mine & ~fx.r)
        by_mine = (t_rows & mine_s[:, None, :]).any(2)
        n9Le = valid & fx.m & by_lord & ~by_mine
        n9Lk = E & n9Le[:, None, :] & (dist <= reach)
        n9 = (n9k & ~n9Lk).any(2)
    kinds.update(n1=n1k.any(2), n2=n2k.any(2), n2w=n2wk.any(2), n3=n3k.any(2), n4=n4k.any(2), n5=n5k.any(2),
                 n6=n6k.any(2), n7=n7, n9=n9, n9L=n9Lk.any(2), nb=nbk.any(2))
    for t in (n1k, n2wk, n3k, n4k, n5k, n6k, n9Lk):
        dck |= t
    # bands: keys, then the tick's own state
    hold_m = fx.mine & ~fx.r & fx.m
    gap_hold = torch.where(hold_m[:, :, None], fx.gap, torch.full_like(fx.gap, OR.INF)).amin(1)     # [R, K]
    key_band = (_within(dist, bands["reach_m"]) | (fx.m & _within(fx.speed, bands["speed_ms"]))[:, None, :]
                | (fx.m & _within(gap_hold, bands["touch_m"]))[:, None, :]) & E
    dck |= key_band
    own_b = float(bands["own_reach_m"])
    comer = unheld[:, None, :] & cmg if getattr(teacher, "come_free", False) else (fx.enemy & ~fx.r)[:, None, :] & cmg
    drop = _within(g_near, bands["near_m"]) | (comer & ((gap - own_reach[:, None, :]).abs() <= own_b)).any(2)
    # the attack labels' tags
    kk = kc_[..., None]
    g = lambda t: t.gather(2, kk)[..., 0]                                          # noqa: E731
    mt_e = mt.gather(1, kc_)
    # near2 (branches.FlagBranch's rule, the touch of stage 1f): an unheld enemy (not e) comes at a unit of mine in
    # melee touching e, or at g, edge gap within orders.near2_m
    n2m = float(oc["near2_m"])
    foe = unheld[:, None, :] & ~onehot                                              # [R, G, K]
    gap_ae = fx.gap.gather(2, kc_[:, None, :].expand(-1, K, -1)).transpose(1, 2)   # [R, G, A] gap(a, e of g)
    fight = hold_m[:, None, :] & (gap_ae <= SC.TOUCH_M)                            # my fighter touching e
    hit_f = ((cm & (fx.gap <= n2m))[:, None, :, :] & fight[..., None]).any(2)      # [R, G, K]
    hit_i = cmg & (gap <= n2m)
    near2 = ((hit_f | hit_i) & foe).any(2)
    near2_i = (hit_i & foe).any(2)
    own_far = (n6k & foe & (gap > own_reach[:, None, :])).any(2)
    thr = g(PN.through_tag(fx))
    dc_label = att & (thr | g(key_band))
    # health shares (check 8): the row's hp; the pinning unit - my unit in melee nearest e
    hp = fx.rr["rows"][..., RW.RI["hp"]]
    pin_row = torch.where(hold_m[:, :, None], fx.gap, torch.full_like(fx.gap, OR.INF)).argmin(1)          # [R, K]
    hp_pin_k = hp.gather(1, pin_row)                                                                      # [R, K]
    vd = torch.where(E, dist, torch.full_like(dist, OR.INF))
    nearest_e = att & (vd.argmin(2) == kc_)
    key_k = torch.where(cause | n9k, dist, torch.full_like(dist, OR.INF)).argmin(2)                        # [R, G]
    hp_ek, hp_pk = hp.gather(1, key_k), hp_pin_k.gather(1, key_k)
    tags = {"near2": near2 & att, "near2_at_i": near2_i & att, "mt": torch.where(att, mt_e, torch.full_like(mt_e, -1)),
            "fresh": att & (mt_e == 2), "late": att & (mt_e >= 3), "through_own": att & thr, "own_far": att & own_far,
            "dist": torch.where(att, g(dist), torch.full_like(g(dist), -1.0)),
            "side": torch.where(att, g(fx.g_side), torch.full_like(g(fx.g_side), -1)),
            "hp_e": hp.gather(1, kc_), "hp_pin": hp_pin_k.gather(1, kc_), "nearest_e": nearest_e,
            "hp_e_keep": hp_ek, "hp_pin_keep": hp_pk}
    # trivial: nobody of mine in melee within trivial_melee_m of g, no valid enemy within trivial_enemy_m of any free unit
    d_mm = fx.dist.gather(1, fx.g_row[:, :, None].expand(-1, -1, K))
    melee_near = (hold_m[:, None, :] & (d_mm <= float(kc["trivial_melee_m"]))).any(2)
    enemy_near = ((valid[:, None, :] & (dist <= float(kc["trivial_enemy_m"]))).any(2) & free).any(1, keepdim=True)
    trivial = ~melee_near & ~enemy_near
    dck &= ~onehot
    return {"head_free": head, "attack": att & head, "k": k, "trivial": trivial & head & ~att, "dropped": drop & head,
            "dc_label": dc_label & head, "kinds": {n: x & head & ~att for n, x in kinds.items()}, "tags": tags,
            "dc_keys": dck & head[..., None], "band_keys": key_band & ~onehot & head[..., None],
            "kind_keys": {"n2w": n2wk, "n4": n4k, "n5": n5k}}


class Record:
    """The recorder (design section 6 'learn.py' item 2, this slice's part): a wrapper player - the driver's output
    goes to the level; the labeller is asked on the same view (stateless: the order of the calls does not matter;
    built by the caller under its OWN name's options, never the driver's) and its answer tagged (tick). Per battle of
    the chunk (the Tacticians' row): counts of head-free unit-ticks by label, kind, band; the attack labels with their
    tags; at the battle's first tick the largest distance / turn from a unit to its first frame (the zone gate). With
    keep_raw, each head-free tick's raw, card and bounds (the View's) are kept too - the buffer's input (stage 4)."""
    COUNTS = ("head_free", "attack", "keep", "trivial", "dropped", "dc_label", "taught", "dc_key_keep", "hp_keep_n",
              "hp_e_keep", "hp_pin_keep", "band_key_keep") + KINDS

    def __init__(self, driver, labeller, keep_raw=False):
        self.main, self.lab = driver, labeller
        self.allow = getattr(driver, "allow", (SC.HOLD, SC.ATTACK))
        self.name = getattr(driver, "name", "driver")
        self.keep_raw = keep_raw
        self.cnt, self.labels, self.first, self.raw = {}, [], {}, []

    @property
    def last_tags(self):
        return self.main.last_tags

    def _first_frames(self, v, out):
        from net.unit import zmap as Z
        from net.tactic import raw as TR
        fr = out["frames"]
        mr = v.ctx.mine_raw
        t = v.tiles
        us, uf = Z.local(mr[..., TR.I["x"]], mr[..., TR.I["z"]], t.cx[:, None], t.cz[:, None], t.heading[:, None])
        on = fr["on"] & v.fx.able
        d = torch.where(on, torch.sqrt((fr["s"] - us) ** 2 + (fr["f"] - uf) ** 2), torch.zeros_like(us)).amax(1)
        turn = (torch.remainder(mr[..., TR.I["b"]] - t.heading[:, None] + 180.0, 360.0) - 180.0).abs()
        a = torch.where(on, turn, torch.zeros_like(turn)).amax(1)
        for b, x, y in zip(v.idx.tolist(), d.tolist(), a.tolist()):
            self.first.setdefault(b, (x, y))

    def __call__(self, v):
        out = self.main(v)
        lab = self.lab(v)
        if not self.first or any(b not in self.first for b in v.idx.tolist()):
            self._first_frames(v, out)
        T = tick(v, lab, self.lab)
        rows = v.idx.tolist()
        keep = T["head_free"] & ~T["attack"]
        taught = keep & ~T["dropped"] & torch.stack([T["kinds"][n] for n in TAUGHT], 0).any(0)
        hk = keep & ~T["dropped"] & (T["kinds"]["n7"] | T["kinds"]["n9"])
        parts = {"head_free": T["head_free"], "attack": T["attack"], "keep": keep, "trivial": T["trivial"],
                 "dropped": T["dropped"], "dc_label": T["dc_label"], "taught": taught,
                 "dc_key_keep": keep & T["dc_keys"].any(2), "hp_keep_n": hk,
                 "hp_e_keep": T["tags"]["hp_e_keep"] * hk, "hp_pin_keep": T["tags"]["hp_pin_keep"] * hk,
                 "band_key_keep": keep & T["band_keys"].any(2), **T["kinds"]}
        sums = torch.stack([parts[n].float().sum(1) for n in self.COUNTS], 1).tolist()
        for r, b in enumerate(rows):
            c = self.cnt.setdefault(b, [0] * len(self.COUNTS))
            for j, x in enumerate(sums[r]):
                c[j] += x
        now = int(round(float(v.now)))
        if bool(T["attack"].any()):
            tg = T["tags"]
            cols = {k: tg[k] for k in ("near2", "near2_at_i", "mt", "fresh", "late", "through_own", "own_far", "dist",
                                       "side", "hp_e", "hp_pin", "nearest_e")}
            for r, g in T["attack"].nonzero().tolist():
                self.labels.append({"b": rows[r], "sec": now, "slot": int(v.fx.my[r, g]),
                                    "target": int(lab["target"][r, g]), "dc_label": bool(T["dc_label"][r, g]),
                                    "dropped": bool(T["dropped"][r, g]),
                                    **{k: (float(x[r, g]) if x.dtype.is_floating_point else int(x[r, g]))
                                       for k, x in cols.items()}})
        if self.keep_raw and bool(T["head_free"].any()):
            sel = T["head_free"].any(1)
            self.raw.append({"sec": now, "b": [b for b, s in zip(rows, sel.tolist()) if s],
                             "raw": None if v.raw is None else v.raw[sel].cpu(),
                             "card": None if v.card is None else v.card[sel].cpu(), "bounds": v.bounds})
        return out
