"""The tactician's battle in the simulator: the tactician decides every decide_s (5 s) in slot order, the frozen unit
network (unit.train.restore) carries its frames out, one decision a second with the game's latency and the leash.

    play(setup, ...)  one batch played to its end. Who decides: a TacticNet (sampling, or its best), or the teacher's
                      formula (teacher.drive: the exam's yardstick). Stored per tactician's decision: the tactician's raw
                      state, the tiles left, the last output's frames and the choices - rebuilt in the update (as at the
                      unit: only raw numbers are kept)
                      The tactician's choice per unit: keep its frame or change it (decide.py); a noticeable change it chose
                      costs reward.change (a change the code forced costs nothing)
    Посылка отряду    unit_area(): the frame, the pace of the order as it is, the stage by code (going until its centre is
                      inside the frame, in place from then on, going again on a new frame), no target, a shooter's fire
                      sector by code forward along the zone (else it obeys the infantry's rule, unit.view.allowed, and hardly
                      fires)

The reward (config reward): at the contact + if the infantry stands between the enemy and the shooters, - if the enemy
went straight into the shooters; at the end the trade in gold; a small cost per noticeable frame change.
The input stays one to one: the parcel and the order come from above, the bridge gains no field.
"""
import time

import torch

from tools.nn.sim import battle
from tools.nn.sim import orders as O
from tools.nn.tactic import cfg
from tools.nn.tactic import decide as DC
from tools.nn.tactic import drill1 as DR
from tools.nn.tactic import inputs as IN
from tools.nn.tactic import parcel as PC
from tools.nn.tactic import raw as TR
from tools.nn.tactic import teacher as TC
from tools.nn.tactic import uzone as U
from tools.nn.tactic import zone as ZN
from tools.nn.train import cadence, reward
from tools.nn.unit import decode as Dc
from tools.nn.unit import net as Nt
from tools.nn.unit import train as UT
from tools.nn.unit import view as V

UNIT_CFG = {"cell_m": 8.0, "bounds": None}


def unit_area(area, frames, tiles, my, shooter):
    """Writes the frames [R = B, G] (uzone.Seq.frames) into the units' area {x, z, b, front, depth, on, sb, sw} [B, N] in
    place; only the units with a frame. shooter [B, N]: a fire sector forward along the zone."""
    w = U.world(frames, tiles)
    R, G = frames["on"].shape
    k = torch.arange(R, device=my.device)
    sector = float(cfg()["sector_deg"])
    for j in range(G):
        on, sl = frames["on"][:, j], my[:, j].clamp(min=0)
        for key in ("x", "z", "b", "front", "depth"):
            area[key][k, sl] = torch.where(on, w[key][:, j], area[key][k, sl])
        area["sb"][k, sl] = torch.where(on, tiles.heading, area["sb"][k, sl])
        area["sw"][k, sl] = torch.where(on & shooter[k, sl], torch.full_like(tiles.heading, sector), area["sw"][k, sl])


def fresh_area(B, N, own, device):
    a = {k: torch.zeros((B, N), device=device) for k in ("x", "z", "b", "front", "depth", "sb", "sw")}
    a["on"] = own.clone()
    return a


def rows_of(N):
    return min(int(cfg()["rows"]), N)


def frames_input(frames, tiles):
    """The personal areas' layer of the near window: the last output's frames in world coordinates (None: none yet)."""
    return None if frames is None else U.world(frames, tiles)


class Setup:
    """One batch of first-drill battles."""

    def __init__(self, sp, P, device, check=True):
        self.sp = sp
        self.P = P
        self.device = device
        self.st, self.parcel, self.order, self.own, self.shooter, self.foe = DR.build(sp, P, device, check)
        self.card = UT.cards(self.st, device).float()
        self.bi, self.ii = self.own.nonzero(as_tuple=True)
        self.my = DC.my_slots(self.parcel)


class TileKeeper:
    """The tiles of R groups between decisions (zone.Keeper's rule, batched): a unit that falls out (dead or routing) takes
    away the tile nearest its last place, once. One thing more than zone.Keeper: a unit that rallies brings its tile back
    (a new unit in the group, in the page's words): otherwise the group can hold more units than tiles and no zone fits."""

    def __init__(self, parcel, my):
        self.tiles = ZN.Tiles.of(parcel)
        R, G = my.shape
        self.parcel, self.my = parcel, my
        self.out = torch.zeros((R, G), dtype=torch.bool, device=my.device)
        self.last = torch.zeros((R, G, 2), device=my.device)
        self.tile = torch.zeros((R, G), dtype=torch.long, device=my.device)

    def update(self, raw):
        I = TR.I
        my, t = self.my, self.tiles
        k = torch.arange(my.shape[0], device=my.device)
        for j in range(my.shape[1]):
            row = raw[self.parcel.bi, my[:, j].clamp(min=0)]
            able = (row[:, I["alive"]] > 0) & ~(row[:, I["r"]] > 0) & (my[:, j] >= 0)
            self.last[k, j, 0] = torch.where(able, row[:, I["x"]], self.last[k, j, 0])
            self.last[k, j, 1] = torch.where(able, row[:, I["z"]], self.last[k, j, 1])
            new = ~able & ~self.out[:, j] & (my[:, j] >= 0)
            if bool(new.any()):
                before = t.mask.clone()
                t.drop(new, self.last[:, j, 0], self.last[:, j, 1])
                gone = before & ~t.mask
                self.tile[:, j] = torch.where(new, gone.float().argmax(1), self.tile[:, j])
                self.out[:, j] = self.out[:, j] | new
            back = able & self.out[:, j]
            if bool(back.any()):
                m = t.mask.clone()
                m[k[back], self.tile[back, j]] = True
                t.mask = m
                self.out[:, j] = self.out[:, j] & ~back
        return t


@torch.no_grad()
def play(setup, unit_net, net=None, critic=None, greedy=False, formula=False, keep=True, duty=None, unit_greedy=False):
    """One batch to its end. net: a TacticNet, or formula=True: the teacher's formula. Returns (batch, stats); batch is
    what update needs (keep=False: the exam, no stores)."""
    st, P, dev = setup.st, setup.P, setup.device
    B, N = st.B, st.N
    c = cfg()
    rw = c["reward"]
    decide_every = int(c["decide_s"])
    limit = int(c["drill1"]["limit_s"])
    Dmax = limit // decide_every + 1
    spacing = P.sim["formation"]["spacing_m"]
    steps = round(1.0 / P.dt)
    parcel, order, own, shooter, card, my = setup.parcel, setup.order, setup.own, setup.shooter, setup.card, setup.my
    bi, ii = setup.bi, setup.ii
    G = my.shape[1]
    keeper = TileKeeper(parcel, my)
    tiles = keeper.tiles
    seer = TR.Seer(P)
    script = DR.Script()
    ender = DR.Ender(st, own)
    watch = DR.Watch(st, own, shooter, spacing, dev)
    cad = cadence.Cadence()
    leash = Dc.Leash(B, N, dev)
    wlim = Dc.width_limits(st, P)
    area = fresh_area(B, N, own, dev)
    arrived = torch.zeros((B, N), dtype=torch.bool, device=dev)
    pace = torch.full((B,), Nt.PACE_ANY, dtype=torch.long, device=dev)   # (the order's pace as it is: 'any' here)
    T = V.T
    ubuf = torch.zeros((T + 1, B, N, len(V.RAW)), device=dev)
    in_force = O.hold(B, N, dev)
    sim_step = battle.stepper(dev)
    m0 = reward.measure(st)
    K = rows_of(N)
    if keep:
        store = {"raw": torch.zeros((Dmax, B, N, len(TR.RAW)), device=dev),
                 "tiles": torch.zeros((Dmax, B, ZN.MAX_TILES), dtype=torch.bool, device=dev),
                 "prev": {k: torch.zeros((Dmax, B, G), device=dev) for k in ("s", "f", "hw", "hd")},
                 "prev_on": torch.zeros((Dmax, B, G), dtype=torch.bool, device=dev),
                 "has_prev": torch.zeros((Dmax, B), dtype=torch.bool, device=dev),
                 "act": {k: torch.zeros((Dmax, B, G), dtype=torch.long, device=dev) for k in ("keep", "width", "depth", "cell")}}
    lp_old = torch.zeros((Dmax, B), device=dev)
    val = torch.zeros((Dmax, B), device=dev)
    rew = torch.zeros((Dmax, B), device=dev)
    done = torch.zeros((Dmax, B), dtype=torch.bool, device=dev)
    valid = torch.zeros((Dmax, B), dtype=torch.bool, device=dev)
    frames = None
    stats = {"decisions": 0, "fell": 0.0, "frames_overlap": 0, "frames_out": 0, "changes": 0.0, "small_moves": 0.0,
             "chose": 0.0, "kept": 0.0, "forced": 0.0, "ent": {"keep": 0.0, "width": 0.0, "depth": 0.0, "where": 0.0}}
    d = -1
    t_tactic = t_units = t_sim = 0.0
    n_unit = 0
    r_acc = torch.zeros(B, device=dev)
    for sec in range(limit + 1):
        if bool(st.done.all()):
            break
        if duty is not None:
            duty.start()
        raw_t = seer.raw(st)
        if sec % decide_every == 0:
            if d >= 0:
                rew[d] = r_acc
                done[d] = valid[d] & st.done
                r_acc = torch.zeros(B, device=dev)
            d += 1
            t0 = time.time()
            keeper.update(raw_t)
            inp = IN.build(raw_t, card, parcel, tiles, frames_input(frames, tiles), bounds=st.bounds, rows=K)
            if formula:
                ctx = DC.ctx_of(inp, parcel, tiles)
                out = TC.drive(ctx.mine_raw, ctx.able, tiles)
            else:
                out, _, ctx = net(inp, order, parcel, tiles, prev=frames, greedy=greedy)
                lp_old[d] = out["lp"]
                if critic is not None:
                    val[d] = critic(inp, order)
                for k in stats["ent"]:
                    stats["ent"][k] += float(out["ent"][k][~st.done].sum())
            valid[d] = ~st.done & ctx.able.any(1)
            stats["decisions"] += int(valid[d].sum())
            stats["fell"] += float(out["fell"][valid[d]].sum())
            live_u = valid[d][:, None]
            free_u = ctx.able & ~out["must"] & live_u
            stats["chose"] += float(free_u.sum())
            stats["kept"] += float((out["kept"] & free_u).sum())
            stats["forced"] += float((out["forced"] & live_u & ~out["must"]).sum())
            ov, ou = U.bad_frames(out["frames"], tiles)
            stats["frames_overlap"] += int((ov & valid[d]).sum())
            stats["frames_out"] += int((ou & valid[d]).sum())
            if keep:
                store["raw"][d] = raw_t
                store["tiles"][d] = tiles.mask
                if frames is not None:
                    for k in store["prev"]:
                        store["prev"][k][d] = frames[k]
                    store["prev_on"][d] = frames["on"]
                    store["has_prev"][d] = True
                for k in store["act"]:
                    store["act"][k][d] = out[k]
            new = {k: v.clone() for k, v in out["frames"].items()}
            if frames is not None:
                ch = U.changed(new, frames)
                paid = ch & ~out["forced"] & ~out["must"]        # (only a change the network chose)
                r_acc = r_acc - float(rw["change"]) * paid.sum(1).float() * (~st.done).float()
                stats["changes"] += float((ch.sum(1).float() * (~st.done).float()).sum())
                moved = ((new["s"] - frames["s"]).abs() > 1e-3) | ((new["f"] - frames["f"]).abs() > 1e-3) | \
                        ((new["hw"] - frames["hw"]).abs() > 1e-3) | ((new["hd"] - frames["hd"]).abs() > 1e-3)
                stats["small_moves"] += float((moved & new["on"] & frames["on"] & ~ch & live_u).sum())
                k_ = my.clamp(min=0)
                arrived.scatter_(1, k_, arrived.gather(1, k_) & ~moved)
            # a unit without a frame now (routing) keeps its last one
            if frames is not None:
                for k in ("s", "f", "hw", "hd"):
                    new[k] = torch.where(new["on"], new[k], frames[k])
            unit_area(area, new, tiles, my, shooter)
            frames = new
            t_tactic += time.time() - t0
        # the units' second
        t0 = time.time()
        raw_u = V.raw(st, spacing, leash=leash.on, missile=P.sim["missile"])
        ubuf = torch.cat([ubuf[1:], raw_u[None]])
        u = st.u
        inside = Dc.signed_dist(u["x"], u["z"], {k: area[k] for k in ("x", "z", "b", "front", "depth")}) >= 0
        arrived = arrived | (inside & own)
        live = ~st.done[bi] & UT.viewer_ok(st, bi, ii)
        bl, il = bi[live], ii[live]
        intended = O.hold(B, N, dev)
        intended.kind[:] = O.KEEP
        if bl.numel():
            idx = torch.full_like(bl, T)
            v = UT.views(ubuf, idx, bl, il, card, area, UNIT_CFG, wlim=wlim)
            ord_u = Nt.order_of(raw_u[bl, il], {k: a[bl, il] for k, a in area.items() if k != "on"}, pace[bl],
                                arrived[bl, il])
            act = Nt.sample(unit_net(v, ord_u, v["cell_m"]), greedy=unit_greedy)
            intended = Dc.to_orders(act, v, raw_u, v["cell_m"], B, N, wlim=wlim)
            n_unit += int(bl.numel())
        lands = cad.delays(P.dt, B, dev)
        t_units += time.time() - t0
        t0 = time.time()
        for j in range(steps):
            was = st.done.clone()
            now_in = lands <= j
            mine = O.Orders(*(torch.where(now_in[:, None], getattr(intended, f), getattr(in_force, f)) for f in O.FIELDS))
            o = O.merge(script(st), leash.apply(st, mine, area, own), own)
            sim_step(st, o, P)
            st.winner = torch.where(st.done, st.winner, torch.zeros_like(st.winner))
            _, got, hit = watch.step(st, was, P.dt, area)
            finished = ender.after_step(st, watch.t_contact)
            r_acc = r_acc + float(rw["between"]) * got.float() - float(rw["hit"]) * hit.float()
            if bool(finished.any()):
                now = reward.measure(st)
                trade = (m0[:, 1, 3] - now[:, 1, 3]) - (m0[:, 0, 3] - now[:, 0, 3])
                r_acc = r_acc + float(rw["trade"]) * trade * finished.float()
        in_force = intended
        t_sim += time.time() - t0
        if duty is not None:
            duty.end()
    if d >= 0:
        rew[d] = r_acc
        done[d] = valid[d] & st.done
    D = d + 1
    m1 = reward.measure(st)
    trade = (m0[:, 1, 3] - m1[:, 1, 3]) - (m0[:, 0, 3] - m1[:, 0, 3])
    out_stats = dict(stats)
    out_stats.update({"battles": B, "t_tactic": round(t_tactic, 2), "t_units": round(t_units, 2), "t_sim": round(t_sim, 2),
                      "unit_decisions": n_unit, "seconds": sec, "decisions_mean": round(stats["decisions"] / B, 1),
                      "changes_per_battle": round(stats["changes"] / B, 2), "reward_mean": round(float(rew[:D].sum()) / B, 4),
                      "small_moves_per_battle": round(stats["small_moves"] / B, 2),
                      "keep_share": round(stats["kept"] / max(stats["chose"], 1.0), 3)})
    n = max(stats["decisions"], 1)
    out_stats["ent"] = {k: round(v / n, 3) for k, v in stats["ent"].items()}
    out_stats["trade"] = round(float(trade.mean()), 4)
    readings = {"contact": watch.contact, "between": watch.between, "hit": watch.hit, "melee": watch.melee,
                "fired_s": watch.fired_s, "t_contact": watch.t_contact, "fill": watch.fill, "trade": trade,
                "why": ender.why, "ammo_used": _ammo_used(st, shooter), "frames": frames}
    batch = None
    if keep:
        batch = {"raw": store["raw"][:D], "tiles": store["tiles"][:D],
                 "prev": {k: v[:D] for k, v in store["prev"].items()}, "prev_on": store["prev_on"][:D],
                 "has_prev": store["has_prev"][:D], "act": {k: v[:D] for k, v in store["act"].items()},
                 "lp": lp_old[:D], "value": val[:D], "reward": rew[:D], "done": done[:D], "valid": valid[:D],
                 "last_value": torch.zeros(B, device=dev), "card": card, "parcel": parcel, "order": order,
                 "bounds": st.bounds, "K": K, "G": G}
    return batch, out_stats, readings


def _ammo_used(st, shooter):
    """[B] the share of the shooters' ammunition spent."""
    u = st.u
    a0 = (u["ammo0"] * shooter).sum(1).clamp(min=1)
    return ((u["ammo0"] - u["a"]) * shooter).sum(1) / a0


def rebuild(batch, di, bi):
    """The inputs of the samples (decision di, battle bi) from the stored raw numbers: (inp, order, parcel, tiles, act,
    prev: the frames in force)."""
    p = batch["parcel"]
    M = bi.shape[0]
    dev = bi.device
    parcel = PC.Parcel(torch.arange(M, device=dev), p.mine[bi], p.target[bi], p.cx[bi], p.cz[bi], p.sin[bi], p.cos[bi],
                       p.line[bi], p.group[bi])
    tiles = ZN.Tiles.of(parcel)
    tiles.mask = batch["tiles"][di, bi] & tiles.mask
    raw = batch["raw"][di, bi]
    has = batch["has_prev"][di, bi]
    fr = {k: v[di, bi] for k, v in batch["prev"].items()}
    fr["on"] = batch["prev_on"][di, bi] & has[:, None]
    areas = None if not bool(has.any()) else U.world(fr, tiles)
    inp = IN.build(raw, batch["card"][bi], parcel, tiles, areas, bounds=batch["bounds"], rows=batch["K"])
    act = {k: v[di, bi] for k, v in batch["act"].items()}
    return inp, batch["order"][bi], parcel, tiles, act, fr
