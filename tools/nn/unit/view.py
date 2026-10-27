"""A unit's view: its zone, the list of units in it with their memory, and its zone map (build/diag_g49/plan_unit_net.md
section 3; unit_base.md: only the zone, coordinates relative to the unit, only raw data the game gives, never the
battle's time left).

Two steps, so that training stores only raw numbers (a stored view would be ~52 GB a batch, plan section 3):
    raw(st)                     [B, N, len(RAW)] each slot's raw state in the world's frame (what the game shows of it)
    build(now, hist, card, ...) the view of the chosen units from the raw state now and the last T snapshots (one a
                                second, newest first): rebuilt the same way when a decision is collected and in the update

The zone of unit i: itself, and every live unit j (an enemy only when visible) whose formation's edge is within
R(i, j) = the longer missile range of the two + (i's run + j's run) x REACH_S of i's edge (geometry.rect_gap: any part
of the formation counts). At most K rows: i first, then up to K - 1 of the zone nearest by that gap, enemies before
own units (each by the gap). A row always describes the whole formation, however little of it is in the zone.

A row (ROW, raw numbers in the viewer's frame): j's centre right / forward of i's (/ 100 m), j's front against i's
(cos, sin), j's velocity in i's frame (/ 5 m/s), front and depth (/ 50 m), men left (share of its start), health,
MoralePercent, MoraleState / 7, the flags (routing, shattered, wavering, in melee, moving, running, firing), fatigue
state / 5, projectiles left (share), whose (self / own / enemy), 'the target of my order' and (its own row only)
'my order is now replaced by the leash' (decode.py). The card (the unit's
passport, tools/nn/model/passport.py) is a separate input. Memory: the same row of the same unit from each of the T
past snapshots, put into i's frame of NOW (so the network sees how it moved); a snapshot where it was dead, gone or
(an enemy) unseen is empty. Rows not in the zone are zeros under a false mask - nothing of an enemy outside the zone
reaches the view (tests).
"""
import torch

from tools.nn.unit import zmap as Z

K = 16                 # rows: the unit itself + up to 15 others
T = 10                 # memory snapshots, one a second
REACH_S = 15.0         # s: the zone's time (the order's span) - a parameter
RAW = ("x", "z", "b", "vx", "vz", "front", "depth", "fill", "men", "hp", "mp", "ms", "r", "s", "w", "m", "mv", "f",
       "fire", "fat", "ammo", "alive", "vis", "side", "range", "run", "leash")
I = {k: i for i, k in enumerate(RAW)}
ROW = ("right", "forward", "front_cos", "front_sin", "v_right", "v_forward", "front", "depth", "men", "hp", "mp",
       "ms", "r", "s", "w", "m", "mv", "f", "fire", "fat", "ammo", "self", "own", "enemy", "my_target", "leashed")
DEG = torch.pi / 180
SELF, OWN, ENEMY = 0, 1, 2


def raw(st, spacing, melee_depth=0.0, leash=None):
    """[B, N, len(RAW)] the slots' raw state now (world frame; tools/nn/sim/state.py names). alive: present, men left,
    not gone; vis: visible to the other side; fill: the share of the formation's places filled (zmap.rects_of);
    leash [B, N] bool: the unit's order is now replaced by the leash (tools/nn/unit/decode.py; only its own row shows
    it)."""
    u = st.u
    r = Z.rects_of(u, spacing, melee_depth)
    men0 = u["men0"].clamp(min=1)
    ammo = torch.where(u["ammo0"] > 0, u["a"] / u["ammo0"].clamp(min=1), torch.zeros_like(u["a"]))
    alive = (u["side"] > 0) & (u["men"] > 0) & ~u["gone"]
    cols = {"x": u["x"], "z": u["z"], "b": u["b"], "vx": u["vx"], "vz": u["vz"], "front": r["front"],
            "depth": r["depth"], "fill": r["fill"], "men": u["men"] / men0, "hp": u["hp"], "mp": u["mp"], "ms": u["ms"],
            "r": u["r"], "s": u["s"], "w": u["w"], "m": u["m"], "mv": u["mv"], "f": u["f"], "fire": u["fire"],
            "fat": u["fat"], "ammo": ammo, "alive": alive, "vis": u["vis"], "side": u["side"], "range": u["range"],
            "run": u["run"], "leash": leash if leash is not None else torch.zeros_like(alive)}
    return torch.stack([cols[k].float() for k in RAW], -1)


class Memory:
    """The last T raw snapshots [T, B, N, len(RAW)], newest first (an empty one: alive 0). push() once a second."""

    def __init__(self, B, N, device="cpu", steps=T):
        self.hist = torch.zeros((steps, B, N, len(RAW)), device=device)

    def push(self, now):
        self.hist = torch.cat([now[None], self.hist[:-1]])
        return self.hist


def _gap(x, z, b, front, depth):
    from tools.nn.sim import geometry
    return geometry.rect_gap(x, z, b, front, depth)


def _row(j, me, whose, target):
    """[..., len(ROW)]: the raw state j [..., RAW] seen from me [..., RAW] (broadcast), whose [...] SELF / OWN / ENEMY,
    target [...] bool."""
    s, f = Z.local(j[..., I["x"]], j[..., I["z"]], me[..., I["x"]], me[..., I["z"]], me[..., I["b"]])
    rel = (j[..., I["b"]] - me[..., I["b"]]) * DEG
    h = me[..., I["b"]] * DEG
    c, sn = torch.cos(h), torch.sin(h)
    vx, vz = j[..., I["vx"]], j[..., I["vz"]]
    cols = [s / 100, f / 100, torch.cos(rel), torch.sin(rel), (vx * c - vz * sn) / 5, (vx * sn + vz * c) / 5,
            j[..., I["front"]] / 50, j[..., I["depth"]] / 50, j[..., I["men"]], j[..., I["hp"]], j[..., I["mp"]],
            j[..., I["ms"]] / 7] + [j[..., I[k]] for k in ("r", "s", "w", "m", "mv", "f", "fire")] + [
            j[..., I["fat"]] / 5, j[..., I["ammo"]],
            (whose == SELF).float(), (whose == OWN).float(), (whose == ENEMY).float(), target.float(),
            j[..., I["leash"]] * (whose == SELF).float()]
    return torch.stack(cols, -1)


def zone(now, bi, ii, reach_s=REACH_S, everyone=False):
    """For the viewers (bi, ii) [R]: in_zone [R, N] (itself included) and the edge gap [R, N]. everyone: every live
    unit (an enemy only seen) is in the zone - the critic's view of the whole skirmish."""
    S = now[bi]                                              # [R, N, RAW] the viewer's battle
    me = now[bi, ii]                                         # [R, RAW]
    gap = _gap(now[..., I["x"]], now[..., I["z"]], now[..., I["b"]], now[..., I["front"]], now[..., I["depth"]])
    gap = gap[bi, ii]                                        # [R, N]
    reach = torch.maximum(me[:, None, I["range"]], S[..., I["range"]]) + (me[:, None, I["run"]] + S[..., I["run"]]) * reach_s
    alive = S[..., I["alive"]] > 0
    enemy = S[..., I["side"]] != me[:, None, I["side"]]
    seen = ~enemy | (S[..., I["vis"]] > 0)
    N = now.shape[1]
    itself = torch.arange(N, device=now.device)[None] == ii[:, None]
    near = gap <= reach if not everyone else torch.ones_like(alive)
    return (alive & seen & near & (S[..., I["side"]] > 0)) | itself, gap


def build(now, hist, card, who, order_target=None, area=None, cell_m=8.0, bounds=None, rows=K, reach_s=REACH_S,
          everyone=False):
    """The views of the units who [B, N] (bool) wants, R of them in who.nonzero() order.

    now [B, N, RAW], hist [T, B, N, RAW] (Memory.hist: the past snapshots, newest first), card [B, N, P] (passports),
    order_target [B, N] long (the slot each unit's order attacks, -1 none), area: {x, z, b, front, depth[, on]} [B, N]
    each unit's area (a turned rectangle) or None, cell_m: the map's cell, m. Returns a dict:
        rows [R, K, ROW], card [R, K, P], mask [R, K] bool, slot [R, K] long (-1 empty), whose [R, K] long,
        mem [R, K, T, ROW], mem_mask [R, K, T], map [R, len(zmap.LAYERS), 30, 30],
        cells_ok [R, 30, 30] bool (a cell whose centre lies in the area and on the map: the 'where' head's allowed
        cells), target_ok [R, K] bool (an enemy row whose centre is in the area), bi, ii [R]."""
    dev = now.device
    bi, ii = who.nonzero(as_tuple=True)
    R, N = bi.shape[0], now.shape[1]
    inz, gap = zone(now, bi, ii, reach_s, everyone)
    S = now[bi]                                              # [R, N, RAW]
    me = now[bi, ii]
    enemy = S[..., I["side"]] != me[:, None, I["side"]]
    itself = torch.arange(N, device=dev)[None] == ii[:, None]
    # the K - 1 nearest others of the zone, then ordered: itself, enemies by the gap, own units by the gap
    big = torch.finfo(gap.dtype).max / 4
    key = torch.where(inz & ~itself, gap, torch.full_like(gap, big))
    m = min(rows - 1, N)
    near = key.topk(m, dim=1, largest=False)
    pick = near.indices                                      # [R, m]
    ok = near.values < big
    rank = torch.where(ok, enemy.gather(1, pick).logical_not().float() * 1e7 + near.values.clamp(-1e6, 1e6),
                       torch.full_like(near.values, 3e7))
    order = rank.argsort(1)
    pick, ok = pick.gather(1, order), ok.gather(1, order)
    slot = torch.cat([ii[:, None], torch.where(ok, pick, torch.full_like(pick, -1))], 1)       # [R, 1 + m]
    if slot.shape[1] < rows:
        slot = torch.cat([slot, torch.full((R, rows - slot.shape[1]), -1, device=dev, dtype=torch.long)], 1)
    mask = slot >= 0
    sl = slot.clamp(min=0)
    J = S.gather(1, sl[..., None].expand(R, rows, len(RAW)))               # [R, K, RAW]
    whose = torch.where(sl == ii[:, None], SELF, torch.where(enemy.gather(1, sl), ENEMY, OWN))
    whose = torch.where(mask, whose, torch.full_like(whose, -1))
    tgt = order_target[bi, ii] if order_target is not None else torch.full((R,), -1, device=dev, dtype=torch.long)
    is_t = mask & (sl == tgt[:, None]) & (tgt[:, None] >= 0)
    zero = torch.zeros((), device=dev)
    rows_ = torch.where(mask[..., None], _row(J, me[:, None], whose, is_t), zero)
    cards = torch.where(mask[..., None], card[bi].gather(1, sl[..., None].expand(R, rows, card.shape[-1])), zero)
    # memory: the same slots T seconds back, in the viewer's frame now
    H = hist[:, bi]                                          # [T, R, N, RAW]
    Tn = H.shape[0]
    Jh = H.gather(2, sl[None, ..., None].expand(Tn, R, rows, len(RAW))).permute(1, 2, 0, 3)     # [R, K, T, RAW]
    was = (Jh[..., I["alive"]] > 0) & ((whose[..., None] != ENEMY) | (Jh[..., I["vis"]] > 0)) & mask[..., None]
    mem = torch.where(was[..., None], _row(Jh, me[:, None, None], whose[..., None].expand(R, rows, Tn),
                                             is_t[..., None].expand(R, rows, Tn)), zero)
    # the map: the zone's rows only
    role = torch.where(~mask, Z.NONE, torch.where(whose == SELF, Z.SELF, torch.where(whose == ENEMY, Z.ENEMY, Z.OWN)))
    units = {"x": J[..., I["x"]], "z": J[..., I["z"]], "b": J[..., I["b"]], "front": J[..., I["front"]],
             "depth": J[..., I["depth"]], "fill": J[..., I["fill"]] * mask, "role": role}
    cm = torch.full((R,), float(cell_m), device=dev) if not torch.is_tensor(cell_m) else cell_m
    ar = None if area is None else {k: v[bi, ii] for k, v in area.items()}
    zm = Z.build(me[:, I["x"]], me[:, I["z"]], me[:, I["b"]], cm, units, area=ar, bounds=bounds)
    cells_ok, target_ok = allowed(me, J, mask & (whose == ENEMY), ar, cm, bounds)
    return {"rows": rows_, "card": cards, "mask": mask, "slot": torch.where(mask, slot, torch.full_like(slot, -1)),
            "whose": whose, "mem": mem, "mem_mask": was, "map": zm, "cells_ok": cells_ok, "target_ok": target_ok,
            "bi": bi, "ii": ii}


def _inside(x, z, ar):
    """Points (x, z) [R, ...] inside each viewer's area rectangle (ar [R] fields)."""
    sh = (-1,) + (1,) * (x.dim() - 1)
    s, f = Z.local(x, z, ar["x"].view(sh), ar["z"].view(sh), ar["b"].view(sh))
    return (s.abs() <= ar["front"].view(sh) / 2) & (f.abs() <= ar["depth"].view(sh) / 2)


def allowed(me, J, enemy_rows, ar, cell_m, bounds, cells=Z.CELLS):
    """The hard bans' masks: cells_ok [R, C, C] (the cell's centre in the area and on the map; no area: on the map; if
    no cell is left, the one nearest the area's centre) and target_ok [R, K] (an enemy row whose centre is in the area)."""
    R = me.shape[0]
    dev = me.device
    k = torch.arange(cells, device=dev, dtype=torch.float32)
    s, f = Z.cell_centre(k[:, None], k[None, :], cell_m.view(R, 1, 1), cells)          # [R, 1, C], [R, C, 1]
    s, f = s.expand(R, cells, cells), f.expand(R, cells, cells)
    x, z = Z.world(s, f, me[:, I["x"]].view(R, 1, 1), me[:, I["z"]].view(R, 1, 1), me[:, I["b"]].view(R, 1, 1))
    ok = torch.ones((R, cells, cells), dtype=torch.bool, device=dev)
    if bounds is not None:
        ok = ok & (x.abs() <= bounds) & (z.abs() <= bounds)
    t_ok = enemy_rows
    if ar is not None:
        on = ar.get("on", torch.ones(R, dtype=torch.bool, device=dev))
        ok = ok & (_inside(x, z, ar) | ~on.view(R, 1, 1))
        t_ok = t_ok & (_inside(J[..., I["x"]], J[..., I["z"]], ar) | ~on[:, None])
        # no cell left (an area smaller than a cell, or off the map): the cell nearest the area's centre
        none = ~ok.flatten(1).any(1)
        if bool(none.any()):
            d = (x - ar["x"].view(R, 1, 1)) ** 2 + (z - ar["z"].view(R, 1, 1)) ** 2
            best = d.flatten(1).argmin(1)
            one = torch.zeros_like(ok).flatten(1)
            one[torch.arange(R, device=dev), best] = True
            ok = torch.where(none.view(R, 1, 1), one.view(R, cells, cells), ok)
    return ok, t_ok
