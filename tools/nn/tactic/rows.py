"""The tactician's rows of units (design/tactic "Список карточек отрядов", "Живая часть", "Пометка «кто это»",
"Состояние", "Текущие характеристики", "Арсенал"; the user's decisions 30-31.10).

One list of rows; each row one unit, ROW = 185 numbers:
    passport 69         tools/nn/model/passport.py features (the same for own units and enemies)
    live 22             as the unit level's row (tools/nn/unit/view.py _row: the same formulas and scales) but in the
                        ZONE's axes - the zone's centre, "up" its turn (not a unit's): position right / forward of the
                        zone's centre / 100 m, its front against "up" (cos, sin), its velocity right / forward / 5 m/s,
                        front and depth / 50 m, men, health, morale, its state / 7, the 7 flags, fatigue / 5,
                        ammunition, the ground under its centre less the ground under the zone's centre / 20 m (+-4)
    who 5               me (always 0 at the tactician) / own in my group / own other / enemy - my target / enemy other:
                        put by the CODE from the parcel's masks, never by the network
    effects 25 x 2      per effect (tactic/raw.py EFFECTS): on (0 / 1) and the share of its time left (own units;
                        effects without a duration 0). An enemy: `on` of the 23 the player sees, never the seconds
    stats now 15        the passport's number with every effect, own button and received ability on (the same scale)
    arsenal 8 x 3       per button: has it, the share of its recharge left, the share of its active time left; an
                        enemy: has it and active now (0 / 1), never the recharge
An enemy's row fills 150 of the 185 (ENEMY_SEES); the rest is zeros by construction.

The rows' zone of attention (select): my group's units (alive, routing too) and the target's (alive and seen) always;
the others - a live own unit, or an enemy only while seen - when the distance between its formation's edge and the
zone's edge is within max(its range, the group's longest) + (its run + the group's fastest run) x reach_s
(config/nn/tactic.json, 30 s); at most K rows: mine, the target, then the others by that distance, enemies before own
units. A hidden enemy (the target's too) has no row - and nothing of it reaches the input. Rows not used: zeros under a
false mask. Identity by slot: row i carries the battle's slot (slot [R, K]); the analysis keeps it row by row.
"""
import math
from functools import lru_cache

import numpy as np
import torch

from tools.nn.model.passport import SIZE as PASSPORT
from tools.nn.tactic import cfg
from tools.nn.tactic import raw as TR
from tools.nn.unit import zmap as Z

LIVE = ("right", "forward", "front_cos", "front_sin", "v_right", "v_forward", "front", "depth", "men", "hp", "mp",
        "ms", "r", "s", "w", "m", "mv", "f", "fire", "fat", "ammo", "dh")
WHO = ("self", "own_group", "own_other", "enemy_target", "enemy_other")
SELF, OWN_GROUP, OWN_OTHER, ENEMY_TARGET, ENEMY_OTHER = range(5)
NONE = -1
# the stats now: name, the passport's column (passport.features), its scale ("lin", k: v / k; "log", top:
# log1p(v) / log1p(top)) - the passport's own (design/tactic "Текущие характеристики")
CSTATS = (("speed", 4, "lin", 5.0), ("charge_speed", 6, "lin", 10.0), ("attack", 7, "lin", 100.0),
          ("defence", 8, "lin", 100.0), ("leadership", 18, "lin", 100.0), ("charge_bonus", 9, "lin", 100.0),
          ("damage", 10, "log", 500.0), ("ap", 11, "log", 500.0), ("resist_physical", 20, "lin", 100.0),
          ("resist_flame", 23, "lin", 100.0), ("shield", 17, "lin", 100.0), ("accuracy", 34, "lin", 100.0),
          ("reload", 33, "lin", 20.0), ("m_damage", 31, "log", 500.0), ("m_ap", 32, "log", 500.0))
C = len(CSTATS)
CI = {c[0]: i for i, c in enumerate(CSTATS)}
DB_STAT = {("scalar_speed", "mult"): "speed", ("scalar_charge_speed", "mult"): "charge_speed",
           ("stat_melee_attack", "add"): "attack", ("stat_melee_defence", "add"): "defence",
           ("stat_morale", "add"): "leadership", ("stat_charge_bonus", "mult"): "charge_bonus",
           ("stat_melee_damage_base", "mult"): "damage", ("stat_melee_damage_ap", "mult"): "ap",
           ("stat_resistance_physical", "add"): "resist_physical", ("stat_weakness_flame", "add"): "resist_flame",
           ("stat_resistance_flame", "add"): "resist_flame", ("stat_missile_block_chance", "add"): "shield",
           ("stat_accuracy", "add"): "accuracy", ("stat_reloading", "add"): "reload",
           ("scalar_missile_damage_base", "mult"): "m_damage", ("scalar_missile_damage_ap", "mult"): "m_ap"}
RELOAD_MIN_S = 4.0     # the game's floor of a reload cut by the reload skill (tools/nn/sim/effects.py)
RESIST_CAP = 0.9       # resistances add up to 90 %
E, A = TR.E, TR.A
ROW = (tuple(f"p{k}" for k in range(PASSPORT)) + LIVE + WHO + tuple(f"{e[0]}_{k}" for e in TR.EFFECTS for k in ("on", "left"))
       + tuple(f"now_{c[0]}" for c in CSTATS) + tuple(f"{k}_{a}" for a in range(A) for k in ("has", "cd", "on")))
RI = {k: i for i, k in enumerate(ROW)}
assert len(ROW) == 185, len(ROW)


def _enemy_sees():
    m = np.ones(len(ROW), bool)
    for name, _, _, seen in TR.EFFECTS:
        m[RI[f"{name}_left"]] = False
        if not seen:
            m[RI[f"{name}_on"]] = False
    for a in range(A):
        m[RI[f"cd_{a}"]] = False
    return m


ENEMY_SEES = _enemy_sees()            # [185] the columns an enemy's row may fill (150)
assert int(ENEMY_SEES.sum()) == 150, int(ENEMY_SEES.sum())


@lru_cache(maxsize=1)
def stat_sources():
    """(logmult [S, C], add [S, C]) of every source of a stat change: the effects' own (self / hit; "phase" only - a
    "received" effect is its ability's received row), the 8 buttons' own (self), the received abilities (friends /
    enemies, raw.ability_keys order). An addition in the stat's own units; a source with scalar_speed and no
    scalar_charge_speed scales the charge speed too (sim/effects.py)."""
    fx, ab, _ = TR.catalogues()
    keys = TR.ability_keys()
    S = E + A + len(keys)
    lm, ad = np.zeros((S, C), np.float32), np.zeros((S, C), np.float32)

    def put(s, stats, groups):
        speed = charge = None
        for e in stats:
            if not set(e.get("on") or ()) & groups:
                continue
            c = DB_STAT.get((e["stat"], e["how"]))
            if c is None:
                continue
            v = float(e["value"])
            if e["how"] == "mult":
                lm[s, CI[c]] += math.log(max(v, 1e-6))
                if c == "speed":
                    speed = math.log(max(v, 1e-6))
                if c == "charge_speed":
                    charge = True
            else:
                ad[s, CI[c]] += v
        if speed is not None and charge is None:
            lm[s, CI["charge_speed"]] += speed
    for i, (name, src, _, _) in enumerate(TR.EFFECTS):
        if src == "phase":
            put(i, (fx["effects"].get(TR.PHASE_KEY[name]) or {}).get("stats") or (), {"self", "hit"})
    for a, k in enumerate(TR.ARSENAL):
        put(E + a, (ab.get(k) or {}).get("effects") or (), {"self"})
    for r, k in enumerate(keys):
        put(E + A + r, (ab.get(k) or {}).get("effects") or (), {"friends", "enemies"})
    return lm, ad


_DEV = {}


def _on_device(dev):
    key = str(dev)
    if key not in _DEV:
        lm, ad = stat_sources()
        act, full = TR.arsenal_times()
        cols = torch.tensor([c[1] for c in CSTATS], device=dev)
        scale = torch.tensor([c[3] for c in CSTATS], device=dev)
        log = torch.tensor([c[2] == "log" for c in CSTATS], device=dev)
        bits = torch.tensor([1 << r for r in range(len(TR.ability_keys()))], dtype=torch.int64, device=dev)
        dur = torch.tensor([e[2] or 0.0 for e in TR.EFFECTS], device=dev)
        seen = torch.tensor([e[3] for e in TR.EFFECTS], device=dev)
        _DEV[key] = {"lm": torch.tensor(lm, device=dev), "ad": torch.tensor(ad, device=dev), "cols": cols,
                     "scale": scale, "log": log, "act": torch.tensor(act, device=dev),
                     "full": torch.tensor(full, device=dev), "bits": bits, "dur": dur, "seen": seen,
                     "abits": torch.tensor([1 << a for a in range(A)], dtype=torch.int64, device=dev),
                     "enemy_sees": torch.tensor(ENEMY_SEES, device=dev)}
    return _DEV[key]


def stats_now(card, eff_on, a_active, rx):
    """[..., C] the stats now on the passport's scale: card [..., passport.SIZE], eff_on [..., E] bool (the row's
    effects on), a_active [..., A] bool (own buttons active; the General's aura: owned), rx [...] (the received
    abilities' bitmask, float)."""
    d = _on_device(card.device)
    rxb = (rx.long()[..., None] & d["bits"]) != 0
    on = torch.cat([eff_on.float(), a_active.float(), rxb.float()], -1)
    mult = torch.exp(on @ d["lm"])
    add = on @ d["ad"]
    base = card[..., d["cols"]]
    L = torch.log1p(d["scale"])
    raw = torch.where(d["log"], torch.expm1(base * L), base * d["scale"])
    now = raw * mult + add
    rc, sk = CI["reload"], add[..., CI["reload"]]
    rl = raw[..., rc]
    now[..., rc] = torch.where((sk != 0) & (rl > 0), (rl * (1 - sk / 100)).clamp(min=RELOAD_MIN_S), rl)
    for c, lo, hi in (("shield", 0, 100), ("accuracy", 0, 100), ("resist_physical", -1e9, 100 * RESIST_CAP)):
        now[..., CI[c]] = now[..., CI[c]].clamp(lo, hi)
    for c in ("speed", "charge_speed", "damage", "ap", "m_damage", "m_ap", "charge_bonus"):
        now[..., CI[c]] = now[..., CI[c]].clamp(min=0)
    return torch.where(d["log"], torch.log1p(now.clamp(min=0)) / L, now / d["scale"])


def row(j, card, who, cx, cz, heading, h0):
    """[..., 185]: unit j's tactician raw state [..., RAW] in the zone's frame (cx, cz, heading degrees, h0 the ground
    under the zone's centre; broadcast), its passport card [..., 69], who [...] (WHO index)."""
    c = cfg()
    I = TR.I
    pos, spd, size = float(c["pos_m"]), float(c["speed_ms"]), float(c["size_m"])
    s, f = Z.local(j[..., I["x"]], j[..., I["z"]], cx, cz, heading)
    rel = (j[..., I["b"]] - heading) * TR.DEG
    h = torch.as_tensor(heading, dtype=j.dtype, device=j.device) * TR.DEG
    co, sn = torch.cos(h), torch.sin(h)
    vx, vz = j[..., I["vx"]], j[..., I["vz"]]
    live = [s / pos, f / pos, torch.cos(rel), torch.sin(rel), (vx * co - vz * sn) / spd, (vx * sn + vz * co) / spd,
            j[..., I["front"]] / size, j[..., I["depth"]] / size, j[..., I["men"]], j[..., I["hp"]], j[..., I["mp"]],
            j[..., I["ms"]] / 7] + [j[..., I[k]] for k in ("r", "s", "w", "m", "mv", "f", "fire")] + [
            j[..., I["fat"]] / 5, j[..., I["ammo"]],
            ((j[..., I["gh"]] - h0) / float(c["height_m"])).clamp(-float(c["height_cap"]), float(c["height_cap"]))]
    enemy = (who == ENEMY_TARGET) | (who == ENEMY_OTHER)
    mine = (~enemy).float()
    hot = [(who == k).float() for k in range(len(WHO))]
    d = _on_device(j.device)
    e0 = I["e_" + TR.EFFECTS[0][0]]
    clk = j[..., e0:e0 + E]
    on = (clk > 0) & (d["seen"] | ~enemy[..., None])
    dur = d["dur"]
    left = torch.where(dur > 0, (1 - clk / dur.clamp(min=1e-6)).clamp(0, 1), torch.zeros_like(clk)) * on * mine[..., None]
    eff = torch.stack([on.float(), left], -1).flatten(-2)
    has = (j[..., I["a_has"]].long()[..., None] & d["abits"]) != 0
    act, full = d["act"], d["full"]
    a_on = j[..., I["a_on0"]:I["a_on0"] + A]
    a_cd = j[..., I["a_cd0"]:I["a_cd0"] + A]
    active = (a_on > 0) & has
    passive = (full <= 0) & has                             # (the General's aura: owned, always on)
    on_share = torch.where(enemy[..., None], active.float(),
                           torch.where(act > 0, (a_on / act.clamp(min=1e-6)).clamp(0, 1), torch.zeros_like(a_on)) * has)
    cd_share = torch.where(full > 0, (a_cd / full.clamp(min=1e-6)).clamp(0, 1), torch.zeros_like(a_cd)) * has * mine[..., None]
    ars = torch.stack([has.float(), cd_share, on_share], -1).flatten(-2)
    now = stats_now(card, on, active | passive, j[..., I["rx"]])
    out = torch.cat([card, torch.stack(live + hot, -1), eff, now, ars], -1)
    return torch.where(enemy[..., None] & ~d["enemy_sees"], torch.zeros((), device=j.device), out)


def zone_rect(tiles):
    """{x, z, b, front, depth} [R] the zone's rectangle (the full grid of tiles)."""
    return {"x": tiles.cx, "z": tiles.cz, "b": tiles.heading, "front": tiles.front, "depth": tiles.depth}


def edge_gap(S, rect):
    """[R, N] the signed distance between each slot's formation (S [R, N, RAW]) and the zone's rectangle rect [R]
    (geometry.rect_gap: edge to edge; negative inside)."""
    from tools.nn.sim import geometry
    I = TR.I
    cols = [torch.cat([S[..., I[k]], rect[k2][:, None]], 1) for k, k2 in
            (("x", "x"), ("z", "z"), ("b", "b"), ("front", "front"), ("depth", "depth"))]
    g = geometry.rect_gap(*cols)
    return g[:, -1, :-1]


def select(raw, parcel, tiles, rows=None, reach_s=None):
    """(slot [R, K] long (-1 empty), who [R, K] long (NONE empty), gap [R, N]) of each group's rows (module doc)."""
    c = cfg()
    K = int(rows or c["rows"])
    reach_s = float(c["reach_s"] if reach_s is None else reach_s)
    I = TR.I
    S = raw[parcel.bi]                                        # [R, N, RAW]
    R, N = S.shape[:2]
    dev = S.device
    alive = (S[..., I["alive"]] > 0) & (S[..., I["side"]] > 0)
    side = S[..., I["side"]]
    mine = parcel.mine & alive
    my_side = torch.where(parcel.mine, side, torch.zeros_like(side)).amax(1)
    enemy = (side != my_side[:, None]) & (side > 0)
    seen = S[..., I["vis"]] > 0
    target = parcel.target & alive & enemy & seen
    gap = edge_gap(S, zone_rect(tiles))
    g_range = torch.where(mine, S[..., I["range"]], torch.zeros_like(gap)).amax(1)
    g_run = torch.where(mine, S[..., I["run"]], torch.zeros_like(gap)).amax(1)
    reach = torch.maximum(S[..., I["range"]], g_range[:, None]) + (S[..., I["run"]] + g_run[:, None]) * reach_s
    other = alive & ~mine & ~target & (~enemy | seen) & (gap <= reach)
    # (float64: the tiers' offsets keep a slot's or a metre's difference exact)
    slot = torch.arange(N, device=dev, dtype=torch.float64)[None].expand(R, N)
    g = gap.double().clamp(-1e6, 1e6)
    key = torch.full((R, N), float("inf"), device=dev, dtype=torch.float64)
    key = torch.where(other & ~enemy, 2e7 + g, key)
    key = torch.where(other & enemy, 1e7 + g, key)
    key = torch.where(target, -2e7 + slot, key)
    key = torch.where(mine, -3e7 + slot, key)
    m = min(K, N)
    best = key.topk(m, dim=1, largest=False)
    order = best.values.argsort(1)
    pick = best.indices.gather(1, order)
    ok = best.values.gather(1, order) < float("inf")
    sl = torch.where(ok, pick, torch.full_like(pick, -1))
    if m < K:
        sl = torch.cat([sl, torch.full((R, K - m), -1, dtype=torch.long, device=dev)], 1)
    s0 = sl.clamp(min=0)
    who = torch.where(mine.gather(1, s0), OWN_GROUP,
                      torch.where(target.gather(1, s0), ENEMY_TARGET,
                                  torch.where(enemy.gather(1, s0), ENEMY_OTHER, OWN_OTHER)))
    who = torch.where(sl >= 0, who, torch.full_like(who, NONE))
    return sl, who, gap


def zone_ground(raw, parcel):
    """[R] the ground under each zone's centre (0 on a flat battle)."""
    from tools.nn.sim import terrain
    g = {k: v[parcel.bi] for k, v in TR.ground_of(raw).items()}
    return terrain.height(g, parcel.cx[:, None], parcel.cz[:, None])[:, 0]


def build(raw, card, parcel, tiles, rows=None, reach_s=None):
    """The rows of R groups: raw [B, N, RAW] (tactic.raw), card [B, N, 69] (passports), parcel (R groups), tiles
    (zone.Tiles of the groups). Returns {rows [R, K, 185], mask [R, K], slot [R, K], who [R, K], J [R, K, RAW] (the
    rows' raw state, for the map), h0 [R]}."""
    slot, who, _ = select(raw, parcel, tiles, rows, reach_s)
    S = raw[parcel.bi]
    R, K = slot.shape
    s0 = slot.clamp(min=0)
    J = S.gather(1, s0[..., None].expand(R, K, S.shape[-1]))
    cd = card[parcel.bi].gather(1, s0[..., None].expand(R, K, card.shape[-1]))
    mask = slot >= 0
    h0 = zone_ground(raw, parcel)
    sh = (R, 1)
    out = row(J, cd, who, parcel.cx.view(sh), parcel.cz.view(sh), parcel.heading().view(sh), h0.view(sh))
    zero = torch.zeros((), device=raw.device)
    return {"rows": torch.where(mask[..., None], out, zero), "mask": mask, "slot": slot, "who": who,
            "J": torch.where(mask[..., None], J, zero), "h0": h0}
