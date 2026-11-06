"""The operative's whole input of B battles (plan.md "Вход"; the network's side: net.py):

    rows [B, K, 185], mask, slot, who [B, K]   every own alive unit (routing too) by slot, then every seen enemy by slot,
                                 K = 40 (20 + 20); the tactician's row formula (tactic/rows.row, not changed) with the
                                 origin at the unit itself (right / forward 0, the height 0) in the field's axes: the
                                 frozen 'Анализ отряда' never saw +-10. Own units are 'own in my group' (the operative
                                 commands them all), enemies 'enemy other'
    J [B, K, RAW]                the rows' raw state (for the map, the walk)
    place [B, K, 3]              frame.place: right, forward / 1024 m, the height / 20 m
    prev [B, K, 14]              groups.prev: my unit's group, an enemy's target-of
    map [B, 33, 32, 32]          the whole field in the field's axes, a cell of 64 m (layers: LAYERS; the cards' layers
                                 zeros until net paints them), cover [B, K, 32, 32] the cells' shares of the rows
    tokens [B, G, 10]            groups.numbers; members, targets [B, G, K] bool: the rows of each group's members / target
    numbers [B, 1]               the groups busy / G
    command [B, 7]               the order from above (verb 4 + pace 3: tactic/decide.order_vec), the case's for now

One function on the simulator's raw state and on the bridge's (tools/nn/companion/operative_input.py).
"""
import torch

from tools.nn.operative import cfg
from tools.nn.operative import frame as FR
from tools.nn.tactic import raw as TR
from tools.nn.tactic import rows as RW
from tools.nn.tactic import windows as W
from tools.nn.unit import zmap as Z


def layers(card=8):
    return (("mine", "mine_cos", "mine_sin", "enemy", "enemy_cos", "enemy_sin")
            + tuple(f"card_mine{i}" for i in range(card)) + tuple(f"card_enemy{i}" for i in range(card))
            + ("zones",) + W.TERRAIN + W.FIRE)


LAYERS = layers()
LI = {n: i for i, n in enumerate(LAYERS)}


def select(raw, side, K=None):
    """(slot [B, K] long -1 empty, who [B, K] long RW.NONE empty)."""
    K = int(K or cfg()["rows"])
    mine, enemy = FR.sides(raw, side)
    B, N = mine.shape
    dev = raw.device
    sl = torch.arange(N, device=dev, dtype=torch.float64)[None].expand(B, N)
    key = torch.full((B, N), float("inf"), dtype=torch.float64, device=dev)
    key = torch.where(enemy, 1e4 + sl, key)
    key = torch.where(mine, sl, key)
    m = min(K, N)
    best = key.topk(m, dim=1, largest=False)
    order = best.values.argsort(1)
    pick = best.indices.gather(1, order)
    ok = best.values.gather(1, order) < float("inf")
    slot = torch.where(ok, pick, torch.full_like(pick, -1))
    if m < K:
        slot = torch.cat([slot, torch.full((B, K - m), -1, dtype=torch.long, device=dev)], 1)
    s0 = slot.clamp(min=0)
    who = torch.where(mine.gather(1, s0), RW.OWN_GROUP, RW.ENEMY_OTHER)
    return slot, torch.where(slot >= 0, who, torch.full_like(who, RW.NONE))


def build_map(J, mask, who, axes, ground, zones, bounds=None):
    """(map [B, L, C, C], cover [B, K, C, C]). ground {terr, ...} [B, 1]; zones: groups.zones()."""
    c = cfg()
    C = int(c["cells"])
    B = J.shape[0]
    dev = J.device
    cell = torch.full((B,), float(c["cell_m"]), device=dev)
    cx, cz, hd = axes.cx, axes.cz, axes.heading
    mine = mask & (who == RW.OWN_GROUP)
    enemy = mask & (who == RW.ENEMY_OTHER)
    m = torch.zeros((B, len(LAYERS), C, C), device=dev)
    roles = torch.where(mine, Z.OWN, torch.where(enemy, Z.ENEMY, Z.NONE))
    dn = W.densities(J, roles, cx, cz, hd, cell, C)
    m[:, LI["mine"]:LI["mine"] + 3] = dn[:, 3:6]
    m[:, LI["enemy"]:LI["enemy"] + 3] = dn[:, 0:3]
    m[:, LI["zones"]] = W.area_layer(zones, cx, cz, hd, cell, C)
    h0 = FR.ground_at_g(ground, cx, cz)
    fld = axes.field if bool(axes.field["on"].any()) else None
    t0 = LI[W.TERRAIN[0]]
    m[:, t0:t0 + 6] = W.terrain_layers(cx, cz, hd, cell, C, ground, h0, fld, bounds)
    f0 = LI[W.FIRE[0]]
    m[:, f0:f0 + 4] = W.fire_layers(J, enemy, mine, cx, cz, hd, cell, C)
    cover = W.covers(J, mine | enemy, cx, cz, hd, cell, C)
    return m, cover


def build(raw, card, side, axes, groups, command, bounds=None, K=None):
    """raw [B, N, RAW] (tactic.raw), card [B, N, 69], side [B] long, axes (frame.Axes), groups (groups.Groups),
    command [B, 7]. Returns the dict of the module doc."""
    slot, who = select(raw, side, K)
    B, Kk = slot.shape
    s0 = slot.clamp(min=0)
    mask = slot >= 0
    J = raw.gather(1, s0[..., None].expand(B, Kk, raw.shape[-1]))
    cd = card.gather(1, s0[..., None].expand(B, Kk, card.shape[-1]))
    I = TR.I
    out = RW.row(J, cd, who, J[..., I["x"]], J[..., I["z"]], axes.heading[:, None], J[..., I["gh"]])
    zero = torch.zeros((), device=raw.device)
    rows = torch.where(mask[..., None], out, zero)
    J = torch.where(mask[..., None], J, zero)
    own_rows = mask & (who == RW.OWN_GROUP)
    enemy_rows = mask & (who == RW.ENEMY_OTHER)
    place = FR.place(raw, axes).gather(1, s0[..., None].expand(B, Kk, 3)) * mask[..., None]
    prev = groups.prev(slot, own_rows, enemy_rows)
    ground = TR.ground_of(raw)
    mp, cover = build_map(J, mask, who, axes, ground, groups.zones(), bounds)
    mem = groups.members().gather(2, s0[:, None, :].expand(B, groups.G, Kk)) & own_rows[:, None, :]
    tgt = groups.target.gather(2, s0[:, None, :].expand(B, groups.G, Kk)) & enemy_rows[:, None, :] & \
        groups.on[..., None]
    G = groups.G
    numbers = (groups.on.float().sum(1, keepdim=True) / G)
    return {"rows": rows, "mask": mask, "slot": slot, "who": who, "J": J, "place": place, "prev": prev, "map": mp,
            "cover": cover, "own": own_rows, "enemy": enemy_rows, "tokens": groups.numbers(axes, raw),
            "members": mem, "targets": tgt, "numbers": numbers, "command": command}
