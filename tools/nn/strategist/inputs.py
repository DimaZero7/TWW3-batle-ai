"""The strategist's input of B battles (plan.md "Вход"): what the player sees before the battle.

    rows [B, K, 185], mask, slot, who [B, K]   my units by slot, then the enemy's by slot (K = 40: 20 + 20); the
                                 tactician's row formula with the origin at the unit itself (as the operative's) and then
                                 BLIND (blind()): no unit has a place yet, and the enemy's is not seen
    map [B, 8, 32, 32]           in the strategist's axes (zones.axes), a cell of 64 m: the terrain 6 (tactic/windows
                                 terrain_layers), the share of the cell in my deployment zone, in the enemy's
    numbers [B, 2]               my units / 20, the enemy's / 20
    command [B, 7]               the battle's task from above (verb 4 + pace 3: 'destroy' = attack, 'hold'), the case's
    shooter [B, K] bool          a row's unit shoots (its passport's range > 0: the composition)

blind(): every row - its place and all that follows from it (right, forward, the front's cos / sin, the velocities, the
height step, 'fires') is zero; an enemy row also keeps of its live part only the men and the health (the pre-battle
screen shows the cards with their men and health bar), and no effect or button 'on now'. What stays of an enemy: its
passport, its stats, which buttons it has. One function on the simulator's raw state and on the bridge's.
"""
import torch

from tools.nn.operative import frame as FR
from tools.nn.operative import inputs as OIN
from tools.nn.strategist import cfg
from tools.nn.strategist import zones as ZS
from tools.nn.tactic import raw as TR
from tools.nn.tactic import rows as RW
from tools.nn.tactic import windows as W

LAYERS = W.TERRAIN + ("my_zone", "enemy_zone")
LI = {n: i for i, n in enumerate(LAYERS)}
PLACE = ("right", "forward", "front_cos", "front_sin", "v_right", "v_forward", "fire", "dh")
ENEMY_KEEPS = ("men", "hp")


def _cols(names):
    return torch.tensor([RW.RI[n] for n in names], dtype=torch.long)


def blind_masks():
    """([185] bool own columns kept, [185] bool enemy columns kept)."""
    own = torch.ones(len(RW.ROW), dtype=torch.bool)
    own[_cols(PLACE)] = False
    enemy = own.clone()
    enemy[_cols([n for n in RW.LIVE if n not in ENEMY_KEEPS])] = False
    for name, *_ in TR.EFFECTS:
        enemy[RW.RI[f"{name}_on"]] = False
        enemy[RW.RI[f"{name}_left"]] = False
    for a in range(TR.A):
        enemy[RW.RI[f"on_{a}"]] = False
        enemy[RW.RI[f"cd_{a}"]] = False
    return own, enemy


def blind(rows, own_rows, enemy_rows):
    ko, ke = (m.to(rows.device) for m in blind_masks())
    keep = torch.where(own_rows[..., None], ko, torch.where(enemy_rows[..., None], ke, torch.zeros_like(ko)))
    return torch.where(keep, rows, torch.zeros((), device=rows.device))


def build_map(axes, raw, mine_zone, enemy_zone, bounds=None):
    c = cfg()
    C, cm = int(c["cells"]), float(c["cell_m"])
    B = raw.shape[0]
    dev = raw.device
    cell = torch.full((B,), cm, device=dev)
    cx, cz, hd = axes.cx, axes.cz, axes.heading
    ground = TR.ground_of(raw)
    h0 = FR.ground_at_g(ground, cx, cz)
    fld = axes.field if bool(axes.field["on"].any()) else None
    m = torch.zeros((B, len(LAYERS), C, C), device=dev)
    m[:, :len(W.TERRAIN)] = W.terrain_layers(cx, cz, hd, cell, C, ground, h0, fld, bounds)
    m[:, LI["my_zone"]] = W.area_layer(ZS.as_area(mine_zone), cx, cz, hd, cell, C)
    m[:, LI["enemy_zone"]] = W.area_layer(ZS.as_area(enemy_zone), cx, cz, hd, cell, C)
    return m


def build(raw, card, side, axes, mine_zone, enemy_zone, command, bounds=None, K=None):
    """raw [B, N, RAW] (tactic.raw; every unit present and seen - the composition), card [B, N, 69], side [B] long."""
    K = int(K or cfg()["rows"])
    slot, who = OIN.select(raw, side, K)
    B, Kk = slot.shape
    s0 = slot.clamp(min=0)
    mask = slot >= 0
    J = raw.gather(1, s0[..., None].expand(B, Kk, raw.shape[-1]))
    cd = card.gather(1, s0[..., None].expand(B, Kk, card.shape[-1]))
    I = TR.I
    out = RW.row(J, cd, who, J[..., I["x"]], J[..., I["z"]], axes.heading[:, None], J[..., I["gh"]])
    own_rows = mask & (who == RW.OWN_GROUP)
    enemy_rows = mask & (who == RW.ENEMY_OTHER)
    rows = blind(torch.where(mask[..., None], out, torch.zeros((), device=raw.device)), own_rows, enemy_rows)
    numbers = torch.stack([own_rows.sum(1).float() / 20.0, enemy_rows.sum(1).float() / 20.0], 1)
    shooter = (J[..., I["range"]] > 0) & mask
    return {"rows": rows, "mask": mask, "slot": slot, "who": who, "own": own_rows, "enemy": enemy_rows,
            "shooter": shooter, "map": build_map(axes, raw, mine_zone, enemy_zone, bounds), "numbers": numbers,
            "command": command}
