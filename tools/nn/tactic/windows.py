"""The tactician's map: three windows of CELLS x CELLS (30) cells in the zone's axes - "up" is the zone's turn in every
window (design/tactic "Окно группы", "Окно цели", "Карточка на карте", "Огонь"; the user's decisions 30-31.10). The
same builder as the unit level's map (tools/nn/unit/zmap.py, imported: the exact share of a cell under a turned
rectangle, density = share x the share of the formation's places filled, the front's cos / sin x the density, two units
in a cell their density-weighted mean); the ground through tools/nn/sim/terrain.py and ground.py (the captured map).

    near    centre: the zone's centre; side: the zone's larger side + one tile (80 m) each side; cell = side / 30 -
            always. Layers (34 with the card of 8): tiles 1, personal areas 1, my units 3 (density, front cos, sin),
            the target 3 (if it is in the window), the card of my units 8, the card of the target 8, terrain 6, fire 4
    far     the same centre, cell = the group's longest missile range / 14 - only when the group has shooters or guns
            (zeros otherwise, and the number 'no far window' = 1). Layers (28): enemies 1, own 1, the card of the
            enemies 8, of own units 8, terrain 6, fire 4
    target  centre: the target's actual centre (the mean of its seen units' centres), the near window's cell, "up" MY
            zone's turn. Layers (17): the target 3, its card 8, terrain 6 (zeros without a seen target)
    numbers 4: the link - the distance from my group's actual centre (the mean of my units' centres) to the target's
            centre / 100 m and its direction in the map's axes (cos: forward, sin: right; zeros without a target) -
            and 'no far window' (0 / 1)

terrain 6: the field (the share of the cell inside the playable field: the captured map's frame, terrain.frame_local;
a flat battle - the square +-bounds; no bounds - 1), the height (the ground at the cell's centre less the ground under
the zone's centre / 20 m, +-4), forest (the share of the cell on ground class 1), trunks (1 - exp(-lam x 10 m): the
share of projectiles the cell's trunks stop over 10 m of path; ground.py lam), cannot stand (the share on the
'cannot stand' nodes, ground.blocked), shallow water (class 2); the shares by 3 x 3 points a cell.
fire 4 (0 / 1 by the cell's centre): the enemy's reach (within the range of some seen enemy shooter or gun - beyond its
minimum range: a ring), the enemy's fire arc now (within that and within its fire arc now each side of its front:
view.fire_arc - standing 45 deg, on the move 90 or 0), my group's reach, my group's arc. Shooters: alive, not routing,
with ammunition.

Only the rows' units are painted (rows.py: their zone of attention) - nothing of a unit outside it reaches the map.
The cards' layers need the network ('Анализ отряда' -> 128 -> 8): build() returns them as zeros with the cells'
shares of every row (cover) and which rows go into which card layers (paint); net.paint_cards fills them.
"""
import torch

from tools.nn.tactic import cfg
from tools.nn.tactic import raw as TR
from tools.nn.tactic import rows as RW
from tools.nn.unit import zmap as Z

TERRAIN = ("field", "height", "forest", "trunks", "blocked", "shallow")
FIRE = ("enemy_reach", "enemy_arc", "our_reach", "our_arc")


def near_layers(card=None):
    k = int(card or cfg()["card"])
    return (("tiles", "areas", "mine", "mine_cos", "mine_sin", "target", "target_cos", "target_sin")
            + tuple(f"card_mine{i}" for i in range(k)) + tuple(f"card_target{i}" for i in range(k)) + TERRAIN + FIRE)


def far_layers(card=None):
    k = int(card or cfg()["card"])
    return (("enemies", "own") + tuple(f"card_enemies{i}" for i in range(k)) + tuple(f"card_own{i}" for i in range(k))
            + TERRAIN + FIRE)


def target_layers(card=None):
    k = int(card or cfg()["card"])
    return ("target", "target_cos", "target_sin") + tuple(f"card_target{i}" for i in range(k)) + TERRAIN


# the card layers of each window: (the first card layer's name, the paint selection's key)
CARDS = {"near": (("card_mine0", "near_mine"), ("card_target0", "near_target")),
         "far": (("card_enemies0", "far_enemy"), ("card_own0", "far_own")),
         "target": (("card_target0", "target"),)}
LAYERS = {"near": near_layers, "far": far_layers, "target": target_layers}
NUMBERS = ("link_dist", "link_cos", "link_sin", "no_far")


def cell_points(cx, cz, heading, cell, cells, sub=1):
    """(x, z) [R, C, C, sub x sub] world points: the cells' centres (sub 1) or a sub x sub split of each cell."""
    k = torch.arange(cells, device=cx.device, dtype=torch.float32)
    q = (torch.arange(sub, device=cx.device, dtype=torch.float32) + 0.5) / sub - 0.5
    cm = cell.view(-1, 1, 1, 1)
    s0, f0 = Z.cell_centre(k[:, None], k[None, :], cell.view(-1, 1, 1), cells)      # [R, 1, C], [R, C, 1]
    s = s0[..., None] + (q[None, :, None] * torch.ones(sub, device=cx.device)[None, None, :]).reshape(-1) * cm
    f = f0[..., None] - (torch.ones(sub, device=cx.device)[:, None] * q[None, :]).reshape(-1) * cm
    s, f = torch.broadcast_tensors(s, f)
    sh = (-1, 1, 1, 1)
    return Z.world(s, f, cx.view(sh), cz.view(sh), heading.view(sh))


def covers(J, sel, cx, cz, heading, cell, cells):
    """[R, K, C, C] the share of each cell under each row's formation (J [R, K, RAW]; zero where not sel [R, K])."""
    I = TR.I
    rs, rf = Z.local(J[..., I["x"]], J[..., I["z"]], cx[:, None], cz[:, None], heading[:, None])
    a = (J[..., I["b"]] - heading[:, None]) * TR.DEG
    front = torch.where(sel, J[..., I["front"]], torch.zeros_like(rs))
    depth = torch.where(sel, J[..., I["depth"]], torch.zeros_like(rs))
    cov = Z._cover(rs, rf, a, front, depth, cell, cells)
    return cov * sel[..., None, None].float()


def densities(J, roles, cx, cz, heading, cell, cells):
    """[R, 7, C, C] zmap.build's unit layers (enemy, its cos, sin, own, its cos, sin, self) of the rows J [R, K, RAW]
    with their map roles roles [R, K] (zmap NONE / ENEMY / OWN)."""
    I = TR.I
    units = {"x": J[..., I["x"]], "z": J[..., I["z"]], "b": J[..., I["b"]], "front": J[..., I["front"]],
             "depth": J[..., I["depth"]], "fill": J[..., I["fill"]] * (roles != Z.NONE).float(), "role": roles}
    m = Z.build(cx, cz, heading, cell, units, cells=cells)
    return m[:, Z.L["enemy"]:Z.L["self"] + 1]


def terrain_layers(cx, cz, heading, cell, cells, ground, h0, field=None, bounds=None):
    """[R, 6, C, C] (TERRAIN) of the frames (module doc). ground: {terr, tpx, tpz, tpr} [R, 1] each frame's battle's
    map (terr -1: flat); h0 [R] the ground under the zone's centre; field: {x, z, b, front, depth, on} [R] the playable
    field as a turned rectangle in the battle's frame (on False: none) or None; bounds: a flat battle's square half
    size (None: no edge)."""
    from tools.nn.sim import ground as G
    from tools.nn.sim import terrain
    c = cfg()
    R = cx.shape[0]
    dev = cx.device
    out = torch.zeros((R, len(TERRAIN), cells, cells), device=dev)
    # the field
    fl = torch.ones((R, cells, cells), device=dev)
    if bounds is not None:
        bd = torch.as_tensor(bounds, dtype=torch.float32, device=dev).expand(R)
        zero = torch.zeros(R, device=dev)
        rs, rf = Z.local(zero, zero, cx, cz, heading)
        fl = Z._cover(rs[:, None], rf[:, None], (-heading * TR.DEG)[:, None], (2 * bd)[:, None], (2 * bd)[:, None],
                      cell, cells)[:, 0]
    if field is not None:
        rs, rf = Z.local(field["x"], field["z"], cx, cz, heading)
        a = (field["b"] - heading) * TR.DEG
        fm = Z._cover(rs[:, None], rf[:, None], a[:, None], field["front"][:, None], field["depth"][:, None],
                      cell, cells)[:, 0]
        fl = torch.where(field["on"].view(R, 1, 1), fm, fl)
    out[:, 0] = fl
    if "terr" not in ground or not bool((ground["terr"] >= 0).any()):
        return out
    x, z = cell_points(cx, cz, heading, cell, cells, 1)
    h = terrain.height(ground, x, z)[..., 0]
    cap = float(c["height_cap"])
    out[:, 1] = ((h - h0.view(R, 1, 1)) / float(c["height_m"])).clamp(-cap, cap)
    if G.on(ground):
        px, pz = cell_points(cx, cz, heading, cell, cells, int(c["samples"]))
        cls = G.classes(ground, px, pz)
        out[:, 2] = (cls == 1).float().mean(-1)
        out[:, 5] = (cls == 2).float().mean(-1)
        lam = G._at(ground, "lam", px, pz).mean(-1)
        out[:, 3] = 1 - torch.exp(-lam * float(c["trunk_path_m"]))
        out[:, 4] = G.blocked(ground, px, pz).float().mean(-1)
    return out


def fire_layers(J, enemy_sel, our_sel, cx, cz, heading, cell, cells):
    """[R, 4, C, C] (FIRE) of the rows J [R, K, RAW]: the shooters among enemy_sel / our_sel [R, K] (module doc)."""
    I = TR.I
    x, z = cell_points(cx, cz, heading, cell, cells, 1)
    x, z = x[..., 0].flatten(1)[:, None], z[..., 0].flatten(1)[:, None]          # [R, 1, C*C]
    sx, sz = J[..., I["x"]][..., None], J[..., I["z"]][..., None]                   # [R, K, 1]
    dx, dz = x - sx, z - sz
    d = torch.sqrt(dx * dx + dz * dz)
    rng, rmin = J[..., I["range"]][..., None], J[..., I["min_range"]][..., None]
    shoots = (J[..., I["range"]] > 0) & (J[..., I["alive"]] > 0) & ~(J[..., I["r"]] > 0) & (J[..., I["ammo"]] > 0)
    reach = (d <= rng) & (d >= rmin)
    way = torch.rad2deg(torch.atan2(dx, dz))
    off = (torch.remainder(way - J[..., I["b"]][..., None] + 180.0, 360.0) - 180.0).abs()
    arc = J[..., I["arc"]][..., None]
    in_arc = reach & ((off <= arc) | (arc >= 180.0))
    out = []
    for sel in (enemy_sel, our_sel):
        s = (sel & shoots)[..., None]
        out += [(reach & s).any(1), (in_arc & s).any(1)]
    return torch.stack(out, 1).float().view(J.shape[0], 4, cells, cells)


def tile_layer(tiles, cell, cells):
    """[R, C, C] the share of each cell inside the tiles left (they never overlap)."""
    t = tiles.rects()
    rs, rf = Z.local(t["x"], t["z"], tiles.cx[:, None], tiles.cz[:, None], tiles.heading[:, None])
    a = torch.zeros_like(rs)
    return Z._cover(rs, rf, a, t["front"], t["depth"], cell, cells).sum(1).clamp(max=1.0)


def area_layer(areas, cx, cz, heading, cell, cells):
    """[R, C, C] the share of each cell inside the group's personal areas ({x, z, b, front, depth, on} [R, M]; they
    never overlap: the translator's rule) - the tactician's own last output (a stub of that shape until it exists)."""
    if areas is None:
        return torch.zeros((cx.shape[0], cells, cells), device=cx.device)
    rs, rf = Z.local(areas["x"], areas["z"], cx[:, None], cz[:, None], heading[:, None])
    a = (areas["b"] - heading[:, None]) * TR.DEG
    on = areas["on"].bool()
    cov = Z._cover(rs, rf, a, torch.where(on, areas["front"], torch.zeros_like(rs)),
                   torch.where(on, areas["depth"], torch.zeros_like(rs)), cell, cells)
    return (cov * on[..., None, None].float()).sum(1).clamp(max=1.0)


def build(rows, parcel, tiles, ground, areas=None, field=None, bounds=None):
    """The three windows and the numbers of R groups. rows: rows.build's dict (J, mask, who, h0); parcel, tiles (R);
    ground {terr, tpx, tpz, tpr} [R, 1]; areas / field / bounds as above. Returns {near [R, L, C, C], far, target (the
    card layers zeros), cover {near, far, target} [R, K, C, C], paint {near_mine, near_target, far_enemy, far_own,
    target} [R, K] bool, numbers [R, 4], cell [R, 3] (near, far, target cell, m)}."""
    c = cfg()
    C = int(c["cells"])
    I = TR.I
    J, mask, who, h0 = rows["J"], rows["mask"], rows["who"], rows["h0"]
    R = J.shape[0]
    dev = J.device
    cx, cz, hd = parcel.cx, parcel.cz, parcel.heading()
    mine = mask & (who == RW.OWN_GROUP)
    target = mask & (who == RW.ENEMY_TARGET)
    enemy = mask & ((who == RW.ENEMY_TARGET) | (who == RW.ENEMY_OTHER))
    own = mask & ((who == RW.OWN_GROUP) | (who == RW.OWN_OTHER))
    # the cells
    cell_n = tiles.near_size() / C
    g_range = torch.where(mine, J[..., I["range"]], torch.zeros_like(J[..., 0])).amax(1)
    has_far = g_range > 0
    cell_f = torch.where(has_far, g_range / float(c["range_cells"]), cell_n)
    # the target's actual centre, my group's actual centre
    def centre(sel):
        w = sel.float()
        k = w.sum(1).clamp(min=1)
        return (J[..., I["x"]] * w).sum(1) / k, (J[..., I["z"]] * w).sum(1) / k
    tx, tz = centre(target)
    gx, gz = centre(mine)
    has_t = target.any(1)
    tx, tz = torch.where(has_t, tx, cx), torch.where(has_t, tz, cz)
    L = {"near": near_layers(), "far": far_layers(), "target": target_layers()}
    li = {w: {n: i for i, n in enumerate(names)} for w, names in L.items()}
    near = torch.zeros((R, len(L["near"]), C, C), device=dev)
    far = torch.zeros((R, len(L["far"]), C, C), device=dev)
    tgt = torch.zeros((R, len(L["target"]), C, C), device=dev)
    # near
    near[:, li["near"]["tiles"]] = tile_layer(tiles, cell_n, C)
    near[:, li["near"]["areas"]] = area_layer(areas, cx, cz, hd, cell_n, C)
    roles = torch.where(mine, Z.OWN, torch.where(target, Z.ENEMY, Z.NONE))
    dn = densities(J, roles, cx, cz, hd, cell_n, C)
    near[:, li["near"]["mine"]:li["near"]["mine"] + 3] = dn[:, 3:6]
    near[:, li["near"]["target"]:li["near"]["target"] + 3] = dn[:, 0:3]
    t0 = li["near"]["field"]
    near[:, t0:t0 + 6] = terrain_layers(cx, cz, hd, cell_n, C, ground, h0, field, bounds)
    f0 = li["near"]["enemy_reach"]
    near[:, f0:f0 + 4] = fire_layers(J, enemy, mine, cx, cz, hd, cell_n, C)
    # far
    roles = torch.where(enemy, Z.ENEMY, torch.where(own, Z.OWN, Z.NONE))
    df = densities(J, roles, cx, cz, hd, cell_f, C)
    far[:, li["far"]["enemies"]] = df[:, 0]
    far[:, li["far"]["own"]] = df[:, 3]
    t0 = li["far"]["field"]
    far[:, t0:t0 + 6] = terrain_layers(cx, cz, hd, cell_f, C, ground, h0, field, bounds)
    f0 = li["far"]["enemy_reach"]
    far[:, f0:f0 + 4] = fire_layers(J, enemy, mine, cx, cz, hd, cell_f, C)
    far = far * has_far.view(R, 1, 1, 1).float()
    # target
    roles = torch.where(target, Z.ENEMY, Z.NONE)
    dt = densities(J, roles, tx, tz, hd, cell_n, C)
    tgt[:, 0:3] = dt[:, 0:3]
    t0 = li["target"]["field"]
    tgt[:, t0:t0 + 6] = terrain_layers(tx, tz, hd, cell_n, C, ground, h0, field, bounds)
    tgt = tgt * has_t.view(R, 1, 1, 1).float()
    # the cells' shares of the rows, for the cards' layers
    cover = {"near": covers(J, mine | target, cx, cz, hd, cell_n, C),
             "far": covers(J, (enemy | own) & has_far[:, None], cx, cz, hd, cell_f, C),
             "target": covers(J, target, tx, tz, hd, cell_n, C)}
    paint = {"near_mine": mine, "near_target": target, "far_enemy": enemy & has_far[:, None],
             "far_own": own & has_far[:, None], "target": target}
    # the numbers: the link and 'no far window'
    ls, lf = Z.local(tx, tz, gx, gz, hd)
    dist = torch.sqrt(ls * ls + lf * lf)
    ok = has_t & mine.any(1) & (dist > 1e-6)
    numbers = torch.stack([torch.where(has_t & mine.any(1), dist / float(c["pos_m"]), torch.zeros_like(dist)),
                           torch.where(ok, lf / dist.clamp(min=1e-6), torch.zeros_like(dist)),
                           torch.where(ok, ls / dist.clamp(min=1e-6), torch.zeros_like(dist)),
                           (~has_far).float()], 1)
    return {"near": near, "far": far, "target": tgt, "cover": cover, "paint": paint, "numbers": numbers,
            "cell": torch.stack([cell_n, torch.where(has_far, cell_f, torch.zeros_like(cell_f)), cell_n], 1)}
