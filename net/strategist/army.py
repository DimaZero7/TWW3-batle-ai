"""The strategist's formula for a whole army (08.11, the orchestrator's task; tmp/old-build/diag_label/human_rules.md - the
user's three won deployments on the Kislev plains): the groups of the operative's army formula (operative/army.py) from
the start, laid out as the person lays his out, so that the operative's teacher keeps them at its first decisions
(a group in force of one role and not over its size stays, a group that holds keeps its zone).

The groups: by role (operative/army.roles of the row's passport), cut by slot (the strategist sees no places) into the
teacher's sizes: the line and the halberds (anti) in pairs, the shooters in threes, the snipers that fire flat one
each, the other snipers one group, all the guns one group, the lord alone, the fast in pairs.

The layout, in my zone's axes (s right, f forward), rows from the front (human_rules 1):
    row 1..   the line: its pairs side by side across the middle (the line 180-330 m wide, 2 rows); the snipers that
              fire flat (passport 'direct': jezzails, long rifles) at the outer ends of the first row, the right one
              first, SNIPER_GAP_M clear of the line (33-44 m from the nearest infantry, nothing of mine ahead),
              SNIPER_BACK_M behind the row (plan nudge), a halberd / spear pair at the outer end of the second row
              beside each (the open flank's guard: the enemy's fast units struck the snipers at the ends first);
              the other snipers and the shooters at the flanks of the first line row; the first two halberd pairs at the flanks of the last line row ("one pair left, one right")
    the back  behind the centre: the guns (the artillery in the centre behind the line, -50..0 m), the lord (behind the
              centre, -15..-19 m), the other halberd pairs (the reserve "behind in the centre"); the fast behind the
              flanks (80 m behind the line)
A row holds what fits in my zone's width (edge_m inside); the rest goes to the next row. The walk puts a group's zone
centre on a cell (64 m), so the rows are the cells' rows inside my zone and a group the cells' centre where its
footprint clears the one before (pack); 'hold' takes the back rows, 'attack' the front ones. The walk (net.choose)
takes these points as the formula's wishes - the open cell nearest each, the groups in the order of their numbers:
the front rows first, in each the centre outwards.

    groups(roles, rows)               [(role, [rows])] by the teacher's sizes
    layout(groups, fw, fd, s_grid, f_grid, front, edge, attack)   [(s, f)] the wanted centres
"""
from net.operative import army as AM

SIZE = {"line": 2, "anti": 2, "shooter": 3, "fast": 2, "sniper": 10, "art": 10, "lord": 1}   # operative/army.plan
GAP_M = 2.0               # between two groups side by side (their outer formations: config unit_gap_m)
SNIPER_GAP_M = 8.0        # a direct-fire snipers' group's footprint from its neighbour's: the person's snipers stand
#                           33-44 m (centre to centre) from the nearest infantry (human_rules 1) - a pair of greatswords
#                           and two long rifles' slots 8 m apart put their nearest men's centres ~42 m apart
SNIPER_BACK_M = 10.0      # the direct-fire snipers stand this far behind their front row's centre line (the
#                           orchestrator 08.11: 8-12 m behind the first row's edge - the enemy's fast units strike the
#                           open flank first: in the chain's battles the snipers at the ends met them at 52-71 s)
DIRECT_COL = 66           # passport.features: FLIGHT 'direct' (flat fire: needs a clear line past friends)


def roles(cards):
    return AM.roles(cards)


def groups(rl, rows, flat=()):
    """[(role, [rows])]: rows (in slot order) of my units, rl their roles; flat the rows that fire flat - a sniper of
    them is a group of its own (one each flank, 33-44 m from the infantry: a group of two would stand a cell farther;
    the teacher keeps a group of one role within its size)."""
    by = {}
    for j, r in zip(rows, rl):
        by.setdefault(r, []).append(j)
    out = []
    for r in ("line", "anti", "shooter", "sniper", "art", "fast", "lord"):
        js = by.get(r, [])
        if r == "sniper":
            out += [(r, [j]) for j in js if j in flat]
            js = [j for j in js if j not in flat]
        k = SIZE[r]
        out += [(r, js[i:i + k]) for i in range(0, len(js), k)]
    return out


def direct(gs, cards_of):
    """{group index} the snipers' groups that fire flat (any unit's passport 'direct'): cards_of(row) -> its card."""
    return {i for i, (r, rows) in enumerate(gs) if r == "sniper" and any(cards_of(j)[DIRECT_COL] > 0.5 for j in rows)}


def pack(ws, grid, half, gap=GAP_M, gaps=None):
    """[s] the centres of groups of widths ws side by side on the cells' centres grid (sorted, my zone's s axis), each
    clear of the one before by gap (or by the larger of the two groups' gaps [i]), the row centred; None if it does not
    fit within +-half."""
    if not ws:
        return []
    gaps = gaps or [gap] * len(ws)
    out, right = [], -1e9
    for k, w in enumerate(ws):
        need = max(gap, gaps[k], gaps[k - 1]) if k else gap
        c = next((g for g in grid if g - w / 2 >= right + need and g - w / 2 >= -half - 1e-6), None)
        if c is None or c + w / 2 > half + 1e-6:
            return None
        out.append(c)
        right = c + w / 2
    # centre the row: the whole row moved by whole cells while it stays inside
    step = (grid[1] - grid[0]) if len(grid) > 1 else 0.0
    best = out
    for k in range(-len(grid), len(grid) + 1):
        sh = [c + k * step for c in out]
        if step and all(c - w / 2 >= -half - 1e-6 and c + w / 2 <= half + 1e-6 for c, w in zip(sh, ws)):
            if abs(sh[0] - ws[0] / 2 + sh[-1] + ws[-1] / 2) < abs(best[0] - ws[0] / 2 + best[-1] + ws[-1] / 2):
                best = sh
    return best


def _seq(row):
    """Left to right: the left flank items (the first put outermost), the middle ones, the right flank items (the
    first put outermost)."""
    left = [i for i, w in row if w == "left"]
    right = [i for i, w in row if w == "right"][::-1]
    return left + [i for i, w in row if w == "mid"] + right


def _gaps(seq, flat):
    return [SNIPER_GAP_M if k in flat else GAP_M for k in seq]


def _rows(gs, fw, grid, half, flat=()):
    """[[(group index, place)]] the rows from the front: place 'mid' / 'left' / 'right' (module doc); flat the
    direct-fire snipers' groups - the first row's outer flanks, before the line (nothing of mine ahead of them)."""
    rows = [[]]

    def fits(r, i, where):
        if r >= len(rows):
            return True
        seq = _seq(rows[r] + [(i, where)])
        return pack([fw[k] for k in seq], grid, half, gaps=_gaps(seq, flat)) is not None

    def put(r, i, where):
        while not fits(r, i, where):
            r += 1
        while r >= len(rows):
            rows.append([])
        rows[r].append((i, where))
        return r

    line = [i for i, (r, _) in enumerate(gs) if r == "line"]
    anti = [i for i, (r, _) in enumerate(gs) if r == "anti"]
    ranged = [i for i, (r, _) in enumerate(gs) if r in ("sniper", "shooter") and i not in flat]
    back = [i for i, (r, _) in enumerate(gs) if r in ("art", "lord")]
    fast = [i for i, (r, _) in enumerate(gs) if r == "fast"]
    for k, i in enumerate(sorted(flat)):             # the flat-firing snipers first: the front row's ends
        put(0, i, "right" if k % 2 == 0 else "left")
    # a halberd / spear pair beside each such sniper, nearer the flank, a row behind (the front row has no room for
    # them: 6 cells of 64 m - two line pairs and two snipers fill it), the guard of the open flank; no halberds: a line
    # pair (the last ones)
    n_g = len(flat)
    guards = anti[:n_g] + line[len(line) - max(0, n_g - len(anti)):] if n_g > len(anti) else anti[:n_g]
    for k, i in enumerate(guards):
        put(1, i, "right" if k % 2 == 0 else "left")
    anti = [i for i in anti if i not in guards]
    line = [i for i in line if i not in guards]
    last_line = 0
    for i in line:
        last_line = put(last_line, i, "mid")
    for k, i in enumerate(ranged):                   # the flanks of the first line row, the right one first
        put(0, i, "right" if k % 2 == 0 else "left")
    for k, i in enumerate(anti[:2]):                 # one pair left, one right of the last line row
        put(last_line, i, "left" if k == 0 else "right")
    rb = len(rows) if rows[-1] else len(rows) - 1    # the back row
    for i in back + anti[2:]:
        put(rb, i, "mid")
    for k, i in enumerate(fast):
        put(rb, i, "left" if k % 2 == 0 else "right")
    return [x for x in rows if x]


def layout(gs, fw, fd, s_grid, f_grid, front, edge, attack, flat=()):
    """[(s, f)] per group of gs (module doc); fw, fd the groups' footprints (strategist/deploy.footprint); s_grid,
    f_grid the cells' centres inside my zone along its axes (the walk puts a zone's centre on a cell: the groups go
    on whole cells, clear of each other); front my zone's width, edge the margin inside it."""
    half = front / 2 - edge
    rows = _rows(gs, fw, sorted(s_grid), half, set(flat))
    out = [None] * len(gs)
    fs = sorted(f_grid, reverse=True)                # the rows' places from the front
    n = len(rows)
    if not attack and n <= len(fs):
        fs = fs[len(fs) - n:]                        # 'hold': the block at the back
    for k, row in enumerate(rows):
        seq = _seq(row)
        ss = pack([fw[i] for i in seq], sorted(s_grid), half, gaps=_gaps(seq, set(flat))) or [0.0] * len(seq)
        f = fs[min(k, len(fs) - 1)] if fs else 0.0
        for i, s in zip(seq, ss):
            out[i] = (s, f)
    return out


def order(gs, places):
    """The groups' numbers: the walk places them in this order - the front rows first, in each the centre outwards."""
    return sorted(range(len(gs)), key=lambda i: (-places[i][1], abs(places[i][0])))
