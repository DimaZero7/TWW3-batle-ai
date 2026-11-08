"""The strategist's formula for a whole army (08.11, the orchestrator's task; build/diag_label/human_rules.md - the
user's three won deployments on the Kislev plains): the groups of the operative's army formula (operative/army.py) from
the start, laid out as the person lays his out, so that the operative's teacher keeps them at its first decisions
(a group in force of one role and not over its size stays, a group that holds keeps its zone).

The groups: by role (operative/army.roles of the row's passport), cut by slot (the strategist sees no places) into the
teacher's sizes: the line and the halberds (anti) in pairs, the shooters in threes, all the snipers one group, all the
guns one group, the lord alone, the fast in pairs.

The layout, in my zone's axes (s right, f forward), rows from the front (human_rules 1):
    row 1..   the line: its pairs side by side across the middle (the line 180-330 m wide, 2 rows); the snipers and
              the shooters at the flanks of the first line row, the right one first (33-44 m from the infantry, not
              ahead of it); the first two halberd pairs at the flanks of the last line row ("one pair left, one right")
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
from tools.nn.operative import army as AM

SIZE = {"line": 2, "anti": 2, "shooter": 3, "fast": 2, "sniper": 10, "art": 10, "lord": 1}   # operative/army.plan
GAP_M = 2.0               # between two groups side by side (their outer formations: config unit_gap_m)


def roles(cards):
    return AM.roles(cards)


def groups(rl, rows):
    """[(role, [rows])]: rows (in slot order) of my units, rl their roles."""
    by = {}
    for j, r in zip(rows, rl):
        by.setdefault(r, []).append(j)
    out = []
    for r in ("line", "anti", "shooter", "sniper", "art", "fast", "lord"):
        js = by.get(r, [])
        k = SIZE[r]
        out += [(r, js[i:i + k]) for i in range(0, len(js), k)]
    return out


def pack(ws, grid, half, gap=GAP_M):
    """[s] the centres of groups of widths ws side by side on the cells' centres grid (sorted, my zone's s axis), each
    clear of the one before by gap, the row centred; None if it does not fit within +-half."""
    if not ws:
        return []
    out, right = [], -1e9
    for w in ws:
        c = next((g for g in grid if g - w / 2 >= right + gap and g - w / 2 >= -half - 1e-6), None)
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
    left = [i for i, w in row if w == "left"][::-1]
    return left + [i for i, w in row if w == "mid"] + [i for i, w in row if w == "right"]


def _rows(gs, fw, grid, half):
    """[[(group index, place)]] the rows from the front: place 'mid' / 'left' / 'right' (module doc)."""
    rows = [[]]

    def fits(r, i, where):
        if r >= len(rows):
            return True
        seq = _seq(rows[r] + [(i, where)])
        return pack([fw[k] for k in seq], grid, half) is not None

    def put(r, i, where):
        while not fits(r, i, where):
            r += 1
        while r >= len(rows):
            rows.append([])
        rows[r].append((i, where))
        return r

    line = [i for i, (r, _) in enumerate(gs) if r == "line"]
    anti = [i for i, (r, _) in enumerate(gs) if r == "anti"]
    ranged = [i for i, (r, _) in enumerate(gs) if r in ("sniper", "shooter")]
    back = [i for i, (r, _) in enumerate(gs) if r in ("art", "lord")]
    fast = [i for i, (r, _) in enumerate(gs) if r == "fast"]
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


def layout(gs, fw, fd, s_grid, f_grid, front, edge, attack):
    """[(s, f)] per group of gs (module doc); fw, fd the groups' footprints (strategist/deploy.footprint); s_grid,
    f_grid the cells' centres inside my zone along its axes (the walk puts a zone's centre on a cell: the groups go
    on whole cells, clear of each other); front my zone's width, edge the margin inside it."""
    half = front / 2 - edge
    rows = _rows(gs, fw, sorted(s_grid), half)
    out = [None] * len(gs)
    fs = sorted(f_grid, reverse=True)                # the rows' places from the front
    n = len(rows)
    if not attack and n <= len(fs):
        fs = fs[len(fs) - n:]                        # 'hold': the block at the back
    for k, row in enumerate(rows):
        seq = _seq(row)
        ss = pack([fw[i] for i in seq], sorted(s_grid), half) or [0.0] * len(seq)
        f = fs[min(k, len(fs) - 1)] if fs else 0.0
        for i, s in zip(seq, ss):
            out[i] = (s, f)
    return out


def order(gs, places):
    """The groups' numbers: the walk places them in this order - the front rows first, in each the centre outwards."""
    return sorted(range(len(gs)), key=lambda i: (-places[i][1], abs(places[i][0])))
