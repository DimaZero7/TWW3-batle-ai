"""Deployment of one side: melee lines in front, missile lines behind them, the lord behind all.

Positions are those of config/nn/arenas.json: `forward` towards the enemy, `lateral` to the
right, a unit's point is the centre of its front (the formation stands behind it); forward 0 is
the side's front line, gap_m from the enemy's. A line is centred on lateral 0; the most
expensive units stand in its middle. Everything stays inside the deployment zone of
tools/nn/scenario.py: forward from front_m - deployment_m to front_m, lateral within
+-deployment_m / 2.

Plain python: no numpy, no torch.
"""
LATERAL_GAP_M = 6.0      # between units of a line (the arenas' spearmen: 30 m wide, 36 m apart)
ROW_GAP_M = 15.0         # between the back of a line and the front of the next
MARGIN_M = 5.0           # from the zone's side edges
ZONE_FRONT_M = 40.0      # the zone reaches this far in front of forward 0 (tools/nn/scenario.py)


def _centre_out(units):
    """Units in a line from left to right, the most expensive in the middle."""
    line = []
    for i, u in enumerate(sorted(units, key=lambda u: (-u.cost, u.slot))):
        if i % 2:
            line.insert(0, u)
        else:
            line.append(u)
    return line


def _rows(units, room):
    """Units split into lines no wider than room, in order."""
    rows, row, width = [], [], 0.0
    for u in units:
        need = u.width + (LATERAL_GAP_M if row else 0.0)
        if row and width + need > room:
            rows.append(row)
            row, width, need = [], 0.0, u.width
        row.append(u)
        width += need
    if row:
        rows.append(row)
    return rows


def place(lord, units, deployment_m=300.0):
    """[unit dict of an arena side] for the lord and units (pools.Unit), lord first:
    {"slot", "key", "men", "forward", "lateral", "width", "general"}."""
    room = deployment_m - 2 * MARGIN_M
    melee = sorted((u for u in units if not u.missile), key=lambda u: (-u.cost, u.slot))
    missile = sorted((u for u in units if u.missile), key=lambda u: (-u.cost, u.slot))
    rows = _rows(melee, room) + _rows(missile, room)
    placed, front = [], 0.0
    for row in rows:
        line = _centre_out(row)
        total = sum(u.width for u in line) + LATERAL_GAP_M * (len(line) - 1)
        left = -total / 2
        for u in line:
            placed.append((u, front, left + u.width / 2))
            left += u.width + LATERAL_GAP_M
        front -= max(u.depth for u in row) + ROW_GAP_M
    assert front >= ZONE_FRONT_M - deployment_m, f"the lines do not fit the zone ({front} m)"
    counts, out = {}, [{"slot": "lord", "key": lord.key, "men": lord.men, "forward": round(front if rows else 0.0, 1),
                        "lateral": 0.0, "width": lord.width, "general": True}]
    for u, fwd, lat in placed:
        counts[u.slot] = counts.get(u.slot, 0) + 1
        out.append({"slot": f"{u.slot}_{counts[u.slot]}", "key": u.key, "men": u.men, "forward": round(fwd, 1),
                    "lateral": round(lat, 1), "width": u.width})
    return out


def footprint(unit, depth):
    """(forward min, forward max, lateral min, lateral max) a placed unit covers, m."""
    return (unit["forward"] - depth, unit["forward"], unit["lateral"] - unit["width"] / 2,
            unit["lateral"] + unit["width"] / 2)
