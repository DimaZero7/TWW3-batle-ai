"""The strategist's sample of the live labelling (game/label/live.py): the person deploys his army in the game's own
deployment phase - places, turns, widths by hand, groups by Ctrl+G, the enemy's groups on the enemy cards, the chat -
and presses the game's "Start battle"; the bridge's last deployment document (bridge/lua/apps/battle/live_label.lua: the
deployment document of services.deploy_document - my units whole, the enemy by its composition, both zones - with the
groups, the selection, the chat) becomes one strategist sample.

    the input       blind, by the companion's own code (bridge/companion/strategist_input.StrategistSeer: the same
                    input the strategist gets in the game and in the simulator)
    the decision    the walk's choices (game/label/strategist.act_of / walk_of, the strategist's own road) of the
                    person's deployment read backwards:
                      comp    his groups (Ctrl+G); my units in none - groups more of at most max_group, in the first free
                              numbers (as the operative's windows do: 'rest')
                      cell    where the group's zone centre goes: the centre of its units' places
                      turn    16 sectors from the bearing my zone -> the enemy's zone: the group's mean facing
                      form    line - the group's units stand in one row (their depth spread along the group's facing under
                              LINE_DEPTH_M); else square
                      verb, pace   the operative's start command: hold / any (the person gives orders after the start;
                              his command is not a choice he makes at the deployment)
    raw             each unit's place, bearing, width as he left it, and per group what was read (centre, mean facing,
                    the spreads) - the places the strategist's code would make from these choices are its own (its tiles),
                    not his: MISSING lists what does not go one to one
"""
import math

import torch

from game.label import human as HU
from game.label import strategist as LS
from net.strategist import cfg as scfg
from net.tactic import cfg as tcfg

LINE_DEPTH_M = 15.0
MISSING = [
    "places inside a group: the strategist's code lays the group's units out in its slots (strategist/deploy.places) - "
    "the person's own places, widths and the order inside the group are kept raw (human_places, edit places, act "
    "places), not as a choice; the strategist has no width head yet (its formula keeps the units' own frontage)",
    "the cell is the one nearest the centre of the group's units (the walk's cells: 64 m); the exact centre and places "
    "are the raw, the check goes by them (strategist.exact_check), not by the cell",
    "the form from the units' spread (line: one row); a loose or a mixed layout reads as square",
    "the operative's start command (verb, pace) is not set at the deployment: hold / any",
]


def laid_slot(doc, name):
    return next((i for i, u in enumerate(doc["units"]) if u["n"] == name), None)


def groups_of(doc):
    g = doc.get("groups") or {}
    own = {int(k): list(n) for k, n in (g.get("own") or [])}
    enemy = {int(k): list(n) for k, n in (g.get("enemy") or [])}
    return own, enemy


def circ_mean(bearings):
    s = sum(math.sin(math.radians(b)) for b in bearings)
    c = sum(math.cos(math.radians(b)) for b in bearings)
    return math.degrees(math.atan2(s, c)) % 360.0


def edit_of(doc, own_groups, base, G=None, max_group=None, sectors=None, dims=None):
    """(edit for strategist.act_of, raw, rest numbers): the person's deployment read backwards (module doc); base the
    bearing my zone -> the enemy's zone (strategist/net.Ctx.base); dims {name: (front, depth)} his formations (formations
    of), else the check estimates them. The edit's 'places' - every unit's exact place (the raw the check goes by)."""
    G = int(G or scfg()["groups"])
    max_group = int(max_group or tcfg()["max_group"])
    S = int(sectors or scfg()["sectors"])
    units = doc["units"]
    slot_of = {u["n"]: i for i, u in enumerate(units)}
    mine = {u["n"]: u for u in units if int(u["side"]) == 1 and (u.get("men") or 0) > 0}
    full = {k: [n for n in names if n in mine] for k, names in own_groups.items() if 1 <= k <= G}
    full = {k: v for k, v in full.items() if v}
    grouped = {n for v in full.values() for n in v}
    rest = [n for n in mine if n not in grouped]
    rest_ks = []
    for i in range(0, len(rest), max_group):
        k = next((k for k in range(1, G + 1) if k not in full), None)
        if k is None:
            break
        full[k] = rest[i:i + max_group]
        rest_ks.append(k)
    dims = dims or {}
    edit = {"groups": [], "verb": "hold", "pace": "any",
            "places": {slot_of[n]: [u.get("x"), u.get("z"), u.get("b"), u.get("ow"), *(dims.get(n) or (None, None))]
                       for n, u in mine.items()}}
    raw = {"units": {n: {"x": u.get("x"), "z": u.get("z"), "b": u.get("b"), "width": u.get("ow")} for n, u in mine.items()},
           "groups": {}}
    for k in sorted(full):
        names = full[k]
        xs = [float(mine[n]["x"]) for n in names]
        zs = [float(mine[n]["z"]) for n in names]
        cx, cz = sum(xs) / len(xs), sum(zs) / len(zs)
        heading = circ_mean([float(mine[n].get("b") or 0.0) for n in names])
        turn = int(round(((heading - float(base)) % 360.0) / (360.0 / S))) % S
        a = math.radians(heading)
        fx, fz = math.sin(a), math.cos(a)                 # the front
        along = [(x - cx) * fx + (z - cz) * fz for x, z in zip(xs, zs)]
        across = [(x - cx) * fz - (z - cz) * fx for x, z in zip(xs, zs)]
        depth = max(along) - min(along)
        front = max(across) - min(across)
        form = "line" if len(names) > 1 and depth < LINE_DEPTH_M else "square"
        edit["groups"].append({"g": k - 1, "members": [slot_of[n] for n in names], "form": form, "turn": turn,
                               "goal": [round(cx, 1), round(cz, 1)]})
        raw["groups"][k] = {"members": names, "cx": round(cx, 1), "cz": round(cz, 1), "heading": round(heading, 1),
                            "turn": turn, "form": form, "front_spread": round(front, 1), "depth_spread": round(depth, 1),
                            "rest": k in rest_ks}
    return edit, raw, rest_ks


def formations_of(doc, laid, params):
    """{name: (front, depth)} my formations as the person left them (the simulator's geometry.dims of the unit at his
    place and width: the companion's state_with)."""
    from bridge.companion import deploy as CD
    from sim import geometry
    places = {u["n"]: (float(u["x"]), float(u["z"]), float(u.get("b") or 0.0), float(u["ow"]))
              for u in doc["units"] if int(u["side"]) == 1 and u.get("ow") and u.get("x") is not None}
    st, _, index = CD.state_with(laid, places, params)
    front, depth = geometry.dims(st.u, params.sim["formation"]["spacing_m"])
    return {n: (float(front[0, index[i]]), float(depth[0, index[i]]))
            for i, n in enumerate(u["n"] for u in laid["units"]) if n in places}


def sample(doc, command, seer, chat=None):
    """(sample, text) of the deployment document (module doc): the sample in the strategist's labelling format
    (game/label/strategist.rebuild reads it), act {} and the errors when the walk refuses the choices."""
    inp, ctx, laid = seer.input(doc, command)
    own, enemy = groups_of(doc)
    edit, raw, rest = edit_of(doc, own, float(ctx.base[0]), dims=formations_of(doc, laid, seer.P))
    for n, u in raw["units"].items():
        fd = (edit["places"].get(laid_slot(doc, n)) or [None] * 6)[4:6]
        u["front"], u["depth"] = fd
    act, errs, notes = {}, [], []
    try:
        act = LS.act_of(edit, inp, ctx.axes, inp["own"].device)
        out, errs = LS.walk_of(act, inp, ctx)
        notes = out.get("notes") or []
    except HU.Refused as e:
        errs = list(e.errors)
    if errs:
        act = {}
    names = [u["n"] for u in doc["units"]]
    human = {"own": {k: v for k, v in sorted(own.items())}, "enemy": {k: v for k, v in sorted(enemy.items())},
             "rest": rest}
    smp = {"level": "strategist", "mode": "live", "where": "game", "sec": 0, "raw": seer.raw, "card": seer.tactic.card,
           "side": torch.ones(1, dtype=torch.long, device=inp["own"].device), "command": inp["command"],
           "mine_zone": ctx.mine, "enemy_zone": ctx.enemy, "bounds": seer.P.map_half, "K": int(inp["own"].shape[1]),
           "act": act, "source": "human", "edit": edit, "human_groups": human, "human_places": raw, "errors": errs,
           "notes": notes,
           "chat": chat or [], "names": names, "missing": MISSING}
    text = {"sec": 0, "until": 0, "level": "strategist", "mode": "live", "where": "game", "source": "human",
            "groups": human, "read": raw["groups"], "edit": edit, "errors": errs, "notes": notes, "chat": chat or [],
            "missing": MISSING}
    return smp, text
