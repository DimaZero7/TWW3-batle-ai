"""Wins by faction and by matchup: the imbalance between the factions in an evaluation or a gate.

Plain numpy (no torch): tools.nn.gate uses it in the project's .venv.

    group(won, own, enemy, attacks, gold) -> {"factions": {"EMP": {"attack": cell, "defend": cell}},
                                             "matchups": {"EMP-SKV": cell}}

A cell: {games, wins, win_rate} (+ gold_destroyed, gold_lost, gold_ratio, gold_trade with gold;
+ margin (the mean signed margin, tools/nn/train/skill.py) with `margin`; + no_result with `void`).
A matchup is our faction first: EMP-SKV is our Empire against Skaven.
"""
import numpy as np

SHORT = {"wh_main_emp_empire": "EMP", "wh2_main_skv_skaven": "SKV"}
ROLES = ("attack", "defend")


def short(faction):
    """'EMP', 'SKV'; an unknown faction: the end of its key."""
    return SHORT.get(faction, (faction or "?")[-8:])


def _order(code):
    codes = list(SHORT.values())
    return (codes.index(code) if code in codes else len(codes), code)


def _cell(sel, won, gold=None, void=None, margin=None):
    n = int(sel.sum())
    out = {"games": n, "wins": int(won[sel].sum()), "win_rate": float(won[sel].mean()) if n else None}
    if gold is not None and n:
        own, enemy, bud = (np.asarray(g, dtype=float)[sel] for g in gold)
        out.update({"gold_destroyed": float(enemy.mean()), "gold_lost": float(own.mean()),
                    "gold_ratio": float(enemy.sum() / max(1e-9, own.sum())),
                    "gold_trade": float(((enemy - own) / np.maximum(bud, 1e-9)).mean())})
    if margin is not None and n:
        out["margin"] = float(np.asarray(margin, dtype=float)[sel].mean())
    if void is not None:
        out["no_result"] = int(void[sel].sum())
    return out


def group(won, own, enemy, attacks, gold=None, void=None, margin=None):
    """won, attacks: [n] bool (our side won, our side attacked); own, enemy: [n] faction keys (ours, the
    opponent's); gold: (own gold lost, enemy gold destroyed, budget), [n] each, or None; void: [n] bool,
    battles without an outcome (counted in games, not as wins); margin: [n] signed margins (skill.margin)
    or None. -> {"factions", "matchups"}."""
    won, attacks = np.asarray(won, dtype=bool), np.asarray(attacks, dtype=bool)
    a = np.array([short(f) for f in own], dtype=object)
    e = np.array([short(f) for f in enemy], dtype=object)
    void = None if void is None else np.asarray(void, dtype=bool)
    factions = {}
    for f in sorted(set(a), key=_order):
        mine = a == f
        factions[f] = {r: _cell(mine & (attacks if r == "attack" else ~attacks), won, void=void) for r in ROLES}
    matchups = {}
    for f, g in sorted(set(zip(a, e)), key=lambda p: (_order(p[0]), _order(p[1]))):
        matchups[f"{f}-{g}"] = _cell((a == f) & (e == g), won, gold, void, margin)
    return {"factions": factions, "matchups": matchups}


def _f(v, fmt):
    return "-" if v is None else fmt.format(v)


def text(g):
    """One line of a group(): 'EMP attack 3/4, defend 1/4; SKV ... | EMP-SKV 2/4 0.50 gold 1.20; ...'."""
    if not g:
        return "-"
    fs = "; ".join(f"{f} " + ", ".join(f"{r} {c['wins']}/{c['games']}" for r, c in roles.items() if c["games"])
                   for f, roles in g.get("factions", {}).items())
    ms = "; ".join(f"{m} {c['wins']}/{c['games']} {_f(c['win_rate'], '{:.2f}')}"
                   + (f" gold {_f(c.get('gold_ratio'), '{:.2f}')}" if "gold_ratio" in c else "")
                   + (f" ({c['no_result']} no result)" if c.get("no_result") else "")
                   for m, c in g.get("matchups", {}).items())
    return f"{fs} | {ms}"


def rows(groups):
    """[(title, cells)]: the "by faction" block over several group() results (minutes of a trend, before
    and after); a cell per result: 'win attack / defend' of our faction, 'win / gold ratio' of a matchup.
    The battle counts (the same battles every time) go to the title, from the last result."""
    groups = [g or {} for g in groups]
    out = []
    facs = sorted({f for g in groups for f in g.get("factions", {})}, key=_order)
    for f in facs:
        last = next(g["factions"][f] for g in reversed(groups) if f in g.get("factions", {}))
        cells = []
        for g in groups:
            c = g.get("factions", {}).get(f, {})
            cells.append(" / ".join(_f(c.get(r, {}).get("win_rate"), "{:.3f}") for r in ROLES))
        out.append((f"{f} win attack / defend (n {last['attack']['games']}/{last['defend']['games']})", cells))
    pairs = sorted({m for g in groups for m in g.get("matchups", {})},
                   key=lambda m: tuple(_order(x) for x in m.split("-", 1)))
    for m in pairs:
        last = next(g["matchups"][m] for g in reversed(groups) if m in g.get("matchups", {}))
        cells = [f"{_f(c.get('win_rate'), '{:.3f}')} / {_f(c.get('gold_ratio'), '{:.2f}')}"
                 for c in (g.get("matchups", {}).get(m, {}) for g in groups)]
        out.append((f"{m} win / gold ratio (n {last['games']})", cells))
    return out
