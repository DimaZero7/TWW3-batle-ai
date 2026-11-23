"""Exams M2 (tmp/engine/design_merged.md section 6): the tables of a card's exam from the players' raw files, the
paired reads at the judge's windows, balanced blocks.

    at(d, anchor, w)      [B] plain health at anchor + w s (the series' step, capped at its end) and the cut share
    reads(d, anchor, windows)   {"i_<w>": [B] for each window w (tactic: C's first contact + 300 s - legacy -, + 450 s),
                          "iii": [B] the battle's end (hp_trade), "cut_<w>"} (design section 0: the flank trick's cost
                          opens at contact + 180..360 s)
    paired(a, b, sel)     {diff, two_errors, n} of a - b over sel (finite pairs; n < 2: None)
    pool(parts)           a pooled paired read of several groups (each part's variance from its two errors)
    tables(D, names, card)   per set: battles, cut shares, the comer battles, pairs {pair: {stratum: {i_300, i_450,
                          iii_end, orders}}}, combos (card.COMBOS: a paired linear combination of arms per battle, e.g.
                          N - (TEACH + TEACH2) / 2 for 'half the gap'), orders a unit-minute, attack orders a battle,
                          execution, binds, n_battles - the card gives players, pairs, strata (groups_of), comer,
                          execution and binds. The arms are paired by battle key, never by position: every arm's file
                          must carry the same battle keys (index, its per-battle meta, the chunk seeds) and the same
                          simulator key, as many battles as the set's meta - else ExamRefused. The seed registry's
                          checks run first (seeds.load: a missing registry file is an error)
    Blocks(path)          blocks of 256 battles balanced by role and axis, registered append-only (a re-registration
                          with other seeds, a duplicate seed, an unbalanced block 0 halt); grow(read, line, start, cap) -
                          the doubling 256 -> 512 -> 1,024: a piece whose interval straddles its line asks for the
                          doubled set, up to the cap; decide(diff, two_errors, n, line) - still straddling at the cap =
                          not decided
"""
import json
import math
from pathlib import Path

SERIES_S = 5          # (net/tactic watch PinWatch.SERIES_S: the health series' step, s)
WINDOWS = (300, 450)  # (design section 0: the tactic judge reads + 300 s - legacy - and + 450 s, and the end)


def _series_step():
    try:
        from net.tactic import exam_pin as EP
        return EP.W.PinWatch.SERIES_S
    except Exception:                                    # (no simulator importable: the constant)
        return SERIES_S


def at(d, anchor, w):
    step = _series_step()
    out, cut = [], 0
    for b, ser in enumerate(d["hp_series"]):
        i = int(round((anchor[b] + w) / step))
        cut += i > len(ser) - 1
        out.append(ser[min(i, len(ser) - 1)])
    return out, cut / max(len(d["hp_series"]), 1)


def reads(d, anchor, windows=WINDOWS):
    r = {"iii": d["per_battle"]["hp_trade"]}
    for w in windows:
        r[f"i_{w}"], c = at(d, anchor, w)
        r[f"cut_{w}"] = round(c, 4)
    return r


def mean(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    return round(sum(v) / len(v), 4) if v else float("nan")


def paired(a, b, sel=None):
    idx = range(len(a)) if sel is None else sel
    d = [a[i] - b[i] for i in idx if math.isfinite(a[i]) and math.isfinite(b[i])]
    n = len(d)
    if n < 2:
        return None
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return {"diff": round(m, 4), "two_errors": round(2 * sd / math.sqrt(n), 4), "n": n}


def pool(parts):
    parts = [p for p in parts if p]
    if not parts:
        return None
    n = sum(p["n"] for p in parts)
    m = sum(p["diff"] * p["n"] for p in parts) / n
    ss = sum(((p["two_errors"] / 2) ** 2 * p["n"]) * (p["n"] - 1) + p["n"] * (p["diff"] - m) ** 2 for p in parts)
    sd = math.sqrt(ss / max(n - 1, 1))
    return {"diff": round(m, 4), "two_errors": round(2 * sd / math.sqrt(n), 4), "n": n}


def pool_all(parts):
    """pool, None (not judged) when any part is missing."""
    return None if any(p is None for p in parts) else pool(parts)


def fmt(p):
    return "-" if p is None else f"{p['diff']:+.4f} ± {p['two_errors']:.4f} ({p['n']})"


class ExamRefused(RuntimeError):
    pass


REGISTRY = Path(__file__).resolve().parent / "registry.json"


def battle_keys(nm, d):
    """([(set, index, the arm's per-battle meta)], (chunk seeds, simulator key)) of one arm's file."""
    mt = d.get("meta")
    if not isinstance(mt, list) or len(mt) != len(d["hp_series"]):
        raise ExamRefused(f"set {nm}: an arm's file without a per-battle meta of its {len(d['hp_series'])} battles - "
                          "no battle key to pair on")
    # (brief_A3 V5: the legacy arm files carry per battle only {enemy, n, n_theirs} - 12 distinct over 1,024 - so the
    # key adds the health series' first point (t = 5 s, before any decision acts: equal across every arm of every
    # exam on disk, 0 of 51,392 differ - exam3, after_side, the E1 goldens); a per-battle spec sha in the meta, when an
    # arm file carries one (the engine's own exam play), enters the key through the meta)
    hp0 = [round(float(s[0]), 6) if s else None for s in d["hp_series"]]
    return [(nm, b, json.dumps(m, sort_keys=True), hp0[b]) for b, m in enumerate(mt)], (tuple(d.get("seeds") or ()),
                                                                                       d.get("sim_key"))


def check_arms(nm, pl, n_meta):
    """(module doc) ExamRefused on a missing key, another length, other seeds or another simulator."""
    ref = None
    for p, d in pl.items():
        keys, run = battle_keys(nm, d)
        if len(keys) != n_meta:
            raise ExamRefused(f"set {nm}: arm {p} has {len(keys)} battles, the set's meta {n_meta}")
        if ref is None:
            ref = (p, keys, run)
            continue
        if run != ref[2]:
            raise ExamRefused(f"set {nm}: arm {p}'s chunk seeds / simulator key {run} differ from {ref[0]}'s {ref[2]}")
        bad = [i for i, (a, b) in enumerate(zip(keys, ref[1])) if a != b]
        if bad:
            raise ExamRefused(f"set {nm}: arm {p}'s battle keys differ from {ref[0]}'s at {len(bad)} battles "
                              f"(first {bad[:5]}) - reordered or of other battles; refused")


def lin(R, coef, read, sel):
    """{diff, two_errors, n} of sum_p coef[p] x R[p][read] per battle over sel (paired)."""
    vals = [sum(c * R[p][read][b] for p, c in coef.items()) for b in sel]
    return paired(vals, [0.0] * len(vals))


def tables(D, names, card, windows=WINDOWS, registry=REGISTRY):
    """(module doc) card: a module with PLAYERS, PAIRS, ANCHOR (the anchor player), groups_of(mt, nm, comer),
    comer(mt, pl), exec_of(d, sel) or None, binds_of(mt) or None, COMBOS (optional)."""
    from net.drill import seeds as SD
    SD.load(registry)
    D = Path(D)
    out = {}
    for nm in names:
        files = {p: D / f"{nm}_{p}.json" for p in card.PLAYERS if (D / f"{nm}_{p}.json").exists()}
        if card.ANCHOR not in files:
            continue
        pl = {p: json.loads(f.read_text(encoding="utf-8")) for p, f in files.items()}
        mf = D / f"{nm}_meta.json"
        mt = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else [{"cause": None}] * len(
            pl[card.ANCHOR]["hp_series"])
        check_arms(nm, pl, len(mt) if mf.exists() else len(pl[card.ANCHOR]["hp_series"]))
        anchor = pl[card.ANCHOR]["per_battle"]["t_contact"]
        R = {p: reads(d, anchor, windows) for p, d in pl.items()}
        cm = card.comer(mt, pl)
        tab = {"battles": len(mt), "cut_i": {p: R[p]["cut_300"] for p in R},
               "cut": {p: {str(w): R[p][f"cut_{w}"] for w in windows} for p in R}, "comer_scenes": cm, "pairs": {},
               "orders_min": {}, "attack_orders_battle": {}, "exec": {}, "n_battles": {}}
        if card.binds_of is not None:
            tab["binds"] = card.binds_of(mt)
        for lab, sel in card.groups_of(mt, nm, set(cm["all"])):
            tab["n_battles"][lab] = len(sel)
            for x, y in card.PAIRS:
                if x in pl and y in pl:
                    row = {"i_300": paired(R[x]["i_300"], R[y]["i_300"], sel)}
                    for w in windows:
                        if w != 300:
                            row[f"i_{w}"] = paired(R[x][f"i_{w}"], R[y][f"i_{w}"], sel)
                    row["iii_end"] = paired(R[x]["iii"], R[y]["iii"], sel)
                    row["orders"] = paired(pl[x]["per_battle"]["game_orders_min"],
                                           pl[y]["per_battle"]["game_orders_min"], sel)
                    tab["pairs"].setdefault(f"{x} - {y}", {})[lab] = row
            for cn, coef in getattr(card, "COMBOS", {}).items():
                if all(p in pl for p in coef):
                    row = {f"i_{w}": lin(R, coef, f"i_{w}", sel) for w in windows}
                    row["iii_end"] = lin(R, coef, "iii", sel)
                    tab.setdefault("combos", {}).setdefault(cn, {})[lab] = row
            for p, d in pl.items():
                tab["orders_min"].setdefault(p, {})[lab] = mean([d["per_battle"]["game_orders_min"][b] for b in sel])
                ao = d["per_battle"].get("attack_orders")
                if ao:
                    tab["attack_orders_battle"].setdefault(p, {})[lab] = mean([ao[b] for b in sel])
            if card.exec_of is not None:
                for p, d in pl.items():
                    if "strikes" in d:
                        tab["exec"].setdefault(p, {})[lab] = card.exec_of(d, sel)
        out[nm] = tab
    return out


def print_reads(out, pairs_prefix=None, labels=None, windows=WINDOWS):
    """The paired reads of every set at each window and the end (design section 0: all three printed)."""
    lines = []
    for nm, tab in out.items():
        if not isinstance(tab, dict) or "pairs" not in tab:
            continue
        lines.append(f"## set {nm} ({tab['battles']} battles) - C's first contact + "
                     + " s | + ".join(str(w) for w in windows) + " s | end | orders")
        for pair, rows in tab["pairs"].items():
            if pairs_prefix and not pair.startswith(pairs_prefix):
                continue
            for lab, r in rows.items():
                if labels and lab not in labels:
                    continue
                cells = [fmt(r.get(f"i_{w}")) for w in windows] + [fmt(r["iii_end"]), fmt(r["orders"])]
                lines.append(f"  {pair} {lab}: " + " | ".join(cells))
    return lines


# ------------------------------------------------------------------ balanced blocks, doubling
BLOCK = 256           # (design section 6: blocks of 256)
CAP = 1024            # (design section 6: a set grows up to 1,024)


class BlockHalt(RuntimeError):
    pass


class Blocks:
    """An append-only register of a set's blocks: {set: [{block, seeds, roles, axes}]} in a JSON file."""

    def __init__(self, path):
        self.path = Path(path)
        self.reg = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def register(self, set_name, block, seeds, roles, axes):
        """Block `block` (0, 1, ..) of set_name: its seeds and their role / axis labels; balanced (each role and axis
        value as often as in block 0 - or evenly when it is block 0); a block re-registered with other seeds halts."""
        if len(seeds) != BLOCK:
            raise BlockHalt(f"{set_name} block {block}: {len(seeds)} battles, a block is {BLOCK}")
        if len(set(seeds)) != len(seeds):
            raise BlockHalt(f"{set_name} block {block}: {len(seeds) - len(set(seeds))} duplicate seeds")
        bal = _balance(roles), _balance(axes)
        if block == 0:
            for nm_, b_ in (("roles", bal[0]), ("axes", bal[1])):
                if len(set(b_.values())) > 1:
                    raise BlockHalt(f"{set_name} block 0: its {nm_} not balanced {b_} (each value as often)")
        have = self.reg.setdefault(set_name, [])
        for b in have:
            if b["block"] == block:
                if b["seeds"] != list(seeds):
                    raise BlockHalt(f"{set_name} block {block} registered again with other seeds")
                return b
        if block != len(have):
            raise BlockHalt(f"{set_name}: block {block} after {len(have)} (append-only)")
        if have and (bal[0] != have[0]["roles"] or bal[1] != have[0]["axes"]):
            raise BlockHalt(f"{set_name} block {block}: not balanced as block 0 ({bal} vs {have[0]['roles']}, "
                            f"{have[0]['axes']})")
        seen = {s for b in have for s in b["seeds"]}
        if seen & set(seeds):
            raise BlockHalt(f"{set_name} block {block}: seeds of an earlier block")
        e = {"block": block, "seeds": list(seeds), "roles": bal[0], "axes": bal[1]}
        have.append(e)
        _atomic_text(json.dumps(self.reg, indent=1), self.path)
        return e


def _atomic_text(text, path):
    """.part, fsync of the written handle, os.replace (store.atomic_text without torch)."""
    import os
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _balance(labels):
    c = {}
    for x in labels:
        c[str(x)] = c.get(str(x), 0) + 1
    return dict(sorted(c.items()))


def straddles(diff, two_errors, line):
    """The interval diff +- two errors contains the line (point passing or not)."""
    return diff - two_errors < line < diff + two_errors


def decide(diff, two_errors, n, line, cap=CAP):
    """'decided' | 'double' (n < cap: the doubled set asked) | 'not decided' (straddling at the cap)."""
    if not straddles(diff, two_errors, line):
        return "decided"
    return "double" if n < cap else "not decided"


def grow(read, line, start=BLOCK, cap=CAP):
    """The doubling (design section 6): read(n) -> (diff, two_errors) on the first n battles; while the interval
    straddles the line the set doubles (256 -> 512 -> 1,024), up to cap. Returns (n, diff, two_errors, steps)."""
    n, steps = start, []
    while True:
        d, e = read(n)
        steps.append({"n": n, "diff": d, "two_errors": e, "straddles": straddles(d, e, line)})
        if not steps[-1]["straddles"] or n >= cap:
            return n, d, e, steps
        n = min(2 * n, cap)
