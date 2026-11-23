"""The seed registry (tmp/engine/design_merged.md section 4): every card's seed ranges in ONE registry, twin families
assigned to splits once, before any fit (not learn.held_split's per-battle sample, learn.py:407-412).

A range is an entry {name, card, split, gen, key, lo, hi, subs, why}: gen - the generator its numbers seed (two ranges
of different generators never meet: a scene seed and a Random stream seed are different identities); key - the recipe
or axis inside a generator (None: the whole generator); subs - the realized retry seeds the generator may move to
(scenes_o1.scene: seed + k x 10^6, k <= substitutes, scenes_o1.py:169-183; written into the spec as seed_used).

    registry()             the entries of T0-1 / T0-2 (imitate.seed_ranges' train and kept sets at their exact recipe
                           ranges, learn.round_specs' counts) and O0-1 (scenes_o1: taught 71000+, held 72000 + 10000 j,
                           G 73000 / G_hold 73500; the fresh train / dev / V / probe ranges 40e6-80e6, brief_B) and the
                           chunk torch seeds / Python streams apart
    overlaps(entries)      [(a, b)] - same generator, same or unkeyed key, intervals (substitutes included) meeting;
                           two entries of one card group's one split never leak (a T0-1 twin's retry bases + t x 100003
                           reach the next rounds' train ranges)
    check(entries)         halts (SeedHalt) on any overlap
    write(path) / load(path)   the registry file (sha256 enters the run key); load: a path that does not exist halts
    assert_starts(entries, cards, split, starts)   every realised start (generator, key, value) of a chunk inside the
                           split's ranges of the cards (the collector, once a chunk) - else SeedHalt
    identity(spec), family(spec), spec_sha(spec)   a built battle's identity (generator, key, realized seed; a seedless
                           general: its spec's sha, never -1), its twin family (the base's identity: an S2N twin shares
                           its S2's seed; a T0-1 twin its base's realized (recipe, seed); an O0-1 twin its cause scene's
                           seed) and the canonical playable spec's sha (annotations stripped; O0's mission / opponent
                           fields of 'o1' kept)
    check_specs({split: [spec]})   halts on an identity, a spec sha or a family in two splits
"""
import hashlib
import json
from pathlib import Path

SUBSTITUTE = 1_000_000          # (net/operative/scenes_o1.py SUBSTITUTE)
B_BLOCK = 1_000_000             # (brief_B: a fresh O0-1 range's base seeds below this, substitutes up to +5 x 10^6)
ANNOTATIONS = ("recipe", "family", "twin_of", "held", "seed", "seed_used", "axes", "binds", "kind", "name", "_why",
               "note", "s2", "roles", "o1")


O1_ANNOTATIONS = ("cause", "twin", "held", "probe")   # (scenes_o1: the labels; att, enemy, sally_m, comer, t_dec, pose,
#                                                    starts, goals play)


class SeedHalt(RuntimeError):
    pass


RETRY_STEP = 100003             # (net/tactic/scenes.py:907: a T0-1 twin's base redrawn at seed + t x 100003)


def E(name, card, split, gen, lo, hi, key=None, subs=0, why="", step=SUBSTITUTE):
    s = [(lo + step * k, hi + step * k) for k in range(1, int(subs) + 1)]
    return {"name": name, "card": card, "split": split, "gen": gen, "key": key, "lo": int(lo), "hi": int(hi),
            "subs": s, "why": why}


TWIN_RECIPES = ("N1", "N2", "E1", "E3")      # (net/tactic/scenes.py TWINS)
_RETRY_WHY = ("brief_A3 R2: the generator permits t < scenes.c()['redraw'] (40) retries (net/tactic/scenes.py:905-911; "
              "t = 4 is ~1 % of twins); every retry t >= 1 of the training rounds, V, the exam sets S, F, H and the "
              "probe's scenes is REALISED here (scenes.make on each twin seed) and registered as its exact base, keyed "
              "by its base recipe - a registered range of every reachable t would cross set F and the exam seeds "
              "(50 meets at t <= 39); a realised base inside another split's range is a real leak and halts check()")


_MEMO = {}


def _twin_of(rec, sd, kw):
    from net.tactic import scenes as SN
    k = (rec, int(sd), json.dumps(kw, sort_keys=True))
    if k not in _MEMO:
        _MEMO[k] = tuple(SN.make(rec, sd, **kw)["twin_of"])
    return _MEMO[k]


def _exact(name, card, split, by, why):
    """{key: {seed}} -> entries, one a key, each seed its own [s, s + 1)."""
    out = []
    for b, xs in sorted(by.items()):
        xs = sorted(xs)
        out.append({"name": f"{name} {b} (realised)", "card": card, "split": split, "gen": "tactic.make", "key": b,
                    "lo": xs[0], "hi": xs[0] + 1, "subs": [[x, x + 1] for x in xs[1:]], "why": why})
    return out


def _realised(name, card, split, items):
    """Entries of the twins' bases as realised by scenes.make on items [(recipe, seed, make kwargs)] - t = 0 and every
    retry t >= 1 - one entry a base recipe (key), each base its own [s, s + 1) (_RETRY_WHY)."""
    by = {}
    for rec, sd, kw in items:
        b, bs = _twin_of(rec, sd, kw)
        by.setdefault(b, set()).add(int(bs))
    return _exact(f"{name} twin bases", card, split, by,
                  "scenes.make: a twin's base at seed + t x 100003 (t = 0 its own seed), as realised (_RETRY_WHY)")


def _t0():
    """T0-1 / T0-2 ranges at recipe resolution (imitate.seed_ranges' supersets split by recipe)."""
    from net.stage0 import cfg as scfg
    from net.tactic import imitate as IM
    from net.tactic import learn as LN
    st = scfg()["drills"]
    at = st["attack"]
    L = IM.L()
    sd = L["seeds"]
    out = []
    mix = at["mix"]
    n = int(at["sets"]["round"]["scenes"])
    counts = {r: max(1, round(v * n)) for r, v in mix.items() if r != "E_on_top"}
    e = max(1, round(float(mix.get("E_on_top", 0.0)) * n / 3))
    counts.update({"E1": e, "E2": e, "E3": e})
    rounds = int(L["rounds"])
    for k in range(rounds):
        s0 = int(sd["scenes"]) + int(sd["round_step"]) * k
        for j, (rec, m) in enumerate(counts.items()):
            out.append(E(f"T0-1 scenes r{k + 1} {rec}", "t0_1", "train", "tactic.make", s0 + 10000 * j,
                         s0 + 10000 * j + int(m), key=rec, why="learn.round_specs"))
            if rec in TWIN_RECIPES:
                out += _realised(f"T0-1 scenes r{k + 1} {rec}", "t0_1", "train",
                                 [(rec, s0 + 10000 * j + i, {}) for i in range(int(m))])
        s2 = int(LN.SIDE_LEARN["seeds"]["s2"]) + int(LN.SIDE_LEARN["seeds"]["round_step"]) * k
        out.append(E(f"S2 + S2N twins r{k + 1}", "t0_2", "train", "tactic.side", s2, s2 + int(LN.SIDE_LEARN["s2_round"]),
                     why="imitate.side_round (an S2N twin shares its S2's seed)"))
    out.append(E("general (Random seeds)", "t0_1", "train", "random.general", int(sd["general"]),
                 int(sd["general"]) + rounds, why="SP.seeded(general + k)"))
    rg = IM.seed_ranges(rounds)["kept"]
    gen_of = {"V S2 + S2N": "tactic.side", "probe S2 train": "tactic.side", "probe S2 eval": "tactic.side",
              "probe held axes": "tactic.side", "exam taught": "tactic.side", "exam held": "tactic.side"}
    split_of = {"T0-1 S": "exam:S", "T0-1 V": "V", "T0-1 F": "exam:F", "T0-1 H": "exam:H", "V S2 + S2N": "V",
                "probe S2 train": "probe_train", "probe S2 eval": "probe_held", "probe held axes": "probe_held",
                "T0-1 probe scenes (2 rounds)": "probe_train", "T0-1 probe S'": "probe_held",
                "exam taught": "exam:taught", "exam held": "exam:held"}
    sets = at["sets"]
    for nm, (lo, hi) in rg.items():
        if "(Random)" in nm:
            out.append(E(nm, "t0_1", "V" if "general V" in nm else ("exam:G" if "general G" in nm else "probe"),
                         "random.general", lo, hi))
            continue
        base = nm.split(" (")[0].replace("T0-1 ", "")
        if nm in ("T0-1 S", "T0-1 V", "T0-1 F"):
            cnt = sets[base]["counts"]
            for j, (rec, m) in enumerate(cnt.items()):
                out.append(E(f"{nm} {rec}", "t0_1", split_of[nm], "tactic.make", lo + 10000 * j, lo + 10000 * j + int(m),
                             key=rec))
                out += _twins(f"{nm} {rec}", split_of[nm], rec, lo + 10000 * j, int(m))
            continue
        if nm == "T0-1 probe scenes (2 rounds)":
            pr = at["probe"]["seeds"]
            for k in range(2):
                s0 = int(pr["scenes"]) + int(pr["round_step"]) * k
                for j, (rec, m) in enumerate(counts.items()):
                    out.append(E(f"T0-1 probe scenes r{k + 1} {rec}", "t0_1", "probe_train", "tactic.make",
                                 s0 + 10000 * j, s0 + 10000 * j + int(m), key=rec))
                    out += _twins(f"T0-1 probe scenes r{k + 1} {rec}", "probe_train", rec, s0 + 10000 * j, int(m))
            continue
        if nm == "T0-1 probe S'":
            cnt = sets["S"]["counts"]
            for j, (rec, m) in enumerate(cnt.items()):
                out.append(E(f"{nm} {rec}", "t0_1", "probe_held", "tactic.make", lo + 10000 * j, lo + 10000 * j + int(m),
                             key=rec))
                out += _twins(f"{nm} {rec}", "probe_held", rec, lo + 10000 * j, int(m))
            continue
        if nm == "T0-1 H":
            # (exact: scenes.held_specs - the cause scenes and the twins of each held axis, and the twins' bases)
            from net.tactic import scenes as SN
            st = SN.c()["sets"]["H"]
            items, own = [], {}
            for j, (axis, recs) in enumerate(st["axes"].items()):
                for i in range(int(st["cause"])):
                    own.setdefault(recs[i % len(recs)], set()).add(int(st["seed"]) + 10000 * j + i)
                tw = [t for t in ("N1", "N2") if any(b in recs for b in SN.TWINS[t][1])] or ["N1"]
                for i in range(int(st["twin"])):
                    sd_ = int(st["seed"]) + 10000 * j + 5000 + i
                    own.setdefault(tw[i % len(tw)], set()).add(sd_)
                    items.append((tw[i % len(tw)], sd_, {"held": axis, "bases": recs}))
            assert all(lo <= x < hi for v in own.values() for x in v), "H's seeds outside imitate.seed_ranges' H"
            out += _exact(nm, "t0_1", split_of[nm], own, "scenes.held_specs: each held axis's cause scenes and twins")
            out += _realised(nm, "t0_1", split_of[nm], items)
            continue
        out.append(E(nm, "t0_2" if nm in gen_of else "t0_1", split_of[nm], gen_of.get(nm, "tactic.make"), lo, hi,
                     key=None))
    # the chunk torch seeds and the Python streams (apart from the scene seeds: their own generators)
    ch = int(L["chunk"])
    nb = 4096 // ch + 2                      # (a part's chunks at most: 2,713 battles / 256 in round 1 -> 11)
    for k in range(rounds):
        for off, nm in ((0, "scene teacher"), (25, "scene net"), (50, "general teacher"), (75, "general net")):
            b = int(sd["tseed"]) + 100 * k + off
            out.append(E(f"torch chunks r{k + 1} {nm}", "t0_*", "train", "torch.chunk", b, b + min(nb, 25)))
    for nm, b in (("V", int(sd["V_tseed"])), ("V general", int(sd["V_tseed"]) + 50), ("H", int(sd["H_tseed"]))):
        out.append(E(f"torch chunks {nm}", "t0_*", "V" if nm.startswith("V") else "exam:H", "torch.chunk", b, b + 25))
    out.append(E("collect Random(tseed x 1000 + r)", "t0_*", "train", "python.collect", int(sd["tseed"]) * 1000,
                 int(sd["tseed"]) * 1000 + rounds))
    return out


def _o1():
    from net.operative import cfg as ocfg
    from net.operative import scenes_o1 as SO
    S = ocfg()["o1"]["scenes"]
    subs = int(S["substitutes"])
    out = [E("O0-1 taught", "o0_1", "exam:taught", "o1.scene", int(S["seed"]), int(S["seed"]) + int(S["n"]), subs=subs,
             why="scenes_o1.taught_set (twins share the cause scene's seed)")]
    for j, ax in enumerate(SO.HELD_AXES):
        lo = int(S["held_seed"]) + 10000 * j
        out.append(E(f"O0-1 held {ax}", "o0_1", "exam:held", "o1.scene", lo, lo + int(S["n_held"]), key=None, subs=subs,
                     why="scenes_o1.held_set"))
    g = int(S["general_seed"])
    out += [E("O0-1 G", "o0_1", "exam:G", "random.general", g, g + 2, why="scenes_o1.general_set: SP.seeded(seed), "
              "Random(seed + 1)"),
            E("O0-1 G_hold", "o0_1", "exam:G", "random.general", g + 500, g + 502),
            E("O0-1 G_def", "o0_1", "exam:G_def", "random.general", 74000, 74002,
              why="net/operative/attack.py G_def (the orchestrator 22.11): SP.seeded(74000), Random(74001) - B's NEEDS_A 3"),
            E("O0-1 train generals", "o0_1", "train", "random.general", 46_000_000, 47_000_000,
              why="levels/operative.py GENERAL_SEED (the orchestrator 22.11 F4): round r mission attack SP.seeded("
                  "46,000,000 + 10,000 r), battle i 46,000,000 + 10,000 r + i; mission hold the same + 5,000 (n <= 5,000"
                  " each) - B's NEEDS_A 4")]
    for nm, lo in (("train", 40_000_000), ("dev", 50_000_000), ("V", 60_000_000), ("probe_train", 70_000_000),
                   ("probe_held", 80_000_000)):
        out.append(E(f"O0-1 {nm}", "o0_1", nm, "o1.scene", lo, lo + B_BLOCK, subs=subs,
                     why="brief_B: fresh ranges beyond the named exam splits; base seeds < 10^6 a block"))
    t = int(S["tseed"])
    out.append(E("O0-1 torch chunks G_def", "o0_1", "exam:G_def", "torch.chunk", t + 400, t + 500,
                 why="exam_o1.sets_of: tseed + 100 x 4 (G_def) - B's NEEDS_A 3"))
    for k, nm in enumerate(("taught", "held", "G", "G_hold")):
        out.append(E(f"O0-1 torch chunks {nm}", "o0_1", f"exam:{nm}", "torch.chunk", t + 100 * k, t + 100 * (k + 1),
                     why="exam_o1.sets_of: tseed + 100 x the set's place in SETS (taught, held, G, G_hold)"))
    for nm, lo in (("train", 40_000_000), ("V", 60_000_000)):
        out.append(E(f"O0-1 torch chunks {nm}", "o0_1", nm, "torch.chunk", lo, lo + B_BLOCK,
                     why="levels/operative.train: the chunk seeds s0 + k of the round's first scene seed s0 (train: "
                         "SETS['train'] + 10,000 r; V: SETS['V'], + 1 the network's path)"))
    return out


def _twins(name, split, rec, lo, m):
    return _realised(name, "t0_1", split, [(rec, lo + i, {}) for i in range(m)]) if rec in TWIN_RECIPES else []


def _s1():
    """S0-1's ranges (coder C, tmp/drill_s1/c/registry_s1.py; net/drill/cards/s0_1.py SETS, net/strategist/s1_scenes.py
    - no substitute seeds: a rejected draw is redrawn in the scene's own stream; s1_ground.POOLS one ridge pool a split)."""
    from net.strategist import s1_ground as SG
    c = "s0_1"
    W = ("net/drill/cards/s0_1.py SETS; net/strategist/s1_scenes.py (no substitute seeds: a rejected draw is redrawn in "
         "the scene's own stream)")
    out = [E("S0-1 train", c, "train", "s1.scene", 91_000_000, 91_000_000 + 4 * 10_000, why="4 rounds x ROUND_STRIDE; " + W),
           E("S0-1 train general", c, "train", "s1.general", 91_500_000, 91_500_000 + 4 * 10_000, why=W),
           E("S0-1 dev", c, "dev", "s1.scene", 92_000_000, 92_000_000 + 1024, why=W),
           E("S0-1 V", c, "V", "s1.scene", 93_000_000, 93_000_000 + 1024, why=W),
           E("S0-1 V general", c, "V", "s1.general", 93_500_000, 93_500_000 + 1024, why=W),
           E("S0-1 probe_train", c, "probe_train", "s1.scene", 94_000_000, 94_000_000 + 2048, why=W),
           E("S0-1 probe_train general", c, "probe_train", "s1.general", 94_500_000, 94_500_000 + 2048, why=W),
           E("S0-1 probe_held", c, "probe_held", "s1.scene", 95_000_000, 95_000_000 + 2048, why=W),
           E("S0-1 probe_held general", c, "probe_held", "s1.general", 95_500_000, 95_500_000 + 2048, why=W),
           E("S0-1 taught", c, "exam:taught", "s1.scene", 96_000_000, 96_000_000 + 1024,
             why="exam_s1 SEEDS (doubling to 1,024)"),
           E("S0-1 G", c, "exam:G", "s1.general", 97_000_000, 97_000_000 + 2048,
             why="exam_s1 SEEDS (doubling to 1,024 / 2,048)")]
    for j, ax in enumerate(("ground", "size", "band", "distance", "enemy", "shift", "zone")):
        lo = 96_100_000 + 10_000 * j
        out.append(E(f"S0-1 held {ax}", c, "exam:held", "s1.scene", lo, lo + 1024, why="exam_s1 SEEDS + HELD_STRIDE x j"))
    for nm, lo in (("taught", 98_000_000), ("G", 98_200_000), ("flat", 98_300_000)):
        out.append(E(f"S0-1 torch chunks {nm}", c, f"exam:{nm}", "torch.chunk", lo, lo + 1000,
                     why="exam_s1 TSEED (every arm of a set the same chunk seeds)"))
    out.append(E("S0-1 torch chunks held", c, "exam:held", "torch.chunk", 98_100_000, 98_100_000 + 7000,
                 why="exam_s1 TSEED held + 1000 x the axis"))
    for pool, (s0, n) in SG.POOLS.items():
        out.append(E(f"S0-1 ground pool {pool}", c, pool if pool in ("train", "dev", "V", "probe_train", "probe_held")
                     else f"exam:{'held' if pool.startswith('held') else pool}", "s1.ground", s0, s0 + n,
                     why="net/strategist/s1_ground.py POOLS: one ridge pool a split"))
    return out


def registry():
    """(module doc); DRILL_MAKE_CACHE=<file> keeps the realised twins between runs (a check's iteration only)."""
    import os
    import pickle
    f = os.environ.get("DRILL_MAKE_CACHE")
    if f and Path(f).is_file():
        _MEMO.update(pickle.loads(Path(f).read_bytes()))
    try:
        return _t0() + _o1() + _s1()
    finally:
        if f:
            Path(f).write_bytes(pickle.dumps(dict(_MEMO)))


def _group(card):
    return "t0" if str(card).startswith("t0_") else str(card)


def _meet(a, b):
    if a["gen"] != b["gen"]:
        return False
    if a["split"] == b["split"] and _group(a["card"]) == _group(b["card"]):
        return False                                   # (one card group's one split: no leak - the retry bases)
    if a["key"] is not None and b["key"] is not None and a["key"] != b["key"]:
        return False
    ia = [(a["lo"], a["hi"])] + [tuple(x) for x in a["subs"]]
    ib = [(b["lo"], b["hi"])] + [tuple(x) for x in b["subs"]]
    return any(lo < hi2 and lo2 < hi for lo, hi in ia for lo2, hi2 in ib)


KNOWN_MEETS = {
    ('T0-1 scenes r1 N1 twin bases A1 (realised)', "T0-1 probe S' N1 twin bases A1 (realised)"),
    ('T0-1 scenes r1 N1 twin bases A3 (realised)', "T0-1 probe S' N1 twin bases A3 (realised)"),
    ('T0-1 scenes r2 N1 twin bases A6 (realised)', "T0-1 probe S' N1 twin bases A6 (realised)"),
    ('T0-1 S N1 twin bases A2 (realised)', 'T0-1 probe scenes r1 N1 twin bases A2 (realised)'),
    ('T0-1 S N2 twin bases A1 (realised)', 'T0-1 V N2 twin bases A1 (realised)'),
    ('T0-1 S E1 twin bases A1 (realised)', 'T0-1 V A1'),
    ('T0-1 S E3 twin bases A3 (realised)', 'T0-1 V A3'),
    ('T0-1 probe scenes r1 N1 twin bases A6 (realised)', "T0-1 probe S' N1 twin bases A6 (realised)"),
    ('T0-1 probe scenes r2 N1 twin bases A1 (realised)', "T0-1 probe S' N1 twin bases A1 (realised)"),
    ('T0-1 probe scenes r2 N1 twin bases A2 (realised)', "T0-1 probe S' N1 twin bases A2 (realised)"),
    ('T0-1 probe scenes r2 N1 twin bases A6 (realised)', "T0-1 probe S' N1 twin bases A6 (realised)"),
}
_KNOWN_WHY = ("brief_A3 R2, the realised twins' bases (22.11, tmp/engine/a/r3/reg_try.out): 11 exact meets, each a "
              "twin's base (redrawn at seed + t x 100003) that is another set's scene or base - none between the "
              "training rounds and V or an exam set the engine reads (taught, held, G, S, F, H); 4 are training rounds "
              "1-2 against T0-1's legacy probe read S' (the legacy probe, not run by the engine), 7 between T0-1's "
              "exam S, V and its legacy probe sets. Listed by name: any other meet halts; these print")


def overlaps(entries, known=False):
    m = [(a["name"], b["name"]) for i, a in enumerate(entries) for b in entries[i + 1:] if _meet(a, b)]
    return [x for x in m if (x in KNOWN_MEETS) == known]


def check(entries):
    bad = overlaps(entries)
    if bad:
        raise SeedHalt(f"seed ranges meet: {bad[:10]}{' ...' if len(bad) > 10 else ''} ({len(bad)})")
    return entries


def write(path, entries=None):
    entries = check(entries if entries is not None else registry())
    text = json.dumps({"entries": entries}, indent=1)
    from net.drill.store import atomic_text
    atomic_text(text, path)
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    """The registry's entries; a path that does not exist is an error (the registry is mandatory)."""
    p = Path(path)
    if not p.is_file():
        raise SeedHalt(f"the seed registry {path}: no such file (the registry is mandatory)")
    return check(json.loads(p.read_text(encoding="utf-8"))["entries"])


def _inside(e, gen, key, v):
    if e["gen"] != gen or (e["key"] is not None and key is not None and e["key"] != key):
        return False
    return any(lo <= v < hi for lo, hi in [(e["lo"], e["hi"])] + [tuple(x) for x in e["subs"]])


def assert_starts(entries, cards, split, starts):
    """Every realised start (gen, key, value) inside an entry of `split` of one of `cards` (a card group's own,
    't0_*' the T0 shared ones); else SeedHalt naming the first ones outside (once a chunk: cheap)."""
    own = [e for e in entries if e["split"] == split and (e["card"] in cards or (e["card"] == "t0_*" and any(
        str(c).startswith("t0_") for c in cards)))]
    bad = [s for s in starts if not any(_inside(e, *s) for e in own)]
    if bad:
        raise SeedHalt(f"{len(bad)} realised starts outside the registry's {split} ranges of {sorted(cards)}: {bad[:5]}")
    return len(starts)


# ------------------------------------------------------------------ built battles
def _gen_of(sp):
    if sp.get("kind") == "s1":                         # (S0-1: net/strategist/s1_scenes - its own generators)
        return "s1.general" if (sp.get("s1") or {}).get("general") else "s1.scene"
    if sp.get("kind") == "o1":
        return "o1.scene"
    if sp.get("recipe") in ("S2", "S2N"):
        return "tactic.side"
    if sp.get("kind") == "general" or "recipe" not in sp:
        return "random.general"
    return "tactic.make"


def identity(sp):
    g = _gen_of(sp)
    if "seed" not in sp and "seed_used" not in sp:
        return (g, sp.get("recipe"), "spec:" + spec_sha(sp))   # (a seedless general: its spec, never -1)
    if g in ("s1.scene", "s1.general"):                # (= net/strategist/s1_scenes.identity)
        return (g, sp.get("twin"), int(sp["seed"]))
    if g == "o1.scene":
        return (g, (sp.get("o1") or {}).get("twin"), int(sp.get("seed_used", sp["seed"])))
    if g == "tactic.side":
        return (g, sp["recipe"], int(sp["seed"]))
    return (g, sp.get("recipe"), int(sp.get("seed", -1)))


def family(sp):
    g = _gen_of(sp)
    if g in ("s1.scene", "s1.general"):                # (= s1_scenes.family: a twin and its cause scene)
        return (g, int(sp["seed"]))
    if g in ("o1.scene", "tactic.side"):
        return (g, int(sp.get("seed_used", sp["seed"])) if g == "o1.scene" else int(sp["seed"]))
    tw = sp.get("twin_of")
    if tw:
        return ("tactic.make", tw[0], int(tw[1]))
    return identity(sp)


def spec_sha(sp):
    """The canonical playable spec's sha (annotations stripped; keys sorted)."""
    x = {k: v for k, v in sp.items() if k not in ANNOTATIONS}
    if isinstance(sp.get("o1"), dict):                 # (O0's mission / opponent fields are playable: kept)
        x["o1"] = {k: v for k, v in sp["o1"].items() if k not in O1_ANNOTATIONS}
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def check_specs(by_split):
    """{split: [spec]} -> {split: n}; halts on an identity, a spec sha or a family in two splits."""
    seen = {"identity": {}, "spec": {}, "family": {}}
    bad = []
    for split, specs in by_split.items():
        for sp in specs:
            for kind, v in (("identity", identity(sp)), ("spec", spec_sha(sp)), ("family", family(sp))):
                s0 = seen[kind].setdefault(v, split)
                if s0 != split:
                    bad.append(f"{kind} {v} in {s0} and {split}")
    if bad:
        raise SeedHalt(f"{len(bad)} in two splits: {bad[:6]}")
    return {k: len(v) for k, v in by_split.items()}


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "net/drill/registry.json"
    print(write(p))
