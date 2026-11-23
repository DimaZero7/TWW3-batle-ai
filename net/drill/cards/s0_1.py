"""Card S0-1 'the line on the crest' (the strategist's first drill; tmp/drill_s1/design_merged.md, design_opus 10):
one blind decision a battle at tick 0 - KEEP (the centre line) or 'on the crest' + a pointer over the finder's <= 3
candidates (net/strategist/crest.py); the teacher crest.teach (mission attack -> KEEP; else the largest band if band >=
0.08, band - band(centre) >= 0.05, >= 60 % of its points >= 0.05 - ASSUMPTIONS judged by the exam's bins); arms DEFAULT
(KEEP), DUMB (the foot), TEACH, PLACE (report) over the stack as it is (mission hold: the operative keeps; the tactic's
TEACH, the unit script); the generator net/strategist/s1_scenes.py (captured plains / river + synthetic ridges, twins
flat / shift / low / role, seven held axes, kho_wastes_01 the held ground); the head StratNet1 (net/strategist/s1_net).

    SETS         the fresh seed ranges (generators s1.scene / s1.general / torch.chunk / s1.ground; NEEDS_A.md lists them
                 for the registry): train 91e6 (+ ROUND_STRIDE x round), its general battles 91.5e6, dev 92e6, V 93e6 (+
                 93.5e6), probe_train 94e6 (+ 94.5e6), probe_held 95e6 (+ 95.5e6), the exam's taught 96e6, held 96.1e6 +
                 10,000 j, G 97e6, the exam's torch chunks 98e6..98.31e6 (net/strategist/exam_s1.py)
    start_of(spec)   a built battle's (generator, key, seed) for the registry's start check
    judge(tables, arm)   the exam's pieces (exam_s1.table) as {id, kind, status, cells}
    pieces_before / pieces_after (tables, probe)   TEACH / NET + the label probe's gate (probe_piece)
"""
ID = "S0-1"
WORDS = "the line on the crest"
LEVEL = "strategist"
STAGE = "crest"
PARENT = None
ARMS = {"DEFAULT": "the centre line (KEEP)", "DUMB": "the foot: the legal line of the lowest band",
        "TEACH": "crest.teach", "PLACE": "the teacher's line on the flat twin (report)"}
BELOW = {"operative": "none - mission hold (O0-1 keeps the zone)", "tactic": "TEACH (T0-1's teacher)",
         "unit": "net/unit/scripts.Unit"}
OPPONENT = ("human", "nearest", "hold_shoot (general battles)", "ai_game (held)")
SETS = {"train": 91_000_000, "train_general": 91_500_000, "dev": 92_000_000, "V": 93_000_000, "V_general": 93_500_000,
        "probe_train": 94_000_000, "probe_train_general": 94_500_000, "probe_held": 95_000_000,
        "probe_held_general": 95_500_000, "taught": 96_000_000, "held": 96_100_000, "G": 97_000_000}
_SETS_WHY = ("fresh ranges above every registered one (the registry's highest 81e6, 22.11): no substitute seeds - a "
             "rejected draw is redrawn inside the scene's own Python stream (s1_scenes.TRIES), so a range is its seeds")
ROUND_STRIDE = 10_000
_ROUND_WHY = "a round's scenes start ROUND_STRIDE after the last one's (n <= 10,000 a round); rounds_registered 4"
FILES = ("net/strategist/crest.py", "net/strategist/s1_scenes.py", "net/strategist/s1_ground.py",
         "net/strategist/s1_net.py", "net/strategist/exam_s1.py", "net/strategist/probe_s1.py",
         "net/config/strategist.json", "net/drill/levels/strategist.py", "net/drill/cards/s0_1.py")
MARKS = {"followed (CREST labels)": ("followed", ">=", 0.95), "false (KEEP labels)": ("false", "<=", 0.05)}
_MARKS_WHY = ("design_opus 6 'Exam after': training stops at labels >= 0.95 per class on held seeds - CREST labels "
              "followed (the teacher's candidate) >= 0.95, KEEP labels kept (a crest chosen <= 0.05) - on the UNFORCED "
              "rows (levels/strategist.marks; a forced KEEP is no success, Astra 10); census minima 50 a class and 50 "
              "POINTER rows (CREST labels with >= 2 valid crests: the pointer's own gradient)")
PROBE_ABLATE = {"rows": ("dh",), "place": ("height",),
                "cand": ("band", "band_p25", "share05", "band_slot_min", "delta_band", "rise60", "rise100", "back20",
                         "profile*"),
                "seat": ("att",)}
_PROBE_WHY = ("design_opus 6 'label probe': intact >= 0.9; heights, band and the mission erased <= the majority share + "
              "0.05 on the UNFORCED rows (>= 1 valid candidate; brief_C3 X3); the columns by name (rows: tactic/rows RI; "
              "place / cand / seat: net/strategist/s1_net PLACE_COLS / CAND_COLS / SEAT_COLS; 'x*' a prefix) - "
              "net/strategist/probe_s1.py ablate reads this")


def teacher(spec, c=None):
    from net.strategist import s1_scenes as SS
    return SS.label(spec)


def start_of(spec):
    gen = "s1.general" if spec["s1"].get("general") else "s1.scene"
    return (gen, None, int(spec["seed"]))


def _classes(batch):
    t = batch.teacher
    return {"KEEP": t == 0, "CREST": t > 0}


def _marks(v):
    out = {}
    for name, (k, _, _) in MARKS.items():
        out[name] = {}
        for path in ("T", "N"):
            c = (v.get(path) or {}).get(k)
            out[name][path] = None if c is None else {"share": c["p"], "two_errors": c["two_errors"], "n": c["n"]}
    return out


# ------------------------------------------------------------------ the exam's pieces (exam_s1.table)
_PIECES = (("exec", "skill"), ("sign", "skill"), ("quiet", "safety"), ("noharm", "safety"), ("flat_bit", "safety"),
           ("orders", "safety"), ("freeze", "safety"), ("bins", "skill"), ("place", "safety"), ("value", "report"),
           ("strata", "report"))


def _status(t, kind):
    st = str(t.get("status", "NOT JUDGED"))
    if st.startswith("fail"):
        return "fail"                          # (a report line's STOP fails too: PLACE 'not the height', exam_s1 doc)
    if kind == "report" or st.startswith("reported"):
        return "reported" if not st.startswith("NOT") else st
    return st


def judge(tables, arm="TEACH"):
    """[{id, kind, status, cells}] of exam_s1.table(D, arm) - the held axes one piece each; a missing line or axis is
    NOT JUDGED (a fail of the exam)."""
    from net.strategist import s1_scenes as SS
    res = []
    for pid, kind in _PIECES:
        t = tables.get(pid)
        if t is None:
            res.append({"id": f"S0-1:{pid}:{arm}", "kind": kind, "status": "NOT JUDGED", "cells": {}, "note": "no table"})
            continue
        res.append({"id": f"S0-1:{pid}:{arm}", "kind": kind, "status": _status(t, kind), "cells": t})
    held = tables.get("held") or {}
    for ax in SS.HELD:
        t = held.get(f"held:{ax}")
        res.append({"id": f"S0-1:held:{ax}:{arm}", "kind": "skill", "status": "NOT JUDGED" if t is None else t["status"],
                    "cells": t or {}})
    return res


def pieces(arm):
    from net.drill.cards import Piece
    out = []
    for pid, kind in _PIECES:
        out.append(Piece(id=f"S0-1:{pid}:{arm}", kind=kind, legacy=False, arms=(arm, "DEFAULT", "DUMB", "PLACE"),
                         strata=("taught", "held", "G"), denominators=(pid,), n_min=0,
                         read=lambda tables, stratum, den, p=pid: (tables.get(p), None, 0),
                         line=lambda v, e, p=pid: (v or {}).get("status", "undecided")))
    return out


def probe_piece(path):
    import json
    from pathlib import Path
    f = Path(path) if path else None
    if f is None or not f.exists():
        return {"id": "S0-1:probe", "kind": "safety", "status": "NOT JUDGED", "file": str(f), "note": "no probe result"}
    v = json.loads(f.read_text(encoding="utf-8"))["verdict"]
    st = "NOT JUDGED" if not v["n_ok"] else ("pass" if v["pass"] else "fail")
    return {"id": "S0-1:probe", "kind": "safety", "status": st, "file": str(f), "verdict": v}


def pieces_before(tables, probe=None):
    return judge(tables, "TEACH") + [probe_piece(probe)]


def pieces_after(tables, probe=None):
    return judge(tables, "NET") + [probe_piece(probe)]


def _m2(tables, probe=None):
    return pieces_after(tables, probe)


# ------------------------------------------------------------------ the verdicts (net/drill/verdict.py entries)
GATE_FILE = "value_gate.json"


def verdict_before(D, probe=None):
    """The teacher's exam before (exam_s1 table of TEACH + the label probe): the M2 verdict; the value gate decided
    from taught's cause trade (>= 4 standard errors and >= 0.02 -> 'half the gap', else 'not worse than the teacher')
    and FROZEN in D/value_gate.json - an existing file with another branch is refused (the gate is decided once)."""
    import hashlib
    import json
    from pathlib import Path
    from net.drill import verdict as VD
    from net.strategist import exam_s1 as EX
    D = Path(D)
    t = EX.table(D, "TEACH")
    probe = probe or str(D.parent / "probe" / "probe.json")
    v = VD.m2(pieces_before(t, probe), card=card())
    gate = {"card": ID, "gate": t["value"]["gate"], "hp_trade": t["value"]["hp_trade"], "line": [4.0, 0.02],
            "table_sha": hashlib.sha256(json.dumps(t, sort_keys=True, default=str).encode()).hexdigest()[:16],
            "verdict_before": v["verdict"]}
    f = D / GATE_FILE
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old.get("gate") != gate["gate"]:
            raise SystemExit(f"{f}: the value gate was frozen as {old.get('gate')!r}, now {gate['gate']!r} - refused")
    elif gate["gate"] is not None:
        f.write_text(json.dumps(gate, indent=1, default=str), encoding="utf-8")
    v["value_gate"] = gate
    return v


def _net_vs(t_dir, gate):
    """The value piece after (METHOD step 6): taught's cause scenes, paired by battle - 'half the gap': NET - the
    midpoint of TEACH and DEFAULT lower bound >= 0; 'not worse than the teacher': NET - TEACH lower bound >= -0.005."""
    from net.strategist import exam_s1 as EX
    N, T, Dd = (EX._load(t_dir, "taught", a) for a in ("NET", "TEACH", "DEFAULT"))
    sp = EX.spec_keys(t_dir, "taught")
    ks = [k for k, r in sp.items() if r["cause"] and N and T and Dd and k in N and k in T and k in Dd]
    if gate == "half the gap":
        st = EX.stat([N[k]["hp_trade"] - (T[k]["hp_trade"] + Dd[k]["hp_trade"]) / 2 for k in ks])
        ok, rule = st["lo"] is not None and st["lo"] >= 0, "NET - (TEACH + DEFAULT) / 2 lower bound >= 0"
    else:
        st = EX.stat([N[k]["hp_trade"] - T[k]["hp_trade"] for k in ks])
        ok, rule = st["lo"] is not None and st["lo"] >= -0.005, "NET - TEACH lower bound >= -0.005"
    nmin = EX.MIN_N["value"]
    return {"id": "S0-1:value_after:NET", "kind": "skill", "status": EX._judge(ok, st["n"], nmin), "n": st["n"],
            "min": nmin, "gate": gate, "rule": rule, "hp_trade": st}


def _gap(t_dir):
    """Taught - held (METHOD step 6): the share of cause scenes where NET's line is the teacher's, taught against the
    held sets pooled; the gap within two errors of the difference (else memorised: widen the generator)."""
    import math
    from net.strategist import exam_s1 as EX
    from net.strategist import s1_scenes as SS

    def share(names):
        k = n = 0
        for nm in names:
            N, T = EX._load(t_dir, nm, "NET"), EX._load(t_dir, nm, "TEACH")
            for key, r in EX.spec_keys(t_dir, nm).items():
                if r["cause"] and N and T and key in N and key in T:
                    n += 1
                    k += N[key]["line"] == T[key]["line"]
        return k, n
    kt, nt = share(["taught"])
    kh, nh = share([f"held:{a}" for a in SS.HELD])
    if not nt or not nh:
        return {"id": "S0-1:gap:NET", "kind": "skill", "status": "NOT JUDGED", "n": min(nt, nh), "min": 64}
    pt, ph = kt / nt, kh / nh
    e = 2 * math.sqrt(max(pt * (1 - pt), 1e-4) / nt + max(ph * (1 - ph), 1e-4) / nh)
    return {"id": "S0-1:gap:NET", "kind": "skill", "status": EX._judge(pt - ph <= e, min(nt, nh), 64),
            "n": min(nt, nh), "min": 64, "taught": round(pt, 4), "held": round(ph, 4), "two_errors": round(e, 4),
            "rule": "taught - held share of the teacher's line <= two errors"}


def _control(c_dir):
    """The fresh-KEEP control (a fresh StratNet1 = DEFAULT played as NET in c_dir): its execution MUST fail."""
    from pathlib import Path
    from net.strategist import exam_s1 as EX
    if not Path(c_dir).exists():
        return {"id": "S0-1:control:NET", "kind": "safety", "status": "NOT JUDGED", "note": f"no {c_dir}"}
    st = EX.table(c_dir, "NET")["exec"]["status"]
    return {"id": "S0-1:control:NET", "kind": "safety", "status": "pass" if st == "fail" else
            ("NOT JUDGED" if st == "NOT JUDGED" else "fail"), "control_exec": st,
            "rule": "a fresh network (always KEEP) fails execution - else the piece checks nothing"}


def _sha(t_dir):
    """The candidate's weights: every NET file's key 'net' = the sha of D/candidate.json's checkpoint."""
    import json
    from pathlib import Path
    from net.drill import store as ST
    D = Path(t_dir)
    f = D / "candidate.json"
    if not f.exists():
        return {"id": "S0-1:weights", "kind": "safety", "status": "NOT JUDGED", "note": "no candidate.json"}
    want = ST.sha256(json.loads(f.read_text(encoding="utf-8"))["ckpt"])[:16]
    seen = {g.name: json.loads(g.read_text(encoding="utf-8"))["key"].get("net") for g in D.glob("*__NET.json")}
    bad = [k for k, v in seen.items() if v != want]
    return {"id": "S0-1:weights", "kind": "safety", "status": "pass" if seen and not bad else "NOT JUDGED",
            "want": want, "bad": bad, "files": len(seen)}


def verdict(D):
    """The exam after (METHOD step 6), python -m net.drill.verdict after --card s0_1 --dir D: D holds exam_s1's sets
    with the NET arm (run --net, the candidate in D/candidate.json {ckpt}), D/value_gate.json (copied from the exam
    before: frozen), D/control (the same sets with a fresh network as NET) and the label probe D/../probe/probe.json.
    Pieces: the table's lines on NET, the value piece against the frozen gate, the taught - held gap, the fresh-KEEP
    control, the weights' sha. No earlier strategist drill: no retention piece (the card has no parent)."""
    import json
    from pathlib import Path
    from net.drill import verdict as VD
    from net.strategist import exam_s1 as EX
    D = Path(D)
    gf = D / GATE_FILE
    gate = json.loads(gf.read_text(encoding="utf-8"))["gate"] if gf.exists() else None
    t = EX.table(D, "NET")
    pcs = judge(t, "NET") + [probe_piece(str(D.parent / "probe" / "probe.json"))]
    pcs.append(_net_vs(D, gate) if gate else {"id": "S0-1:value_after:NET", "kind": "skill", "status": "NOT JUDGED",
                                               "note": f"no frozen {GATE_FILE}"})
    pcs += [_gap(D), _control(D / "control"), _sha(D)]
    return VD.m2(pcs, card=card())


def card():
    from net.drill.cards import Card, Scenes, SetSpec
    from net.strategist import s1_net as SN
    from net.strategist import s1_scenes as SS
    ax = {k: v["held"] for k, v in SS.AXES.items()}
    sets = {"train": SetSpec(SETS["train"], 512, 128), "dev": SetSpec(SETS["dev"], 32, 64),
            "V": SetSpec(SETS["V"], 128, 128), "probe_train": SetSpec(SETS["probe_train"], 512, 128),
            "probe_held": SetSpec(SETS["probe_held"], 256, 128), "taught": SetSpec(SETS["taught"], 256, 256), "flat": SetSpec(SETS["taught"], 256, 256),
            "G": SetSpec(SETS["G"], 512, 256),
            **{f"held:{h}": SetSpec(SETS["held"] + 10_000 * j, 64, 256, held=h) for j, h in enumerate(SS.HELD)}}
    return Card(id=ID, words=WORDS, cause="we hold, the enemy comes, and our zone holds a crest facing him", level=LEVEL,
                parent=PARENT, stage=STAGE, teacher=teacher, disturb=None, arms=ARMS, below=BELOW, opponent=OPPONENT,
                scenes=Scenes(make=SS.scene, axes=ax, twins=tuple(SS.TWINS), family=SS.family,
                              binds_spec=SS.binds),
                sets=sets, mass=1.0, classes=_classes, band_share_max=None, masks_changed={}, marks=_marks,
                marks_lines={k: (op, line) for k, (_, op, line) in MARKS.items()},
                pieces_before=tuple(pieces("TEACH")), pieces_after=tuple(pieces("NET")), value_gate=(4.0, 0.02),
                head=lambda path=None: SN.StratNet1(), watch=None, probe_ablate=PROBE_ABLATE, lane=None, files=FILES,
                level_args={"lr": 1e-3, "minibatch": 256, "max_grad": 1.0, "tseed": 11, "chunk": 128,
                            "rounds_registered": 4},
                stages={"n": 512, "general_share": 0.25, "seed0": SETS["train"], "gseed0": SETS["train_general"],
                        "stride": ROUND_STRIDE},
                masks={"strat": "strategist"}, census={"strat": {"KEEP": 50, "CREST": 50, "POINTER": 50}}, retention=None, m2=_m2,
                exec_denominator="taught cause: first-melee units")
