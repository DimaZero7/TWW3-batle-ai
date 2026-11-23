"""Card O0-1 'advance on the holder' (tmp/engine/design_merged.md 3; the operative's line stage): the teacher
advance.teach on the level's facts, candidates KEEP and LS (the stand-off line; L0 closed, LM out), arms DEFAULT / DUMB /
TEACH over the tactic's TEACH, the enemy sally.Sally / WalkIn, scenes scenes_o1 with twins come / role, head OpNet1.

    SETS        seeds: the exam's named sets (taught 71000, held 72000 + 10000 j, G 73000 / G_hold 73500 - scenes_o1) and
                the fresh ranges of the engine (train 40e6, dev 50e6, V 60e6, probe_train 70e6, probe_held 80e6; a range's
                substitutes + k x 1e6, k <= 5, stay below the next range)
    scenes(name, n)        n cause scenes of the range + n twins (even i come, odd i role) - scenes_o1's taught rule
    disturb(rnd, specs)    {battle index: delay}: in DISTURB_SHARE of the cause battles the teacher's first LS is held
                one or two ticks (the states off the teacher's path; its label stays LS)
    Disturbed(delays)      the teacher-driven player with those delays
    teacher(f, c)          advance.teach
"""
import random

from net.operative import advance as AD
from net.operative import scenes_o1 as SO

ID = "O0-1"
WORDS = "advance on the holder"
LEVEL = "operative"
STAGE = "line"
PARENT = None
ARMS = {"DEFAULT": "the stack without the operative (KEEP)", "DUMB": "Hold(LM) once", "TEACH": "advance.teach"}
BELOW = {"tactic": "TEACH"}
OPPONENT = ("sally.Sally", "sally.WalkIn", "sally.Enemy")
SETS = {"taught": 71000, "held": 72000, "G": 73000, "G_hold": 73500,
        "train": 40_000_000, "dev": 50_000_000, "V": 60_000_000, "probe_train": 70_000_000, "probe_held": 80_000_000,
        "probe_held2": 80_500_000}
_HELD2_WHY = ("fresh probe families (Astra's second o1 review 2: the earlier held families were used for recipe "
              "selection) inside the registered probe_held range 80,000,000-81,000,000")
_SETS_WHY = ("the named exam sets are scenes_o1's (operative.json o1.scenes); the fresh ranges 10e6 apart: a seed's "
             "substitutes (+ k x 1e6, k <= 5, scenes_o1.SUBSTITUTE) never reach the next range, clear of the tactic's "
             "train.log line 2 and the exam seeds")
DISTURB_SHARE = 0.25
DELAYS = (1, 2)
_DISTURB_WHY = "design_merged 3 (O0-1): the first L60 delayed one or two ticks in 25 % of the teacher-driven cause battles"
FILES = ("net/operative/advance.py", "net/operative/lines.py", "net/operative/sally.py", "net/operative/scenes_o1.py",
         "net/operative/heads.py", "net/operative/gframe.py", "net/operative/scenes_fair.py",
         "net/operative/attack.py", "net/drill/levels/operative.py", "net/drill/cards/o0_1.py")


def teacher(f, c=None):
    return AD.teach(f, c)


def scenes(name, n):
    """(module doc) through scenes_fair (Astra's o1 review 2: the layouts' tells besides the cause removed) - the
    engine's sets only; the exam's named sets stay scenes_o1's (exam2's arms)."""
    return scenes_at(SETS[name], n)


def disturb(rnd, specs):
    out = {}
    for b, s in enumerate(specs):
        if s["o1"]["cause"] and rnd.random() < DISTURB_SHARE:
            out[b] = rnd.choice(DELAYS)
    return out


class Disturbed:
    """The teacher with its first LS held delays[b] ticks in battle b of the chunk (wants_view: the side passes b)."""
    name = "TEACH~"
    wants_view = True

    def __init__(self, delays):
        self.delays, self.held = dict(delays), {}

    def __call__(self, f, order, c=None, obs=None, b=None):
        lab = AD.teach(f, c)[0]
        if lab is not None and self.held.get(b, 0) < self.delays.get(b, 0):
            self.held[b] = self.held.get(b, 0) + 1
            return None
        if lab is not None:
            self.held[b] = 10 ** 6                       # (the first LS went out: no later one is held)
        return lab


def make_disturbed(specs, seed):
    """make_player for exam_o1.play: the chunk's Disturbed from a Python stream of its own (not torch's)."""
    rnd = random.Random(seed)
    d = disturb(rnd, specs)

    def mk(ci, ii):
        return Disturbed({k: d[i] for k, i in enumerate(ii) if i in d})
    return mk, d


# ------------------------------------------------------------------ the card as data (cards/__init__.py)
def _line_of(tables, arm):
    return tables.get("lines" if arm == "TEACH" else f"lines_{arm}") or {}


def _share(x):
    return (None, None, 0) if not x else (x["p"], x["two_errors"], x["n"])


def _pair(x):
    return (None, None, 0) if not x else (x["diff"], x["two_errors"], x["n"])


def _reader(line, arm):
    """read(tables, stratum, denominator) -> (value, two errors, n) of exam_o1.lines' line `line` for the arm."""
    def read(tables, stratum, den):
        L = _line_of(tables, arm).get("6" if line == "6t" else line) or {}
        if line == "1":                                    # (both execution populations: all, and the comer battles out)
            return _share(L.get({"arm": arm, "arm-comer": f"{arm}_-comer"}.get(den, "DEFAULT")))
        if line in ("St", "Sh", "Sn"):                     # (a stratum's n BEFORE pooling: the set's own group / pairs)
            if line == "Sn":
                nm, lab = stratum, "no cause [-comer]"
            else:
                nm, lab = ("taught", "cause") if line == "St" else ("held", f"held {stratum}, cause")
            t = tables.get(nm) or {}
            if den == "contact":
                c = (((t.get("groups") or {}).get(lab) or {}).get(arm) or {}).get("contact")
                return (None, None, 0) if not c else (c["p"], c["two_errors"], c["n"])
            other = "DEFAULT" if (line == "Sn" or arm == "TEACH") else "TEACH"
            x = (((t.get("pairs") or {}).get(f"{arm} - {other}") or {}).get(lab) or {}).get("j600")
            return (None, None, 0) if not x else (x["diff"], x["two_errors"], x["n"])
        if line == "NvT":                                  # (NET - TEACH on the cause scenes, pooled after the strata)
            from net.operative import exam_o1 as EXO
            return _pair(EXO._pool_pair([EXO._pp(tables, nm, "cause", f"{arm} - TEACH") for nm in ("taught", "held")]))
        if line == "2":
            return _share(L.get("changed")) if den == "changed" else _pair((L.get("trade_-comer") or {}).get("j600"))
        if line == "3":
            return _pair((L.get("all") or {}).get("j600"))
        if line == "4":
            v = (L.get("with_lord") or {}).get(stratum)
            return (v, 0.0, (L.get("n") or {}).get(stratum) or 0)
        if line == "5":
            x = L.get("pooled") if stratum == "pooled" else (L.get("by_axis") or {}).get(stratum)
            return (None, None, 0) if not x else (x["gap"], x["two_errors"], x["held"]["n"])
        if line == "6":
            return (L.get(f"{arm}_stall"), 0.0, L.get("n") or 0)
        if line == "6t":
            return _pair(L.get(f"{arm} - DEFAULT j600 (G)"))
        if line == "Gq":                                   # (read from the set's own table: exam_o1 has no line for it)
            g = tables.get(QUIET_SET) or {}
            if den == "changed":
                return _share(((g.get("groups") or {}).get("all") or {}).get(arm, {}).get("changed"))
            return _pair((((g.get("pairs") or {}).get(f"{arm} - DEFAULT") or {}).get("all") or {}).get("j600"))
        if line == "binds":
            r = (L.get("by_axis") or {}).get(stratum) or {}
            return (r.get("binds", 0) - r.get("scenes", 0), 0.0, r.get("scenes", 0))
        raise KeyError(line)
    return read


def _lines_of(line):
    """line(value, two errors) -> pass | fail | undecided, the exam's rules as coded (exam_o1.lines)."""
    def f1a(v, e):
        return "undecided" if v is None else ("pass" if v >= 0.95 else "fail")

    def f1d(v, e):
        return "undecided" if v is None else ("pass" if v <= 0.05 else "fail")
    rules = {"1": f1a, "1d": f1d,
             "2": lambda v, e: "undecided" if v is None else ("pass" if v <= 0.03 else "fail"),
             "2t": lambda v, e: "undecided" if v is None else ("pass" if v - e >= -0.005 else "fail"),
             "S": lambda v, e: "undecided" if v is None else "pass",
             "NvT": lambda v, e: "undecided" if v is None else ("pass" if v - e >= -0.005 else "fail"),
             "3": lambda v, e: "pass",
             "4": lambda v, e: "undecided" if v is None else ("pass" if v <= 1.2 else "fail"),
             "5": lambda v, e: "undecided" if v is None else ("pass" if v >= -e else "fail"),
             "6": lambda v, e: "undecided" if v is None else ("pass" if v <= 0.05 else "fail"),
             "6t": lambda v, e: "pass",
             "Gq": lambda v, e: "undecided" if v is None else ("pass" if v <= 0.03 else "fail"),
             "Gqt": lambda v, e: "undecided" if v is None else ("pass" if v - e >= -0.005 else "fail"),
             "binds": lambda v, e: "pass" if v == 0 else "fail"}
    return rules[line]


_RULES = {"1": ("1", "1", "1d"), "2": ("2", "2t"), "St": ("S",), "Sh": ("S",), "Sn": ("S",), "NvT": ("NvT",), "3": ("3",), "4": ("4",), "5": ("5",), "6": ("6",), "6t": ("6t",),
          "Gq": ("Gq", "Gqt"), "binds": ("binds",)}
QUIET_SET = "G_def"
NO_HARM_WHY = (
    "the orchestrator 22.11: no harm against DEFAULT is NOT judged where DEFAULT stalls - no contact is no battle, and in "
    "the game the attacker who stands loses by the timer (game/probe/evidence/holder_game.json); on general battles with "
    "the mission 'attack' (G) the gate is line 6 (stalls <= 0.05) and TEACH - DEFAULT j600 is printed beside it as a "
    "report (6t); no harm IS gated on general battles with the mission 'hold' (QUIET_SET): the arm changes the order in "
    "<= 3 % of them and its trade's lower bound (TEACH - DEFAULT j600 minus two errors) is >= -0.005. The exam's present "
    "G_hold is NOT that set (att 1 against a holder that never comes: the teacher changes the order in every battle by "
    "design) - it stays reported in line 6; QUIET_SET is not built yet (no attacking enemy kind in sally.Enemy): the "
    "piece is NOT JUDGED until its table exists")
LOWER_BOUND = ("2t", "NvT", "Gqt")
DOUBLE_MAX = 1024
_DOUBLE_WHY = "METHOD step 4: 256 a set, doubled to 1,024 while the line is not decided (an interval straddling it)"
_F6_WHY = ("Astra's o1 review 4 / the orchestrator F6: line 1 reads both execution populations (all, the comer battles "
           "out); every stratum's n checked BEFORE pooling (St: taught's 192 cause scenes, Sh: 32 a held axis); the "
           "no-cause trade gated by its lower bound >= -0.005 (not 'the interval holds zero'); NET against TEACH on "
           "the cause scenes not worse (lower bound >= -0.005: the teacher's gap is under the value gate)")
# (id, kind, strata, denominators, n_min): n_min the exam's own - FULL sets 384 (design 6: lines 1, 2, 4-6 accept
# 256 / 512 / 1,024 by blocks, so 256), line 2 PAIRS_MIN 100; line 5 per held axis judged on its 32 cause scenes
_PIECES = (("St", "safety", ("taught:cause",), ("contact", "pairs"), 192),
           ("Sh", "safety", None, ("contact", "pairs"), 32),
           ("Sn", "safety", ("taught", "held"), ("pairs",), 100),
           ("1", "skill", ("taught+held:cause",), ("arm", "arm-comer", "DEFAULT"), 256),
           ("2", "safety", ("taught+held:no cause[-comer]",), ("changed", "trade"), 100),
           ("NvT", "safety", ("taught+held:cause",), ("j600",), 256),
           ("3", "report", ("taught+held:cause",), ("j600",), 2),
           ("4", "safety", ("taught", "held", "G"), ("orders",), 256),
           ("5", "skill", None, ("contact gap",), 32),
           ("6", "skill", ("G:all",), ("stall",), 256),
           ("6t", "report", ("G:all",), ("j600",), 2),
           ("Gq", "safety", (QUIET_SET + ":all",), ("changed", "trade"), 64),
           ("binds", "safety", None, ("binds",), 1))


def pieces(arm):
    """The exam's lines 1-6 and binds (exam_o1.lines, as coded) as Pieces of the arm: TEACH before the training,
    NET after; denominators in the order of _RULES' rules (line 1: the arm's contact AND DEFAULT's)."""
    from net.drill.cards import Piece
    from net.operative import scenes_o1 as SOx
    H = tuple(SOx.HELD_AXES)
    out = []
    for pid, kind, strata, dens, nmin in _PIECES:
        if pid == "NvT" and arm == "TEACH":
            continue                                       # (the network against the teacher: the exam after only)
        strata = strata or ((("pooled",) + H) if pid == "5" else H)
        out.append(Piece(id=f"O0-1:{pid}:{arm}", kind=kind, legacy=True, arms=(arm, "DEFAULT", "DUMB"), strata=strata,
                         denominators=dens, n_min=nmin, read=_reader(pid, arm), line=_lines_of(_RULES[pid][0])))
    return out


def judge(tables, arm="TEACH"):
    """[{id, kind, status, cells}] - every piece of the arm on the exam's tables (exam_o1.json): a cell (stratum x
    denominator) with n < n_min is NOT JUDGED; a piece passes when every cell passes (report pieces: reported)."""
    res = []
    for p in pieces(arm):
        pid = p.id.split(":")[1]
        cells, st = {}, "pass"
        for s in p.strata:
            for k, d in enumerate(p.denominators):
                v, e, n = p.read(tables, s, d)
                r = _lines_of(_RULES[pid][min(k, len(_RULES[pid]) - 1)])(v, e)
                if n < p.n_min:
                    r = "NOT JUDGED"
                elif (r == "fail" and _RULES[pid][min(k, len(_RULES[pid]) - 1)] in LOWER_BOUND and v is not None
                      and v + e >= -0.005 and n < DOUBLE_MAX):
                    r = "double"                            # (the interval straddles its line: double the set)
                cells[f"{s} | {d}"] = {"value": v, "two_errors": e, "n": n, "line": r}
                if p.kind == "report":
                    continue
                if r == "NOT JUDGED":
                    st = "NOT JUDGED"
                elif r == "double" and st == "pass":
                    st = "double"
                elif r not in ("pass", "double") and st != "NOT JUDGED":
                    st = "fail"
        x = {"id": p.id, "kind": p.kind, "status": "reported" if p.kind == "report" else st, "cells": cells}
        if st == "double":
            x["double"] = [s_.split(":")[0] for s_ in p.strata]
        res.append(x)
    return res


CARRIERS_WHY = (
    "the cause 'the enemy holds' has two carriers in the input: its motion (rows' velocity, moving and running flags; "
    "gframe's velocities and closing) and its 'braced' effect - a holder stands braced, in the game too (the bracing "
    "of standing infantry, game/manual.md 'Натиск и упор'); on the probe's held pairs braced is 1.0 on every cause "
    "decision and 0.0 on every come one (tmp/engine/b/leak_cols). The probe's ablated arm erases both (probe_ablate).")
HEAD_WHY = (
    "the network: OpNet1 with the per-candidate-type state readout on (pick, heads.PICK_WHY) at lr 1e-3, minibatch 256 "
    "- the pair the three-seed probe passed with motion and braced erased (tmp/engine/b/probe/probe.json = "
    "probe_pick_b; the plain head's files beside it: probe_lr3e4_b, probe_lr3e4, probe_mb64, probe_fit3)")
FAIR_WHY = (
    "the engine's scenes (training, V, the probe) go through net/operative/scenes_fair.py: the come twin's walkers no "
    "longer lag their pose (14 m median), the holders' ranks and lord offset jittered alike in a family - the tells "
    "Astra's o1 review 2 found besides the cause; the exam's named sets stay scenes_o1's (exam2's arms)")
PROBE = "probe.json"     # (net/drill/probe.py's result: {verdict: {seeds: {seed: {arm_last | arm_best_loss}}, n_ok, pass}})


def probe_piece(path):
    """The label probe on the real input as a GATE (design 5, Astra's engine-code reviews 7 and o1-2 4): pass only when
    its file passed AND carries this card's gate key (levels/operative.gate_key: these scenes, this teacher, this
    head's config); no file, n under the probe's minimum, or another key: NOT JUDGED (the verdict BLOCKED)."""
    import json
    from pathlib import Path
    from net.drill.levels import operative as OPL
    f = Path(path) if path else None
    if f is None or not f.exists():
        return {"id": "O0-1:probe", "kind": "safety", "status": "NOT JUDGED", "file": str(f), "note": "no probe result"}
    d = json.loads(f.read_text(encoding="utf-8"))
    v, want = d["verdict"], OPL.gate_key(card())
    if d.get("gate_key") != want:
        return {"id": "O0-1:probe", "kind": "safety", "status": "NOT JUDGED", "file": str(f),
                "note": f"gate_key {d.get('gate_key')} != this card's {want}"}
    st = "NOT JUDGED" if not v["n_ok"] else ("pass" if v["pass"] else "fail")
    return {"id": "O0-1:probe", "kind": "safety", "status": st, "file": str(f), "gate_key": want, "verdict": v}


def pieces_before(tables, probe=None):
    return judge(tables, "TEACH") + [probe_piece(probe)]


def pieces_after(tables, probe=None):
    return judge(tables, "NET") + [probe_piece(probe)]


def _classes(batch):
    t = batch.teacher
    return {"KEEP": t == 0, "LS": t == 1}


MARKS = {"cause first advance followed": ("cause_first_followed", ">=", 0.90),
         "come twin first decision false": ("come_first_false", "<=", 0.03),
         "role twin false advance": ("role_false", "<=", 0.03)}
_MARKS_WHY = ("the decisions that matter, per battle on V (Astra's o1 review 6, the orchestrator 22.11 F3): a cause "
              "scene's first advance taken at the teacher's tick or within one tick >= 0.90, a come twin's first "
              "decision advanced <= 0.03, a role twin advanced at all <= 0.03, each n >= 50 (V: 128 cause, 64 come, "
              "64 role); the per-tick followed / false printed beside (levels/operative.brief) - missing all 64 come "
              "twins costs only 1.66 % of ticks")


def _marks(v):
    """{mark: {path: {share, two_errors, n}}} from levels/operative.v_read's {T, N: marks}."""
    out = {}
    for name, (k, _, _) in MARKS.items():
        out[name] = {}
        for path in ("T", "N"):
            c = (v.get(path) or {}).get(k)
            out[name][path] = None if c is None else {"share": c["p"], "two_errors": c["two_errors"], "n": c["n"]}
    return out


def _m2(tables, probe=None):
    return pieces_after(tables, probe)


PROBE_FILE = "tmp/engine/b/probe4/" + PROBE     # (probe3 re-collected under the 23.11 gate key: the adapter and ROUND_N changed; same scenes)      # (the accepted three-seed result of net/drill/probe.py, copied there)


TEACHER_FILE = "tmp/o1/exam3/verdict_before.json"   # (the teacher's exam on the fair starts: verdict_before writes it)
FRESH_SUFFIX = "_fresh"        # (the fresh-network control's exam folder: <D>_fresh - exam_o1 run --players NET --net fresh)
FRESH_MIN = 256


def neg_piece(D, fresh=None):
    """The negative control of a card without a parent (Astra's o1 reviews 4-5): the UNTRAINED network (the card's
    OpNet1, fresh) played on the exam's OWN battles - exam_o1 run --players NET --net fresh into <D>_fresh with the
    same starts - must fail line 1: its contact on the cause scenes <= 0.05 on >= FRESH_MIN battles, its arm files
    keyed net_sha 'fresh' and paired with D's (the same starts, seeds, chunk, simulator, tactic, visibility)."""
    import json
    from pathlib import Path
    from net.operative import exam_o1 as EX
    F = Path(fresh) if fresh else Path(str(D) + FRESH_SUFFIX)
    files = sorted(F.glob("*_NET.json")) if F.exists() else []
    if not files:
        return {"id": "O0-1:neg", "kind": "safety", "status": "NOT JUDGED", "note": f"no fresh-network arm in {F}"}
    shas = {f.name: (json.loads(f.read_text(encoding="utf-8")).get("key") or {}).get("net_sha") for f in files}
    for f in files:                                 # (the fresh arm played the same battles as D's arms)
        nm = f.name[:-len("_NET.json")]
        ref = Path(D) / f"{nm}_TEACH.json"
        if ref.exists():
            a = json.loads(ref.read_text(encoding="utf-8"))
            b = json.loads(f.read_text(encoding="utf-8"))
            EX._same_battles(nm, {"TEACH": a, "NET": b}, json.loads((Path(D) / f"{nm}_meta.json").read_text(
                encoding="utf-8")))
    t = EX.table(F, names=tuple(f.name[:-len("_NET.json")] for f in files))
    c = ((t.get("lines_NET") or {}).get("1") or {}).get("NET")
    ok = bool(c) and c["n"] >= FRESH_MIN and c["p"] <= 0.05 and all(v == "fresh" for v in shas.values())
    st = "NOT JUDGED" if not c or c["n"] < FRESH_MIN else ("pass" if ok else "fail")
    return {"id": "O0-1:neg", "kind": "safety", "status": st, "contact": c, "files": shas,
            "rule": "the fresh network on the exam's own battles: cause contact <= 0.05 on >= 256 (it fails line 1)"}


def weights_piece(D, run=None):
    """The NET arm's files bound to the candidate (Astra's o1 review 3-4): every <set>_NET.json in D carries the
    checkpoint's sha (exam_o1 run: key net_sha) and it equals the run's run.json sha256; no --run: NOT JUDGED (the
    weights unauthenticated - an archived teacher's counts would pass as NET)."""
    import json
    from pathlib import Path
    files = sorted(Path(D).glob("*_NET.json"))
    shas = {f.name: (json.loads(f.read_text(encoding="utf-8")).get("key") or {}).get("net_sha") for f in files}
    want = None
    if run:
        rj = json.loads((Path(run) / "run.json").read_text(encoding="utf-8"))
        want = rj.get("sha256") if rj.get("passed") else None
    if not files or want is None:
        st = "NOT JUDGED"
    else:
        st = "pass" if all(v == want for v in shas.values()) else "fail"
    return {"id": "O0-1:weights", "kind": "safety", "status": st, "files": shas, "run_sha256": want,
            "rule": "every NET arm file's net_sha = the candidate's final.pt sha (run.json, passed)"}


def value_gate(t):
    """The value branch from the teacher's exam (METHOD step 4): 'half gap' when TEACH - DEFAULT on the cause scenes
    (j600, lords in) is >= 4 standard errors AND >= 0.02, else 'not worse than the teacher'."""
    x = ((((t.get("lines") or {}).get("3") or {}).get("TEACH - DEFAULT, cause")) or {}).get("j600")
    if not x:
        return {"branch": None, "note": "no TEACH - DEFAULT cause pair"}
    se = x["two_errors"] / 2
    half = x["diff"] >= 4 * se and x["diff"] >= 0.02
    return {"branch": "half gap" if half else "not worse than the teacher", "gap": x,
            "rule": "METHOD step 4: half the gap asked only when the teacher's gap >= 4 errors and >= 0.02"}


EXAM_SETS = ("taught", "held", "G", "G_hold", "G_def")


def _tables(D):
    """exam_o1's tables rebuilt from D's raw arm files (never a detached summary): the pairing checks run."""
    from net.operative import exam_o1 as EX
    return EX.table(D, names=EXAM_SETS)


def verdict_before(D, probe=PROBE_FILE):
    """The teacher's exam (`python -m net.drill.verdict before --card o0_1 --dir D --probe P`): the tables rebuilt from
    D's raw arms, the TEACH pieces, the probe gate; the value branch frozen into D/value_gate.json; the result with
    this card's gate key written to D/verdict_before.json (TEACHER_FILE: the training's start reads it)."""
    import json
    from pathlib import Path
    from net.drill import verdict as VD
    from net.drill.levels import operative as OPL
    t = _tables(D)
    vg = value_gate(t)
    (Path(D) / "value_gate.json").write_text(json.dumps(vg, indent=1), encoding="utf-8")
    v = VD.m2(pieces_before(t, probe or PROBE_FILE), None, card=card())
    v["value_gate"] = vg
    v["gate_key"] = OPL.gate_key(card())
    (Path(D) / "verdict_before.json").write_text(json.dumps(v, indent=1, default=str), encoding="utf-8")
    return v


def _value_piece(t, vg):
    """The value branch frozen before: 'not worse than the teacher' is piece NvT; 'half gap' asks NET - DEFAULT on the
    cause scenes (lower bound) >= half the teacher's gap; no frozen branch: NOT JUDGED."""
    from net.operative import exam_o1 as EX
    b = vg.get("branch")
    if b is None:
        return {"id": "O0-1:value", "kind": "safety", "status": "NOT JUDGED", "note": "no value_gate.json"}
    if b != "half gap":
        return {"id": "O0-1:value", "kind": "safety", "status": "pass", "note": "branch 'not worse': piece NvT gates"}
    x = EX._pool_pair([EX._pp(t, nm, "cause", "NET - DEFAULT") for nm in ("taught", "held")])
    if not x or x["n"] < 256:
        return {"id": "O0-1:value", "kind": "safety", "status": "NOT JUDGED", "pair": x}
    ok = x["diff"] - x["two_errors"] >= vg["gap"]["diff"] / 2
    return {"id": "O0-1:value", "kind": "safety", "status": "pass" if ok else "fail", "pair": x,
            "half": vg["gap"]["diff"] / 2}


def verdict(D, probe=PROBE_FILE, run=None):
    """The exam after (`python -m net.drill.verdict after --card o0_1 --dir D --probe P --run R`): the tables rebuilt
    from D's raw arms (the teacher's exam's arms + NET on the same battles), the NET pieces with the value branch
    frozen before (D/value_gate.json), the probe gate, the fresh-network control on the exam's own battles (<D>_fresh)
    and the weights' binding (every NET arm file's sha = the run's)."""
    import json
    from pathlib import Path
    from net.drill import verdict as VD
    t = _tables(D)
    f = Path(D) / "value_gate.json"
    vg = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"branch": None}
    pcs = pieces_after(t, probe or PROBE_FILE) + [neg_piece(D), weights_piece(D, run), _value_piece(t, vg)]
    v = VD.m2(pcs, None, card=card())
    v["neg"] = {"note": "no parent: the negative control is the untrained network on the exam's battles (O0-1:neg)"}
    return v


def card():
    from net.drill.cards import Card, Scenes, SetSpec
    from net.operative import exam_o1 as EXO
    from net.operative import heads as HD
    from net.operative import scenes_o1 as SOx
    ax = {k: SOx.cs()["axes"][k].get("held") for k in SOx.HELD_AXES}
    sets = {"train": SetSpec(SETS["train"], 512, 256), "dev": SetSpec(SETS["dev"], 32, 64),
            "V": SetSpec(SETS["V"], 128, 128), "probe_train": SetSpec(SETS["probe_train"], 192, 64),
            "probe_held": SetSpec(SETS["probe_held"], 128, 64), "taught": SetSpec(SETS["taught"], 192, 384),
            "G": SetSpec(SETS["G"], 384, 384), "G_hold": SetSpec(SETS["G_hold"], 64, 384),
            **{f"held:{h}": SetSpec(SETS["held"] + 10000 * j, 32, 384, held=h) for j, h in enumerate(SOx.HELD_AXES)}}
    return Card(id=ID, words=WORDS, cause="our mission is attack and the enemy holds", level=LEVEL, parent=PARENT,
                stage=STAGE, teacher=teacher, disturb=disturb, arms=ARMS, below=BELOW, opponent=OPPONENT,
                scenes=Scenes(make=SOx.scene, axes=ax, twins=("come", "role"), binds_spec=SOx.binds),
                sets=sets, mass=1.0, classes=_classes, band_share_max=None, masks_changed={}, marks=_marks,
                marks_lines={k: (op, line) for k, (_, op, line) in MARKS.items()}, pieces_before=tuple(pieces("TEACH")),
                pieces_after=tuple(pieces("NET")), head=lambda path=None: HD.OpNet1(), watch=EXO.O1Watch,
                probe_ablate=("rows.v_right", "rows.v_forward", "rows.mv", "rows.f", "gframe.g_v_right",
                              "gframe.g_v_fwd", "gframe.g_close", "rows.braced_on", "rows.braced_left",
                              "order.zone_to_LS", "cand.to_C", "cand.to_zone", "enemy lord: rows.right / forward, "
                              "place, gframe.g_right / g_fwd / g_near"),
                lane="holder_game", files=FILES,
                level_args={"pick": True, "lr": 1e-3, "minibatch": 256, "max_grad": 1.0, "tseed": 11, "chunk": 256},
                stages={"n": ROUND_N, "net_share": 0.5, "seed0": SETS["train"], "stride": ROUND_STRIDE,
                        "general_share": GENERAL_SHARE},
                masks={"op": "operative"}, census={"op": {"KEEP": 50, "LS": 50, "LS:first": FIRST_MIN,
                                                           **{"KEEP:" + k: 50 for k in REASONS_MIN}}},
                retention=None, m2=_m2,
                exec_denominator=None)



GENERAL_SHARE = 0.25
ROUND_N = 384
_ROUND_N_WHY = ("Astra o1_train_review3 section 9 (the coordinator 23.11: METHOD's 6 GPU hours a drill): a round of 384 "
                "cause scenes + 384 twins + 128 G + 128 G_def = 1,024 battles, four chunks of 256, instead of 512 + 512 "
                "+ 171 + 170 = 1,365 (six chunks); 384 + 128 nominal first advances keep FIRST_MIN 400 reachable (the "
                "teacher's first LS in 0.987 of exam3's cause scenes, 1.0 of G); the census checks it, not changed after")
_GENERAL_WHY = ("METHOD step 5: a share of the stage's general battles in every collection, for the generalisation - "
                "the orchestrator 22.11 F4: 25 % of a round's battles, half mission attack against the sally holder "
                "(G's recipe), half mission hold against the stage's attackers (G_def's), fresh train seeds")
FIRST_MIN = 400
REASONS_MIN = ("role", "comer", "comes", "near")
_CENSUS_WHY = ("Astra's o1 review 6: minima per KEEP reason with a gradient (50 each, the engine's default class "
               "minimum) and of distinct first advances a round: 400 of the round's 384 cause scenes + 128 attack "
               "generals (the teacher's first LS in 0.99 of cause scenes, exam2 line 1; the disturbance delays, never "
               "drops it)")
ROUND_STRIDE = 10_000
_ROUND_WHY = "a training round's scenes start ROUND_STRIDE after the last one's (n <= 10,000 a round), inside train's range"


def scenes_at(s0, n):
    """scenes() from an explicit first seed (the training rounds: SETS['train'] + ROUND_STRIDE x round)."""
    assert n <= ROUND_STRIDE, n
    from net.operative import scenes_fair as SF
    return [SF.scene(s0 + i) for i in range(n)] + [SF.scene(s0 + i, twin="come" if i % 2 == 0 else "role")
                                                   for i in range(n)]
