"""Card T0-1 'атаковать' (attack): the tactic's target stage (net/tactic/pin.py TEACH) - the first tactic card, no
parent (tmp/engine/design_merged.md section 3). Its stop is relative to the accepted legacy network REFERENCE (brief_A3
R1: marks(tv) on path T - and the N-only ones on path N - not worse than the reference's V read by more than two
errors, path N not worse than T; every cell n >= 50); imitate.m3's absolute lines are printed beside. Its exam after
is the legacy `learn after` (imitate.after: S, H, G - its after.json), read here as M2 pieces.

    card()             the card's data for the run (cards/__init__.py): level tactic, drill 'attack', TEACH
    absolute(tv)       the stop's lines: imitate.m3 on V (followed N pooled >= 0.90, cells, followed T, false a tick /
                       a kind, no-cause false attacks, uncaused) - (ok, fails, judged); not judged when a pooled share
                       it reads has n < MARK_MIN_N
    pieces_m2(after)   the exam after's pieces from after.json: M3 per set (skill), M4 on S (skill, n >= 50), M5 per
                       held axis (skill), M7 per set (safety: orders a unit-minute, N - TEACH <= 0.03); a missing set
                       or axis: NOT JUDGED
    verdict(D)         M2 on D/after.json (no parent: no NEG, said so); exit code 0 only on PASSED
"""
import json
from pathlib import Path

N_MIN_M4 = 50
_N_MIN_WHY = "M4's YES labels on S below this: NOT JUDGED (brief_A2 Part 3's n >= 50 for every share a gate reads)"


def _cell(x):
    return x if isinstance(x, dict) else None


REFERENCE = {"path": "weights/tactic/attack1/final.pt",
             "sha256": "14b64dbf6fbf87ba8ca9d0d6e5260ace1d38aa3bf54c6b593c1be2d6369ec2a9",
             "why": "a retrain must reproduce the accepted network (net/drills.json T0-1); M3's absolute marks were the "
                    "project's wish, failed by the accepted network itself (followed 0.71-0.85) - brief_A3 R1, the line "
                    "written before the run in tmp/astra/KEY.md engine_code2"}


def marks(tv):
    """The stop's marks on V: followed pooled and false a tick on both paths; the N-only no-cause false attacks and
    uncaused orders (each judged on path N against the reference's)."""
    g = lambda *k: _dig(tv, *k)                                                       # noqa: E731
    return {"followed pooled": {"T": g("followed_T", "pooled"), "N": g("followed_N", "pooled")},
            "false a tick": {"T": g("false_T", "tick"), "N": g("false_N", "tick")},
            "no-cause scenes with a false attack (N)": {"N": g("nocause_false")},
            "uncaused orders (N)": {"N": g("uncaused")}}


def _dig(x, *k):
    for a in k:
        x = (x or {}).get(a)
    return _cell(x)


def _lines():
    from net.tactic import imitate as IM
    M = IM.L()["marks"]
    return {"followed pooled": (">=", float(M["followed_t"])), "false a tick": ("<=", float(M["false_tick"])),
            "no-cause scenes with a false attack (N)": ("<=", float(M["nocause_false"])),
            "uncaused orders (N)": ("<=", float(M["uncaused"]))}


def absolute(tv):
    from net.drill.cards import MARK_MIN_N as MN
    from net.tactic import imitate as IM
    # (every cell M3 reads at the minimum the card advertises: cells and kinds at MARK_MIN_N, not the config's 30;
    # no-cause / uncaused judged only from MARK_MIN_N observations - brief_A3 R1)
    ok, fails = IM.m3(tv, {**IM.L()["marks"], "cell_min_n": MN})
    judged = True
    for nm in ("nocause_false", "uncaused"):
        c = _cell(tv.get(nm))
        if not c or int(c.get("n") or 0) < MN:
            judged = False
            fails = fails + [f"M3 {nm}: n {(c or {}).get('n')} < {MN} - not judged"]
    reads = {"followed_N pooled": (tv.get("followed_N") or {}).get("pooled"),
             "false_N tick": (tv.get("false_N") or {}).get("tick")}
    if "followed_T" in tv:
        reads["followed_T pooled"] = tv["followed_T"].get("pooled")
    for nm, c in reads.items():
        if not c or c.get("share") is None or int(c.get("n") or 0) < MN:
            judged = False
            fails = fails + [f"M3 {nm}: n {(c or {}).get('n')} < {MN} - not judged"]
    return ok and judged, [f"M3: {x}" for x in fails], judged


def _p(pid, kind, status, **x):
    return {"id": pid, "kind": kind, "status": status, **x}


def pieces_m2(after):
    out = []
    for s in ("S", "H", "G"):
        t = after.get(s)
        if not t or "M3" not in t:
            out.append(_p(f"M3:{s}", "skill", "NOT JUDGED", note="set missing"))
            out.append(_p(f"M7:{s}", "safety", "NOT JUDGED", note="set missing"))
            continue
        fol = (t.get("followed_N") or {}).get("pooled") or {}
        out.append(_p(f"M3:{s}", "skill", "NOT JUDGED" if int(fol.get("n") or 0) < N_MIN_M4 else
                      ("pass" if t["M3"]["passed"] else "fail"), fails=t["M3"]["fails"]))
        m7 = t.get("M7") or {}
        d = m7.get("N - TEACH")
        out.append(_p(f"M7:{s}", "safety", "NOT JUDGED" if not d else ("pass" if m7.get("passed") else "fail"),
                      read=d, N=m7.get("N"), TEACH=m7.get("TEACH")))
    m4 = (after.get("S") or {}).get("M4")
    out.append(_p("M4:S", "skill", "NOT JUDGED" if not m4 or int(m4.get("n") or 0) < N_MIN_M4 else
                  ("pass" if after["S"].get("M4_passed") else "fail"), read=m4))
    m5 = after.get("M5") or {}
    if not m5:
        out.append(_p("M5", "skill", "NOT JUDGED", note="no held axis read"))
    for ax, r in sorted(m5.items()):
        f = r.get("followed") or {}
        out.append(_p(f"M5:{ax}", "skill", "NOT JUDGED" if int(f.get("n") or 0) < N_MIN_M4 or "followed_ok" not in r
                      else ("pass" if r["passed"] else "fail"), read=r))
    return out


def verdict(D):
    from net.drill import verdict as VD
    after = json.loads((Path(D) / "after.json").read_text(encoding="utf-8"))
    return VD.m2(pieces_m2(after), None, card=card())


def card():
    from net.drill.cards import Card
    return Card(
        id="T0-1", words="attack", cause="a target the teacher attacks", level="tactic", parent=None, stage="target",
        teacher="TEACH", disturb=None, arms={"TEACH": "pin's teacher", "C": "the stack without the tactic",
                                             "N": "the network"},
        below={"unit": "the unit scripts"}, opponent=("the scene enemy", "the general battles' AI stand-in"),
        scenes=None, sets={}, mass=1.0, classes=None, band_share_max=None, masks_changed={},
        files=("net/drill/levels/tactic.py",),
        level_args={"drill": "attack", "tags": ["T0-1"]},
        stages={"teacher": "TEACH (disturbed)", "scenes": "learn.round_specs", "general": True,
                "net_share": "train.learn.net_share from round 2"},
        masks={"pointer": "tactic_pointer"}, census={"pointer": {"attack": 50, "keep": 50}},
        dont_care_max={"pointer": {"attack": 0.35, "keep": 0.20}}, absolute=absolute,
        marks=marks, marks_lines=_lines(), reference=REFERENCE)


if __name__ == "__main__":
    import sys
    v = verdict(sys.argv[1])
    from net.drill import verdict as VD
    print("\n".join(VD.m2_lines(v)))
    sys.exit(0 if v["verdict"] == "PASSED" else 1)
