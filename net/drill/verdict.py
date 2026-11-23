"""The blocking verdict (tmp/engine/design_merged.md section 6; brief_A2 Part 6): the legacy verdict (pieces written
before the training) and the M2 verdict reported apart. The card gives the views, the pieces and its declarations.

    legacy_lines(lines, pieces)   the legacy PIECE / VERDICT lines, exactly as imitate.after_side printed them
    m2(pieces, neg=None, card=None, neg_sha=None, parent_sha=None)   the M2 verdict, PASSED only when every skill and
                    safety piece passes. BLOCKED when: a piece is NOT JUDGED / NOT DECIDED (a missing arm, set, held
                    axis or binding, a stratum's n under its minimum before pooling, an interval still straddling its
                    line at 1,024) or asks for a doubled set ('double': the verdict names the set, it does not play);
                    a status or kind is unknown; the card has a parent and NEG is absent, NEG's weights sha is unknown
                    or not the parent's, or NEG does not fail every skill piece. A card without a parent has no NEG
                    and says so. FAILED: a piece fails.
    m2_lines(v)     the M2 lines (printed apart from the legacy ones)
    arm_sha(D, arm)   the weights sha256 an arm of exam folder D was played with (its log lines: "player", "ckpt_sha256")
    gate_before(D, card)   the teacher's exam before: tables, lines, and the value gate decided and stored in
                    D/value_gate.json (exam files of the card)
    after(D, card, ...)   a card's exam after from its raw files: tables (+300, +450 s, end), the legacy pieces, the M2
                    pieces and verdict; NEG's verdict when neg_dir is given
    main()          python -m net.drill.verdict {before | after} --card ID --dir D [...]; exit code 0 only on PASSED
"""
import hashlib
import json
from pathlib import Path

from net.drill import exam as EX

STATUSES = {"pass", "fail", "NOT JUDGED", "NOT DECIDED", "double", "reported", "open", "closed", "NOT IN FILES"}
KINDS = ("skill", "safety", "report")


def legacy_lines(ln, pieces):
    out = []
    for p in pieces:
        x = {k: v for k, v in ln[p].items() if k not in ("rule", "by_axis")}
        out.append(f"PIECE {p}: {ln[p]['status']} - {json.dumps(x)}")
    out.append(f"VERDICT: {ln['verdict']}" + (f" - {'; '.join(ln['reasons'])}" if ln["reasons"] else ""))
    return out


def m2(pieces, neg=None, card=None, neg_sha=None, parent_sha=None):
    blocked, failed, double = [], [], []
    for p in pieces:
        if p.get("kind") not in KINDS:
            blocked.append(f"{p.get('id')}: unknown kind {p.get('kind')!r}")
            continue
        if p.get("status") not in STATUSES:
            blocked.append(f"{p['id']}: unknown status {p.get('status')!r}")
            continue
        if p["kind"] == "report":
            continue
        if p["status"] in ("NOT JUDGED", "NOT DECIDED", "NOT IN FILES"):
            blocked.append(f"{p['id']} {p['status']}")
        elif p["status"] == "double":
            blocked.append(f"{p['id']} undecided: double {p.get('double')}")
            double += list(p.get("double") or [])
        elif p["status"] == "fail":
            failed.append(p["id"])
    has_parent = card is not None and getattr(card, "parent", None) is not None
    neg_res = None
    if not has_parent and card is not None:
        neg_res = {"note": "the card has no parent: no NEG"}
    elif neg is None:
        blocked.append("NEG absent (a card with a parent: the parent's weights as the arm)")
    else:
        sk = {p["id"]: p["status"] for p in neg if p.get("kind") == "skill"}
        bad = [k for k, s in sk.items() if s != "fail"]
        neg_res = {"skill": sk, "parent_fails_every_skill_piece": bool(sk) and not bad, "not_failed": bad,
                   "neg_sha": neg_sha, "parent_sha": parent_sha}
        if not sk:
            blocked.append("NEG: no skill piece read")
        if bad:
            blocked.append("NEG: the parent " + ", ".join(f"{k} {sk[k]}" for k in bad))
        if not neg_sha or not parent_sha:
            blocked.append(f"NEG: weights sha unknown (NEG {neg_sha}, parent {parent_sha})")
        elif neg_sha != parent_sha:
            blocked.append(f"NEG: its weights {neg_sha[:16]} are not the parent's {parent_sha[:16]}")
    verdict = "BLOCKED" if blocked else ("FAILED" if failed else "PASSED")
    return {"verdict": verdict, "blocked": blocked, "failed": failed, "double": sorted(set(double)), "neg": neg_res,
            "pieces": pieces}


def m2_lines(v):
    out = [f"M2 PIECE {p['id']} ({p['kind']}): {p['status']} - "
           + json.dumps({k: x for k, x in p.items() if k not in ("id", "kind", "status")}) for p in v["pieces"]]
    if v["neg"] is not None:
        out.append(f"M2 NEG: {json.dumps(v['neg'])}")
    if v.get("double"):
        out.append(f"M2 DOUBLE: {v['double']} (the verdict asks for these sets doubled; it does not play)")
    out.append(f"M2 VERDICT: {v['verdict']}" + (f" - blocked: {v['blocked']}" if v["blocked"] else "")
               + (f" - failed: {v['failed']}" if v["failed"] else ""))
    return out


def arm_sha(D, arm="N", set_=None):
    """(module doc) None when no log names it; a halt when two logs disagree; set_ - only the lines of that set (or
    naming none: Astra engine_code3 finding 11 - a held-set line never authenticates G)."""
    D = Path(D)
    logs = [D.parent / f"{D.name}.log"] + sorted(D.glob("*.log"))
    seen = set()
    for f in logs:
        if not f.is_file():
            continue
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"ckpt_sha256"' not in ln:
                continue
            try:
                x = json.loads(ln)
            except ValueError:
                continue
            if x.get("player") == arm and x.get("ckpt_sha256") and (set_ is None or x.get("set") in (None, set_)):
                seen.add(x["ckpt_sha256"])
    if len(seen) > 1:
        raise SystemExit(f"{D}: arm {arm} played with several weights {sorted(seen)}")
    return next(iter(seen), None)


def gate_before(D, card, names=("taught", "held", "G")):
    out = EX.tables(D, list(names), card.BEFORE)
    out["lines"] = card.lines_before(out)
    gate = card.value_gate_of(out)
    EX._atomic_text(json.dumps(gate, indent=1), Path(D) / "value_gate.json")
    out["value_gate"] = gate
    return out


def weights_piece(D, names, cand_sha, parent_sha, arms=("N", "N1")):
    """(brief_A3 V4) the exam's arm files authenticated: every set's N file played with the candidate's weights
    (cand_sha: the run's final.pt / run.json sha256), every N1 file with the parent's; a missing or another sha ->
    NOT JUDGED (BLOCKED: the exam is not of these weights)."""
    D = Path(D)
    want = {"N": cand_sha, "N1": parent_sha}
    seen, bad = {}, []
    for nm in names:
        for arm in arms:
            f = D / f"{nm}_{arm}.json"
            if not f.exists():
                # (finding 11: every set x arm file present - a missing one is not an authenticated exam)
                bad.append(f"{nm}_{arm}: missing")
                continue
            sha = json.loads(f.read_text(encoding="utf-8")).get("ckpt_sha256") or arm_sha(D, arm, nm)
            seen[f"{nm}_{arm}"] = sha
            if not want[arm] or sha != want[arm]:
                bad.append(f"{nm}_{arm}: {str(sha)[:16]} (want {str(want[arm])[:16]})")
    st = "pass" if seen and not bad else "NOT JUDGED"
    return {"id": "weights", "kind": "safety", "status": st, "files": seen, "bad": bad,
            "rule": "N arms = the candidate's final.pt sha, N1 arms = the parent's accepted sha"}


def after(D, card, t01=None, neg_dir=None, names=("taught", "held", "G"), gate=None, probe=None, parent_sha=None,
          cand_sha=None):
    out = EX.tables(D, list(names), card.AFTER)
    ln = card.pieces_after(out, t01)
    m2v = getattr(card, "AFTER_M2", card.AFTER)
    out2 = EX.tables(D, list(names), m2v) if m2v is not card.AFTER else out
    kw = {"gate": gate, "probe": probe, "t01": t01}
    neg = neg_sha = None
    if neg_dir is not None:
        neg = card.m2_pieces(EX.tables(neg_dir, list(names), m2v), **kw)
        neg_legacy = card.pieces_after(EX.tables(neg_dir, list(names), card.AFTER), None)
        neg_sha = arm_sha(neg_dir, "N")
    pcs = card.m2_pieces(out2, **kw) + [weights_piece(D, names, cand_sha, parent_sha)]
    v = m2(pcs, neg, card=card.card(), neg_sha=neg_sha, parent_sha=parent_sha)
    res = {"tables": out, "legacy": ln, "m2": v}
    if neg_dir is not None:
        res["neg_legacy"] = {p: neg_legacy[p]["status"] for p in card.PIECES_AFTER}
    res["text"] = (EX.print_reads(out, pairs_prefix=("N", "TEACH2 - TEACH"),
                                  labels=("all", "cause", "no cause", "all [-comer]", "cause [-comer]",
                                          "no cause [-comer]", "comer"))
                   + legacy_lines(ln, card.PIECES_AFTER) + m2_lines(v))
    return res


def main(argv=None):
    import argparse
    import importlib
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("after", "before"))
    ap.add_argument("--card", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gate", default="", help="the value gate decided before (value_gate.json of the teacher's exam)")
    ap.add_argument("--probe", default="", help="the label probe's result file (a gate)")
    ap.add_argument("--t01-new", default="")
    ap.add_argument("--t01-ref", default="")
    ap.add_argument("--neg", default="")
    ap.add_argument("--parent-sha", default="", help="the parent card's accepted weights sha256 (its run.json)")
    ap.add_argument("--run", default="", help="the candidate's run folder: its run.json sha256 authenticates the N arms")
    a = ap.parse_args(argv)
    card = importlib.import_module(f"net.drill.cards.{a.card}")
    if a.what == "before" and hasattr(card, "verdict_before"):
        # (a card whose exam before is its own table - S0-1: net/strategist/exam_s1; NEEDS_A.md item 6)
        v = card.verdict_before(a.dir, a.probe or None)
        print("\n".join(m2_lines(v)))
        return 0 if v["verdict"] == "PASSED" else 1
    if a.what == "before":
        out = gate_before(a.dir, card)
        print("\n".join(EX.print_reads(out)))
        print(json.dumps({"lines": out["lines"], "value_gate": out["value_gate"]}, indent=1))
        ok = all(x.get("passed") for x in out["lines"].values() if isinstance(x, dict) and "passed" in x)
        return 0 if ok else 1
    if not hasattr(card, "AFTER"):
        # (a card whose exam after is the legacy after.json - T0-1: its own verdict(D); one entry for every card,
        # brief_A3 V1)
        if not hasattr(card, "verdict"):
            raise SystemExit(f"--card {a.card}: neither the exam tables (AFTER) nor verdict(D) - no exam after")
        import inspect
        ps = inspect.signature(card.verdict).parameters
        # (a card's own verdict gets --probe / --run when it takes them - O0-1: verdict(D, probe, run); B's NEEDS_A 5)
        v = card.verdict(a.dir, **{k: x for k, x in (("probe", a.probe or None), ("run", a.run or None))
                                   if k in ps and x is not None})
        print("\n".join(m2_lines(v)))
        return 0 if v["verdict"] == "PASSED" else 1
    t01 = card.t01_compare(a.t01_new, a.t01_ref) if a.t01_new and a.t01_ref else None
    gate = json.loads(Path(a.gate).read_text(encoding="utf-8")) if a.gate and Path(a.gate).is_file() else None
    cand = None
    if a.run:
        rj = json.loads((Path(a.run) / "run.json").read_text(encoding="utf-8"))
        if not rj.get("passed") or not rj.get("sha256"):
            raise SystemExit(f"--run {a.run}: run.json passed {rj.get('passed')} / sha {rj.get('sha256')} - no candidate")
        cand = rj["sha256"]
        # (finding 11: the parent the N1 arms are checked against is the candidate's own lineage - its checkpoint's
        # init_sha256 - never only the caller's --parent-sha)
        import torch
        fin = Path(rj.get("final") or Path(a.run) / "final.pt")
        if not fin.is_file() or hashlib.sha256(fin.read_bytes()).hexdigest() != cand:
            raise SystemExit(f"--run {a.run}: {fin} missing or its sha256 != run.json's - no candidate")
        init = torch.load(fin, map_location="cpu", weights_only=False).get("init_sha256")
        if not init or (a.parent_sha and a.parent_sha != init):
            raise SystemExit(f"--parent-sha {str(a.parent_sha)[:16]} != the candidate's init_sha256 {str(init)[:16]} "
                             "(refused: the N1 arms are checked against the candidate's own parent)")
        a.parent_sha = init
    res = after(a.dir, card, t01, a.neg or None, gate=gate, probe=a.probe or None, parent_sha=a.parent_sha or None,
                cand_sha=cand)
    print("\n".join(res["text"]))
    return 0 if res["m2"]["verdict"] == "PASSED" else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
