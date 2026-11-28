"""The lord's T2 script, the same decision in the simulator and through the bridge (contract/parity.md "лорд"): the
batched executor net/unit/lord.T2 (the simulator's, on the input's columns lord.lord_inputs_doc) and the bridge's
bridge/companion/lord_script.LordScript on the SAME recorded game states - the T2 series in the game (27.11, game 9.0
"Normal", ksl_plains_01, arm T of tmp/lordline/t2/runs.table.txt; build/nn-arena/runs/<id>).

The documents: the companion log keeps the full state document only for its first 60 s, so each second's document is
rebuilt from the bridge's own event log (events.jsonl): the units from nn_sample (the state document minus v, fx and
key), fx from the nn_effects stream, abilities_used / abilities_count from nn_ability (status used), key from the first
document, v from the logged document where there is one, else true (the sample does not carry it). On the 60 s with a
logged document the rebuilt one equals it in every field the two scripts read (asserted). Each second both scripts get
the document and the reply's lines for our other units (the order in force: the run-in's onset).

A decision tick = a second the lord stands. Agreement = the same order kind, the same target, the point within 1 m, the
same press. Lines (written before the run, the orchestrator's brief 27.11): agreement >= 0.98 of the ticks, >= 2000
ticks from >= 6 runs; every disagreement in a named class:
    off_call    the press differs where the document holds a call of that button the replayed bridge script never
                took (the recorded run's own retry after a refusal - the documents were played by the script with
                READY_PAD_S 0, so with 1 s the replay requests a second later than the recorded calls): the bridge
                reads a call outside its request window as nobody's and presses into the real recharge, the executor
                reads the button's timer (exchange.ability_timers) and waits
    wall        the bridge script stopped that button after LS.TRIES misses (its stop rule, on those same off-policy
                refusals) while the executor, which has no stop rule of its own beyond the count, presses
    other       anything else - none expected
With READY_PAD_S 0 - the documents' own script - every tick agrees and the replayed bridge decisions are the recorded
ones (order line and press) - the rebuilt documents' check.

The tactic mode (the executor commanded by the tactic level, net/unit/lord.py / lord_script.py 'modes'): the recorded
runs never carried a tactic lord command, so the replay builds one from the recorded states - the tactic level
(tactic/side.Side, the infantry player K that keeps, the lord player LT of tactic/lordtac.py) ticked every 5 s on the
documents' raw, its lord store's target fed to both executors each second (the bridge's by name), net/unit/lord.engaged_of
of the document's raw as the bridge's 'engaged', the torch executor's drop mirrored into the store (and the bridge's
drop must be the same); buttons off in both (the button classes above are the t2 mode's). The same lines: agreement
(kind, target, point, drop) >= 0.98 of >= 2000 ticks from >= 6 runs, no 'other'.

The group mode (the lord a group of one anchored to the infantry, tactic/side.Side group mode): the same, the command
the lord's row's target in the common order store under the infantry's player T (no lord player), both executors given
the anchor's members; the same lines.
"""
import json
import math
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "build" / "nn-arena" / "runs"
# arm T of the series T2 (tmp/lordline/t2/runs.table.txt, the 16 sound attempts; code fp f9e13a8ec9c60b5d)
T_RUNS = ("20261127-155637", "20261127-160539", "20261127-160906", "20261127-161913", "20261127-162220",
          "20261127-163202", "20261127-163520", "20261127-164447", "20261127-164755", "20261127-165651",
          "20261127-165934", "20261127-170909", "20261127-171215", "20261127-172231", "20261127-172534",
          "20261127-173442")
SHARE, TICKS, MIN_RUNS, POINT_M = 0.98, 2000, 6, 1.0


def rebuild(run):
    """[(document, the logged document or None, the companion's reply row)] of a run, one a second (module doc)."""
    S, E, A = {}, [], []
    for line in open(run / "events.jsonl", encoding="utf-8"):
        j = json.loads(line)
        e = j.get("event")
        if e == "nn_sample":
            S[int(j["t"])] = j
        elif e == "nn_effects":
            E.append(j)
        elif e == "nn_ability" and j.get("status") == "used":
            A.append(j)
    D, R = {}, {}
    for line in open(run / "companion.jsonl", encoding="utf-8"):
        j = json.loads(line)
        ch = j.get("chain")
        if not isinstance(ch, dict):
            continue
        if ch.get("doc"):
            D[int(ch["doc"]["t"])] = ch["doc"]
        if "orders" in j:
            R[int(ch["sec"]) * 1000] = j
    doc0 = D[min(D)]
    keys = {u["n"]: u.get("key") for u in doc0["units"]}
    head = {k: v for k, v in doc0.items() if k not in ("units", "move", "t", "abilities_used", "abilities_count")}
    out = []
    for t in sorted(S):
        fx, used, cnt = {}, {}, {}
        for e in E:
            if int(e["t"]) <= t:
                fx[e["u"]] = e["fx"]
        for a in A:
            if int(a["t"]) < t:                   # (the call answers the document of its own ms: seen on the next)
                used.setdefault(a["u"], {})[a["key"]] = int(a["t"])
                cnt.setdefault(a["u"], {})[a["key"]] = cnt.get(a["u"], {}).get(a["key"], 0) + 1
        real = D.get(t)
        vis = {u["n"]: u.get("v") for u in real["units"]} if real else {}
        units = []
        for u in S[t]["units"]:
            r = {k: v for k, v in u.items() if k not in ("mge", "sv", "pcr", "phr")}
            r.update(key=keys.get(u["n"]), v=vis.get(u["n"], True), fx=fx.get(u["n"], {}))
            units.append(r)
        out.append(({**head, "move": t // 1000 + 1, "t": t, "units": units, "abilities_used": used,
                     "abilities_count": cnt}, real, R.get(t)))
    return out


def _fx(v):
    return sorted(v.values()) if isinstance(v, dict) else sorted(v or [])


def replay(run, pad):
    """Per tick {ok, why, bridge, torch, recorded_ok} of one run under READY_PAD_S = pad."""
    import torch
    from bridge.companion import lord_script as LS
    from bridge.companion import tactic_input as TI
    from net.unit import lord as L
    from sim import orders as O
    kind_of = {"keep": O.KEEP, "move": O.MOVE, "attack": O.ATTACK, "withdraw": O.WITHDRAW, "hold": O.HOLD}
    old = LS.READY_PAD_S
    LS.READY_PAD_S = pad
    try:
        rows = [(d, real, rep) for d, real, rep in rebuild(run) if rep is not None]
        d0 = rows[0][0]
        names = [u["n"] for u in d0["units"]]
        keys = [u.get("key") or "" for u in d0["units"]]
        ls = LS.LordScript(names, keys, [int(u["side"]) for u in d0["units"]])
        li = names.index(ls.lord)
        seer = TI.TacticSeer(device="cpu", allow_flat=True)
        ex = L.T2(torch.tensor([li]), [keys[li]], L.rank_of_names(names), "cpu")
        took = {k: set() for k in ls.btn}
        out = []
        for doc, real, rep in rows:
            if real is not None:                         # (the rebuilt document = the logged one, what both read)
                for a, b in zip(doc["units"], real["units"]):
                    for k in ("n", "x", "z", "b", "hp", "men", "m", "w", "r", "s", "mv", "t", "v", "side"):
                        assert a.get(k) == b.get(k), (run.name, doc["t"], a["n"], k)
                    assert _fx(a["fx"]) == _fx(b.get("fx")), (run.name, doc["t"], a["n"])
                assert doc["abilities_used"] == (real.get("abilities_used") or {}), (run.name, doc["t"])
            lines = [o for o in rep["orders"] if not o.get("out")]
            lo, uses = ls.decide(doc, lines)
            ls.commit(ls.lord in {o["unit"] for o in lines})
            for x in ls.note.get("took", ()):
                took[x["key"]].add(round(x["call"], 1))
            raw, tgt, a_on, a_cd = L.lord_inputs_doc(doc, names, seer.raw(doc))
            kinds = torch.full((1, len(names)), O.KEEP, dtype=torch.long)
            for o in lines:
                kinds[0, names.index(o["unit"])] = kind_of[o["kind"]]
            d = ex.decide(raw, tgt, a_on, a_cd, kinds, torch.tensor([doc["t"] / 1000.0]), torch.tensor([True]))
            k = int(d["kind"][0])
            b_ = {"kind": kind_of[lo["kind"]], "target": lo.get("target"),
                  "p": (lo["x"], lo["z"]) if "x" in lo else None, "press": uses[0]["key"] if uses else None}
            t_ = {"kind": k, "target": names[int(d["target"][0])] if k == O.ATTACK else None,
                  "p": (float(d["x"][0]), float(d["z"][0])) if k in (O.MOVE, O.WITHDRAW) else None,
                  "press": L.PRIORITY[int(d["key"][0])] if int(d["key"][0]) >= 0 else None}
            ok = b_["kind"] == t_["kind"] and b_["target"] == t_["target"] and b_["press"] == t_["press"]
            if ok and b_["p"] is not None:
                ok = math.hypot(b_["p"][0] - t_["p"][0], b_["p"][1] - t_["p"][1]) <= POINT_M
            why = None
            if not ok:
                key = b_["press"] or t_["press"]
                if b_["kind"] != t_["kind"] or b_["target"] != t_["target"] or b_["press"] == t_["press"] or key is None:
                    why = "other"
                elif ls.btn[key]["st"] in ("wall", "wall_wait") or (t_["press"] and ls.btn[key]["st"] != "ready"):
                    why = "wall" if ls.btn[key]["st"] in ("wall", "wall_wait") else "other"
                else:
                    c = (doc.get("abilities_used") or {}).get(ls.lord, {}).get(key)
                    why = "off_call" if c is not None and round(c / 1000.0, 1) not in took[key] else "other"
            # the recorded decision (the run's own lord line and press)
            rec = next((o for o in rep["orders"] if o["unit"] == ls.lord), None)
            r_press = [a["key"] for a in rep.get("abilities") or () if a.get("unit") == ls.lord]
            rec_ok = rec is not None and rec["kind"] == lo["kind"] and rec.get("target") == lo.get("target") and \
                rec.get("x") == lo.get("x") and rec.get("z") == lo.get("z") and r_press == [u["key"] for u in uses]
            st = LS.standing({u["n"]: u for u in doc["units"]}.get(ls.lord))
            out.append({"t": doc["t"] / 1000.0, "tick": st, "ok": ok, "why": why, "rec_ok": rec_ok,
                        "bridge": b_, "torch": t_})
        return out
    finally:
        LS.READY_PAD_S = old


def _have():
    pytest.importorskip("torch")
    runs = [RUNS / r for r in T_RUNS if (RUNS / r / "events.jsonl").exists() and (RUNS / r / "companion.jsonl").exists()]
    if len(runs) < MIN_RUNS:
        pytest.skip(f"the T2 game records: {len(runs)} of {len(T_RUNS)} runs under {RUNS}")
    return runs


def test_the_lord_script_sim_equals_bridge():
    """READY_PAD_S as committed (1 s): the lines of the module doc; the classes printed."""
    runs = _have()
    ticks = agree = 0
    cls, per = Counter(), {}
    for run in runs:
        recs = [r for r in replay(run, 1.0) if r["tick"]]
        per[run.name] = (sum(r["ok"] for r in recs), len(recs))
        ticks += len(recs)
        agree += sum(r["ok"] for r in recs)
        cls.update(r["why"] for r in recs if not r["ok"])
    print(json.dumps({"ticks": ticks, "agree": agree, "share": round(agree / ticks, 4), "runs": len(per),
                      "classes": dict(cls), "per_run": per}))
    assert len(per) >= MIN_RUNS and ticks >= TICKS
    assert agree / ticks >= SHARE, (agree, ticks, dict(cls))
    assert cls.get("other", 0) == 0, dict(cls)


def test_on_the_documents_own_script_all_agree():
    """READY_PAD_S 0 (the script the documents were played by): every tick agrees, and the replayed bridge script's
    decisions are the recorded ones - the rebuilt documents are what the bridge read."""
    runs = _have()[:4]
    for run in runs:
        recs = replay(run, 0.0)
        bad = [r for r in recs if r["tick"] and not r["ok"]]
        assert not bad, (run.name, bad[:3])
        nrec = [r for r in recs if not r["rec_ok"]]
        assert not nrec, (run.name, len(nrec), nrec[:2])


def replay_tactic(run):
    """Per tick {ok, why, tick, bridge, torch} of one run in mode tactic (module doc)."""
    import torch
    from bridge.companion import lord_script as LS
    from bridge.companion import tactic_input as TI
    from net.tactic import lordtac as LT
    from net.tactic import pin as PN
    from net.tactic import raw as TR
    from net.tactic import side as TS
    from net.unit import lord as L
    from sim import orders as O
    kind_of = {"keep": O.KEEP, "move": O.MOVE, "attack": O.ATTACK, "withdraw": O.WITHDRAW, "hold": O.HOLD}
    rows = [(d, real, rep) for d, real, rep in rebuild(run) if rep is not None]
    d0 = rows[0][0]
    names = [u["n"] for u in d0["units"]]
    keys = [u.get("key") or "" for u in d0["units"]]
    ls = LS.LordScript(names, keys, [int(u["side"]) for u in d0["units"]], mode="tactic", buttons=False)
    li = names.index(ls.lord)
    seer = TI.TacticSeer(device="cpu", allow_flat=True)
    ex = L.T2(torch.tensor([li]), [keys[li]], L.rank_of_names(names), "cpu", buttons=False, mode="tactic")
    side, nxt, out = None, None, []
    for doc, real, rep in rows:
        r = seer.raw(doc)
        sec = int(round(doc["t"] / 1000.0))
        if side is None:
            x, z, b, mine, hide = TS.of_raw(r, seer.card, 1)
            hide = torch.zeros_like(hide)
            hide[0, li] = True
            mine = mine & ~hide
            side = TS.Side(PN.Keep(), True, "cpu", lord_player=LT.player("LT")).start(x, z, b, mine, hide)
            nxt = sec
        if sec >= nxt:
            side.tick(r, seer.card, seer.P.map_half, now=float(sec))
            while nxt <= sec:
                nxt += 5
        tac = side.tac
        t = int(tac.l_tgt[0])
        cmd = names[t] if t >= 0 else None
        eng = bool(L.engaged_of(r[..., :TR.BASE], torch.tensor([li]), 1)[0])
        lines = [o for o in rep["orders"] if not o.get("out")]
        lo, _ = ls.decide(doc, lines, cmd=cmd, engaged=eng)
        ls.commit(ls.lord in {o["unit"] for o in lines})
        raw, tgt, a_on, a_cd = L.lord_inputs_doc(doc, names, r)
        kinds = torch.full((1, len(names)), O.KEEP, dtype=torch.long)
        for o in lines:
            kinds[0, names.index(o["unit"])] = kind_of[o["kind"]]
        d = ex.decide(raw, tgt, a_on, a_cd, kinds, torch.tensor([doc["t"] / 1000.0]), torch.tensor([True]),
                      cmd=tac.l_tgt.clone())
        for code in (L.DROP_GONE, L.DROP_STEP):
            tac.lord_drop(ex.drop == code, code)
        tac.l_busy = ex.busy.clone()
        k = int(d["kind"][0])
        b_ = {"kind": kind_of[lo["kind"]], "target": lo.get("target"), "p": (lo["x"], lo["z"]) if "x" in lo else None,
              "drop": ls.drop}
        t_ = {"kind": k, "target": names[int(d["target"][0])] if k == O.ATTACK else None,
              "p": (float(d["x"][0]), float(d["z"][0])) if k in (O.MOVE, O.WITHDRAW) else None,
              "drop": None if int(ex.drop[0]) == L.DROP_NONE else L.DROP_NAMES[int(ex.drop[0])]}
        ok = b_["kind"] == t_["kind"] and b_["target"] == t_["target"] and b_["drop"] == t_["drop"]
        if ok and b_["p"] is not None:
            ok = math.hypot(b_["p"][0] - t_["p"][0], b_["p"][1] - t_["p"][1]) <= POINT_M
        st = LS.standing({u["n"]: u for u in doc["units"]}.get(ls.lord))
        out.append({"t": doc["t"] / 1000.0, "tick": st, "ok": ok, "why": None if ok else "other", "cmd": cmd,
                    "bridge": b_, "torch": t_})
    return out


def test_the_tactic_mode_sim_equals_bridge():
    """Mode tactic on the replayed LT commands (module doc): the lines; the commands' count printed."""
    runs = _have()
    ticks = agree = cmds = drops = 0
    per = {}
    for run in runs:
        recs = [r for r in replay_tactic(run) if r["tick"]]
        per[run.name] = (sum(r["ok"] for r in recs), len(recs))
        ticks += len(recs)
        agree += sum(r["ok"] for r in recs)
        cmds += sum(r["cmd"] is not None for r in recs)
        drops += sum(r["torch"]["drop"] is not None for r in recs)
    print(json.dumps({"ticks": ticks, "agree": agree, "share": round(agree / ticks, 4), "runs": len(per),
                      "ticks_with_command": cmds, "drops": drops, "per_run": per}))
    assert len(per) >= MIN_RUNS and ticks >= TICKS
    assert cmds > 0, "no LT command in the replay: the tactic mode not exercised"
    assert agree / ticks >= SHARE, (agree, ticks)


def replay_group(run, drop_m_every=0):
    """Per tick {ok, why, tick, cmd, bridge, torch} of one run in mode group: the lord a group of one anchored to the
    infantry (tactic/side.Side group mode, the infantry player T - its attacks of the bound enemies give the lord's row
    its commands from the common order store). Two independent Sides - one fed by the bridge's LordScript
    (tactic_chain.group_mirror, its peek_post), one by the torch executor (its drops, busy, post, peek_post) - each
    with its own per-second Side.second; a tick agrees when both executors' decisions agree and both Sides' commands,
    masks, busy flags and posts agree. drop_m_every k > 0: every k-th document has the lord's m taken away (unread) -
    the bridge keeps 'melee unread', the torch executor gets m_known False."""
    import copy as _copy
    import torch
    from bridge.companion import lord_script as LS
    from bridge.companion import tactic_chain as TCH
    from bridge.companion import tactic_input as TI
    from net.tactic import pin as PN
    from net.tactic import raw as TR
    from net.tactic import side as TS
    from net.unit import lord as L
    from sim import orders as O
    kind_of = {"keep": O.KEEP, "move": O.MOVE, "attack": O.ATTACK, "withdraw": O.WITHDRAW, "hold": O.HOLD}
    rows = [(d, real, rep) for d, real, rep in rebuild(run) if rep is not None]
    d0 = rows[0][0]
    names = [u["n"] for u in d0["units"]]
    keys = [u.get("key") or "" for u in d0["units"]]
    ls = LS.LordScript(names, keys, [int(u["side"]) for u in d0["units"]], mode="group", buttons=False)
    li = names.index(ls.lord)
    seer = TI.TacticSeer(device="cpu", allow_flat=True)
    ex = L.T2(torch.tensor([li]), [keys[li]], L.rank_of_names(names), "cpu", buttons=False, mode="group")
    sb = st = None
    nxt, out, members, mine = None, [], None, None
    one = torch.ones(1, dtype=torch.bool)
    f = lambda v: torch.tensor([float(v)])                                                # noqa: E731
    for j, (doc, real, rep) in enumerate(rows):
        if drop_m_every and j % drop_m_every == 0:
            doc = _copy.deepcopy(doc)
            for u in doc["units"]:
                if u["n"] == ls.lord:
                    u.pop("m", None)
        lrow = {u["n"]: u for u in doc["units"]}.get(ls.lord) or {}
        m_known = torch.tensor([lrow.get("m") is not None])
        r = seer.raw(doc)
        sec = int(round(doc["t"] / 1000.0))
        if sb is None:
            x, z, b, mine, hide = TS.of_raw(r, seer.card, 1)
            hide = torch.zeros_like(hide)
            hide[0, li] = True
            mine = mine & ~hide
            members = [n for i, n in enumerate(names) if bool(mine[0, i])]
            sb = TS.Side(PN.player("T"), True, "cpu", group=True).start(x, z, b, mine, hide)
            st = TS.Side(PN.player("T"), True, "cpu", group=True).start(x, z, b, mine, hide)
            nxt = sec
        raw, tgt, a_on, a_cd = L.lord_inputs_doc(doc, names, r)
        eng = bool(L.engaged_of(r[..., :TR.BASE], torch.tensor([li]), 1)[0])
        tv = torch.tensor([doc["t"] / 1000.0])
        if sec >= nxt:
            cb = int(sb.lord_cmd()[0])
            post, h = ls.peek_post(doc, cmd=names[cb] if cb >= 0 else None, engaged=eng, members=members)
            if post is not None and h is not None:
                sb.posts(f(post[0]), f(post[1]), f(h), one)
            st.posts(*ex.peek_post(raw, tgt, a_on, a_cd, tv, one, cmd=st.lord_cmd().clone(), members=mine))
            sb.tick(r, seer.card, seer.P.map_half, now=float(sec))
            st.tick(r, seer.card, seer.P.map_half, now=float(sec))
            while nxt <= sec:
                nxt += 5
        sb.second(r, seer.card)
        st.second(raw, seer.card)
        c = int(sb.lord_cmd()[0])
        cmd = names[c] if c >= 0 else None
        lines = [o for o in rep["orders"] if not o.get("out")]
        lo, _ = ls.decide(doc, lines, cmd=cmd, engaged=eng, members=members)
        ls.commit(ls.lord in {o["unit"] for o in lines})
        TCH.group_mirror(sb, ls)
        kinds = torch.full((1, len(names)), O.KEEP, dtype=torch.long)
        for o in lines:
            kinds[0, names.index(o["unit"])] = kind_of[o["kind"]]
        d = ex.decide(raw, tgt, a_on, a_cd, kinds, tv, one, cmd=st.lord_cmd().clone(), members=mine, m_known=m_known)
        for code in (L.DROP_GONE, L.DROP_STEP):
            st.lord_drop(ex.drop == code, L.DROP_NAMES[code])
        st.busy = ex.busy.clone()
        st.posts(ex.lpx, ex.lpz, ex.h, ex.has_lpost & ex.has_h)
        k = int(d["kind"][0])
        b_ = {"kind": kind_of[lo["kind"]], "target": lo.get("target"), "p": (lo["x"], lo["z"]) if "x" in lo else None,
              "drop": ls.drop}
        t_ = {"kind": k, "target": names[int(d["target"][0])] if k == O.ATTACK else None,
              "p": (float(d["x"][0]), float(d["z"][0])) if k in (O.MOVE, O.WITHDRAW) else None,
              "drop": None if int(ex.drop[0]) == L.DROP_NONE else L.DROP_NAMES[int(ex.drop[0])]}
        ok = b_["kind"] == t_["kind"] and b_["target"] == t_["target"] and b_["drop"] == t_["drop"]
        if ok and b_["p"] is not None:
            ok = math.hypot(b_["p"][0] - t_["p"][0], b_["p"][1] - t_["p"][1]) <= POINT_M
        why = None if ok else "other"
        same = (int(sb.lord_cmd()[0]) == int(st.lord_cmd()[0]) and bool(sb.busy[0]) == bool(st.busy[0])
                and torch.equal(sb.mask, st.mask))
        if ok and not same:
            ok, why = False, "sides"
        out.append({"t": doc["t"] / 1000.0, "tick": LS.standing(lrow or None), "ok": ok, "why": why, "cmd": cmd,
                    "bridge": b_, "torch": t_, "unread": not bool(m_known[0])})
    return out


def test_the_group_mode_sim_equals_bridge():
    """Mode group (replay_group, the tactic mode's lines): agreement >= 0.98 of >= 2000 ticks of >= 6 runs, the
    store's command exercised."""
    runs = _have()
    ticks = agree = cmds = drops = 0
    per = {}
    for run in runs:
        recs = [r for r in replay_group(run) if r["tick"]]
        per[run.name] = (sum(r["ok"] for r in recs), len(recs))
        ticks += len(recs)
        agree += sum(r["ok"] for r in recs)
        cmds += sum(r["cmd"] is not None for r in recs)
        drops += sum(r["torch"]["drop"] is not None for r in recs)
    print(json.dumps({"mode": "group", "ticks": ticks, "agree": agree, "share": round(agree / ticks, 4),
                      "runs": len(per), "ticks_with_command": cmds, "drops": drops, "per_run": per}))
    assert len(per) >= MIN_RUNS and ticks >= TICKS
    assert cmds > 0, "no command from the lord's group row in the replay: mode group not exercised"
    assert agree / ticks >= SHARE, (agree, ticks)


def test_the_group_mode_with_unread_melee():
    """Mode group, every 7th document without the lord's m (unread): the bridge keeps 'melee unread', the torch
    executor (m_known False) the same - agreement >= 0.98 on the unread ticks of the runs."""
    runs = _have()
    n = ok = 0
    for run in runs:
        recs = [r for r in replay_group(run, drop_m_every=7) if r["tick"] and r["unread"]]
        n += len(recs)
        ok += sum(r["ok"] for r in recs)
    print(json.dumps({"mode": "group_unread_m", "ticks": n, "agree": ok}))
    assert n > 0 and ok / n >= SHARE, (ok, n)
