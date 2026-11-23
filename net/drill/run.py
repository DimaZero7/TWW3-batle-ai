"""The engine's one training entry point (tmp/engine/design_merged.md section 4; brief_A2 Parts 1-2): any card of any
level, the level and the card read through their interfaces (net/drill/levels/__init__.py, net/drill/cards/__init__.py).

    python -m net.drill.run train --card ID --out DIR [--init PATH] [--parent-run DIR] [--schedule JSON] [--smoke N]
        [--no-v] [--no-general] [--resume PATH] [--registry FILE] [--v-cache DIR] [--dump-fit] [--device cpu]
    python -m net.drill.run adopt --card ID --out DIR --weights PATH --reason TEXT

adopt   DIR/adopt.json, written by this command only: the card's accepted reference (--weights, its sha256 = the card's
        reference's) adopted as the card's accepted weights for a run in DIR that did not pass - the reason, the sha,
        the run's run.json and each round's done-record (round, sha256, stop, passed, run key). Refused: a run.json
        passed true, other weights, an empty reason, an adopt record of other weights there.

train   the schedule (net/config/drill.json "schedule" or --schedule; absent / malformed: exit naming the key; kind
        'aggregate' refused - not built; n_max beyond the level's registered rounds refused). The seed registry is
        mandatory (default net/drill/registry.json; a missing file is an error; its checks run first; the collector
        checks every chunk's realised starts against this card's training ranges). A card with a parent needs --init
        (its sha256 checked against the parent's run folder - --parent-run, or the folder of --init: a run.json passed
        true with that sha, or an adopt record of the parent card with that sha; neither: refused); V0 = the parent's marks, read before any child weight is loaded, kept in DIR/v_round_00.json with the
        parent's sha and the run key (another sha or key: refused). Per round: one collection (level.collect_round into
        DIR/collect_rNN/, a writer's lock, the manifest keyed by run key and round), the mixture, the weights, the census
        (census.check with the card's minima; a card with a parent: old_masks against the parent's masks), `passes`
        passes (fit.passes with level.batch_of), V and the stop (validate.relative_stop), the round's checkpoint and its
        done-record DIR/round_NN.done.json (checkpoint sha256, run key, stop, passed, the manifest's sha256).
Stop: 'relative' - the marks judged and passed, from n_min; at n_max with the marks not passed: "cap, marks not
        passed" (run.json passed false, no final.pt, exit code 1); schedule stop 'cap': at n_max, passed = the marks.
        DIR/STOP: a done-record of its own (stop.done.json), passed false, exit 1.
Resume: the done rounds of THIS run key (a checkpoint of another key refused; each done-record's checkpoint and
        manifest digests verified); a done round beyond this cap refused (the folder holds a longer run); a finished
        run (a done-record carrying a stop) republishes THAT round and reason; --resume PATH: PATH's own done-record
        (next to it) verified, a finished one published from PATH itself.
Run key: the card id, the engine's files (run.ENGINE, levels/__init__.py and this level's levels/<level>*.py), the
        modules of this card and its parents (+ cards/__init__.py and the cards' declared files), the level's
        IMPL_FILES, the level's configs with every cap removed, the census config, the parent's sha, the schedule less
        n_max, the minibatch order, smoke / no_general / no_v, the simulator key, the registry's sha, the device kind.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from net.drill import store as ST

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "net" / "config" / "drill.json"
REGISTRY = ROOT / "net" / "drill" / "registry.json"
SCHEDULE_KEYS = {"kind": ("rounds", "aggregate"), "passes": int, "n_min": int, "n_max": int,
                 "stop": ("relative", "cap")}
CAP_FAIL = "cap, marks not passed"
NOT_JUDGED = "V not judged"
_NOT_JUDGED_WHY = ("Astra t02_fix_review (visible) and the coordinator 23.11: a round beyond n_min is trained only on a "
                   "NAMED MEASURED failed V mark - a V whose cells did not gather their minimum (validate.relative_stop "
                   "judged False) stops the run as not judged instead of training on; --no-v runs keep the cap")
ADOPT = "adopt.json"


def conf():
    return json.loads(CONF.read_text(encoding="utf-8"))


def schedule_of(given):
    """The schedule dict, validated; SystemExit naming the key when absent or malformed."""
    src = "--schedule"
    if given:
        try:
            s = json.loads(Path(given).read_text(encoding="utf-8")) if Path(given).is_file() else json.loads(given)
        except ValueError as e:
            raise SystemExit(f"--schedule: not JSON ({e})")
    else:
        rel = CONF.relative_to(ROOT).as_posix() if CONF.is_relative_to(ROOT) else CONF.as_posix()
        src = f"{rel} key 'schedule'"
        s = conf().get("schedule")
        if s is None:
            raise SystemExit(f"{src}: absent - the schedule slot is empty (give --schedule explicitly)")
    if not isinstance(s, dict):
        raise SystemExit(f"{src}: malformed - not an object")
    for k, t in SCHEDULE_KEYS.items():
        if k not in s:
            raise SystemExit(f"{src}: malformed - key 'schedule.{k}' absent")
        v = s[k]
        if isinstance(t, tuple):
            if v not in t:
                raise SystemExit(f"{src}: malformed - key 'schedule.{k}' = {v!r}, one of {t}")
        elif not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise SystemExit(f"{src}: malformed - key 'schedule.{k}' = {v!r}, an integer >= 1")
    if s["n_max"] < s["n_min"]:
        raise SystemExit(f"{src}: malformed - key 'schedule.n_max' {s['n_max']} < n_min {s['n_min']}")
    if s["kind"] != "rounds":
        raise SystemExit(f"{src}: key 'schedule.kind' = {s['kind']!r} - only 'rounds' is built (v1)")
    return s


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _files_sha(files):
    h = hashlib.sha256()
    for p in sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix()):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


ENGINE = ("__init__", "binds", "census", "collect", "exam", "fit", "masks", "record", "run", "seeds", "store",
          "validate", "verdict")
_ENGINE_WHY = ("the level-blind modules a training reads, listed: a level's own files (net/drill/probe.py - the "
               "operative's probe) and another level's adapter stay out of the key (a restart on 22.11 was refused "
               "after an edit of probe.py alone)")


def engine_files(level_name):
    d = ROOT / "net" / "drill"
    fs = [d / f"{m}.py" for m in ENGINE] + [d / "levels" / "__init__.py"] + list((d / "levels").glob(
        f"{level_name}*.py"))
    return [p for p in fs if p.is_file()]


def card_files(chain, cards):
    from net.drill.cards import module_file
    fs = [ROOT / "net" / "drill" / "cards" / "__init__.py"] + [module_file(c) for c in chain]
    for c in cards:
        fs += [ROOT / f for f in c.files if (ROOT / f).is_file()]
    return fs


def impl_files(globs):
    out = []
    for g in globs:
        out += [Path(p) for p in glob.glob(str(ROOT / g), recursive=True) if "__pycache__" not in p]
    return out


def run_key(a, sched, card_chain, cards, level, lconf, parent_sha, registry_sha, v_key=None):
    x = {"card": a.card, "engine": _files_sha(engine_files(level.NAME)), "cards": _files_sha(card_files(card_chain,
         cards)), "impl": _files_sha(impl_files(level.IMPL_FILES)), "level": lconf["key"],
         "census": conf().get("census"), "parent": parent_sha,
         "schedule": {k: v for k, v in sched.items() if k != "n_max"},
         "minibatch_order": "torch.randperm(gen) a pass, gen = Generator().manual_seed(the level's tseed)",
         "smoke": a.smoke, "no_general": a.no_general, "no_v": a.no_v, "sim_key": level.sim_key(),
         "registry": registry_sha, "device": str(a.device).split(":")[0]}
    if v_key is not None:
        x["v_key"] = v_key                              # (Astra engine_code3 finding 1: V's play key pinned)
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _done(out, r):
    return Path(out) / f"round_{r:02d}.done.json"


def _verify(ck_path, rec, key):
    if rec.get("run_key") != key:
        raise SystemExit(f"{ck_path}: its done-record's run key {rec.get('run_key')} != this run's {key} (refused)")
    if rec.get("sha256") != _sha(ck_path):
        raise SystemExit(f"{ck_path}: sha256 differs from its done-record")
    mf = Path(ck_path).parent / f"collect_r{int(rec['round']):02d}" / "manifest.json"
    if rec.get("shards_manifest") is not None and (not mf.exists() or ST.sha256(mf) != rec["shards_manifest"]):
        raise SystemExit(f"{mf}: the shards' manifest differs from the done-record's digest (or is missing)")


def resume_state(a, out, key, sched):
    """(the latest done record or None, its checkpoint path, a finished record or None) - module doc."""
    if a.resume:
        p = Path(a.resume)
        r = int(p.stem.split("_")[1]) if p.stem.startswith("round_") else None
        dr = _done(p.parent, r) if r is not None else None
        if dr is None or not dr.exists():
            raise SystemExit(f"--resume {p}: no done-record next to it (round_NN.done.json) - refused")
        rec = json.loads(dr.read_text(encoding="utf-8"))
        _verify(p, rec, key)
        if rec["round"] > sched["n_max"]:
            raise SystemExit(f"--resume {p}: round {rec['round']} beyond this schedule's n_max {sched['n_max']}")
        return rec, p, (rec if rec.get("stop") else None)
    best = fin = None
    for p in sorted(out.glob("round_[0-9][0-9].pt")):
        ck_key = torch.load(p, map_location="cpu", weights_only=False).get("run_key")
        if ck_key != key:
            raise SystemExit(f"{p}: a checkpoint of another run (run key {ck_key} != {key}) in --out {out} - give a "
                             "clean folder or --resume explicitly")
        dr = _done(out, int(p.stem.split("_")[1]))
        if not dr.exists():
            continue                                    # (no done-record: an unfinished round, collected again)
        rec = json.loads(dr.read_text(encoding="utf-8"))
        _verify(p, rec, key)
        if fin is None and rec.get("stop"):
            fin = (rec, p)
        best = (rec, p)
    if (out / "stop.done.json").exists():
        rec = json.loads((out / "stop.done.json").read_text(encoding="utf-8"))
        if rec.get("run_key") == key and fin is None:
            fin = (rec, Path(rec["checkpoint"]) if rec.get("checkpoint") else None)
    # (the cap check BEFORE a finished record is returned - brief_A3 R4a: a restart under a lowered cap never
    # republishes a round beyond it)
    for x in (best, fin):
        if x is not None and int(x[0].get("round") or 0) > sched["n_max"]:
            raise SystemExit(f"{out}: done round {x[0]['round']} beyond this schedule's n_max {sched['n_max']} - the "
                             "folder holds a longer run (refused: give a clean folder)")
    if fin is not None:
        return fin[0], fin[1], fin[0]
    return (best[0], best[1], None) if best else (None, None, None)


def publish(out, key, rec, ck_path, log, card=None):
    """run.json (and final.pt when passed) of the finished round `rec` from its own checkpoint `ck_path`; card - the
    card's module name (parent_of's passed-run identity)."""
    passed = bool(rec.get("passed"))
    res = {"round": rec.get("round"), "stop": rec.get("stop"), "passed": passed, "run_key": key,
           "checkpoint": str(ck_path) if ck_path else None, "card": card}
    if passed:
        part = out / "final.pt.part"
        shutil.copyfile(ck_path, part)
        with open(part, "rb+") as f:                   # (fsync the written handle, then the replace)
            f.flush()
            os.fsync(f.fileno())
        os.replace(part, out / "final.pt")
        res.update(final=str(out / "final.pt"), sha256=_sha(out / "final.pt"))
    ST.atomic_text(json.dumps(res, indent=1), out / "run.json")
    log({"done": res})
    return res


def v0_of(f0, parent_sha, key, read, log=print, brief=None, who="parent_sha", v_key=None):
    """V0 (module doc): the stored file when its parent sha, run key and V play key are this run's (else refused - a
    V0 of another play never meets this run's marks); otherwise read (the parent's network, nothing loaded over it
    yet) and stored with them. who='reference_sha': the reference's V read (v_round_ref.json, the same form)."""
    f0 = Path(f0)
    if f0.exists():
        x = json.loads(f0.read_text(encoding="utf-8"))
        vk = x.get("v_key", (x.get("v") or {}).get("cache", {}).get("key"))
        if x.get(who) != parent_sha or x.get("run_key") != key or (v_key is not None and vk != v_key):
            raise SystemExit(f"{f0}: {f0.stem} of {who} {str(x.get(who))[:16]} / run key {x.get('run_key')} / V key "
                             f"{vk} - this run's {str(parent_sha)[:16]} / {key} / {v_key} (refused)")
        return x["v"]
    tv0 = read()
    got = (tv0 or {}).get("cache", {}).get("key") if isinstance(tv0, dict) else None
    if v_key is not None and got is not None and got != v_key:
        raise SystemExit(f"{f0}: read under V key {got} != the run's pinned {v_key} (refused)")
    ST.atomic_text(json.dumps({who: parent_sha, "run_key": key, "v_key": v_key, "v": tv0}, indent=1, default=str), f0)
    log({f0.stem: brief(tv0) if brief else "read"})
    return tv0


def reference_of(card, level, dev, log):
    """(net, sha) of card.reference - its sha256 checked against the card's (refused otherwise)."""
    ref = card.reference
    p = ROOT / ref["path"]
    sha = _sha(p)
    if sha != ref["sha256"]:
        raise SystemExit(f"reference {p}: sha {sha[:16]} != the card's {ref['sha256'][:16]} (refused)")
    log({"reference": ref["path"], "sha256": sha, "why": ref.get("why")})
    return level.net_reference(card, str(p), dev), sha


def _rng_kept(fn):
    """fn() with the torch (CPU, CUDA), random and numpy states restored after it: a read before the training leaves
    the training's draws as they were."""
    import random
    import numpy as np
    st = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
          random.getstate(), np.random.get_state())
    try:
        return fn()
    finally:
        torch.set_rng_state(st[0])
        if st[1] is not None:
            torch.cuda.set_rng_state_all(st[1])
        random.setstate(st[2])
        np.random.set_state(st[3])


def parent_of(a, card, level, dev, log):
    """(net, parent sha) - from zero (no parent) or the parent's accepted weights (--init, sha checked)."""
    if card.parent is None:
        if a.init:
            raise SystemExit(f"--card {a.card}: no parent - --init is not taken (the network starts from zero)")
        return level.net_new(card, dev), None
    if not a.init:
        raise SystemExit(f"--card {a.card}: its parent is {card.parent} - give --init <the parent's accepted "
                         "final.pt> (no default weights)")
    sha = _sha(a.init)
    d = Path(a.parent_run) if a.parent_run else Path(a.init).parent
    pr, pa = d / "run.json", d / ADOPT
    r = json.loads(pr.read_text(encoding="utf-8")) if pr.exists() else None
    if r is not None and r.get("passed"):
        if r.get("sha256") != sha or r.get("card") != card.parent:
            raise SystemExit(f"--init {a.init}: sha {sha[:16]} / card {card.parent} against the parent's {pr} (passed, "
                             f"sha {str(r.get('sha256'))[:16]}, card {r.get('card')}) - refused")
        log({"parent_run": str(pr), "sha256": sha})
    elif pa.exists():
        ad = json.loads(pa.read_text(encoding="utf-8"))
        # (Astra engine_code3 finding 5: the record's claim rechecked on consumption - the parent card's declared
        # reference, the fields only the command writes, the folder's own run.json and done-records)
        from net.drill import cards as CD
        pref = getattr(CD.get(card.parent), "reference", None) if card.parent else None
        bad = []
        if pref is None or pref.get("sha256") != sha:
            bad.append(f"the parent card {card.parent}'s reference sha {str((pref or {}).get('sha256'))[:16]} != "
                       f"--init's {sha[:16]}")
        if ad.get("sha256") != sha or ad.get("card") != card.parent:
            bad.append(f"the record's sha {str(ad.get('sha256'))[:16]} / card {ad.get('card')}")
        if not all(ad.get(k) for k in ("reason", "time")) or "rounds" not in ad or "run" not in ad:
            bad.append("not written by `run adopt` (reason / time / rounds / run missing)")
        elif ad["run"] != (None if r is None else {k: r.get(k) for k in ("round", "stop", "passed", "run_key",
                                                                           "checkpoint")}):
            bad.append("its run differs from the folder's run.json")
        elif ad["rounds"] != _rounds_of(d):
            bad.append("its rounds differ from the folder's done-records")
        if bad:
            raise SystemExit(f"--init {a.init}: the adopt record {pa} refused: {bad}")
        log({"parent_adopted": str(pa), "sha256": sha, "reason": ad.get("reason"),
             "run": ad.get("run"), "run_keys": ad.get("run_keys")})
    else:
        raise SystemExit(f"--init {a.init}: {d} holds neither a passed run.json of the parent"
                         f"{'' if r is None else ' (its run.json says passed false)'} nor an adopt record ({ADOPT}: "
                         f"python -m net.drill.run adopt) - refused; give --parent-run <the parent's run folder>")
    return level.net_parent(card, a.init, dev), sha


def _rounds_of(out):
    return [{k: r.get(k) for k in ("round", "sha256", "stop", "passed", "run_key")}
            for r in (json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(out).glob("round_[0-9][0-9].done.json")))]


def probe_of(a):
    """(METHOD step 4; Astra engine_code3 finding 9) a card module defining PROBE trains (not a smoke) only with
    --probe: a probe record whose verdict passed with n_ok (levels/tactic_probe's probe.json) of this card."""
    import importlib
    mod = importlib.import_module(f"net.drill.cards.{a.card}")
    if getattr(mod, "PROBE", None) is None or a.smoke:
        return None
    if not a.probe or not Path(a.probe).is_file():
        raise SystemExit(f"--card {a.card} defines a label probe: give --probe <its passed probe.json> (METHOD step 4: "
                         "the probe passes before the training) - refused")
    pr = json.loads(Path(a.probe).read_text(encoding="utf-8"))
    v = pr.get("verdict") or {}
    if not (v.get("pass") and v.get("n_ok")) or pr.get("card", a.card) not in (a.card, CD_id(a.card)):
        raise SystemExit(f"--probe {a.probe}: verdict pass {v.get('pass')} / n_ok {v.get('n_ok')} / card "
                         f"{pr.get('card')} - a passed probe of --card {a.card} is required (refused)")
    return _sha(a.probe)


def CD_id(card):
    from net.drill import cards as CD
    return CD.get(card).id


def adopt(a):
    """`adopt` (module doc): DIR/adopt.json - the card's accepted reference adopted as its accepted weights for a run
    that did not pass. Refused: a run.json passed true (its final.pt is the parent), weights other than the card's
    reference, an empty reason, an adopt record already there for other weights."""
    from net.drill import cards as CD
    card = CD.get(a.card)
    out = Path(a.out)
    ref = getattr(card, "reference", None)
    if ref is None:
        raise SystemExit(f"adopt --card {a.card}: the card has no reference - nothing accepted to adopt")
    if not a.reason.strip():
        raise SystemExit("adopt: --reason is required (the line the decision was written in)")
    sha = _sha(a.weights)
    if sha != ref["sha256"]:
        raise SystemExit(f"adopt --weights {a.weights}: sha {sha[:16]} != the card's reference {ref['path']} "
                         f"({ref['sha256'][:16]}) - only the accepted network is adopted")
    rj = out / "run.json"
    run = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else None
    if run is not None and run.get("passed"):
        raise SystemExit(f"adopt: {rj} says passed true - its final.pt is the parent, nothing to adopt")
    rounds = _rounds_of(out)
    keys = sorted({r["run_key"] for r in rounds if r.get("run_key")})
    rec = {"card": a.card, "card_id": card.id, "weights": str(a.weights), "sha256": sha, "reason": a.reason,
           "run": None if run is None else {k: run.get(k) for k in ("round", "stop", "passed", "run_key", "checkpoint")},
           "run_keys": keys, "rounds": rounds, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    f = out / ADOPT
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old.get("sha256") != sha or old.get("card") != a.card:
            raise SystemExit(f"{f}: an adopt record of {old.get('card')} / {str(old.get('sha256'))[:16]} already - refused")
    out.mkdir(parents=True, exist_ok=True)
    ST.atomic_text(json.dumps(rec, indent=1), f)
    print(json.dumps({"adopt": str(f), "sha256": sha[:16], "rounds": len(rounds), "run_keys": keys}))
    return {"passed": True, "adopt": str(f)}


def train(a):
    from sim.params import load
    from net.drill import cards as CD
    from net.drill import census as CN
    from net.drill import collect as CL
    from net.drill import fit as FT
    from net.drill import levels as LV
    from net.drill import seeds as SD
    from net.drill import validate as VA
    sched = schedule_of(a.schedule)
    card = CD.get(a.card)
    chain = CD.chain(a.card)
    cards = [CD.get(c) for c in chain]
    level = LV.get(card.level)
    lconf = level.conf(card)
    if sched["n_max"] > int(lconf.get("rounds_registered", sched["n_max"])):
        raise SystemExit(f"schedule n_max {sched['n_max']} beyond the {lconf['rounds_registered']} rounds the seed "
                         "registry holds for this level (refused)")
    probe_of(a)
    reg_path = Path(a.registry) if a.registry else REGISTRY
    entries = SD.load(reg_path)                         # (mandatory; its overlap checks)
    reg_sha = _sha(reg_path)
    P = load()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a", encoding="utf-8")

    def log(x):
        s = json.dumps(x, default=str) if not isinstance(x, str) else x
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    dev = a.device
    # (the legacy order of the random draws: the seed, the network, the optimiser, the minibatch generator)
    torch.manual_seed(int(lconf["tseed"]))
    net, parent_sha = parent_of(a, card, level, dev, log)
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=float(lconf["lr"]))
    gen = torch.Generator().manual_seed(int(lconf["tseed"]))
    vkey = level.v_key(card, a, int(lconf["chunk"])) if (hasattr(level, "v_key") and not a.no_v) else None
    key = run_key(a, sched, chain, cards, level, lconf, parent_sha, reg_sha, vkey)
    log({"start": time.strftime("%H:%M"), "card": a.card, "level": level.NAME, "parent": card.parent,
         "schedule": sched, "run_key": key, "device": dev, "smoke": a.smoke, "no_v": a.no_v, "init": a.init or None,
         "registry": str(reg_path), "v_key": vkey})
    if hasattr(level, "seed_check"):
        level.seed_check(card, log)
    ctx = SimpleNamespace(device=dev, sim=a.sim, smoke=a.smoke, no_general=a.no_general, params=P,
                          chunk=int(lconf["chunk"]), log=log, registry=entries, out=out, run_key=key, check_starts=None,
                          check_starts_v=CL.starts_check(entries, chain, "V"), v_key=vkey)
    vcache = Path(a.v_cache) if a.v_cache else out / "v_cache"
    rec, ck_path, fin = resume_state(a, out, key, sched)
    if fin is not None:
        log({"FINISHED_BEFORE": fin.get("stop"), "round": fin.get("round")})
        res = publish(out, key, fin, ck_path, log, a.card)
        logf.close()
        return res
    # V0: the parent's marks, before any child weight is loaded
    tv0 = None
    if card.parent is not None and not a.no_v:
        tv0 = v0_of(out / "v_round_00.json", parent_sha, key, lambda: level.v_read(card, net, ctx, vcache), log,
                    level.brief, v_key=vkey)
    # the reference's marks (card.reference, brief_A3 R1): read once, kept beside V0's form; the RNG left as it was
    tv_ref = None
    if card.reference is not None and not a.no_v:
        f_ref = out / "v_round_ref.json"
        if f_ref.exists():
            tv_ref = v0_of(f_ref, card.reference["sha256"], key, None, log, None, who="reference_sha", v_key=vkey)
        else:
            rnet, rsha = reference_of(card, level, dev, log)
            tv_ref = v0_of(f_ref, rsha, key, lambda: _rng_kept(lambda: level.v_read(card, rnet, ctx, vcache)), log,
                           level.brief, who="reference_sha", v_key=vkey)
            del rnet
    r0 = 0
    if rec is not None:
        ck = level.load(ck_path, dev)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen_state"].cpu())
        torch.set_rng_state(ck["rng"].cpu())
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        r0 = int(rec["round"])
        log({"RESUME": f"round {r0}", "checkpoint": str(ck_path)})
    last = (rec, ck_path) if rec is not None else None
    for r in range(r0, sched["n_max"]):
        if (out / "STOP").exists():
            srec = {"round": last[0]["round"] if last else 0, "run_key": key, "stop": "STOP file", "passed": False,
                    "checkpoint": str(last[1]) if last else None}
            ST.atomic_text(json.dumps(srec, indent=1), out / "stop.done.json")
            log({"STOP": "the STOP file"})
            res = publish(out, key, srec, last[1] if last else None, log, a.card)
            logf.close()
            return res
        t0 = time.time()
        store, stc = CL.round_of(level, card, r, net, ctx, out, chain)
        rows, cards_t, meta = ST.mixture_of([store], conf().get("ram_gb"), getattr(level, "remap", None),
                                            getattr(level, "pad", None))
        w, winfo = level.weights(card, rows, meta)
        log({"round": r + 1, "rows": int(rows["b"].shape[0]), "bytes": store.bytes(), **stc})
        log({"weights": winfo})
        batch = level.census_batch(rows)
        cen = CN.check(level.census_classes(card, batch, w, meta), meta, batch.b, card, a.smoke,
                       level.census_extra(card, batch, rows, meta, w), stc.get("done_rows"))
        log({"census": cen})
        if card.parent is not None and hasattr(level, "parent_masks"):
            pf, cf, clf = level.parent_masks(card)
            mv = CN.moved(CN.old_masks(batch, pf, cf, clf(batch, meta)), card)
            log({"old_masks": mv})
        del batch
        if a.dump_fit:
            ST.atomic_save({"rows": rows, "cards": cards_t, "meta": meta, "w": w, "K": int(stc["NK"][1]),
                            "net": {k: v.clone() for k, v in net.state_dict().items()},
                            "opt": opt.state_dict(), "gen": gen.get_state().clone(),
                            "rng": torch.get_rng_state().clone()}, out / f"fit_args_r{r + 1:02d}.pt")
        hist, terms = FT.passes(level, net, opt, rows, cards_t, w, dev, log, gen, sched["passes"],
                                int(lconf["minibatch"]), float(lconf["max_grad"]), pre_pass=bool(lconf["pre_pass"]),
                                batch_of=level.batch_of)
        del rows, cards_t, w
        n_done = r + 1
        stop, passed, vrow = None, False, None
        ok = False
        if not a.no_v:
            tv = level.v_read(card, net, ctx, vcache)
            ST.atomic_text(json.dumps(tv, indent=1, default=str), out / f"v_round_{n_done:02d}.json")
            rep = []
            ok, why, judged = VA.relative_stop(card, tv, tv0, tv_ref, rep)
            vrow = {"V": level.brief(tv), "stop_ok": ok, "judged": judged, "why": why, "report": rep,
                    "teacher_path": tv.get("cache", {}).get("teacher_path")}
            log({"V_round": n_done, **vrow})
        if sched["stop"] == "relative" and ok and n_done >= sched["n_min"]:
            stop, passed = "relative", True
        elif not a.no_v and not judged and n_done >= sched["n_min"]:
            stop, passed = NOT_JUDGED, False
        elif n_done >= sched["n_max"]:
            stop, passed = ("cap", True) if (sched["stop"] == "cap" and ok) else (CAP_FAIL, False)
        ck_path = out / f"round_{n_done:02d}.pt"
        state = {"run_key": key, "round": n_done, "gen_state": gen.get_state(), "rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "shards": store.digest(), "schedule": sched, "stop": stop, "card": a.card,
                 "unit_commit": a.unit, "sim_commit": a.sim, "init": a.init or None, "init_sha256": parent_sha,
                 "row": {"round": n_done, "loss": hist, "terms": terms, "V": vrow,
                         "min": round((time.time() - t0) / 60, 1)}}
        level.save(ck_path, net, opt, card, state)
        drec = {"round": n_done, "run_key": key, "sha256": _sha(ck_path), "stop": stop, "passed": passed,
                "shards_manifest": store.digest(), "v_key": vkey}
        ST.atomic_text(json.dumps(drec, indent=1), _done(out, n_done))
        log({"ROUND": state["row"], "ckpt": str(ck_path), "stop": stop})
        last = (drec, ck_path)
        if stop:
            break
    if last is None:
        raise SystemExit("no round finished")
    res = publish(out, key, last[0], last[1], log, a.card)
    logf.close()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("train", "adopt"))
    ap.add_argument("--card", required=True, help="a card module of net/drill/cards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="", help="adopt: the card's accepted reference weights")
    ap.add_argument("--reason", default="", help="adopt: why (the line the decision was written in)")
    ap.add_argument("--schedule", default="", help="the schedule as JSON (or a JSON file); default drill.json's")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--no-general", action="store_true")
    ap.add_argument("--no-v", action="store_true")
    ap.add_argument("--init", default="", help="the parent card's accepted weights (required for a card with a parent)")
    ap.add_argument("--parent-run", default="", help="the parent's run folder (its run.json checks --init's sha)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--probe", default="", help="train: a card with a label probe needs its passed probe.json")
    ap.add_argument("--registry", default="", help="the seed registry (default net/drill/registry.json; mandatory)")
    ap.add_argument("--v-cache", default="", help="the teacher path's cache of V (default DIR/v_cache; keyed)")
    ap.add_argument("--dump-fit", action="store_true", help="each round's fit arguments to fit_args_rNN.pt (checks)")
    ap.add_argument("--sim", default="")
    ap.add_argument("--unit", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    return adopt(a) if a.what == "adopt" else train(a)


if __name__ == "__main__":
    sys.exit(0 if (main() or {}).get("passed") else 1)
