"""The label probe on the network's REAL input (tmp/engine/design_merged.md 5; Astra's engine review item 4) - the gate
before any training of O0-1: can OpNet1 see the cause, and only the cause?

    collect    a teacher-driven collection of card O0-1's scenes (cards/o0_1.scenes: n cause + n twins, even i come,
               odd i role) through the live runner with the recorder (decided rows only, no raw): probe_train under the
               card's disturbed teacher, probe_held under the plain teacher; one store.Store shard a set
    fit        a fresh OpNet1 a arm fitted on probe_train (record.loss, levels/operative.balanced: LS half the mass, KEEP's
               half split over the teacher's KEEP reasons), read after every
               pass on probe_held's FIRST decisions (start_s) of complete disjoint cause / come pairs (the same seed's
               cause scene and come twin, both with a decided first row, the teacher LS on the cause and KEEP on the come):
        intact     followed (the cause's first decision LS) >= FOLLOWED, false (the come's first decision LS) <= FALSE,
                   each n >= N_MIN
        ablated    refitted with the motion zeroed (ABLATE: the rows' velocity right / forward, moving and running flags;
                   gframe's velocities and closing): balanced accuracy on the same pairs within 0.5 +- BAND
               beside: every held change tick / KEEP tick, the train rows' followed share, the losses by pass
    python -m net.drill.probe collect --set probe_train --n 192 --out tmp/engine/b/probe
        every seed of --seeds (init seed s, shuffle s + 1) fits both arms; the gate holds on EVERY seed at the last
        pass AND at the pass of the lowest training loss (no chosen pass); -> <dir>/<name>.json
    python -m net.drill.probe fit --dir tmp/engine/b/probe [--passes 20 --lr 1e-3 --mb 256 --seeds 5,6,7 --name probe
                                                            --device cuda]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from net.drill import record as RC
from net.drill import store as ST
from net.drill.cards import o0_1 as CARD
from net.drill.levels import operative as OPL
from net.operative import gframe as GF
from net.operative import heads as HD
from net.operative import lines as LN
from net.tactic import rows as RW

FOLLOWED, FALSE, N_MIN, BAND = 0.90, 0.03, 50, 0.10
_WHY = "design_merged 5 'Probe': intact followed >= 0.90, false <= 0.03 on >= 50 pairs; ablated balanced accuracy 0.5 +- 0.10"
MOTION_ROW = tuple(69 + RW.LIVE.index(k) for k in ("v_right", "v_forward", "mv", "f"))   # (f: the running flag)
MOTION_G = tuple(GF.COLS.index(k) for k in ("g_v_right", "g_v_fwd", "g_close"))
BRACED_ROW = (RW.RI["braced_on"], RW.RI["braced_left"])
_BRACED_WHY = ("the cause 'the enemy holds' has a second carrier: the enemy's 'braced' effect (a standing unit braces) - "
               "on the held pairs' E rows 1.0 on every cause decision, 0.0 on every come one (tmp/engine/b/leak_cols); "
               "the per-candidate readout (heads.PICK_WHY) separated the pairs on it with the motion zeroed (probe_pick, "
               "ablated BA 1.0). --ablate-braced 1 zeroes it with the motion: the ablated arm then asks 'is anything "
               "besides the hold itself telling'")
ABLATE_BRACED = True
ABLATE_GEOM = True        # (the orchestrator 22.11 F5: the geometry Astra's o1 review 2 found besides the cause - the
#                           stand-off line's distances (order 'zone to LS', both candidates' distances) and the enemy
#                           lord's place (its row's position, place and gframe position columns) - erased too)
ORDER_GEOM = (5,)                                  # (heads.order_feats: the zone in force to the stand-off line)
CAND_GEOM = (2, 3)                                 # (heads.cand_feats: a candidate's distance to C and to the zone)
ROW_POS = tuple(69 + RW.LIVE.index(k) for k in ("right", "forward"))
G_POS = tuple(GF.COLS.index(k) for k in ("g_right", "g_fwd", "g_near"))      # (the gate's ablated arm erases the motion AND braced - the orchestrator 22.11)


def ablate(obs):
    o = dict(obs)
    r = o["rows"].clone()
    r[..., list(MOTION_ROW) + (list(BRACED_ROW) if ABLATE_BRACED else [])] = 0.0
    g = o["gframe"].clone()
    g[..., list(MOTION_G)] = 0.0
    if ABLATE_GEOM:
        from net.tactic import side as SD_
        lord = (r[..., SD_.LORD_CARD] > 0.5) & ((r[..., RW.RI["enemy_target"]] > 0) | (r[..., RW.RI["enemy_other"]] > 0))
        lm = lord[..., None]
        r[..., list(ROW_POS)] = torch.where(lm, torch.zeros_like(r[..., list(ROW_POS)]), r[..., list(ROW_POS)])
        g[..., list(G_POS)] = torch.where(lm, torch.zeros_like(g[..., list(G_POS)]), g[..., list(G_POS)])
        pl = o["place"].clone()
        pl = torch.where(lm, torch.zeros_like(pl), pl)
        od = o["order"].clone()
        od[..., list(ORDER_GEOM)] = 0.0
        cd = o["cand"].clone()
        cd[..., list(CAND_GEOM)] = 0.0
        o["place"], o["order"], o["cand"] = pl, od, cd
    o["rows"], o["gframe"] = r, g
    return o


def collect(a):
    D = Path(a.out)
    specs = CARD.scenes(a.set, a.n)
    seed = CARD.SETS[a.set]
    if a.set == "probe_train":
        mk, delays = CARD.make_disturbed(specs, seed)
    else:
        mk, delays = None, {}
    rec = OPL.Recorder(keep_raw=False, keep_delegated=False)
    seeds = [seed + k for k in range((len(specs) + a.chunk - 1) // a.chunk)]
    t0 = time.time()
    OPL.LEVEL.play("TEACH", specs, seeds, a.chunk, a.device, make_player=mk, recorder=rec,
                   compile=torch.device(a.device).type == "cuda")
    rows = rec.rows()
    kinds = [s["o1"]["twin"] or "cause" for s in specs]
    meta = [{"seed": s["seed"], "kind": k, "delay": delays.get(i, 0), "i": i % a.n} for i, (s, k) in
            enumerate(zip(specs, kinds))]
    st = ST.Store(D / a.set)
    st.put(a.set, rows, meta, cards=None, stats=OPL.census(rows))
    ST.atomic_text(json.dumps({"gate_key": OPL.gate_key(CARD.card())}), D / a.set / "gate_key.json")   # (the scenes' key)
    print(json.dumps({"set": a.set, "battles": len(specs), "rows": int(len(rows["b"])), "census": OPL.census(rows),
                      "min": round((time.time() - t0) / 60, 1)}), flush=True)


def first_pairs(rows, meta):
    """[(cause row, come row)] of complete disjoint pairs (module doc) and the counts left out."""
    t0 = int(LN.conf()["start_s"])
    first = {}
    for i in range(len(rows["b"])):
        if int(rows["sec"][i]) == t0:
            first[int(rows["b"][i])] = i
    by = {}
    for b, m in enumerate(meta):
        if m["kind"] in ("cause", "come"):
            by.setdefault(m["seed"], {})[m["kind"]] = b
    pairs, out = [], {"no_twin": 0, "no_first_row": 0, "labels": 0}
    for sd, v in by.items():
        if "cause" not in v or "come" not in v:
            out["no_twin"] += 1
            continue
        if v["cause"] not in first or v["come"] not in first:
            out["no_first_row"] += 1
            continue
        c, m = first[v["cause"]], first[v["come"]]
        if int(rows["teacher"][c]) != OPL.LS or int(rows["teacher"][m]) != OPL.KEEP:
            out["labels"] += 1
            continue
        pairs.append((c, m))
    return pairs, out


def _pred(net, rows, idx, abl, chunk=512):
    """Greedy choices on rows idx, in chunks (a 6,000-row forward at once took the 6 GB container: FIT 137)."""
    out = []
    idx = torch.as_tensor(idx, dtype=torch.long)
    for i in range(0, idx.numel(), chunk):
        j = idx[i:i + chunk]
        b = OPL.to_batch(rows, j)
        obs = {**b.obs, "legal": rows["obs_legal"][j]}
        if abl:
            obs = ablate(obs)
        dev = next(net.parameters()).device
        with torch.no_grad():
            out.append(net.eval()({k: v.to(dev) for k, v in obs.items()}).argmax(1).cpu())
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def share(k, n):
    return {"p": round(k / n, 4) if n else None, "k": int(k), "n": int(n)}


def read(net, tr, held, hmeta, pairs, abl):
    cz = [c for c, _ in pairs]
    mz = [m for _, m in pairs]
    pc, pm = _pred(net, held, cz, abl), _pred(net, held, mz, abl)
    tpr, fpr = float((pc == 1).float().mean()), float((pm == 1).float().mean())
    el = OPL.eligible(held)
    ch = el.nonzero()[:, 0]
    ph = _pred(net, held, ch.tolist(), abl)
    te = held["teacher"][ch]
    elt = OPL.eligible(tr).nonzero()[:, 0]
    pt = _pred(net, tr, elt.tolist(), abl)
    tt = tr["teacher"][elt]
    return {"pairs_followed": share(int((pc == 1).sum()), len(cz)), "pairs_false": share(int((pm == 1).sum()), len(mz)),
            "pairs_balanced_accuracy": round((tpr + 1 - fpr) / 2, 4),
            "held_change_ticks_followed": share(int(((ph == 1) & (te == 1)).sum()), int((te == 1).sum())),
            "held_keep_ticks_false": share(int(((ph == 1) & (te == 0)).sum()), int((te == 0).sum())),
            "train_change_ticks_followed": share(int(((pt == 1) & (tt == 1)).sum()), int((tt == 1).sum())),
            "train_keep_ticks_false": share(int(((pt == 1) & (tt == 0)).sum()), int((tt == 0).sum()))}


def passes(net, rows, w, n, lr, mb, seed, obs_fn=None, after_pass=None, log=None, max_grad=1.0, device="cpu"):
    """The probe's own small loop: n passes over the rows with weight > 0 in minibatches of mb (one permutation a
    pass, generator seeded seed), Adam(lr) on the trainable parameters, record.loss per minibatch; the pass loss
    weighted by each minibatch's weight sum. -> [pass loss]."""
    use = (w[:, 0] > 0).nonzero()[:, 0]
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(int(seed))
    hist = []
    for p in range(n):
        net.train()
        perm = use[torch.randperm(use.numel(), generator=gen)]
        tot, ws = 0.0, 0.0
        for i in range(0, perm.numel(), mb):
            idx = perm[i:i + mb]
            b = OPL.to_batch(rows, idx)
            if obs_fn is not None:
                b.obs = obs_fn(b.obs)
            b = b.to(device)
            wi = w[idx].to(device)
            loss, _ = RC.loss(OPL.LEVEL, net, b, wi)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(max_grad))
            opt.step()
            tot += float(loss.detach()) * float(wi.sum())
            ws += float(wi.sum())
        hist.append(round(tot / max(ws, 1e-9), 5))
        net.eval()
        row = {"pass": p, "loss": hist[-1]}
        if after_pass is not None:
            row.update(after_pass(p, net))
        if log is not None:
            log(row)
    return hist


def fit_arm(tr, held, hmeta, pairs, abl, a, log, seed):
    torch.manual_seed(seed)
    net = HD.OpNet1(pick=a.pick).to(a.device)
    w = OPL.balanced(tr)
    curve = []

    def after(p, net):
        r = {"arm": "ablated" if abl else "intact", "seed": seed, "keep_bias": round(float(net.keep_bias.detach()), 3),
             **read(net, tr, held, hmeta, pairs, abl)}
        curve.append(r)
        return r
    hist = passes(net, tr, w, a.passes, a.lr, a.mb, seed + 1, obs_fn=ablate if abl else None, after_pass=after, log=log,
                  device=a.device)
    for c, h in zip(curve, hist):
        c["loss"] = h
    return curve


def _gate(r, abl):
    f, x = r["pairs_followed"], r["pairs_false"]
    ok_n = f["n"] >= N_MIN and x["n"] >= N_MIN
    if abl:
        return ok_n and abs(r["pairs_balanced_accuracy"] - 0.5) <= BAND
    return ok_n and f["p"] >= FOLLOWED and x["p"] <= FALSE


def verdict(runs):
    """runs {seed: {"intact": curve, "ablated": curve}}: each arm's gate on every seed at the last pass and at the
    lowest-loss pass."""
    out, ok = {}, True
    for sd, r in runs.items():
        v = {}
        for arm in ("intact", "ablated"):
            c = r[arm]
            best = min(range(len(c)), key=lambda k: c[k]["loss"])
            for tag, k in (("last", len(c) - 1), ("best_loss", best)):
                g = _gate(c[k], arm == "ablated")
                v[f"{arm}_{tag}"] = {"pass_index": k, "loss": c[k]["loss"], "followed": c[k]["pairs_followed"],
                                     "false": c[k]["pairs_false"], "balanced_accuracy": c[k]["pairs_balanced_accuracy"],
                                     "pass": g}
                ok = ok and g
        out[str(sd)] = v
    n_ok = all(c["pairs_followed"]["n"] >= N_MIN and c["pairs_false"]["n"] >= N_MIN
               for r in runs.values() for arm in ("intact", "ablated") for c in r[arm][-1:])
    return {"seeds": out, "n_ok": n_ok, "pass": ok and n_ok, "lines_why": _WHY + "; on every seed at the last and the lowest-loss pass"}


def fit(a):
    D = Path(a.dir)
    tr, _, tmeta = ST.Store(D / "probe_train").mixture()
    held, _, hmeta = ST.Store(D / a.held).mixture()
    pairs, left = first_pairs(held, hmeta)
    say = lambda r: print(json.dumps(r), flush=True)                 # noqa: E731
    say({"train": OPL.census(tr), "held": OPL.census(held), "pairs": len(pairs), "left_out": left,
         "ablate_row_cols": list(MOTION_ROW), "ablate_gframe_cols": list(MOTION_G)})
    seeds = [int(x) for x in str(a.seeds).split(",")]
    global ABLATE_BRACED
    ABLATE_BRACED = bool(a.ablate_braced)
    global ABLATE_GEOM
    ABLATE_GEOM = bool(a.ablate_geom)
    out = {"pairs": len(pairs), "left_out": left, "passes": a.passes, "lr": a.lr, "mb": a.mb, "seeds": seeds,
           "pick": a.pick, "ablate_braced": bool(a.ablate_braced),
           "ablate_geom": bool(a.ablate_geom), "held": a.held, "runs": {}}
    if a.pick and a.lr and a.mb:     # (the gate key of the card when the probe ran the card's config)
        from net.drill.cards import o0_1 as CARD_
        c_ = CARD_.card()
        la = dict(c_.level_args)
        want = OPL.gate_key(c_)
        kf = [D / x / "gate_key.json" for x in ("probe_train", a.held)]     # (both stores collected on these scenes)
        same = all(f.exists() and json.loads(f.read_text(encoding="utf-8")).get("gate_key") == want for f in kf)
        if (bool(la.get("pick")) == bool(a.pick) and float(la["lr"]) == a.lr and int(la["minibatch"]) == a.mb
                and a.ablate_braced and a.ablate_geom and same):
            out["gate_key"] = want
        else:
            say({"gate_key": None, "why": "not the card's recipe (pick / lr / mb / ablate flags) or stores of other "
                                          "scenes: this probe cannot open the training"})
    stopped = None
    for arm in ("intact", "ablated"):        # (every seed's intact arm first: one failing seed fails the recipe)
        for sd in seeds:
            c = fit_arm(tr, held, hmeta, pairs, arm == "ablated", a, say, sd)
            out["runs"].setdefault(sd, {})[arm] = c
            ST.atomic_text(json.dumps(out, indent=1), D / f"{a.name}.partial.json")
            best = min(range(len(c)), key=lambda k: c[k]["loss"])
            if a.stop_on_fail and not (_gate(c[-1], arm == "ablated") and _gate(c[best], arm == "ablated")):
                stopped = f"{arm} seed {sd} failed: the rest not run (--stop-on-fail)"
                break
        if stopped:
            break
    if stopped:
        say({"stopped": stopped})
        out["verdict"] = {"pass": False, "n_ok": True, "stopped": stopped, "lines_why": _WHY}
        (D / f"{a.name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        raise SystemExit(1)
    out["verdict"] = verdict(out["runs"])
    say({"verdict": out["verdict"]})
    (D / f"{a.name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if not out["verdict"]["pass"]:              # (Astra's second o1 review 4: a failed probe exits non-zero)
        raise SystemExit(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("collect", "fit"))
    ap.add_argument("--set", default="probe_train")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", default="tmp/engine/b/probe")
    ap.add_argument("--dir", default="tmp/engine/b/probe")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--seeds", default="5,6,7")
    ap.add_argument("--name", default="probe")
    ap.add_argument("--pick", type=int, default=None, help="OpNet1's state readout per candidate (heads.PICK_WHY)")
    ap.add_argument("--stop-on-fail", type=int, default=1)
    ap.add_argument("--ablate-braced", type=int, default=1, help="_BRACED_WHY")
    ap.add_argument("--ablate-geom", type=int, default=1, help="ABLATE_GEOM")
    ap.add_argument("--held", default="probe_held", help="the held set's store folder under --dir")
    a = ap.parse_args(argv)
    return collect(a) if a.cmd == "collect" else fit(a)


if __name__ == "__main__":
    main()
