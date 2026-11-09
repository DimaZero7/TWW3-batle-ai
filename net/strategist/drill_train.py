"""Drill S-1 (strategist/drill_s1.py): the exam, the transfer exam and the strategist's training by search.

The training (one decision a battle; no rule tuned by hand): imitation of the formula on a pool of the drill's and the
whole matchups' inputs (no battles), then rounds of the search: for each new seed K candidate deployments - the
formula's cells (no nudge), the formula's walk drawn wider round its wishes (cell_temp), and from round 1 the
network's own (greedy and sampled) - each played in the simulator; the best by its mark (the drill: sniper_kept +
trade at 150 s; a whole matchup, the share of general battles: the trade at 150 s) is the label; the network learns
the labels of all rounds (and the formula's pool with a falling weight).

    bash ops/dock.sh net.strategist.drill_train exam --who teacher,exposed[,<ckpt>] --out <json>
    bash ops/dock.sh net.strategist.drill_train transfer --who teacher,exposed[,<ckpt>] --out <jsonl>
    bash ops/dock.sh net.strategist.drill_train train --minutes 35 --out tmp/old-build/diag_drill_s1/run1
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from sim.params import load
from net.strategist import cfg
from net.strategist import drill_s1 as D
from net.strategist import net as NN
from net.strategist import teacher as TC
from net.strategist import train as ST
from net.tactic import train as TT

EXAM_SEED = 4242          # the drill's exam: held-out seeds (training seeds from 100)
EXAM_N = 96
TRANSFER_SEED = 7         # tmp/old-build/diag_m1/sim/exam2.py: random.Random(7) per matchup
CAND_TEMPS = (1.5, 2.0, 2.0, 3.0, 3.0)


def deployer(who, dev):
    if who in ("teacher", "exposed", "teacher0", "deep", "behind", "deep_zone"):
        return who
    net, _, _ = ST.restore(who, dev)
    net.eval()
    return net


def decide_all(batch, whos, dev, inp=None, ctx=None):
    """The batch cut into len(whos) equal slices, slice k decided by whos[k]."""
    inp = batch.inp if inp is None else inp
    ctx = batch.ctx if ctx is None else ctx
    B = inp["own"].shape[0]
    k = len(whos)
    S = B // k
    outs = []
    for i, w in enumerate(whos):
        idx = torch.arange(i * S, (i + 1) * S, device=dev)
        a, c = D.take(inp, ctx, idx)
        if isinstance(w, tuple):                     # ('sample', temp) | ('net', net, greedy)
            if w[0] == "sample":
                out = D.decide(a, c, "sample", temp=torch.full((S,), float(w[1]), device=dev))
            else:
                with torch.no_grad():
                    out = w[1](a, c, greedy=w[2])
        else:
            out = D.decide(a, c, w)
        outs.append(out)
    return D.cat_out(outs)


def exam(whos, dev, unit_net, P, n=EXAM_N, seed=EXAM_SEED, limit=D.LIMIT_S):
    """{who: summary} on the drill's held-out battles (every deployer the same battles; the teacher first - the
    damage the measure is reckoned against)."""
    sp = D.specs(n, random.Random(seed))
    names = list(whos)
    ds = [deployer(w, dev) for w in names]
    batch = D.Batch(sp * len(ds), P, dev)
    out = decide_all(batch, ds, dev)
    m, st = D.play(batch, out, unit_net, P, dev, limit=limit)
    res = {}
    for i, w in enumerate(names):
        sel = torch.zeros(len(sp) * len(ds), dtype=torch.bool, device=dev)
        sel[i * n:(i + 1) * n] = True
        res[w] = D.summary(m, sel)
        res[w]["by_enemy"] = {e: D.summary(m, sel & torch.tensor([s["enemy"] == e for s in batch.sp], device=dev))
                              ["sniper_kept"] for e in ("ai_game", "nearest", "hold_shoot")}
        fast = torch.tensor([s["n_fast"] > 0 for s in batch.sp], device=dev)
        res[w]["with_fast"] = {k: D.summary(m, sel & fast)[k] for k in ("fire_share", "sniper_end", "sn_first",
                                                                          "fast60_s")}
    res["seconds"] = st.get("wall_s")
    return res, m


def transfer(whos, dev, unit_net, P, n=32, limit=900, names=("m1", "m2", "m3")):
    """{who: {matchup: summary}} of the whole matchups against ai_game (32 x 900 s each), my side deployed by who."""
    sp = []
    for name in names:
        sp += D.full_specs([name], n, random.Random(TRANSFER_SEED))
    ds = [deployer(w, dev) for w in whos]
    batch = D.Batch(sp * len(ds), P, dev, per_side=20)
    out = decide_all(batch, ds, dev)
    m, st = D.play(batch, out, unit_net, P, dev, limit=limit)
    N = len(sp)
    res = {}
    for i, w in enumerate(whos):
        res[w] = {}
        for j, name in enumerate(names):
            sel = torch.zeros(N * len(ds), dtype=torch.bool, device=dev)
            sel[i * N + j * n:i * N + (j + 1) * n] = True
            s = D.summary(m, sel)
            gd = m["lost_enemy"][sel] - m["lost_own"][sel]
            s["gold_diff"] = (round(float(gd.mean()), 3), round(1.96 * float(gd.std()) / n ** 0.5, 3))
            res[w][name] = s
    res["seconds"] = st.get("wall_s")
    return res, m


# ---------------------------------------------------------------- the training
def pool(n_drill, n_full, seed, P, dev, chunk=256):
    """[(inp, ctx, labels)] chunks: the formula's labels (no battles)."""
    rnd = random.Random(seed)
    out = []
    sp = D.specs(n_drill, rnd) + D.full_specs(D.SOURCES, n_full, rnd)
    random.Random(seed + 1).shuffle(sp)
    for i in range(0, len(sp), chunk):
        b = D.Batch(sp[i:i + chunk], P, dev, per_side=20)
        lab = D.decide(b.inp, b.ctx, "teacher0")["act"]
        out.append((b.inp, b.ctx, lab))
    return out


def fit(net, opt, chunks, steps, mb, weights=None, log=None):
    """steps of the cross-entropies against the labels (teacher.loss), a chunk drawn by its weight each step."""
    params = [p for p in net.parameters() if p.requires_grad]
    w = torch.tensor(weights if weights is not None else [1.0] * len(chunks))
    agree, terms = {}, []
    for s in range(steps):
        k = int(torch.multinomial(w, 1))
        inp, ctx, lab = chunks[k]
        B = inp["own"].shape[0]
        idx = torch.randint(B, (min(mb, B),), device=inp["own"].device)
        i2, c2 = D.take(inp, ctx, idx)
        term, st = TC.loss(net, i2, c2, {h: v[idx] for h, v in lab.items()})
        opt.zero_grad()
        term.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if D.DUTY is not None:
            D.DUTY.end()
        terms.append(float(term.detach()))
        for h, v in st.items():
            if v is not None:
                agree.setdefault(h, []).append(v)
        if log and (s % 100 == 0 or s == steps - 1):
            log({"fit": s, "term": round(sum(terms[-20:]) / len(terms[-20:]), 4),
                 "agree": {h: round(sum(v[-20:]) / len(v[-20:]), 3) for h, v in agree.items()}})
    return {"term": round(sum(terms[-50:]) / max(len(terms[-50:]), 1), 4),
            "agree": {h: round(sum(v[-50:]) / len(v[-50:]), 3) for h, v in agree.items()}}


@torch.no_grad()
def agreement(net, chunks):
    """{head: share} of the network's greedy choices that match the labels (teacher.loss's agreement)."""
    was = net.training
    net.eval()
    acc = {}
    for inp, ctx, lab in chunks:
        _, st = TC.loss(net, inp, ctx, lab)
        for h, v in st.items():
            if v is not None:
                acc.setdefault(h, []).append(v)
    net.train(was)
    return {h: round(sum(v) / len(v), 3) for h, v in acc.items()}


def play_chunks(sp, out, enemies, dev, unit_net, P, chunk=512):
    """drill_s1.play of the battles sp (out their decisions, enemies their scripts) in pieces of chunk battles: the
    readings joined."""
    ms = []
    for a in range(0, len(sp), chunk):
        b = min(len(sp), a + chunk)
        batch = D.Batch(sp[a:b], P, dev)
        m, st = D.play(batch, D.take_out(out, torch.arange(a, b, device=dev)), unit_net, P, dev, enemies=enemies[a:b])
        ms.append(m)
    return {k: torch.cat([m[k] for m in ms]) for k in ms[0]}, st


def search(sp, cands, dev, unit_net, P, mark, rnd, R=None, F=2):
    """(batch of sp, best act [S], marks [K, S], the chosen candidate [S], readings [K, S], the finalists' marks [S]):
    every candidate of every seed played once (the seed's own enemy); the F best of each seed played R - 1 times more
    (others of the mix: another script and another draw of the dice), the label the finalist of the best mean of its R
    battles (step 5: 3 battles for all 8 cost 24 a seed, this 8 + F (R - 1) = 12)."""
    S, K = len(sp), len(cands)
    R = int(R or D.DC1["repeats"])
    batch = D.Batch(sp * K, P, dev)
    out = decide_all(batch, cands, dev)
    m, st = play_chunks(sp * K, out, [s_["enemy"] for s_ in sp] * K, dev, unit_net, P)
    score = m[mark].view(K, S)
    top = score.topk(min(F, K), 0).indices                               # [F, S]
    fin = (top * S + torch.arange(S, device=dev)[None]).reshape(-1)      # finalist f of seed s: f*S + s
    reps = []
    for _ in range(R - 1):
        en = [D._enemy_script(rnd) for _ in sp] * top.shape[0]
        m2, _ = play_chunks(sp * top.shape[0], D.take_out(out, fin), en, dev, unit_net, P)
        reps.append(m2[mark])
    mean = (score.gather(0, top) + sum(r.view(top.shape) for r in reps)) / R          # [F, S]
    best = top.gather(0, mean.argmax(0)[None])[0]                        # ties: the better of the first battle
    idx = best * S + torch.arange(S, device=dev)
    lab = D.take_out(out, idx)["act"]
    one = D.Batch(sp, P, dev)
    m["final_mark"] = mean.amax(0)
    return one, lab, score, best, m, st


def where_snipers(batch, act, dev):
    """The chosen deployment's snipers: metres behind the middle of my infantry (+ behind), metres beyond the
    infantry's end across (+ outside), mean per battle - what the search found."""
    from net.strategist import deploy as DP
    from net.strategist import battle as BT
    with torch.no_grad():
        out = NN.choose(NN.null_enc(*batch.inp["own"].shape, int(cfg()["groups"]), dev), batch.inp, batch.ctx,
                        NN.NullHeads(batch.inp["own"].shape[0], dev), act=act)
    plan = out["plan"]
    mem = BT.slots_of(plan, batch.inp, batch.st0.N)
    pl = DP.places(mem, batch.shooter, plan["cx"], plan["cz"], plan["heading"], plan["line"], batch.width0,
                   DP.slot_of(plan))
    from net.strategist import zones as ZS
    s, f = ZS.local(batch.mine_zone, pl["x"], pl["z"])
    rl = D.roles_of_state(batch.st0)
    back, out_ = [], []
    for b in range(batch.st0.B):
        inf = [i for i, r in enumerate(rl[b]) if r in ("line", "anti") and int(batch.st0.u["side"][b, i]) == 1]
        sn = [i for i, r in enumerate(rl[b]) if r == "sniper" and int(batch.st0.u["side"][b, i]) == 1]
        if not inf or not sn:
            continue
        fm = float(f[b, inf].median())
        lo, hi = float(s[b, inf].min()), float(s[b, inf].max())
        for i in sn:
            back.append(fm - float(f[b, i]))
            si = float(s[b, i])
            out_.append(si - hi if si > hi else (lo - si if si < lo else -min(si - lo, hi - si)))
    mean = lambda v: round(sum(v) / max(len(v), 1), 1)          # noqa: E731
    return {"behind_m": mean(back), "outside_m": mean(out_), "n": len(back)}


def train(a):
    P = load()
    dev = a.device
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a", encoding="utf-8")

    def say(row):
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    unit_net = TT.unit_network(dev)
    if a.init:
        net, critic, _ = ST.restore(a.init, dev)
    else:
        net, critic = NN.StrategistNet().to(dev), NN.StrategistCritic().to(dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr)
    with TT.lock(dev, f"strategist drill_s1 {out.name}"):
        _train(a, P, dev, out, say, unit_net, net, critic, opt)
    log.close()


def _train(a, P, dev, out, say, unit_net, net, critic, opt):
    t0 = time.time()
    base = pool(a.pool_drill, a.pool_full, 100, P, dev)
    say({"pool": len(base), "s": round(time.time() - t0, 1)})
    if a.imitate > 0:
        r = fit(net, opt, base, a.imitate, a.mb, log=say)
        say({"imitate": a.imitate, **r, "minutes": round((time.time() - t0) / 60, 2)})
        ST.save(out / "imitated.pt", net, critic, opt, 0, a.imitate)
    # the labels of the earlier steps (their seeds' specs and the chosen act: the inputs rebuilt from the specs)
    labelled, saved = [], []
    for f in [x for x in a.labels.split(",") if x]:
        for sp, act in torch.load(f, weights_only=False):
            act = {k: v.to(dev) for k, v in act.items()}
            b = D.Batch(sp, P, dev)
            labelled.append((b.inp, b.ctx, act))
            saved.append((sp, act))
    say({"labels_loaded": len(labelled)})
    rnd = random.Random(a.seed + 7)
    rd = 0
    t_r = time.time()                      # (--minutes: the search rounds' budget, after the imitation)
    while time.time() - t_r < a.minutes * 60 and rd < a.rounds:
        sp_d = D.specs(a.seeds, rnd)
        sp_f = D.full_specs(D.SOURCES, a.full_seeds, rnd, enemy="mixed")
        for s in sp_f:
            s["enemy"] = D._enemy_script(rnd)
        cands = ["teacher"] + [("sample", t) for t in CAND_TEMPS]
        if rd > 0 or a.init:
            net.eval()
            cands += [("net", net, True), ("net", net, False)]
        row = {"round": rd}
        for kind, sp, mark in (("drill", sp_d, "score3"), ("full", sp_f, "trade")):
            if not sp:
                continue
            t1 = time.time()
            one, lab, score, best, m, st = search(sp, cands, dev, unit_net, P, mark, rnd)
            # the network's choice against the search's on these new seeds, before it learns them
            row.setdefault("agree_new", {})[kind] = agreement(net, [(one.inp, one.ctx, lab)])
            labelled.append((one.inp, one.ctx, lab))
            saved.append((sp, lab))
            K = len(cands)
            S = len(sp)
            V = lambda k: m[k].view(K, S)                                                    # noqa: E731
            final = round(float(m.pop("final_mark").mean()), 3)
            pick = lambda k: round(float(V(k).gather(0, best[None])[0].mean()), 3)           # noqa: E731
            row[kind] = {"seeds": S, "cands": K, "teacher": round(float(score[0].mean()), 3), "final_mean": final,
                         "best": round(float(score.max(0).values.mean()), 3),
                         "net_greedy": round(float(score[K - 2].mean()), 3) if (rd > 0 or a.init) else None,
                         "chosen": torch.bincount(best, minlength=K).tolist(),
                         **{f"{k}_teacher": round(float(V(k)[0].mean()), 3) for k in
                            ("fire_share", "sniper_end", "sn_first", "sniper_shots")},
                         **{f"{k}_best": pick(k) for k in ("fire_share", "sniper_end", "sn_first", "sniper_shots")},
                         "where_best": where_snipers(one, lab, dev), "s": round(time.time() - t1, 1)}
        net.train()
        n_l = len(labelled)
        w_base = a.base_weight * max(0.0, 1.0 - rd / max(a.base_decay, 1))
        ws = [w_base / len(base)] * len(base) + [(1.0 - w_base) / n_l] * n_l
        row["fit"] = fit(net, opt, base + labelled, a.steps, a.mb, ws, log=say)
        row["agree_all_labels"] = agreement(net, labelled)
        torch.save([(sp, {k: v.cpu() for k, v in act.items()}) for sp, act in saved], out / "labels.pt")
        row["minutes"] = round((time.time() - t0) / 60, 2)
        say(row)
        ST.save(out / "latest.pt", net, critic, opt, rd + 1, a.imitate)
        rd += 1
    say({"rounds": rd, "minutes": round((time.time() - t0) / 60, 2)})


def overfit(a):
    """Can the network learn a small set of the search's labels by heart (the drill's seeds of --labels, the first
    --n of them)? The agreement per head every 100 steps; --init the start (none: a fresh network)."""
    P = load()
    dev = a.device
    torch.manual_seed(a.seed)
    if a.init:
        net, critic, _ = ST.restore(a.init, dev)
    else:
        net, critic = NN.StrategistNet().to(dev), NN.StrategistCritic().to(dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr)
    sp, acts = [], []
    for f in [x for x in a.labels.split(",") if x]:
        for spx, act in torch.load(f, weights_only=False):
            if spx[0].get("kind") != "drill":
                continue
            for k, one in enumerate(spx):
                sp.append(one)
                acts.append({h: v[k] for h, v in act.items()})
    n = min(a.n or 64, len(sp))
    b = D.Batch(sp[:n], P, dev)
    lab = {h: torch.stack([x[h] for x in acts[:n]]).to(dev) for h in acts[0]}
    chunk = [(b.inp, b.ctx, lab)]
    rows = []
    for k in range(0, a.steps, 100):
        r = fit(net, opt, chunk, 100, min(a.mb, n))
        row = {"steps": k + 100, "term": r["term"], "agree": agreement(net, chunk), "seeds": n}
        print(json.dumps(row), flush=True)
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("exam", "transfer", "train", "overfit"))
    ap.add_argument("--who", default="teacher,exposed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="tmp/net/drill_s1/exam.json")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--matchups", default="m1,m2,m3")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--full-seeds", type=int, default=12)
    ap.add_argument("--pool-drill", type=int, default=1536)
    ap.add_argument("--pool-full", type=int, default=512)
    ap.add_argument("--imitate", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--mb", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base-weight", type=float, default=0.3)
    ap.add_argument("--base-decay", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="a strategist checkpoint to start from")
    ap.add_argument("--labels", default="", help="comma-separated labels.pt of earlier steps")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--gpu-duty", type=float, default=1.0, help="the GPU's share of the time (the game mode: 0.45)")
    a = ap.parse_args(argv)
    if a.gpu_duty < 1.0:
        D.DUTY = D.GpuShare(a.device, a.gpu_duty)
    if a.what == "train":
        return train(a)
    if a.what == "overfit":
        return overfit(a)
    P = load()
    unit_net = TT.unit_network(a.device)
    whos = a.who.split(",")
    t0 = time.time()
    if a.what == "exam":
        res, _ = exam(whos, a.device, unit_net, P, n=a.n or EXAM_N, limit=a.limit or D.LIMIT_S)
    else:
        res, _ = transfer(whos, a.device, unit_net, P, n=a.n or 32, limit=a.limit or 900,
                          names=tuple(a.matchups.split(",")))
    res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"what": a.what, "who": whos, **res}) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
