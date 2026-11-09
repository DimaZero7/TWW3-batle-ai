"""The strategist's short training (plan.md "Обучение"). One decision a battle - a bandit:

    1. imitation without battles: a pool of generated cases (the input and the formula's choices as labels), the
       cross-entropies of net.choose(labels=...) - the 'pre-training'; minutes
    2. PPO of one decision: a batch of battles (the network samples), the reward at the end = the case's measure +
       the trade in gold (config reward), the advantage = the reward - the critic; the teacher's term with a weight
       falling to zero; under the strategist the operative (config operative: its formula, or its network), the tactician's formula, the frozen unit network

Checkpoints <out>/latest.pt (kind strategist_v1), the log <out>/log.jsonl, the exam at the end <out>/exam.json.

    DOCK_NAME=agent-strategist bash ops/dock.sh net.strategist.train --device cuda --minutes 30 --out tmp/old-build/diag_strategist/run1
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sim.params import load
from net.strategist import battle as BT
from net.strategist import cases as CS
from net.strategist import cfg
from net.strategist import exam as EX
from net.strategist import net as NN
from net.strategist import teacher as TC
from net.strategist import zones as ZS
from net.tactic import train as TT
from net.common import ppo
from net.strategist.ckpt import KIND, restore, save  # noqa: F401 (re-exported)


def take(inp, ctx, idx):
    i2 = {k: v[idx] for k, v in inp.items()}
    return i2, NN.Ctx(ctx.axes.take(idx), ZS.take(ctx.mine, idx), ZS.take(ctx.enemy, idx))


def take_act(act, idx):
    return {k: v[idx] for k, v in act.items()}


def imitate(net, opt, P, dev, steps, batch, pool, gen, log):
    """Stage 1 (module doc). Returns the last agreement."""
    sp = CS.specs(pool, gen)
    pre = CS.Pre(sp, P, dev)
    inp, ctx = BT.inputs_of(pre)
    with torch.no_grad():
        lab = TC.labels_of(inp, ctx)
    params = [p for p in net.parameters() if p.requires_grad]
    agree = {}
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(pool, (batch,), device=dev)
        i2, c2 = take(inp, ctx, idx)
        term, st = TC.loss(net, i2, c2, take_act(lab, idx))
        opt.zero_grad()
        term.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if s % 50 == 0 or s == steps - 1:
            agree = {h: (round(v, 3) if v is not None else None) for h, v in st.items()}
            row = {"imitate": s, "term": round(float(term.detach()), 4), "agree": agree, "s": round(time.time() - t0, 1)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
    return agree


def update(net, critic, opt, inp, ctx, act, lp_old, ret, tc, teach_weight):
    with torch.no_grad():
        v_old = critic(inp)
    adv = ret - v_old
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    B = ret.shape[0]
    pa = [p for p in net.parameters() if p.requires_grad]
    pc = [p for p in critic.parameters() if p.requires_grad]
    out = {"pi": [], "v": [], "clip": [], "kl": [], "teach": [], "ent": [], "agree": {h: [] for h in NN.HEADS}}
    stop = False
    for _ in range(int(tc["epochs"])):
        if stop:
            break
        perm = torch.randperm(B, device=ret.device)
        for i in range(0, B, int(tc["minibatch"])):
            j = perm[i:i + int(tc["minibatch"])]
            i2, c2 = take(inp, ctx, j)
            res = net(i2, c2, act=take_act(act, j))
            lp = res["lp"]
            pl, clip = ppo.policy_loss(lp[:, None], lp_old[j][:, None], adv[j], torch.ones_like(lp[:, None]),
                                       float(tc["clip"]))
            ent = sum(res["ent"].values()).mean()
            vl = F.mse_loss(critic(i2), ret[j])
            loss = pl + float(tc["vf"]) * vl - float(tc["entropy"]) * ent
            term, st = TC.loss(net, i2, c2)
            if teach_weight > 0:
                loss = loss + teach_weight * term
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pa, float(tc["max_grad"]))
            torch.nn.utils.clip_grad_norm_(pc, float(tc["max_grad"]))
            opt.step()
            out["pi"].append(float(pl.detach()))
            out["v"].append(float(vl.detach()))
            out["clip"].append(float(clip))
            out["kl"].append(float((lp_old[j] - lp.detach()).mean()))
            out["teach"].append(float(term.detach()))
            out["ent"].append(float(ent.detach()))
            for h, v in st.items():
                if v is not None:
                    out["agree"][h].append(v)
            if out["kl"][-1] > float(tc["target_kl"]):
                stop = True
                break
    mean = lambda v: round(float(np.mean(v)), 4) if v else None                     # noqa: E731
    return {"policy_loss": mean(out["pi"]), "value_loss": mean(out["v"]), "clip": mean(out["clip"]),
            "kl": mean(out["kl"]), "teach_loss": mean(out["teach"]), "teach_weight": round(teach_weight, 3),
            "ent": mean(out["ent"]), "steps": len(out["pi"]), "agree": {h: mean(v) for h, v in out["agree"].items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc = cfg()["train"]
    ap.add_argument("--battles", type=int, default=int(tc["battles"]))
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--imitate", type=int, default=int(tc["imitate_steps"]))
    ap.add_argument("--init", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="tmp/net/strategist/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-exam", action="store_true")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--op", default=None, help="the operative under: net (config operative_ckpt) or formula (default: config operative)")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    P = load()
    dev = a.device
    net, critic = NN.StrategistNet().to(dev), NN.StrategistCritic().to(dev)
    done, imitated = 0, 0
    if a.init:
        net, critic, ck = restore(a.init, dev)
        done, imitated = int(ck.get("updates", 0)), int(ck.get("imitated", 0))
    params = [p for p in net.parameters() if p.requires_grad] + [p for p in critic.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(tc["lr"]))
    unit_net = TT.unit_network(dev)
    op_net = EX.operative_net(dev) if (a.op or cfg()["operative"]) == "net" else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed + 1000 * done + 31)
    print(json.dumps({"weights": NN.report(net), "critic": sum(p.numel() for p in critic.parameters()),
                      "battles": a.battles, "device": dev}), flush=True)
    start = time.time()
    first = done                       # the teacher's weight falls from this run's first update (a run from --init too)
    with TT.lock(dev, f"strategist {out.name}") if not a.no_lock else contextlib.nullcontext():
        with (out / "log.jsonl").open("a", encoding="utf-8") as logf:
            if a.imitate > 0:
                imitate(net, opt, P, dev, a.imitate, int(tc["imitate_batch"]), int(tc["imitate_pool"]), gen, logf)
                imitated += a.imitate
                save(out / "latest.pt", net, critic, opt, done, imitated)
                save(out / "imitated.pt", net, critic, opt, done, imitated)
            for _ in range(a.updates):
                if a.minutes > 0 and time.time() - start > a.minutes * 60:
                    break
                t0 = time.time()
                pre = CS.Pre(CS.specs(a.battles, gen, cases=tuple(tc["cases"])), P, dev)
                inp, ctx = BT.inputs_of(pre)
                with torch.no_grad():
                    o = net(inp, ctx)
                rd, rs = BT.play(pre, o, inp, unit_net, op_net=op_net)
                m = BT.measures(pre, rd)
                meas = BT.measure_of(pre, rd)
                rw = cfg()["reward"]
                ret = float(rw["measure"]) * meas + float(rw["trade"]) * rd["trade"]
                t1 = time.time()
                us = update(net, critic, opt, inp, ctx, o["act"], o["lp"].detach(), ret, tc, TC.weight_at(done - first))
                t2 = time.time()
                done += 1
                by = {cs: round(float(m[cs][pre.case == i].mean()), 3) for i, cs in enumerate(CS.CASES)
                      if bool((pre.case == i).any())}
                rs.pop("errors", None)
                row = {"update": done, "measure": by, "reward": round(float(ret.mean()), 4), **rs, **us,
                       "groups": round(float(o["plan"]["on"].sum(1).float().mean()), 3),
                       "rollout_s": round(t1 - t0, 1), "update_s": round(t2 - t1, 1),
                       "minutes": round((t2 - start) / 60, 2)}
                print(json.dumps(row), flush=True)
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                save(out / "latest.pt", net, critic, opt, done, imitated)
    if not a.no_exam:
        EX.run(unit_net, P, dev, net=net, out_path=out / "exam.json", who="network", op_net=op_net)
    print(json.dumps({"updates": done, "minutes": round((time.time() - start) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
