"""The operative's short training on its three cases: PPO over the operative's decisions (one a 15 s; tools/nn/train/
ppo.py gae, policy_loss, explained_variance), the tacticians (the formula) and the frozen unit network carry it out
(env.py), the teacher's term on a small share of the decisions with a weight falling to zero (teacher.py; the battles are
the network's own). Checkpoints: <out>/latest.pt (kind operative_v1), the log <out>/log.jsonl. The GPU lock as the
chain's (test5.gpu_lock) on the CUDA device.

    DOCK_NAME=agent-operative bash tools/nn/dock.sh tools.nn.operative.train --device cuda --minutes 30 --out build/diag_operative/run1
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.nn.operative import cfg
from tools.nn.operative import cases as CS
from tools.nn.operative import env as EN
from tools.nn.operative import exam as EX
from tools.nn.operative import net as NN
from tools.nn.operative import teacher as TC
from tools.nn.sim.params import load
from tools.nn.tactic import train as TT
from tools.nn.train import ppo
from tools.nn.unit import train as UT

KIND = "operative_v1"


def params_of(net, critic):
    return [p for p in net.parameters() if p.requires_grad], [p for p in critic.parameters() if p.requires_grad]


def update(net, critic, opt, batch, tc, device, duty, teach_weight):
    adv, ret = ppo.gae(batch["reward"], batch["value"], batch["done"], batch["last_value"], tc["gamma"], tc["lam"])
    d_idx, b_idx = batch["valid"].nonzero(as_tuple=True)
    M = d_idx.shape[0]
    a_all = adv[d_idx, b_idx]
    a_all = (a_all - a_all.mean()) / (a_all.std() + 1e-8)
    r_all = ret[d_idx, b_idx]
    pa, pc = params_of(net, critic)
    share = float(cfg()["teacher"]["share"])
    first_w = float(cfg()["teacher"]["first_weight"])
    out = {"pi": [], "v": [], "clip": [], "kl": [], "ent": {h: [] for h in NN.HEADS}, "teach": [], "preds": [],
           "rets": [], "agree": {h: [] for h in NN.HEADS}}
    stop = False
    for _ in range(int(tc["epochs"])):
        perm = torch.randperm(M, device=device)
        for i in range(0, M, int(tc["minibatch"])):
            if duty is not None:
                duty.start()
            j = perm[i:i + int(tc["minibatch"])]
            di, bi = d_idx[j], b_idx[j]
            inp, ctx, act = EN.rebuild(batch, di, bi)
            res, _ = net(inp, ctx, act=act)
            lp = res["lp"]
            pl, clip = ppo.policy_loss(lp[:, None], batch["lp"][di, bi][:, None], a_all[j], torch.ones_like(lp[:, None]),
                                       float(tc["clip"]))
            ent = sum(res["ent"].values())
            value = critic(inp)
            vl = F.mse_loss(value, r_all[j])
            loss = pl + float(tc["vf"]) * vl - float(tc["entropy"]) * ent.mean()
            # the teacher: on a share of the decisions with the falling weight; on every battle's first decision (the
            # groups and the first zones are made there) with first_weight, which does not fall (config teacher)
            first = di == 0
            rest = (torch.rand(j.shape[0], device=device) < share) & ~first
            for sel, w in ((first, first_w), (rest, teach_weight)):
                pick = sel.nonzero()[:, 0]
                if not pick.numel():
                    continue
                inp2, ctx2, _ = EN.rebuild(batch, di[pick], bi[pick])
                term, st = TC.loss(net, inp2, ctx2, batch["destroy"][bi[pick]])
                if w > 0:
                    loss = loss + w * term
                out["teach"].append(float(term.detach()))
                for h, v in st.items():
                    if v is not None:
                        out["agree"][h].append(v)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pa, float(tc["max_grad"]))
            torch.nn.utils.clip_grad_norm_(pc, float(tc["max_grad"]))
            opt.step()
            out["pi"].append(float(pl.detach()))
            out["v"].append(float(vl.detach()))
            out["clip"].append(float(clip))
            kl = float((batch["lp"][di, bi] - lp.detach()).mean())
            out["kl"].append(kl)
            for h in out["ent"]:
                out["ent"][h].append(float(res["ent"][h].mean().detach()))
            out["preds"].append(value.detach())
            out["rets"].append(r_all[j])
            if duty is not None:
                duty.end()
            if kl > float(tc["target_kl"]):
                stop = True
                break
        if stop:
            break
    ev = float(ppo.explained_variance(torch.cat(out["preds"]), torch.cat(out["rets"])))
    mean = lambda v: round(float(np.mean(v)), 4) if v else None                          # noqa: E731
    return {"policy_loss": mean(out["pi"]), "value_loss": mean(out["v"]), "clip": mean(out["clip"]), "kl": mean(out["kl"]),
            "explained_var": round(ev, 3), "samples": M, "steps": len(out["pi"]), "teach_loss": mean(out["teach"]),
            "teach_weight": round(teach_weight, 4), "agree": {h: mean(v) for h, v in out["agree"].items()},
            "ent_update": {h: mean(v) for h, v in out["ent"].items()}}


def save(path, net, critic, opt, cfg_, updates):
    torch.save({"kind": KIND, "net": net.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg_, "updates": updates}, path)


def restore(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {path}")
    net, critic = NN.OperativeNet().to(device), NN.OperativeCritic().to(device)
    ck["padded"] = False
    for mod, key in ((net, "net"), (critic, "critic")):
        own, sd = mod.state_dict(), dict(ck[key])
        for name, p in own.items():
            if name not in sd or sd[name].shape != p.shape:
                # a checkpoint from before a head's change (net.Heads where_a, join, cmd): this layer starts afresh
                sd[name] = p
                ck["padded"] = True
        mod.load_state_dict(sd)
    return net, critic, ck


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc = cfg()["train"]
    ap.add_argument("--battles", type=int, default=int(tc["battles"]))
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default="build/diag_operative/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teach-updates", type=int, default=None)
    ap.add_argument("--exam-battles", type=int, default=None)
    ap.add_argument("--no-exam", action="store_true")
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    P = load()
    dev = a.device
    net, critic = NN.OperativeNet().to(dev), NN.OperativeCritic().to(dev)
    pa, pc = params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=float(tc["lr"]))
    done, teach_from = 0, 0
    if a.init:
        net, critic, ck = restore(a.init, dev)
        pa, pc = params_of(net, critic)
        opt = torch.optim.Adam(pa + pc, lr=float(tc["lr"]))
        if not ck["padded"]:
            opt.load_state_dict(ck["opt"])          # (a padded network starts a fresh Adam: its moments do not fit)
        done = int(ck.get("updates", 0))
    if a.teach_updates is not None:
        teach_from = done
    unit_net = TT.unit_network(dev)
    duty = UT.Duty(dev, a.gpu_duty)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed + 1000 * done + 29)
    print(json.dumps({"weights": NN.report(net), "critic": sum(p.numel() for p in critic.parameters()),
                      "battles": a.battles, "device": dev}), flush=True)
    start = time.time()
    with TT.lock(dev, f"operative {out.name}") if not a.no_lock else contextlib.nullcontext():
        with (out / "log.jsonl").open("a", encoding="utf-8") as logf:
            for _ in range(a.updates):
                if a.minutes > 0 and time.time() - start > a.minutes * 60:
                    break
                t0 = time.time()
                setup = CS.Setup(CS.specs(a.battles, gen, cases=tuple(tc["cases"])), P, dev)
                batch, rs, rd = EN.play(setup, unit_net, net=net, critic=critic, duty=duty)
                t1 = time.time()
                us = update(net, critic, opt, batch, tc, dev, duty, TC.weight_at(done - teach_from, a.teach_updates))
                if torch.device(dev).type == "cuda":
                    torch.cuda.empty_cache()
                t2 = time.time()
                done += 1
                m = EN.measures(setup, rd)
                by = {cs: round(float(m[cs][setup.case == i].mean()), 3) for i, cs in enumerate(CS.CASES)
                      if bool((setup.case == i).any())}
                rs.pop("errors", None)
                row = {"update": done, "measure": by, **rs, **us, "rollout_s": round(t1 - t0, 1),
                       "update_s": round(t2 - t1, 1), "minutes": round((t2 - start) / 60, 2),
                       "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                       if torch.device(dev).type == "cuda" else None}
                print(json.dumps(row), flush=True)
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                save(out / "latest.pt", net, critic, opt, tc, done)
    if not a.no_exam:
        EX.run(unit_net, P, dev, net=net, n=a.exam_battles, out_path=out / "exam.json", who="network")
    print(json.dumps({"updates": done, "minutes": round((time.time() - start) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
