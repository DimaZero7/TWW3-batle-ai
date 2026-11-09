"""The tactician's training on its first drill: PPO over the tactician's decisions (one a 5 s; net/common/ppo.py gae,
policy_loss, explained_variance), the frozen unit network carries the frames out (env.py), a teacher's term of a small
share of the decisions with a weight falling to zero (teacher.py; the battles are the network's own: PPO does not learn
from someone else's actions). Checkpoints: <out>/latest.pt (kind tactic_v1), the log <out>/log.jsonl. The GPU lock as the
chain's (test5.gpu_lock) on the CUDA device.

    DOCK_NAME=agent-tactic bash ops/dock.sh net.tactic.train --device cuda --minutes 30 --out tmp/net/tactic/run
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
from net.tactic import cfg
from net.tactic import decide as DC
from net.tactic import drill1 as DR
from net.tactic import env as EN
from net.tactic import exam1 as EX
from net.tactic import teacher as TC
from net.common import ppo
from net.unit import train as UT
from net.tactic.ckpt import KIND, restore, save, unit_network  # noqa: F401 (re-exported)


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
    share = float(cfg()["teach"]["share"])
    out = {"pi": [], "v": [], "clip": [], "kl": [], "ent": {k: [] for k in ("keep", "width", "depth", "where")}, "teach": [],
           "keep_ok": [], "keep_said": [], "size_ok": [], "cell_ok": [], "preds": [], "rets": []}
    stop = False
    for _ in range(int(tc["epochs"])):
        perm = torch.randperm(M, device=device)
        for i in range(0, M, int(tc["minibatch"])):
            if duty is not None:
                duty.start()
            j = perm[i:i + int(tc["minibatch"])]
            di, bi = d_idx[j], b_idx[j]
            inp, order, parcel, tiles, act, prev = EN.rebuild(batch, di, bi)
            res, enc, ctx = net(inp, order, parcel, tiles, prev=prev, act=act)
            lp = res["lp"]
            pl, clip = ppo.policy_loss(lp[:, None], batch["lp"][di, bi][:, None], a_all[j], torch.ones_like(lp[:, None]),
                                       float(tc["clip"]))
            ent = sum(res["ent"].values())
            value = critic(inp, order)
            vl = F.mse_loss(value, r_all[j])
            loss = pl + float(tc["vf"]) * vl - float(tc["entropy"]) * ent.mean()
            # the teacher's term on a small share of the decisions (reported even once its weight is 0)
            pick = torch.rand(j.shape[0], device=device) < share
            if bool(pick.any()):
                from net.tactic import zone as ZN
                sub = {k: v[pick] for k, v in enc.items()}
                t_sub = ZN.Tiles(tiles.cx[pick], tiles.cz[pick], tiles.heading[pick], tiles.rows[pick], tiles.cols[pick],
                                 tiles.mask[pick])
                term, st = TC.loss(sub, ctx.take(pick, t_sub), net.heads)
                if teach_weight > 0:
                    loss = loss + teach_weight * term
                out["teach"].append(float(term.detach()))
                out["keep_ok"].append(st["keep_ok"])
                out["keep_said"].append(st["keep_said"])
                out["size_ok"].append(st["size_ok"])
                out["cell_ok"].append(st["cell_ok"])
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
            for k in out["ent"]:
                out["ent"][k].append(float(res["ent"][k].mean().detach()))
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
            "teach_keep_ok": mean(out["keep_ok"]), "teach_keep_said": mean(out["keep_said"]),
            "teach_size_ok": mean(out["size_ok"]), "teach_cell_ok": mean(out["cell_ok"]), "teach_weight": round(teach_weight, 4),
            "ent_update": {k: mean(v) for k, v in out["ent"].items()}}


def lock(device, label):
    if torch.device(device).type != "cuda":
        return contextlib.nullcontext()
    from net.common.gpu import gpu_lock
    return gpu_lock(label)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tc = cfg()["train"]
    ap.add_argument("--battles", type=int, default=int(tc["battles"]))
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--init", default=None, help="go on from this checkpoint: weights, Adam, counter")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-duty", type=float, default=1.0)
    ap.add_argument("--out", default="tmp/net/tactic/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exam-every", type=int, default=0, help="the exam every this many updates (0: at the end only)")
    ap.add_argument("--exam-battles", type=int, default=None)
    ap.add_argument("--teach-updates", type=int, default=None, help="the teacher's weight falls to zero over this many "
                    "updates from the start of this run (default: config teach.decay_updates, counted from update 0; a "
                    "checkpoint from before a new head: from the start of this run)")
    ap.add_argument("--no-lock", action="store_true", help="no GPU lock (a short agent's check beside nothing else)")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    P = load()
    dev = a.device
    net, critic = DC.TacticNet().to(dev), DC.TacticCritic().to(dev)
    pa, pc = params_of(net, critic)
    opt = torch.optim.Adam(pa + pc, lr=float(tc["lr"]))
    done = 0
    teach_from = 0
    if a.init:
        net, critic, ck = restore(a.init, dev)
        pa, pc = params_of(net, critic)
        opt = torch.optim.Adam(pa + pc, lr=float(tc["lr"]))
        if not ck["padded"]:
            opt.load_state_dict(ck["opt"])          # (a padded network starts a fresh Adam: its moments do not fit)
        done = int(ck.get("updates", 0))
        if ck["padded"]:
            teach_from = done                       # (a new head: the teacher's decay starts afresh)
    if a.teach_updates is not None:
        teach_from = done
    unit_net = unit_network(dev)
    duty = UT.Duty(dev, a.gpu_duty)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed + 1000 * done + 17)
    print(json.dumps({"weights": DC.report(net), "critic": sum(p.numel() for p in critic.parameters()),
                      "battles": a.battles, "device": dev}), flush=True)
    start = time.time()
    with lock(dev, f"tactic {out.name}") if not a.no_lock else contextlib.nullcontext():
        with (out / "log.jsonl").open("a", encoding="utf-8") as logf:
            for k in range(a.updates):
                if a.minutes > 0 and time.time() - start > a.minutes * 60:
                    break
                t0 = time.time()
                setup = EN.Setup(DR.specs(a.battles, gen), P, dev)
                batch, rs, rd = EN.play(setup, unit_net, net=net, critic=critic, duty=duty)
                t1 = time.time()
                us = update(net, critic, opt, batch, tc, dev, duty, TC.weight_at(done - teach_from, a.teach_updates))
                if torch.device(dev).type == "cuda":
                    torch.cuda.empty_cache()           # (the update's cached blocks back for the battles)
                t2 = time.time()
                done += 1
                sh = {s: round(float(rd["between"][[x["start"] == s for x in setup.sp]].float().mean()), 3) for s in DR.STARTS}
                row = {"update": done, "between": round(float(rd["between"].float().mean()), 3), "between_by": sh,
                       "contact": round(float(rd["contact"].float().mean()), 3), "hit": round(float(rd["hit"].float().mean()), 3),
                       "melee": round(float(rd["melee"].float().mean()), 3), **rs, **us,
                       "rollout_s": round(t1 - t0, 1), "update_s": round(t2 - t1, 1),
                       "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.device(dev).type == "cuda" else None,
                       "decisions_s": round(rs["decisions"] / max(t2 - t0, 1e-9), 1), "minutes": round((t2 - start) / 60, 2)}
                print(json.dumps(row), flush=True)
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                save(out / "latest.pt", net, critic, opt, tc, done)
                if a.exam_every and done % a.exam_every == 0:
                    EX.run(unit_net, P, dev, net, n=a.exam_battles, out_path=out / f"exam_u{done}.json")
    EX.run(unit_net, P, dev, net, n=a.exam_battles, out_path=out / "exam.json")
    print(json.dumps({"updates": done, "minutes": round((time.time() - start) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
