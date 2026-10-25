"""The mission tourney (S28): the general's mission kind learns from copies of the battle, not from the critic.

Why (build/diag_intel): the general's kind gets a real reward only over its plan (~22 s), then the critic's estimate,
and the critic sees neither the plan nor the missions; the chosen kind explains ~0 of the outcome 60 s on. Here every
decision is compared with others directly, the same moment replayed.

A round: B battles of the learner (the general + its frozen executor) against S27 and a few scripts (v3 armies, the
fighters lab as the g43 line), T decisions on (auto reset: new battles as they end). The general's re-plans of those
decisions are captured by branch.Brancher (one own group of the plan, the snapshot before the decision's steps); at the
round's end each snapshot is played `copies` + 1 ways, horizon_s on (branch.play_kinds, the partial open loop: only the
swapped group re-decides, common random numbers): "chosen" (the plan as is) and copy c with the group's kind = alt_c:
  copy 0     - the teacher's kind and target for the group (Runtime.teacher_choice), where it has one and it differs;
               else as the others;
  copies 1.. - a kind drawn from the general's own policy over the other kinds it could give (not the chosen: the
               policy is sharp, ~80 % of free draws repeated it and priced exactly 0), its target the natural one of the kind (branch.alt_targets; the group's own where
               the kind takes it).
A variant's value = its trade at the horizon less "chosen"'s (the v2 reward's gold term, budget share): 0 for "chosen".
The advantage (variant_advantage) = value - the mean over the snapshot's variants ("chosen" included), / the running root
mean square of the advantages (at least ADV_FLOOR), within +-ADV_CLIP. The update: -sum over variants of A x log pi(kind)
of the group (the plan's record computed again, net.General "kind_logp"), minus entropy x the kind's entropy; only the
kind head (general.kind_q.) learns, the rest is frozen (executor.freeze_but_general). Not trained: the target head (the
alternatives' targets are the natural ones, not drawn from it).

    python -m tools.nn.general.mission_tourney --init build/nn-train/line_g43/g43_1_grafted.pt \\
        --out build/nn-train/missions/g44 --minutes 20 --device cuda --fighters

Checkpoints: latest.pt every round (the training's format: graft / run.py --init read it), the optimizer in "train".
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools.nn.general import branch as gbranch
from tools.nn.general.net import CHOICES

ADV_FLOOR = 0.002           # the advantages' scale's floor (budget share)
ADV_CLIP = 5.0
SCALE_KEEP = 0.8            # the scale's running mean keeps this share of the last
PREFIXES = ("general.kind_q.",)     # the trained parameters (the kind head; the kind table stays: the targets read it)


def variant_advantage(values):
    """[E, V] the variants' advantages: each value less the mean of its row (the snapshot's variants)."""
    return values - values.mean(1, keepdim=True)


class TourneyBrancher(gbranch.Brancher):
    """branch.Brancher with the tourney's variants: copy 0 the teacher's (else a policy draw), the others policy draws
    over the kinds the general could give the group (module doc)."""

    def teacher_alternatives(self, env, rt, nz, g, kind, valid):
        alt_t0, w, t_t0, t_w0, vs = super().teacher_alternatives(env, rt, nz, g, kind, valid)
        k, K = valid.shape
        C = self.cfg.copies
        ar = torch.arange(k, device=nz.device)
        logp = rt.r["kind_logp"][nz][ar, g].float()                                # [k, K]
        ok = valid.clone()
        ok[ar, kind] = False            # (the chosen one is the "chosen" variant already: a repeat prices exactly 0)
        p = logp.exp() * ok.float()
        p = torch.where(p.sum(1, keepdim=True) > 1e-8, p, ok.float())
        p = torch.where(p.sum(1, keepdim=True) > 0, p, torch.ones_like(p))
        draw = torch.multinomial(p, C, replacement=True, generator=self.gen)       # [k, C]
        d_t, d_w = gbranch.alt_targets(rt, nz, g, draw)
        first = vs[:, None] & (torch.arange(C, device=nz.device) == 0)[None]
        alt = torch.where(first, alt_t0, draw)
        alt_t = torch.where(first, t_t0, d_t)
        alt_w = torch.where(first[..., None], t_w0, d_w.float())
        return alt, w, alt_t, alt_w, vs


def layout(B, mix, seed):
    """league.Layout of B battles: the learner's side alternating, the opponents by the mix {name: share}."""
    from tools.nn.train import league
    rng = np.random.default_rng(seed)
    names = list(mix)
    p = np.asarray([mix[n] for n in names], dtype=float)
    ctrl = np.asarray([league.CODE[names[i]] for i in rng.choice(len(names), B, p=p / p.sum())])
    side = 1 + np.arange(B) % 2
    return league.Layout(np.zeros(B, dtype=int), side, ctrl)


def update(actor, opt, res, scale, entropy, epochs, minibatch, grad_clip):
    """The round's update on res {"rec", "group", "kinds" [E, V], "adv" [E, V] (scaled)} -> the log's numbers."""
    gen = actor.general
    rec, group, kinds, adv = res["rec"], res["group"], res["kinds"], res["adv"]
    E = len(group)
    params = [p for n, p in actor.named_parameters() if n.startswith(PREFIXES)]
    sums = {"pg": 0.0, "ent": 0.0, "n": 0}
    ar = torch.arange(E, device=group.device)
    with torch.no_grad():
        _, d0 = gen(rec, act={k: rec[k] for k in CHOICES})
        lp0 = d0["kind_logp"][ar, group].float()                                    # [E, K]
    gtor = torch.Generator().manual_seed(int(time.time()))
    for _ in range(epochs):
        order = torch.randperm(E, generator=gtor).to(group.device)
        for i in range(0, E, minibatch):
            idx = order[i:i + minibatch]
            sub = {k: v[idx] for k, v in rec.items()}
            _, d = gen(sub, act={k: sub[k] for k in CHOICES})
            a = torch.arange(len(idx), device=idx.device)
            logp = d["kind_logp"][a, group[idx]].float()                            # [m, K]
            lpv = logp.gather(1, kinds[idx])                                       # [m, V]
            pg = -(adv[idx] * lpv).sum(1).mean()
            p = logp.exp()
            ent = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(1).mean()
            loss = pg - entropy * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            sums["pg"] += float(pg.detach()) * len(idx)
            sums["ent"] += float(ent.detach()) * len(idx)
            sums["n"] += len(idx)
    with torch.no_grad():
        _, d1 = gen(rec, act={k: rec[k] for k in CHOICES})
        lp1 = d1["kind_logp"][ar, group].float()
        kl = float((lp0.exp() * (lp0 - lp1)).sum(1).mean())
        chose_best = float((lp1.gather(1, kinds).argmax(1) == res["value"].argmax(1)).float().mean())
    n = max(1, sums["n"])
    return {"pg": round(sums["pg"] / n, 5), "ent_kind": round(sums["ent"] / n, 4), "kl": round(kl, 5),
            "pi_top_is_best": round(chose_best, 3)}


def main(argv=None):
    from tools.nn.general.runtime import Runtime
    from tools.nn.general.executor import freeze_but_general
    from tools.nn.train import checkpoint, randomise, rollout, scenes
    from tools.nn.train.exam_sim import resolve
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--enemy", default="build/v2/itS27_4.pt", help="the past slot (S27)")
    ap.add_argument("--mix", default='{"past":0.6,"ai_like":0.15,"nearest":0.1,"hold_shoot":0.15}',
                    help="the opponents' shares (league names; past = --enemy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--battles", type=int, default=64)
    ap.add_argument("--decisions", type=int, default=32, help="main decisions a round")
    ap.add_argument("--snaps", type=int, default=64, help="snapshots (branched plans) a round at most")
    ap.add_argument("--copies", type=int, default=4, help="variants besides the chosen")
    ap.add_argument("--horizon", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=512, help="training army seeds in the bank")
    ap.add_argument("--armies", default="v3")
    ap.add_argument("--fighters", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-learn", action="store_true", help="only play and log (the signal's check)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    dev = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = checkpoint.read(resolve(args.init), dev)
    actor = checkpoint.load_policy(resolve(args.init), dev)
    critic = checkpoint.load_critic(resolve(args.init), dev)
    enemy = checkpoint.load_policy(resolve(args.enemy), dev)
    frozen, trainable = freeze_but_general(actor, critic, PREFIXES)
    params = [p for n, p in actor.named_parameters() if n.startswith(PREFIXES)]
    opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    ctx, men = None, None
    if args.fighters:
        from tools.nn.train import exam_sim
        ctx = exam_sim.lab(True, args.seed)
        men = ctx.__enter__()
    rng = np.random.default_rng(args.seed)
    seeds = [int(x) for x in rng.integers(0, 1_000_000_000, args.seeds)]
    params_sim = rollout.params_with_limit(900.0)
    source = scenes.Generated(seeds, 19, params_sim, dev, seed=args.seed, armies=args.armies)
    env = rollout.Battles(layout(args.battles, json.loads(args.mix), args.seed), (), device=dev, params=params_sim,
                          spread=randomise.Spread(), seed=args.seed, auto_reset=True, source=source,
                          men=men() if men is not None else None)
    env.set_past(enemy)
    rt = Runtime(env, actor.cfg, actor.gcfg)
    env.general_rt = rt
    bcfg = gbranch.Config(share=1.0, copies=args.copies, horizon_s=args.horizon, cap=args.snaps, batch=args.snaps,
                          alt="teacher", kinds="all", openloop=True, critic=False, norm="chunk", center=False,
                          seed=args.seed)
    gb = TourneyBrancher(bcfg, dev)
    rt.brancher = gb
    head = {"init": args.init, "enemy": args.enemy, "mix": args.mix, "args": vars(args), "branch": dataclasses.asdict(bcfg),
            "frozen": frozen, "trainable": trainable}
    print(json.dumps(head), flush=True)
    with open(out / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"head": head}) + "\n")
    t0 = time.time()
    rnd, scale = 0, None
    try:
        while True:
            if args.minutes > 0 and time.time() - t0 > args.minutes * 60:
                break
            if args.minutes <= 0 and rnd >= args.rounds:
                break
            t1 = time.time()
            actor.eval()
            gb.begin(env, args.decisions)
            with torch.no_grad():
                for _ in range(args.decisions):
                    env.step(actor, None, False)
            t_main = time.time() - t1
            got = gb.finish(env, actor)
            t_branch = time.time() - t1 - t_main
            if got is None:
                print(f"round {rnd}: no branches ({gb.stats})", flush=True)
                rnd += 1
                continue
            diff = gb.copies                                                        # [E, C] chosen - other
            E, C = diff.shape
            value = torch.cat([torch.zeros(E, 1, device=diff.device), -diff], 1)   # [E, V]: chosen 0
            kinds = torch.cat([got["kind"][:, None], got["alt"].clamp(min=0)], 1).long()
            raw = variant_advantage(value)
            rms = float(raw.pow(2).mean().sqrt())
            scale = max(ADV_FLOOR, rms if scale is None else SCALE_KEEP * scale + (1 - SCALE_KEEP) * rms)
            adv = (raw / scale).clamp(-ADV_CLIP, ADV_CLIP)
            info = {}
            if not args.no_learn:
                info = update(actor, opt, {"rec": got["rec"], "group": got["group"], "kinds": kinds, "adv": adv,
                                           "value": value}, scale, args.entropy, args.epochs, args.minibatch,
                              args.grad_clip)
            vs = gb.stats.get("vs_teacher_n", 0)
            row = {"round": rnd, "snaps": E, "variants": C + 1, "branch_battles": gb.stats.get("battles"),
                   "value_sd_within": round(float(value.std(1).mean()), 5),        # the spread of the variants' values
                   "adv_rms": round(rms, 5), "scale": round(scale, 5),
                   "best_is_chosen": round(float((value.argmax(1) == 0).float().mean()), 3),
                   "teacher_n": vs, "teacher_minus_chosen": -gb.stats.get("vs_teacher_price", 0.0),
                   "policy_minus_chosen": round(float(value[:, 2:].mean()), 5) if C > 1 else None,
                   "same_kind_share": round(float((kinds[:, 1:] == kinds[:, :1]).float().mean()), 3),
                   "gold_budget_share_sd": round(float(diff.std()), 5),
                   "s_main": round(t_main, 1), "s_branch": round(t_branch, 1), "s_round": round(time.time() - t1, 1),
                   "s_total": round(time.time() - t0), **info}
            with open(out / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            rnd += 1
            train = {"mission_optim": opt.state_dict(), "state": {"mission_tourney": {"round": rnd, "scale": scale}}}
            meta = dict(data.get("meta") or {}, mission_tourney={"round": rnd, "init": args.init,
                                                                 "log": str(out / "log.jsonl")})
            checkpoint.save(out / "latest.pt", actor, critic, meta=meta, train=train)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    print(f"done: {rnd} rounds, {time.time() - t0:.0f} s; {out / 'latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
