"""The learned group collection's own training branch (GeneralConfig.groups "learned"; docs/ru/training/general.md
"Сбор групп: турнир"): only the collection (general.assign., general.k_head.) learns, the rest of the network - the
general's other parts, the executor, the critic - is frozen; the signal is the tourney's (tourney.py), not PPO's.

    python -m tools.nn.general.tourney_train --init build/nn-train/test5/g38_general/m40.pt \\
        --assign-init build/learned_groups/fit_g38.pt --out build/nn-train/groups/t1 --minutes 30 --device cuda

A round (synchronous: the update after the round's battles end): tourney.play (S starts x K set x copies), then
  membership: a policy gradient with PPO's clipped ratio on every plan of the learner (the membership's
              log-probability: assign.log_prob of the plan's own membership, all units at once) with the battle's
              advantage - its trade less the mean of the same start and K (tourney.advantages), / the running root
              mean square of the advantages (at least ADV_FLOOR), within +-ADV_CLIP; minus entropy x the membership's
              entropy a unit;
  K head:     on each start's deployment plan, -sum over the K played of pi(K) x A(K) (A: the K's mean trade less the
              mean over the K played, / the same scale), minus k_entropy x the head's entropy;
  anchor:     the cross-entropy to the rule's groups and K (assign_learn.anchor) on the round's plans, its weight w0
              down to a floor over anchor_decay rounds (signal.imitate_weight; run.py imitate_floor's way); --anchor 0:
              none (the collection free of the rule);
  mixed:      --mixed-penalty C (0 off): a plan's advantage less C x (its share of mixed groups - the round's mean;
              groups.mixed: shooters and melee in one group), in trade units before the scale (plan_advantage);
  kept plans (GeneralConfig.groups_sticky: a timer re-plan kept the last groups, tourney "kept") are left out of the
              membership's term - their membership was not chosen.
The council's cards are computed once a round without gradients (frozen). The log (one JSON line a round, log.jsonl;
a short line on the console): trade by K, the advantages' scale, the losses, the ARI to the rule, our groups a plan
(exactly K_eff), the role mix, MI(group; role), the cosine of the group cards, MI(group; kind), the K head's choice.
Checkpoints: latest.pt every round (the branch's optimizer and round in "train": --resume goes on), r<N>.pt every
--save-every rounds; the same format as the training's (exam_sim, graft.py read them).
"""
import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools.nn.general import assign_learn as al
from tools.nn.general import groups as ggroups
from tools.nn.general import signal as gsignal
from tools.nn.general import tourney

ADV_FLOOR = 0.01            # the advantages' scale's floor (budget shares)
ADV_CLIP = 5.0
SCALE_KEEP = 0.8            # the scale's running mean keeps this share of the last


@dataclasses.dataclass
class Learn:
    lr: float = 1e-4
    epochs: int = 4
    minibatch: int = 256
    clip: float = 0.2
    entropy: float = 0.003               # x the membership's entropy a unit
    k_entropy: float = 0.01
    k_weight: float = 1.0                # x the K head's term
    anchor: float = 0.5                  # the anchor's weight at round 0 ...
    anchor_decay: int = 40               # ... down to anchor_min over this many rounds (0: no decay)
    anchor_min: float = 0.05
    grad_clip: float = 1.0
    kl_max: float = 0.02                 # the epochs stop when the KL a unit (from the plans' policy) passes it
    mixed_penalty: float = 0.0           # x the plan's share of mixed groups (groups.mixed: shooters and melee in one
    #                                      group), taken off its advantage in trade units (less the round's mean); 0 off


def _rows(d, idx):
    return {k: v[idx] for k, v in d.items()}


def raw_of(rec):
    """[M, N] the plans' units' places before the numbering (perm[main]; -1 none)."""
    main = rec["main"]
    return torch.where(main >= 0, rec["perm"].gather(1, main.clamp(min=0)), main)


def k_term(gen, rec, res, ak, scale, k_entropy, council=None):
    """The K head's loss of a round: each start's deployment plan (its first battle's), -sum_K pi(K) A(K) over the
    K played (ak [S, len(ks)] / scale) - k_entropy x H(pi). -> (loss, pi [S, G] numpy)."""
    ks = res["ks"]
    S = ak.shape[0]
    first = {}
    battle = rec["battle"].tolist()
    dep = rec["deploy"].tolist()
    for i, (b, d) in enumerate(zip(battle, dep)):
        s = int(res["state"][b])
        if d and s not in first:
            first[s] = i
    if not first:
        return None, None
    st = sorted(first)
    idx = torch.as_tensor([first[s] for s in st], device=rec["battle"].device)
    sub = _rows({k: v for k, v in rec.items() if k in al.INPUT_KEYS}, idx)
    t = gen.assign_terms(sub, None, torch.ones(len(idx), dtype=torch.long, device=idx.device),
                         council=None if council is None else (council[0][idx], council[1][idx]))
    lg = t["k_logits"].masked_fill(~t["k_ok"], -1e9)
    logp = torch.log_softmax(lg, -1)
    p = logp.exp()
    a = torch.as_tensor(ak[st] / scale, dtype=p.dtype, device=p.device).clamp(-ADV_CLIP, ADV_CLIP)
    kk = torch.as_tensor([k - 1 for k in ks], device=p.device)
    gain = (p[:, kk] * a).sum(-1)
    ent = -(p * torch.where(t["k_ok"], logp, torch.zeros_like(logp))).sum(-1)
    return -(gain.mean() + k_entropy * ent.mean()), p.detach().cpu().numpy()


def plan_advantage(a_battle, share, coef, chosen, scale):
    """[M] the plans' advantages: a_battle [M] (their battle's, trade units) less coef x (share [M], the plan's share of
    mixed groups, - its mean over the chosen plans `chosen` [M]), / scale, within +-ADV_CLIP."""
    adv = a_battle.float()
    if coef:
        w = chosen.float()
        mean = (share * w).sum() / w.sum().clamp(min=1.0)
        adv = adv - coef * (share.float() - mean)
    return (adv / scale).clamp(-ADV_CLIP, ADV_CLIP)


def update(actor, opt, res, learn, rnd, scale):
    """One round's update (module doc) -> the log's numbers."""
    gen = actor.general
    rec = res["records"]
    inp = {k: v for k, v in rec.items() if k in al.INPUT_KEYS}
    M = len(rec["battle"])
    dev = rec["battle"].device
    a_b, ak, mean = tourney.advantages(res["trade"], res["state"], res["K"], res["ks"])
    raw = raw_of(rec)
    G = gen.cfg.places
    who_all = rec["own"] & rec["free"]
    chosen = ~rec["kept"].bool() if "kept" in rec else torch.ones(M, dtype=torch.bool, device=dev)
    share = ggroups.mixed_share(raw, who_all, ggroups.roles(rec["num"], rec["lab"]), G)
    adv = plan_advantage(torch.as_tensor(a_b, dtype=torch.float32, device=dev)[rec["battle"]], share,
                         learn.mixed_penalty, chosen, scale)
    K = rec["K"].long()
    old = rec["lp_assign"].float()
    w_anchor = gsignal.imitate_weight(learn.anchor, learn.anchor_decay, rnd, learn.anchor_min)
    gen.eval()
    with torch.no_grad():
        units, before, rule = [], [], []
        for i in range(0, M, 1024):
            sl = _rows(inp, torch.arange(i, min(M, i + 1024), device=dev))
            u, _, b = gen.council_out(sl)
            units.append(u)
            before.append(b)
            rule.append(al.rule_raw(gen, sl))
        units, before, rule = torch.cat(units), torch.cat(before), torch.cat(rule)
    params = al.branch_params(actor)
    gtor = torch.Generator().manual_seed(rnd)
    sums = {"pg": 0.0, "ent": 0.0, "anchor": 0.0, "kl": 0.0, "clip": 0.0, "n": 0}
    k_loss, pi = None, None
    kl_first, kl_unit, stopped, steps = None, 0.0, False, 0
    for ep in range(learn.epochs):
        if stopped:
            break
        order = torch.randperm(M, generator=gtor).to(dev)
        for i in range(0, M, learn.minibatch):
            idx = order[i:i + learn.minibatch]
            sub = _rows(inp, idx)
            t = gen.assign_terms(sub, raw[idx], K[idx], council=(units[idx], before[idx]))
            n_u = t["who"].sum(1).clamp(min=1).float()
            with torch.no_grad():
                # the KL a unit from the plans' own policy (k3); the epochs stop beyond kl_max (learn.py's way)
                lr0 = (t["lp"] - old[idx]).clamp(-20, 20)
                c0 = chosen[idx].float()                # (a kept plan's membership was not chosen: no KL)
                kl_unit = float(((torch.expm1(lr0) - lr0) / n_u * c0).sum() / c0.sum().clamp(min=1.0))
                if kl_first is None:
                    kl_first = kl_unit             # (before any step: ~0 when play and update agree)
            if steps and learn.kl_max and kl_unit > learn.kl_max:
                stopped = True
                break
            ratio = torch.exp((t["lp"] - old[idx]).clamp(-20, 20))
            a = adv[idx]
            ch = chosen[idx].float()
            n_ch = ch.sum().clamp(min=1.0)
            surr = torch.minimum(ratio * a, ratio.clamp(1 - learn.clip, 1 + learn.clip) * a) * ch
            ent = (t["ent"] / n_u * ch).sum() / n_ch
            loss = -surr.sum() / n_ch - learn.entropy * ent
            if w_anchor > 0:
                an = al.anchor(gen, sub, rule[idx], (units[idx], before[idx]))
                loss = loss + w_anchor * (an["ce"] + learn.k_weight * an["ce_k"])
                sums["anchor"] += float(an["ce"].detach()) * len(idx)
            if i == 0:
                # the K head's term once an epoch, with its first minibatch (its starts are few)
                k_loss, pi = k_term(gen, rec, res, ak, scale, learn.k_entropy, (units, before))
                if k_loss is not None:
                    loss = loss + learn.k_weight * k_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, learn.grad_clip)
            opt.step()
            steps += 1
            with torch.no_grad():
                lr_ = (t["lp"] - old[idx]).clamp(-20, 20)
                sums["kl"] += float(((torch.expm1(lr_) - lr_) * ch).sum() / n_ch) * len(idx)
                sums["clip"] += float((((ratio - 1).abs() > learn.clip).float() * ch).sum() / n_ch) * len(idx)
                sums["pg"] += float(-surr.sum().detach() / n_ch) * len(idx)
                sums["ent"] += float(ent.detach()) * len(idx)
                sums["n"] += len(idx)
    n = max(1, sums["n"])
    out = {"w_anchor": round(w_anchor, 4), "pg": round(sums["pg"] / n, 5), "ent_unit": round(sums["ent"] / n, 4),
           "anchor_ce": round(sums["anchor"] / n, 4) if w_anchor > 0 else None, "kl": round(sums["kl"] / n, 5),
           "clip": round(sums["clip"] / n, 4), "k_loss": None if k_loss is None else round(float(k_loss.detach()), 5),
           "kl_first_unit": None if kl_first is None else round(kl_first, 6), "kl_unit_last": round(kl_unit, 5),
           "steps": steps, "kl_stop": stopped, "kept_share": round(float(1.0 - chosen.float().mean()), 4),
           "mixed_share": round(float(share[chosen].mean()) if bool(chosen.any()) else 0.0, 4)}
    if pi is not None:
        out["k_pi"] = [round(float(x), 3) for x in pi.mean(0)]
    return out, a_b, ak, mean


@torch.no_grad()
def round_stats(actor, res, n=512):
    """The round's collection numbers on its plans (assign_learn.stats on up to n plans; the ARI of every plan)."""
    gen = actor.general
    rec = res["records"]
    G = gen.cfg.places
    who = rec["own"] & rec["free"]
    rule = al.rule_raw(gen, {k: v for k, v in rec.items() if k in al.INPUT_KEYS})
    ari = ggroups.ari(raw_of(rec).clamp(0, G - 1), rule.clamp(0, G - 1), who, G)
    act = rec["active"][:, :G].sum(1)
    k_eff = torch.minimum(rec["K"].long(), who.sum(1))
    M = len(rec["battle"])
    idx = torch.randperm(M, generator=torch.Generator().manual_seed(0))[:n].to(rec["battle"].device)
    sub = {k: v[idx] for k, v in rec.items() if k in al.INPUT_KEYS + ("main", "perm")}
    st = al.stats(gen, sub)
    st.update(ari_all=float(ari.mean()), groups_all=float(act.float().mean()),
              exact_k=float((act == k_eff).float().mean()), plans=M)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()}


def main(argv=None):
    from tools.nn.train import checkpoint
    from tools.nn.train.exam_sim import resolve
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="the network (a checkpoint with a general)")
    ap.add_argument("--assign-init", help="a checkpoint whose collection (general.assign., general.k_head.) starts "
                                          "it (assign_learn fit's); default: --init's own")
    ap.add_argument("--resume", action="store_true", help="--init is this branch's latest.pt: its optimizer and round")
    ap.add_argument("--enemy", default="build/v2/itS27_4.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=0.0, help="stop after this long (0: --rounds)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--states", type=int, default=32)
    ap.add_argument("--ks", default="2,3,5,8")
    ap.add_argument("--ks-random", type=int, default=0, help="> 0: each round this many K of 1..--ks-max")
    ap.add_argument("--ks-max", type=int, default=10)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--limit-s", type=float, default=900.0)
    ap.add_argument("--armies", default="v3")
    ap.add_argument("--fighters", action="store_true", help="the fighters lab (as the training's --fighters)")
    ap.add_argument("--mode", default="batch", choices=("batch", "lanes"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lo", type=float, help="GeneralConfig.assign_lo (default: the checkpoint's)")
    ap.add_argument("--hi", type=float, help="GeneralConfig.assign_hi")
    for f in dataclasses.fields(Learn):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = ap.parse_args(argv)
    learn = Learn(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Learn)})
    dev = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = checkpoint.read(resolve(args.init), dev)
    actor = checkpoint.load_policy(resolve(args.init), dev)
    critic = checkpoint.load_critic(resolve(args.init), dev)
    over = {k: v for k, v in (("assign_lo", args.lo), ("assign_hi", args.hi)) if v is not None}
    al.with_learned(actor, **over)
    if args.assign_init:
        src = checkpoint.read(resolve(args.assign_init), dev)["actor"]
        own = actor.state_dict()
        n = 0
        for k, v in src.items():
            if k.startswith(al.PREFIXES):
                own[k] = v
                n += 1
        actor.load_state_dict(own)
        print(f"the collection from {args.assign_init}: {n} tensors", flush=True)
    frozen, trainable = al.freeze_but(actor, critic)
    params = al.branch_params(actor)
    opt = torch.optim.Adam(params, lr=learn.lr, eps=1e-5)
    rnd0, scale = 0, None
    tr = data.get("train") or {}
    if args.resume and "tourney" in (tr.get("state") or {}):
        st = tr["state"]["tourney"]
        rnd0, scale = int(st["round"]), float(st["scale"])
        try:
            opt.load_state_dict(tr["tourney_optim"])
        except (ValueError, KeyError) as e:
            print(f"the optimizer's state not taken ({e}): Adam afresh", flush=True)
    enemy = checkpoint.load_policy(resolve(args.enemy), dev)
    cfg = tourney.Config(states=args.states, ks=tuple(int(k) for k in args.ks.split(",")), copies=args.copies,
                         limit_s=args.limit_s, armies=args.armies, fighters=args.fighters, mode=args.mode,
                         seed=args.seed, ks_random=args.ks_random, ks_max=args.ks_max)
    men = None
    ctx = None
    if args.fighters:
        from tools.nn.train import exam_sim
        ctx = exam_sim.lab(True, args.seed)
        men = ctx.__enter__()
    head = {"init": args.init, "assign_init": args.assign_init, "enemy": args.enemy, "tourney": dataclasses.asdict(cfg),
            "learn": dataclasses.asdict(learn), "gcfg": dataclasses.asdict(actor.gcfg), "frozen": frozen,
            "trainable": trainable, "device": dev}
    print(json.dumps(head), flush=True)
    with open(out / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"head": head}) + "\n")
    t0 = time.time()
    rnd = rnd0
    try:
        while True:
            if args.minutes > 0 and time.time() - t0 > args.minutes * 60:
                break
            if args.minutes <= 0 and rnd - rnd0 >= args.rounds:
                break
            t1 = time.time()
            actor.eval()
            res = tourney.play(actor, enemy, cfg, rnd=rnd, device=dev, men=men)
            t_play = time.time() - t1
            if not res["records"]:
                print(f"round {rnd}: no plans", flush=True)
                rnd += 1
                continue
            a_b = tourney.advantages(res["trade"], res["state"], res["K"], res["ks"])[0]
            rms = float(np.sqrt(np.mean(a_b ** 2)))
            scale = max(ADV_FLOOR, rms if scale is None else SCALE_KEEP * scale + (1 - SCALE_KEEP) * rms)
            st0 = round_stats(actor, res)
            info, a_b, ak, mean = update(actor, opt, res, learn, rnd, scale)
            row = {"round": rnd, "battles": len(res["trade"]), "ks": list(res["ks"]), "decisions": res["decisions"],
                   "trade": round(float(np.mean(res["trade"])), 4), "won": round(float(np.mean(res["won"])), 4),
                   "trade_by_k": [round(float(x), 4) for x in mean.mean(0)], "adv_rms": round(rms, 4),
                   "scale": round(scale, 4), "battle_s": round(float(np.mean(res["t"])), 1), **info, **st0,
                   "s_play": round(t_play), "s_round": round(time.time() - t1), "s_total": round(time.time() - t0)}
            with open(out / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"round {rnd}: trade {row['trade']:+.4f} by K {dict(zip(res['ks'], row['trade_by_k']))} adv_rms "
                  f"{rms:.4f} | ari {st0['ari_all']:.3f} groups {st0['groups_all']:.2f} exact_k {st0['exact_k']:.2f} "
                  f"mixed {st0['mixed']:.2f} mi_role {st0['mi_role']:.3f} cos {st0['cos']:.3f} mi_kind "
                  f"{st0['mi_kind']:.3f} | kl {info['kl']:.4f} clip {info['clip']:.3f} anchor {info['anchor_ce']} "
                  f"k_pi {info.get('k_pi')} | {row['s_round']} s", flush=True)
            rnd += 1
            # (the branch's optimizer apart from the training's "optim": run.py --init does not take it for its own)
            train = {"tourney_optim": opt.state_dict(), "state": {"tourney": {"round": rnd, "scale": scale}}}
            meta = dict(data.get("meta") or {}, tourney={"round": rnd, "init": args.init, "log": str(out / "log.jsonl")})
            checkpoint.save(out / "latest.pt", actor, critic, meta=meta, train=train)
            if args.save_every and rnd % args.save_every == 0:
                checkpoint.save(out / f"r{rnd}.pt", actor, critic, meta=meta)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    print(f"done: {rnd - rnd0} rounds, {time.time() - t0:.0f} s; {out / 'latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
