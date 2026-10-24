"""The learned group collection's anchor to the rule and its numbers (GeneralConfig.groups "learned", assign.py;
docs/ru/training/general.md "Сбор групп: обучение").

Anchor: on a plan's inputs the collection is asked for the rule's number of groups (K_rule: the rule's groups of our
free units, groups.collect) and its logits (the places below K_rule, no capacity mask) are pulled to the rule's
membership by a cross-entropy - the rule's groups matched to the places first (match: the pair sharing the most units
first, greedily, as net.match_groups), so the labels' order does not matter. The K head gets a cross-entropy to
K_rule alike. Its weight decays to a floor in the tourney (signal.imitate_weight): the rule is where the collection
starts, not what it must stay.

Numbers (stats()): ARI against the rule (groups.ari) of the greedy sequential membership with the capacity mask ("ari")
and of the plain most likely group of each unit ("ari_free": the mask's ceiling apart), groups a plan, the units'
role mix (mixed: the share of our groups of 2+ units with more than one of the 4 roles; mi_role: the mutual information
of a unit's group and role, nats), the cosine of our group cards of a plan, MI(group; mission kind) (the mean KL of a
group's kind distribution from the plan's mean, as build/diag_listen/probe.py).

Fit (stage 2 of the branch): `python -m tools.nn.general.assign_learn collect <ckpt> --out <records.pt>` plays a few
tourney battles of the checkpoint (its own groups: the rule) and keeps its plans' inputs; `fit <ckpt> --data
<records.pt> --out <ckpt'>` trains only the collection (assign., k_head.) on the anchor and writes the checkpoint with
groups "learned" (its held-out ARI printed).
"""
import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.nn.general import assign as gassign
from tools.nn.general import groups as ggroups
from tools.nn.general.net import NEG, match_groups  # noqa: F401  (match_groups: the same greedy rule)

PREFIXES = ("general.assign.", "general.k_head.")          # the branch's parameters (the rest frozen)


def rule_raw(gen, inp):
    """[P, N] the rule's places (groups.collect) of the plans' inputs."""
    cfg = gen.cfg
    return ggroups.collect(inp["num"], inp["lab"], inp["pos"], inp["own"], inp["free"], cfg.places, cfg.group_r,
                           cfg.lord_join_m)


def rule_k(rule, who, G):
    """[P] the rule's groups of our free units (at least 1)."""
    oh = F.one_hot(rule.clamp(0, G - 1), G) * who[..., None]
    return (oh.sum(1) > 0).sum(1).clamp(min=1)


def match(pred, rule, who, K, G):
    """[P, G] the place matched to each of the rule's groups (0..K-1 both): the pair sharing the most units first,
    greedily (ties: the same number); groups >= K -> themselves."""
    P = pred.shape[0]
    a = F.one_hot(rule.clamp(0, G - 1), G).float() * who[..., None]
    b = F.one_hot(pred.clamp(0, G - 1), G).float() * who[..., None]
    score = torch.einsum("pni,pnj->pij", a, b) + 1e-3 * torch.eye(G, device=pred.device)
    live = torch.arange(G, device=pred.device)[None] < K[:, None]
    score = score.masked_fill(~(live[:, :, None] & live[:, None, :]), float("-inf"))
    out = torch.arange(G, device=pred.device).repeat(P, 1)
    rows = torch.arange(P, device=pred.device)
    for _ in range(G):
        flat = score.reshape(P, -1)
        best, at = flat.max(-1)
        ok = torch.isfinite(best)
        i, j = at // G, at % G
        out[rows[ok], i[ok]] = j[ok]
        score[rows, i, :] = float("-inf")
        score[rows, :, j] = float("-inf")
    return out


def anchor(gen, inp, raw_rule=None, council=None):
    """The anchor's terms of plans' inputs -> {"ce" (the membership's, a mean over our free units), "ce_k" (the K
    head's, a mean over the plans), "pred" [P, N], "rule" [P, N], "K" [P], "logits", "who"}."""
    G = gen.cfg.places
    raw_rule = rule_raw(gen, inp) if raw_rule is None else raw_rule
    who = inp["own"] & inp["free"]
    rule = torch.where(who, raw_rule, torch.full_like(raw_rule, -1))
    K = rule_k(rule, who, G)
    t = gen.assign_terms(inp, None, K, council=council)
    live = torch.arange(G, device=K.device)[None, None] < K[:, None, None]
    x = t["logits"].float().masked_fill(~live, NEG)
    pred = x.detach().argmax(-1)
    place = match(pred, rule, who, K, G)
    target = place.gather(1, rule.clamp(min=0))
    n = who.sum().clamp(min=1)
    ce = (F.cross_entropy(x.reshape(-1, G), target.reshape(-1), reduction="none").reshape(target.shape) * who).sum() / n
    kl = t["k_logits"].masked_fill(~t["k_ok"], NEG)
    ce_k = F.cross_entropy(kl, (K - 1).clamp(min=0))
    return {"ce": ce, "ce_k": ce_k, "pred": torch.where(who, pred, torch.full_like(pred, -1)), "rule": rule, "K": K,
            "logits": t["logits"], "who": who}


@torch.no_grad()
def ari_of(gen, inp, a=None, council=None):
    """{"ari" [P] (the greedy sequential membership with the capacity mask, at K_rule), "ari_free" [P] (the plain most
    likely group, at K_rule), "k_rule" [P]} of plans' inputs against the rule."""
    G = gen.cfg.places
    a = anchor(gen, inp, council=council) if a is None else a
    who, K = a["who"], a["K"]
    k_eff, m, cap = gassign.capacity(who.sum(1), K, gen.cfg.assign_lo, gen.cfg.assign_hi)
    seq, _, _ = gassign.sample(a["logits"], who, k_eff, m, cap, greedy=True)
    r = a["rule"]
    return {"ari": ggroups.ari(seq.clamp(min=0), r.clamp(min=0), who, G),
            "ari_free": ggroups.ari(a["pred"].clamp(min=0), r.clamp(min=0), who, G), "k_rule": K}


def ari_ceiling(gen, inp):
    """[P] the best ARI against the rule any membership within the capacity bounds at K_rule can reach (greedy: the
    rule's groups cut / filled to the bounds by moving the fewest units - an estimate from below), and [P] whether the
    rule's own membership is within them."""
    G = gen.cfg.places
    raw_rule = rule_raw(gen, inp)
    who = inp["own"] & inp["free"]
    rule = torch.where(who, raw_rule, torch.full_like(raw_rule, -1))
    K = rule_k(rule, who, G)
    k_eff, m, cap = gassign.capacity(who.sum(1), K, gen.cfg.assign_lo, gen.cfg.assign_hi)
    # the sequential choice with the logits of the rule itself (+10 on its group): the nearest membership the mask lets
    logits = F.one_hot(rule.clamp(0, G - 1), G).float() * 10.0
    seq, _, _ = gassign.sample(logits, who, k_eff, m, cap, greedy=True)
    cnt = (F.one_hot(rule.clamp(0, G - 1), G) * who[..., None]).sum(1)
    live = torch.arange(G, device=K.device)[None] < K[:, None]
    inside = (((cnt >= m[:, None]) & (cnt <= cap[:, None])) | ~live).all(1)
    return ggroups.ari(seq.clamp(min=0), rule.clamp(min=0), who, G), inside


@torch.no_grad()
def stats(gen, inp, c=None, d=None):
    """The collection's numbers (module doc) of plans' inputs made by it (inp with "main", "perm" of the plans, or
    without: made here, greedy): {"groups", "mixed", "mi_role", "cos", "mi_kind", "ari"} (means)."""
    G = gen.cfg.places
    if c is None or d is None:
        c, d = gen(inp, greedy=True)
    main = c["main"]
    who = inp["own"] & inp["free"]
    role = ggroups.roles(inp["num"], inp["lab"])
    mem = (F.one_hot(main.clamp(0, G - 1), G) * (who & (main >= 0) & (main < G))[..., None]).float()   # [P, N, G]
    size = mem.sum(1)                                                                              # [P, G]
    act = size > 0
    nr = len(ggroups.ROLES)
    rh = F.one_hot(role.clamp(0, nr - 1), nr).float()
    joint = torch.einsum("png,pnr->pgr", mem, rh)                                                   # [P, G, roles]
    roles_in = (joint > 0).sum(-1)
    big = size >= 2
    mixed = float(((roles_in > 1) & big).sum() / big.sum().clamp(min=1))
    n = size.sum(1, keepdim=True).clamp(min=1)
    pj = joint / n[..., None]
    pg, pr = pj.sum(2, keepdim=True), pj.sum(1, keepdim=True)
    mi = (pj * (torch.log(pj.clamp(min=1e-12)) - torch.log((pg * pr).clamp(min=1e-12)))).sum((1, 2))
    gc = F.normalize(c["groups"][:, :G].float(), dim=-1)
    cs = gc @ gc.transpose(1, 2)
    pm = act[:, :, None] & act[:, None, :] & ~torch.eye(G, dtype=torch.bool, device=gc.device)[None]
    cos = float(cs[pm].mean()) if bool(pm.any()) else float("nan")
    kp = d["kind_logp"].float().exp()                                                              # [P, G, K]
    sel = act
    if bool(sel.any()):
        p = kp[sel]
        mk = p.mean(0, keepdim=True)
        mi_kind = float((p * (p.clamp(min=1e-30).log() - mk.clamp(min=1e-30).log())).sum(-1).mean())
    else:
        mi_kind = float("nan")
    rule = rule_raw(gen, inp)
    raw = torch.where(main >= 0, c["perm"].gather(1, main.clamp(min=0)), main)
    ari = ggroups.ari(raw.clamp(0, G - 1), rule.clamp(0, G - 1), who, G)
    return {"groups": float(act.sum(1).float().mean()), "mixed": mixed, "mi_role": float(mi.mean()), "cos": cos,
            "mi_kind": mi_kind, "ari": float(ari.mean())}


def freeze_but(actor, critic=None, prefixes=PREFIXES):
    """executor.freeze_but_general with the prefixes given (the branch: the collection alone)."""
    from tools.nn.general.executor import freeze_but_general
    return freeze_but_general(actor, critic, prefixes)


def branch_params(actor, prefixes=PREFIXES):
    return [p for n, p in actor.named_parameters() if n.startswith(prefixes)]


def with_learned(actor, **over):
    """The actor's general switched to groups "learned" (set_rules: the same shape), with the config's fields over."""
    gcfg = dataclasses.replace(actor.gcfg, groups="learned", **over)
    actor.set_rules(gcfg)
    return actor


# --- stage 2: the anchor's fit on the plans' inputs --------------------------------------------------------------------
INPUT_KEYS = ("num", "lab", "pos", "on", "own", "free", "tiles", "tile_ok", "zone", "deploy", "prev", "has_prev",
              "prev_main", "prev_kind", "fight", "K")


def collect(args):
    from tools.nn.general import tourney
    from tools.nn.train import checkpoint
    from tools.nn.train.exam_sim import resolve
    actor = checkpoint.load_policy(resolve(args.checkpoint), args.device)
    enemy = checkpoint.load_policy(resolve(args.enemy), args.device)
    cfg = tourney.Config(states=args.states, ks=(3,), copies=1, limit_s=args.limit_s, armies=args.armies,
                         fighters=False, seed=args.seed)
    out = {}
    t0 = time.time()
    for rnd in range(args.rounds):
        res = tourney.play(actor, enemy, cfg, rnd=rnd, device=args.device)
        rec = {k: v.cpu() for k, v in res["records"].items() if k in INPUT_KEYS + ("battle",)}
        rec["battle"] = rec["battle"] + rnd * len(res["trade"])
        for k, v in rec.items():
            out.setdefault(k, []).append(v)
        print(f"round {rnd}: {len(rec['battle'])} plans, {res['decisions']} decisions, {time.time() - t0:.0f} s",
              flush=True)
    data = {k: torch.cat(v) for k, v in out.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    print(f"written: {args.out} ({len(data['battle'])} plans)", flush=True)


def _rows(d, idx):
    return {k: v[idx] for k, v in d.items()}


def fit(args):
    from tools.nn.train import checkpoint
    from tools.nn.train.exam_sim import resolve
    dev = args.device
    actor = checkpoint.load_policy(resolve(args.checkpoint), dev)
    over = {}
    if args.lo is not None:
        over["assign_lo"] = args.lo
    if args.hi is not None:
        over["assign_hi"] = args.hi
    with_learned(actor, **over)
    gen = actor.general
    parts = [torch.load(f, weights_only=True) for f in args.data.split(",")]
    for i, d in enumerate(parts):
        d["battle"] = d["battle"] + 1_000_000 * i              # (each file's battles apart)
    data = {k: torch.cat([d[k] for d in parts]).to(dev) for k in parts[0]}
    inp = {k: v for k, v in data.items() if k in INPUT_KEYS}
    M = len(data["battle"])
    # held out: every 5th battle's plans
    val = (data["battle"] % 5) == 0
    tr_idx, va_idx = (~val).nonzero().squeeze(1), val.nonzero().squeeze(1)
    gen.eval()
    with torch.no_grad():
        units, before, rule = [], [], []
        for i in range(0, M, 512):
            sl = _rows(inp, torch.arange(i, min(M, i + 512), device=dev))
            u, _, b = gen.council_out(sl)
            units.append(u)
            before.append(b)
            rule.append(rule_raw(gen, sl))
        units, before, rule = torch.cat(units), torch.cat(before), torch.cat(rule)
    params = branch_params(actor)
    for p in actor.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=args.lr)
    gtor = torch.Generator(device="cpu").manual_seed(args.seed)

    def evaluate(idx):
        sl = _rows(inp, idx)
        a = anchor(gen, sl, rule[idx], (units[idx], before[idx]))
        r = ari_of(gen, sl, a)
        return float(a["ce"]), float(r["ari"].mean()), float(r["ari_free"].mean())

    t0 = time.time()
    with torch.no_grad():
        ceil, inside = ari_ceiling(gen, _rows(inp, va_idx))
    log = {"plans": M, "train": len(tr_idx), "val": len(va_idx), "ceiling_ari": round(float(ceil.mean()), 4),
           "rule_within_bounds": round(float(inside.float().mean()), 4),
           "lo": gen.cfg.assign_lo, "hi": gen.cfg.assign_hi}
    print(json.dumps(log), flush=True)
    hist = []
    for ep in range(args.epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=gtor).to(dev)]
        tot = 0.0
        for i in range(0, len(perm), args.minibatch):
            idx = perm[i:i + args.minibatch]
            a = anchor(gen, _rows(inp, idx), rule[idx], (units[idx], before[idx]))
            loss = a["ce"] + args.k_weight * a["ce_k"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        with torch.no_grad():
            ce, ari, ari_free = evaluate(va_idx)
            _, ari_tr, _ = evaluate(tr_idx[:len(va_idx)])
        row = {"epoch": ep + 1, "train_loss": round(tot / len(perm), 4), "train_ari": round(ari_tr, 4),
               "val_ce": round(ce, 4), "val_ari": round(ari, 4), "val_ari_free": round(ari_free, 4),
               "s": round(time.time() - t0)}
        hist.append(row)
        print(json.dumps(row), flush=True)
    with torch.no_grad():
        st = stats(gen, _rows(inp, va_idx[:512]))
    print(json.dumps({"val_stats": {k: round(v, 4) for k, v in st.items()}}), flush=True)
    if args.out:
        for p in actor.parameters():
            p.requires_grad_(True)
        meta = dict(checkpoint.meta(resolve(args.checkpoint)), assign_fit={"log": log, "history": hist,
                                                                           "data": str(args.data)})
        checkpoint.save(args.out, actor, checkpoint.load_critic(resolve(args.checkpoint), dev), meta=meta)
        print(f"written: {args.out}", flush=True)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="play tourney battles of a checkpoint, keep its plans' inputs")
    c.add_argument("checkpoint")
    c.add_argument("--enemy", default="build/v2/itS27_4.pt")
    c.add_argument("--out", required=True)
    c.add_argument("--states", type=int, default=16)
    c.add_argument("--rounds", type=int, default=1)
    c.add_argument("--limit-s", type=float, default=600.0)
    c.add_argument("--armies", default="v3")
    c.add_argument("--seed", type=int, default=11)
    c.add_argument("--device", default="cpu")
    f = sub.add_parser("fit", help="train the collection on the anchor alone")
    f.add_argument("checkpoint")
    f.add_argument("--data", required=True, help="collect's files, comma separated")
    f.add_argument("--out")
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--minibatch", type=int, default=256)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--k-weight", type=float, default=0.2)
    f.add_argument("--lo", type=float)
    f.add_argument("--hi", type=float)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    (collect if args.cmd == "collect" else fit)(args)


if __name__ == "__main__":
    main()
