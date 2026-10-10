"""Cut the eyes' feedback out of a trained v2 checkpoint (docs/en/training/model.md "Eyes").

    bash tools/nn/dock.sh tools.nn.model.eyes_cut --src build/v2/itW1.pt --dst build/v2/itW1_cut.pt

The eyes (eyes.py) stay as an auxiliary task: their heads and loss are kept. Only the way back is cut: the layers that
added their predictions to the unit and sector tokens (eyes.FEEDBACK_PARAMS) are dropped and the config gets
eyes_feedback False. Everything else is copied as it is: the actor's other weights, the critic (it never saw the eyes),
the meta and the "train" part run.py --init goes on from - Adam's moments with the dropped weights' entries taken out
(the rest renumbered in the optimizer's order), the trainer's counters. So the next part of a chain starts from the cut
file as from any latest.pt.

The cut network computes the decision without the feedback, so its choices move a little; the check (--check-battles,
default 16 simulator battles x 48 decisions against ai_like, on the CPU) prints how much: per head of the chain, the
mean KL(uncut || cut) and the share of rows whose most likely choice changed, on the rows where that head is used
(head_entropy.rows; both networks under the actions the uncut one took, so every head is compared on the same
condition). The memory (GRU) comes before the eyes, so it is the same in both.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.nn.model import critic as model_critic
from tools.nn.model import eyes as meyes
from tools.nn.model import heads as hd
from tools.nn.model import policy as model_policy

HEADS = ("kind", "target", "sector", "cell", "hold", "run", "ability")


def cut_config(cfg):
    """The config without the feedback (ValueError when there is nothing to cut)."""
    if not (cfg.sectors and cfg.eyes and cfg.eyes_feedback):
        raise ValueError("not a v2 network whose eyes feed back (sectors, eyes and eyes_feedback): nothing to cut")
    return dataclasses.replace(cfg, eyes_feedback=False)


def cut_actor_state(state):
    """The actor's state_dict without the feedback layers."""
    return {k: v for k, v in state.items() if not k.startswith(meyes.FEEDBACK_PARAMS)}


def param_names(cfg):
    """The names of the parameters Adam holds, in its order (run.optimizer: the actor's, then the critic's)."""
    return ([f"actor.{n}" for n, _ in model_policy.Actor(cfg).named_parameters()]
            + [f"critic.{n}" for n, _ in model_critic.Critic(cfg).named_parameters()])


def cut_optim(saved, old_names, new_names):
    """Adam's state_dict over old_names -> over new_names (every new name must be an old one): the dropped parameters'
    entries out, the rest renumbered. None when it is not one group over all of old_names (then Adam starts afresh)."""
    groups = saved["param_groups"]
    if len(groups) != 1 or list(groups[0]["params"]) != list(range(len(old_names))):
        return None
    old_index = {n: i for i, n in enumerate(old_names)}
    keep = [old_index[n] for n in new_names]
    state = {j: saved["state"][i] for j, i in enumerate(keep) if i in saved["state"]}
    return {"state": state, "param_groups": [dict(groups[0], params=list(range(len(keep))))]}


def cut(data, src=None):
    """A checkpoint dict (checkpoint.read) -> (the cut checkpoint dict, a note on Adam's state)."""
    from tools.nn.train import checkpoint
    cfg = checkpoint.config_of(data)
    new_cfg = cut_config(cfg)
    actor = model_policy.Actor(new_cfg)
    actor.load_state_dict(cut_actor_state(data["actor"]))           # strict: every other weight is there
    out = dict(data, config=dataclasses.asdict(new_cfg),
               actor={k: v.detach().cpu() for k, v in actor.state_dict().items()},
               meta=dict(data.get("meta", {}), eyes_cut_from=None if src is None else str(src)))
    note = "no optimizer state in the checkpoint"
    train = data.get("train")
    if train is not None:
        train = dict(train)
        if "optim" in train:
            optim = cut_optim(train["optim"], param_names(cfg), param_names(new_cfg))
            if optim is None:
                train.pop("optim")
                note = "the optimizer's state is not one group over the actor and critic: dropped (Adam starts afresh)"
            else:
                train["optim"] = optim
                note = (f"Adam's state kept: {len(train['optim']['state'])} parameters "
                        f"(of {len(data['train']['optim']['state'])})")
        out["train"] = train
    return out, note


def cut_file(src, dst):
    from tools.nn.train import checkpoint
    data = checkpoint.read(src)
    out, note = cut(data, src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(out, tmp)
    tmp.replace(dst)
    return data, out, note


# --- the check: how far the cut network's decisions moved ---

def _kl_rows(head, la, lb):
    """[R, N] each: KL(uncut || cut) of the head's distribution, its total variation (the share of probability that
    moved to other choices), the most likely choice changed, 2 or more choices allowed."""
    from tools.nn.model.heads import NEG
    from tools.nn.train.head_entropy import LOGITS
    key = LOGITS.get(head, head)
    a, b = (torch.nan_to_num(x[key].double(), nan=NEG) for x in (la, lb))
    if head == "run":
        pa = torch.sigmoid(a)
        kl = pa * (F.logsigmoid(a) - F.logsigmoid(b)) + (1 - pa) * (F.logsigmoid(-a) - F.logsigmoid(-b))
        return kl, (pa - torch.sigmoid(b)).abs(), (a > 0) != (b > 0), torch.ones_like(a, dtype=torch.bool)
    lpa, lpb = F.log_softmax(a, -1), F.log_softmax(b, -1)
    kl = (lpa.exp() * (lpa - lpb)).sum(-1)
    tv = 0.5 * (lpa.exp() - lpb.exp()).abs().sum(-1)
    return kl, tv, a.argmax(-1) != b.argmax(-1), (a > NEG / 2).sum(-1) >= 2


def compare(la, lb, action, obs):
    """{head: {"kl": mean KL(uncut || cut), "tv": mean total variation, "flip": share of rows whose most likely choice
    changed, "rows": n}} on the rows where the head is used and has 2 or more choices (logits flattened [R, N, ...]),
    and "any": the share of deciding units for which some head they use flipped."""
    from tools.nn.train import head_entropy, ppo
    free = ppo.deciding(obs)
    ctrl = obs["ctrl"]
    out, any_flip = {}, torch.zeros_like(free)
    for h in HEADS:
        key = head_entropy.LOGITS.get(h, h)
        if key not in la or (h == "hold" and action.commit is None) or (h == "ability" and action.ability is None):
            continue
        kl, tv, flip, choices = _kl_rows(h, la, lb)
        m = (ctrl if h == "ability" else head_entropy.rows(h, action, free, ctrl)) & choices
        n = int(m.sum())
        mean = (lambda v: float(v[m].double().mean()) if n else 0.0)
        out[h] = {"kl": mean(kl), "tv": mean(tv), "flip": mean(flip), "rows": n}
        any_flip |= flip & m
    out["any"] = {"flip": float(any_flip[free].float().mean()), "rows": int(free.sum())}
    return out


@torch.no_grad()
def check(uncut, cut_actor, critic, n=16, decisions=48, seed=0, device="cpu"):
    """`n` simulator battles (random armies, up to 19 units a side) played by the uncut network against ai_like for
    `decisions` decisions; both networks on them (compare())."""
    from tools.nn.armies import generate
    from tools.nn.train import league, ppo, randomise, rollout, scenes
    params = rollout.params_with_limit(3600.0)
    seeds = range(generate.TRAIN_SEEDS.start + 7919 * seed, generate.TRAIN_SEEDS.start + 7919 * seed + n)
    src = scenes.Generated(seeds, 19, params, device, seed=seed, sequential=True)
    lay = league.layout(n, len(scenes.SCENES), {"ai_like": 1.0}, scene_attacker=scenes.attackers())
    env = rollout.Battles(lay, scenes.SCENES, device, params, randomise.Spread(), seed=seed, source=src)
    torch.manual_seed(seed)
    batch = rollout.collect(env, uncut, critic, decisions)
    obs = rollout.full_obs(batch["obs"], batch["abil_static"])
    idx = torch.arange(batch["reward"].shape[1], device=batch["reward"].device)
    act = ppo.chunk_action(batch["action"], idx)
    la, _ = uncut.sequence(obs, batch["h0"], batch["reset"], act)
    lb, _ = cut_actor.sequence(obs, batch["h0"], batch["reset"], act)
    flat = lambda d: {k: v.reshape(-1, *v.shape[2:]) for k, v in d.items()}
    fa = hd.Action(**{f: getattr(act, f).reshape(-1, *getattr(act, f).shape[2:]) for f in act.parts()})
    return compare(flat(la), flat(lb), fa, flat(obs))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="a v2 checkpoint whose eyes feed back")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--check-battles", type=int, default=16, help="simulator battles of the check (0: none)")
    ap.add_argument("--check-decisions", type=int, default=48)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from tools.nn.train import checkpoint
    data, out, note = cut_file(args.src, args.dst)
    n = model_policy.parameters
    old, new = model_policy.Actor(checkpoint.config_of(data)), model_policy.Actor(checkpoint.config_of(out))
    print(f"{args.src} -> {args.dst}: actor {n(old):,} -> {n(new):,} parameters; {note}", flush=True)
    if args.check_battles and data.get("critic") is not None:
        uncut = checkpoint.load_policy(args.src, args.device)
        cut_actor = checkpoint.load_policy(args.dst, args.device)
        critic = checkpoint.load_critic(args.src, args.device)
        res = check(uncut, cut_actor, critic, args.check_battles, args.check_decisions, device=args.device)
        print(f"   check: {args.check_battles} simulator battles x {args.check_decisions} decisions against ai_like",
              flush=True)
        for h, r in res.items():
            if h == "any":
                print(f"   any head flipped: {r['flip']:.4f} of {r['rows']} deciding unit-decisions", flush=True)
            else:
                print(f"   {h:8} KL {r['kl']:.5f}  probability moved {r['tv']:.4f}  top choice changed {r['flip']:.4f}"
                      f"  ({r['rows']} rows)", flush=True)


if __name__ == "__main__":
    main()
