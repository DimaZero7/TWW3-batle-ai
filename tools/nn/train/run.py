"""Train the battle network with PPO in the simulator (docs/en/training/training.md).

    bash tools/nn/dock.sh tools.nn.train.run --minutes 15 --name r1          # in the container, on the GPU
    bash tools/nn/dock.sh tools.nn.train.run --minutes 60 --name r2 --init build/nn-train/runs/r1/best.pt

The network starts from build/nn-train/random.pt (written if missing) or --init, plays `--battles`
battles at once against itself, its past versions and the scripted opponents, and learns after
every `--steps` decisions. The networks decide as in the game (tools/nn/train/cadence.py: every
--decide-s 1 s of battle, the orders --order-latency 0.36 s late; the simulator's step stays 0.5 s);
--gamma (and GAE's lambda) are per 0.5 s of battle and become per decision here, the horizon in
seconds the same. Writes (build/ is not in Git), in build/nn-train/runs/<name>/:

    latest.pt, best.pt     the network (tools/nn/train/checkpoint.py); best: the best window of
                           training battles against the scripts (the worst opponent-role counts)
    pool/v*.pt, pool/q.json  past versions, the opponents of self-play, and their quality scores (league.Pool;
                           --pool-dir: a pool folder of its own, shared by the parts of a night)
    log.jsonl              one line per update: losses, entropy, reward, win rates by opponent and role
    eval.json              the final evaluation (tools/nn/train/evaluate.py)
    replays/*/             battles of the final network, like the game's recordings
and copies the final network to build/nn-train/latest.pt (the companion's default).
"""
import argparse
import contextlib
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools.nn.general import signal as gsignal
from tools.nn.general.config import MASK_PARTS, MISSION_KINDS
from tools.nn.model import config as model_config
from tools.nn.model import critic as model_critic
from tools.nn.model import policy as model_policy
from tools.nn.train import cadence as cad
from tools.nn.train import checkpoint, drills, evaluate, league, matchups, ppo, randomise, reward, rollout, scenes
from tools.nn.train import ability_branch, bankproc, head_entropy, head_reset, kitesp, teach_auto
from tools.nn.train import dr as uncertain
from tools.nn.train.drills import teach as drill_teach

SCRIPTED = ("nearest", "hold_shoot", "hold", "ai_like")


def compatible(path, preset):
    """The checkpoint exists and matches the current code's network (actor and critic)."""
    try:
        data = checkpoint.read(path)
        cfg = checkpoint.config_of(data)
        checkpoint.actor_of(data).load_state_dict(data["actor"])
        model_critic.Critic(cfg).load(data["critic"])
        # (a v2 network of an older v2 config, e.g. without the eyes, is not the preset's)
        return data.get("preset") == preset and (not cfg.sectors or cfg == model_config.PRESETS[preset])
    except (FileNotFoundError, ValueError, RuntimeError, KeyError):
        return False


def networks(preset, device, start=None, critic_from=None):
    """(actor, critic) of `start` (default: random.pt, written if missing). The critic comes from
    critic_from when given (e.g. the run's latest.pt beside a test5 m<minute>.pt saved without it); a
    checkpoint without one starts a fresh critic (then give --critic-warmup updates)."""
    if start is None:
        start = checkpoint.RANDOM_V2 if model_config.PRESETS[preset].sectors else checkpoint.RANDOM
        if not compatible(start, preset):
            checkpoint.write_random(start, preset)
    data = checkpoint.read(start)
    cfg = checkpoint.config_of(data)
    actor, critic = checkpoint.actor_of(data), model_critic.Critic(cfg)
    actor.load_state_dict(data["actor"])
    state = checkpoint.read(critic_from)["critic"] if critic_from else data.get("critic")
    if state is None:
        print(f"{start}: no critic in the checkpoint, a fresh one (give it --critic-warmup updates)", flush=True)
    else:
        critic.load(state)
    return actor.to(device).eval(), critic.to(device).eval()


def with_grid(actor, grid):
    """The actor with the grid move-point head (ModelConfig.grid, tools/nn/model/heads.py; 0: as it is): an actor
    of the bin head keeps every weight but the point head's, the grid head starts fresh."""
    if not grid or actor.cfg.grid == grid:
        return actor
    device = next(actor.parameters()).device
    new = model_policy.Actor(dataclasses.replace(actor.cfg, grid=grid))
    new.load_state_dict({k: v for k, v in actor.state_dict().items() if not k.startswith("heads.point.")})
    return new.to(device).eval()


def sized(cfg, N, slots=22, width=1.0):
    """The PPO settings for battles of N slots: fewer decisions per minibatch for bigger battles, so a
    minibatch holds about as many unit tokens as at 22 slots (the memory of a 16 GB card). A wider
    network (width: its token width / the small one's 128) keeps the minibatch and computes it in
    ceil(width) parts (gradient accumulation): the widened network (x 2) with the small one's minibatch
    did not fit in 16 GB, and a minibatch halved instead (03.10) doubled the optimizer's steps per update:
    Adam's steps are about lr each, so the policy moved ~2 times as far and the KL (quadratic in the
    step) ~4 times (the first update's KL over the whole batch 0.033 against the small network's 0.0073
    on the same batch; with the parts 0.010) - the wide run w1 collapsed within 5 minutes."""
    return dataclasses.replace(cfg, minibatch=max(256, int(cfg.minibatch * min(1.0, slots / N))),
                               accum=max(1, math.ceil(width - 1e-9)))


def optimizer(actor, critic, lr, width=1.0, ability_lr=1.0, general_lr=1.0, kind_lr=1.0):
    """Adam over the actor and the critic. In a widened network (width > 1, tools/nn/model/widen.py) the
    weights that read the copied token stream (widen.stream_readers: fan-in x width) take lr / width:
    every copy gets the gradient the small weight got and Adam moves each about lr, so their sum (what
    the network computes) moved width times as far as in the small network (as muP's lr ~ 1 / fan-in).
    ability_lr (--ability-lr-mult): the ability head's parameters (ppo.ABILITY_HEAD) take lr x it - Adam moves each
    weight about lr whatever the loss's weight, so a heavier ability term barely speeds the head; this does. Not 1:
    the parameters in groups by runs of the same lr, in the actor's then the critic's order (the order a one-group
    optimizer has: resume maps the moments by it, both ways); 1: the one group as before. general_lr (--general-lr-mult):
    the general's network's parameters (tools/nn/general/executor.general_params) take lr x it, the same way; kind_lr
    (--general-kind-lr-mult under --general-kind-own-signal 1): the general's kind head (learn.kind_head_params) lr x
    general_lr x it."""
    params = list(actor.parameters()) + list(critic.parameters())
    if ability_lr != 1.0 or general_lr != 1.0 or kind_lr != 1.0:
        if width > 1.0:
            raise SystemExit("--ability-lr-mult / --general-lr-mult are for an unwidened network (v2)")
        head = {id(p) for p in ppo.ability_head_params(actor)}
        from tools.nn.general.executor import general_params
        gen = {id(p) for p in general_params(actor)}
        kind = set()
        if kind_lr != 1.0:
            from tools.nn.general.learn import kind_head_params
            kind = {id(p) for p in kind_head_params(actor.general)}
        groups = []
        for p in params:
            plr = (lr * ability_lr if id(p) in head else lr * general_lr * kind_lr if id(p) in kind
                   else lr * general_lr if id(p) in gen else lr)
            if not groups or groups[-1]["lr"] != plr:
                groups.append({"params": [], "lr": plr})
            groups[-1]["params"].append(p)
        return torch.optim.Adam(groups, lr=lr, eps=1e-5)
    if width <= 1.0:
        return torch.optim.Adam(params, lr=lr, eps=1e-5)
    from tools.nn.model import widen
    readers, rest = [], []
    for model, spec in ((actor, widen.ACTOR), (critic, widen.CRITIC)):
        names = widen.stream_readers(model, spec)
        for n, p in model.named_parameters():
            (readers if n in names else rest).append(p)
    return torch.optim.Adam([{"params": readers, "lr": lr / width}, {"params": rest, "lr": lr}], lr=lr, eps=1e-5)


def resume(opt, path, fresh=()):
    """Adam's state from the checkpoint `path` (its "train", checkpoint.py) into opt, when it has one of the same
    parameters (count and shapes, in order, besides `fresh`); the learning rates stay opt's (the options'). fresh: the
    parameters the checkpoint has none of (an actor's parts added since it was saved, fresh_params): they start
    Adam afresh. A first layer whose inputs grew (the token's columns: observation.old_columns; the ability features
    appended at the end: encoder.pad_inputs) gets its moments at the columns' places, zero for the new ones. -> the
    trainer's counters saved with it ({} if none), or None when nothing was resumed (why: printed)."""
    tr = checkpoint.train_state(path) if path else None
    if not tr or "optim" not in tr:
        return None
    saved = tr["optim"]
    counters = dict(tr.get("state") or {})
    skip = {id(p) for p in fresh}
    # the parameters in the groups' order, matched one by one whatever the groups (a one-group optimizer's state into
    # one with the ability head's group, --ability-lr-mult, and back: optimizer() keeps the order)
    saved_ids = [i for g in saved["param_groups"] for i in g["params"]]
    mine = [p for g in opt.param_groups for p in g["params"] if id(p) not in skip]
    if len(saved_ids) != len(mine):
        # (the trainer's counters go on all the same: the chain's updates keep the observation term off, its seconds
        # the entropy decay)
        print(f"{path}: the optimizer's state is of other parameters: Adam starts afresh, the counters go on", flush=True)
        return counters
    from tools.nn.model import observation as ob
    pairs = list(zip(saved_ids, mine))
    for i, p in pairs:
        st = saved["state"].get(i, {})
        if "exp_avg" in st and tuple(st["exp_avg"].shape) != tuple(p.shape):
            old = st["exp_avg"]
            if old.dim() == 2 and p.dim() == 2 and old.shape[0] == p.shape[0] and old.shape[1] < p.shape[1]:
                # the token encoder's first layer of a checkpoint from before the new units (observation.old_columns):
                # its moments to the columns' new places; other inputs appended at the end (encoder.pad_inputs); zero
                # for the new columns (as its weights, encoder.py)
                cols = ob.old_columns(old.shape[1]) if p.shape[1] == ob.TOKEN else None
                cols = list(range(old.shape[1])) if cols is None else cols
                for k in ("exp_avg", "exp_avg_sq"):
                    new = st[k].new_zeros(p.shape)
                    new[:, cols] = st[k]
                    st[k] = new
                continue
            print(f"{path}: the optimizer's state is of other shapes: Adam starts afresh, the counters go on", flush=True)
            return counters
    # the saved state renumbered as opt numbers its parameters (the fresh ones without a state)
    number = {id(p): n for n, p in enumerate(p for g in opt.param_groups for p in g["params"])}
    state = {number[id(p)]: saved["state"][i] for i, p in pairs if i in saved["state"]}
    # (the groups' options are opt's own: the same Adam, its lr per group restored below)
    param_groups = [dict({k: v for k, v in g.items() if k != "params"}, params=[number[id(p)] for p in g["params"]])
                    for g in opt.param_groups]
    lrs = [g["lr"] for g in opt.param_groups]
    opt.load_state_dict({"state": state, "param_groups": param_groups})
    for g, lr in zip(opt.param_groups, lrs):
        g["lr"] = lr
    return counters


def fresh_params(model, path):
    """The parameters of `model` (an actor) that the checkpoint's actor has none of (resume's fresh)."""
    have = set(checkpoint.read(path)["actor"])
    return [p for n, p in model.named_parameters() if n not in have]


def same_file(a, b):
    return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()


@torch.no_grad()
def follow(reference, actor, share):
    """The reference moves `share` of the way to the actor (Polyak averaging of the weights: --anchor-ema)."""
    for r, a in zip(reference.state_dict().values(), actor.state_dict().values()):
        if r.is_floating_point():
            r.lerp_(a, share)
        else:
            r.copy_(a)


def schedule(start, end, share):
    """Linear from start (share 0) to end (share 1)."""
    share = min(1.0, max(0.0, share))
    return start + (end - start) * share


def decay_share(minutes, seconds, run_share):
    """The entropy schedule's share done: --entropy-decay's seconds over its minutes (carried over a chain's parts),
    else the run's own share."""
    return seconds / (minutes * 60) if minutes else run_share


def pinned_matchups(stats):
    """{"pinned/<role>/<own>-<enemy>": {games, wins, win_rate, gold_trade}} of (games, wins, mean gold trade) counts
    (rollout.Battles.take_pin_stats; run.py --past-pin)."""
    return {k: {"games": g, "wins": w, "win_rate": round(w / g, 3), "gold_trade": round(t, 4)}
            for k, (g, w, t) in sorted(stats.items()) if g}


def rates(stats):
    return {k: round(w / g, 3) for k, (g, w, _) in stats.items() if g}


def score(window):
    """The worst win rate against the scripts over (opponent, role) with enough battles; None if too few."""
    rs = [w / g for k, (g, w, _) in window.items() if k.split("/")[0] in SCRIPTED and g >= 20]
    keys = [k for k, (g, _, _) in window.items() if k.split("/")[0] in SCRIPTED and g >= 20]
    return (min(rs), keys) if len(rs) >= 5 else (None, keys)


def normal_drills(spec):
    """--teach-normal -> ({drill: fixed share} or None for adaptive, [drill names]): "auto" every drills.TRAIN
    drill with moments; "kiting,hold_fire" those, adaptive; json {"kiting": 0.1} fixed shares; "" none."""
    if not spec:
        return None, []
    if spec.strip().startswith("{"):
        fixed = {k: float(v) for k, v in json.loads(spec).items()}
        return fixed, list(fixed)
    names = [n for n in drills.TRAIN] if spec == "auto" else [n.strip() for n in spec.split(",") if n.strip()]
    return None, names


def eval_drills(args):
    """drills.set_extra: the drills this run trains (--drill-weights above 0) or teaches in normal battles
    (--teach-normal) are evaluated beside drills.READY (their drill block, transfer, references) -> the extra ones."""
    names = []
    if args.drills and args.drill_weights:
        names += [n for n, w in json.loads(args.drill_weights).items() if float(w) > 0]
    names += normal_drills(args.teach_normal)[1]
    return drills.set_extra(names)


# --reward v2 (the base v2's reward, 08.10.2026): the battle's end (win / loss), the gold trade and the order change
# cost (keep and a repeated order free); no lord term, no idle cost, no retarget cost
REWARD_PRESETS = {"v2": {"order_cost": 0.006, "lord": 0.0, "idle": 0.0, "retarget": 0.0}}


def reward_preset(args):
    """--reward NAME: its values replace the options' (in place)."""
    for k, v in REWARD_PRESETS.get(args.reward or "", {}).items():
        setattr(args, k, v)
    return args


class Parser(argparse.ArgumentParser):
    """An ArgumentParser that records which options the command line gave: args.cli_given, their dests (sorted list).
    A setting saved in --init yields only to an option given there, never to an option's default (general_config)."""

    def parse_known_args(self, args=None, namespace=None):
        ns, rest = super().parse_known_args(args, namespace)
        saved = [(a, a.default) for a in self._actions]
        try:                                     # the same line again with no defaults: only the given dests are set
            for a, _ in saved:
                a.default = argparse.SUPPRESS
            given, _ = super().parse_known_args(args, argparse.Namespace())
        finally:
            for a, d in saved:
                a.default = d
        ns.cli_given = sorted(vars(given))
        return ns, rest


# GeneralConfig field -> its run.py option's dest
GENERAL_FLAGS = {"card": "general_card", "reader_hidden": "general_hidden", "council_layers": "general_council_layers",
                 "council_heads": "general_heads", "group_layers": "general_group_layers", "places": "general_places",
                 "kinds": "general_kinds", "target_grid": "general_target_grid", "rounds": "general_rounds",
                 "center": "general_center", "sharp": "general_sharp", "stick": "general_stick",
                 "kind_stick": "general_kind_stick", "task_mask": "general_task_mask",
                 "task_mask_parts": "general_task_mask_parts",
                 "shoot_share": "general_shoot_share", "melee_stay": "general_melee_stay",
                 "waver_share": "general_waver_share", "shoot_line_m": "general_shoot_line_m",
                 "tiles": "general_tiles", "effects": "general_effects", "zone": "general_zone",
                 "deploy": "general_deploy", "replan_s": "general_replan_s", "replan_min_s": "general_replan_min_s",
                 "replan_near_m": "general_replan_near_m", "teach_stall_s": "general_teach_stall_s",
                 "teach_wait_s": "general_teach_wait_s",
                 "groups": "general_groups", "group_r": "general_group_r", "lord_join_m": "general_lord_join_m",
                 "group_centre": "general_group_centre",
                 "split_kinds": "general_split_kinds", "tile_feats": "general_tile_feats",
                 "tile_norm": "general_tile_norm", "threat": "general_threat", "task_group": "general_task_group",
                 "group_tiles": "general_group_tiles"}
# the fields that make the network's shape (the saved weights fit only them; center: 0 has no weight, any other value
# is only the learned share's start - the saved weight is the share): an option cannot change them on an --init's general
GENERAL_SHAPE = ("card", "reader_hidden", "council_layers", "council_heads", "group_layers", "places", "kinds",
                 "target_grid", "center", "sharp", "stick", "tiles", "effects")


def general_given(args):
    """The dests of the general's options given on the command line (args.cli_given of Parser); a Namespace made
    without it: the options that differ from the parser's defaults."""
    if hasattr(args, "cli_given"):
        return set(args.cli_given) & set(GENERAL_FLAGS.values())
    ap = parser()
    return {d for d in GENERAL_FLAGS.values() if getattr(args, d) != ap.get_default(d)}


def general_config(args, saved=None):
    """(GeneralConfig of the run, overrides {field: (saved, new)}). Without saved (a fresh general): of the options.
    With saved (--init's general): its sizes and rules, the rules of the options given on the command line over them;
    an option given against a shape field (GENERAL_SHAPE) stops the run - the saved weights fit only their shape."""
    from tools.nn.general import config as gconfig
    if saved is None:
        return gconfig.GeneralConfig(**{f: getattr(args, d) for f, d in GENERAL_FLAGS.items()}), {}
    given = general_given(args)
    over = {f: (getattr(saved, f), getattr(args, d)) for f, d in GENERAL_FLAGS.items()
            if d in given and getattr(args, d) != getattr(saved, f)}
    shape = [f for f in over if f in GENERAL_SHAPE]
    if shape:
        raise SystemExit("the --init's general has its own shape; these options would change it (its weights would not "
                         "fit): " + ", ".join(f"--{GENERAL_FLAGS[f].replace('_', '-')} {over[f][1]} (saved {over[f][0]})"
                                              for f in shape))
    return dataclasses.replace(saved, **{f: new for f, (_, new) in over.items()}), over


def with_general(actor, args):
    """(actor, the general's LearnConfig or None): --general wraps the unit network into the executor with a general
    (tools/nn/general/executor.py: the old weights, the task input at zero, a fresh general; an --init that has one
    keeps it, its rules yield to the options given on the command line: general_config); without --general an
    executor still learns its general when it has one."""
    from tools.nn.general import executor, learn as glearn
    if not getattr(args, "general", False) and not executor.has_task(actor):
        return actor, None
    if not executor.has_task(actor):
        if not actor.cfg.sectors:
            raise SystemExit("--general needs a v2 unit network (ModelConfig.sectors > 0)")
        actor = executor.wrap(actor, general_config(args)[0])
        n = sum(p.numel() for p in executor.general_params(actor))
        print(f"general: a fresh general on the unit network ({n} weights; the task input at zero: the executor plays "
              f"as the unit network did)", flush=True)
    else:
        saved = actor.gcfg
        gcfg, over = general_config(args, saved)
        actor.set_rules(gcfg)
        print(f"general: from the checkpoint {saved}; overridden by flags: "
              + (", ".join(f"{f} {a!r} -> {b!r}" for f, (a, b) in over.items()) or "none"), flush=True)
    cfg = glearn.LearnConfig(entropy_kind=args.general_entropy_kind, entropy_target=args.general_entropy_target,
                             entropy_place=args.general_entropy_place, epochs=args.general_epochs, value=args.general_value,
                             minibatch=args.general_minibatch, branches=args.general_branches,
                             branch_weight=args.general_branch_weight,
                             distinct=args.general_distinct, distinct_decay=args.general_distinct_decay,
                             exec=args.general_exec, exec_decay=args.general_exec_decay, teacher=args.general_teacher,
                             imitate=args.general_imitate, imitate_decay=args.general_imitate_decay,
                             imitate_min=args.general_imitate_min,
                             exec_general=args.general_exec_general, exec_kind=args.general_exec_kind,
                             teacher_hold_m=args.general_teacher_hold_m, teacher_fire_s=args.general_teacher_fire_s,
                             teacher_flank=bool(args.general_teacher_flank), flank_measure=args.general_flank_measure,
                             kl=args.general_kl, kind_own=bool(args.general_kind_own_signal),
                             kind_weight=args.general_kind_weight, kind_epochs=args.general_kind_epochs,
                             kind_q=args.general_kind_q)
    print(f"general: {actor.gcfg}; learning {cfg}; return {args.general_return}; lr x {args.general_lr_mult:g}",
          flush=True)
    return actor, cfg


def general_only_args(args):
    """--general-only 1: (a copy of the options without what trains only the executor, [what was turned off]) - the
    ability branches (their replays, the ability critic, the ability head's own signal and its extra passes: the
    executor's ability head; their saved state goes through to the run's checkpoints as --init had it), the
    demonstrations (--observe-dir: an executor's term) and the critic's warm-up (the critic is frozen too). The caller's
    options stay as they were."""
    off, names = {}, []
    if args.ability_branch:
        off.update(ability_branch=0.0, ability_critic=False, ability_own_signal=False, ability_epochs=1,
                   ability_price="auto")
        names.append("--ability-branch (its critic, own signal and epochs)")
    if args.observe_dir:
        off["observe_dir"] = None
        names.append("--observe-dir")
    if args.critic_warmup:
        off["critic_warmup"] = 0
        names.append("--critic-warmup")
    return argparse.Namespace(**{**vars(args), **off}), names


# the ability branches' state in a checkpoint's "train" (blob): under --general-only it goes through as --init had it
ABILITY_STATE = ("ability_critic", "ability_critic_openloop", "ability_critic_kind", "ability_norm")


def general_brancher(args, glearn_cfg, device):
    """The mission branches (tools/nn/general/branch.py) of --general-branches P > 0, else None (nothing changes)."""
    if glearn_cfg is None or not glearn_cfg.branches:
        return None
    from tools.nn.general import branch as gbranch
    return gbranch.Brancher(gbranch.Config(
        share=glearn_cfg.branches, copies=args.general_branch_copies, horizon_s=args.general_branch_horizon,
        cap=args.general_branch_cap, batch=args.general_branch_batch, alt=args.general_branch_alt,
        price=args.general_branch_price, greedy=args.general_branch_greedy, kinds=args.general_branch_kinds,
        openloop=bool(args.general_branch_openloop),
        critic=bool(args.general_kind_critic) and glearn_cfg.kind_q == "branch",     # (its estimate used only there)
        critic_warmup=args.general_kind_critic_warmup, critic_buffer=args.general_kind_critic_buffer,
        norm=args.general_kind_norm, center=bool(args.general_branch_center), seed=args.seed), device)


def general_outcome_q(args, glearn_cfg, gcfg, device, decision_s=1.0):
    """The outcome Q of the mission kinds (tools/nn/general/kind_q.py) of --general-kind-q outcome|both, else None;
    decision_s: the battles' decision step (the features' wait in chunks: kind_q.wait_chunks)."""
    if glearn_cfg is None or glearn_cfg.kind_q == "branch":
        return None
    from tools.nn.general import kind_q as gkq
    return gkq.OutcomeQ(gcfg.kinds, gcfg.card, device, buffer=args.general_kind_q_buffer,
                        warmup=args.general_kind_q_warmup, ev_min=args.general_kind_q_ev_min,
                        norm=args.general_kind_norm, shrink=bool(args.general_kind_q_shrink), seed=args.seed,
                        wait=gkq.wait_chunks(args.general_kind_q_horizon, decision_s, args.steps))


def imitate_floor(args, cfg, carried):
    """The general's LearnConfig with the imitation's floor (--general-imitate-min) of the run: the --init's saved one
    (its train state "general_imitate_min") unless the command line gives the option."""
    given = getattr(args, "cli_given", None)
    saved = (carried or {}).get("general_imitate_min")
    if saved is None or (given is not None and "general_imitate_min" in given):
        return cfg
    if given is None and args.general_imitate_min != parser().get_default("general_imitate_min"):
        return cfg                                 # (a Namespace without cli_given: an option off its default is given)
    return dataclasses.replace(cfg, imitate_min=float(saved))


MISSION_NAMES = {k.name for k in MISSION_KINDS}


def general_text(g):
    """One log line of the general's stats (row["general"])."""
    kinds = " ".join(f"{x:.2f}" for x in g.get("kinds", []))
    return (f"general: plans {g.get('plans', 0)} (deploy {g.get('deploy_plans', 0)}, by net {g.get('deploy_net_share', 0):.2f}), "
            f"groups own {g.get('groups_own', 0):.1f} enemy {g.get('groups_enemy', 0):.1f}, replans/battle "
            f"{g.get('replans_per_battle', 0):.1f} ({g.get('replans_per_min', 0):.2f} a minute of "
            f"{g.get('battle_min', 0):.1f} min battles) {g.get('replan_reasons', {})}; kinds [{kinds}] entropy "
            f"{g.get('kind_entropy', 0):.2f}/{g.get('kind_entropy_max', 0):.2f}; places {g.get('places_top', {})}; "
            f"targets: groups {g.get('target_group', 0):.2f} (own {g.get('target_own', 0):.2f} enemy "
            f"{g.get('target_enemy', 0):.2f}) tiles {g.get('target_tile', 0):.2f}; re-plans: group churn "
            f"{g.get('group_churn', 0):.3f} kind switch {g.get('kind_switch', 0):.3f}; "
            f"pl {g.get('pl', 0):+.3f} ent {g.get('ent', 0):.2f} (kind {g.get('ent_kind', 0):.2f} target "
            f"{g.get('ent_target', 0):.2f} place {g.get('ent_place', 0):.2f}) kl {g.get('kl', 0):.4f} (plans' own "
            f"{g.get('kl_plan', 0):.4f}) clip {g.get('clip', 0):.2f} (fresh {g.get('clip_fresh', 0):.2f} old "
            f"{g.get('clip_old', 0):.2f}, old weight {g.get('stale_w', 1):.2f}) "
            f"value mse {g.get('value_mse', 0):.5f} ev {g.get('value_ev', 0):+.2f} return {g.get('return_mean', 0):+.4f}"
            + (f" critic ev {g['critic_ev']:+.2f}" if "critic_ev" in g else "")
            + f"; adv {g.get('adv_mean', 0):+.4f} sd {g.get('adv_std', 0):.4f}; task_in |W| {g.get('task_in_w', 0):.3f} "
            f"(group {g.get('task_in_group', 0):.3f} kind {g.get('task_in_kind', 0):.3f} geo {g.get('task_in_geo', 0):.3f})"
            + (f"; branches {g['branch_branches']} of {g.get('branch_plans', 0)} plans, price "
               f"{g.get('branch_price_mean', 0):+.4f} sd {g.get('branch_price_std', 0):.4f} (real share "
               f"{g.get('branch_price_rel', float('nan')):.2f}), chosen better {g.get('branch_better', 0):.2f} worse "
               f"{g.get('branch_worse', 0):.2f}, {g.get('branch_seconds', 0):.1f} s, term kl {g.get('branch_kl', 0):.4f}"
               + (f", vs teacher {g.get('branch_vs_teacher_n', 0)} (fallback {g['branch_alt_fallback']:.2f}) better "
                  f"{g.get('branch_vs_teacher_better', 0):.2f} worse {g.get('branch_vs_teacher_worse', 0):.2f} price "
                  f"{g.get('branch_vs_teacher_price', 0):+.4f}" if "branch_alt_fallback" in g else "")
               + (f" (raw {g['branch_price_raw_mean']:+.4f}, centre {g['branch_center']:+.4f})"
                  if "branch_center" in g else "")
               if "branch_branches" in g else "")
            + (f"; kind Q {g.get('kind_q_samples', 0)} groups on {g.get('kind_q_on', 0)} ev "
               f"{g.get('kind_q_ev', float('nan')):.3f} (kind part {g.get('kind_q_ev_kind', float('nan')):+.4f}, coef "
               f"{g.get('kind_q_coef', float('nan')):+.2f} se {g.get('kind_q_coef_se', float('nan')):.2f}, shrink "
               f"{g.get('kind_q_shrink', float('nan')):.2f}), +{g.get('kind_q_matured', 0)} plans (ended "
               f"{g.get('kind_q_ended_share', 0):.2f}, lost {g.get('kind_q_lost', 0)}), agrees with the branches "
               f"{g.get('kind_q_agree', float('nan')):+.3f}" if "kind_q_samples" in g else "")
            + (f"; distinct acc {g.get('distinct_acc', 0):.3f} (chance {g.get('distinct_chance', 0):.2f}) gain "
               f"{g.get('distinct_gain', 0):+.3f} bonus {g.get('distinct_bonus', 0):+.4f} w {g.get('distinct_weight', 0):.5f} "
               f"scale {g.get('distinct_scale', 0):.4f}"
               if "distinct_weight" in g else "")
            + ("; missions done " + " ".join(f"{k[5:]} {v:.2f}" for k, v in g.items()
                                             if k.startswith("exec_") and k[5:] in MISSION_NAMES)
               + f" (w {g.get('exec_weight', 0):g}, bonus |{g.get('exec_bonus_abs', 0):.3f}| a unit-decision, adv sd "
               f"{g.get('exec_adv_std', 0):.3f}), masked pairs {g.get('masked_pairs', 0):.2f}"
               + (f", teacher {g['teach_share']:.2f}" if "teach_share" in g else "")
               if "exec_weight" in g else "")
            + (f"; task masks {g['mask_share']:.2f} of the units deciding, fall-back {g.get('mask_fallback', 0):.4f} "
               f"({g.get('mask_fallback_n', 0)}), volley lock {g.get('volley_lock', 0):.3f}, shooters firing "
               f"{g.get('shooters_firing', 0):.2f}" if "mask_share" in g else "")
            + (f"; shoot without a shooter {g.get('shoot_noshooter', 0):.3f} of the groups, "
               f"{g.get('shoot_units_nomissile', 0):.2f} of its units without missiles" if "shoot_noshooter" in g else "")
            + (f"; imitate nll {g.get('imitate_nll', float('nan')):.3f} acc kind "
               f"{g.get('imitate_acc_kind', float('nan')):.2f} target {g.get('imitate_acc_target', float('nan')):.2f} "
               f"w {g['imitate_w']:.4f}" if "imitate_w" in g else "")
            + (f"; exec general |{g['exec_general_term']:.4f}| ({g.get('exec_general_mean', 0):+.4f}) adv sd"
               if "exec_general_term" in g else ""))


def fighters_installed(args):
    """--fighters: the fighters lab in the simulator for the run (tools/nn/train/men.py: the hook installed, nothing
    built adopted, the units' shape for the network); else nothing."""
    if not getattr(args, "fighters", False):
        return contextlib.nullcontext()
    from tools.nn.fighters import hook as fighters_hook
    # shape "lazy": the units' shape only before an observation (every decision), not after every simulator step
    return fighters_hook.installed(seed=args.seed, shape="lazy", adopt=False, outside=False)


def fighters_men(args, M="args"):
    """--fighters: a new men.Men for a batch (M: places a battle; "args": --fighters-men, None: the batch's own -
    evaluation); else None."""
    if not getattr(args, "fighters", False):
        return None
    from tools.nn.train import men
    return men.Men(M=args.fighters_men if M == "args" else M, seed=args.seed)


def cuda_alloc(mode, device):
    """--cuda-alloc: the caching allocator's way of holding video memory (CUDA only). "expandable" (default): segments
    that grow in place (PyTorch's expandable_segments) instead of fixed blocks; "default": PyTorch's own. With fixed
    blocks the cache held 15.5-16.7 GB for a 12.7 GB peak (line S with fighters): past the card, into shared memory
    (start_kl's pass 3.2 s instead of 0.2, the collection ~10 % slower); expandable: 13.1 GB."""
    if mode == "default" or torch.device(device).type != "cuda":
        return
    setting = getattr(torch._C, "_accelerator_setAllocatorSettings", None) or torch.cuda.memory._set_allocator_settings
    setting("expandable_segments:True")


def train(args, every=None, teacher=None, normal=None):
    """_train with the fighters lab installed for it (--fighters)."""
    with fighters_installed(args):
        return _train(args, every, teacher, normal)


def _train(args, every=None, teacher=None, normal=None):
    """every: (minutes, hook) - hook(actor, critic, minute, update) after every `minutes` of training
    (its time not counted as training: e.g. a full evaluation). teacher: with --drill-teach auto, the
    adaptive teacher (teach_auto.Auto; default: one with the init checkpoint's last drill numbers,
    teach_auto.prior) - the hook may change its shares (Auto.observe), read before every update. normal: with
    an adaptive --teach-normal, the teacher in normal battles (teach_auto.Transfer; default: one with the init
    checkpoint's last transfer numbers), likewise."""
    # --general-only 1: only the general learns; the executor (the unit network inside the network) and the critic are
    # frozen (executor.freeze_but_general), their own terms are not computed (ppo.update's executor False)
    general_only = bool(getattr(args, "general_only", 0))
    turned_off = []
    if general_only:
        args, turned_off = general_only_args(args)
    if args.ability_critic and not args.ability_branch:
        raise SystemExit("--ability-critic needs --ability-branch")
    own = bool(getattr(args, "ability_own_signal", False))
    if own and not args.ability_branch:
        raise SystemExit("--ability-own-signal needs --ability-branch (the branches price the presses)")
    body_grad, own_w = getattr(args, "ability_body_grad", None), getattr(args, "ability_weight", None)
    ab_body = (body_grad if body_grad is not None else 0.1) if own else 1.0
    ab_own_w = own_w if own_w is not None else 1.0
    price = getattr(args, "ability_price", "auto")
    ab_bonus = price == "bonus" or (price == "auto" and own)
    if ab_bonus and not args.ability_branch:
        raise SystemExit("--ability-price bonus needs --ability-branch")
    ab_lr, ab_epochs = float(getattr(args, "ability_lr_mult", 1.0)), int(getattr(args, "ability_epochs", 1))
    if ab_epochs > 1 and not args.ability_branch:
        raise SystemExit("--ability-epochs needs --ability-branch (the extra passes are of the ability term)")
    device = torch.device(args.device)
    cuda_alloc(getattr(args, "cuda_alloc", "default"), device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    drills.BROAD = args.drill_broad              # the drills' broad frames: their share of every drill's battles
    drills.EMBED = args.drill_embed              # the embedded frames: their share (of the drills that have one)
    eval_drills(args)                            # the run's own drills beside READY (the hook's evaluations)
    reward_preset(args)
    out = checkpoint.DIR / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    if args.preset is None:          # the record follows the network: --init's preset (e.g. "wide"), else small
        args.preset = checkpoint.read(args.init).get("preset", "small") if args.init else "small"
    actor, critic = networks(args.preset, device, args.init, args.critic_init)
    actor = with_grid(actor, args.grid)
    actor, glearn_cfg = with_general(actor, args)
    if general_only and glearn_cfg is None:
        raise SystemExit("--general-only 1 needs a general to learn: --general (or an --init with a general)")
    # --reset-heads: those heads start again as a new network's, the rest as loaded (head_reset.py); the KL anchors'
    # frozen networks below get the same fresh heads, Adam's moments of them are dropped
    reset_params = head_reset.reset(actor, head_reset.parse(args.reset_heads), args.seed)
    if reset_params:
        print(f"--reset-heads {args.reset_heads}: {len(reset_params)} tensors start afresh "
              f"({sum(dict(actor.named_parameters())[n].numel() for n in reset_params)} weights): "
              + ", ".join(sorted({n.split('.')[1] for n in reset_params})), flush=True)
    head_targets = head_entropy.parse(args.head_entropy)
    reference = None
    if args.anchor:
        # the grid's reference too (its fresh head is never compared: the anchor's KL is on the kind and target),
        # so --anchor-roll can copy the learner into it
        reference = with_grid(checkpoint.load_policy(args.reference or args.init, device), args.grid)
        for p in reference.parameters():
            p.requires_grad_(False)
        head_reset.copy(actor, reference, set(reset_params))
    # The network the run started from, frozen: the distance from the start (ppo.distance, start_kl in
    # the log) next to the rating, whatever the reference does (--anchor-roll moves it).
    start = None
    if args.init:
        start = checkpoint.load_policy(args.init, device)
        for p in start.parameters():
            p.requires_grad_(False)
        head_reset.copy(actor, start, set(reset_params))
    cadence = cad.of_args(args)
    # gamma and lambda per decision: the --gamma given per 0.5 s (and the default lambda) over the decision's
    # seconds, so the horizon in seconds stays whatever the cadence (cadence.py)
    cfg = ppo.PPOConfig(lr=args.lr, gamma=cadence.discount(args.gamma), lam=cadence.discount(ppo.PPOConfig.lam),
                        epochs=args.epochs, minibatch=args.minibatch,
                        entropy=args.entropy, anchor=args.anchor, adv_norm=args.adv_norm, eyes=args.eyes_weight)
    width = actor.cfg.d / model_config.SMALL.d
    if general_only:
        from tools.nn.general.executor import freeze_but_general
        frozen, trainable = freeze_but_general(actor, critic)
        print(f"general only: {frozen} weights frozen, {trainable} trainable (general)"
              + (f"; turned off: {', '.join(turned_off)}" if turned_off else ""), flush=True)
    # (v2 is trained from scratch, not widened from a small network: no lr / width on the stream's readers)
    opt = optimizer(actor, critic, cfg.lr, 1.0 if actor.cfg.sectors else width, ab_lr,
                    float(getattr(args, "general_lr_mult", 1.0)) if glearn_cfg is not None else 1.0,
                    float(getattr(args, "general_kind_lr_mult", 1.0))
                    if glearn_cfg is not None and glearn_cfg.kind_own else 1.0)
    # Adam's moments and the trainer's counters from --init (saved with latest.pt and m<minute>.pt): the next part of
    # a chain goes on as one run. Only when the critic comes from the same file (else its moments are another's).
    carried = None
    if args.init and args.resume and (args.critic_init is None or same_file(args.critic_init, args.init)):
        carried = resume(opt, args.init, fresh_params(actor, args.init))
    warmup = args.critic_warmup
    if carried is not None:
        warmup = 0
        print(f"resumed Adam's state and the counters {carried} from {args.init}"
              + (f" (no --critic-warmup {args.critic_warmup}: the critic and Adam go on)" if args.critic_warmup else ""),
              flush=True)
    carried = carried or {}
    head_reset.forget(opt, actor, reset_params)
    chain_s0, chain_u0 = float(carried.get("chain_s", 0.0)), int(carried.get("chain_updates", 0))
    decay_s0 = float(carried.get("entropy_decay_s", 0.0))
    observe = observation(args, actor, start, cfg, int(carried.get("observe_updates", 0)), device)
    if args.entropy_decay and args.entropy_end is None:
        raise SystemExit("--entropy-decay needs --entropy-end (the weight it goes to)")
    weights = reward.Weights(order_change=args.order_cost, idle=args.idle, lord=args.lord, retarget=args.retarget,
                             idle_tau_s=args.idle_tau, idle_cap=args.idle_cap, idle_pause_s=args.idle_pause,
                             idle_step=args.idle_step, idle_rate=args.idle_rate, idle_window_s=args.idle_window,
                             gold=args.gold, rout_share=args.rout_share, lord_rout=args.lord_rout)

    def small_arg():
        if not args.small:
            return None
        share, units = args.small.split(":")
        return float(share), int(units)
    entropy_end = args.entropy if args.entropy_end is None else args.entropy_end
    anchor_end = args.anchor if args.anchor_end is None else args.anchor_end
    # The pool of past versions with quality scores (league.Pool); --pool-dir: shared by the parts of a night (it
    # carries on from its q.json: the untrained network and --init only go into a new pool); a chain step's new pool
    # carries on from the previous step's (league.inherit_pool: the --init's or --critic-init's run folder has
    # pool/q.json; --pool-fresh: not); --pool-no-untrained: no untrained network in it
    pool, notes = league.open_pool(
        Path(args.pool_dir) if args.pool_dir else out / "pool", size=args.pool, seed=args.seed, eta=args.pool_eta,
        init=args.init, critic=args.critic_init, untrained=lambda: checkpoint.random_for(actor.cfg),
        inherit=not args.pool_fresh and not args.pool_dir, no_untrained=args.pool_no_untrained,
        extra=list(filter(None, (args.pool_extra or "").split(","))))
    for note in notes:
        print(note, flush=True)
    print(pool.text(), flush=True)
    # The past slots (rollout.Battles): each holds a version; a battle keeps its slot's version from its start
    slots = max(1, args.past_slots)
    slot_path, slot_p = [None] * slots, [1.0] * slots
    pasts = {}                                   # (slot, ModelConfig) -> its actor
    deferred = 0                                 # draws that found no free slot (the active one goes on)
    mix = json.loads(args.mix) if args.mix else league.MIX
    if args.drills:
        # drills: --drills of the battles, shared by --drill-weights (default: the verified ones equally)
        shares = json.loads(args.drill_weights) if args.drill_weights else {n: 1.0 for n in drills.TRAIN}
        mix = league.with_drills(mix, args.drills, shares)
    # the scripts met only as the defender in training (the learner attacks); the evaluation keeps league.ATTACK_ONLY
    defend_only = league.names_list(args.defend_only)
    lay = league.layout(args.battles, len(scenes.SCENES), mix, scene_attacker=scenes.attackers(), attack_only=defend_only)
    # the drills' teacher (drills/teach.py): manual {drill: the imitation term's starting weight}, annealed
    # to 0; or auto (teach_auto.py): every READY drill the run plays, a share of its battles from its deficit
    played = set(np.unique(lay.opponent).tolist())
    auto = args.drill_teach == "auto"
    if auto:
        names = [n for n in drills.evaluated() if league.CODE[drills.opponent(n)] in played]
        if not names:
            raise SystemExit("--drill-teach auto: the run plays no READY drill (--drills, --drill-weights)")
        if teacher is None:
            teacher = teach_auto.Auto(args.drill_teach_k, args.drill_teach_cap, args.drill_teach_weight,
                                      *teach_auto.prior(args.init))
        rows = teacher.bind(names)
        teach0 = dict.fromkeys(names, teacher.weight)
        print(f"drill teacher auto (k {teacher.k:g}, cap {teacher.cap:g}, weight {teacher.weight:g}), from "
              f"{rows[0]['source'] if rows else '-'}:\n" + "\n".join(teach_auto.table(rows)), flush=True)
    else:
        teacher = None
        teach0 = json.loads(args.drill_teach) if args.drill_teach else {}
    absent = [n for n in teach0 if n not in drills.NAMES or league.CODE[drills.opponent(n)] not in played]
    if absent:
        raise SystemExit(f"--drill-teach {absent}: not drills the run plays (--drills, --drill-weights)")
    # the teacher in normal battles (--teach-normal): the drill's teacher script at its moments in the rows of no
    # drill, a share of those battles (fixed, or adaptive from the transfer gap: teach_auto.Transfer)
    fixed_normal, normal_names = normal_drills(args.teach_normal)
    loaded_normal = drills.load(normal_names) if normal_names else {}
    bad = [n for n in normal_names if n not in loaded_normal or loaded_normal[n].moments is None]
    if bad:
        raise SystemExit(f"--teach-normal {bad}: no such drill with moments (drills.Drill.moments)")
    if normal_names and fixed_normal is None:
        if normal is None:
            normal = teach_auto.Transfer(args.teach_normal_k, args.teach_normal_cap, args.teach_normal_weight,
                                         *teach_auto.prior(args.init, "transfer"), match=args.teach_normal_match)
        rows = normal.bind(normal_names)
        print(f"teacher in normal battles auto (k {normal.k:g}, cap {normal.cap:g}, weight {normal.weight:g}, match "
              f"{normal.match:g}), from {rows[0]['source'] if rows else '-'}:\n"
              + "\n".join(teach_auto.table(rows, ref="ai_like applied", net="net applied")), flush=True)
    else:
        normal = None
    normal_w = ({drills.normal_name(n): args.teach_normal_weight for n in normal_names} if fixed_normal is not None
                else {})
    normal_s = {drills.normal_name(n): v for n, v in (fixed_normal or {}).items()}
    params = rollout.params_with_limit(args.limit)
    dr_cfg = uncertain.load(args.dr, params) if args.dr else None
    rng = np.random.default_rng(args.seed)

    def draw():
        """The next bank's random numbers (bankproc.Spec): fresh train seeds of random battles (tools/nn/armies), with
        the drills' banks; --kite-share: kite battles (tools/nn/train/kitesp.py) beside them, taken by the self / past
        rows."""
        from tools.nn.armies import generate
        seeds = rng.integers(generate.TRAIN_SEEDS.start, generate.TRAIN_SEEDS.stop, args.bank)
        seed = int(rng.integers(1 << 30))
        spec = bankproc.Spec(seeds, seed, args.max_units, params, lay, small_arg(), bool(args.drills), args.drill_bank,
                             armies=args.armies)
        if args.kite_share:
            spec.kite_share = args.kite_share
            spec.kite_seeds = rng.integers(generate.TRAIN_SEEDS.start, generate.TRAIN_SEEDS.stop, args.kite_bank)
            spec.kite_seed = int(rng.integers(1 << 30))
        return spec

    def source():
        """A bank of battles built here, on the device."""
        return bankproc.build(draw(), device)

    if args.kite_share and not kitesp.rows_of(lay).any():
        raise SystemExit("--kite-share: the run has no self / past battles (--mix)")
    env = (kitesp.KiteBattles if args.kite_share else rollout.Battles)(lay, scenes.SCENES, device, params, randomise.Spread(), weights, seed=args.seed,
                          source=source(), cadence=cadence, past_slots=slots + bool(args.past_pin),
                          teach=drills.load(list(teach0)) if teach0 else None,
                          teach_normal={n: loaded_normal[n] for n in normal_names} if normal_names else None,
                          attack_only=defend_only, dr=dr_cfg, men=fighters_men(args))
    if env.men is not None:
        print(f"fighters lab: every man of the batch (tools/nn/train/men.py), {env.men.M} places a battle, the "
              f"network sees the units' shape", flush=True)
    if dr_cfg is not None:
        print(dr_cfg.text(), flush=True)
    if normal_s:
        env.set_teach_shares(normal_s)
    # --ability-branch: presses priced by replaying them in the simulator (ability_branch.py), the ability head's term
    ab_updates0 = int(carried.get("ability_updates", 0))
    if args.ability_branch:
        env.brancher = ability_branch.Brancher(ability_branch.Config(
            share=args.ability_branch, missed=args.ability_branch_missed, copies=args.ability_branch_copies,
            horizon_s=args.ability_branch_horizon, cap=args.ability_branch_cap, batch=args.ability_branch_batch,
            hold_s=args.ability_branch_hold, openloop=args.ability_branch_openloop,
            greedy=args.ability_branch_greedy, seed=args.seed, critic=args.ability_critic,
            critic_weight=args.ability_critic_weight, critic_warmup=args.ability_critic_warmup,
            critic_buffer=args.ability_critic_buffer, bonus=ab_bonus, bonus_file=getattr(args, "ability_price_file", ""),
            own=own, norm=getattr(args, "ability_norm", "shared"),
            spell_price=getattr(args, "spell_price", "direct")), device)
        if env.brancher.norm is not None:
            # each ability's running scale goes on from --init's "train" (fresh otherwise)
            ts = (checkpoint.train_state(args.init) or {}) if args.init and args.resume else {}
            loaded = env.brancher.norm.load(ts.get("ability_norm"))
            print(f"ability prices scaled per ability: {'resumed from ' + str(args.init) if loaded else 'fresh'}",
                  flush=True)
        if own:
            print(f"ability branches: {env.brancher.cfg}; the ability head's own signal: weight {ab_own_w:g} (no decay), "
                  f"body gradient x {ab_body:g}, price {env.brancher.price_kind()}, head lr x {ab_lr:g}, "
                  f"ability epochs {ab_epochs}", flush=True)
        else:
            print(f"ability branches: {env.brancher.cfg}, weight {args.ability_branch_weight:g} -> 0 over "
                  f"{args.ability_branch_decay} updates (from update {ab_updates0} of the chain)", flush=True)
        if env.brancher.critic is not None:
            # the ability critic's model, Adam and buffer go on from --init's "train" (a fresh one otherwise)
            ts = (checkpoint.train_state(args.init) or {}) if args.init and args.resume else {}
            saved = ts.get("ability_critic")
            # prices of another kind (open-loop or not, by the bonuses or not; a state without the mark is closed-loop
            # by the branches) start afresh
            kind = ts.get("ability_critic_kind") or ("openloop" if ts.get("ability_critic_openloop", False)
                                                     else "closed")
            if saved is not None and kind != env.brancher.price_kind():
                saved = None
                print(f"ability critic: the saved buffer holds another kind of price ({kind}, now "
                      f"{env.brancher.price_kind()})", flush=True)
            if env.brancher.critic.load(saved):
                print(f"ability critic: resumed from {args.init} ({len(env.brancher.critic)} samples)", flush=True)
            else:
                print("ability critic: fresh", flush=True)
    # v2 (d 256): 2 parts. Until the update freed the critic's activations and the full entropy's graph early
    # (ppo.update; 8.1 -> 5.6 MB a decision) it took one part more: 3 parts peaked at 12.0 GB of the 16 GB card, and a
    # little more pushed the card into its shared memory (an update 10 s -> 157 s); now 2 parts peak at 12.2 GB, 3 at
    # 10.1 GB (line W's batch, 1024 battles x 64 decisions)
    step_cfg = dataclasses.replace(sized(cfg, env.N, width=width), ability_own=own, ability_body_grad=ab_body,
                                   ability_epochs=ab_epochs, amp=getattr(args, "amp", "") or "")
    if args.parts:                                # --parts: the minibatch in this many parts, whatever the size
        step_cfg = dataclasses.replace(step_cfg, accum=args.parts)
    elif env.men is not None:
        # --fighters: with the batch's men on the card through the update (+1.2 GB at 1024 battles x 7168 places, the
        # update's peak +2.5 GB) 2 parts peaked at 14.8 GB, the card spilled into its shared memory (a decision 0.23 ->
        # 2.2 s, an update 18 -> 158 s); with them parked in host memory (the default, Men.park) still ~13.6 GB. 3 parts:
        # 11.5 GB (2 parts fit with --amp bf16h: 12.0 GB)
        step_cfg = dataclasses.replace(step_cfg, accum=max(step_cfg.accum, 3))
    log = (out / "log.jsonl").open("w", encoding="utf-8", newline="\n")
    print(f"{args.name}: battles {env.B} (learner rows {env.R}), slots {env.N}, up to {args.max_units} units a side "
          f"(armies {args.armies}), "
          f"steps per update {args.steps} ({args.steps * env.decision_s:g} s of battle), "
          f"minibatch {step_cfg.minibatch} decisions in {step_cfg.accum} part(s), {cadence.text(params.dt)}, "
          f"model {args.preset} {model_policy.parameters(actor) / 1e6:.2f} M actor, limit {args.limit:.0f} s, "
          f"ppo {dataclasses.asdict(cfg)}, reward {dataclasses.asdict(weights)}, mix {mix}"
          + (f", drill teacher {teach0} -> 0 over {args.drill_teach_minutes:g} min" if teach0 and not auto else "")
          + (f", drill teacher auto {teacher.meta()}" if auto else "")
          + (f", teacher in normal battles {normal.meta()}" if normal is not None else "")
          + (f", teacher in normal battles fixed shares {normal_s} weight {args.teach_normal_weight:g}" if normal_s else ""),
          flush=True)

    def load_slot(k, path, untrained, p):
        if path != slot_path[k] or slots == 1:
            data = checkpoint.read(path, device)
            cfg_past = checkpoint.config_of(data)
            # a pool version of another size (random.pt beside a widened learner) or with a general (S28): its own
            # actor, made once
            key = (k, cfg_past, checkpoint.general_of(data))
            if key not in pasts:
                pasts[key] = checkpoint.actor_of(data).to(device).eval()
            past = pasts[key]
            past.load_state_dict(data["actor"])
            env.set_past(past, untrained, slot=k)
        slot_path[k], slot_p[k] = path, p

    def pick_past():
        """A version drawn from the pool for the battles that start from now on: into a slot that holds no battle
        (one that has it already first), made the active one; the same version as the active slot's goes on there.
        One slot (--past-slots 1): the version changes in the running battles too, their memory afresh."""
        nonlocal deferred
        path, untrained, p = pool.sample()
        a = env.active
        if slots == 1 or slot_path[a] is None or path == slot_path[a]:
            load_slot(a, path, untrained, p)
            return
        counts = env.slot_counts()
        free = [k for k in range(slots) if k != a and counts[k] == 0]
        if not free:
            deferred += 1
            return
        k = next((k for k in free if slot_path[k] == path), free[0])
        load_slot(k, path, untrained, p)
        env.activate(k)

    if args.past_pin:
        # --past-pin: a fixed past version in a slot of its own (past slot `slots`), apart from the pool: a past battle
        # that starts takes it with probability --past-pin-share (rollout.Battles.pin_slot)
        past_share = float(np.mean(lay.opponent == league.CODE["past"]))
        if not past_share:
            raise SystemExit("--past-pin: the run has no past battles (--mix)")
        data = checkpoint.read(args.past_pin, device)
        pinned = checkpoint.actor_of(data).to(device).eval()
        pinned.load_state_dict(data["actor"])
        env.pin_slot(slots, pinned, args.past_pin_share, args.seed)
        print(f"past pinned: {Path(args.past_pin).name} in slot {slots}, apart from the pool; {args.past_pin_share:g} of "
              f"the past battles (~{args.past_pin_share * past_share:.0%} of all), counted as pinned/attack|defend",
              flush=True)
    pick_past()
    dlearn = None                                 # --general-distinct: the classifier (distinct.Learner)
    d_updates0 = int(carried.get("general_distinct_updates", 0))
    e_updates0 = int(carried.get("general_exec_updates", 0))   # --general-exec-decay's counter over the chain
    i_updates0 = int(carried.get("general_imitate_updates", 0))   # --general-imitate-decay's (from its first step)
    if glearn_cfg is not None:
        glearn_cfg = imitate_floor(args, glearn_cfg, carried)
    t_warm = time.time()
    env.step(actor, critic)                       # compiles the simulator's step (not counted)
    if env.general_rt is not None:
        # the general's plans: the return's rule and the discount per decision (the trainer's)
        env.general_rt.ret, env.general_rt.gamma = args.general_return, cfg.gamma
        if glearn_cfg is not None and glearn_cfg.distinct > 0:
            # --general-distinct: the classifier of missions by behaviour (its weights, Adam and the decay's counter go
            # on from --init's "train"; fresh otherwise)
            from tools.nn.general import distinct as gdistinct, runtime as gruntime
            dlearn = gdistinct.Learner(len(gruntime.BEH), actor.gcfg.kinds, device)
            ts = (checkpoint.train_state(args.init) or {}) if args.init and args.resume else {}
            loaded = dlearn.load(ts.get("general_distinct"))
            env.general_rt.distinct = dlearn
            print(f"general distinct bonus: W {glearn_cfg.distinct:g} -> 0 over {glearn_cfg.distinct_decay} updates "
                  f"(from update {d_updates0} of the chain); classifier "
                  f"{'resumed from ' + str(args.init) if loaded else 'fresh'}", flush=True)
        if glearn_cfg is not None:
            # the missions' measures every decision (logged), the executor's bonus for them at --general-exec W (its
            # decay's counter goes on over the chain), the teacher's share of the battles
            env.general_rt.measure = True
            env.general_rt.exec_w = gsignal.distinct_weight(glearn_cfg.exec, glearn_cfg.exec_decay, e_updates0)
            env.general_rt.set_teacher(glearn_cfg.teacher)
            env.general_rt.teach_hold_m = glearn_cfg.teacher_hold_m
            env.general_rt.teach_fire_s = glearn_cfg.teacher_fire_s
            env.general_rt.teach_flank = glearn_cfg.teacher_flank
            env.general_rt.flank_measure = glearn_cfg.flank_measure
            from tools.nn.general.learn import exec_kinds
            env.general_rt.exec_kind = torch.tensor(exec_kinds(glearn_cfg.exec_kind, env.general_rt.gcfg.kinds),
                                                    device=env.general_rt.dev)
            print(f"general missions: the executor's bonus W {glearn_cfg.exec:g}"
                  + (f" -> 0 over {glearn_cfg.exec_decay} updates (from update {e_updates0} of the chain)"
                     if glearn_cfg.exec_decay else "") + f"; the teacher's share of the battles {glearn_cfg.teacher:g}"
                  + (f"; its imitation W {glearn_cfg.imitate:g}" + (f" -> {min(glearn_cfg.imitate_min, glearn_cfg.imitate):g} over {glearn_cfg.imitate_decay} updates "
                                                                   f"(from update {i_updates0})"
                                                                   if glearn_cfg.imitate_decay else "")
                     if glearn_cfg.teacher > 0 else "")
                  + f"; the general's execution bonus S {glearn_cfg.exec_general:g}; the kinds' weights "
                  + f"{glearn_cfg.exec_kind}; the task's masks {'on' if env.general_rt.gcfg.task_mask else 'off'}",
                  flush=True)
        # --general-branches P: a share of the learner's plans replayed with one group's mission kind changed
        env.general_rt.brancher = general_brancher(args, glearn_cfg, device)
        if env.general_rt.brancher is not None:
            gb = env.general_rt.brancher
            if glearn_cfg.kind_own:
                print(f"general mission branches: {gb.cfg}; the kind head's own signal: weight "
                      f"{glearn_cfg.kind_weight:g} (no decay), kind lr x {args.general_kind_lr_mult:g}, kind epochs "
                      f"{glearn_cfg.kind_epochs}", flush=True)
            else:
                print(f"general mission branches: {gb.cfg}, term weight {glearn_cfg.branch_weight:g}", flush=True)
            # the kind critic and each kind's scale go on from --init's "train" (fresh otherwise; prices of another
            # loop - open or closed - start afresh)
            ts = (checkpoint.train_state(args.init) or {}) if args.init and args.resume else {}
            saved = ts.get("general_kind")
            if saved is not None and saved.get("openloop") != gb.cfg.openloop:
                saved = None
            if saved is not None and bool(saved.get("centered", False)) != gb.cfg.center:
                saved = dict(saved, critic=None)  # (the critic's prices centred or not: its buffer starts afresh)
            got = gb.load(saved)
            print(f"general kind critic / scales: {'resumed ' + ', '.join(got) if got else 'fresh'}", flush=True)
        # --general-kind-q outcome|both: every ended plan's groups against the trade their kinds led to
        env.general_rt.outcome_q = general_outcome_q(args, glearn_cfg, env.general_rt.gcfg, device, env.decision_s)
        if env.general_rt.outcome_q is not None:
            env.general_rt.q_h = args.general_kind_q_horizon
            ts = (checkpoint.train_state(args.init) or {}) if args.init and args.resume else {}
            gq = env.general_rt.outcome_q
            loaded = gq.load(ts.get("general_kind_q"))
            print(f"general kind Q ({glearn_cfg.kind_q}): the trade over {args.general_kind_q_horizon:g} s after each "
                  f"plan (waits {gq.wait_n} chunks), buffer {gq.cap}, ready at {gq.warmup} groups and held-out "
                  f"(every 10th battle) ev >= {gq.ev_min:g}, shrink (coefficient - 2 se) "
                  f"{'on' if gq.use_shrink else 'off'}, scale {gq.norm_kind}"
                  + (f", branches' weight beside it {glearn_cfg.branch_weight:g}" if glearn_cfg.kind_q == "both" else "")
                  + f"; {'resumed (' + str(len(gq)) + ' groups)' if loaded else 'fresh'}", flush=True)
        env.general_rt.take_stats()
    env.take_stats()
    env.take_pin_stats()
    env.orders_per_minute()
    env.kinds()
    env.lords()
    print(f"warm-up {time.time() - t_warm:.0f} s", flush=True)
    t0 = time.time()
    t_bank = t0
    # The next bank of armies is built in a process of its own while training goes on (--bank-ahead, bankproc.py):
    # building one is mostly Python (the armies, the drills' battles: 17-48 s that held the GPU idle every
    # --bank-refresh minutes; in a thread it shared the GIL with the collection, 7.5 s -> 16-35 s); the swap then only
    # waits if it is not ready yet.
    builder = bankproc.Ahead() if args.bank_ahead else None
    ahead = builder.submit(draw()) if builder is not None else None
    update, decisions, total, window, best = 0, 0, {}, {}, -1.0
    pin_total = {}                                  # --past-pin: (games, wins, gold trade summed) by matchup
    floor_w = None                                # the entropy floor's current weight (--entropy-target)
    paused = 0.0
    rolls, rolled_at = 0, 0.0                     # --anchor-roll: renewals of the reference, training s of the last
    followed_at = 0.0                             # --anchor-ema: training s of the reference's last step
    next_keep = args.keep_every                   # --keep-every: training minutes of the next kept copy
    saved_at = time.time()                        # --save-every: wall time of the last latest.pt
    if args.anchor_ema and args.anchor_roll:
        raise SystemExit("--anchor-ema and --anchor-roll: one of them")
    next_mark = every[0] if every else None
    if "floor_w" in carried and carried["floor_w"] is not None and args.entropy_target:
        floor_w = float(carried["floor_w"])      # the entropy floor's weight goes on where it was
    # the per-head floors' weights (--head-entropy): from the start weight, or where the chain left them
    head_w = {h: float((carried.get("head_w") or {}).get(h, args.head_entropy_start)) for h in head_targets}
    if head_targets:
        print("per-head entropy floors (share of the maximum): " + ", ".join(f"{h} {t:g}" for h, t in head_targets.items())
              + f"; weights from {head_w} within [{args.head_entropy_start:g}, {args.head_entropy_max:g}] "
              f"x {args.entropy_rate:g} an update", flush=True)

    # --general-only: the ability branches' state of --init goes through untouched (they are off: general_only_args)
    passed = ({k: v for k, v in (checkpoint.train_state(args.init) or {}).items() if k in ABILITY_STATE}
              if general_only and args.init and args.resume else {})

    def blob():
        """What --init needs to go on seamlessly from a checkpoint (checkpoint.py "train")."""
        trained = time.time() - t0 - paused
        return {"optim": opt.state_dict(),
                "state": {"floor_w": floor_w, "head_w": head_w,
                          "entropy_decay_s": decay_s0 + (trained if args.entropy_decay and not general_only else 0.0),
                          "chain_s": chain_s0 + trained, "chain_updates": chain_u0 + update,
                          "ability_updates": ab_updates0 + (update if args.ability_branch else 0),
                          "observe_updates": observe.updates if observe is not None
                          else int(carried.get("observe_updates", 0)),
                          "general_distinct_updates": d_updates0 + (update if dlearn is not None else 0),
                          "general_exec_updates": e_updates0 + (update if glearn_cfg is not None else 0),
                          "general_imitate_updates": i_updates0 + (update if glearn_cfg is not None else 0),
                          **({"general_imitate_min": glearn_cfg.imitate_min} if glearn_cfg is not None else {})},
                **({"general_distinct": dlearn.state()} if dlearn is not None else {}), **passed,
                **({"ability_critic": env.brancher.critic.state(),
                    "ability_critic_openloop": bool(env.brancher.cfg.openloop),
                    "ability_critic_kind": env.brancher.price_kind()}
                   if args.ability_branch and env.brancher.critic is not None else {}),
                **({"ability_norm": env.brancher.norm.state()}
                   if args.ability_branch and env.brancher.norm is not None else {}),
                **({"general_kind": dict(env.general_rt.brancher.state(),
                                         openloop=env.general_rt.brancher.cfg.openloop,
                                         centered=env.general_rt.brancher.cfg.center)}
                   if getattr(env, "general_rt", None) is not None and env.general_rt.brancher is not None else {}),
                **({"general_kind_q": env.general_rt.outcome_q.state()}
                   if getattr(env, "general_rt", None) is not None
                   and getattr(env.general_rt, "outcome_q", None) is not None else {})}

    def share_done():
        """The share of the run done: of the updates when --updates is set, else of the minutes."""
        if args.updates:
            return update / args.updates
        return (time.time() - t0 - paused) / (args.minutes * 60)

    while share_done() < 1.0 and (not args.updates or time.time() - t0 - paused < args.minutes * 60):
        t_u = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()    # the log's gpu_gb: this update's peak
        done_share = share_done()
        if time.time() - t_bank >= args.bank_refresh * 60:
            if ahead is not None:
                env.source = builder.take(ahead, device)
                ahead = builder.submit(draw())
            else:
                env.source = source()
            env.bank = env.source.bank
            t_bank = time.time()
        if teacher is not None:
            env.set_teach_shares(teacher.shares)
        if normal is not None:
            env.set_teach_shares(normal.shares)
        if env.men is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()                  # --fighters: the update's cached blocks back for the men's steps
        if env.men is not None:
            env.men.unpark(env.st)                    # (--fighters-park: the men back on the card)
        if dlearn is not None:
            # --general-distinct: the bonus's weight of this collection (W down to 0 over --general-distinct-decay)
            dlearn.weight = gsignal.distinct_weight(glearn_cfg.distinct, glearn_cfg.distinct_decay, d_updates0 + update)
        if glearn_cfg is not None and env.general_rt is not None:
            # --general-exec: the mission bonus's weight of this collection (W down to 0 over --general-exec-decay)
            env.general_rt.exec_w = gsignal.distinct_weight(glearn_cfg.exec, glearn_cfg.exec_decay, e_updates0 + update)
        batch = rollout.collect(env, actor, critic, args.steps)
        if glearn_cfg is not None and env.general_rt is not None and env.general_rt.outcome_q is not None:
            # --general-kind-q outcome|both: the outcome Q's estimate of every ended plan's groups (the kind term)
            env.general_rt.outcome_q.attach(batch, actor)
        elif glearn_cfg is not None and env.general_rt is not None and env.general_rt.brancher is not None:
            # --general-kind-critic: its estimate of every ended plan's groups (learn.update's kind term)
            env.general_rt.brancher.attach(batch, actor)
        if env.men is not None and args.fighters_park:
            env.men.park(env.st)                      # the men off the card while the network learns (Men.park)
        if env.men is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()                  # ... and the steps' blocks back for the update
        t_c = time.time()
        # Schedules: the kind's entropy bonus and the KL to the reference go linearly from their start to
        # their end value over the run (e.g. exploration fades out).
        # With --entropy-target the weight is the floor's (ppo.entropy_weight), never below the schedule.
        # --entropy-decay: over that many minutes of training with it on, carried over the parts of a chain
        decay_s = decay_s0 + (time.time() - t0 - paused if args.entropy_decay else 0.0)
        scheduled = schedule(args.entropy, entropy_end, decay_share(args.entropy_decay, decay_s, done_share))
        floor_w = scheduled if floor_w is None else max(scheduled, floor_w)
        ab_w = (ab_own_w if own else
                ability_branch.weight(args.ability_branch_weight, args.ability_branch_decay, ab_updates0 + update)
                if args.ability_branch else 0.0)
        u_cfg = dataclasses.replace(step_cfg, entropy=floor_w if args.entropy_target else scheduled,
                                    anchor=schedule(args.anchor, anchor_end, done_share), heads=tuple(head_w.items()),
                                    ability=ab_w, ability_own=own, ability_body_grad=ab_body)
        trains = update >= warmup
        teach_w = (teacher.weights() if teacher is not None
                   else drill_teach.weights(teach0, args.drill_teach_minutes, (time.time() - t0 - paused) / 60))
        teach_w = {**teach_w, **(normal.weights() if normal is not None else normal_w)}
        # (--general-imitate: the teacher's imitation's weight of this update, W down to the floor --general-imitate-min
        # over --general-imitate-decay)
        g_cfg = (dataclasses.replace(glearn_cfg, imitate=gsignal.imitate_weight(
            glearn_cfg.imitate, glearn_cfg.imitate_decay, i_updates0 + update, glearn_cfg.imitate_min))
                 if glearn_cfg is not None else None)
        st = ppo.update(actor, critic, opt, batch, u_cfg, train_policy=trains, reference=reference, teach=teach_w,
                        observe=observe, general=g_cfg, executor=not general_only)
        taught = st.pop("teach", None)
        observed = st.pop("observe", None)
        if dlearn is not None:
            # the classifier learns on the ended plans' groups (its accuracy before the steps: general_distinct_acc)
            st.update({"general_" + k: v for k, v in
                       dlearn.fit((batch.get("general") or {}).get("ready")).items()})
            # the bonus's scale: the executor's advantage std (signal.distinct_scale), a running mean over the updates
            stds = [st[k] for k in ("adv_std_attack", "adv_std_defend") if k in st]
            dlearn.scale = gsignal.distinct_scale(dlearn.scale, sum(stds) / len(stds) if stds else None)
            st["general_distinct_scale"] = dlearn.scale
        if start is not None:
            st["start_kl"] = ppo.distance(actor, start, batch, u_cfg.minibatch)
        if args.entropy_target and trains and not general_only:     # (--general-only: no executor's entropy)
            floor_w = ppo.entropy_weight(floor_w, st["entropy"], args.entropy_target, scheduled,
                                         max(scheduled, args.entropy_max), args.entropy_rate)
        if head_targets and trains and not general_only:
            head_w = head_entropy.next_weights(head_w, {h: st.get(f"head_share_{h}") for h in head_targets},
                                               head_targets, args.head_entropy_start, args.head_entropy_max,
                                               args.entropy_rate)
        del batch
        update += 1
        if args.gpu_duty < 1.0:
            # --gpu-duty: rest the GPU for this share of the time (quieter fans, a responsive desktop);
            # the rest does not count as training time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rest = (time.time() - t_u) * (1.0 / max(args.gpu_duty, 0.1) - 1.0)
            time.sleep(rest)
            paused += rest
        # --anchor-roll: every that many seconds of training the reference becomes the current actor (a
        # leash that moves with the network: it bounds the drift within a window, not the whole run).
        trained_s = time.time() - t0 - paused
        # --anchor-ema: after every update the reference moves towards the actor, by the share that halves
        # their distance in --anchor-ema seconds of training (a slow leash: a sudden collapse moves it a
        # little and the KL pulls back; steady progress it follows)
        if args.anchor_ema and reference is not None:
            follow(reference, actor, 1.0 - 0.5 ** ((trained_s - followed_at) / args.anchor_ema))
            followed_at = trained_s
        if args.anchor_roll and reference is not None and trained_s - rolled_at >= args.anchor_roll:
            reference.load_state_dict(actor.state_dict())
            rolls, rolled_at = rolls + 1, trained_s
            print(f"   anchor roll {rolls}: the KL reference is now the actor of update {update} "
                  f"({trained_s / 60:.1f} min of training)", flush=True)
        if args.save_every and time.time() - saved_at >= args.save_every * 60:
            # --save-every: latest.pt (with the critic) every that many minutes, whatever the snapshots' rhythm (a run
            # that dies restarts from it: build/steps/v2_iter.sh)
            checkpoint.save(out / "latest.pt", actor, critic, args.preset,
                            {"update": update, "battles": env.battles, "seconds": round(trained_s), "run": args.name,
                             "cadence": cadence.meta()}, blob())
            saved_at = time.time()
        if args.keep_every and trained_s >= next_keep * 60:
            # --keep-every: a copy of the network (with its critic) every that many minutes of training: m<minute>.pt
            checkpoint.save(out / f"m{int(round(next_keep))}.pt", actor, critic, args.preset,
                            {"update": update, "battles": env.battles, "minute": next_keep, "run": args.name,
                             "cadence": cadence.meta()}, blob())
            next_keep += args.keep_every
        decisions += env.B * args.steps
        stats = env.take_stats()
        pin_m = env.take_pin_stats()          # --past-pin: the pinned version's battles by matchup ({} without)
        for k, (g, w, t) in pin_m.items():
            G, W, T = pin_total.get(k, (0, 0, 0.0))
            pin_total[k] = (G + g, W + w, T + t * g)
        # the learner's results against each slot's version -> the pool's quality scores (before a slot changes); the
        # pinned slot (--past-pin, the last one) stays out of the pool
        for k, (g, w) in enumerate(env.take_slot_stats()[:slots]):
            if g and slot_path[k] is not None:
                pool.result(slot_path[k], g, w, slot_p[k])
        pool.tick()
        pool.write()
        lords = env.lords()
        for k, (g, w, s) in stats.items():
            for acc in (total, window):
                G, W, S = acc.get(k, (0, 0, 0.0))
                acc[k] = (G + g, W + w, S + s * g)
        row = {"update": update, "seconds": round(time.time() - t0, 1), "battles": env.battles,
               "timeouts": env.timeouts, "battle_steps_per_s": round(env.B * args.steps / (time.time() - t_u)),
               "battle_seconds_per_s": round(env.B * args.steps * env.decision_s / (time.time() - t_u)),
               "collect_s": round(t_c - t_u, 2), "update_s": round(time.time() - t_c, 2),
               **{k: round(v, 4) for k, v in st.items()},
               "entropy_weight": round(u_cfg.entropy, 5), "anchor_weight": round(u_cfg.anchor, 4),
               "anchor_rolls": rolls, "head_entropy_weight": {h: round(w, 5) for h, w in dict(u_cfg.heads).items()},
               "lord_dead_own": round(lords["own"], 3), "lord_dead_enemy": round(lords["enemy"], 3),
               "abilities_per_battle": round(env.abilities(), 2), "nan_fixed": env.nan_fixed(),
               "switches_per_minute": round(lords["switches_per_minute"], 2),
               "orders_per_minute": round(env.orders_per_minute(), 2), "kinds": env.kinds(),
               "reward_parts": {r: {k: round(v, 4) for k, v in p.items()} for r, p in env.reward_parts().items()},
               "games": {k: g for k, (g, _, _) in stats.items()}, "win_rate": rates(stats),
               "seconds_per_battle": {k: round(s) for k, (_, _, s) in stats.items()},
               "past": slot_path[env.active].name, "past_deferred": deferred,
               "pool": {"n": len(pool), "wide": int((pool.probs() > 0.05).sum()),
                        "top": [[n, round(q, 3), round(p, 3), age] for n, q, p, age in pool.top(5)]}}
        if pin_m:
            row["pinned_matchups"] = pinned_matchups(pin_m)
        if taught:
            row["teach"] = taught
        if torch.cuda.is_available():
            row["gpu_gb"] = {"peak": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                             "alloc": round(torch.cuda.memory_allocated() / 2**30, 2),
                             "reserved": round(torch.cuda.memory_reserved() / 2**30, 2)}
        if args.ability_branch:
            row["ability_branch"] = dict(env.brancher.stats, weight=round(ab_w, 5))
        if observed:
            row["observe"] = {k: round(v, 4) for k, v in observed.items()}
        if args.kite_share:
            row["kite"] = env.kite_stats()
        if env.general_rt is not None:
            row["general"] = dict(env.general_rt.take_stats(),
                                  **{k[len("general_"):]: round(v, 5) for k, v in st.items() if k.startswith("general_")})
            if env.general_rt.brancher is not None:
                row["general"].update({"branch_" + k: v for k, v in env.general_rt.brancher.stats.items()})
            if env.general_rt.outcome_q is not None:
                row["general"].update({"kind_q_" + k: v for k, v in env.general_rt.outcome_q.stats.items()})
        log.write(json.dumps(row) + "\n")
        log.flush()
        if update % args.print_every == 0:
            sv = (lambda k: st.get(k, float("nan")))      # (--general-only: the executor's stats are not computed)
            print(f"u{update:4d} {row['seconds']:6.0f}s battles {env.battles:6d} (timeouts {env.timeouts:5d}) "
                  f"{row['battle_steps_per_s']:6d} st/s ({row['battle_seconds_per_s']} battle-s/s) pl {sv('policy_loss'):+.3f} vl {st['value_loss']:.4f} "
                  f"ent {sv('entropy'):.2f}/{sv('entropy_all'):.2f} kl {sv('kl'):.3f} R {st['reward']:+.3f} "
                  f"anchor {sv('anchor_kl'):.3f} start {st.get('start_kl', 0.0):.3f} orders/min {row['orders_per_minute']:5.1f} "
                  f"lords dead own {lords['own']:.2f} enemy {lords['enemy']:.2f} abil {row['abilities_per_battle']:.1f} "
                  f"kinds " + " ".join(f"{k[:2]} {v:.2f}" for k, v in row["kinds"].items()), flush=True)
            print(f"      ev {st.get('ev', 0):.3f} (attack {st.get('ev_attack', 0):.3f} defend {st.get('ev_defend', 0):.3f}) "
                  f"adv std attack {st.get('adv_std_attack', 0):.3f} defend {st.get('adv_std_defend', 0):.3f}; "
                  "reward a minute " + "; ".join(f"{r} " + " ".join(f"{k} {v:+.3f}" for k, v in p.items())
                                                 for r, p in row["reward_parts"].items()), flush=True)
            if any(k.startswith("head_share_") for k in st):
                print("      heads' spread (share of the maximum / nats) " + " ".join(
                    f"{h} {st['head_share_' + h]:.2f}/{st['head_ent_' + h]:.2f}" for h in head_entropy.HEADS
                    if "head_share_" + h in st)
                      + ("; floors' weights " + " ".join(f"{h} {w:.4f}" for h, w in u_cfg.heads) if u_cfg.heads else ""),
                      flush=True)
            if row["nan_fixed"] or st.get("skipped"):
                print(f"      non-finite: logits taken as masked {row['nan_fixed']:.0f}, update steps skipped "
                      f"{st.get('skipped', 0):.0f}", flush=True)
            if "eyes_own10" in st:
                print("      eyes mse / explained " + "; ".join(
                    f"{h} {st['eyes_' + h]:.4f} / {st['eyes_ev_' + h]:+.2f}" for h in ("own10", "own30", "threat30",
                                                                                     "danger10")), flush=True)
            if args.kite_share:
                print("      " + kitesp.text(row["kite"]), flush=True)
            if "general" in row:
                print("      " + general_text(row["general"]), flush=True)
            if args.ability_branch:
                print("      ability branches " + " ".join(f"{k} {v}" for k, v in row["ability_branch"].items()),
                      flush=True)
            if observed and "loss" not in observed:     # the term off (its weight negligible, observe/store.py)
                print(f"      observe off (weight {observed['weight_now']:.1e}); anchor weight {observed['anchor_now']:.4f} "
                      f"KL to the start {observed['anchor_kl']:.4f}", flush=True)
            elif observed:
                print(f"      observe weight {observed['weight_now']:.4f} loss {observed['loss']:.3f} nll "
                      f"{observed['nll']:.3f} mean weight {observed['mean_c']:.2f} kind agree {observed['kind_acc']:.3f} "
                      f"target agree {observed['target_acc']:.3f} ({observed['labels']:.0f} selected labels, "
                      f"{observed['attack_labels']:.0f} attacks); "
                      f"anchor weight {observed['anchor_now']:.4f} KL to the start {observed['anchor_kl']:.4f}", flush=True)
            if taught:
                active = (lambda v: "-" if v["agree_active"] is None else f"{v['agree_active']:.3f}")
                share = (lambda v: f"share {v['share']:.2f} (labelled {v['labelled']:.2f}) " if "share" in v else "")
                print("      teacher " + "; ".join(f"{k} {share(v)}weight {v['weight']:.3f} ce {v['ce']:.3f} agree {v['agree']:.3f} "
                                                  f"(kind {v['agree_kind']:.3f}, active {active(v)} of {v['active']}; "
                                                  f"{v['units']} unit-decisions)" for k, v in taught.items()), flush=True)
        if update % args.snapshot_every == 0:
            meta = {"update": update, "battles": env.battles, "seconds": row["seconds"], "run": args.name,
                    "cadence": cadence.meta()}
            pool.save(actor, None, args.preset, meta, name=f"{args.name}_v{update:05d}.pt" if args.pool_dir else None)
            checkpoint.save(out / "latest.pt", actor, critic, args.preset, meta, blob())
            print("   " + pool.text() + (f"; draws without a free slot {deferred}" if slots > 1 else ""), flush=True)
            sc, _ = score(window)
            print(f"   window: " + ", ".join(f"{k} {w / g:.2f} ({g})" for k, (g, w, _) in sorted(window.items())),
                  flush=True)
            if sc is not None:
                if sc > best:
                    best = sc
                    checkpoint.save(out / "best.pt", actor, critic, args.preset, dict(meta, score=sc))
                    print(f"   best: worst scripted win rate {sc:.2f}", flush=True)
                window = {}
        if update % args.past_every == 0:
            pick_past()
        if every and time.time() - t0 - paused >= next_mark * 60 and time.time() - t0 - paused < args.minutes * 60:
            t_e = time.time()
            torch.cuda.empty_cache()
            every[1](actor, critic, next_mark, update)
            actor.eval()
            critic.eval()
            torch.cuda.empty_cache()
            paused += time.time() - t_e
            next_mark += every[0]
    seconds = time.time() - t0
    if builder is not None:
        builder.close()                       # (the bank being built is not needed)
    meta = {"update": update, "battles": env.battles, "seconds": round(seconds), "decisions": decisions,
            "battles_at_once": env.B, "steps_per_update": args.steps, "limit_s": args.limit, "run": args.name,
            "cadence": cadence.meta(),
            "ppo": dataclasses.asdict(cfg), "reward": dataclasses.asdict(weights)}
    checkpoint.save(out / "latest.pt", actor, critic, args.preset, meta, blob())
    print(pool.text(), flush=True)
    if not (out / "best.pt").exists():
        checkpoint.save(out / "best.pt", actor, critic, args.preset, meta)
    log.close()
    summary = {"updates": update, "seconds": round(seconds), "battles": env.battles, "timeouts": env.timeouts,
               "battle_steps_per_s": round(decisions / seconds),
               "battle_seconds_per_s": round(decisions * env.decision_s / seconds),
               "battles_per_min": round(env.battles / (seconds / 60)), "cadence": cadence.meta(),
               "win_rate_all": {k: round(w / g, 3) for k, (g, w, _) in sorted(total.items()) if g},
               "games": {k: g for k, (g, _, _) in sorted(total.items())},
               "mean_battle_s": {k: round(s / g) for k, (g, _, s) in sorted(total.items()) if g}, "best_score": best}
    if pin_total:
        summary["pinned_matchups"] = pinned_matchups({k: (g, w, t / g) for k, (g, w, t) in pin_total.items() if g})
    print("trained:", json.dumps(summary), flush=True)
    return actor, summary, out


def observation(args, actor, start, cfg, updates, device):
    """--observe-dir: the demonstrations' term for ppo.update (tools/nn/observe/store.py Observe), or None."""
    if not args.observe_dir:
        return None
    from tools.nn.observe import store
    if args.observe_anchor and start is None:
        raise SystemExit("--observe-anchor needs --init (the frozen network the KL holds the policy near)")
    T = args.observe_steps or args.steps
    # (the term off from the start - its weight negligible after the chain's updates: the store is not loaded)
    demos = (store.Demos(args.observe_dir, T, args.observe_adv, args.observe_beta, args.observe_clip, actor.cfg)
             if args.observe_weight * args.observe_decay ** updates >= store.NEGLIGIBLE else None)
    chunks = max(1, round(args.observe_share * cfg.minibatch / T))
    obs = store.Observe(demos, chunks, args.observe_weight, args.observe_decay, start if args.observe_anchor else None,
                        args.observe_anchor, args.observe_anchor_decay, args.observe_refresh, updates, device, args.seed,
                        anchor_alone=args.observe_anchor_alone)
    print(demos.text() if demos is not None else "observation: the term is off (its weight is negligible), the "
          "demonstrations are not loaded", flush=True)
    print(f"observation: {chunks} chunks of {T} decisions beside every minibatch of {cfg.minibatch}; weight "
          f"{args.observe_weight:g} x {args.observe_decay:g} an update (now {obs.weight:.4f} after {updates} updates "
          f"of the chain), KL to --init {args.observe_anchor:g} x {args.observe_anchor_decay:g} (now {obs.anchor:.4f}; "
          + ("on after the term too" if args.observe_anchor_alone else "off with the term") + "), "
          f"memories again every {args.observe_refresh} updates", flush=True)
    return obs


def show(name, r):
    print(f"{name}: orders/min {r['orders_per_minute']:.1f}, kinds "
          + ", ".join(f"{k} {v:.2f}" for k, v in r["kinds"].items()), flush=True)
    for opp, o in r["by_opponent"].items():
        for role, x in o["roles"].items():
            if not x.get("games"):
                continue
            print(f"  v {opp:10} {role:6} win {x['win_rate']:.3f} ({x['wins']}/{x['games']}), {x['seconds']:5.0f} s, "
                  f"HP lost own {x['hp_own_lost']:.2f} enemy {x['hp_enemy_lost']:.2f}, "
                  f"gold lost own {x.get('gold_lost', float('nan')):.0f} enemy {x.get('gold_destroyed', float('nan')):.0f} "
                  f"(ratio {x.get('gold_ratio', float('nan')):.2f}), timeouts {x['timeouts']:.2f}, "
                  f"lord dead own {x.get('lord_dead_own', float('nan')):.2f} enemy {x.get('lord_dead_enemy', float('nan')):.2f}")
        print(f"      kinds " + ", ".join(f"{k} {v:.2f}" for k, v in o["kinds"].items()), flush=True)
        if "matchups" in o:
            print(f"      by faction: {matchups.text(o)}", flush=True)
        if len(r["by_scene"][opp]) <= 12:
            print("      " + "; ".join(f"{k} {v['win_rate']:.2f}" for k, v in r["by_scene"][opp].items()), flush=True)


def final_eval(path, args, summary, out):
    device = torch.device(args.device)
    cadence = cad.of_args(args)
    actor = checkpoint.load_policy(path, device)
    untrained = checkpoint.load_policy(args.eval_past or checkpoint.RANDOM, device)
    res = {"training": summary, "checkpoint": str(path)}
    t = time.time()
    men = (lambda: fighters_men(args, None)) if getattr(args, "fighters", False) else None
    with fighters_installed(args):
        res["trained"] = evaluate.play(actor, past=untrained, device=device, limit_s=args.limit,
                                       generated=args.eval_generated, max_units=args.max_units, cadence=cadence,
                                       fighters=men)
    print(f"eval {path.name} {time.time() - t:.0f} s", flush=True)
    show(path.name, res["trained"])
    (out / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    with fighters_installed(args):
        paths = evaluate.record(actor, out / "replays", device=device, limit_s=args.limit, source=str(path),
                                cadence=cadence, fighters=men)
    print(f"replays: {len(paths)} in {out / 'replays'}", flush=True)
    return res


def parser():
    ap = Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="run")
    ap.add_argument("--init", help="start from this checkpoint (default: random.pt)")
    ap.add_argument("--minutes", type=float, default=5.0, help="wall time of training (the warm-up not counted)")
    ap.add_argument("--updates", type=int, default=0,
                    help="train this many updates instead (--minutes then only caps the time; schedules follow the updates)")
    ap.add_argument("--battles", type=int, default=1024, help="battles at once")
    ap.add_argument("--steps", type=int, default=64, help="decisions per chunk (between updates)")
    ap.add_argument("--preset", choices=sorted(model_config.PRESETS),
                    help="the size of a network started from random.pt (default: --init's, else small; a checkpoint "
                         "brings its own sizes)")
    ap.add_argument("--limit", type=float, default=3600.0, help="battle time limit, s (the defender wins at it)")
    ap.add_argument("--fighters", action="store_true",
                    help="the fighters lab in the training's simulator (tools/nn/train/men.py, "
                         "docs/en/training/fighters_lab.md): every man a point with his health and place, the network "
                         "sees the units' shape; the final evaluation too (the baselines stay the plain simulator's); "
                         "off by default")
    ap.add_argument("--no-fighters-park", dest="fighters_park", action="store_false",
                    help="--fighters: keep the batch's men on the card while the network learns (by default they wait in "
                         "host memory, men.Men.park: 1.2 GB less at the update's peak)")
    ap.add_argument("--fighters-men", type=int, default=7168,
                    help="--fighters: places a battle (the batch's fixed size; a battle of more men is picked again)")
    ap.add_argument("--dr", nargs="?", const=str(uncertain.DEFAULT), default=None, metavar="JSON",
                    help="domain randomisation of the simulator's uncertain numbers (tools/nn/train/dr.py): each training "
                         "battle draws them inside the ranges of this file (no value: config/nn/dr.json); off by default")
    ap.add_argument("--lr", type=float, default=ppo.PPOConfig.lr)
    ap.add_argument("--gamma", type=float, default=ppo.PPOConfig.gamma,
                    help="discount per 0.5 s of battle (per decision: gamma ** (--decide-s / 0.5); GAE's lambda too)")
    cad.add_args(ap)
    ap.add_argument("--epochs", type=int, default=ppo.PPOConfig.epochs)
    ap.add_argument("--minibatch", type=int, default=ppo.PPOConfig.minibatch, help="decisions per minibatch")
    ap.add_argument("--parts", type=int, default=0,
                    help="compute each minibatch in this many parts (gradient accumulation; 0: by the network's size, sized())")
    ap.add_argument("--entropy", type=float, default=ppo.PPOConfig.entropy, help="weight of the kind's entropy")
    ap.add_argument("--entropy-end", type=float, help="... at the end of the run (linear; default: no change)")
    ap.add_argument("--entropy-decay", type=float, default=0.0,
                    help="> 0: --entropy goes to --entropy-end over this many minutes of training with this option on, "
                         "counted over the parts of a chain (saved with the checkpoint) instead of over one run "
                         "(e.g. 0.01 -> 0.003 over 120; 0: off)")
    ap.add_argument("--entropy-target", type=float, default=0.0,
                    help="> 0: an entropy floor - the weight goes up x --entropy-rate every update while the kind's "
                         "entropy is below this, back down to the schedule above it (0: the schedule only)")
    ap.add_argument("--entropy-max", type=float, default=0.1, help="the floor's weight at most this")
    ap.add_argument("--entropy-rate", type=float, default=1.25, help="the floor's factor per update (also the "
                                                                      "per-head floors')")
    ap.add_argument("--head-entropy", default="",
                    help="v2: a floor of its own for each named head, as the share of its maximum entropy (0: one "
                         "choice, 1: all alike; head_entropy.py), e.g. 'sector=0.4,cell=0.5,hold=0.5'; heads: "
                         + ", ".join(head_entropy.HEADS))
    ap.add_argument("--head-entropy-start", type=float, default=0.003,
                    help="the per-head floors' starting and lowest weight")
    ap.add_argument("--head-entropy-max", type=float, default=0.05, help="... and highest")
    ap.add_argument("--reset-heads", default="",
                    help="v2: these heads' layers start again as a new network's when --init is loaded, the rest "
                         "kept (head_reset.py; against the primacy bias), e.g. 'sector,cell,hold'; heads: "
                         + ", ".join(head_reset.PARTS) + ". Once: give it to one step of a chain, not to every step")
    ap.add_argument("--order-cost", type=float, default=reward.Weights.order_change)
    ap.add_argument("--eyes-weight", type=float, default=ppo.PPOConfig.eyes,
                    help="weight of the eyes' loss (v2 with eyes: auxiliary heads on the simulator's truth)")
    ap.add_argument("--reward", choices=sorted(REWARD_PRESETS),
                    help="a reward preset: its values replace --order-cost, --lord, --idle, --retarget (v2: win / loss "
                         "+ gold trade + order change 0.006; lord, idle and retarget 0)")
    ap.add_argument("--grid", type=int, default=0,
                    help="> 0: the move point as a cell of a grid x grid map grid (tools/nn/model/heads.py); a checkpoint "
                         "of the bin head keeps every other weight, the grid head starts fresh")
    ap.add_argument("--idle", type=float, default=reward.Weights.idle)
    ap.add_argument("--gold", type=float, default=reward.Weights.gold,
                    help="(enemy gold destroyed - own gold lost) / budget, per step")
    ap.add_argument("--rout-share", type=float, default=reward.Weights.rout_share,
                    help="a routing unit (it may rally) loses this share of the gold it has left")
    ap.add_argument("--lord", type=float, default=reward.Weights.lord, help="the enemy lord's death - own lord's death")
    ap.add_argument("--lord-rout", type=float, default=reward.Weights.lord_rout,
                    help="> 0: in the lord term a shattered lord counts as dead, a routing one as this share of a death")
    ap.add_argument("--retarget", type=float, default=reward.Weights.retarget,
                    help="cost of switching an attack to another target while the old one stands")
    ap.add_argument("--mix", help='opponent shares as json, e.g. {"self": 0.2, "nearest": 0.4}')
    ap.add_argument("--defend-only", default=",".join(league.ATTACK_ONLY),
                    help="training: these opponents (comma-separated) play only the defender, the learner attacks "
                         "(default: league.ATTACK_ONLY; the evaluation always uses that one)")
    ap.add_argument("--critic-warmup", type=int, default=0, help="first updates train only the critic")
    ap.add_argument("--anchor", type=float, default=ppo.PPOConfig.anchor, help="KL weight to the reference actor")
    ap.add_argument("--anchor-end", type=float, help="... at the end of the run (linear; default: no change)")
    ap.add_argument("--reference", help="the reference actor (default: --init)")
    ap.add_argument("--anchor-roll", type=float, default=0.0,
                    help="seconds of training between renewals of the KL reference (--reference or --init): it "
                         "becomes the current actor (a rolling anchor; 0: fixed for the run)")
    ap.add_argument("--anchor-ema", type=float, default=0.0,
                    help="> 0: after every update the reference moves towards the actor (Polyak averaging of the "
                         "weights) with this half-life in seconds of training (0: off; not with --anchor-roll)")
    ap.add_argument("--critic-init", help="take the critic from this checkpoint (default: --init's own)")
    ap.add_argument("--adv-norm", default=ppo.PPOConfig.adv_norm, choices=("batch", "role"),
                    help="normalise the side's advantage over the minibatch or over each role apart")
    ap.add_argument("--pool", type=int, default=8, help="past versions in the pool at most (the lowest q leaves)")
    ap.add_argument("--pool-extra", help="more past opponents for the pool (checkpoints, comma-separated; one already "
                                         "in keeps its q)")
    ap.add_argument("--pool-dir", help="the pool's folder (snapshots and q.json, league.Pool): shared by the parts of a "
                                       "night, a part carries on from its q.json (default: the run's own pool/)")
    ap.add_argument("--pool-no-untrained", action="store_true",
                    help="no untrained network in the pool (a new one gets --init alone; an inherited one loses it): "
                         "in a step of the chain it took up to 11 %% of the battles at an equal p")
    ap.add_argument("--pool-fresh", action="store_true",
                    help="a new pool from scratch: do not carry on from the previous step's pool/q.json beside "
                         "--init or --critic-init")
    ap.add_argument("--pool-eta", type=float, default=0.01,
                    help="the quality score's step: a win against version i lowers its q by this / (N p_i) (OpenAI 0.01)")
    ap.add_argument("--past-slots", type=int, default=3,
                    help="past versions playing at once, each battle keeps the one it began with (1: one version, "
                         "changed in the running battles with their memory afresh, the old way)")
    ap.add_argument("--past-pin", help="a fixed past opponent (a checkpoint) in a slot of its own, apart from the pool "
                                       "(no quality score, never replaced): it takes --past-pin-share of the past "
                                       "battles; its results in the log as pinned/attack|defend (rollout.Battles.pin_slot)")
    ap.add_argument("--past-pin-share", type=float, default=0.65,
                    help="--past-pin: the share of the starting past battles that take the pinned version (0.65 of "
                         "past 0.38 / 0.80 of the mix ~ 31 %% of all battles)")
    ap.add_argument("--past-every", type=int, default=6,
                    help="updates between draws of the past version for the battles that start next (6 ~ a battle)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="do not take Adam's state and the counters from --init (default: taken when it has them and "
                         "the critic is its own; then no --critic-warmup)")
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--save-every", type=float, default=5.0,
                    help="minutes between saves of latest.pt (with the critic) besides the snapshots' (0: only those)")
    ap.add_argument("--keep-every", type=float, default=0.0,
                    help="> 0: also keep a copy of the network every this many minutes of training (m<minute>.pt in "
                         "the run's folder; 0: only latest.pt and best.pt)")
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--eval-past", help="the network behind 'past' in the final evaluation (default: random.pt)")
    ap.add_argument("--small", help="share:units - that share of every generated bank with at most `units` a side")
    ap.add_argument("--idle-tau", type=float, default=reward.Weights.idle_tau_s,
                    help="s: before its first damage the attacker's idle cost is --idle x (exp(t / this) - 1)")
    ap.add_argument("--idle-pause", type=float, default=reward.Weights.idle_pause_s,
                    help="s: after it, a step up every this many seconds without damage (0 before the first)")
    ap.add_argument("--idle-step", type=float, default=reward.Weights.idle_step,
                    help="... --idle x (exp(steps x this) - 1); new damage sets it back to 0")
    ap.add_argument("--idle-cap", type=float, default=reward.Weights.idle_cap,
                    help="the idle cost at most --idle x this")
    ap.add_argument("--idle-rate", type=float, default=reward.Weights.idle_rate,
                    help="0: any damage resets the idle timer; > 0: only a damage rate of at least this share of "
                         "the budget a minute (defender gold lost, mean over --idle-window)")
    ap.add_argument("--idle-window", type=float, default=reward.Weights.idle_window_s,
                    help="s: the time constant of that damage rate")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--max-units", type=int, default=19, help="random armies (tools/nn/armies): units a side at most")
    ap.add_argument("--bank", type=int, default=2048, help="generated battles ready at once")
    ap.add_argument("--armies", choices=("v1", "v2", "v3"), default="v1",
                    help="the training banks' army scheme (tools/nn/armies/generate.py): v1 a budget, then armies under "
                         "it; v2 (user 11.10) the numbers of units first (15-19 40%%, 10-14 30%%, 5-9 20%%, 1-4 10%%), "
                         "a ceiling 2000-15000, the other side for the same gold; v3 (23.10) as v2 with the numbers "
                         "evenly 1-19, 5%% of battles both sides full, a ceiling 2000-20000. Evaluations keep v1, the "
                         "exam's proxy v2")
    ap.add_argument("--drills", type=float, default=0.0,
                    help="share of the battles that are drills (tools/nn/train/drills; e.g. 0.08), taken from the "
                         "other opponents in proportion")
    ap.add_argument("--drill-weights", help="the drills' shares of --drills as json, e.g. {\"pincer\": 1, \"kiting\": 2} "
                                            "(default: drills.TRAIN equally - kiting, hold_fire; counter is READY but "
                                            "not trained by default: it costs in normal battles)")
    ap.add_argument("--kite-share", type=float, default=0.0,
                    help="share of the self / past battles started that are kite battles (tools/nn/train/kitesp.py: "
                         "fast missile units against slower infantry, equal gold, the network on both sides, no "
                         "script or teacher; e.g. 0.1); the log's \"kite\": the shooters' kiting episodes and trade")
    ap.add_argument("--kite-bank", type=int, default=kitesp.BANK_DEFAULT,
                    help="kite battles ready at once (renewed with the bank)")
    ap.add_argument("--drill-bank", type=int, default=256, help="battles of each drill ready at once (renewed with the bank)")
    ap.add_argument("--drill-broad", type=float, default=drills.BROAD_DEFAULT,
                    help="share of every drill's battles from its broad frame (the situation in a messier battle: more "
                         "unit types, uninvolved units, lords; tools/nn/train/drills), the rest from the clean frame; "
                         "also the drill evaluation's mix (test5)")
    ap.add_argument("--drill-embed", type=float, default=drills.EMBED_DEFAULT,
                    help="share of the battles of a drill that has an embedded frame (kiting, hold_fire) from it: the "
                         "situation inside a normal generated battle (both lords, normal sizes, no other cue); the "
                         "rest clean / broad by --drill-broad; also the drill evaluation's mix (test5); 0: the old mix")
    ap.add_argument("--teach-normal", default="",
                    help="the teacher in NORMAL battles (tools/nn/train/teach_auto.py Transfer): the drill's teacher "
                         "script labels our units in the ordinary training battles at the drill's moments only (its "
                         "transfer detector's situation), a share of those battles: 'auto' (every drills.TRAIN drill, "
                         "adaptive: share = min(cap, k x gap), gap = max(0, ai_like's applied share - the network's) / "
                         "ai_like's from every test5 evaluation's transfer block, 0 within --teach-normal-match), "
                         "'kiting' (names, adaptive) or json {\"kiting\": 0.1} (fixed shares)")
    ap.add_argument("--teach-normal-k", type=float, default=teach_auto.NORMAL_K, help="--teach-normal adaptive: k")
    ap.add_argument("--teach-normal-cap", type=float, default=teach_auto.NORMAL_CAP,
                    help="--teach-normal adaptive: the share of the normal battles labelled at most this")
    ap.add_argument("--teach-normal-weight", type=float, default=teach_auto.NORMAL_WEIGHT,
                    help="--teach-normal: the imitation weight on a labelled unit (the pull is weight x share)")
    ap.add_argument("--teach-normal-match", type=float, default=teach_auto.NORMAL_MATCH,
                    help="--teach-normal adaptive: no labels while the gap is at most this")
    ap.add_argument("--drill-teach", help="the drills' teacher: 'auto' (tools/nn/train/teach_auto.py: every READY drill "
                                          "the run plays, a share of its battles labelled from how far the network is "
                                          "behind the skilled script, renewed after every test5 evaluation) or, manual, "
                                          "json {drill: weight}, e.g. {\"kiting\": 0.5}: in that drill's battles + "
                                          "weight x cross-entropy of the policy against the drill's skilled script "
                                          "(labels only; tools/nn/train/drills/teach.py), the weight going linearly to 0 "
                                          "over --drill-teach-minutes")
    ap.add_argument("--drill-teach-k", type=float, default=teach_auto.K,
                    help="auto: share = min(cap, k x deficit), deficit = max(0, skilled - net win) / skilled")
    ap.add_argument("--drill-teach-cap", type=float, default=teach_auto.CAP, help="auto: the share at most this")
    ap.add_argument("--drill-teach-weight", type=float, default=teach_auto.WEIGHT,
                    help="auto: the imitation weight on a labelled unit (fixed; the pull is weight x share)")
    ap.add_argument("--drill-teach-minutes", type=float, default=10.0,
                    help="minutes of training over which the teacher's weight goes to 0")
    ap.add_argument("--observe-dir", help="learning by observation (tools/nn/observe): the converted battles "
                                          "(build/observe/data); off without it")
    ap.add_argument("--observe-share", type=float, default=0.1,
                    help="demonstration decisions beside every minibatch, a share of the minibatch's (0.1-0.25)")
    ap.add_argument("--observe-weight", type=float, default=0.1, help="the imitation term's weight at the start")
    ap.add_argument("--observe-decay", type=float, default=0.95,
                    help="... x this every update (carried over the chain's parts; 1: constant)")
    ap.add_argument("--observe-anchor", type=float, default=0.2,
                    help="weight of KL(policy || --init frozen) on the PPO rows while observing (0: off)")
    ap.add_argument("--observe-anchor-decay", type=float, default=0.995, help="... x this every update")
    ap.add_argument("--observe-anchor-alone", action="store_true",
                    help="keep the KL anchor after the term is off (its weight negligible, observe/store.py): by "
                         "default it ends with the term")
    ap.add_argument("--observe-adv", default="critic", choices=("critic", "mc", "window20"),
                    help="which advantage selects the demonstrations (A > 0) and weighs them: critic (GAE of the "
                         "reward v2 to the battle's end minus the network's critic), mc (the plain return minus the "
                         "critic), window20 (the fallback without the critic: the gold trade of the next 20 s minus "
                         "the battle's mean)")
    ap.add_argument("--observe-beta", type=float, default=1.0,
                    help="the weight exp(A / (beta x std A)) of a selected demonstration")
    ap.add_argument("--observe-clip", type=float, default=20.0, help="... at most this")
    ap.add_argument("--observe-steps", type=int, default=0, help="decisions per demonstration chunk (0: --steps)")
    ap.add_argument("--observe-refresh", type=int, default=20,
                    help="updates between recomputing the demonstrations' stored memories")
    ap.add_argument("--bank-refresh", type=float, default=5.0, help="minutes between new banks of armies")
    ap.add_argument("--no-bank-ahead", dest="bank_ahead", action="store_false",
                    help="build each new bank when it is due (the training waits) instead of in a process ahead "
                         "(tools/nn/train/bankproc.py)")
    ap.add_argument("--ability-branch", type=float, default=0.0, metavar="SHARE",
                    help="abilities by their effect (ability_branch.py): this share of the learner's presses is replayed "
                         "in the simulator pressed and not pressed (common random numbers), the trade's difference is "
                         "the ability head's extra advantage; 0: off")
    ap.add_argument("--ability-branch-missed", type=float, default=ability_branch.Config.missed,
                    help="... the chance a moment of a ready ability not pressed (a unit, a decision) is replayed too")
    ap.add_argument("--ability-branch-copies", type=int, default=ability_branch.Config.copies,
                    help="... copies of each branch")
    ap.add_argument("--ability-branch-horizon", type=float, default=ability_branch.Config.horizon_s,
                    help="... seconds of battle each branch plays")
    ap.add_argument("--ability-branch-hold", type=float, default=ability_branch.Config.hold_s,
                    help="... in the 'not pressed' branch the slot also stays unused this many seconds (0: the network "
                         "goes on and may press it a decision later)")
    ap.add_argument("--ability-branch-openloop", action="store_true",
                    help="... 'not pressed' plays the orders 'pressed' gave (both sides, every decision) instead of "
                         "deciding again: only the press's own effect differs (far less noise)")
    ap.add_argument("--ability-branch-greedy", action="store_true",
                    help="... the branches' networks take their most likely action instead of sampling")
    ap.add_argument("--ability-branch-cap",type=int, default=ability_branch.Config.cap,
                    help="... snapshots a chunk at most (0: no cap)")
    ap.add_argument("--ability-branch-batch", type=int, default=ability_branch.Config.batch,
                    help="... snapshots per branch batch (x copies battles in each branch)")
    ap.add_argument("--ability-branch-weight", type=float, default=0.1,
                    help="... the ability head's term's weight at the start (ppo.ability_term)")
    ap.add_argument("--ability-branch-decay", type=int, default=300,
                    help="... it goes to 0 linearly over this many updates of the chain with it on (0: no decay)")
    ap.add_argument("--ability-critic", action="store_true",
                    help="... the ability critic (ability_critic.py): learns the replayed prices, and once it beats the "
                         "mean it prices every moment an ability was ready (needs --ability-branch)")
    ap.add_argument("--ability-critic-weight", type=float, default=ability_branch.Config.critic_weight,
                    help="... its predicted price x this at the moments not replayed")
    ap.add_argument("--ability-critic-warmup", type=int, default=ability_branch.Config.critic_warmup,
                    help="... replayed moments in its buffer before it is used")
    ap.add_argument("--ability-critic-buffer", type=int, default=ability_branch.Config.critic_buffer,
                    help="... replayed moments it keeps (saved with the checkpoint)")
    ap.add_argument("--ability-own-signal", action="store_true",
                    help="the ability head learns only from its own term (needs --ability-branch): its log-probability "
                         "and entropy are out of the main surrogate; the term's weight is --ability-weight (no decay); "
                         "the body gets the term's gradient x --ability-body-grad (ppo.py)")
    ap.add_argument("--ability-body-grad", type=float, default=None,
                    help="... the gradient into the ability head's input x this (default 0.1 under the own signal; "
                         "0: none)")
    ap.add_argument("--ability-weight", type=float, default=None,
                    help="... the ability head's own term's weight (default 1.0 under the own signal)")
    ap.add_argument("--ability-lr-mult", type=float, default=1.0,
                    help="the ability head's parameters (ppo.ABILITY_HEAD) learn at --lr x this (an Adam group of their "
                         "own; 1: as the rest)")
    ap.add_argument("--ability-epochs", type=int, default=1,
                    help="passes of the ability term over each update's priced rows: the main one + this - 1 of the "
                         "ability head alone on the stored thoughts, each clipped on the ability ratio (needs "
                         "--ability-branch; ppo.ability_epochs)")
    ap.add_argument("--ability-price", default="auto", choices=("auto", "branch", "bonus"),
                    help="... the press's price: bonus - a valid ability's by its bonuses (config/ability_price.json, "
                         "abilities_price/train_price.py), the others by the branches; branch - all by the branches; "
                         "auto - bonus under the own signal, else branch")
    ap.add_argument("--ability-norm", default="shared", choices=ability_branch.NORMS,
                    help="... the prices' scale before the clip: shared - one root mean square of all abilities' prices "
                         "(the biggest set it); per-ability - each ability's own running one (kept in the checkpoint; "
                         "ability_branch.AbilityNorm)")
    ap.add_argument("--ability-price-file", default="",
                    help="... the bonus price's numbers (default config/ability_price.json)")
    ap.add_argument("--spell-price", default="direct", choices=ability_branch.SPELL_PRICES,
                    help="... a damage spell's cast: direct - by direct accounting (its bombardment's enemy gold less the "
                         "own, less k_winds x the winds spent; tools/nn/train/spell_price.py), not replayed; branch - "
                         "replayed as any ability")
    ap.add_argument("--gpu-mem", type=float, default=1.0,
                    help="share of the GPU's memory training may hold (game mode: e.g. 0.5, so a game keeps memory)")
    ap.add_argument("--amp", default="", choices=("", "bf16", "bf16h"),
                    help="the PPO update's passes under bfloat16 autocast (ppo.PPOConfig.amp)")
    ap.add_argument("--cuda-alloc", default="expandable", choices=("default", "expandable"),
                    help="the CUDA allocator's segments (cuda_alloc; default: expandable)")
    ap.add_argument("--gpu-duty", type=float, default=1.0,
                    help="share of the time the GPU works (e.g. 0.9: after each update rest 1/9 of its time; "
                         "quieter fans, a responsive desktop; the rest is not counted as training time)")
    ap.add_argument("--general", action="store_true",
                    help="S28: the general (tools/nn/general, docs/en/training/general.md): the unit network becomes its "
                         "executor (the old weights, a new task input at zero) with a fresh general that plans groups, "
                         "missions and the deployment; both learn by PPO. An --init with a general keeps its own (its "
                         "sizes and rules; a rule's option given on the command line goes over the saved one, a size's "
                         "option against the saved size stops the run: general_config); off by default")
    ap.add_argument("--general-card", type=int, default=64, help="general: the cards' width (64-128; simplification 3)")
    ap.add_argument("--general-hidden", type=int, default=128, help="general: the army reader's first layer")
    ap.add_argument("--general-council-layers", type=int, default=2, help="general: the council's layers")
    ap.add_argument("--general-group-layers", type=int, default=1, help="general: the group council's layers")
    ap.add_argument("--general-heads", type=int, default=4, help="general: attention heads of both councils")
    ap.add_argument("--general-places", type=int, default=10, help="general: group places a side")
    ap.add_argument("--general-kinds", type=int, default=6,
                    help="general: mission kinds (the first of tools/nn/general/config.py MISSION_KINDS)")
    ap.add_argument("--general-target-grid", type=int, default=4,
                    help="general: a mission's target is a group or a tile of this coarse grid of the side's frame "
                         "(4: 4 x 4 tiles of 400 m; it must split the unit network's sectors evenly)")
    ap.add_argument("--general-rounds", type=int, default=3, help="general: rounds of the group collection")
    ap.add_argument("--general-sharp", type=float, default=15.0,
                    help="general: the group collection's keys and queries to unit length, their cosine x a learned "
                         "sharpness starting here (0: the plain dot product - a fresh general then split every unit "
                         "evenly over the places)")
    ap.add_argument("--general-kind-stick", type=float, default=1.0,
                    help="general: a re-plan's bonus on a group's logit of its last plan's mission kind (the group that "
                         "kept its number and some of its units; a fixed rule, saved with the checkpoint; 0: none)")
    ap.add_argument("--general-task-mask", type=lambda v: bool(int(v)), default=True,
                    help="general: 1 the mission limits the executor - a unit's kinds, targets and sectors cut to its "
                         "group's mission (tools/nn/general/taskmask.py) - and 'shoot' only for a group with a shooter; "
                         "0: neither (the executor as before, to the bit). Saved with the checkpoint")
    ap.add_argument("--general-task-mask-parts", default=MASK_PARTS,
                    help="general: with --general-task-mask 1, which masks (a comma list of tools/nn/general/taskmask.py "
                         "PARTS: withdraw_attack, withdraw_move, shoot_melee, shoot_target, attack_target, attack_move, "
                         "hold_target, hold_move, screen_move, shoot_group, shoot_volley; 'all'; 'none'; default: those harmless at "
                         "minute 0, docs 'Task masks'). Saved with the checkpoint")
    ap.add_argument("--general-shoot-share", type=float, default=0.5,
                    help="general: with --general-task-mask 1, 'shoot' only for a group whose shooters (a missile range "
                         "on the card) are at least this share of its units (a rule, saved with the checkpoint)")
    ap.add_argument("--general-melee-stay", type=lambda v: bool(int(v)), default=True,
                    help="general: 1 'withdraw' not for a group fighting in melee that has not wavered (some unit in "
                         "melee, under general-waver-share of it wavering or worse); the teacher and the executor's "
                         "mask (a steady unit in melee: no move / withdraw) keep the rule too; 0: withdraw anywhere")
    ap.add_argument("--general-shoot-line-m", type=float, default=20.0,
                    help="general: shooters under 'shoot' go no farther than this ahead of our infantry groups under "
                         "'hold' (the executor's mask part shoot_line); 0: off")
    ap.add_argument("--general-waver-share", type=float, default=0.5,
                    help="general: a group has wavered (withdraw allowed in melee) when at least this share of its "
                         "units waver, rout or are shattered")
    ap.add_argument("--general-groups", default="rule", choices=("rule", "net"),
                    help="general: the group collection - 'rule' (tools/nn/general/groups.py: roles, then clusters by "
                         "place, both sides alike; the attention's collection learns nothing) or 'net' (the learned slot "
                         "attention; the atlas of g17r: degenerate). Saved with the checkpoint (an older one: net)")
    ap.add_argument("--general-group-r", type=float, default=100.0,
                    help="general: with --general-groups rule, units of a role join while their clusters' centres are "
                         "closer than this (m)")
    ap.add_argument("--general-lord-join-m", type=float, default=60.0,
                    help="general: with --general-groups rule, a lone lord or hero joins the nearest melee group within "
                         "this (m)")
    ap.add_argument("--general-group-centre", type=lambda v: bool(int(v)), default=True,
                    help="general: with --general-groups rule, 1 a group card reads its units' values less their side's "
                         "mean (how the group differs from its army)")
    ap.add_argument("--general-split-kinds", type=lambda v: bool(int(v)), default=True,
                    help="general: 1 the executor reads its own mission kind table (the group card detached): its epochs "
                         "do not move the general's choice; 0: one table (an older checkpoint: 0)")
    ap.add_argument("--general-tile-feats", default="v2", choices=("v1", "v2"),
                    help="general: the tiles' numbers - v2: the three always zero in the simulator replaced by own and "
                         "enemy shooters' and enemy fast units' strength; v1: the sectors' 8 (an older checkpoint: v1)")
    ap.add_argument("--general-tile-norm", type=lambda v: bool(int(v)), default=True,
                    help="general: 1 the tiles' numbers standardised over the map (an older checkpoint: 0)")
    ap.add_argument("--general-threat", type=lambda v: bool(int(v)), default=True,
                    help="general: 1 the groups' threat numbers (nearest enemy group's distance, strength ratio, "
                         "shooter shares, contact; the shares under fire and in melee) on their cards before the kind "
                         "and target (an older checkpoint: 0)")
    ap.add_argument("--general-group-tiles", type=lambda v: bool(int(v)), default=True,
                    help="general: 1 the tiles under every group's centre (within 1.5 tiles, by distance) on its card - "
                         "ours before the kind and target, theirs as targets (a layer from zero: an older checkpoint "
                         "plays as before until it learns)")
    ap.add_argument("--general-task-group", type=lambda v: bool(int(v)), default=False,
                    help="general: 1 the executor's task carries the group card; 0 zeros there (the width kept; an "
                         "older checkpoint: 1)")
    ap.add_argument("--general-kl", default="k3", choices=("k1", "k3"),
                    help="general: the KL stop's estimate - k3 (r - 1) - log r >= 0; k1 the mean log-ratio (went below "
                         "0 as the policy sharpened: no stop)")
    ap.add_argument("--general-stick", type=float, default=1.0,
                    help="general: a re-plan's bonus on a unit's logit of its last group in the group collection (a "
                         "learned value, starts here; the groups' inertia - 0: none)")
    ap.add_argument("--general-center", type=float, default=0.7,
                    help="general: the group collection's keys less this x their side's mean key (a learned share; "
                         "0: plain slot attention - a fresh general then put nearly every unit into one group)")
    ap.add_argument("--general-tiles", default="sectors", choices=("sectors", "cnn"),
                    help="general: the map tiles - the unit network's sectors (simplification 1; cnn: not built yet)")
    ap.add_argument("--general-effects", default="off", choices=("off", "on"),
                    help="general: the effect cards (simplification 2: off = zeros; on: not built yet)")
    ap.add_argument("--general-zone", default="place", choices=("place", "map"),
                    help="general: the deployment zone - place.py's area (simplification 5; map: not built yet)")
    ap.add_argument("--general-deploy", default="net", choices=("net", "script"),
                    help="general: net - the general places its groups in the zone, a rule lays the units out; script - "
                         "place.py's placement as before (the general still forms groups and gives missions)")
    ap.add_argument("--general-replan-s", type=float, default=30.0, help="general: seconds of battle between plans")
    ap.add_argument("--general-replan-min-s", type=float, default=3.0,
                    help="general: an event's re-plan (a group broken, a lord fallen, the enemy near) not sooner than "
                         "this after a plan")
    ap.add_argument("--general-teach-stall-s", type=float, default=60.0,
                    help="general: the teacher's wait runs out (all attack, a re-plan) when its farthest waiting "
                         "group has come no nearer the enemy for this long (s; 0: off)")
    ap.add_argument("--general-teach-wait-s", type=float, default=120.0,
                    help="general: ... or when the wait has gone on this long since it began (s; 0: off)")
    ap.add_argument("--general-replan-near-m", type=float, default=250.0,
                    help="general: the re-plan's event 'the enemy near' - a standing enemy unit within this (m) of one "
                         "of our groups' centres that was farther at the plan; 0: no such event")
    ap.add_argument("--general-entropy-kind", type=float, default=0.001,
                    help="general: the entropy bonus on the mission kind's choice (simplification 4: on; each head "
                         "apart: the kind's ~ln 10 and the target's ~ln 36 are not one sum)")
    ap.add_argument("--general-entropy-target", type=float, default=0.001,
                    help="general: the entropy bonus on the target's choice")
    ap.add_argument("--general-entropy-place", type=float, default=0.001,
                    help="general: the entropy bonus on the deployment place's choice")
    ap.add_argument("--general-epochs", type=int, default=2, help="general: PPO epochs over the ended plans")
    ap.add_argument("--general-minibatch", type=int, default=512, help="general: plans a minibatch")
    ap.add_argument("--general-value", type=float, default=0.5, help="general: the plan value's loss weight")
    ap.add_argument("--general-lr-mult", type=float, default=1.0, help="general: its network learns at --lr x this")
    ap.add_argument("--general-only", type=int, default=0, choices=(0, 1),
                    help="general: 1 - only the general learns (its plan PPO, the imitation, the kind Q); the executor "
                         "(the unit network, its task input) and the critic are frozen bit for bit, their terms not "
                         "computed (the executor's PPO, the critic, the heads' entropy, the ability branches and "
                         "critic, the mission bonus, the demonstrations); needs a general (--general)")
    ap.add_argument("--general-return", default="span", choices=("span", "td", "critic"),
                    help="general: a plan's return - span: the reward from its decision to the next plan or the end "
                         "(the design); td: that + the next plan's discounted value (the value then of the battle's "
                         "rest); critic: the plan's discounted reward + gamma^n x the unit critic's value where it ended "
                         "(0 at the battle's end), its advantage less the critic's value where it began (the plan "
                         "value learns the same target beside it)")
    ap.add_argument("--general-branches", type=float, default=0.0,
                    help="general: P, the share of the learner's plans replayed with one group's mission kind changed "
                         "(tools/nn/general/branch.py): the price 'this kind against another' is the kind choice's "
                         "advantage (0: off, nothing changes)")
    ap.add_argument("--general-branch-copies", type=int, default=4,
                    help="general branches: copies of each pair (common random numbers), each with its own other kind")
    ap.add_argument("--general-branch-horizon", type=float, default=60.0,
                    help="general branches: seconds of battle each branch plays (the plan frozen)")
    ap.add_argument("--general-branch-cap", type=int, default=128,
                    help="general branches: snapshots an update at most (0: no cap); <= the batch: one batch an update")
    ap.add_argument("--general-branch-batch", type=int, default=64,
                    help="general branches: snapshots a branch batch (x copies battles in each of its two branches)")
    ap.add_argument("--general-branch-alt", default="teacher", choices=("teacher", "uniform", "policy", "same"),
                    help="general branches: the other kind - teacher: the kind and target the teacher's rule would give "
                         "the group there (the price: my choice against the teacher's; where it gives the same kind or "
                         "none: uniform, the log's alt_fallback), uniform over the others, or by the general's "
                         "probabilities without the chosen one (the price then x (1 - pi(chosen))); same (a diagnosis): "
                         "the chosen kind again with another seed - the mean raw price is the loop's own bias, nothing "
                         "trains on it")
    ap.add_argument("--general-branch-price", default="trade", choices=("trade", "reward"),
                    help="general branches: the price - the gold exchange at the horizon, or the side's reward over it")
    ap.add_argument("--general-branch-kinds", default="all", choices=("all", "target"),
                    help="general branches: the other kinds - all (every kind the group could take, each with its own "
                         "target) or target (only those taking the group's target: biased, withdraw never met attack)")
    ap.add_argument("--general-branch-greedy", action="store_true",
                    help="general branches: the networks in the branches take their most likely action")
    ap.add_argument("--general-branch-weight", type=float, default=1.0,
                    help="general branches: the weight of their term in the general's loss (without the kind head's "
                         "own signal; with it: --general-kind-weight); under --general-kind-q both also the branches' "
                         "weight beside the outcome Q's groups")
    ap.add_argument("--general-branch-center", type=int, default=1, choices=(0, 1),
                    help="general branches: 1 - the prices less their running mean over all pairs (~10 updates; the "
                         "partial open loop's bias), before the advantage, the kind critic and the log's better/worse "
                         "(the raw ones logged too); 0 - as measured")
    ap.add_argument("--general-branch-openloop", type=int, default=1, choices=(0, 1),
                    help="general branches: 1 - partial open loop: the 'other' copy plays the 'chosen' copy's orders "
                         "(both sides, the scripts) for every unit but the swapped group's, only that group decides "
                         "anew (branch.py); 0 - closed loop, everybody decides anew in both")
    ap.add_argument("--general-kind-critic", type=int, default=1, choices=(0, 1),
                    help="general: 1 - the kind critic (tools/nn/general/kind_critic.py) learns the branches' prices and "
                         "gives every group of every ended plan the kind's advantage Q(chosen) - E_pi Q once ready")
    ap.add_argument("--general-kind-critic-warmup", type=int, default=1000,
                    help="general kind critic: prices before it may be used")
    ap.add_argument("--general-kind-critic-buffer", type=int, default=25000,
                    help="general kind critic: prices it keeps")
    ap.add_argument("--general-kind-norm", default="per-kind", choices=("per-kind", "chunk"),
                    help="general branches: the prices' scale - each kind's running root mean square over ~10 updates, "
                         "or the chunk's over all rows (before)")
    ap.add_argument("--general-kind-q", default="outcome", choices=("outcome", "branch", "both"),
                    help="general: the kind head's signal - outcome (the default): the outcome Q "
                         "(tools/nn/general/kind_q.py: every group of every ended plan against the trade over "
                         "--general-kind-q-horizon s after the plan in the battle itself; Q(chosen) - E_pi Q once "
                         "ready; the branches, if on, only measured and logged: kind_q_agree, alt same); branch: the "
                         "branches and the kind critic on their prices (before; the critic learns only here); both: the "
                         "Q's groups + --general-branch-weight x the branches' rows")
    ap.add_argument("--general-kind-q-horizon", type=float, default=60.0,
                    help="general kind Q: seconds of battle after a plan's start its outcome covers (the battle's end "
                         "earlier: up to it, marked)")
    ap.add_argument("--general-kind-q-warmup", type=int, default=10000,
                    help="general kind Q: groups in its buffer before it may be used")
    ap.add_argument("--general-kind-q-buffer", type=int, default=30000,
                    help="general kind Q: groups it keeps (~4 updates)")
    ap.add_argument("--general-kind-q-ev-min", type=float, default=0.05,
                    help="general kind Q: used once its held-out explained variance of the outcome is at least this")
    ap.add_argument("--general-kind-q-shrink", type=int, default=1, choices=(0, 1),
                    help="general kind Q: 1 - Q x (the held-out coefficient of its kind part less two standard "
                         "errors by battle, within [0, 1]); 0 - as is")
    ap.add_argument("--general-kind-own-signal", type=int, default=1, choices=(0, 1),
                    help="general: 1 - the kind head's own signal: its log-probability out of the plan's PPO, it learns "
                         "from the branches and the kind critic (weight --general-kind-weight, no decay) and the "
                         "imitation; 0 - the plan's advantage + --general-branch-weight x the branches' term")
    ap.add_argument("--general-kind-weight", type=float, default=1.0,
                    help="general: the kind term's weight under --general-kind-own-signal 1")
    ap.add_argument("--general-kind-lr-mult", type=float, default=2.0,
                    help="general: the kind head's lr x this (x --general-lr-mult) under --general-kind-own-signal 1")
    ap.add_argument("--general-kind-epochs", type=int, default=2,
                    help="general: passes of the kind term under --general-kind-own-signal 1 (the extra ones move the "
                         "kind head alone)")
    ap.add_argument("--general-distinct", type=float, default=0.0,
                    help="general: W, the executor's bonus for missions that tell apart (tools/nn/general/distinct.py): "
                         "a classifier guesses a group's mission kind from its units' behaviour during the plan; each "
                         "group's W x (log q(kind) - log(1/K)) x the executor's advantage std (a running mean of "
                         "ppo's adv_std_*: W = 1 pays one advantage std a nat of the guess) goes into the side's "
                         "reward where the plan ends (0: off)")
    ap.add_argument("--general-distinct-decay", type=int, default=600,
                    help="general: the distinct bonus's W down to 0 linearly over this many updates of the chain (the "
                         "counter goes on over its parts; 0: no decay); the classifier goes on learning and is logged")
    ap.add_argument("--general-exec", type=float, default=1.0,
                    help="general: W, the executor's bonus for carrying out its missions (tools/nn/general/mission.py: "
                         "every decision each unit's measure 0..1 of its group's mission kind - attack, hold, flank, "
                         "screen, shoot, withdraw - towards its target): W x (the measure - its kind's running mean) "
                         "into the unit's own advantage as a reward-to-go (in the normalised advantage's units: W = 1 "
                         "pays one advantage std for a measure 1 above the mean) - never into the reward, so the critic "
                         "and the general's return do not see it (0: measured and logged, not paid)")
    ap.add_argument("--general-exec-decay", type=int, default=0,
                    help="general: the mission bonus's W down to 0 linearly over this many updates of the chain (0: no "
                         "decay)")
    ap.add_argument("--general-teacher", type=float, default=0.0,
                    help="general: P, the share of the learner's battles whose missions a scripted teacher gives "
                         "(tools/nn/general/teacher.py: attack the nearest enemy group, shooters shoot, one group "
                         "screens the shooters, a threatened lord withdraws); those plans train the executor only, not "
                         "the general (0: off)")
    ap.add_argument("--general-teacher-hold-m", type=float, default=250.0,
                    help="general: the teacher's groups hold their tile while the nearest enemy unit is farther than "
                         "this (m) ...")
    ap.add_argument("--general-teacher-fire-s", type=float, default=30.0,
                    help="general: ... and until our shooters have been firing this long (s); then they attack")
    ap.add_argument("--general-teacher-flank", type=int, default=1, choices=(0, 1),
                    help="general: 1 - where two or more of the teacher's groups attack one enemy group, the one nearest "
                         "to the target's front pins it (attack), the others flank it (teacher.py); 0 - all attack")
    ap.add_argument("--general-flank-measure", default="contact", choices=("contact", "angle"),
                    help="general: the flank mission's measure in melee - 'contact': the side of the target the "
                         "contact began on (the simulator's hit_sec, morale 'attacked in the flank / rear'); 'angle': "
                         "where the unit stands now off the target unit's facing (mission.py)")
    ap.add_argument("--general-exec-kind", default="withdraw=2,screen=2",
                    help="general: the executor's mission bonus x each kind's weight, 'name=w,...' (kinds not named: 1)")
    ap.add_argument("--general-imitate", type=float, default=0.3,
                    help="general: W, the teacher's imitation - on the teacher's ended plans (--general-teacher) the "
                         "general's loss + W x the mean negative log-probability of the teacher's kind and target over "
                         "the groups it gave a mission (the groups are the network's; the general's weights only; 0: "
                         "off, the nll and accuracies still logged)")
    ap.add_argument("--general-imitate-decay", type=int, default=300,
                    help="general: the imitation's W down to 0 linearly over this many updates of the chain (its own "
                         "counter, from its first update; 0: no decay)")
    ap.add_argument("--general-imitate-min", type=float, default=0.15,
                    help="general: the floor M of the imitation's decay - W goes down to M, not to 0 (the general's own "
                         "signal is weaker than the teacher's for now: g20-g21 without it lost the rating); saved in the "
                         "checkpoint, the option given on the command line over it")
    ap.add_argument("--general-exec-general", type=float, default=0.2,
                    help="general: S, its small bonus for missions carried out - each group's plan advantage + S x the "
                         "group's measure less its kind's running mean over the plan (the executor's centring: no gain "
                         "for easy kinds), in the normalised advantage's units (0: off, the plain PPO)")
    ap.add_argument("--eval-generated", type=int, default=512, help="random battles per opponent (EVAL_SEEDS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args()
    if args.gpu_mem < 1.0 and torch.cuda.is_available():
        # game mode (rule 15): cap the caching allocator so a game beside training keeps video memory (10.10: training
        # held 15.7 of 16.3 GB and the user's game fell from 200 to 21 fps; --gpu-duty caps time, not memory)
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem)
    actor, summary, out = train(args)
    checkpoint.save(checkpoint.LATEST, actor, None, args.preset, checkpoint.meta(out / "latest.pt"))
    if not args.no_eval:
        final_eval(out / "latest.pt", args, summary, out)


if __name__ == "__main__":
    main()
