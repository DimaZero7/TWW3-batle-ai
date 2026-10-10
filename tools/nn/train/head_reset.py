"""A partial reset of the v2 chained heads (run.py --reset-heads; docs/en/training/training.md "Per-head spread").

The named heads' layers start again as a new network's (the same init as tools/nn/model/chain.ChainHeads': the
sector's near weight softplus 1, the target's pair weight 0), everything else stays: the trunk, the other heads, the
critic. Against the primacy bias (Nikishin et al. 2022, "The Primacy Bias in Deep Reinforcement Learning": a network
that over-fitted its early experience is helped by resetting its last layers, the rest - and the data - kept): the
heads of the place and the commitment had collapsed to one choice (build/net_audit, build/observe: the chosen sector's
median probability 0.9994, the cell's 1.0), so neither PPO nor the imitation of other players could find another one.

Outside tools/nn/model on purpose: a file there changes the simulator's version (version.VERSION_FILES).
"""
import torch

# The heads by name and their layers (ChainHeads' attributes). "cond": the condition every part after the kind
# reads (sector, cell, hold and run share it).
PARTS = {"kind": ("kind",), "target": ("q", "k", "target_pair"), "sector": ("place_q", "place_k", "place_near"),
         "cell": ("fine",), "hold": ("commit",), "run": ("run",), "cond": ("cond",)}


def parse(spec):
    """'sector,cell,hold' -> ['sector', 'cell', 'hold'] (checked against PARTS); '' / None -> []."""
    names = [n.strip() for n in (spec or "").split(",") if n.strip()]
    bad = [n for n in names if n not in PARTS]
    if bad:
        raise ValueError(f"--reset-heads {bad}: not heads of {sorted(PARTS)}")
    return names


def reset(actor, names, seed=0):
    """Re-initialise the named heads' layers of a v2 actor in place (module doc). -> the reset parameters' full names
    (actor.named_parameters'). The global random state is left as it was."""
    if not names:
        return []
    if not getattr(actor.cfg, "sectors", 0):
        raise ValueError("--reset-heads: only the v2 chained heads (a v2 network)")
    from tools.nn.model.chain import ChainHeads
    heads = actor.heads
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    fresh = ChainHeads(actor.cfg)
    torch.set_rng_state(state)
    done = []
    for name in names:
        for attr in PARTS[name]:
            mine, new = getattr(heads, attr, None), getattr(fresh, attr, None)
            if mine is None or new is None:          # (target_pair: only with cfg.pairs)
                continue
            mine.load_state_dict(new.state_dict())
            done += [f"heads.{attr}.{k}" for k, _ in mine.named_parameters()]
    return done


def copy(src, dst, params):
    """The parameters `params` (full names) of src into dst (e.g. the reset heads into the KL anchors' frozen
    networks, so they hold the policy near the reset heads, not the collapsed ones)."""
    if dst is None or not params:
        return
    mine = dict(src.named_parameters())
    with torch.no_grad():
        for n, p in dst.named_parameters():
            if n in params:
                p.copy_(mine[n])


def forget(opt, actor, params):
    """Adam's moments of the reset parameters dropped (they start as for a new parameter: the old moments are of the
    collapsed weights)."""
    mine = dict(actor.named_parameters())
    for n in params:
        opt.state.pop(mine[n], None)
