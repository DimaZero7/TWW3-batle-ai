"""The commitment (v2, ModelConfig.sectors > 0): with every new order a unit also chooses how long it keeps it.

The chained heads (chain.py) choose, with a new order (hold, move, attack, withdraw), one of DURATIONS (2, 4, 8,
16 s). Until it runs out the unit's decisions are forced to keep (the order in force goes on): the order kind's
choice is only keep (masked: its log-probability is 0, no gradient, no entropy), except when an INTERRUPT happens -
then the unit decides again at once:
* it came into melee (a fight began: it attacked or was attacked; the token's `melee` was off at the last decision);
* an enemy came to threaten its flank or rear (the token's threat_left / right / rear came on: a charge coming in);
* its attack's target died, routs or shatters, or is no longer seen (no longer in Obs.target_ok, or routing);
* it routed or rallied (its routing state changed);
* its own lord died (the context's own-lord-slain came on).
A unit that chooses keep itself starts no commitment. Routing units take no orders at all (Obs.ctrl).

All from the side's own observation (the unit tokens, Obs.target_ok, the context) and the battle time: the simulator
(tools/nn/train/rollout.py) and the companion (tools/nn/companion, the game's state each second) keep it the same way.
The state is a dict of tensors [B, N] (until: battle time it ends, -1 none; target: the committed attack's target,
-1 none; melee, threat, rout: the flags at the last decision) and lord [B] (own lord dead at the last decision).

The network sees it (Obs "commit" [B, N, 2]): the seconds left / LEFT_MAX and whether the unit is held now; and
Obs "free" [B, N]: the units that decide now (take orders and are not held).
"""
import torch

from tools.nn.model import factions
from tools.nn.model import observation as ob
from tools.nn.sim.orders import ATTACK, KEEP

DURATIONS = (2.0, 4.0, 8.0, 16.0)
LEFT_MAX = 16.0
OWN_LORD = factions.SIZE + 8          # the context's own lord slain (observation._context + lords)
THREAT = tuple(ob.INDEX[n] for n in ("threat_left", "threat_right", "threat_rear"))
KEYS = ("until", "target", "melee", "threat", "rout", "lord")


def start(B, N, device=None):
    """No commitment, nothing seen yet."""
    z = torch.zeros((B, N), device=device)
    f = torch.zeros((B, N), dtype=torch.bool, device=device)
    return {"until": z - 1, "target": torch.full((B, N), -1, dtype=torch.long, device=device), "melee": f,
            "threat": f.clone(), "rout": f.clone(), "lord": torch.zeros(B, dtype=torch.bool, device=device)}


def _flags(obs_t):
    tok = obs_t["tokens"]
    melee = tok[..., ob.INDEX["melee"]] > 0.5
    threat = (tok[..., list(THREAT)] > 0.5).any(-1)
    rout = (tok[..., ob.INDEX["state_routing"]] + tok[..., ob.INDEX["state_shattered"]]) > 0.5
    lord = obs_t["ctx"][:, OWN_LORD] > 0.5
    return melee, threat, rout, lord


def interrupts(state, obs_t):
    """[B, N] the units whose commitment an event ends now (module doc)."""
    melee, threat, rout, lord = _flags(obs_t)
    tgt = state["target"]
    has = tgt >= 0
    t = tgt.clamp(min=0)
    gone = has & (~obs_t["target_ok"].gather(1, t) | rout.gather(1, t))
    return ((melee & ~state["melee"]) | (threat & ~state["threat"]) | (rout != state["rout"]) | gone
            | (lord & ~state["lord"])[:, None])


def inputs(state, obs_t, t):
    """obs_t with the commitment's inputs: "free" [B, N] (decides now), "commit" [B, N, 2] (seconds left / LEFT_MAX,
    held now). t: the battle time [B] (s)."""
    t = torch.as_tensor(t, device=obs_t["own"].device).float().reshape(-1, 1)
    held = obs_t["ctrl"] & (state["until"] > t) & ~interrupts(state, obs_t)
    left = torch.where(held, state["until"] - t, torch.zeros_like(state["until"]))
    out = dict(obs_t)
    out["free"] = obs_t["ctrl"] & ~held
    out["commit"] = torch.stack([left / LEFT_MAX, held.float()], -1)
    return out


def apply(state, obs_t, t, action):
    """The state after the decision (obs_t from inputs(), action with .commit): a free unit's new order starts its
    commitment; a held unit keeps it; an interrupted one, a keep chosen or a unit out of orders: none."""
    t = torch.as_tensor(t, device=obs_t["own"].device).float().reshape(-1, 1)
    held = obs_t["commit"][..., 1] > 0.5
    new = obs_t["free"] & (action.kind != KEEP)
    dur = torch.as_tensor(DURATIONS, device=t.device)[action.commit.clamp(0, len(DURATIONS) - 1)]
    until = torch.where(new, t + dur, torch.where(held, state["until"], torch.full_like(state["until"], -1.0)))
    target = torch.where(new, torch.where(action.kind == ATTACK, action.target, torch.full_like(action.target, -1)),
                         torch.where(held, state["target"], torch.full_like(state["target"], -1)))
    melee, threat, rout, lord = _flags(obs_t)
    return {"until": until, "target": target, "melee": melee, "threat": threat, "rout": rout, "lord": lord}


def reset(state, rows):
    """The rows [B] (bool) start afresh (a new battle), in place."""
    fresh = start(*state["until"].shape, device=state["until"].device)
    for k in KEYS:
        r = rows.reshape(-1, *([1] * (state[k].dim() - 1)))
        state[k].copy_(torch.where(r, fresh[k], state[k]))
