"""One decision of a side: observation -> actor -> sampled orders for the simulator (and later the game).

    obs, memory = observation.observe(state, setup, side, memory)
    orders, h, logits, action = decide.act(actor, obs, setup, h)

Only the side's units that take orders get one; every other unit (enemy, dead, routing) holds.
"""
import numpy as np
import torch

from tools.nn.model import heads as hd
from tools.nn.model import observation as ob
from tools.nn.model import policy
from tools.nn.model.frame import Frame
from tools.nn.sim.orders import Orders


def frame_to(frame, device):
    t = lambda v: torch.as_tensor(np.asarray(v) if not torch.is_tensor(v) else v, device=device).float()
    return Frame(t(frame.cx), t(frame.cz), t(frame.ux), t(frame.uz))


def to_orders(cfg, action, obs_t, frame, bounds):
    """An Action -> the simulator's Orders (tools/nn/sim/orders.py), [B, N] in the state's slots."""
    ctrl = obs_t["ctrl"]
    kind = torch.where(ctrl, action.kind, torch.full_like(action.kind, hd.HOLD))
    point = hd.point_world(cfg, action, obs_t, frame, bounds.to(obs_t["pos"].dtype))
    move = (kind == hd.MOVE) | (kind == hd.WITHDRAW)
    here = torch.stack(frame.world(obs_t["pos"][..., 0] * ob.POS, obs_t["pos"][..., 1] * ob.POS), -1)
    point = torch.where(move[..., None], point, here)
    target = torch.where(kind == hd.ATTACK, action.target, torch.full_like(action.target, -1))
    run = action.run & ((kind == hd.MOVE) | (kind == hd.ATTACK))
    return Orders(kind=kind, x=point[..., 0].float(), z=point[..., 1].float(), target=target, run=run)


@torch.no_grad()
def act(actor, obs, setup, h=None, greedy=False, temperature=1.0):
    """(orders, new memory, logits, action) for the batch of one side's observation."""
    device = next(actor.parameters()).device
    obs_t = policy.to_torch(obs, device)
    logits, h = actor(obs_t, h)
    action = hd.sample(logits, greedy, temperature)
    bounds = torch.as_tensor(setup.bounds, device=device)
    orders = to_orders(actor.cfg, action, obs_t, frame_to(obs.frame, device), bounds)
    return orders, h, logits, action
