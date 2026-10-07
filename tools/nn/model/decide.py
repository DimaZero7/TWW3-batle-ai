"""One decision of a side: observation -> actor -> sampled orders for the simulator (and later the game).

    obs, memory = observation.observe(state, setup, side, memory)
    orders, h, logits, action = decide.act(actor, obs, setup, h)

Only the side's units that take orders get one; every other unit (enemy, dead, routing) holds.
A unit may get KEEP (code 4): no new order, the one in force goes on (x, z = its place, target -1).
Orders.ability: the slot of an ability to use now (only where obs.abil_ok allowed it), -1 none.
v2 (cfg.sectors): a unit held by its commitment keeps (commit.py); the companion keeps the commitment's state.
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
    ability = None
    if action.ability is not None:
        ability = torch.where(ctrl & (action.ability >= 0), action.ability, torch.full_like(action.ability, -1))
    return Orders(kind=kind, x=point[..., 0].float(), z=point[..., 1].float(), target=target, run=run,
                  ability=ability)


def v2_inputs(actor, obs_t, frame, bounds, commit_state, t):
    """v2 (actor.cfg.sectors): obs_t with the commitment's inputs (commit.inputs, from commit_state; None: nothing
    held) and the geometry (sectors.geo). Other actors: obs_t as it is."""
    if not actor.cfg.sectors:
        return obs_t
    from tools.nn.model import commit, sectors
    B, N = obs_t["own"].shape
    state = commit_state if commit_state else commit.start(B, N, obs_t["own"].device)
    t = torch.zeros(B, device=obs_t["own"].device) if t is None else t
    out = commit.inputs(state, obs_t, t)
    out["geo"] = sectors.geo(frame, bounds)
    return out


@torch.no_grad()
def act(actor, obs, setup, h=None, greedy=False, temperature=1.0, abilities=True, commit_state=None, t=None):
    """(orders, new memory, logits, action) for the batch of one side's observation. abilities: the
    network also chooses its abilities (Orders.ability). v2: commit_state, a dict (commit.py) kept between the
    decisions of a battle (an empty one at its start), is updated in place; t: the battle time [B] (s)."""
    device = next(actor.parameters()).device
    obs_t = policy.to_torch(obs, device)
    bounds = torch.as_tensor(setup.bounds, device=device)
    frame = frame_to(obs.frame, device)
    if t is not None:
        t = torch.as_tensor(np.asarray(t, dtype=np.float32).reshape(-1), device=device)
    obs_t = v2_inputs(actor, obs_t, frame, bounds, commit_state, t)
    logits, action, h = actor.act(obs_t, h, greedy, temperature, abilities)
    orders = to_orders(actor.cfg, action, obs_t, frame, bounds)
    if actor.cfg.sectors and commit_state is not None:
        from tools.nn.model import commit
        B = obs_t["own"].shape[0]
        state = commit_state if commit_state else commit.start(B, obs_t["own"].shape[1], device)
        commit_state.update(commit.apply(state, obs_t, torch.zeros(B, device=device) if t is None else t, action))
    return orders, h, logits, action
