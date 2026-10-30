"""One decision of a side: observation -> actor -> sampled orders for the simulator (and later the game).

    obs, memory = observation.observe(state, setup, side, memory)
    orders, h, logits, action = decide.act(actor, obs, setup, h)

Only the side's units that take orders get one; every other unit (enemy, dead, routing) holds.
A unit may get KEEP (code 4): no new order, the one in force goes on (x, z = its place, target -1).
Orders.ability: the slot of an ability to use now (only where obs.abil_ok allowed it), -1 none; a spell's slot
(abilities.SPELL_AT + k) casts at Orders.cast_target, the target pointer's choice.
v2 (cfg.sectors): a unit held by its commitment keeps (commit.py); the companion keeps the commitment's state.
v2 with cfg.form: Orders.face / Orders.width of the formation's heads (chain.py; keep: -1) - the facing a direction of the
side's frame as the world's bearing (face_world), the width a step between the unit's narrowest and widest frontage of
obs "wlim" (width_metres: act's wlim, the simulator's reform limits - tools/nn/general/form.py).
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


def face_world(face, frame, faces=None):
    """[B, N] the world's bearing (degrees 0..360, as the state's b) of the facing head's choice face [B, N] (0 keep ->
    -1; 1 + k: k x 360 / faces clockwise from the side's forward, chain.FACES)."""
    from tools.nn.model import chain
    n = faces or chain.FACES
    fwd = torch.rad2deg(torch.atan2(frame.ux, frame.uz)).reshape(-1, *([1] * (face.dim() - 1)))
    deg = torch.remainder(fwd + (face - 1).to(fwd.dtype) * (360.0 / n), 360.0)
    deg = torch.where(deg >= 360.0, torch.zeros_like(deg), deg)            # (float rounding: 359.99999 -> 360)
    return torch.where(face > 0, deg, torch.full_like(deg, -1.0))


def width_metres(width, wlim, widths=None):
    """[B, N] the ordered width (m) of the width head's choice width [B, N] (0 keep -> -1; 1..W-1 evenly from the
    narrowest wlim[..., 0] to the widest wlim[..., 1], chain.WIDTHS); wlim None: -1 everywhere."""
    from tools.nn.model import chain
    if wlim is None:
        return torch.full(width.shape, -1.0, device=width.device)
    W = widths or chain.WIDTHS
    lo, hi = wlim[..., 0].float(), wlim[..., 1].float()
    frac = ((width.float() - 1) / max(W - 2, 1)).clamp(0, 1)
    return torch.where(width > 0, lo + frac * (hi - lo), torch.full_like(lo, -1.0))


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
    ability = cast = None
    if action.ability is not None:
        ability = torch.where(ctrl & (action.ability >= 0), action.ability, torch.full_like(action.ability, -1))
        # a spell (Orders.ability SPELL_AT + k) falls on the target pointer's choice (heads.py)
        cast = torch.where(hd.is_spell(ability), action.target, torch.full_like(action.target, -1))
    face = width = None
    if getattr(action, "face", None) is not None:
        # (only with a hold, a move or a withdraw: chain.form_able; never under attack or keep)
        formed = (kind == hd.HOLD) | (kind == hd.MOVE) | (kind == hd.WITHDRAW)
        face = torch.where(formed, face_world(action.face, frame).float(), torch.full_like(point[..., 0], -1.0).float())
        width = torch.where(formed, width_metres(action.width, obs_t.get("wlim")).float(),
                            torch.full_like(face, -1.0))
    return Orders(kind=kind, x=point[..., 0].float(), z=point[..., 1].float(), target=target, run=run,
                  ability=ability, cast_target=cast, face=face, width=width)


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
def act(actor, obs, setup, h=None, greedy=False, temperature=1.0, abilities=True, commit_state=None, t=None,
        wlim=None):
    """(orders, new memory, logits, action) for the batch of one side's observation. abilities: the
    network also chooses its abilities (Orders.ability). v2: commit_state, a dict (commit.py) kept between the
    decisions of a battle (an empty one at its start), is updated in place; t: the battle time [B] (s). wlim [B, N, 2]:
    each unit's narrowest and widest frontage, m (cfg.form's width head, tools/nn/general/form.py; None: no width)."""
    device = next(actor.parameters()).device
    obs_t = policy.to_torch(obs, device)
    if wlim is not None and getattr(actor.cfg, "form", False):
        obs_t["wlim"] = torch.as_tensor(np.asarray(wlim, dtype=np.float32), device=device).reshape(
            *obs_t["own"].shape, 2)
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
