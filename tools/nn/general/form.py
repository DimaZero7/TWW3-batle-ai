"""The executor's formation heads (tools/nn/model/chain.py cfg.form: the facing and the width; docs/ru/training/general.md
"Тонкий исполнитель"): turning them on in a loaded network and each unit's width limits they choose between.

    actor = form.with_form(actor)                 # run.py --exec-form 1: the heads added, fresh, on "keep"
    obs_t["wlim"] = form.limits(st.u, params)     # [B, N, 2] the narrowest and the widest frontage, m

Outside tools/nn/model on purpose (a file there changes the simulator's version): the limits read the simulator's
reform rule (tools/nn/sim/reform.py), the same numbers for the simulator's battles and the game's companion.
"""
import dataclasses

import torch

from tools.nn.unit.decode import WIDTH_MAX_M


def with_form(actor, on=True):
    """The actor with the formation's heads (cfg.form) - every weight it had, the two heads fresh (zero weights, keep's
    bias: chain.keep_linear), on its device and in its mode. on False, or the heads already there: the actor as it is.
    Only a v2 network (the chained heads)."""
    if not on or getattr(actor.cfg, "form", False):
        return actor
    if not actor.cfg.sectors:
        raise ValueError("--exec-form: only a v2 network (the chained heads)")
    dev = next(actor.parameters()).device
    cfg = dataclasses.replace(actor.cfg, form=True)
    gcfg = getattr(actor, "gcfg", None)
    new = type(actor)(cfg, gcfg) if gcfg is not None else type(actor)(cfg)
    missing, unexpected = new.load_state_dict(actor.state_dict(), strict=True)
    left = [k for k in missing if not k.startswith(("heads.face.", "heads.width."))]
    if left or unexpected:
        raise ValueError(f"--exec-form: the network does not fit: missing {left[:5]}, unexpected {list(unexpected)[:5]}")
    return new.to(dev).train(actor.training)


def limits_of(men0, wmin_k, sp_h, sp_v, params):
    """[..., 2] each unit's (narrowest, widest) frontage, m, for the width head (chain.py): the simulator's limits of its
    men at the start (reform.limits: >= 2 files and >= wmin_k of the men; one rank of all), the widest within
    WIDTH_MAX_M (80 m: the game's 80 m orders gave 77-79 m; the unit network's, unit/decode.py) - the same numbers the
    whole battle, so a choice kept is the same order. A unit whose width does not change (a lone man, artillery:
    wmin_k 0; unknown: NaN) gets (0, 0): only keep."""
    from tools.nn.sim import reform
    rf = params.sim["formation"].get("reform") or {}
    spacing = params.sim["formation"]["spacing_m"]
    t = lambda v: torch.nan_to_num(torch.as_tensor(v, dtype=torch.float32), nan=0.0)
    men0, wmin_k, sp_h, sp_v = t(men0), t(wmin_k), t(sp_h), t(sp_v)
    u = {"men": men0, "wmin_k": wmin_k, "sp_h": sp_h, "sp_v": sp_v}
    lo, hi = reform.limits(u, rf, spacing, men=men0)
    hi = torch.maximum(torch.minimum(hi, torch.full_like(hi, WIDTH_MAX_M)), lo)
    able = (men0 > 1) & (wmin_k > 0) & bool(rf.get("on"))
    zero = torch.zeros_like(lo)
    return torch.stack([torch.where(able, lo, zero), torch.where(able, hi, zero)], -1)


def limits(u, params):
    """limits_of the simulator's state fields u (men0, wmin_k, sp_h, sp_v [B, N]) -> [B, N, 2], on u's device."""
    return limits_of(u["men0"], u["wmin_k"], u["sp_h"], u["sp_v"], params).to(u["men0"].device)


def limits_static(keys, params):
    """limits_of each unit's passport (params.static: the companion, whose units are keys) -> [N, 2]; a key the
    simulator does not know: (0, 0)."""
    rows = [params.static(k) if k in params.units else None for k in keys]
    get = lambda f: [float(r[f]) if r is not None else float("nan") for r in rows]
    return limits_of(get("men0"), get("wmin_k"), get("sp_h"), get("sp_v"), params)
