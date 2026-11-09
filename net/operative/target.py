"""The group's target from its seed (plan.md "Цель группы"; a departure from the decision of 29.10 for v1): the network
picks one seen enemy, the code grows the target by the tactician's squares' rule (tactic/parcel.py target_boxes) - the
enemies touching it, nearest the seed first, while the frame fits the group's near window. So the target is always
connected, and two groups may take one target.

    grow(raw, bi, seed, cand, gx, gz, near)      [R, N] bool: the target, in the axes facing the seed from the group
    base_heading(raw, bi, target, gx, gz, up)    [R]: the bearing from the group's centre to the target's centre (none:
                                                 up) - the turn's sector 0
    turn_ok(raw, bi, target, base, near)         [R, S] bool: the sectors whose zone's axes keep the target connected
                                                 and its frame inside the near window (parcel.check's rules)
"""
import torch

from net.operative import cfg
from net.tactic import parcel as PC
from net.tactic import raw as TR


def _parcel(bi, target, cx, cz, heading):
    R, N = target.shape
    h = heading * (torch.pi / 180)
    z = torch.zeros(R, dtype=torch.bool, device=target.device)
    return PC.Parcel(bi, torch.zeros_like(target), target, cx, cz, torch.sin(h), torch.cos(h), z,
                     torch.zeros(R, dtype=torch.long, device=target.device))


def bearing(x0, z0, x1, z1):
    return torch.remainder(torch.rad2deg(torch.atan2(x1 - x0, z1 - z0)), 360.0)


def grow(raw, bi, seed, cand, gx, gz, near):
    """seed [R] long (-1: none -> no target), cand [R, N] bool the enemies it may take (alive, seen)."""
    R, N = cand.shape
    dev = cand.device
    I = TR.I
    S = raw[bi]
    has = seed >= 0
    s0 = seed.clamp(min=0)
    k = torch.arange(R, device=dev)
    sx, sz = S[k, s0, I["x"]], S[k, s0, I["z"]]
    head = bearing(gx, gz, sx, sz)
    t = torch.zeros((R, N), dtype=torch.bool, device=dev)
    t[k[has], s0[has]] = True
    cand = cand | t
    p = _parcel(bi, t, gx, gz, head)
    s, f, hs, hf, _ = PC.target_boxes(raw, p, sel=cand)
    touch = ((s[:, :, None] - s[:, None, :]).abs() <= hs[:, :, None] + hs[:, None, :]) & \
            ((f[:, :, None] - f[:, None, :]).abs() <= hf[:, :, None] + hf[:, None, :])
    d = (s - s[k, s0][:, None]) ** 2 + (f - f[k, s0][:, None]) ** 2
    big = 1e9
    for _ in range(N):
        lo_s = torch.where(t, s - hs, torch.full_like(s, big)).amin(1, keepdim=True)
        hi_s = torch.where(t, s + hs, torch.full_like(s, -big)).amax(1, keepdim=True)
        lo_f = torch.where(t, f - hf, torch.full_like(f, big)).amin(1, keepdim=True)
        hi_f = torch.where(t, f + hf, torch.full_like(f, -big)).amax(1, keepdim=True)
        w = torch.maximum(hi_s, s + hs) - torch.minimum(lo_s, s - hs)
        dd = torch.maximum(hi_f, f + hf) - torch.minimum(lo_f, f - hf)
        fits = (w <= near[:, None] + 1e-3) & (dd <= near[:, None] + 1e-3)
        can = cand & ~t & (touch & t[:, None, :]).any(2) & fits & has[:, None]
        if not bool(can.any()):
            break
        j = torch.where(can, d, torch.full_like(d, big)).argmin(1)
        add = can.any(1)
        t[k[add], j[add]] = True
    return t


def centre(raw, bi, target):
    """([R], [R], [R] bool) the target's actual centre (its alive, seen units)."""
    I = TR.I
    S = raw[bi]
    w = (target & (S[..., I["alive"]] > 0) & (S[..., I["vis"]] > 0)).float()
    k = w.sum(1)
    return (S[..., I["x"]] * w).sum(1) / k.clamp(min=1), (S[..., I["z"]] * w).sum(1) / k.clamp(min=1), k > 0


def base_heading(raw, bi, target, gx, gz, up):
    tx, tz, has = centre(raw, bi, target)
    ok = has & (((tx - gx) ** 2 + (tz - gz) ** 2) > 1e-6)
    return torch.where(ok, bearing(gx, gz, tx, tz), up)


def headings(base, S=None):
    """[R, S] the sectors' bearings."""
    S = int(S or cfg()["sectors"])
    return torch.remainder(base[:, None] + torch.arange(S, device=base.device)[None] * (360.0 / S), 360.0)


def turn_ok(raw, bi, target, base, near, gx, gz):
    R = target.shape[0]
    S = int(cfg()["sectors"])
    hd = headings(base, S)
    out = torch.ones((R, S), dtype=torch.bool, device=target.device)
    if not bool(target.any()):
        return out
    rep = lambda v: v.repeat_interleave(S, 0)                                       # noqa: E731
    p = _parcel(rep(bi), rep(target), rep(gx), rep(gz), hd.reshape(-1))
    s, f, hs, hf, on = PC.target_boxes(raw, p)
    ok = PC.connected(s, f, hs, hf, on)
    w, d = PC.frame_size(s, f, hs, hf, on)
    nn_ = rep(near)
    ok = ok & (w <= nn_ + 1e-3) & (d <= nn_ + 1e-3)
    return ok.view(R, S)
