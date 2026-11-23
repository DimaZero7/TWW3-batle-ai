"""The operative's group frame (tmp/engine/design_merged.md 5): per selected row (inputs.select's slots) six numbers in
the frame of the own group's centre C facing the enemy group E's centre - nothing of the teacher's thresholds in it.

    gframe(raw, slot, mem, E) -> [B, K, 6]   right, forward (/ lines.PLACE_M), v_right, v_forward, the closing speed on
                     C (+ = towards C; / lines.SPEED_MS) and the centre distance to the nearest own member (/ PLACE_M);
                     zeros on empty rows and in a battle without a member or without E
    COLS             the column names
The frame turns with the group: a battle rotated about C (positions, velocities, headings) gives the same numbers.
"""
import torch

from net.tactic import raw as TR

COLS = ("g_right", "g_fwd", "g_v_right", "g_v_fwd", "g_close", "g_near")
PLACE_M = 1024.0      # (= lines.PLACE_M: operative.json place_m)
SPEED_MS = 3.0        # (= lines.SPEED_MS: the infantry's run, sim/config/units.json speed.run)


def gframe(raw, slot, mem, E):
    """(module doc) raw [B, N, RAW], slot [B, K] long (-1 empty), mem / E [B, N] bool."""
    I = TR.I
    x, z, vx, vz = raw[..., I["x"]], raw[..., I["z"]], raw[..., I["vx"]], raw[..., I["vz"]]
    wm, we = mem.float(), E.float()
    nm, ne = wm.sum(1), we.sum(1)
    cx, cz = (x * wm).sum(1) / nm.clamp(min=1), (z * wm).sum(1) / nm.clamp(min=1)
    ex, ez = (x * we).sum(1) / ne.clamp(min=1), (z * we).sum(1) / ne.clamp(min=1)
    dx, dz = ex - cx, ez - cz
    dn = torch.sqrt(dx * dx + dz * dz)
    ok = (nm > 0) & (ne > 0) & (dn > 1e-6)
    ux, uz = dx / dn.clamp(min=1e-6), dz / dn.clamp(min=1e-6)          # forward; right = (uz, -ux)
    px, pz = x - cx[:, None], z - cz[:, None]
    ux_, uz_ = ux[:, None], uz[:, None]
    right, fwd = px * uz_ - pz * ux_, px * ux_ + pz * uz_
    v_r, v_f = vx * uz_ - vz * ux_, vx * ux_ + vz * uz_
    pn = torch.sqrt(px * px + pz * pz)
    close = -(vx * px + vz * pz) / pn.clamp(min=1e-6)
    d = torch.sqrt((x[:, :, None] - x[:, None, :]) ** 2 + (z[:, :, None] - z[:, None, :]) ** 2)       # [B, N, N]
    near = torch.where(mem[:, None, :], d, torch.full_like(d, float("inf"))).amin(2)
    near = torch.where(torch.isfinite(near), near, torch.zeros_like(near))
    f = torch.stack([right / PLACE_M, fwd / PLACE_M, v_r / SPEED_MS, v_f / SPEED_MS, close / SPEED_MS,
                     near / PLACE_M], -1)                                                    # [B, N, 6]
    f = torch.where(ok[:, None, None], f, torch.zeros_like(f))
    B, K = slot.shape
    s0 = slot.clamp(min=0)
    out = f.gather(1, s0[..., None].expand(B, K, f.shape[-1]))
    return torch.where((slot >= 0)[..., None], out, torch.zeros_like(out))
