"""Neighbours by cells, every size fixed (torch.compile-friendly: no data-dependent shapes, no loop over men, no sync).

The men of every battle are sorted by a hashed cell key (battle b, cell (floor(x / c), floor(z / c)) -> b x H +
hash(cell), H a power of 2); per slot the table keeps where its men start in the sorted order and how many there are.
A query looks up each of the 3 x 3 cells round its point and takes the first `cap` men of each slot. A man thus sees
every man within c of him (more, up to 2c, in the corners); a cell holding more than cap men or two cells sharing a
hash slot can hide a few (counted by `overflow`). The GPU kernels (scans.py, kernels.py) read the same table and the
men's numbers in the sorted order (sorted_of): a slot's men lie side by side."""
import torch

_P1, _P2 = 73856093, 19349663


def _hash(ix, iz, H):
    return torch.remainder(ix * _P1 + iz * _P2, H)


class Table:
    """The cell table of one set of points [B, M] (x, z, valid): perm the men in the sorted order (flat b x M + m), man
    their places m, start / count [B x H + 1] each slot's first man in that order and its men (the last slot empty)."""

    def __init__(self, x, z, valid, cell_m, H, cap=None):
        B, M = x.shape
        self.B, self.M, self.c, self.H = B, M, float(cell_m), int(H)
        assert self.H > 0 and self.H & (self.H - 1) == 0, "cells: H a power of 2 (the GPU kernels' remainder)"
        ix = torch.floor(x / self.c).long()
        iz = torch.floor(z / self.c).long()
        b = torch.arange(B, device=x.device)[:, None]
        n_slots = B * self.H
        key = b * self.H + _hash(ix, iz, self.H)
        key = torch.where(valid, key, torch.full_like(key, n_slots))       # the invalid at the very end
        self.sorted, self.perm = torch.sort(key.reshape(-1))
        n = self.sorted.numel()
        idx = torch.arange(n, device=x.device)
        first = torch.searchsorted(self.sorted, self.sorted)                 # the slot's first man in the sorted order
        at = torch.where((idx == first) & (self.sorted < n_slots), self.sorted, torch.full_like(idx, n_slots))
        self.start = torch.zeros(n_slots + 1, dtype=torch.int32, device=x.device).scatter(0, at, idx.to(torch.int32))
        inside = torch.where(self.sorted < n_slots, self.sorted, torch.full_like(idx, n_slots))
        cnt = torch.zeros(n_slots + 1, dtype=torch.int32, device=x.device).scatter_add(
            0, inside, torch.ones_like(idx, dtype=torch.int32))
        self.count = torch.where(torch.arange(n_slots + 1, device=x.device) < n_slots, cnt, torch.zeros_like(cnt))
        self.man = (self.perm - torch.div(self.perm, M, rounding_mode="floor") * M).to(torch.int32)
        self.dev = x.device

    def sorted_of(self, t):
        """t [B, M] in the sorted order [B x M] (a slot's men side by side: the GPU kernels read them so)."""
        return t.reshape(-1)[self.perm].contiguous()

    def slots(self, qx, qz):
        """[B, Q, 9] the hashed slots (b x H + slot) of the 3 x 3 cells round each point."""
        B = qx.shape[0]
        ix = torch.floor(qx / self.c).long()
        iz = torch.floor(qz / self.c).long()
        b = torch.arange(B, device=qx.device)[:, None, None]
        d = torch.arange(-1, 2, device=qx.device)
        dx = d.repeat_interleave(3)
        dz = d.repeat(3)
        return b * self.H + _hash(ix[..., None] + dx, iz[..., None] + dz, self.H)

    def query(self, qx, qz, cap):
        """[B, Q, 9 * cap] indices into M of the men stored in the 3 x 3 cells round each point (qx, qz [B, Q]), -1
        where empty."""
        B, Q = qx.shape
        s = self.slots(qx, qz)
        k = torch.arange(cap, device=qx.device)
        at = self.start[s].long()[..., None] + k
        ok = k < self.count[s][..., None]
        m = self.man[at.clamp(max=self.man.numel() - 1)].long()
        return torch.where(ok, m, torch.full_like(m, -1)).reshape(B, Q, 9 * cap)

    def overflow(self, cap):
        """Cells holding more than cap men (their extra men may be missed by queries)."""
        return int((self.count > cap).sum())


def gather2(t, idx):
    """t [B, M] at idx [B, Q, K] (-1 -> index 0; mask it yourself)."""
    B, Q, K = idx.shape
    return t.gather(1, idx.clamp(min=0).reshape(B, Q * K)).reshape(B, Q, K)
