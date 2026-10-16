"""Neighbours by cells, every size fixed (torch.compile-friendly: no data-dependent shapes, no loop over men).

The men of all battles are sorted by a hashed cell key (battle b, cell (floor(x / c), floor(z / c)) -> b x H + hash);
a query looks up each of the 3 x 3 cells round its point by a binary search for the cell's key and takes the first
`cap` men stored there. A man thus sees every man within c of him (more, up to 2c, in the corners); a cell holding
more than cap men or two cells sharing a hash slot can hide a few (counted by `overflow`)."""
import torch

_P1, _P2 = 73856093, 19349663


def _hash(ix, iz, H):
    return torch.remainder(ix * _P1 + iz * _P2, H)


class Table:
    """The sorted cell table of one set of points [B, M] (x, z, valid)."""

    def __init__(self, x, z, valid, cell_m, H):
        B, M = x.shape
        self.B, self.M, self.c, self.H = B, M, float(cell_m), int(H)
        ix = torch.floor(x / self.c).long()
        iz = torch.floor(z / self.c).long()
        b = torch.arange(B, device=x.device)[:, None]
        key = b * self.H + _hash(ix, iz, self.H)
        key = torch.where(valid, key, torch.full_like(key, B * self.H))      # the invalid at the very end
        flat = key.reshape(-1)
        self.sorted, self.perm = torch.sort(flat)
        self.dev = x.device

    def query(self, qx, qz, cap):
        """[B, Q, 9 * cap] indices into M of the men stored in the 3 x 3 cells round each point (qx, qz [B, Q]), -1 where
        empty."""
        B, Q = qx.shape
        ix = torch.floor(qx / self.c).long()
        iz = torch.floor(qz / self.c).long()
        b = torch.arange(B, device=qx.device)[:, None, None]
        d = torch.arange(-1, 2, device=qx.device)
        dx = d.repeat_interleave(3)
        dz = d.repeat(3)
        k = b * self.H + _hash(ix[..., None] + dx, iz[..., None] + dz, self.H)          # [B, Q, 9]
        start = torch.searchsorted(self.sorted, k.reshape(-1)).reshape(B, Q, 9)
        slot = start[..., None] + torch.arange(cap, device=qx.device)                   # [B, Q, 9, cap]
        slot_c = slot.clamp(max=self.sorted.numel() - 1)
        ok = (slot < self.sorted.numel()) & (self.sorted[slot_c] == k[..., None])
        flat = self.perm[slot_c]                                                       # flat index b * M + m
        m = flat - b[..., None] * self.M
        m = torch.where(ok, m, torch.full_like(m, -1))
        return m.reshape(B, Q, 9 * cap)

    def overflow(self, cap):
        """Cells holding more than cap men (their extra men may be missed by queries)."""
        counts = torch.unique_consecutive(self.sorted, return_counts=True)[1]
        return int((counts > cap).sum())


def gather2(t, idx):
    """t [B, M] at idx [B, Q, K] (-1 -> index 0; mask it yourself)."""
    B, Q, K = idx.shape
    return t.gather(1, idx.clamp(min=0).reshape(B, Q * K)).reshape(B, Q, K)
