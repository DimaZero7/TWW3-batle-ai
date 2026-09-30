"""Memory over time: a GRU per token (per unit, and one for the army's context token).

Why a GRU and not attention over the last frames: the cost of a decision stays the same however
long the memory, the state is one vector per unit (easy to carry between decisions, and
the same on every computer in co-op), and the 10-20 s horizon (20-80 decisions at 2-4 per second)
is learned instead of fixed by a buffer. Tokens that are not looked at (padding, dead) keep zero.
"""
import torch
from torch import nn


class TokenMemory(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cell = nn.GRUCell(d, d)
        self.norm = nn.LayerNorm(d)

    def initial(self, batch, length, device=None):
        return torch.zeros(batch, length, self.cell.hidden_size, device=device)

    def forward(self, x, h, keep):
        """x, h [B, L, d]; keep [B, L] -> (x + memory, new h)."""
        B, L, d = x.shape
        new = self.cell(self.norm(x).reshape(B * L, d), h.reshape(B * L, d)).reshape(B, L, d)
        new = new * keep[..., None]
        return x + new, new
