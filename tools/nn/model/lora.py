"""LoRA adapters: a small low-rank addition per pair "faction + role" on top of the shared weights.

y = W x + (alpha / r) * B_a (A_a x), a = the adapter of this battle's side. B starts at zero, so an
adapter changes nothing until it is trained. rank 0 (default): a plain linear layer.
"""
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, n_in, n_out, rank=0, adapters=0, alpha=8.0, bias=True):
        super().__init__()
        self.base = nn.Linear(n_in, n_out, bias=bias)
        self.rank = rank if adapters > 0 else 0
        self.enabled = self.rank > 0
        if self.rank:
            self.A = nn.Parameter(torch.randn(adapters, n_in, rank) / n_in ** 0.5)
            self.B = nn.Parameter(torch.zeros(adapters, rank, n_out))
            self.scale = alpha / rank

    def forward(self, x, adapter=None):
        y = self.base(x)
        if not (self.enabled and adapter is not None):
            return y
        flat = x.reshape(x.shape[0], -1, x.shape[-1])                   # [B, T, in]
        delta = torch.bmm(torch.bmm(flat, self.A[adapter]), self.B[adapter])
        return y + self.scale * delta.reshape(y.shape)


def adapter_parameters(module):
    """The adapters' weights (train these only, with the base frozen)."""
    return [p for m in module.modules() if isinstance(m, LoRALinear) and m.rank for p in (m.A, m.B)]


def freeze_base(module):
    """Freeze everything except the adapters."""
    keep = {id(p) for p in adapter_parameters(module)}
    for p in module.parameters():
        p.requires_grad_(id(p) in keep)
