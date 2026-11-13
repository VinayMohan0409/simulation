"""Low-rank adapters (LoRA, Hu et al. 2022) on the nn.Linear layers of chosen submodules, as LAC-WM fine-tunes its world model (rank 2); merge() folds them back into plain weights for a checkpoint."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = base(x) + scale * (x A^T) B^T, with the base frozen, A small random and B zero (so it starts as the base)."""

    def __init__(self, base: nn.Linear, rank=2, alpha=2.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    def merged(self):
        """A plain nn.Linear with the adapter folded into its weight."""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None, device=self.base.weight.device)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin


def add_lora(module: nn.Module, rank=2, alpha=2.0):
    """Wrap every nn.Linear under module (in place); returns the adapter parameters."""
    params = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            wrapped = LoRALinear(child, rank, alpha)
            setattr(module, name, wrapped)
            params += [wrapped.A, wrapped.B]
        elif not isinstance(child, LoRALinear):
            params += add_lora(child, rank, alpha)
    return params


def merge(module: nn.Module):
    """Replace every LoRALinear under module by its merged nn.Linear (in place)."""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged())
        else:
            merge(child)
