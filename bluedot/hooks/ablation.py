"""Direction ablation hooks: project the residual stream off a direction."""

import torch
from torch import Tensor

from .base import _unpack, _repack

_EPS = 1e-8


def _ablate(act: Tensor, direction: Tensor, coeff: float) -> Tensor:
    direction = direction / (direction.norm(dim=-1, keepdim=True) + _EPS)
    direction = direction.to(act)
    return act - coeff * (act @ direction).unsqueeze(-1) * direction


def ablation_pre_hook(direction: Tensor, coeff: float = 1.0):
    def hook_fn(module, input):
        act, rest = _unpack(input)
        act = _ablate(act, direction, coeff)
        return _repack(act, rest)
    return hook_fn


def ablation_fwd_hook(direction: Tensor, coeff: float = 1.0):
    def hook_fn(module, input, output):
        act, rest = _unpack(output)
        act = _ablate(act, direction, coeff)
        return _repack(act, rest)
    return hook_fn
