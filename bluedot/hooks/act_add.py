"""Activation-addition hooks: add `coeff * direction` to the residual stream."""

from typing import List, Optional

import torch
from torch import Tensor

from .base import _unpack, _repack

_EPS = 1e-8


def act_add_pre_hook(
    direction: Tensor,
    coeff: float,
    positions: Optional[List[int]] = None,
    normalize: bool = True,
):
    """
    Adds `coeff * direction_unit` to the input activation of a block.

    Args:
        direction: [d_model] tensor to inject
        coeff:     scaling factor (unit-scale when normalize=True; raw scale otherwise)
        positions: token positions to intervene at (None = every position)
        normalize: divide `direction` by its L2 norm before adding. Strongly
                   recommended — diff-of-means vectors can have norms in the
                   hundreds, which without normalization will blow the
                   residual stream out of distribution and produce garbled
                   output (UTF-8 replacement chars).
    """
    if normalize:
        direction = direction / (direction.norm(dim=-1, keepdim=True) + _EPS)

    def hook_fn(module, input):
        act, rest = _unpack(input)
        v = direction.to(act)
        delta = coeff * v
        if positions is None:
            act = act + delta
        else:
            for p in positions:
                act[:, p, :] = act[:, p, :] + delta
        return _repack(act, rest)
    return hook_fn
