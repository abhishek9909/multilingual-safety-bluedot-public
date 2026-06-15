"""Direction-ablation intervention (Arditi et al. style)."""

from typing import List, Optional, Union

import torch
from torch import Tensor

from .runner import Intervention
from ..models.base import ModelWrapper
from ..directions.storage import Direction
from ..hooks.ablation import ablation_pre_hook, ablation_fwd_hook


class Ablation(Intervention):
    """
    Project the residual stream off the given direction at every layer
    (and optionally inside attn/mlp outputs too).

    Use a SINGLE direction (a [d] vector); if a per-layer direction was
    used to *select* the best layer, pass `direction.at_layer(L)` here.
    """

    def __init__(
        self,
        direction: Union[Direction, Tensor],
        *,
        coeff: float = 1.0,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        ablate_attn: bool = True,
        ablate_mlp: bool = True,
    ):
        if isinstance(direction, Direction):
            assert not direction.is_per_layer, (
                "Ablation requires a single direction vector. "
                "Use Direction.at_layer(L) to slice first."
            )
            self.vec = direction.vec
        else:
            assert direction.ndim == 1, "Ablation expects a [d_model] vector."
            self.vec = direction
        self.coeff = coeff
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.ablate_attn = ablate_attn
        self.ablate_mlp = ablate_mlp

    def hooks(self, model: ModelWrapper):
        end = self.end_layer or model.n_layers
        layers = range(self.start_layer, end)

        pre_hooks = [
            (model.block_modules[L], ablation_pre_hook(self.vec, self.coeff))
            for L in layers
        ]
        fwd_hooks = []
        if self.ablate_attn:
            fwd_hooks += [
                (model.attn_modules[L], ablation_fwd_hook(self.vec, self.coeff))
                for L in layers
            ]
        if self.ablate_mlp:
            fwd_hooks += [
                (model.mlp_modules[L], ablation_fwd_hook(self.vec, self.coeff))
                for L in layers
            ]
        return pre_hooks, fwd_hooks
