"""Activation-addition intervention."""

from typing import List, Optional, Union

import torch
from torch import Tensor

from .runner import Intervention
from ..models.base import ModelWrapper
from ..directions.storage import Direction
from ..hooks.act_add import act_add_pre_hook


class ActAdd(Intervention):
    """
    Add `coeff * direction` at the given layers, optionally at given positions.

    Args:
        direction: Direction (per-layer) or raw [n_layers, d] / [d] tensor
        layers:    list of layer indices to intervene at
        coeff:     scalar (positive = inject the direction, negative = subtract)
        positions: token positions (None = every position)
    """

    def __init__(
        self,
        direction: Union[Direction, Tensor],
        layers: List[int],
        coeff: float = 1.0,
        positions: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        """
        Args:
            direction: per-layer [n_layers, d] or single [d] vector
            layers:    layer indices to intervene at
            coeff:     scale factor. With normalize=True this is unit-scale
                       (try 1-8 for a single layer, 0.1-1.0 when intervening
                       across many layers — effects compound).
            positions: token positions (None = every position)
            normalize: divide direction by its L2 norm before adding.
                       Strongly recommended; without it the diff-of-means
                       vector dominates the residual stream and produces
                       garbled output.
        """
        self.direction = direction
        self.layers = layers
        self.coeff = coeff
        self.positions = positions
        self.normalize = normalize

    def _vec_at(self, layer: int) -> Tensor:
        if isinstance(self.direction, Direction):
            return self.direction.at_layer(layer) if self.direction.is_per_layer else self.direction.vec
        if self.direction.ndim == 2:
            return self.direction[layer]
        return self.direction

    def hooks(self, model: ModelWrapper):
        pre_hooks = [
            (
                model.block_modules[layer],
                act_add_pre_hook(
                    self._vec_at(layer), self.coeff, self.positions,
                    normalize=self.normalize,
                ),
            )
            for layer in self.layers
        ]
        return pre_hooks, []
