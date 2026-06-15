"""Cosine probe: dot-product / cosine similarity with a saved direction."""

import torch
from torch import Tensor

from .base import Probe
from ..directions.storage import Direction
from ..directions.ops import normalize


class CosineProbe(Probe):
    """
    score(h) = h_unit . direction_unit   (or raw dot product if normalize=False)

    Takes a Direction or raw tensor + a layer index to select the slice.
    """

    def __init__(
        self,
        direction: "Direction | Tensor",
        layer: int = None,
        *,
        normalize_inputs: bool = True,
        normalize_direction: bool = True,
    ):
        if isinstance(direction, Direction):
            vec = direction.at_layer(layer) if (direction.is_per_layer and layer is not None) else direction.vec
        else:
            vec = direction[layer] if (direction.ndim == 2 and layer is not None) else direction
        if normalize_direction:
            vec = normalize(vec)
        self.vec = vec
        self.normalize_inputs = normalize_inputs

    def score(self, hidden: Tensor) -> Tensor:
        v = self.vec.to(hidden)
        h = normalize(hidden) if self.normalize_inputs else hidden
        return (h * v).sum(dim=-1)
