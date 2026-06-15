"""
Portable direction artifact: tensor + metadata.

Every saved direction carries enough info to be loaded later and used against
a *different* dataset or model with the same architecture, without re-running
the extraction.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

import torch
from torch import Tensor


@dataclass
class Direction:
    """
    A direction vector (or per-layer stack).

    vec shape conventions:
      - per-layer: [n_layers, d_model]
      - single   : [d_model]
    """
    vec: Tensor
    kind: str                   # 't_inst' or 't_post_inst' (or custom)
    model_name: str
    template_name: str
    layer: Optional[int] = None # None = stored per-layer
    source_pos_dataset: str = ""
    source_neg_dataset: str = ""
    lang: str = "en"
    n_pos: int = 0
    n_neg: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # --- convenience -------------------------------------------------------
    @property
    def is_per_layer(self) -> bool:
        return self.vec.ndim == 2

    def at_layer(self, layer: int) -> Tensor:
        if not self.is_per_layer:
            raise ValueError("Direction is not stored per-layer.")
        return self.vec[layer]

    def normalized(self, eps: float = 1e-8) -> "Direction":
        v = self.vec / (self.vec.norm(dim=-1, keepdim=True) + eps)
        return Direction(**{**asdict(self), "vec": v})


def save_direction(direction: Direction, path: str) -> None:
    """Save to `path.pt` (tensor) and `path.json` (metadata)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".pt"):
        path = path[:-3]
    torch.save(direction.vec.cpu(), path + ".pt")
    meta = {k: v for k, v in asdict(direction).items() if k != "vec"}
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_direction(path: str, map_location: str = "cpu") -> Direction:
    if path.endswith(".pt"):
        path = path[:-3]
    vec = torch.load(path + ".pt", map_location=map_location)
    with open(path + ".json") as f:
        meta = json.load(f)
    return Direction(vec=vec, **meta)
