"""
Extract hidden states from a model at given token positions.

Two modes:
  - extract_activations(...)      -> full [N, n_positions, n_layers, d_model]
  - extract_mean_activations(...) -> mean [n_positions, n_layers, d_model]
                                     (streaming, memory-friendly)

Neither knows anything about harmful/harmless. Pass *one* group of prompts
and *one* PositionSpec; the caller composes diffs in directions/.
"""

from typing import List, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from ..models.base import ModelWrapper
from ..positions.tokens import PositionSpec
from ..hooks.base import add_hooks
from ..hooks.capture import capture_pre_hook, running_mean_pre_hook


class ActivationExtractor:
    """
    Reusable extractor bound to a ModelWrapper and PositionSpec.
    Use the same instance to extract activations for any number of prompt sets.
    """

    def __init__(
        self,
        model: ModelWrapper,
        positions: PositionSpec,
        batch_size: int = 8,
        system: Optional[str] = None,
    ):
        self.model = model
        self.positions = positions
        self.batch_size = batch_size
        self.system = system

    def extract(self, prompts: List[str], reduce: str = "none") -> Tensor:
        """
        reduce:
          "none" -> [N, n_positions, n_layers, d_model]  (float32, CPU)
          "mean" -> [n_positions, n_layers, d_model]     (float64, on model device)
        """
        if reduce == "mean":
            return extract_mean_activations(
                self.model, prompts, self.positions,
                batch_size=self.batch_size, system=self.system,
            )
        return extract_activations(
            self.model, prompts, self.positions,
            batch_size=self.batch_size, system=self.system,
        )


def extract_activations(
    model: ModelWrapper,
    prompts: List[str],
    positions: PositionSpec,
    *,
    batch_size: int = 8,
    system: Optional[str] = None,
) -> Tensor:
    """Return [N, n_positions, n_layers, d_model] (float32 on CPU)."""
    n_layers = model.n_layers
    cache: List[List[Tensor]] = [[] for _ in range(n_layers)]
    pos = list(positions)

    pre_hooks = [
        (model.block_modules[layer], capture_pre_hook(layer, cache, pos))
        for layer in range(n_layers)
    ]

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="extract"):
            inputs = model.tokenize(prompts[i:i + batch_size], system=system)
            with add_hooks(pre_hooks=pre_hooks):
                model.model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )

    # cache[layer] is a list of [batch, n_pos, d] tensors → concat along batch
    per_layer = [torch.cat(cache[layer], dim=0) for layer in range(n_layers)]
    # stack into [n_layers, N, n_pos, d] then permute → [N, n_pos, n_layers, d]
    stacked = torch.stack(per_layer, dim=0).permute(1, 2, 0, 3).contiguous()
    return stacked


def extract_mean_activations(
    model: ModelWrapper,
    prompts: List[str],
    positions: PositionSpec,
    *,
    batch_size: int = 8,
    system: Optional[str] = None,
) -> Tensor:
    """Return [n_positions, n_layers, d_model] (float64 on model.device)."""
    n_pos = len(positions)
    n_layers = model.n_layers
    n_samples = len(prompts)
    d = model.d_model
    pos = list(positions)

    cache = torch.zeros((n_pos, n_layers, d), dtype=torch.float64, device=model.device)

    pre_hooks = [
        (model.block_modules[layer], running_mean_pre_hook(layer, cache, n_samples, pos))
        for layer in range(n_layers)
    ]

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="mean"):
            inputs = model.tokenize(prompts[i:i + batch_size], system=system)
            with add_hooks(pre_hooks=pre_hooks):
                model.model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )

    return cache
