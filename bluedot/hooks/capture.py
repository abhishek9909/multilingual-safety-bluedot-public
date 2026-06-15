"""Hooks for capturing hidden states at specified token positions."""

from typing import List

import torch
from torch import Tensor


def capture_pre_hook(layer: int, cache: List[List[Tensor]], positions: List[int]):
    """
    Pre-hook on a residual-stream block. Appends [batch, len(positions), d_model]
    to cache[layer].
    """
    def hook_fn(module, input):
        act = input[0] if isinstance(input, tuple) else input
        # left-padded inputs: position -k indexes the k-th-from-last real token.
        cache[layer].append(act[:, positions, :].detach().to("cpu", torch.float32))
    return hook_fn


def running_mean_pre_hook(
    layer: int,
    cache: Tensor,                 # shape [n_positions, n_layers, d_model], float64
    n_samples: int,
    positions: List[int],
):
    """
    Accumulates a running mean into `cache[:, layer]`. Avoids holding all
    activations in memory. Caller must call this once per sample with
    n_samples=total_samples.
    """
    def hook_fn(module, input):
        act = input[0] if isinstance(input, tuple) else input
        contrib = act[:, positions, :].to(cache)
        cache[:, layer] += (1.0 / n_samples) * contrib.sum(dim=0)
    return hook_fn
