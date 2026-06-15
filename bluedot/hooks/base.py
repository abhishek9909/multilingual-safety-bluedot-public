"""Forward-hook primitives. Pure utilities — no model-family logic here."""

import contextlib
import functools
from typing import Callable, List, Tuple

import torch


@contextlib.contextmanager
def add_hooks(
    pre_hooks: List[Tuple[torch.nn.Module, Callable]] = (),
    fwd_hooks: List[Tuple[torch.nn.Module, Callable]] = (),
    **kwargs,
):
    """Register hooks for the duration of the `with` block, then remove."""
    handles = []
    try:
        for module, hook in pre_hooks:
            handles.append(module.register_forward_pre_hook(functools.partial(hook, **kwargs)))
        for module, hook in fwd_hooks:
            handles.append(module.register_forward_hook(functools.partial(hook, **kwargs)))
        yield
    finally:
        for h in handles:
            h.remove()


def _unpack(input_or_output):
    if isinstance(input_or_output, tuple):
        return input_or_output[0], input_or_output[1:]
    return input_or_output, None


def _repack(activation, rest):
    if rest is None:
        return activation
    return (activation, *rest)
