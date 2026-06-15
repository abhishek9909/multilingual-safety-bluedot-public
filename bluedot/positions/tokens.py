"""
Compute negative-indexed token positions for t_inst / t_post_inst.

Tokenizers are left-padded (see models.base.load_model), so negative indices
robustly point at the last real tokens of every sequence in a batch.
"""

from dataclasses import dataclass
from typing import List, Literal

from ..models.base import ModelWrapper


Kind = Literal["t_inst", "t_post_inst", "eoi", "inst+eoi"]


@dataclass(frozen=True)
class PositionSpec:
    """Negative-indexed positions to extract from a sequence."""
    kind: Kind
    positions: List[int]

    def __iter__(self):
        return iter(self.positions)

    def __len__(self):
        return len(self.positions)


def t_inst_position(model: ModelWrapper) -> PositionSpec:
    """Position of the last instruction token (harmfulness probe)."""
    n_eoi = model.n_eoi_toks
    return PositionSpec(kind="t_inst", positions=[-n_eoi - 1])


def t_post_inst_position(model: ModelWrapper) -> PositionSpec:
    """Position of the last post-instruction token (refusal probe)."""
    return PositionSpec(kind="t_post_inst", positions=[-1])


def eoi_positions(model: ModelWrapper) -> PositionSpec:
    """All post-instruction token positions."""
    n_eoi = model.n_eoi_toks
    return PositionSpec(kind="eoi", positions=list(range(-n_eoi, 0)))


def inst_plus_eoi_positions(model: ModelWrapper, n_context: int = 2) -> PositionSpec:
    """
    n_context preceding instruction tokens + all eoi tokens.

    Matches the "extract_hidden.py" convention where the last `n_context`
    instruction tokens precede the eoi region (so index 0 = first context tok,
    index n_context-1 = t_inst, index -1 = t_post_inst).
    """
    n_eoi = model.n_eoi_toks
    start = -n_eoi - n_context
    return PositionSpec(kind="inst+eoi", positions=list(range(start, 0)))
