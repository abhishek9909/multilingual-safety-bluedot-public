"""
Diff-of-means direction extraction.

Given two prompt sets (positive / negative — e.g. harmful / harmless) and a
PositionSpec, compute the per-layer mean-activation difference at each
position. The result is a per-layer direction tensor.

Higher-level helpers:
  - extract_direction(...) -> single Direction
  - extract_t_inst_and_t_post_inst(...) -> dict with both directions in one pass
"""

from typing import List, Optional, Dict

import torch
from torch import Tensor

from ..models.base import ModelWrapper
from ..positions.tokens import (
    PositionSpec,
    t_inst_position,
    t_post_inst_position,
    inst_plus_eoi_positions,
)
from ..activations.extractor import extract_mean_activations
from .storage import Direction


class DirectionExtractor:
    """
    Bind a ModelWrapper + PositionSpec; call .extract(pos, neg) for any prompt sets.

    The same extractor can produce directions across many dataset pairs
    (English harmful/harmless, Japanese harmful/harmless, etc.) without
    reconfiguring the model.
    """

    def __init__(
        self,
        model: ModelWrapper,
        positions: PositionSpec,
        *,
        batch_size: int = 8,
        system: Optional[str] = None,
    ):
        self.model = model
        self.positions = positions
        self.batch_size = batch_size
        self.system = system

    def extract(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        *,
        kind: Optional[str] = None,
        lang: str = "en",
        source_pos_dataset: str = "",
        source_neg_dataset: str = "",
    ) -> Direction:
        mean_pos = extract_mean_activations(
            self.model, positive_prompts, self.positions,
            batch_size=self.batch_size, system=self.system,
        )
        mean_neg = extract_mean_activations(
            self.model, negative_prompts, self.positions,
            batch_size=self.batch_size, system=self.system,
        )
        # [n_positions, n_layers, d_model]
        diff = (mean_pos - mean_neg).to(torch.float32).cpu()
        assert not diff.isnan().any()

        # If positions spec has exactly one position, squeeze it; the result is
        # a per-layer direction [n_layers, d_model].
        if diff.shape[0] == 1:
            vec = diff[0]
        else:
            vec = diff   # [n_positions, n_layers, d_model]

        return Direction(
            vec=vec,
            kind=kind or self.positions.kind,
            model_name=self.model.name,
            template_name=self.model.template.name,
            source_pos_dataset=source_pos_dataset,
            source_neg_dataset=source_neg_dataset,
            lang=lang,
            n_pos=len(positive_prompts),
            n_neg=len(negative_prompts),
        )


# --- convenience one-shots -------------------------------------------------

def extract_direction(
    model: ModelWrapper,
    positive_prompts: List[str],
    negative_prompts: List[str],
    *,
    kind: str = "t_post_inst",
    batch_size: int = 8,
    system: Optional[str] = None,
    lang: str = "en",
    source_pos_dataset: str = "",
    source_neg_dataset: str = "",
) -> Direction:
    """Extract a single direction at either t_inst or t_post_inst."""
    if kind == "t_inst":
        spec = t_inst_position(model)
    elif kind == "t_post_inst":
        spec = t_post_inst_position(model)
    else:
        raise ValueError(f"kind must be 't_inst' or 't_post_inst', got {kind!r}")
    return DirectionExtractor(
        model, spec, batch_size=batch_size, system=system
    ).extract(
        positive_prompts, negative_prompts,
        kind=kind, lang=lang,
        source_pos_dataset=source_pos_dataset,
        source_neg_dataset=source_neg_dataset,
    )


def extract_t_inst_and_t_post_inst(
    model: ModelWrapper,
    positive_prompts: List[str],
    negative_prompts: List[str],
    *,
    batch_size: int = 8,
    system: Optional[str] = None,
    n_context: int = 2,
    lang: str = "en",
    source_pos_dataset: str = "",
    source_neg_dataset: str = "",
) -> Dict[str, Direction]:
    """
    Extract BOTH t_inst (harmfulness) and t_post_inst (refusal) directions in
    a SINGLE forward-pass sweep over the data. Returns
        {'t_inst': Direction, 't_post_inst': Direction}.
    """
    spec = inst_plus_eoi_positions(model, n_context=n_context)
    extractor = DirectionExtractor(model, spec, batch_size=batch_size, system=system)
    full = extractor.extract(
        positive_prompts, negative_prompts,
        kind="inst+eoi", lang=lang,
        source_pos_dataset=source_pos_dataset,
        source_neg_dataset=source_neg_dataset,
    )
    # full.vec shape: [n_positions, n_layers, d_model], with
    #   index n_context - 1 == t_inst (last context tok)
    #   index -1           == t_post_inst (last eoi tok)
    t_inst_vec = full.vec[n_context - 1]      # [n_layers, d_model]
    t_post_vec = full.vec[-1]                  # [n_layers, d_model]

    base = dict(
        model_name=model.name,
        template_name=model.template.name,
        lang=lang,
        source_pos_dataset=source_pos_dataset,
        source_neg_dataset=source_neg_dataset,
        n_pos=len(positive_prompts),
        n_neg=len(negative_prompts),
    )
    return {
        "t_inst": Direction(vec=t_inst_vec, kind="t_inst", **base),
        "t_post_inst": Direction(vec=t_post_vec, kind="t_post_inst", **base),
    }
