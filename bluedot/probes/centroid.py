"""Centroid probe: distance-to-mean classifier."""

import torch
from torch import Tensor

from .base import Probe


class CentroidProbe(Probe):
    """
    Fit two means (positive class, negative class). Score by
        score(h) = -||h - mu_pos||^2 + ||h - mu_neg||^2
    Higher = more like positive class.
    """

    def __init__(self):
        self.mu_pos: Tensor = None
        self.mu_neg: Tensor = None

    def fit(self, hidden_pos: Tensor, hidden_neg: Tensor) -> "CentroidProbe":
        self.mu_pos = hidden_pos.mean(dim=0)
        self.mu_neg = hidden_neg.mean(dim=0)
        return self

    def score(self, hidden: Tensor) -> Tensor:
        mu_pos = self.mu_pos.to(hidden)
        mu_neg = self.mu_neg.to(hidden)
        d_pos = ((hidden - mu_pos) ** 2).sum(dim=-1)
        d_neg = ((hidden - mu_neg) ** 2).sum(dim=-1)
        return d_neg - d_pos
