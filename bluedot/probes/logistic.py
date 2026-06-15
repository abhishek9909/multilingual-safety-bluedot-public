"""Logistic regression probe (sklearn-backed)."""

import torch
from torch import Tensor

from .base import Probe


class LogisticProbe(Probe):
    """
    Standard linear logistic regression on the chosen layer's hidden states.

    Useful as a learned alternative to the diff-of-means direction.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.clf = None

    def fit(self, hidden: Tensor, labels) -> "LogisticProbe":
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter)
        self.clf.fit(hidden.detach().cpu().numpy(), labels)
        return self

    def score(self, hidden: Tensor) -> Tensor:
        if self.clf is None:
            raise RuntimeError("LogisticProbe is not fit yet.")
        # use decision_function so caller can pick a threshold
        s = self.clf.decision_function(hidden.detach().cpu().numpy())
        return torch.from_numpy(s)
