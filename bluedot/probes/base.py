"""
Probe API. Probes consume hidden states (Tensor) + (optional) labels
and produce per-example scores. They do NOT run models.

This separation means a probe can be applied to ANY (model, dataset)
combination as long as you can get hidden states.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from torch import Tensor


@dataclass
class ProbeResult:
    scores: Tensor                    # [N] scalar score per example
    predictions: Optional[Tensor] = None   # [N] binary 0/1 if applicable
    threshold: Optional[float] = None
    metrics: Dict[str, Any] = None    # auc, accuracy, etc. if labels were given


class Probe(ABC):
    """Abstract probe.

    Convention: higher score = more "positive class" (e.g. more harmful).
    """

    @abstractmethod
    def score(self, hidden: Tensor) -> Tensor:
        """hidden: [N, d_model]  ->  scores: [N]"""

    def evaluate(
        self,
        hidden: Tensor,
        labels: Optional[List[int]] = None,
        threshold: Optional[float] = None,
    ) -> ProbeResult:
        scores = self.score(hidden)
        pred = None
        metrics = None
        if threshold is not None:
            pred = (scores > threshold).long()
        if labels is not None:
            labels_t = torch.tensor(labels)
            metrics = {}
            if pred is not None:
                metrics["accuracy"] = (pred == labels_t).float().mean().item()
            try:
                from sklearn.metrics import roc_auc_score
                metrics["auroc"] = float(
                    roc_auc_score(labels_t.numpy(), scores.detach().cpu().numpy())
                )
            except Exception:
                pass
        return ProbeResult(scores=scores, predictions=pred, threshold=threshold, metrics=metrics)
