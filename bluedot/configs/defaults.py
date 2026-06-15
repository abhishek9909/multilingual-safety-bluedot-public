"""
Lightweight experiment config. Use it or don't — every module accepts plain
kwargs so configs are optional.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ExperimentConfig:
    # model
    model_path: str = "Qwen/Qwen2-7B-Instruct"
    template: str = "qwen2"

    # data
    poly_root: Optional[str] = None      # path to PolyRefuse dataset
    source_lang: str = "en"
    n_train: int = 64
    n_val: int = 32
    seed: int = 42

    # extraction
    batch_size: int = 8
    direction_kinds: List[str] = field(default_factory=lambda: ["t_inst", "t_post_inst"])

    # intervention
    coeff: float = 1.0
    ablation_start_layer: int = 0

    # artifacts
    artifact_dir: str = "artifacts"
