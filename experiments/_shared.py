"""
Shared infrastructure for all four experiments.

Locked decisions from PLAN.md live here as module-level constants. Every
experiment script imports from this file so configuration changes propagate
to all of them at once.

Public API:
    CONFIG                 — frozen experiment config
    get_model()            — lazy, cached model load (called at most once per process)
    load_pair(lang, split) — (harmful_prompts, harmless_prompts) for a language
    extract_or_load_direction(model, lang, kind)
                           — returns Direction, caching to disk
    extract_or_load_acts(model, lang, kind, split)
                           — returns hidden states at (kind position, every layer)
    direction_path(lang, kind)
    acts_path(lang, kind, split)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch

from bluedot.models import load_model, ModelWrapper
from bluedot.datasets import load, to_prompts
from bluedot.directions import (
    Direction,
    extract_direction,
    save_direction,
    load_direction,
)
from bluedot.positions import t_inst_position, t_post_inst_position
from bluedot.activations import extract_activations
from bluedot.interventions import generate_with_intervention
from bluedot.utils import set_seed


# ---------------------------------------------------------------------------
# Locked config (see PLAN.md)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Config:
    model_path: str
    template: str
    langs: Tuple[str, ...]
    layer: int
    n_train: int
    n_val: int
    n_test_probe: int     # per (lang, label) for the probe arm
    n_test_causal: int    # per (src, tgt) for the causal arm
    seed: int
    poly_root: str
    artifact_dir: str
    batch_size: int
    # Train-set filtering (Wang et al.): before extracting a direction, drop
    # any "harmful" prompt the model doesn't actually refuse at baseline, and
    # any "harmless" prompt it doesn't actually comply with. Cleaner diffs.
    filter_train: bool
    filter_max_new_tokens: int    # short prefix is enough to detect a refusal


CONFIG = _Config(
    model_path=os.environ.get("BLUEDOT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
    # Chat template family. Defaults match the default model. Override with
    # BLUEDOT_TEMPLATE=qwen2 for Qwen2 / Qwen2.5 runs.
    template=os.environ.get("BLUEDOT_TEMPLATE", "llama3"),
    # NOTE: PolyRefuse is missing harmless_test_translated_zh.json, so we
    # drop zh from the default set. To opt back in (e.g. for the train-only
    # direction extraction in Exp 4), pass --langs en de fr ru ar ja zh on
    # the script and ensure it never touches harmless_test for zh.
    langs=("en", "de", "fr", "ru", "ar", "ja"),
    # Mid-layer for direction extraction / ablation. Llama3-8B-Instruct has
    # 32 layers, L=14 is ~44% depth. For Qwen2.5-7B-Instruct (28 layers),
    # L=12 is the proportional equivalent. Override with BLUEDOT_LAYER.
    layer=int(os.environ.get("BLUEDOT_LAYER", "14")),
    n_train=128,
    n_val=32,
    n_test_probe=100,
    n_test_causal=50,
    seed=42,
    poly_root=os.environ.get(
        "BLUEDOT_POLY_ROOT",
        os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "Multilingual-Refusal", "dataset"
        )),
    ),
    artifact_dir=os.environ.get(
        "BLUEDOT_ARTIFACTS",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts")),
    ),
    batch_size=int(os.environ.get("BLUEDOT_BS", "8")),
    filter_train=os.environ.get("BLUEDOT_FILTER_TRAIN", "1") == "1",
    filter_max_new_tokens=32,
)


# ---------------------------------------------------------------------------
# Path conventions
# ---------------------------------------------------------------------------

def direction_path(lang: str, kind: str, suffix: str = "") -> str:
    """Standard path for a saved direction. `suffix` lets variants live
    alongside (e.g. 'inv' for inversion-task directions)."""
    name = f"{lang}_{kind}" + (f"_{suffix}" if suffix else "")
    return os.path.join(CONFIG.artifact_dir, "directions", name)


def acts_path(lang: str, kind: str, split: str) -> str:
    return os.path.join(
        CONFIG.artifact_dir, "activations", f"{lang}_{kind}_{split}.pt"
    )


def results_path(exp: str, name: str) -> str:
    out = os.path.join(CONFIG.artifact_dir, exp)
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, name)


# ---------------------------------------------------------------------------
# Cached model
# ---------------------------------------------------------------------------

_MODEL: ModelWrapper | None = None


def get_model() -> ModelWrapper:
    """Lazy-load the model exactly once per process."""
    global _MODEL
    if _MODEL is None:
        set_seed(CONFIG.seed)
        _MODEL = load_model(CONFIG.model_path, CONFIG.template)
    return _MODEL


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pair(lang: str, split: str, n: int) -> Tuple[List[str], List[str]]:
    """Return (harmful_prompts, harmless_prompts) for a language and split."""
    harmful = load(
        "polyrefuse", root=CONFIG.poly_root,
        harmtype="harmful", split=split, lang=lang, n=n, seed=CONFIG.seed,
    )
    harmless = load(
        "polyrefuse", root=CONFIG.poly_root,
        harmtype="harmless", split=split, lang=lang, n=n, seed=CONFIG.seed,
    )
    return to_prompts(harmful), to_prompts(harmless)


# ---------------------------------------------------------------------------
# Train-set filtering (Wang et al. baseline-behavior screen)
# ---------------------------------------------------------------------------

def _filter_path(lang: str) -> str:
    return os.path.join(CONFIG.artifact_dir, "filtered_train", f"{lang}.json")


def filter_train_by_behavior(
    model: ModelWrapper,
    harmful: List[str],
    harmless: List[str],
    lang: str,
) -> Tuple[List[str], List[str]]:
    """
    Generate short prefixes for every train prompt; keep only those whose
    baseline behavior matches their label.

      kept_harmful  := harmful prompts the model refuses at baseline
      kept_harmless := harmless prompts the model complies with at baseline

    Cached to JSON per language so direction extraction is reproducible.
    """
    # Avoid an import cycle.
    from .judge import is_refusal

    cache = _filter_path(lang)
    if os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        return d["harmful"], d["harmless"]

    print(f"[filter] generating baseline behavior for lang={lang} "
          f"(n_harm={len(harmful)}, n_harmless={len(harmless)})")
    harm_out = generate_with_intervention(
        model, harmful, intervention=None,
        batch_size=CONFIG.batch_size,
        max_new_tokens=CONFIG.filter_max_new_tokens,
    )
    harmless_out = generate_with_intervention(
        model, harmless, intervention=None,
        batch_size=CONFIG.batch_size,
        max_new_tokens=CONFIG.filter_max_new_tokens,
    )

    kept_harm = [p for p, o in zip(harmful, harm_out)
                 if is_refusal(o["response"], lang)]
    kept_harmless = [p for p, o in zip(harmless, harmless_out)
                     if not is_refusal(o["response"], lang)]

    print(f"[filter] kept {len(kept_harm)}/{len(harmful)} harmful  "
          f"and {len(kept_harmless)}/{len(harmless)} harmless")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"harmful": kept_harm, "harmless": kept_harmless}, f,
                  ensure_ascii=False, indent=2)
    return kept_harm, kept_harmless


# ---------------------------------------------------------------------------
# Direction caching
# ---------------------------------------------------------------------------

def extract_or_load_direction(
    model: ModelWrapper,
    lang: str,
    kind: str,
    *,
    suffix: str = "",
    n_train: int | None = None,
    prompts_override: Tuple[List[str], List[str]] | None = None,
    filter_train: bool | None = None,
) -> Direction:
    """
    Return a per-layer Direction at the given kind for the given language.
    Caches to disk; subsequent calls are free.

    `prompts_override` lets callers supply already-prepared prompts (used by
    Exp 2 for inversion variants). When `prompts_override` is set the
    training-set behavior filter is bypassed — the caller has already chosen
    their prompts deliberately.

    `filter_train` overrides CONFIG.filter_train if set.
    """
    p = direction_path(lang, kind, suffix=suffix)
    if os.path.exists(p + ".pt"):
        return load_direction(p)

    do_filter = CONFIG.filter_train if filter_train is None else filter_train

    if prompts_override is not None:
        harmful, harmless = prompts_override
    else:
        harmful, harmless = load_pair(lang, "train", n_train or CONFIG.n_train)
        if do_filter:
            harmful, harmless = filter_train_by_behavior(
                model, harmful, harmless, lang
            )

    d = extract_direction(
        model, harmful, harmless,
        kind=kind, batch_size=CONFIG.batch_size, lang=lang,
        source_pos_dataset=f"polyrefuse:harmful:{lang}"
                           + (":filtered" if do_filter else ""),
        source_neg_dataset=f"polyrefuse:harmless:{lang}"
                           + (":filtered" if do_filter else ""),
    )
    save_direction(d, p)
    return d


# ---------------------------------------------------------------------------
# Activation caching (used by probe arms and Exp 3)
# ---------------------------------------------------------------------------

def extract_or_load_acts(
    model: ModelWrapper,
    lang: str,
    kind: str,
    split: str = "test",
    *,
    n: int | None = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Returns (acts, labels) where:
        acts:   [2N, n_layers, d_model]  hidden states at the kind position,
                stacked as [harmful (N), harmless (N)]
        labels: list of 2N ints (1=harmful, 0=harmless)

    Cached to disk.
    """
    n = n or CONFIG.n_test_probe
    path = acts_path(lang, kind, split)
    if os.path.exists(path):
        bundle = torch.load(path, map_location="cpu")
        return bundle["acts"], bundle["labels"]

    pos_fn = t_inst_position if kind == "t_inst" else t_post_inst_position
    pos = pos_fn(model)

    harmful, harmless = load_pair(lang, split, n)
    prompts = harmful + harmless
    labels = [1] * len(harmful) + [0] * len(harmless)

    # extract_activations returns [N, n_positions, n_layers, d_model]
    acts4 = extract_activations(model, prompts, pos, batch_size=CONFIG.batch_size)
    # Single position — squeeze.
    acts = acts4[:, 0, :, :].contiguous()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"acts": acts, "labels": labels}, path)
    return acts, labels
