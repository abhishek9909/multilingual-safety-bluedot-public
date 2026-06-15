"""
Loaders for the PolyRefuse multilingual dataset.

Expected layout (matching ../Multilingual-Refusal/dataset):
    {root}/splits/{harmtype}_{split}.json
    {root}/splits_multi/{harmtype}_{split}_translated_{lang}.json

Each json file is a list of dicts with at least 'instruction'
(and 'instruction_translated' for multi).
"""

import json
import os
import random
from typing import List, Optional

from .common import Example
from .registry import register_loader


def _read_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def polyrefuse_split(
    root: str,
    harmtype: str,
    split: str = "train",
    lang: str = "en",
    n: Optional[int] = None,
    seed: int = 0,
    fallback_split: Optional[str] = None,
) -> List[Example]:
    """
    Load PolyRefuse rows. If the requested file is missing AND
    `fallback_split` is set, try that split instead and log the substitution.
    Used to paper over gaps like missing harmless_test for zh.
    """
    def _path(s):
        if lang == "en":
            return os.path.join(root, "splits", f"{harmtype}_{s}.json")
        return os.path.join(
            root, "splits_multi", f"{harmtype}_{s}_translated_{lang}.json",
        )

    path = _path(split)
    if not os.path.exists(path) and fallback_split is not None:
        alt = _path(fallback_split)
        if os.path.exists(alt):
            print(f"[polyrefuse] {path} missing — falling back to {alt}")
            path = alt
    rows = _read_json(path)

    examples = []
    for r in rows:
        prompt = r.get("instruction_translated") or r["instruction"]
        examples.append(Example(
            prompt=prompt,
            label=harmtype,
            lang=lang,
            prompt_en=r.get("instruction"),
            meta={k: v for k, v in r.items() if k not in ("instruction", "instruction_translated")},
        ))

    if n is not None and n < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, n)
    return examples


register_loader("polyrefuse", polyrefuse_split)
