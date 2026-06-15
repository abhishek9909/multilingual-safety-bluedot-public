"""
Generic JSON / JSONL loaders. Useful for the other repo's data formats
(e.g. LLMs_Encode_*) and for ad-hoc experiment datasets.
"""

import json
import os
from typing import List, Optional

from .common import Example
from .registry import register_loader


def json_array(
    path: str,
    *,
    prompt_key: str = "instruction",
    label: Optional[str] = None,
    lang: str = "en",
) -> List[Example]:
    """Standard `[{prompt_key: "...", ...}, ...]` JSON file."""
    with open(path) as f:
        rows = json.load(f)
    return [
        Example(
            prompt=r[prompt_key], label=label, lang=lang,
            meta={k: v for k, v in r.items() if k != prompt_key},
        )
        for r in rows
    ]


def jsonl(
    path: str,
    *,
    prompt_key: str = "instruction",
    label: Optional[str] = None,
    lang: str = "en",
) -> List[Example]:
    """One-JSON-per-line file."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(Example(
                prompt=r[prompt_key], label=label, lang=lang,
                meta={k: v for k, v in r.items() if k != prompt_key},
            ))
    return out


def processed_dir(
    root: str,
    name: str,
    *,
    label: Optional[str] = None,
    lang: str = "en",
) -> List[Example]:
    """Loader for {root}/processed/{name}.json (Multilingual-Refusal layout)."""
    path = os.path.join(root, "processed", f"{name}.json")
    return json_array(path, label=label, lang=lang)


register_loader("json_array", json_array)
register_loader("jsonl", jsonl)
register_loader("processed", processed_dir)
