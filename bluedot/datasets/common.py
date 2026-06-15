"""Common dataset format. All loaders return List[Example]."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Example:
    prompt: str                          # the prompt to feed the model
    label: Optional[str] = None          # 'harmful' / 'harmless' / etc
    lang: str = "en"
    prompt_en: Optional[str] = None      # English original if translated
    meta: Dict[str, Any] = field(default_factory=dict)


def to_prompts(examples: List[Example]) -> List[str]:
    return [e.prompt for e in examples]


def to_labels(examples: List[Example]) -> List[Optional[str]]:
    return [e.label for e in examples]
