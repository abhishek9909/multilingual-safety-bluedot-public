"""
Per-response reasoning-chain auditing for the pilot experiments.

Two cheap, dependency-free instruments:

(1) Reasoning-language detection
    Identify the dominant Unicode script of the <reasoning> block and check
    whether it matches the target language. Llama3 tends to drift back to
    English reasoning even when instructed in another language; this flags
    every response where that drift happened.

(2) Reasoning-action consistency
    Parse the LAST digit (Exp 1) or LAST severity label (Exp 2) mentioned in
    the reasoning block and compare with the final <answer> tag. A mismatch
    means the model's CoT reasoned toward one answer but emitted a different
    one in the answer slot — a CoT-faithfulness failure.

Both produce per-response fields that are logged into results.json for the
analysis script to pivot on later.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# (1) Reasoning-language detection
# ---------------------------------------------------------------------------

# Each language is mapped to the Unicode-block name we expect to dominate its
# reasoning text. Languages sharing the Latin script can only be distinguished
# coarsely by this method, but the binary "is it in the expected script
# family at all?" check is sufficient for the drift diagnostic — the
# substantive failure mode we're catching is Llama3 falling back to English
# (Latin) when instructed in Chinese / Arabic / Bengali / etc.
LANG_TO_SCRIPT: Dict[str, str] = {
    "en": "LATIN",
    "fr": "LATIN", "de": "LATIN", "it": "LATIN", "es": "LATIN",
    "vi": "LATIN", "hu": "LATIN", "cs": "LATIN", "ms": "LATIN", "id": "LATIN",
    "zh": "CJK",
    "ja": "JAPANESE",        # mixed CJK + Hiragana + Katakana
    "ko": "HANGUL",
    "ru": "CYRILLIC", "sr": "CYRILLIC", "uk": "CYRILLIC", "bg": "CYRILLIC",
    "ar": "ARABIC",
    "bn": "BENGALI",
    "th": "THAI",
}


def _script_of_char(ch: str) -> Optional[str]:
    """Coarse script categorisation for a single character."""
    cp = ord(ch)
    if cp < 0x80:
        return "LATIN" if ch.isalpha() else None
    name = unicodedata.name(ch, "")
    if not name:
        return None
    if "CJK UNIFIED" in name:
        return "CJK"
    if "HIRAGANA" in name or "KATAKANA" in name:
        return "JAPANESE"
    if "HANGUL" in name:
        return "HANGUL"
    if "CYRILLIC" in name:
        return "CYRILLIC"
    if "ARABIC" in name:
        return "ARABIC"
    if "BENGALI" in name:
        return "BENGALI"
    if "THAI" in name:
        return "THAI"
    if "LATIN" in name:
        return "LATIN"
    return None


def dominant_script(text: str) -> Optional[str]:
    """Return the most frequent script among letter-characters, or None."""
    counts: Dict[str, int] = {}
    for ch in text:
        s = _script_of_char(ch)
        if s is None:
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def script_matches_language(text: str, lang: str) -> Optional[bool]:
    """
    True if the text's dominant script matches `lang`'s expected script,
    False if it falls back to a different script, None if undetectable.
    For Japanese, accept either JAPANESE or CJK as a match.
    """
    expected = LANG_TO_SCRIPT.get(lang)
    if expected is None:
        return None
    dom = dominant_script(text)
    if dom is None:
        return None
    if expected == "JAPANESE":
        return dom in ("JAPANESE", "CJK")
    return dom == expected


# ---------------------------------------------------------------------------
# (2) Reasoning-action consistency
# ---------------------------------------------------------------------------

_REASONING_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL
)


def extract_reasoning(response: str) -> str:
    """Return the text inside <reasoning></reasoning>, or the whole response
    up to the <answer> tag if no reasoning block is present."""
    m = _REASONING_RE.search(response)
    if m:
        return m.group(1).strip()
    # Fall back: everything before <answer> if any
    if "<answer>" in response.lower():
        return response.lower().split("<answer>", 1)[0].strip()
    return response.strip()


_DIGIT_RE      = re.compile(r"(?<!\d)([12])(?!\d)")
_SEVERITY_RE   = re.compile(r"\b(L[0-3])\b", re.IGNORECASE)


def last_digit_in_reasoning(text: str) -> Optional[int]:
    """Return the last standalone 1 or 2 mentioned in `text`, or None."""
    matches = _DIGIT_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def last_severity_in_reasoning(text: str) -> Optional[str]:
    """Return the last L0..L3 token mentioned in `text`, or None."""
    matches = _SEVERITY_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


# ---------------------------------------------------------------------------
# One-shot audit functions for each pilot experiment
# ---------------------------------------------------------------------------

def audit_exp1_response(response: str, *, lang: str, answer: Optional[int]) -> Dict:
    """
    For a single Exp 1 response, return the audit fields:
      reasoning                  — extracted reasoning text
      reasoning_lang_match       — bool|None
      reasoning_endorses_digit   — int|None (1 or 2 from the reasoning chain)
      reasoning_action_match     — bool|None (endorsement == answer tag?)
    """
    reasoning = extract_reasoning(response)
    endorsement = last_digit_in_reasoning(reasoning)
    lang_match = script_matches_language(reasoning, lang)
    action_match = (endorsement == answer) if endorsement is not None and answer is not None else None
    return {
        "reasoning":                reasoning,
        "reasoning_lang_match":     lang_match,
        "reasoning_endorses_digit": endorsement,
        "reasoning_action_match":   action_match,
    }


def audit_exp2_response(response: str, *, lang: str, answer: Optional[str]) -> Dict:
    """
    For a single Exp 2 perception response:
      reasoning                  — extracted reasoning text
      reasoning_lang_match       — bool|None
      reasoning_endorses_severity— str|None (L0/L1/L2/L3 from reasoning chain)
      reasoning_action_match     — bool|None (endorsement == answer tag?)
    """
    reasoning = extract_reasoning(response)
    endorsement = last_severity_in_reasoning(reasoning)
    lang_match = script_matches_language(reasoning, lang)
    if endorsement is not None and answer is not None:
        action_match = endorsement.upper() == answer.upper()
    else:
        action_match = None
    return {
        "reasoning":                  reasoning,
        "reasoning_lang_match":       lang_match,
        "reasoning_endorses_severity": endorsement,
        "reasoning_action_match":     action_match,
    }
