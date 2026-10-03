"""
intent/normalizer.py

Phase 10: Input normalizer.

Turns raw text / voice transcripts into a canonical lowercase form:
    - unicode + whitespace normalisation
    - punctuation stripping (keeps digits, '%', '.', ',' as clause separator)
    - politeness / wake-word filler removal ("could you please", "hey vision")
    - article removal ("the", "a", "an", "my")
    - number words → digits ("fifty" → 50)
    - "percent" → "%"
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List


FILLER_PHRASES: List[str] = sorted([
    "could you please", "can you please", "would you please", "will you please",
    "could you", "can you", "would you", "will you", "would you mind",
    "i want you to", "i need you to", "i would like you to", "i'd like you to",
    "i would like to", "i'd like to", "i want to", "go ahead and",
    "please", "kindly", "for me", "right now", "thank you", "thanks",
    "hey vision", "ok vision", "okay vision", "hey vatsrani", "vatsrani",
], key=len, reverse=True)

ARTICLES = {"the", "a", "an", "my", "our", "your"}

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


@dataclass
class NormalizedInput:
    original: str
    text: str
    tokens: List[str] = field(default_factory=list)
    fillers_removed: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text


def _words_to_numbers(tokens: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "hundred" and out and out[-1].isdigit():
            out[-1] = str(int(out[-1]) * 100)
        elif t in _TENS:
            val = _TENS[t]
            if i + 1 < len(tokens) and tokens[i + 1] in _UNITS and 0 < _UNITS[tokens[i + 1]] < 10:
                val += _UNITS[tokens[i + 1]]
                i += 1
            out.append(str(val))
        elif t in _UNITS and t != "one":
            out.append(str(_UNITS[t]))
        elif t == "one" and i + 1 < len(tokens) and tokens[i + 1] in ("hundred", "percent", "%"):
            out.append("1")
        else:
            out.append(t)
        i += 1
    return out


def normalize(raw: str) -> NormalizedInput:
    original = raw if isinstance(raw, str) else ""
    text = unicodedata.normalize("NFKC", original).lower().strip()
    text = text.replace("’", "'")

    # Clause separators become " , " so the multi-action splitter sees them.
    text = re.sub(r"[;!?]+", " , ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " , ", text)
    text = re.sub(r"[^a-z0-9%.,'\s-]", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,")

    removed: List[str] = []
    for phrase in FILLER_PHRASES:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text):
            removed.append(phrase)
            text = re.sub(pattern, " ", text)

    tokens = [t for t in text.replace(",", " , ").split() if t not in ARTICLES]
    tokens = _words_to_numbers(tokens)
    text = " ".join(tokens)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:percent|per cent|%)", r"\1%", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"(,\s*)+", ", ", text).strip(" ,")
    text = re.sub(r"\s+", " ", text)

    return NormalizedInput(original=original, text=text, tokens=text.split(), fillers_removed=removed)
