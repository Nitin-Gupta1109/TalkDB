"""
Lightweight rule-based intent classifier.

Intent informs the shape validator and confidence scorer. A more sophisticated
classifier (LLM-based) could replace this later without changing any callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class IntentType(str, Enum):
    AGGREGATION = "aggregation"  # single numeric value ("how many", "total", "average")
    RANKING = "ranking"  # ordered top/bottom N
    LOOKUP = "lookup"  # list or fetch multiple rows ("list all", "show me")
    COMPARISON = "comparison"  # period-over-period, a vs b
    DISTRIBUTION = "distribution"  # grouped aggregation ("by region", "breakdown")
    UNKNOWN = "unknown"


@dataclass
class Intent:
    type: IntentType
    is_single_value: bool  # true if we expect exactly 1 row, 1 column
    matched_keywords: list[str]


_PATTERNS: list[tuple[IntentType, re.Pattern[str], bool]] = [
    # (type, pattern, is_single_value)
    (IntentType.AGGREGATION, re.compile(r"\b(how many|how much|count of|total|sum of|average|mean|median|maximum|minimum|max|min)\b", re.I), True),
    (IntentType.RANKING, re.compile(r"\b(top|bottom|best|worst|highest|lowest|largest|smallest)\s+\d*\b", re.I), False),
    (IntentType.COMPARISON, re.compile(r"\b(compare|versus|vs\.?|vs|compared to|against)\b", re.I), False),
    (IntentType.DISTRIBUTION, re.compile(r"\b(by|per|grouped by|breakdown|distribution|across)\b", re.I), False),
    (IntentType.LOOKUP, re.compile(r"\b(list|show|which|what are|find|get|all the)\b", re.I), False),
]


def classify_intent(question: str) -> Intent:
    matched_types: list[tuple[IntentType, bool, str]] = []
    for intent_type, pattern, single in _PATTERNS:
        m = pattern.search(question)
        if m:
            matched_types.append((intent_type, single, m.group(0)))

    if not matched_types:
        return Intent(type=IntentType.UNKNOWN, is_single_value=False, matched_keywords=[])

    # Priority: AGGREGATION beats RANKING beats others. But AGGREGATION + DISTRIBUTION = DISTRIBUTION
    # ("revenue by region" is a grouped aggregation, not a single value).
    types = {t for t, _, _ in matched_types}
    keywords = [kw for _, _, kw in matched_types]

    if IntentType.DISTRIBUTION in types and IntentType.AGGREGATION in types:
        return Intent(type=IntentType.DISTRIBUTION, is_single_value=False, matched_keywords=keywords)
    if IntentType.AGGREGATION in types:
        return Intent(type=IntentType.AGGREGATION, is_single_value=True, matched_keywords=keywords)
    if IntentType.RANKING in types:
        return Intent(type=IntentType.RANKING, is_single_value=False, matched_keywords=keywords)
    if IntentType.COMPARISON in types:
        return Intent(type=IntentType.COMPARISON, is_single_value=False, matched_keywords=keywords)
    if IntentType.DISTRIBUTION in types:
        return Intent(type=IntentType.DISTRIBUTION, is_single_value=False, matched_keywords=keywords)

    return Intent(type=IntentType.LOOKUP, is_single_value=False, matched_keywords=keywords)
