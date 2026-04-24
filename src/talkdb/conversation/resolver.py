"""
Heuristic detector for whether a question is a follow-up that needs rewriting.

Intentionally cheap — no LLM call. The rewriter itself is tolerant of false positives
(it will pass through already-standalone questions unchanged), so missing a follow-up
is worse than over-flagging. We err on the side of "yes, rewrite" when a session has turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from talkdb.conversation.session import Session

# Clear follow-up signals: pronouns, implicit subject starters, refinement verbs.
_FOLLOW_UP_PATTERNS = [
    re.compile(r"\b(that|those|them|it|these|this)\b", re.I),
    re.compile(r"^\s*(now|just|only|also|instead|but|and)\b", re.I),
    re.compile(r"^\s*(by|per|grouped by|broken down by)\b", re.I),
    re.compile(r"^\s*(exclude|include|filter|remove|add|drop)\b", re.I),
    re.compile(r"^\s*(sort|order|limit|top|bottom)\b", re.I),
    re.compile(r"\b(last year|this year|last month|this month|last week|this week|yesterday)\b", re.I),
    re.compile(r"\b(compare|versus|vs\.?|against)\b", re.I),
]


@dataclass
class ResolvedQuestion:
    original: str
    is_follow_up: bool
    trigger: str | None  # Which pattern fired, if any


class ReferenceResolver:
    """Cheap follow-up detector. The rewriter handles the actual context injection."""

    def resolve(self, question: str, session: Session | None) -> ResolvedQuestion:
        if session is None or not session.has_turns():
            return ResolvedQuestion(original=question, is_follow_up=False, trigger=None)

        for pattern in _FOLLOW_UP_PATTERNS:
            m = pattern.search(question)
            if m:
                return ResolvedQuestion(original=question, is_follow_up=True, trigger=m.group(0))

        # When a session is active, very short questions (< 7 words) are almost always follow-ups.
        if len(question.split()) < 7:
            return ResolvedQuestion(original=question, is_follow_up=True, trigger="short_question")

        return ResolvedQuestion(original=question, is_follow_up=False, trigger=None)
