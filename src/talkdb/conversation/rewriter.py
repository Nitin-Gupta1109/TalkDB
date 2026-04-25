"""
Follow-up question rewriter.

Takes a short follow-up ("just Q4", "break that down by region") plus the recent turns
of a session and emits a standalone question that the SQL generator can process without
any knowledge of the conversation. This keeps the SQL pipeline stateless.

Per CLAUDE.md: conversation rewriting, not SQL mutation. We rewrite the *question*,
not the SQL — the generator always receives a clean, unambiguous input.
"""

from __future__ import annotations

import litellm

from talkdb.config.settings import Settings
from talkdb.conversation.session import Session

REWRITER_SYSTEM = """You rewrite follow-up questions into standalone questions.

You are given:
- The most recent turns of a conversation (each turn = user question + SQL generated + result summary).
- A new user message that may be a follow-up referring to the previous turns.

Your job:
- If the new message IS a follow-up (refers to pronouns like "that"/"those", implicit subjects, adds filters, changes groupings, shifts time windows, etc.), rewrite it as a complete standalone question that includes all context needed to answer it from scratch.
- If the new message is ALREADY a complete, unambiguous question, return it unchanged.
- Return ONLY the rewritten question. No preamble. No explanation. No quotes.
- Preserve the user's intent exactly. Do not add analysis or opinions.
"""

REWRITER_SYSTEM_GROUNDED = """You rewrite follow-up questions into standalone questions using full conversational grounding.

You are given:
- Recent turns, each showing the user's question, the SQL that was run, AND sample result rows.
- A new user message.

Your job:
- If the new message contains pronouns ("him", "her", "that", "those") or proper nouns that were introduced in a prior turn's results (a specific customer name, product, stadium, etc.), replace them with the CONCRETE VALUES from the prior rows. Example: "what is the capacity of Balmoor?" following a turn that returned stadium rows including Balmoor → rewrite as: "What is the capacity of the stadium named Balmoor?"
- If the new message adds a filter/grouping/ordering ("just Q4", "by region"), merge it with the prior standalone question.
- If the new message is already a complete, unambiguous question, return it unchanged.
- Return ONLY the rewritten question. No preamble, explanation, or quotes.
- NEVER invent values that aren't in the prior turns. If a reference is unresolvable, leave it as-is.
"""


class QuestionRewriter:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def rewrite(self, question: str, session: Session) -> str:
        if not session.has_turns():
            return question

        grounded = getattr(self.settings, "context_grounded_rewriter", False)
        system_prompt = REWRITER_SYSTEM_GROUNDED if grounded else REWRITER_SYSTEM
        user_msg = self._build_user_message(question, session, grounded=grounded)
        response = await litellm.acompletion(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=400,
        )
        content = (response.choices[0].message.content or "").strip()
        # Strip wrapping quotes if the model added them despite instructions.
        if len(content) >= 2 and content[0] in "\"'" and content[-1] == content[0]:
            content = content[1:-1]
        return content or question

    @staticmethod
    def _build_user_message(question: str, session: Session, grounded: bool = False) -> str:
        recent = session.recent_turns(n=3)
        lines: list[str] = ["Recent conversation:"]
        for t in recent:
            lines.append(f"Turn {t.turn_number} — user: {t.question}")
            if t.rewritten_question != t.question:
                lines.append(f"  (rewritten to standalone: {t.rewritten_question})")
            lines.append(f"  SQL: {t.sql}")
            lines.append(f"  Result: {t.results_summary}")
            if grounded and t.sample_rows:
                # Include up to 3 concrete result rows so the LLM can resolve pronouns
                # and proper nouns to actual values from the data.
                lines.append(f"  Sample rows: {t.sample_rows[:3]}")
        lines.append("")
        lines.append(f"New user message: {question}")
        lines.append("")
        lines.append("Rewrite the new message as a standalone question (or return unchanged if already standalone).")
        return "\n".join(lines)
