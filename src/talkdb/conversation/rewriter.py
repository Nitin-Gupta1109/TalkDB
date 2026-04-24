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


class QuestionRewriter:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def rewrite(self, question: str, session: Session) -> str:
        if not session.has_turns():
            return question

        user_msg = self._build_user_message(question, session)
        response = await litellm.acompletion(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": REWRITER_SYSTEM},
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
    def _build_user_message(question: str, session: Session) -> str:
        recent = session.recent_turns(n=3)
        lines: list[str] = ["Recent conversation:"]
        for t in recent:
            lines.append(f"Turn {t.turn_number} — user: {t.question}")
            if t.rewritten_question != t.question:
                lines.append(f"  (rewritten to standalone: {t.rewritten_question})")
            lines.append(f"  SQL: {t.sql}")
            lines.append(f"  Result: {t.results_summary}")
        lines.append("")
        lines.append(f"New user message: {question}")
        lines.append("")
        lines.append("Rewrite the new message as a standalone question (or return unchanged if already standalone).")
        return "\n".join(lines)
