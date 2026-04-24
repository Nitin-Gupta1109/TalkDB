from __future__ import annotations

import re

import litellm

from talkdb.config.settings import Settings

DECOMPOSE_SYSTEM_PROMPT = """You are a careful SQL engineer. Break the user's question into a plan, then write the SQL.

PROCESS (follow in order):
1. Identify the target metric or data the question is asking for.
2. Identify the tables that contain that data (from the RELEVANT CONTEXT).
3. List the joins needed — ONLY using the join rules provided. Never invent a join.
4. List the filters (WHERE conditions) — including any implicit ones from named metric definitions.
5. List the groupings and sort order if applicable.
6. Compose the final SQL from the pieces above.

OUTPUT FORMAT (strict):
- Write the final SQL ONLY, on the last line(s). No prose, no markdown fences.
- You MAY write a brief bulleted plan before the SQL, but the FINAL line(s) must be executable {dialect} SQL.
- If the question cannot be answered from the provided context, respond with exactly: CANNOT_ANSWER: <brief reason>

RULES:
- Only reference tables and columns that appear in the RELEVANT CONTEXT. NEVER invent identifiers.
- Basic aggregations (COUNT, SUM, AVG, MIN, MAX) are fine without a named metric.
- When the user refers to a named metric defined in the context, use its exact calculation (including filters).
- Use table aliases when joining for clarity.

DATABASE DIALECT: {dialect}

RELEVANT CONTEXT:
{context}
"""


SYSTEM_PROMPT = """You are an expert SQL engineer. Generate a single SQL query to answer the user's question.

RULES:
- Only reference tables and columns that appear in the RELEVANT CONTEXT below. NEVER invent tables or columns.
- Basic aggregations (COUNT, SUM, AVG, MIN, MAX) over existing columns are always fine — you do not need a named metric for them.
- When the user refers to a named metric that IS defined in the context, use that metric's exact calculation (including its filters).
- When a join rule is provided, use it verbatim instead of guessing the join condition.
- Only refuse (`CANNOT_ANSWER: <reason>`) when the question genuinely cannot be answered from the available tables and columns — e.g., the requested data is not in any column.
- Generate {dialect} SQL syntax.
- Use table aliases for clarity when joining.
- Return ONLY the SQL. No prose, no markdown fences, no explanation.

DATABASE DIALECT: {dialect}

RELEVANT CONTEXT:
{context}
"""


class GenerationRefusal(Exception):
    """Raised when the LLM declines to answer (CANNOT_ANSWER response)."""


class SQLGenerator:
    """LiteLLM-backed SQL generator. Prompt is built from focused retrieved context, not full schema dumps."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate(self, question: str, context: str, dialect: str) -> str:
        system = SYSTEM_PROMPT.format(dialect=dialect, context=context)
        response = await litellm.acompletion(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        content = response.choices[0].message.content or ""
        return self._extract_sql(content)

    async def generate_decomposed(self, question: str, context: str, dialect: str) -> str:
        """
        Path B for dual-path verification. Uses a decompose-then-compose strategy that
        forces the model to reason structurally differently than the direct path.
        The hope: if both paths independently arrive at the same result, the answer is reliable.
        """
        system = DECOMPOSE_SYSTEM_PROMPT.format(dialect=dialect, context=context)
        response = await litellm.acompletion(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        content = response.choices[0].message.content or ""
        # Strip any "Plan:" reasoning block the model may emit before the SQL.
        return self._extract_sql(_strip_plan_prefix(content))

    async def generate_retry(
        self,
        question: str,
        context: str,
        dialect: str,
        previous_sql: str,
        error_message: str,
    ) -> str:
        """One-shot retry: hand the previous SQL and the error back to the LLM for correction."""
        system = SYSTEM_PROMPT.format(dialect=dialect, context=context)
        retry_prompt = (
            f"Your previous attempt had an error. Fix it and return only the corrected SQL.\n\n"
            f"Original question: {question}\n\n"
            f"Previous SQL:\n{previous_sql}\n\n"
            f"Error:\n{error_message}\n\n"
            f"Return only the corrected SQL."
        )
        response = await litellm.acompletion(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        content = response.choices[0].message.content or ""
        return self._extract_sql(content)

    @staticmethod
    def _extract_sql(raw: str) -> str:
        text = raw.strip()
        if text.startswith("CANNOT_ANSWER"):
            reason = text.split(":", 1)[1].strip() if ":" in text else "no reason given"
            raise GenerationRefusal(reason)

        fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

        return text.rstrip(";").strip()


def _strip_plan_prefix(raw: str) -> str:
    """
    Path B may emit a bulleted plan before the SQL. Extract the last SELECT statement
    (or CANNOT_ANSWER line) — everything before that is reasoning we don't need.
    """
    text = raw.strip()
    if text.startswith("CANNOT_ANSWER"):
        return text

    upper = text.upper()
    for keyword in ("SELECT", "WITH "):
        idx = upper.rfind(keyword)
        if idx != -1:
            return text[idx:]
    return text
