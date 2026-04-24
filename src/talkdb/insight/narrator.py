"""
Insight narrator — the single LLM call in the insight pipeline.

Inputs the analyzer's structured findings. Outputs 2–4 sentences of prose. The
prompt is locked down so the model can ONLY reference numbers already computed
by the analyzer — no hallucinated stats, no invented comparisons.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import litellm

from talkdb.config.settings import Settings
from talkdb.insight.analyzer import AnalysisResult

NARRATOR_SYSTEM = """You are a data analyst writing a short summary of query results.

You are given:
- The user's original question.
- A structured JSON object with facts already computed from the data (totals, trends, top values, anomalies).

Your job:
- Write 2-4 sentences summarizing the result in plain English.
- Lead with the headline number (the most important fact).
- Explain the "why" only if the facts directly support it. Do not speculate.
- Flag anomalies if any are listed.
- End with ONE suggested follow-up question the user might want to ask next.
- CRITICAL: use only numbers present in the FACTS object. Do not invent values. Do not compute new metrics. Do not compare to periods or groups that are not in the facts.
- Use plain sentences. No bullets. No markdown. No preamble like "Here's a summary".
"""


class InsightNarrator:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def narrate(self, question: str, analysis: AnalysisResult) -> str:
        facts = _analysis_to_facts(analysis)
        user_msg = (
            f"Question: {question}\n\n"
            f"FACTS (JSON):\n{json.dumps(facts, default=str, indent=2)}\n\n"
            "Write the summary now."
        )
        response = await litellm.acompletion(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": NARRATOR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        content = (response.choices[0].message.content or "").strip()
        return content


def _analysis_to_facts(analysis: AnalysisResult) -> dict:
    """Flatten the AnalysisResult into a JSON-serializable fact sheet."""
    out: dict = {
        "row_count": analysis.row_count,
        "columns": analysis.columns,
        "intent": analysis.intent_type,
        "key_findings": analysis.key_findings,
    }
    if analysis.single_value is not None:
        out["single_value"] = asdict(analysis.single_value)
    if analysis.time_series is not None:
        out["time_series"] = asdict(analysis.time_series)
    if analysis.categorical is not None:
        out["categorical"] = asdict(analysis.categorical)
    return out
