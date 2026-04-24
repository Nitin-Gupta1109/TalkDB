"""
Combine validation signals into a 0-100 confidence score.

Weights match PROJECT_SPEC.md section 9e. Dual-path and pattern-similarity slots
are wired but inert in Phase 3 — filled in by Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from talkdb.validation.execution_validator import ExecutionResult
from talkdb.validation.schema_validator import SchemaValidationResult
from talkdb.validation.shape_validator import ShapeResult


@dataclass
class ConfidenceScore:
    value: int  # 0-100
    breakdown: dict[str, int]
    refused: bool
    explanation: str = ""
    warnings: list[str] = field(default_factory=list)


DEFAULT_THRESHOLD = 50


def calculate_confidence(
    schema_result: SchemaValidationResult,
    execution_result: ExecutionResult,
    shape_result: ShapeResult,
    *,
    retrieval_similarity: float = 0.5,
    semantic_coverage: float = 0.5,
    dual_path_agreement: float | None = None,
    threshold: int = DEFAULT_THRESHOLD,
) -> ConfidenceScore:
    """
    Returns a 0-100 score. A short-circuit: any hard failure (schema invalid or
    execution failed) caps the score at 0 — those are unrecoverable.

    retrieval_similarity, semantic_coverage: 0-1 floats from the retriever.
    dual_path_agreement: 0-1 float, or None if dual-path is disabled.
    """

    if not schema_result.valid:
        return ConfidenceScore(
            value=0,
            breakdown={"schema": 0},
            refused=True,
            explanation="Schema validation failed: " + schema_result.error_message(),
        )

    if not execution_result.ok:
        return ConfidenceScore(
            value=0,
            breakdown={"execution": 0},
            refused=True,
            explanation=f"Execution failed: {execution_result.error}",
        )

    # Base signals (all 0-1 normalized, then weighted).
    schema_signal = 1.0  # binary pass/fail already checked above
    execution_signal = 1.0
    shape_signal = 1.0 if shape_result.matches else 0.5
    similarity = max(0.0, min(1.0, retrieval_similarity))
    coverage = max(0.0, min(1.0, semantic_coverage))
    dual_path_signal = 0.5 if dual_path_agreement is None else max(0.0, min(1.0, dual_path_agreement))

    weights = {
        "schema": 25,
        "execution": 20,
        "shape": 10,
        "dual_path": 20,
        "pattern_similarity": 15,
        "semantic_coverage": 10,
    }

    breakdown = {
        "schema": int(schema_signal * weights["schema"]),
        "execution": int(execution_signal * weights["execution"]),
        "shape": int(shape_signal * weights["shape"]),
        "dual_path": int(dual_path_signal * weights["dual_path"]),
        "pattern_similarity": int(similarity * weights["pattern_similarity"]),
        "semantic_coverage": int(coverage * weights["semantic_coverage"]),
    }

    value = sum(breakdown.values())
    warnings = list(shape_result.warnings)

    refused = value < threshold
    explanation = ""
    if refused:
        explanation = (
            f"Confidence {value} is below threshold {threshold}. "
            "Refusing to return results rather than risk a silent wrong answer."
        )

    return ConfidenceScore(
        value=value,
        breakdown=breakdown,
        refused=refused,
        explanation=explanation,
        warnings=warnings,
    )
