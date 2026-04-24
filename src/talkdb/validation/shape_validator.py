"""Shape validator: does the returned result shape match the question's intent?"""

from __future__ import annotations

from dataclasses import dataclass, field

from talkdb.core.intent import Intent, IntentType


@dataclass
class ShapeResult:
    matches: bool
    warnings: list[str] = field(default_factory=list)


def validate_shape(intent: Intent, columns: list[str], row_count: int) -> ShapeResult:
    warnings: list[str] = []
    col_count = len(columns)

    if intent.type == IntentType.AGGREGATION and intent.is_single_value:
        if row_count != 1 or col_count > 2:
            warnings.append(
                f"Question implies a single value but result has {row_count} rows x {col_count} cols"
            )
    elif intent.type == IntentType.RANKING:
        if row_count == 0:
            warnings.append("Ranking question returned 0 rows")
        if col_count < 2:
            warnings.append("Ranking question typically returns at least 2 columns (label + metric)")
    elif intent.type == IntentType.DISTRIBUTION:
        if row_count <= 1:
            warnings.append("Distribution/breakdown question returned <=1 row; grouping may have collapsed")
        if col_count < 2:
            warnings.append("Distribution question typically returns a group column plus a metric")
    elif intent.type == IntentType.LOOKUP:
        if row_count == 0:
            warnings.append("Lookup returned 0 rows")

    return ShapeResult(matches=not warnings, warnings=warnings)
