from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from talkdb.config.settings import get_settings
from talkdb.core.engine import Engine, UnsafeSQLError

mcp = FastMCP("talkdb")
_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine(get_settings())
    return _engine


@mcp.tool()
async def ask(question: str, database: str | None = None, session_id: str | None = None) -> dict:
    """
    Ask a question in plain English, get SQL + results.

    Pass session_id to continue a multi-turn conversation. Follow-ups like
    "now by region" or "exclude refunds" are resolved against previous turns.

    Args:
        question: Natural language question.
        database: Optional database id (uses default if omitted).
        session_id: Optional session id. If provided and unknown, a new session is created.
    """
    engine = _get_engine()
    try:
        result = await engine.ask(question, database=database, session_id=session_id)
    except UnsafeSQLError as e:
        return {"error": f"unsafe_sql: {e}", "sql": "", "results": []}
    return result.model_dump()


@mcp.tool()
async def analyze(question: str, database: str | None = None) -> dict:
    """
    Ask a question and get results + narrative + chart (insight pipeline always on).

    Same as `ask` but forces the insight layer: runs pandas-based statistical analysis,
    auto-generates a chart when appropriate, and produces a short natural-language summary.
    """
    engine = _get_engine()
    try:
        result = await engine.analyze(question, database=database)
    except UnsafeSQLError as e:
        return {"error": f"unsafe_sql: {e}", "sql": "", "results": []}
    return result.model_dump()


@mcp.tool()
async def follow_up(refinement: str, session_id: str) -> dict:
    """
    Refine the previous query in a session.

    Examples: "break that down by region", "just Q4", "exclude refunds", "sort descending".

    Args:
        refinement: The follow-up text.
        session_id: An existing session id from a prior `ask` call.
    """
    engine = _get_engine()
    try:
        result = await engine.follow_up(refinement, session_id=session_id)
    except UnsafeSQLError as e:
        return {"error": f"unsafe_sql: {e}", "sql": "", "results": []}
    return result.model_dump()


@mcp.tool()
def list_databases() -> list[dict]:
    """List all configured database connections."""
    return _get_engine().list_databases()


@mcp.tool()
def describe_database(database: str | None = None) -> dict:
    """Show tables, columns, and relationships for a database."""
    return _get_engine().describe_database(database)


@mcp.tool()
def validate_sql(sql: str, database: str | None = None) -> dict:
    """Validate a SQL query without executing it. Returns schema issues and any sample-run errors."""
    return _get_engine().validate_sql(sql, database=database)


@mcp.tool()
def correct_query(
    original_question: str,
    wrong_sql: str,
    correct_sql: str,
    database: str | None = None,
) -> dict:
    """
    Submit a correction. The correct SQL is stored as a proven pattern and indexed
    into the retriever so future similar questions benefit.

    Args:
        original_question: The user's original question.
        wrong_sql: The SQL that was incorrect (stored as a negative example).
        correct_sql: The SQL that actually answers the question.
        database: Optional database id (for scoping the pattern).
    """
    return _get_engine().correct_query(
        original_question=original_question,
        wrong_sql=wrong_sql,
        correct_sql=correct_sql,
        database=database,
    )


@mcp.tool()
async def watch(
    name: str,
    question: str,
    schedule: str = "every 1 hour",
    alert_condition: str = "",
    database: str | None = None,
    webhook_url: str | None = None,
    slack_webhook_url: str | None = None,
) -> dict:
    """
    Save a query as a proactive watch. Runs on schedule and alerts when conditions are met.

    Args:
        name: Human-readable name.
        question: Natural language question to run periodically.
        schedule: "every N minute/hour/day", "daily at HH:MM am/pm", or a 5-field cron expression.
        alert_condition: NL alert trigger (e.g. "drops more than 20% below 7-day average").
        database: Database id. Uses default if omitted.
        webhook_url: Generic JSON webhook for alerts.
        slack_webhook_url: Slack incoming webhook URL.
    """
    engine = _get_engine()
    try:
        w = await engine.watchdog.add_watch(
            name=name,
            question=question,
            schedule=schedule,
            alert_condition=alert_condition,
            database=database,
            webhook_url=webhook_url,
            slack_webhook_url=slack_webhook_url,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {
        "name": w.name,
        "sql": w.sql,
        "schedule": w.schedule,
        "alert_condition": _condition_to_dict(w.alert_condition),
        "delivery_channels": w.delivery_channels,
        "created_at": w.created_at.isoformat(),
    }


@mcp.tool()
def list_watches() -> list[dict]:
    """List all watches and their last-run status."""
    engine = _get_engine()
    out: list[dict] = []
    for w in engine.watchdog.list_watches():
        out.append(
            {
                "name": w.name,
                "question": w.question,
                "schedule": w.schedule,
                "alert_condition": _condition_to_dict(w.alert_condition),
                "last_run": w.last_run.isoformat() if w.last_run else None,
                "last_value": w.last_value,
                "last_status": w.last_status,
                "last_message": w.last_message,
                "is_active": w.is_active,
            }
        )
    return out


@mcp.tool()
def remove_watch(name: str) -> dict:
    """Remove a saved watch."""
    removed = _get_engine().watchdog.remove_watch(name)
    return {"removed": removed, "name": name}


@mcp.tool()
async def run_watch(name: str) -> dict:
    """Manually execute a watch now (useful for testing alert conditions)."""
    engine = _get_engine()
    try:
        run = await engine.watchdog.run_watch(name)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {
        "watch_name": run.watch_name,
        "ran_at": run.ran_at.isoformat(),
        "value": run.value,
        "baseline": run.baseline,
        "baseline_type": run.baseline_type,
        "triggered": run.triggered,
        "status": run.status,
        "message": run.message,
    }


def _condition_to_dict(cond) -> dict:
    return {
        "kind": cond.kind,
        "threshold_value": cond.threshold_value,
        "threshold_direction": cond.threshold_direction,
        "change_percent": cond.change_percent,
        "baseline_type": cond.baseline_type,
        "anomaly_std_devs": cond.anomaly_std_devs,
        "description": cond.description,
    }


@mcp.tool()
def get_session(session_id: str) -> dict | None:
    """Return the current state of a conversation session (turns, SQL, results summary)."""
    return _get_engine().get_session(session_id)


def run_stdio() -> None:
    engine = _get_engine()
    if engine.settings.watchdog_enabled:
        engine.watchdog.start()
    mcp.run(transport="stdio")


def run_sse(host: str = "0.0.0.0", port: int = 8000) -> None:
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="sse")
