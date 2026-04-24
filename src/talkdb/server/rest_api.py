"""
FastAPI wrapper around the same Engine that powers the MCP server.

Mirrors the MCP tools as HTTP endpoints. Same engine instance, same caches,
same vector store, same pattern store. This is the secondary distribution channel —
for web clients, dashboards, or environments where MCP isn't available.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from talkdb.config.settings import Settings, get_settings
from talkdb.core.engine import Engine, UnsafeSQLError


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = Engine(settings)
    app = FastAPI(
        title="TalkDB",
        description="Autonomous data analyst — REST API",
        version="0.1.0",
    )

    # ----- Request schemas -----

    class AskRequest(BaseModel):
        question: str
        database: str | None = None
        session_id: str | None = None
        with_insights: bool | None = None

    class FollowUpRequest(BaseModel):
        refinement: str
        session_id: str

    class AnalyzeRequest(BaseModel):
        question: str
        database: str | None = None

    class ValidateRequest(BaseModel):
        sql: str
        database: str | None = None

    class CorrectionRequest(BaseModel):
        original_question: str
        wrong_sql: str
        correct_sql: str
        database: str | None = None

    class WatchRequest(BaseModel):
        name: str
        question: str
        schedule: str = "every 1 hour"
        alert_condition: str = ""
        database: str | None = None
        webhook_url: str | None = None
        slack_webhook_url: str | None = None

    class InstallRequest(BaseModel):
        source: str = Field(..., description="Package name, local directory path, tarball, or URL.")

    # ----- Routes -----

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/api/v1/ask")
    async def ask(req: AskRequest) -> dict:
        try:
            result = await engine.ask(
                req.question,
                database=req.database,
                session_id=req.session_id,
                with_insights=req.with_insights,
            )
        except UnsafeSQLError as e:
            raise HTTPException(status_code=400, detail=f"unsafe_sql: {e}") from e
        return result.model_dump()

    @app.post("/api/v1/follow-up")
    async def follow_up(req: FollowUpRequest) -> dict:
        try:
            result = await engine.follow_up(req.refinement, session_id=req.session_id)
        except UnsafeSQLError as e:
            raise HTTPException(status_code=400, detail=f"unsafe_sql: {e}") from e
        return result.model_dump()

    @app.post("/api/v1/analyze")
    async def analyze(req: AnalyzeRequest) -> dict:
        try:
            result = await engine.analyze(req.question, database=req.database)
        except UnsafeSQLError as e:
            raise HTTPException(status_code=400, detail=f"unsafe_sql: {e}") from e
        return result.model_dump()

    @app.post("/api/v1/validate")
    def validate(req: ValidateRequest) -> dict:
        return engine.validate_sql(req.sql, database=req.database)

    @app.get("/api/v1/databases")
    def list_databases() -> list[dict]:
        return engine.list_databases()

    @app.get("/api/v1/databases/{database}/schema")
    def describe(database: str) -> dict:
        return engine.describe_database(database)

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        state = engine.get_session(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"session {session_id} not found")
        return state

    @app.post("/api/v1/feedback")
    def correct(req: CorrectionRequest) -> dict:
        return engine.correct_query(
            original_question=req.original_question,
            wrong_sql=req.wrong_sql,
            correct_sql=req.correct_sql,
            database=req.database,
        )

    # Watchdog

    @app.post("/api/v1/watches")
    async def create_watch(req: WatchRequest) -> dict:
        try:
            w = await engine.watchdog.add_watch(
                name=req.name,
                question=req.question,
                schedule=req.schedule,
                alert_condition=req.alert_condition,
                database=req.database,
                webhook_url=req.webhook_url,
                slack_webhook_url=req.slack_webhook_url,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "name": w.name,
            "sql": w.sql,
            "schedule": w.schedule,
            "delivery_channels": w.delivery_channels,
        }

    @app.get("/api/v1/watches")
    def list_watches() -> list[dict]:
        return [
            {
                "name": w.name,
                "question": w.question,
                "schedule": w.schedule,
                "last_run": w.last_run.isoformat() if w.last_run else None,
                "last_value": w.last_value,
                "last_status": w.last_status,
            }
            for w in engine.watchdog.list_watches()
        ]

    @app.delete("/api/v1/watches/{name}")
    def remove_watch(name: str) -> dict:
        return {"removed": engine.watchdog.remove_watch(name), "name": name}

    @app.post("/api/v1/watches/{name}/run")
    async def run_watch(name: str) -> dict:
        try:
            r = await engine.watchdog.run_watch(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {
            "watch_name": r.watch_name,
            "ran_at": r.ran_at.isoformat(),
            "value": r.value,
            "baseline": r.baseline,
            "triggered": r.triggered,
            "status": r.status,
            "message": r.message,
        }

    # Registry

    @app.post("/api/v1/registry/install")
    def install(req: InstallRequest) -> dict:
        return engine.install_package(req.source)

    @app.delete("/api/v1/registry/packages/{name}")
    def uninstall(name: str) -> dict:
        return engine.uninstall_package(name)

    @app.get("/api/v1/registry/installed")
    def list_installed() -> list[dict]:
        return engine.list_installed_packages()

    @app.get("/api/v1/registry/search")
    def search(q: str) -> list[dict]:
        return engine.search_registry(q)

    return app


# Convenience for `uvicorn talkdb.server.rest_api:app`
app = create_app()
