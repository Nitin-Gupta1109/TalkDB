from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

from talkdb.config.settings import Settings
from talkdb.connectors.base import BaseConnector, get_connector
from talkdb.conversation.resolver import ReferenceResolver
from talkdb.conversation.rewriter import QuestionRewriter
from talkdb.conversation.session import (
    ConversationTurn,
    InMemorySessionStore,
    Session,
    SessionStore,
    summarize_result,
)
from talkdb.core.generator import GenerationRefusal, SQLGenerator
from talkdb.core.intent import Intent, classify_intent
from talkdb.insight.analyzer import AnalysisResult, InsightAnalyzer
from talkdb.insight.charter import InsightCharter
from talkdb.insight.narrator import InsightNarrator
from talkdb.learning.feedback import FeedbackRecorder
from talkdb.learning.pattern_store import PatternStore
from talkdb.watchdog.manager import WatchdogManager
from talkdb.retrieval.embeddings import EmbeddingClient
from talkdb.retrieval.hybrid_retriever import HybridRetriever, RetrievedDoc
from talkdb.retrieval.vector_store import ChromaVectorStore, VectorStore
from talkdb.schema.introspector import SchemaIntrospector
from talkdb.schema.models import DatabaseSchema, QueryResult
from talkdb.schema.semantic_model import SemanticModel
from talkdb.validation.confidence import calculate_confidence
from talkdb.validation.dual_path import DualPathResult, compare_results
from talkdb.validation.execution_validator import ExecutionResult, ExecutionValidator
from talkdb.validation.schema_validator import SchemaValidator
from talkdb.validation.shape_validator import validate_shape

MUTATING_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
)


class UnsafeSQLError(ValueError):
    """Raised when generated SQL is not a pure SELECT."""


class Engine:
    """
    Phase 4 orchestrator:
    (optional) rewrite follow-up via session -> retrieve context -> generate -> validate ->
    execute -> score -> append turn to session -> return.
    """

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore | None = None,
        session_store: SessionStore | None = None,
    ):
        self.settings = settings
        self.generator = SQLGenerator(settings)
        self.rewriter = QuestionRewriter(settings)
        self.resolver = ReferenceResolver()
        self.embedder = EmbeddingClient(settings.embedding_model)
        self.vector_store = vector_store or ChromaVectorStore(settings.chroma_path)
        self.retriever = HybridRetriever(self.vector_store, self.embedder)
        self.session_store = session_store or InMemorySessionStore()
        self.analyzer = InsightAnalyzer()
        self.charter = InsightCharter()
        self.narrator = InsightNarrator(settings)
        self.pattern_store = PatternStore()
        self.feedback = FeedbackRecorder(self.pattern_store, self.vector_store, self.embedder)
        self.watchdog = WatchdogManager(self, settings)

        self._connectors: dict[str, BaseConnector] = {}
        self._schemas: dict[str, DatabaseSchema] = {}
        self._semantic_models: list[SemanticModel] = SemanticModel.load_directory(
            settings.semantic_model_path
        )
        self._retriever_loaded = False

    def connector_for(self, database: str | None) -> BaseConnector:
        conn_str = self.settings.connection_for(database)
        if conn_str not in self._connectors:
            self._connectors[conn_str] = get_connector(conn_str)
        return self._connectors[conn_str]

    def schema_for(self, database: str | None) -> DatabaseSchema:
        connector = self.connector_for(database)
        if connector.connection_string not in self._schemas:
            self._schemas[connector.connection_string] = SchemaIntrospector(connector).introspect()
        return self._schemas[connector.connection_string]

    def invalidate_schema_cache(self, database: str | None = None) -> None:
        if database is None:
            self._schemas.clear()
        else:
            conn_str = self.settings.connection_for(database)
            self._schemas.pop(conn_str, None)

    def build_index(self, database: str | None = None) -> int:
        schema = self.schema_for(database)
        count = self.retriever.build_index(schema, self._semantic_models, self.pattern_store)
        self._retriever_loaded = True
        return count

    def _ensure_retriever_loaded(self, schema: DatabaseSchema) -> None:
        if self._retriever_loaded:
            return
        if self.vector_store.count() > 0:
            self.retriever.load_bm25_from_existing(schema, self._semantic_models, self.pattern_store)
        self._retriever_loaded = True

    def _build_context(self, question: str, schema: DatabaseSchema) -> tuple[str, list[RetrievedDoc]]:
        self._ensure_retriever_loaded(schema)
        hits = self.retriever.retrieve(question, k=10)
        if not hits:
            return schema.to_prompt_text(), []
        lines = [f"[{h.doc_type}] {h.text}" for h in hits]
        return "\n".join(lines), hits

    async def ask(
        self,
        question: str,
        database: str | None = None,
        session_id: str | None = None,
        with_insights: bool | None = None,
    ) -> QueryResult:
        """
        Execute a question. If session_id is provided, previous turns are used to resolve follow-ups.
        If with_insights is True (or settings.insight_enabled), runs the insight pipeline on the result.
        """
        session: Session | None = None
        if session_id is not None:
            session = self.session_store.get_or_create(
                session_id=session_id,
                database=database,
                ttl_minutes=self.settings.session_ttl_minutes,
            )

        resolved = self.resolver.resolve(question, session)
        standalone_question = question
        if session is not None and resolved.is_follow_up:
            try:
                standalone_question = await self.rewriter.rewrite(question, session)
            except Exception as e:  # noqa: BLE001 — any rewriter failure should not break the turn
                standalone_question = question
                _ = e  # Keep a trace-friendly binding; actual error handling via logs is out of scope here.

        result = await self._execute(standalone_question, database)

        enable_insights = with_insights if with_insights is not None else self.settings.insight_enabled
        if enable_insights and result.sql and result.results:
            await self._attach_insights(result, standalone_question)

        if session is not None:
            turn = ConversationTurn(
                turn_number=session.next_turn_number(),
                question=question,
                rewritten_question=standalone_question,
                sql=result.sql,
                results_summary=summarize_result(result.columns, result.results[:3]),
                columns=result.columns,
                row_count=result.row_count,
                sample_rows=result.results[:3],
            )
            session.add_turn(turn)
            self.session_store.save(session)
            # Surface session metadata via explanation field for MCP clients that don't yet
            # read it separately. We prepend, so any refusal reason remains visible.
            extra = f"[session={session.session_id} turn={turn.turn_number}]"
            if standalone_question != question:
                extra += f" rewritten=\"{standalone_question}\""
            result.explanation = extra if not result.explanation else f"{extra} {result.explanation}"

        return result

    async def follow_up(self, refinement: str, session_id: str) -> QueryResult:
        """Convenience wrapper: forces the rewriter path by requiring a session_id."""
        session = self.session_store.get(session_id)
        if session is None:
            return _refusal(
                "unknown",
                f"Session {session_id} not found or expired. Start a new session via `ask`.",
            )
        return await self.ask(refinement, database=session.database, session_id=session_id)

    def correct_query(
        self,
        original_question: str,
        wrong_sql: str,
        correct_sql: str,
        database: str | None = None,
    ) -> dict:
        """
        Record a user correction. The correct SQL becomes a proven pattern and is
        indexed into the vector store so future retrievals see it.
        """
        outcome = self.feedback.record_correction(
            question=original_question,
            correct_sql=correct_sql,
            database=database,
            wrong_sql=wrong_sql,
        )
        # Invalidate the retriever cache so the next ask() rebuilds BM25 with the new pattern.
        # (Vector search already sees the new pattern because the feedback recorder upserted it
        # into the vector store directly; BM25 is the in-memory piece that needs a refresh.)
        self._retriever_loaded = False
        return {
            "pattern_id": outcome.pattern_id,
            "indexed": outcome.indexed,
            "message": (
                "Correction recorded. Future questions similar to this one will see the corrected SQL."
                if outcome.indexed
                else "Correction recorded but indexing into the vector store failed; it will still be in the local store."
            ),
        }

    async def analyze(self, question: str, database: str | None = None) -> QueryResult:
        """
        Deeper analytical mode. Same as `ask` but always runs the insight pipeline,
        regardless of settings.insight_enabled.
        """
        return await self.ask(question, database=database, with_insights=True)

    async def _attach_insights(self, result: QueryResult, question: str) -> None:
        """Run the analyzer + charter + narrator on a successful result. Mutates `result` in place."""
        intent = classify_intent(question)
        analysis = self.analyzer.analyze(result.results, result.columns, intent)
        result.key_findings = list(analysis.key_findings)

        chart_spec, reason = self.charter.generate(result.results, result.columns, intent, analysis)
        if chart_spec is not None:
            result.chart = {
                "type": chart_spec.chart_type,
                "image_base64": chart_spec.image_base64,
                "title": chart_spec.title,
            }
        else:
            result.chart_skipped_reason = reason

        try:
            narrative = await self.narrator.narrate(question, analysis)
        except Exception as e:  # noqa: BLE001
            narrative = ""
            result.warnings.append(f"narrator_failed: {e}")
        result.insight = narrative or None

    def get_session(self, session_id: str) -> dict | None:
        session = self.session_store.get(session_id)
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "database": session.database,
            "created_at": session.created_at.isoformat(),
            "last_active": session.last_active.isoformat(),
            "turn_count": len(session.turns),
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "question": t.question,
                    "rewritten_question": t.rewritten_question,
                    "sql": t.sql,
                    "row_count": t.row_count,
                    "columns": t.columns,
                }
                for t in session.turns
            ],
        }

    async def _execute(self, question: str, database: str | None) -> QueryResult:
        """Core validation + execution pipeline (no conversation concerns)."""
        connector = self.connector_for(database)
        schema = self.schema_for(database)
        context, hits = self._build_context(question, schema)

        try:
            sql = await self.generator.generate(question, context=context, dialect=schema.dialect)
        except GenerationRefusal as e:
            return _refusal(schema.dialect, f"Cannot answer: {e}")

        schema_validator = SchemaValidator(schema)
        schema_result = schema_validator.validate(sql)
        exec_validator = ExecutionValidator(connector, timeout_seconds=self.settings.query_timeout)
        exec_result: ExecutionResult | None = None

        if schema_result.valid:
            exec_result = exec_validator.validate(sql)

        if not schema_result.valid or (exec_result is not None and not exec_result.ok):
            error_msg = (
                schema_result.error_message()
                if not schema_result.valid
                else (exec_result.error if exec_result else "unknown error")
            )
            try:
                sql = await self.generator.generate_retry(
                    question=question,
                    context=context,
                    dialect=schema.dialect,
                    previous_sql=sql,
                    error_message=error_msg,
                )
            except GenerationRefusal as e:
                return _refusal(schema.dialect, f"Cannot answer after retry: {e}")
            schema_result = schema_validator.validate(sql)
            if schema_result.valid:
                exec_result = exec_validator.validate(sql)

        self._assert_select_only(sql, dialect=schema.dialect)

        if not schema_result.valid:
            return _refusal(
                schema.dialect,
                f"Schema validation failed: {schema_result.error_message()}",
                sql=sql,
            )
        if exec_result is None or not exec_result.ok:
            return _refusal(
                schema.dialect,
                f"Execution validation failed: {exec_result.error if exec_result else 'unknown'}",
                sql=sql,
            )

        columns, rows = connector.execute(
            sql,
            read_only=True,
            timeout_seconds=self.settings.query_timeout,
        )

        intent = classify_intent(question)
        shape = validate_shape(intent, columns, len(rows))
        retrieval_similarity, semantic_coverage = _score_retrieval(hits)

        dual_path_signal: float | None = None
        dual_path_note: str | None = None
        if self.settings.dual_path_enabled:
            dual_path_signal, dual_path_note = await self._run_dual_path(
                question=question,
                context=context,
                dialect=schema.dialect,
                connector=connector,
                primary_columns=columns,
                primary_rows=rows,
            )

        confidence = calculate_confidence(
            schema_result=schema_result,
            execution_result=exec_result,
            shape_result=shape,
            retrieval_similarity=retrieval_similarity,
            semantic_coverage=semantic_coverage,
            dual_path_agreement=dual_path_signal,
            threshold=self.settings.confidence_threshold,
        )

        if confidence.refused:
            return _refusal(schema.dialect, confidence.explanation, sql=sql, confidence=confidence.value)

        warnings = list(confidence.warnings)
        if dual_path_note:
            warnings.append(dual_path_note)

        return QueryResult(
            sql=sql,
            results=rows,
            row_count=len(rows),
            columns=columns,
            dialect=schema.dialect,
            confidence=confidence.value,
            warnings=warnings,
        )

    async def _run_dual_path(
        self,
        *,
        question: str,
        context: str,
        dialect: str,
        connector: BaseConnector,
        primary_columns: list[str],
        primary_rows: list[dict],
    ) -> tuple[float | None, str | None]:
        """
        Generate SQL via Path B (decompose-then-compose), execute it, compare to Path A's result.
        Returns (agreement_score in [0,1], warning_note). Any failure yields (None, warning).
        """
        try:
            path_b_sql = await self.generator.generate_decomposed(
                question=question, context=context, dialect=dialect
            )
        except GenerationRefusal:
            return (0.5, "dual_path: path B refused (path A accepted; partial signal only)")

        # Safety + schema validation on path B.
        try:
            self._assert_select_only(path_b_sql, dialect=dialect)
        except UnsafeSQLError as e:
            return (0.0, f"dual_path: path B produced unsafe SQL: {e}")

        # Execute Path B with the same read-only + timeout constraints as the main path.
        try:
            b_cols, b_rows = connector.execute(
                path_b_sql, read_only=True, timeout_seconds=self.settings.query_timeout
            )
        except Exception as e:  # noqa: BLE001
            return (0.3, f"dual_path: path B execution failed: {e}")

        comparison: DualPathResult = compare_results(
            path_a_columns=primary_columns,
            path_a_rows=primary_rows,
            path_b_columns=b_cols,
            path_b_rows=b_rows,
        )

        note: str | None = None
        if comparison.agreement_level == "full":
            note = "dual_path: agreement (path A and path B produced matching results)"
        elif comparison.agreement_level == "partial":
            note = f"dual_path: partial agreement — {comparison.divergence_note}"
        elif comparison.agreement_level == "disagreement":
            note = f"dual_path: DISAGREEMENT — {comparison.divergence_note}. Path B SQL: {path_b_sql}"
        return (comparison.agreement_score, note)

    def validate_sql(self, sql: str, database: str | None = None) -> dict:
        connector = self.connector_for(database)
        schema = self.schema_for(database)

        try:
            self._assert_select_only(sql, dialect=schema.dialect)
        except UnsafeSQLError as e:
            return {
                "valid": False,
                "issues": [{"kind": "unsafe", "identifier": str(e)}],
                "tables_referenced": [],
                "columns_referenced": [],
                "sample_rows": [],
            }

        schema_result = SchemaValidator(schema).validate(sql)
        exec_validator = ExecutionValidator(connector, timeout_seconds=self.settings.query_timeout)
        exec_result = exec_validator.validate(sql) if schema_result.valid else None

        return {
            "valid": schema_result.valid and (exec_result is None or exec_result.ok),
            "issues": [
                {"kind": i.kind, "identifier": i.identifier, "suggestion": i.suggestion}
                for i in schema_result.issues
            ]
            + (
                [{"kind": "execution_error", "identifier": exec_result.error}]
                if exec_result and not exec_result.ok
                else []
            ),
            "tables_referenced": schema_result.tables_referenced,
            "columns_referenced": [
                {"table": t, "column": c} for t, c in schema_result.columns_referenced
            ],
            "sample_rows": exec_result.rows if exec_result and exec_result.ok else [],
        }

    def list_databases(self) -> list[dict]:
        entries = [
            {"id": cfg.id, "connection": _redact(cfg.connection), "dialect": cfg.dialect}
            for cfg in self.settings.databases.values()
        ]
        if not entries:
            entries.append(
                {"id": "default", "connection": _redact(self.settings.default_db), "dialect": None}
            )
        return entries

    def describe_database(self, database: str | None = None) -> dict:
        schema = self.schema_for(database)
        return {
            "dialect": schema.dialect,
            "tables": [
                {
                    "name": t.name,
                    "row_count": t.row_count,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.data_type,
                            "nullable": c.is_nullable,
                            "primary_key": c.is_primary_key,
                            "foreign_key": c.foreign_key_references,
                        }
                        for c in t.columns
                    ],
                }
                for t in schema.tables
            ],
            "foreign_keys": [fk.model_dump() for fk in schema.foreign_keys],
        }

    @staticmethod
    def _assert_select_only(sql: str, dialect: str) -> None:
        try:
            parsed = sqlglot.parse(sql, read=dialect)
        except sqlglot.errors.ParseError as e:
            raise UnsafeSQLError(f"Could not parse generated SQL: {e}") from e

        for statement in parsed:
            if statement is None:
                continue
            if isinstance(statement, MUTATING_EXPRESSIONS):
                raise UnsafeSQLError(f"Refusing to execute non-SELECT statement: {statement.key}")
            for node in statement.walk():
                if isinstance(node, MUTATING_EXPRESSIONS):
                    raise UnsafeSQLError(f"Refusing to execute nested mutating statement: {node.key}")


def _refusal(dialect: str, explanation: str, sql: str = "", confidence: int = 0) -> QueryResult:
    return QueryResult(
        sql=sql,
        results=[],
        row_count=0,
        columns=[],
        dialect=dialect,
        confidence=confidence,
        explanation=explanation,
        warnings=[explanation],
    )


def _score_retrieval(hits: list[RetrievedDoc]) -> tuple[float, float]:
    if not hits:
        return 0.3, 0.0
    top = hits[:3]
    similarity = sum(h.score for h in top) / max(len(top), 1) if hasattr(top[0], "score") else 0.5
    similarity = min(max(similarity, 0.3), 1.0)
    semantic_types = {"metric", "example", "insight_hint", "join"}
    coverage = sum(1 for h in hits if h.doc_type in semantic_types) / len(hits)
    return similarity, coverage


def _redact(connection_string: str) -> str:
    try:
        from sqlalchemy.engine import make_url

        url = make_url(connection_string)
        return str(url.set(password="***")) if url.password else str(url)
    except Exception:
        return connection_string
