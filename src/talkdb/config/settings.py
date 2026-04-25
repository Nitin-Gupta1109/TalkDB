from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    id: str
    connection: str
    dialect: str | None = None
    semantic_model: str | None = None


class Settings(BaseSettings):
    """
    Global settings. Loaded from environment variables (prefix TALKDB_) and/or a .env file.
    A talkdb.yaml file can override env vars — see load_yaml_overrides() below.
    """

    model_config = SettingsConfigDict(
        env_prefix="TALKDB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_db: str = Field(default="sqlite:///./data/example.db")

    llm_model: str = Field(default="claude-sonnet-4-6")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=2000)

    embedding_model: str = Field(default="text-embedding-3-small")

    vector_store: str = Field(default="chroma")
    chroma_path: str = Field(default="./data/chroma")

    confidence_threshold: int = Field(default=50)
    query_timeout: int = Field(default=10)
    dual_path_enabled: bool = Field(default=False)

    # --- Experimental benchmark knobs (default off so baseline behavior is preserved) ---

    # Schema-linker pre-pass: before retrieval, ask the LLM which tables are relevant
    # and filter the retrieval context to those tables only. Reduces hallucination on
    # large schemas. Adds one cheap LLM call per question.
    schema_linking_enabled: bool = Field(default=False)

    # Context-grounded rewriter: after each successful turn, store a short
    # "currently referencing" summary in session state so follow-ups can resolve
    # pronouns ("him", "Balmoor") against concrete values, not just prior questions.
    context_grounded_rewriter: bool = Field(default=False)

    # Auto-approve high-signal successful queries into the pattern store so future
    # similar questions retrieve them as proven examples. Gated on confidence +
    # warnings + dual-path agreement + nonzero rows to keep bad queries out.
    auto_approve_enabled: bool = Field(default=False)
    auto_approve_confidence: int = Field(default=80)

    insight_enabled: bool = Field(default=False)
    chart_format: str = Field(default="png")

    watchdog_enabled: bool = Field(default=False)
    watchdog_db: str = Field(default="./data/watchdog.sqlite")
    slack_webhook: str = Field(default="")
    alert_email: str = Field(default="")

    semantic_model_path: str = Field(default="./semantic_models/")

    registry_url: str = Field(default="https://registry.talkdb.dev")
    registry_packages_path: str = Field(default="./data/packages/")

    session_ttl_minutes: int = Field(default=60)
    session_store: str = Field(default="memory")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # In-memory registry of named database connections. Populated from talkdb.yaml or CLI.
    databases: dict[str, DatabaseConfig] = Field(default_factory=dict)

    def connection_for(self, database: str | None) -> str:
        """Resolve a database identifier to a connection string. Falls back to default_db."""
        if database and database in self.databases:
            return self.databases[database].connection
        return self.default_db


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call get_settings.cache_clear() to reload."""
    settings = Settings()
    yaml_path = Path("talkdb.yaml")
    if yaml_path.exists():
        _apply_yaml(settings, yaml_path)
    return settings


def _apply_yaml(settings: Settings, path: Path) -> None:
    """Apply overrides from talkdb.yaml onto an existing Settings instance."""
    import yaml

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    if "llm" in data:
        llm = data["llm"]
        if "model" in llm:
            settings.llm_model = llm["model"]
        if "temperature" in llm:
            settings.llm_temperature = llm["temperature"]
        if "max_tokens" in llm:
            settings.llm_max_tokens = llm["max_tokens"]

    if "validation" in data:
        v = data["validation"]
        if "confidence_threshold" in v:
            settings.confidence_threshold = v["confidence_threshold"]
        if "query_timeout_seconds" in v:
            settings.query_timeout = v["query_timeout_seconds"]
        if "dual_path_enabled" in v:
            settings.dual_path_enabled = v["dual_path_enabled"]

    if "databases" in data:
        for entry in data["databases"]:
            cfg = DatabaseConfig(**entry)
            settings.databases[cfg.id] = cfg
