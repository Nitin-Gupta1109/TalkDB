# TalkDB

An open-source, MCP-native autonomous data analyst. Converts natural-language questions into validated SQL, then goes further: multi-turn conversations, statistical insight generation, proactive metric monitoring, dual-path result verification, and a learning loop from user corrections.

Not just text-to-SQL — a full analyst loop in a single tool.

## What it does

- **Ask questions in plain English** over Postgres, SQLite, and more. Answers include SQL, results, a narrative summary, and an auto-generated chart.
- **Multi-turn conversations.** "Revenue by month" → "just Q4" → "break that down by region" — follow-ups resolve to the previous turn's context.
- **Semantic layer (YAML).** Define business metrics once (`revenue = SUM(orders.total_amount) WHERE status='completed'`); the LLM uses your definitions instead of guessing.
- **Dual-path verification.** Every novel query is generated two structurally different ways; results are compared. Divergence drops confidence and surfaces a warning — catching semantic errors that schema validation can't.
- **Confidence scoring with graceful refusal.** Queries below threshold aren't silently wrong — they return a refusal explaining what's uncertain.
- **Statistical insight agent.** After results land, a pandas-based analyzer detects trends, anomalies, and concentrations. A chart is auto-generated from data shape. An LLM narrator writes 2–4 sentences using only the analyzer's facts (no hallucinated numbers).
- **Proactive watchdog.** Save any query as a scheduled watch. APScheduler runs it on your cadence, compares to a rolling baseline, and fires a webhook/Slack/stdout alert when conditions trigger.
- **Self-improving via corrections.** `correct_query(question, wrong_sql, correct_sql)` stores the pattern and indexes it into retrieval, so future similar questions benefit.
- **Community registry.** `talkdb registry install stripe-semantic` drops in a full semantic model for Stripe's schema — metrics, join rules, and proven query patterns — so you don't start from zero on common SaaS databases.

## Interfaces

- **MCP server (primary).** Works with Claude Desktop, Cursor, VS Code, and any MCP-compatible client. 16 tools: `ask`, `analyze`, `follow_up`, `list_databases`, `describe_database`, `validate_sql`, `correct_query`, `watch`, `list_watches`, `remove_watch`, `run_watch`, `get_session`, `install_semantic_package`, `uninstall_semantic_package`, `list_installed_packages`, `search_registry`.
- **REST API.** FastAPI wrapper with 17 endpoints mirroring every MCP tool — for web clients, dashboards, and environments where MCP isn't available. `talkdb api --port 8000`.
- **CLI.** `talkdb ask`, `talkdb chat`, `talkdb init`, `talkdb index`, `talkdb watchdog add/list/remove/run/start`, `talkdb registry install/uninstall/list/search`, `talkdb serve`, `talkdb api`.

## Tech stack

Python 3.11+ · FastMCP · FastAPI · LiteLLM (Claude / GPT / Gemini / Ollama) · SQLAlchemy 2 · sqlglot · ChromaDB + BM25 hybrid retrieval · pandas + matplotlib + seaborn · APScheduler · Pydantic.

## Quick start

```bash
# Install from PyPI
pip install talkdb-ai

# ...or from source
git clone https://github.com/Nitin-Gupta1109/TalkDB.git
cd TalkDB
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env: set OPENAI_API_KEY or ANTHROPIC_API_KEY, and TALKDB_DEFAULT_DB

# Seed a demo ecommerce DB (optional — or point at your own)
python scripts/seed_example_db.py

# Index schema + semantic model for retrieval
talkdb index

# Ask a question
talkdb ask "What is our total revenue?"

# Multi-turn chat
talkdb chat

# Start MCP server (for Claude Desktop / Cursor)
talkdb serve --transport stdio
```

## Wiring into Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "talkdb": {
      "command": "/absolute/path/to/.venv/bin/talkdb",
      "args": ["serve", "--transport", "stdio"],
      "cwd": "/absolute/path/to/project"
    }
  }
}
```

Restart Claude Desktop. The `ask`, `analyze`, `follow_up`, etc. tools appear in the tool picker.

## Semantic model

Define business meaning once in `semantic_models/<db>.yaml`:

```yaml
metrics:
  - name: revenue
    description: "Total revenue from completed orders."
    calculation: "SUM(orders.total_amount) WHERE orders.status = 'completed'"
    table: orders

tables:
  - name: customers
    columns:
      - name: tier
        valid_values: ["bronze", "silver", "gold", "platinum"]
        aliases: ["loyalty level", "membership tier"]

joins:
  - left: orders
    right: customers
    on: "orders.customer_id = customers.id"
    type: "INNER JOIN"
```

Run `talkdb init --database mydb --output semantic_models/mydb.yaml` to auto-generate a skeleton from your DB.

## Watchdog example

```bash
talkdb watchdog add \
  --name "Revenue monitor" \
  --question "What is today's total revenue?" \
  --schedule "every 1 hour" \
  --alert "drops more than 20% below 7-day average" \
  --slack-webhook "https://hooks.slack.com/services/..."

talkdb watchdog list
talkdb watchdog start   # blocks; runs the scheduler
```

Alerts render like:
> 🔴 Revenue monitor — Current value: $38.2k — 27% below baseline $52.4k (7_day_avg). Suggested follow-up: "Why is today's revenue below baseline?"

## Community registry

Install community-maintained semantic packages:

```bash
talkdb registry install stripe-semantic     # from the registry (when published)
talkdb registry install ./packages/stripe-semantic   # from a local directory
talkdb registry list
talkdb registry search "stripe"
```

Packages are YAML-only — metric definitions, table/column docs, join rules, and proven query patterns. No executable code, ever (security by design). Once installed, the retriever surfaces their definitions automatically when a question matches.

See [packages/stripe-semantic/](packages/stripe-semantic/) for the reference package (5 metrics, 5 tables, 4 joins, 6 proven queries covering MRR, active subscriptions, net revenue, customer LTV).

## Benchmark

Ships with a 47-case benchmark on the seeded DB for regression tracking:

```bash
python -m tests.benchmarks.run_benchmark
```

Current baseline: **37/47 (78%) execution accuracy, 42/47 (89%) lenient (containment match), 0 silent-wrong answers**. `gpt-4o-mini` and `gpt-4o` tied at 78% on this suite — model upgrade alone didn't move the needle. Per-phase regression baselines are checked into `tests/benchmarks/`.

## Project layout

```
src/talkdb/
├── core/          # Engine, SQL generator, intent classifier, dialect transpiler
├── conversation/  # Session state, rewriter, reference resolver
├── schema/        # Introspector, data models, semantic model loader
├── retrieval/     # ChromaDB + BM25 hybrid retriever, embeddings
├── validation/    # Schema, execution, shape, dual-path, confidence
├── insight/       # Analyzer (pandas), charter (matplotlib), narrator (LLM)
├── watchdog/      # Scheduler, baseline, alerter, storage
├── learning/      # Pattern store, feedback recorder
├── registry/      # Community package loader + local index + install client
├── connectors/    # Postgres, SQLite (more dialects via sqlglot)
├── server/        # FastMCP server + FastAPI REST wrapper
└── config/        # Pydantic settings
```

## Design principles

- **Never dump full schema into prompts.** Hybrid retrieval (BM25 + vector) surfaces only relevant context.
- **Never return results below confidence threshold.** Refuse and explain — silent wrong answers destroy trust.
- **SELECT only.** sqlglot AST walk rejects every mutating statement before execution.
- **Read-only validation.** All validation queries run in `READ ONLY` transactions with `LIMIT 10` and a 10-second timeout.
- **LLM-provider agnostic.** Every LLM call goes through LiteLLM — swap Claude ↔ GPT ↔ Gemini ↔ Ollama with one config change.
- **Vector store abstraction.** ChromaDB for dev, pgvector can plug in behind the same interface.
- **Temperature 0 for SQL generation.** Deterministic output.
- **Insight stats are pandas, not LLM.** Only narration uses the LLM (so the numbers in insights are always correct).
- **Dual-path uses structurally different prompts.** Path A direct, Path B decompose-then-compose. Catches correlated errors that self-correction misses.
- **Conversation rewriting, not SQL mutation.** Follow-ups are rewritten into standalone questions before SQL is regenerated from scratch.

## License

MIT.
