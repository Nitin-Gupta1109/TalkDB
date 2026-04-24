"""Generate the 4 example notebooks under examples/.

One-shot generator. Run once; commit the executed outputs. Not part of the package.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def write(nb: nbf.NotebookNode, path: Path) -> None:
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    path.write_text(nbf.writes(nb))


SETUP_CELL = (
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "# Locate project root (so relative paths data/, packages/, .env resolve regardless of\n"
    "# where the notebook was launched from).\n"
    "PROJECT_ROOT = Path.cwd()\n"
    "while not (PROJECT_ROOT / 'pyproject.toml').exists():\n"
    "    if PROJECT_ROOT.parent == PROJECT_ROOT:\n"
    "        raise RuntimeError('could not locate project root (no pyproject.toml found upward)')\n"
    "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
    "os.chdir(PROJECT_ROOT)\n"
    "\n"
    "from dotenv import load_dotenv\n"
    "load_dotenv()\n"
    "if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):\n"
    "    raise RuntimeError('Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env before running this notebook.')\n"
    "print('project root:', PROJECT_ROOT)"
)


EXAMPLES = Path("examples")
EXAMPLES.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 01_quickstart.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.cells = [
    md(
        "# TalkDB quickstart\n\n"
        "Five-minute walkthrough: seed a demo ecommerce DB, build the hybrid retrieval index, "
        "and ask two questions in plain English. You get back validated SQL, the rows, and a confidence score.\n\n"
        "**Prereq:** `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your `.env`. "
        "See [.env.example](../.env.example)."
    ),
    code(SETUP_CELL),
    md("## Seed the demo database\n\nCreates `data/example.db` with 200 customers, 50 products, ~800 orders."),
    code(
        "import subprocess, sys\n"
        "\n"
        "subprocess.run([sys.executable, 'scripts/seed_example_db.py'], check=True)"
    ),
    md("## Build the engine + index"),
    code(
        "from talkdb.config.settings import get_settings\n"
        "from talkdb.core.engine import Engine\n"
        "\n"
        "engine = Engine(get_settings())\n"
        "doc_count = engine.build_index()\n"
        "print(f'Indexed {doc_count} documents into the hybrid retriever.')"
    ),
    md("## Ask a question"),
    code(
        "result = await engine.ask('How many customers are there?')\n"
        "print('SQL:       ', result.sql)\n"
        "print('Rows:      ', result.results)\n"
        "print('Confidence:', result.confidence)"
    ),
    md("## A harder aggregation — joining three tables\n\nRevenue by product category, completed orders only."),
    code(
        "result = await engine.ask(\n"
        "    'Which product category has the highest total revenue from completed orders?'\n"
        ")\n"
        "print('SQL:')\n"
        "print(result.sql)\n"
        "print()\n"
        "print('Top rows:')\n"
        "for row in result.results[:5]:\n"
        "    print(' ', row)\n"
        "print()\n"
        "print('Confidence:', result.confidence)"
    ),
]
write(nb, EXAMPLES / "01_quickstart.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# 02_conversation.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.cells = [
    md(
        "# Multi-turn conversations\n\n"
        "Pass a `session_id` to stack follow-ups. TalkDB rewrites each follow-up into a standalone question "
        "before running it through the normal pipeline — you can inspect the rewritten form in the session log."
    ),
    code(SETUP_CELL),
    code(
        "from talkdb.config.settings import get_settings\n"
        "from talkdb.core.engine import Engine\n"
        "\n"
        "engine = Engine(get_settings())\n"
        "engine.build_index()\n"
        "print('engine ready')"
    ),
    md("## Turn 1 — initial question"),
    code(
        "import uuid\n"
        "\n"
        "session_id = f'demo_{uuid.uuid4().hex[:8]}'\n"
        "\n"
        "r1 = await engine.ask('What is total revenue by month?', session_id=session_id)\n"
        "print('SQL:')\n"
        "print(r1.sql)\n"
        "print(f'\\n{r1.row_count} rows, confidence {r1.confidence}')"
    ),
    md("## Turn 2 — follow-up (`just Q4`)\n\n'Q4' has no meaning on its own — it makes sense only relative to turn 1."),
    code(
        "r2 = await engine.follow_up('just Q4', session_id=session_id)\n"
        "print('SQL:')\n"
        "print(r2.sql)\n"
        "print(f'\\n{r2.row_count} rows')"
    ),
    md("## Turn 3 — `break that down by payment method`"),
    code(
        "r3 = await engine.follow_up('break that down by payment method', session_id=session_id)\n"
        "print('SQL:')\n"
        "print(r3.sql)\n"
        "print()\n"
        "for row in r3.results[:8]:\n"
        "    print(' ', row)"
    ),
    md("## Inspect the session — see what each turn was rewritten to"),
    code(
        "state = engine.get_session(session_id)\n"
        "for turn in state['turns']:\n"
        "    print(f\"Turn {turn['turn_number']}: {turn['question']}\")\n"
        "    if turn['question'] != turn['rewritten_question']:\n"
        "        print(f\"  ↳ rewritten: {turn['rewritten_question']}\")\n"
        "    print(f\"  SQL: {turn['sql'][:100]}{'...' if len(turn['sql']) > 100 else ''}\")\n"
        "    print()"
    ),
]
write(nb, EXAMPLES / "02_conversation.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# 03_insight_agent.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.cells = [
    md(
        "# Insight agent\n\n"
        "After SQL returns rows, TalkDB's insight pipeline runs. "
        "Stats are pure pandas (trends, anomalies via Median Absolute Deviation, concentrations). "
        "A chart is auto-picked from data shape. A short LLM narrator writes 2–4 sentences **using only the "
        "analyzer's computed facts** — so the numbers in the narrative are always correct."
    ),
    code(SETUP_CELL),
    code(
        "from talkdb.config.settings import get_settings\n"
        "from talkdb.core.engine import Engine\n"
        "\n"
        "engine = Engine(get_settings())\n"
        "engine.build_index()\n"
        "print('engine ready')"
    ),
    md("## Ask a time-series question with insights on"),
    code(
        "result = await engine.ask(\n"
        "    'What is total revenue by month?',\n"
        "    with_insights=True,\n"
        ")\n"
        "print('SQL:')\n"
        "print(result.sql)\n"
        "print(f'\\n{result.row_count} rows, confidence {result.confidence}')"
    ),
    md("## Narrative"),
    code("print(result.insight or '(no narrative produced)')"),
    md("## Key findings detected by the pandas analyzer"),
    code(
        "if result.key_findings:\n"
        "    for f in result.key_findings:\n"
        "        print(f' • {f}')\n"
        "else:\n"
        "    print('(none)')"
    ),
    md("## Auto-generated chart"),
    code(
        "import base64\n"
        "from IPython.display import Image, display\n"
        "\n"
        "if result.chart and result.chart.get('image_base64'):\n"
        "    print(f\"chart type: {result.chart.get('type')}\")\n"
        "    print(f\"title:      {result.chart.get('title')}\")\n"
        "    display(Image(data=base64.b64decode(result.chart['image_base64'])))\n"
        "else:\n"
        "    print(f\"(no chart — reason: {result.chart_skipped_reason or 'unknown'})\")"
    ),
]
write(nb, EXAMPLES / "03_insight_agent.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# 04_community_registry.ipynb
# ─────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.cells = [
    md(
        "# Community registry\n\n"
        "Install a semantic-model package from a local directory (or later, from a tarball / URL / "
        "hosted registry name). Once installed, the retriever surfaces its definitions alongside your "
        "own YAML semantic models.\n\n"
        "Here we install the bundled `stripe-semantic` package and verify its metrics and examples are "
        "loaded. We don't have a Stripe schema wired up, so we don't ask Stripe-specific questions end-to-end — "
        "but the package format and retrieval integration are the interesting parts."
    ),
    code(SETUP_CELL),
    code(
        "from talkdb.config.settings import get_settings\n"
        "from talkdb.core.engine import Engine\n"
        "\n"
        "engine = Engine(get_settings())\n"
        "print('engine ready')"
    ),
    md("## Install the bundled `stripe-semantic` package\n\nIdempotent — reinstalls on top of any existing copy."),
    code(
        "result = engine.install_package('./packages/stripe-semantic')\n"
        "for k, v in result.items():\n"
        "    print(f'  {k}: {v}')"
    ),
    md("## List installed packages"),
    code(
        "for p in engine.list_installed_packages():\n"
        "    print(f\"  {p['name']}@{p['version']} — {p['example_count']} examples, schema_type={p['schema_type']}\")"
    ),
    md("## Search the registry"),
    code(
        "for hit in engine.search_registry('stripe'):\n"
        "    print(f\"  [{'✓' if hit['verified'] else ' '}] {hit['name']}@{hit['version']} ({hit['schema_type']})\")\n"
        "    if hit.get('description'):\n"
        "        print(f\"      {hit['description']}\")"
    ),
    md(
        "## Inspect what the package actually contains\n\n"
        "Every package is pure YAML — no executable code, by design. Let's load it and look at its metric definitions and example queries."
    ),
    code(
        "from talkdb.registry.package import SemanticPackage\n"
        "\n"
        "pkg = SemanticPackage.load('./packages/stripe-semantic')\n"
        "print(f'{pkg.manifest.name}@{pkg.manifest.version} — {pkg.manifest.description}')\n"
        "print(f'{len(pkg.semantic_model.metrics)} metrics, {len(pkg.semantic_model.tables)} tables, {len(pkg.all_examples)} examples')\n"
        "print()\n"
        "print('Metrics:')\n"
        "for m in pkg.semantic_model.metrics:\n"
        "    print(f'  • {m.name} — {m.description}')\n"
        "    print(f'      {m.calculation}')"
    ),
    md("## Example queries shipped with the package"),
    code(
        "for ex in pkg.all_examples[:4]:\n"
        "    print(f\"Q: {ex.question}\")\n"
        "    print(f\"A: {ex.sql}\")\n"
        "    print()"
    ),
    md("## Uninstall when done"),
    code(
        "result = engine.uninstall_package('stripe-semantic')\n"
        "print(result)"
    ),
]
write(nb, EXAMPLES / "04_community_registry.ipynb")

print("wrote 4 notebooks under examples/")
