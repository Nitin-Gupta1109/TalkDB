from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from talkdb.config.settings import get_settings
from talkdb.core.engine import Engine, UnsafeSQLError

console = Console()


@click.group()
def cli() -> None:
    """TalkDB — autonomous data analyst."""


@cli.command()
@click.option("--transport", type=click.Choice(["stdio", "sse"]), default="stdio")
@click.option("--host", default=None)
@click.option("--port", type=int, default=None)
def serve(transport: str, host: str | None, port: int | None) -> None:
    """Start the MCP server."""
    from talkdb.server.mcp_server import run_sse, run_stdio

    if transport == "stdio":
        run_stdio()
    else:
        settings = get_settings()
        run_sse(host=host or settings.host, port=port or settings.port)


@cli.command()
@click.argument("question")
@click.option("--database", default=None, help="Database id (uses default if omitted).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a formatted table.")
def ask(question: str, database: str | None, as_json: bool) -> None:
    """Ask a question and print the SQL + results."""
    engine = Engine(get_settings())
    try:
        result = asyncio.run(engine.ask(question, database=database))
    except UnsafeSQLError as e:
        console.print(f"[red]Refusing to execute generated SQL:[/red] {e}")
        raise SystemExit(2) from e

    if as_json:
        click.echo(json.dumps(result.model_dump(), indent=2, default=str))
        return

    console.print(f"[bold cyan]SQL[/bold cyan] ({result.dialect}):")
    console.print(result.sql or "[dim](no SQL generated)[/dim]")
    if result.explanation:
        console.print(f"[yellow]{result.explanation}[/yellow]")

    if not result.results:
        console.print(f"[dim]0 rows.[/dim] confidence={result.confidence}")
        return

    table = Table(show_header=True, header_style="bold magenta")
    for col in result.columns:
        table.add_column(col)
    for row in result.results[:50]:
        table.add_row(*[str(row.get(c, "")) for c in result.columns])
    console.print(table)
    extra = f" (showing first 50 of {result.row_count})" if result.row_count > 50 else ""
    console.print(f"[dim]{result.row_count} row(s){extra}. confidence={result.confidence}[/dim]")


@cli.command()
@click.option("--database", default=None)
def describe(database: str | None) -> None:
    """Describe a database's schema."""
    engine = Engine(get_settings())
    info = engine.describe_database(database)
    click.echo(json.dumps(info, indent=2, default=str))


@cli.command(name="init")
@click.option("--database", default=None, help="Database id (uses default if omitted).")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output path for the semantic model YAML.",
)
def init_cmd(database: str | None, output: Path) -> None:
    """Introspect the database and write a starter semantic model YAML."""
    engine = Engine(get_settings())
    schema = engine.schema_for(database)

    doc = {
        "version": "1.0",
        "database": database or "default",
        "tables": [
            {
                "name": t.name,
                "description": t.description or f"TODO: describe {t.name}",
                "columns": [
                    {
                        "name": c.name,
                        "description": c.description or f"TODO: describe {c.name}",
                        **({"valid_values": c.sample_values} if _looks_categorical(c) else {}),
                    }
                    for c in t.columns
                ],
            }
            for t in schema.tables
        ],
        "joins": [
            {
                "left": fk.from_table,
                "right": fk.to_table,
                "on": f"{fk.from_table}.{fk.from_columns[0]} = {fk.to_table}.{fk.to_columns[0]}",
                "type": "INNER JOIN",
            }
            for fk in schema.foreign_keys
        ],
        "metrics": [],
        "examples": [],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    console.print(f"[green]Wrote {output}[/green]")
    console.print(f"Tables: {len(doc['tables'])}  Joins: {len(doc['joins'])}")
    console.print("[dim]Next: fill in metric definitions and example questions, then run `talkdb index`.[/dim]")


@cli.command(name="index")
@click.option("--database", default=None)
def index_cmd(database: str | None) -> None:
    """Build the vector + BM25 retrieval index from the schema and semantic models."""
    engine = Engine(get_settings())
    count = engine.build_index(database)
    console.print(f"[green]Indexed {count} documents into {engine.settings.vector_store}[/green]")


@cli.command()
@click.option("--database", default=None, help="Database id (uses default if omitted).")
def chat(database: str | None) -> None:
    """Interactive multi-turn REPL. Type '/exit' to quit, '/history' to show turns."""
    import uuid

    engine = Engine(get_settings())
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    console.print(f"[dim]session: {session_id}[/dim]  [dim](Ctrl-D or /exit to quit)[/dim]\n")

    while True:
        try:
            question = click.prompt(">>>", prompt_suffix=" ", show_default=False).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not question:
            continue
        if question in ("/exit", "/quit", ":q"):
            break
        if question == "/history":
            state = engine.get_session(session_id) or {"turns": []}
            for t in state["turns"]:
                console.print(f"[cyan]{t['turn_number']}.[/cyan] {t['question']}")
                if t["question"] != t["rewritten_question"]:
                    console.print(f"   [dim]-> {t['rewritten_question']}[/dim]")
                console.print(f"   [yellow]{t['sql']}[/yellow]")
            continue

        try:
            result = asyncio.run(engine.ask(question, database=database, session_id=session_id))
        except UnsafeSQLError as e:
            console.print(f"[red]Refused:[/red] {e}")
            continue

        if result.sql:
            console.print(f"[yellow]{result.sql}[/yellow]")
        if result.explanation:
            console.print(f"[dim]{result.explanation}[/dim]")

        if result.results:
            table = Table(show_header=True, header_style="bold magenta")
            for col in result.columns:
                table.add_column(col)
            for row in result.results[:20]:
                table.add_row(*[str(row.get(c, "")) for c in result.columns])
            console.print(table)
            extra = f" (showing first 20 of {result.row_count})" if result.row_count > 20 else ""
            console.print(f"[dim]{result.row_count} row(s){extra}. confidence={result.confidence}[/dim]\n")
        else:
            console.print(f"[dim]0 rows. confidence={result.confidence}[/dim]\n")


def _looks_categorical(col) -> bool:
    return (
        col.data_type.upper().startswith(("TEXT", "VARCHAR", "CHAR"))
        and 0 < len(col.sample_values) <= 10
    )


@cli.group()
def watchdog() -> None:
    """Watchdog commands (add/list/remove scheduled watches; start the scheduler)."""


@watchdog.command("add")
@click.option("--name", required=True)
@click.option("--question", required=True, help="Natural language question to run on schedule.")
@click.option("--schedule", default="every 1 hour", help="'every N minute/hour/day', 'daily at HH:MM', or a 5-field cron expression.")
@click.option("--alert", "alert_condition", default="", help="Natural-language alert condition (e.g. 'drops more than 20% below 7-day average').")
@click.option("--database", default=None)
@click.option("--webhook", "webhook_url", default=None)
@click.option("--slack-webhook", "slack_webhook_url", default=None)
def watch_add(name, question, schedule, alert_condition, database, webhook_url, slack_webhook_url):
    """Add a new watch."""
    engine = Engine(get_settings())
    w = asyncio.run(
        engine.watchdog.add_watch(
            name=name,
            question=question,
            schedule=schedule,
            alert_condition=alert_condition,
            database=database,
            webhook_url=webhook_url,
            slack_webhook_url=slack_webhook_url,
        )
    )
    console.print(f"[green]Watch '{w.name}' created[/green]")
    console.print(f"  SQL: {w.sql}")
    console.print(f"  Schedule: {w.schedule}")
    console.print(f"  Condition: {w.alert_condition.description or w.alert_condition.kind}")


@watchdog.command("list")
def watch_list():
    """List all watches."""
    engine = Engine(get_settings())
    watches = engine.watchdog.list_watches()
    if not watches:
        console.print("[dim]No watches defined.[/dim]")
        return
    table = Table(show_header=True, header_style="bold magenta")
    for col in ("Name", "Schedule", "Last run", "Last value", "Status"):
        table.add_column(col)
    for w in watches:
        table.add_row(
            w.name,
            w.schedule,
            w.last_run.isoformat() if w.last_run else "-",
            f"{w.last_value:.2f}" if isinstance(w.last_value, int | float) else "-",
            w.last_status or "-",
        )
    console.print(table)


@watchdog.command("remove")
@click.argument("name")
def watch_remove(name):
    """Remove a watch by name."""
    engine = Engine(get_settings())
    removed = engine.watchdog.remove_watch(name)
    console.print(f"{'[green]Removed' if removed else '[yellow]Not found'}: {name}[/]")


@watchdog.command("run")
@click.argument("name")
def watch_run(name):
    """Manually run a watch now (useful for testing)."""
    engine = Engine(get_settings())
    run = asyncio.run(engine.watchdog.run_watch(name))
    console.print(f"value={run.value}  baseline={run.baseline}  status={run.status}")
    console.print(f"message: {run.message}")


@watchdog.command("start")
def watchdog_start():
    """Start the scheduler and keep it running (blocks until Ctrl-C)."""
    engine = Engine(get_settings())
    engine.watchdog.start()
    console.print("[green]Watchdog scheduler started. Ctrl-C to stop.[/green]")
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        engine.watchdog.shutdown()


if __name__ == "__main__":
    cli()
