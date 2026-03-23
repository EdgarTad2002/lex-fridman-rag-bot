"""
chat.py — Interactive CLI for the Lex Fridman RAG bot.

Usage:
    python chat.py

Type your question and press Enter. Type 'quit' or 'exit' to stop.
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.rule import Rule
from rich.prompt import Prompt
from rich import box
import sys

import rag

console = Console()


def print_banner(meta: dict):
    console.print()
    console.print(Panel(
        f"[bold cyan]🎙️  Lex Fridman Podcast RAG Bot[/]\n\n"
        f"[white]Episode:[/]  [bold yellow]{meta['video_url']}[/]\n"
        f"[white]Chunks:[/]   [bold]{meta['num_chunks']}[/] indexed transcript segments\n"
        f"[white]Model:[/]    [bold]gemini-2.0-flash[/]  |  Embeddings: [bold]{meta['embed_model']}[/]\n\n"
        f"[dim]Ask anything about the podcast. Type [bold]quit[/bold] to exit.[/dim]",
        title="[bold]Welcome[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def print_sources(sources: list[dict]):
    table = Table(
        title="📎 Source Segments",
        box=box.ROUNDED,
        border_style="dim",
        show_lines=True,
        title_style="bold dim",
    )
    table.add_column("#",         style="dim",    width=3,  no_wrap=True)
    table.add_column("Timestamp", style="yellow", width=10, no_wrap=True)
    table.add_column("Excerpt",   style="white",  ratio=1)
    table.add_column("Link",      style="cyan",   width=12, no_wrap=True)

    for i, src in enumerate(sources, 1):
        snippet = src["text"][:160].replace("\n", " ") + "…"
        table.add_row(
            str(i),
            src["timestamp"],
            snippet,
            f"[link={src['url']}]▶ Open[/link]",
        )
    console.print(table)
    console.print()


def run():
    console.print("[dim]Loading index…[/dim]")
    try:
        meta = rag.get_meta()
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)
    except EnvironmentError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        sys.exit(1)

    print_banner(meta)

    while True:
        try:
            query = Prompt.ask("[bold cyan]You[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q", "bye"}:
            console.print("[dim]Goodbye![/dim]")
            break

        console.print()
        with console.status("[cyan]Thinking…[/cyan]", spinner="dots"):
            try:
                result = rag.answer(query, k=5)
            except EnvironmentError as e:
                console.print(f"[bold red]Error:[/] {e}")
                continue
            except Exception as e:
                console.print(f"[bold red]Unexpected error:[/] {e}")
                continue

        # ── Answer ──
        console.print(Rule("[bold green]Answer[/bold green]", style="green"))
        console.print(Markdown(result["answer"]))
        console.print()

        # ── Sources ──
        print_sources(result["sources"])
        console.print(Rule(style="dim"))
        console.print()


if __name__ == "__main__":
    run()
