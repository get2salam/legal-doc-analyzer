"""Command-line interface for Legal Document Analyzer.

Provides ``analyze``, ``extract``, and ``summarize`` commands with
rich terminal output using the ``click`` and ``rich`` libraries.

Usage::

    legal-doc-analyzer analyze contract.pdf
    legal-doc-analyzer extract --types indemnification,termination contract.pdf
    legal-doc-analyzer summarize contract.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analyzer import LegalAnalyzer
from .models import RiskLevel

console = Console()


def _get_risk_style(level: RiskLevel) -> str:
    """Return a rich style string for a risk level."""
    return {
        RiskLevel.HIGH: "bold red",
        RiskLevel.MEDIUM: "bold yellow",
        RiskLevel.LOW: "dim green",
        RiskLevel.INFO: "dim",
    }.get(level, "")


def _get_risk_icon(level: RiskLevel) -> str:
    """Return an emoji icon for a risk level."""
    return {
        RiskLevel.HIGH: "🔴",
        RiskLevel.MEDIUM: "🟡",
        RiskLevel.LOW: "🟢",
        RiskLevel.INFO: "ℹ️",
    }.get(level, "")


@click.group()
@click.version_option(package_name="legal-doc-analyzer")
def main() -> None:
    """📄 Legal Document Analyzer — AI-powered contract analysis.

    Analyze legal documents to extract clauses, detect risks, and
    generate summaries.
    """
    pass


@main.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich",
              help="Output format.")
@click.option("--save", "-s", type=click.Path(path_type=Path), default=None,
              help="Save results to a JSON file.")
def analyze(file: Path, output: str, save: Path | None) -> None:
    """Run full analysis on a legal document.

    Extracts clauses, entities, and risks, then displays a comprehensive
    summary.

    Example: legal-doc-analyzer analyze contract.pdf
    """
    analyzer = LegalAnalyzer()

    with console.status("[bold blue]Analyzing document...", spinner="dots"):
        try:
            result = analyzer.analyze(file)
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            sys.exit(1)

    if output == "json":
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _render_analysis(result)

    if save:
        save.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        console.print(f"\n[dim]Results saved to {save}[/]")


@main.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--types", "-t", default=None,
              help="Comma-separated clause types (e.g., indemnification,termination).")
@click.option("--output", "-o", type=click.Choice(["rich", "json"]), default="rich",
              help="Output format.")
def extract(file: Path, types: str | None, output: str) -> None:
    """Extract clauses from a legal document.

    Optionally filter by clause type.

    Example: legal-doc-analyzer extract --types termination,liability contract.pdf
    """
    analyzer = LegalAnalyzer()
    type_list = [t.strip() for t in types.split(",")] if types else None

    with console.status("[bold blue]Extracting clauses...", spinner="dots"):
        try:
            clauses = analyzer.extract_clauses(file, types=type_list)
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            sys.exit(1)

    if output == "json":
        click.echo(json.dumps([c.to_dict() for c in clauses], indent=2))
    else:
        _render_clauses(clauses, file.name)


@main.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["text", "bullet-points", "json"]),
              default="text", help="Summary format.")
def summarize(file: Path, fmt: str) -> None:
    """Generate a summary of a legal document.

    Example: legal-doc-analyzer summarize contract.pdf
    """
    analyzer = LegalAnalyzer()

    with console.status("[bold blue]Summarizing document...", spinner="dots"):
        try:
            result = analyzer.analyze(file)
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            sys.exit(1)

    if fmt == "json":
        click.echo(json.dumps({
            "filename": result.filename,
            "summary": result.summary,
            "risk_score": result.risk_score,
            "clause_count": len(result.clauses),
            "entity_count": len(result.entities),
        }, indent=2))
    elif fmt == "bullet-points":
        console.print(Panel(
            result.summary,
            title=f"📄 Summary: {result.filename}",
            border_style="blue",
        ))
    else:
        console.print()
        console.print(result.summary)
        console.print()


# ------------------------------------------------------------------
# Rich rendering helpers
# ------------------------------------------------------------------

def _render_analysis(result) -> None:
    """Render a full AnalysisResult with rich formatting."""
    from .models import EntityType

    console.print()

    # Header
    console.print(Panel(
        f"[bold]{result.filename}[/]\n"
        f"Pages: {result.metadata.get('page_count', '?')} | "
        f"Clauses: {len(result.clauses)} | "
        f"Entities: {len(result.entities)} | "
        f"Risks: {len(result.risks)}",
        title="📄 Legal Document Analysis",
        border_style="blue",
    ))

    # Summary
    console.print(Panel(result.summary, title="Summary", border_style="dim"))

    # Clauses table
    if result.clauses:
        _render_clauses(result.clauses, result.filename)

    # Entities table
    if result.entities:
        table = Table(title="Extracted Entities", show_lines=False)
        table.add_column("Type", style="cyan", width=15)
        table.add_column("Text", style="white")
        table.add_column("Page", justify="center", width=6)

        for entity in result.entities[:30]:  # Cap display at 30
            table.add_row(
                entity.type.value.replace("_", " ").title(),
                entity.text[:80],
                str(entity.page or "-"),
            )

        if len(result.entities) > 30:
            table.add_row("...", f"({len(result.entities) - 30} more)", "")

        console.print(table)
        console.print()

    # Risks
    if result.risks:
        console.print("[bold]Risk Assessment[/]")
        for risk in result.risks:
            icon = _get_risk_icon(risk.level)
            style = _get_risk_style(risk.level)
            console.print(f"  {icon} [{style}]{risk.level.value.upper()}[/]: {risk.description}")
            if risk.suggestion:
                console.print(f"      💡 {risk.suggestion}")
        console.print()

    # Risk score bar
    score = result.risk_score
    if score > 0.7:
        score_style = "bold red"
    elif score > 0.3:
        score_style = "bold yellow"
    else:
        score_style = "bold green"

    console.print(f"Overall Risk Score: [{score_style}]{score:.0%}[/]")
    console.print()


def _render_clauses(clauses: list, filename: str) -> None:
    """Render clauses as a rich table."""
    table = Table(title=f"Clauses — {filename}", show_lines=True)
    table.add_column("#", justify="right", width=4)
    table.add_column("Type", style="cyan", width=20)
    table.add_column("Text (excerpt)", style="white", max_width=60)
    table.add_column("Conf.", justify="center", width=6)
    table.add_column("Page", justify="center", width=6)
    table.add_column("Risk", justify="center", width=8)

    for i, clause in enumerate(clauses, 1):
        risk_style = _get_risk_style(clause.risk_level)
        risk_text = Text(clause.risk_level.value.upper(), style=risk_style)
        excerpt = clause.text[:120].replace("\n", " ") + ("..." if len(clause.text) > 120 else "")

        table.add_row(
            str(i),
            clause.type.value.replace("_", " ").title(),
            excerpt,
            f"{clause.confidence:.0%}",
            str(clause.page or "-"),
            risk_text,
        )

    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
