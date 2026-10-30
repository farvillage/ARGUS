"""Typer CLI entry points: `argus check` and `argus rules`."""

import importlib
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from argus import __version__
from argus.evm import (
    LiveRPCSimulator,
    OfflineFixtureSimulator,
    PayloadError,
    decode_calldata,
    load_payload,
)
from argus.formatters import render_json_report, render_rich_report, render_sarif_report
from argus.models import LintReport, Severity
from argus.rules import all_rules, compute_risk_score, run_rules

app = typer.Typer(
    name="argus",
    help="Pre-sign security linter for unsigned EVM transactions.",
    no_args_is_help=True,
    add_completion=False,
)

_err = Console(stderr=True)

_SEVERITY_STYLE = {
    Severity.LOW: "cyan",
    Severity.MEDIUM: "yellow",
    Severity.HIGH: "red",
    Severity.CRITICAL: "bold white on red",
}


class OutputFormat(StrEnum):
    """Supported CLI output formats (uses Python 3.11+ StrEnum per Ruff UP042)."""

    RICH = "rich"
    JSON = "json"
    SARIF = "sarif"


def _fail(message: str) -> NoReturn:
    """Print an escaped error message to stderr and exit with code 2 (operational error)."""
    _err.print(f"[bold red]error:[/bold red] {escape(message)}")
    raise typer.Exit(code=2)


def _version_callback(value: bool) -> None:
    """Print package version immediately and exit when `--version` is passed."""
    if value:
        typer.echo(f"argus {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """argus - lint unsigned EVM transactions before you sign them."""


@app.command()
def check(
    payload_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to an unsigned-tx JSON payload.",
        ),
    ],
    rpc_url: Annotated[
        str | None,
        typer.Option(
            "--rpc-url",
            envvar="ARGUS_RPC_URL",
            help="JSON-RPC endpoint for live simulation (needs debug_traceCall for diffs).",
        ),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Use the payload's `_mock_context`; no network, no RPC key.",
        ),
    ] = False,
    explain: Annotated[
        bool,
        typer.Option(
            "--explain",
            help="Add ML risk score + SHAP attributions (needs `.[ml]`).",
        ),
    ] = False,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", case_sensitive=False, help="Output format."),
    ] = OutputFormat.RICH,
    fail_on: Annotated[
        Severity,
        typer.Option(
            "--fail-on",
            case_sensitive=False,
            help="Exit with code 1 if any finding is at or above this severity.",
        ),
    ] = Severity.HIGH,
) -> None:
    """Load, decode, simulate, and lint a single unsigned EVM transaction payload."""
    try:
        tx, raw = load_payload(payload_path)
    except PayloadError as exc:
        _fail(str(exc))

    decoded = decode_calldata(tx.data)

    try:
        if offline:
            simulator = OfflineFixtureSimulator(raw)
        elif rpc_url:
            simulator = LiveRPCSimulator(rpc_url)
        else:
            simulator = None
    except ValueError as exc:
        _fail(str(exc))

    if simulator is None:
        _fail("provide --rpc-url (or ARGUS_RPC_URL) for live simulation, or pass --offline.")

    try:
        ctx = simulator.simulate(tx, decoded)
    except ValidationError as exc:
        _fail(f"invalid `_mock_context` in {payload_path}:\n{exc}")
    except Exception as exc:
        _fail(f"simulation failed ({type(exc).__name__}): {exc}")

    findings = run_rules(ctx)
    findings.sort(key=lambda f: (-f.severity.rank, f.rule_id))
    report = LintReport(
        findings=findings,
        deterministic_risk_score=compute_risk_score(findings),
        passed=not any(f.severity.meets(fail_on) for f in findings),
    )

    if explain:
        try:
            ml_scorer = importlib.import_module("argus.ml_scorer")
            report.ml_risk_score, report.shap_attributions = ml_scorer.explain(ctx, findings)
        except ImportError as exc:
            _fail(
                "--explain needs the ML extras and argus.ml_scorer "
                f"(pip install -e '.[ml]'). Detail: {exc}"
            )
        except Exception as exc:
            _fail(f"ML scoring failed ({type(exc).__name__}): {exc}")

    if output_format == OutputFormat.RICH:
        render_rich_report(report, ctx, Console())
    elif output_format == OutputFormat.JSON:
        typer.echo(render_json_report(report, ctx, source=payload_path.as_posix()))
    else:
        typer.echo(render_sarif_report(report, artifact_uri=payload_path.as_posix()))

    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def rules() -> None:
    """Print a Rich table listing every registered ARGUS security rule and its severity."""
    table = Table(title="argus rules", show_lines=True)
    table.add_column("ID", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Title")
    table.add_column("What it catches")
    for rule in all_rules():
        top = max(rule.severities, key=lambda sev: sev.rank)
        sev_cell = f"[{_SEVERITY_STYLE[top]}]{rule.severity_label}[/]"
        table.add_row(rule.rule_id, sev_cell, rule.title, rule.summary)
    Console().print(table)


if __name__ == "__main__":  # pragma: no cover
    app(prog_name="argus", args=sys.argv[1:])