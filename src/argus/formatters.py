"""Output formatters: Rich terminal dashboard, JSON, and SARIF 2.1.0."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from argus import __version__
from argus.models import LintReport, Severity, SimulationContext
from argus.rules import all_rules

INFORMATION_URI = "https://github.com/farvillage/ARGUS"

_SEVERITY_STYLE: dict[Severity, str] = {
    Severity.LOW: "cyan",
    Severity.MEDIUM: "yellow",
    Severity.HIGH: "red",
    Severity.CRITICAL: "bold white on red",
}

_UNLIMITED = 2**255


def _text(value: Any, style: str = "") -> Text:
    return Text(str(value), style=style)


def _format_arg(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and value >= _UNLIMITED:
        return f"{value}  (>= 2^255, effectively unlimited)"
    return str(value)


def _format_eth(wei: int) -> str:
    return f"{wei / 10**18:+.6f} ETH"


def _decoded_call_panel(ctx: SimulationContext) -> Panel:
    tx, call = ctx.tx, ctx.decoded_call
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    grid.add_row("From", _text(tx.from_addr))
    grid.add_row("To", _text(tx.to_addr or "<contract creation>"))
    grid.add_row("Chain ID", _text(tx.chain_id))
    grid.add_row("Value", _text(f"{tx.value} wei ({tx.value / 10**18:.6f} ETH)"))
    grid.add_row("Selector", _text(call.selector))
    grid.add_row(
        "Function",
        _text(call.function_name, "green" if call.is_known_selector else "bold yellow"),
    )
    if not call.is_known_selector:
        grid.add_row("Warning", _text("Unknown 4-byte selector with non-empty calldata", "yellow"))
    for name, value in call.arguments.items():
        grid.add_row(_text(f"  {name}", "dim"), _text(_format_arg(value)))
    if tx.authorization_list:
        for index, item in enumerate(tx.authorization_list):
            grid.add_row(
                _text(f"  auth[{index}]", "dim"),
                _text(f"delegate={item.address} chain_id={item.chain_id} nonce={item.nonce}"),
            )
    return Panel(grid, title="Decoded Call", border_style="blue")


def _state_diff_panel(ctx: SimulationContext) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    if not ctx.trace_available:
        table.add_row("Trace", _text("unavailable - state diff unknown", "yellow"))
    eth_style = "red" if ctx.eth_delta_wei < 0 else "green" if ctx.eth_delta_wei > 0 else ""
    table.add_row("ETH delta", _text(_format_eth(ctx.eth_delta_wei), eth_style))
    if ctx.erc20_deltas:
        for token, delta in ctx.erc20_deltas.items():
            table.add_row(
                _text(f"  {token}", "dim"),
                _text(f"{delta:+d}", "red" if delta < 0 else "green"),
            )
    else:
        table.add_row("Token deltas", _text("none", "dim"))
    table.add_row("Approvals granted", _text(len(ctx.approvals_granted)))
    table.add_row("Call depth", _text(ctx.call_depth))
    table.add_row("Reverted", _text(ctx.reverted, "red" if ctx.reverted else "green"))
    table.add_row("Target is contract", _text(ctx.is_contract))
    verified_style = "green" if ctx.is_verified else "yellow"
    table.add_row("Target verified", _text(ctx.is_verified, verified_style))
    age = "unknown" if ctx.deployment_age_blocks is None else f"{ctx.deployment_age_blocks} blocks"
    table.add_row("Deployment age", _text(age))
    table.add_row("Opcodes analysed", _text(len(ctx.opcodes)))
    for note in ctx.notes:
        table.add_row(_text("Note", "dim"), _text(note, "dim"))
    return Panel(table, title="State-Diff Summary", border_style="blue")


def _findings_table(report: LintReport) -> Table | Panel:
    if not report.findings:
        return Panel(_text("No findings.", "green"), title="Findings", border_style="green")
    table = Table(title="Findings", show_lines=True, expand=True)
    table.add_column("Rule", no_wrap=True)
    table.add_column("Severity", no_wrap=True)
    table.add_column("Title")
    table.add_column("Details / Remediation", ratio=2)
    for finding in report.findings:
        detail = Text(finding.description)
        detail.append("\nFix: ", style="bold")
        detail.append(finding.remediation)
        table.add_row(
            finding.rule_id,
            _text(finding.severity.name, _SEVERITY_STYLE[finding.severity]),
            _text(finding.title),
            detail,
        )
    return table


def _shap_panel(report: LintReport) -> Panel | None:
    if not report.shap_attributions:
        return None
    attributions = sorted(
        report.shap_attributions.items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    peak = max(abs(value) for _, value in attributions) or 1.0
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right")
    for name, value in attributions:
        bar = "█" * max(1, round(abs(value) / peak * 24))
        bar_color = "red" if value > 0 else "green"
        grid.add_row(_text(name), _text(bar, bar_color), _text(f"{value:+.4f}"))
    title = "ML Explanation (SHAP) - red raises risk, green lowers it"
    if report.ml_risk_score is not None:
        title += f" | ML risk {report.ml_risk_score:.2%}"
    return Panel(grid, title=title, border_style="magenta")


def _verdict_banner(report: LintReport) -> Panel:
    score = f"deterministic risk score {report.deterministic_risk_score}/100"
    if not report.passed:
        return Panel(
            Align.center(_text(f"DO NOT SIGN - {score}", "bold white")),
            style="on red",
            title="Verdict",
        )
    if report.findings:
        return Panel(
            Align.center(_text(f"PASS WITH WARNINGS - {score}", "bold black")),
            style="on yellow",
            title="Verdict",
        )
    return Panel(
        Align.center(_text(f"PASS - {score}", "bold black")), style="on green", title="Verdict"
    )


def render_rich_report(
    report: LintReport, ctx: SimulationContext, console: Console | None = None
) -> None:
    """Print the full terminal dashboard."""
    out = console or Console()
    out.print(_decoded_call_panel(ctx))
    out.print(_state_diff_panel(ctx))
    out.print(_findings_table(report))
    shap_panel = _shap_panel(report)
    if shap_panel is not None:
        out.print(shap_panel)
    out.print(_verdict_banner(report))


def render_json_report(
    report: LintReport, ctx: SimulationContext | None = None, source: str | None = None
) -> str:
    payload: dict[str, Any] = {
        "tool": {"name": "argus", "version": __version__},
        "source": source,
        "report": report.model_dump(mode="json"),
    }
    if ctx is not None:
        payload["transaction"] = ctx.tx.model_dump(mode="json")
        payload["decoded_call"] = ctx.decoded_call.model_dump(mode="json")
    return json.dumps(payload, indent=2, default=str)


_SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
_SARIF_LEVEL: dict[Severity, str] = {
    Severity.LOW: "note",
    Severity.MEDIUM: "warning",
    Severity.HIGH: "error",
    Severity.CRITICAL: "error",
}
_SECURITY_SEVERITY: dict[Severity, str] = {
    Severity.LOW: "3.0",
    Severity.MEDIUM: "5.5",
    Severity.HIGH: "8.0",
    Severity.CRITICAL: "9.5",
}


def render_sarif_report(report: LintReport, artifact_uri: str = "payload.json") -> str:
    """Render a SARIF 2.1.0 log suitable for GitHub code scanning upload."""
    rules = all_rules()
    rule_index = {rule.rule_id: index for index, rule in enumerate(rules)}

    sarif_rules = []
    for rule in rules:
        top = max(rule.severities, key=lambda sev: sev.rank)
        sarif_rules.append(
            {
                "id": rule.rule_id,
                "name": rule.rule_id,
                "shortDescription": {"text": rule.title},
                "fullDescription": {"text": rule.summary},
                "help": {"text": rule.remediation},
                "defaultConfiguration": {"level": _SARIF_LEVEL[top]},
                "properties": {
                    "security-severity": _SECURITY_SEVERITY[top],
                    "tags": ["security", "web3", "evm"],
                },
            }
        )

    results = []
    for finding in report.findings:
        evidence_json = json.dumps(finding.evidence, sort_keys=True, default=str)
        raw_fp = f"{finding.rule_id}:{artifact_uri}:{evidence_json}".encode()
        fingerprint = hashlib.sha256(raw_fp).hexdigest()[:32]
        results.append(
            {
                "ruleId": finding.rule_id,
                "ruleIndex": rule_index.get(finding.rule_id, -1),
                "level": _SARIF_LEVEL[finding.severity],
                "message": {"text": f"{finding.title}: {finding.description}"},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": artifact_uri},
                            "region": {"startLine": 1},
                        }
                    }
                ],
                "partialFingerprints": {"argus/v1": fingerprint},
                "properties": {
                    "severity": finding.severity.name,
                    "evidence": finding.evidence,
                    "remediation": finding.remediation,
                    "security-severity": _SECURITY_SEVERITY[finding.severity],
                },
            }
        )

    log = {
        "$schema": _SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "argus",
                        "version": __version__,
                        "informationUri": INFORMATION_URI,
                        "rules": sarif_rules,
                    }
                },
                "invocations": [{"executionSuccessful": True}],
                "results": results,
            }
        ],
    }
    return json.dumps(log, indent=2, default=str)