"""Modular rule registry. Each rule inspects a ``SimulationContext`` and returns findings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from argus.models import Finding, Severity, SimulationContext

UNBOUNDED_APPROVAL_THRESHOLD = 2**255
PERMIT_MAX_DEADLINE_SECONDS = 30 * 24 * 60 * 60
FRESH_DEPLOYMENT_BLOCKS = 500

SEVERITY_WEIGHTS: dict[Severity, int] = {
    Severity.LOW: 5,
    Severity.MEDIUM: 15,
    Severity.HIGH: 35,
    Severity.CRITICAL: 60,
}


class BaseRule(ABC):
    rule_id: ClassVar[str]
    title: ClassVar[str]
    severities: ClassVar[tuple[Severity, ...]]
    summary: ClassVar[str]
    remediation: ClassVar[str]

    @property
    def severity_label(self) -> str:
        return " / ".join(sev.name for sev in self.severities)

    @abstractmethod
    def evaluate(self, ctx: SimulationContext) -> list[Finding]:
        """Return zero or more findings for the given simulation context."""

    def _finding(
        self, severity: Severity, description: str, evidence: dict[str, Any]
    ) -> Finding:
        return Finding(
            rule_id=self.rule_id,
            title=self.title,
            severity=severity,
            description=description,
            evidence=evidence,
            remediation=self.remediation,
        )


_REGISTRY: dict[str, BaseRule] = {}


def register(rule_cls: type[BaseRule]) -> type[BaseRule]:
    """Class decorator that instantiates and registers a rule."""
    _REGISTRY[rule_cls.rule_id] = rule_cls()
    return rule_cls


def all_rules() -> list[BaseRule]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def run_rules(ctx: SimulationContext) -> list[Finding]:
    findings: list[Finding] = []
    for rule in all_rules():
        findings.extend(rule.evaluate(ctx))
    return findings


def compute_risk_score(findings: list[Finding]) -> int:
    """Deterministic 0-100 score: sum of severity weights, capped at 100."""
    return min(100, sum(SEVERITY_WEIGHTS[f.severity] for f in findings))


def _is_unlimited_trace_amount(raw_amount: Any) -> bool:
    if raw_amount == "unlimited":
        return True
    if isinstance(raw_amount, int):
        return raw_amount >= UNBOUNDED_APPROVAL_THRESHOLD
    if isinstance(raw_amount, str) and raw_amount.isdigit():
        return int(raw_amount) >= UNBOUNDED_APPROVAL_THRESHOLD
    return False


@register
class UnboundedApprovalRule(BaseRule):
    rule_id = "ARG001"
    title = "Unbounded token approval"
    severities = (Severity.HIGH, Severity.CRITICAL)
    summary = (
        "ERC-20 approve() with an effectively unlimited amount (>= 2^255), "
        "setApprovalForAll(true), or nested unlimited trace approvals. "
        "CRITICAL when the spender is an EOA or unverified contract."
    )
    remediation = (
        "Approve only the exact amount needed, or revoke after use. Never grant blanket "
        "approvals to EOAs or unverified contracts."
    )

    def evaluate(self, ctx: SimulationContext) -> list[Finding]:
        call = ctx.decoded_call
        args = call.arguments
        spender: str | None = None
        kind: str | None = None

        if call.function_name == "approve":
            amount = args.get("amount")
            if isinstance(amount, int) and amount >= UNBOUNDED_APPROVAL_THRESHOLD:
                spender, kind = args.get("spender"), "unbounded ERC-20 approval"
        elif call.function_name == "setApprovalForAll" and args.get("approved") is True:
            spender, kind = args.get("operator"), "setApprovalForAll(true) over a whole collection"
        elif call.function_name != "permit":
            for grant in ctx.approvals_granted:
                if grant.get("approved") is True:
                    spender = grant.get("operator")
                    kind = "nested setApprovalForAll(true) emitted in trace"
                    break
                if _is_unlimited_trace_amount(grant.get("amount")):
                    spender = grant.get("spender")
                    kind = "nested unbounded ERC-20 approval emitted in trace"
                    break

        if kind is None or not isinstance(spender, str):
            return []

        reasons: list[str] = []
        if ctx.is_eoa(spender):
            reasons.append("spender is an EOA (no contract code)")
        if ctx.is_unverified(spender):
            reasons.append("spender is unverified or unknown")
        severity = Severity.CRITICAL if reasons else Severity.HIGH
        evidence: dict[str, Any] = {
            "token_or_collection": ctx.tx.to_addr,
            "spender": spender,
            "function": call.function_name,
            "escalation_reasons": reasons,
        }
        if "amount" in args:
            evidence["amount"] = str(args["amount"])
        return [
            self._finding(
                severity,
                f"Transaction grants a {kind} to {spender}."
                + (f" Escalated: {'; '.join(reasons)}." if reasons else ""),
                evidence,
            )
        ]


@register
class PermitPhishingRule(BaseRule):
    rule_id = "ARG002"
    title = "EIP-2612 permit phishing signature"
    severities = (Severity.CRITICAL,)
    summary = (
        "permit() granting allowance to an EOA or unverified spender, or with a deadline "
        "more than 30 days ahead. Off-chain permits are a favourite drainer primitive."
    )
    remediation = (
        "Do not sign. Legitimate permits go to known verified router contracts with short "
        "deadlines."
    )

    def evaluate(self, ctx: SimulationContext) -> list[Finding]:
        call = ctx.decoded_call
        if call.function_name != "permit":
            return []
        args = call.arguments
        spender = args.get("spender")
        deadline = args.get("deadline")
        reasons: list[str] = []
        evidence: dict[str, Any] = {
            "token": ctx.tx.to_addr,
            "owner": args.get("owner"),
            "spender": spender,
            "value": str(args.get("value")),
        }
        if isinstance(spender, str):
            if ctx.is_eoa(spender):
                reasons.append("spender is an EOA")
            elif ctx.is_unverified(spender):
                reasons.append("spender is an unverified or unknown contract")
        if isinstance(deadline, int):
            now = ctx.reference_time
            evidence["deadline"] = str(deadline)
            evidence["days_until_deadline"] = round((deadline - now) / 86400, 2)
            if deadline - now > PERMIT_MAX_DEADLINE_SECONDS:
                reasons.append("deadline is more than 30 days ahead")
        if not reasons:
            return []
        evidence["reasons"] = reasons
        return [
            self._finding(
                Severity.CRITICAL,
                f"Permit signature looks like phishing: {'; '.join(reasons)}.",
                evidence,
            )
        ]


@register
class Eip7702DelegationRule(BaseRule):
    rule_id = "ARG003"
    title = "EIP-7702 delegation hijack"
    severities = (Severity.CRITICAL,)
    summary = (
        "authorization_list delegates the EOA's code execution to an unverified or unknown "
        "contract, letting that contract act with the account's full authority."
    )
    remediation = (
        "Do not sign. Only delegate to audited, verified wallet implementations you trust."
    )

    def evaluate(self, ctx: SimulationContext) -> list[Finding]:
        offenders: list[dict[str, Any]] = []
        for item in ctx.tx.authorization_list:
            info = ctx.info(item.address)
            if info is None:
                reason = "delegate is unknown"
            elif not info.is_contract:
                reason = "delegate has no contract code"
            elif not info.is_verified:
                reason = "delegate is unverified"
            else:
                continue
            offenders.append(
                {
                    "delegate": item.address,
                    "chain_id": item.chain_id,
                    "nonce": item.nonce,
                    "reason": reason,
                    "chain_id_wildcard": item.chain_id == 0,
                }
            )
        if not offenders:
            return []
        wildcard = any(entry["chain_id_wildcard"] for entry in offenders)
        wildcard_note = (
            " chain_id 0 makes the authorization replayable on every chain."
            if wildcard
            else ""
        )
        return [
            self._finding(
                Severity.CRITICAL,
                f"{len(offenders)} EIP-7702 authorization(s) delegate account code to "
                f"unverified or unknown contracts.{wildcard_note}",
                {"delegations": offenders},
            )
        ]


@register
class FreshUnverifiedContractRule(BaseRule):
    rule_id = "ARG004"
    title = "Freshly deployed unverified contract"
    severities = (Severity.HIGH,)
    summary = "Target contract was deployed < 500 blocks ago and has no verified source."
    remediation = (
        "Wait, verify the source on a block explorer, and confirm the address via an "
        "official channel."
    )

    def evaluate(self, ctx: SimulationContext) -> list[Finding]:
        age = ctx.deployment_age_blocks
        if not ctx.is_contract or ctx.is_verified or age is None:
            return []
        if age >= FRESH_DEPLOYMENT_BLOCKS:
            return []
        return [
            self._finding(
                Severity.HIGH,
                f"Target {ctx.tx.to_addr} was deployed {age} blocks ago and is unverified.",
                {
                    "target": ctx.tx.to_addr,
                    "deployment_age_blocks": age,
                    "threshold_blocks": FRESH_DEPLOYMENT_BLOCKS,
                },
            )
        ]


@register
class DangerousBytecodeRule(BaseRule):
    rule_id = "ARG005"
    title = "Dangerous opcodes in unverified bytecode"
    severities = (Severity.HIGH,)
    summary = (
        "Non-standard DELEGATECALL (no EIP-1967 implementation slot) or SELFDESTRUCT found in "
        "the runtime bytecode of an unverified contract."
    )
    remediation = (
        "Treat the contract as untrusted. Require verified source and an audit before interacting."
    )

    def evaluate(self, ctx: SimulationContext) -> list[Finding]:
        if ctx.is_verified:
            return []
        names = ctx.opcode_mnemonics
        found: list[str] = []
        if ctx.has_nonstandard_delegatecall:
            found.append("DELEGATECALL (non-standard, no EIP-1967 slot)")
        if "SELFDESTRUCT" in names:
            found.append("SELFDESTRUCT")
        if not found:
            return []
        return [
            self._finding(
                Severity.HIGH,
                f"Unverified target contains {', '.join(found)}.",
                {
                    "target": ctx.tx.to_addr,
                    "dangerous_opcodes": found,
                    "opcode_count": len(ctx.opcodes),
                },
            )
        ]


@register
class AsymmetricDrainRule(BaseRule):
    rule_id = "ARG006"
    title = "Asymmetric net asset drain"
    severities = (Severity.CRITICAL,)
    summary = (
        "Simulation shows negative ETH or ERC-20 deltas for the sender with no inbound assets "
        "or protocol receipt tokens."
    )
    remediation = "Do not sign. A legitimate swap, deposit or mint always returns something."

    def evaluate(self, ctx: SimulationContext) -> list[Finding]:
        if not ctx.trace_available or ctx.reverted:
            return []
        outflows: dict[str, str] = {}
        if ctx.eth_delta_wei < 0:
            outflows["ETH (wei)"] = str(ctx.eth_delta_wei)
        for token, delta in ctx.erc20_deltas.items():
            if delta < 0:
                outflows[token] = str(delta)
        if not outflows:
            return []
        has_inbound = (
            ctx.eth_delta_wei > 0
            or any(delta > 0 for delta in ctx.erc20_deltas.values())
            or bool(ctx.receipt_tokens)
        )
        if has_inbound:
            return []
        if ctx.tx.data == "0x" and not ctx.is_contract:
            return []
        return [
            self._finding(
                Severity.CRITICAL,
                "Sender loses assets and receives nothing in return.",
                {
                    "outflows": outflows,
                    "inbound_assets": [],
                    "receipt_tokens": ctx.receipt_tokens,
                    "call_depth": ctx.call_depth,
                },
            )
        ]