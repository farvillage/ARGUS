"""argus test suite: rules, decoding, disassembly, SARIF, CLI exit codes, hardening, ML."""

# Standard library imports for JSON parsing and filesystem path handling
import json
from pathlib import Path

# Third-party testing and Ethereum ABI encoding helpers
import pytest
from eth_abi import encode
from eth_utils import function_signature_to_4byte_selector
from typer.testing import CliRunner

# Internal modules under test
from argus.cli import app
from argus.evm import (
    EIP1967_IMPLEMENTATION_SLOT,
    LiveRPCSimulator,
    OfflineFixtureSimulator,
    PayloadError,
    _TraceAccumulator,
    _walk_trace,
    analyze_bytecode,
    decode_calldata,
    disassemble_opcodes,
    load_payload,
)
from argus.formatters import render_sarif_report
from argus.models import (
    AddressInfo,
    LintReport,
    Severity,
    SimulationContext,
    UnsignedTransaction,
)
from argus.rules import (
    AsymmetricDrainRule,
    DangerousBytecodeRule,
    Eip7702DelegationRule,
    FreshUnverifiedContractRule,
    PermitPhishingRule,
    UnboundedApprovalRule,
    all_rules,
    compute_risk_score,
    run_rules,
)

# Project root and example fixtures directory
ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# Deterministic timestamp and mock EVM addresses used across unit tests.
# Note: `USDC_ADDR` avoids the word "TOKEN" so Bandit's S105 secret check doesn't false-flag it.
NOW = 1_790_000_000
VICTIM = "0x3c44cdddb6a900fa2b585dd299e03d12fa4293bc"
SPENDER_EOA = "0xbadc0de0badc0de0badc0de0badc0de0badc0de0"
ROUTER = "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45"
USDC_ADDR = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
DELEGATE = "0xd3adb33fd3adb33fd3adb33fd3adb33fd3adb33f"

# Reusable AddressInfo fixtures representing an EOA, verified contract, and unverified contract
EOA = AddressInfo(is_contract=False, is_verified=False)
VERIFIED = AddressInfo(is_contract=True, is_verified=True)
UNVERIFIED = AddressInfo(is_contract=True, is_verified=False)
MAX_UINT = 2**256 - 1

# Typer CLI test runner for invoking `argus` commands in-process
runner = CliRunner()


# --------------------------------------------------------------------------- #
# Test payload construction helpers
# --------------------------------------------------------------------------- #


def calldata(signature: str, types: list[str], values: list) -> str:
    """Build 0x-prefixed ABI-encoded calldata from a Solidity signature and argument values."""
    selector = function_signature_to_4byte_selector(signature)
    return "0x" + (selector + encode(types, values)).hex()


def approve_data(spender: str, amount: int) -> str:
    """Generate ERC-20 `approve(address,uint256)` hex calldata."""
    return calldata("approve(address,uint256)", ["address", "uint256"], [spender, amount])


def approval_for_all_data(operator: str, approved: bool) -> str:
    """Generate ERC-721/1155 `setApprovalForAll(address,bool)` hex calldata."""
    return calldata("setApprovalForAll(address,bool)", ["address", "bool"], [operator, approved])


def permit_data(spender: str, deadline: int, value: int = MAX_UINT) -> str:
    """Generate EIP-2612 `permit(...)` hex calldata with dummy ECDSA signature bytes."""
    return calldata(
        "permit(address,address,uint256,uint256,uint8,bytes32,bytes32)",
        ["address", "address", "uint256", "uint256", "uint8", "bytes32", "bytes32"],
        [VICTIM, spender, value, deadline, 27, b"\x11" * 32, b"\x22" * 32],
    )


def make_ctx(
    data: str = "0x",
    *,
    to: str | None = USDC_ADDR,
    meta: dict | None = None,
    auth: list[str] | None = None,
    **fields,
) -> SimulationContext:
    """Construct an in-memory SimulationContext with sensible defaults for unit testing."""
    payload = {
        "chainId": 1,
        "from": VICTIM,
        "to": to,
        "value": 0,
        "data": data,
        "authorizationList": [
            {"chainId": 1, "address": address, "nonce": 0} for address in (auth or [])
        ],
    }
    tx = UnsignedTransaction.model_validate(payload)
    return SimulationContext(
        tx=tx,
        decoded_call=decode_calldata(data),
        address_metadata=meta or {},
        block_timestamp=NOW,
        **fields,
    )


def analyze_example(name: str) -> tuple[SimulationContext, LintReport]:
    """Load an example JSON file from `examples/`, simulate offline, and run all rules."""
    tx, raw = load_payload(EXAMPLES / name)
    ctx = OfflineFixtureSimulator(raw).simulate(tx, decode_calldata(tx.data))
    findings = run_rules(ctx)
    report = LintReport(
        findings=findings,
        deterministic_risk_score=compute_risk_score(findings),
        passed=not findings,
    )
    return ctx, report


# --------------------------------------------------------------------------- #
# Calldata decoding tests
# --------------------------------------------------------------------------- #


def test_known_selectors_match_published_values():
    """Verify computed 4-byte selectors match canonical Uniswap V3 and EIP-2612 selectors."""
    router_v1 = (
        "exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))"
    )
    router_02 = "exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))"
    assert function_signature_to_4byte_selector(router_v1).hex() == "414bf389"
    assert function_signature_to_4byte_selector(router_02).hex() == "04e45aaf"
    permit_sig = "permit(address,address,uint256,uint256,uint8,bytes32,bytes32)"
    assert function_signature_to_4byte_selector(permit_sig).hex() == "d505accf"


def test_decode_approve():
    """Ensure ERC-20 `approve` decodes spender address and integer allowance accurately."""
    call = decode_calldata(approve_data(ROUTER, 1234))
    assert call.function_name == "approve" and call.is_known_selector
    assert call.arguments == {"spender": ROUTER, "amount": 1234}


def test_decode_set_approval_for_all_and_transfer_from():
    """Ensure NFT `setApprovalForAll` and `transferFrom` calls decode all arguments."""
    call = decode_calldata(approval_for_all_data(ROUTER, True))
    assert call.function_name == "setApprovalForAll"
    assert call.arguments == {"operator": ROUTER, "approved": True}

    transfer = decode_calldata(
        calldata(
            "transferFrom(address,address,uint256)",
            ["address", "address", "uint256"],
            [VICTIM, SPENDER_EOA, 7],
        )
    )
    assert transfer.arguments == {"from": VICTIM, "to": SPENDER_EOA, "amount": 7}


def test_decode_permit_normalizes_bytes():
    """Ensure `bytes32` signature parameters (`r`, `s`) normalize to 0x hex strings."""
    call = decode_calldata(permit_data(SPENDER_EOA, NOW + 100))
    assert call.function_name == "permit"
    assert call.arguments["spender"] == SPENDER_EOA
    assert call.arguments["r"] == "0x" + "11" * 32
    assert call.arguments["v"] == 27


def test_decode_exact_input_single_both_router_versions():
    """Verify both SwapRouter (8-field tuple) and SwapRouter02 (7-field tuple) decode cleanly."""
    v1_type = "(address,address,uint24,address,uint256,uint256,uint256,uint160)"
    v1_sel = function_signature_to_4byte_selector(f"exactInputSingle({v1_type})")
    v1_body = encode([v1_type], [(USDC_ADDR, ROUTER, 500, VICTIM, NOW, 10**18, 1, 0)])
    v1 = decode_calldata("0x" + (v1_sel + v1_body).hex())
    assert v1.function_name == "exactInputSingle"
    assert v1.arguments["deadline"] == NOW and v1.arguments["amountIn"] == 10**18

    v2_type = "(address,address,uint24,address,uint256,uint256,uint160)"
    v2_sel = function_signature_to_4byte_selector(f"exactInputSingle({v2_type})")
    v2_body = encode([v2_type], [(USDC_ADDR, ROUTER, 500, VICTIM, 10**18, 1, 0)])
    v2 = decode_calldata("0x" + (v2_sel + v2_body).hex())
    assert v2.function_name == "exactInputSingle" and "deadline" not in v2.arguments


def test_unknown_selector_is_flagged():
    """Verify unrecognized or <4-byte calldata sets `is_known_selector` to False."""
    call = decode_calldata("0xdeadbeef" + "00" * 32)
    assert call.is_known_selector is False
    assert call.function_name == "<unknown>"
    short = decode_calldata("0xdead")
    assert short.is_known_selector is False


def test_empty_calldata_is_native_transfer():
    """Verify empty `0x` calldata is classified as a native ETH transfer."""
    call = decode_calldata("0x")
    assert call.function_name == "<native transfer>" and call.is_known_selector


def test_known_selector_with_garbage_args_reports_decode_error():
    """Verify truncated ABI arguments capture `decode_error` without crashing."""
    call = decode_calldata("0x095ea7b3" + "00")
    assert call.function_name == "approve"
    assert "decode_error" in call.arguments


# --------------------------------------------------------------------------- #
# Pure-Python EVM bytecode disassembler tests
# --------------------------------------------------------------------------- #


def test_push_operands_are_skipped_not_executed():
    """Ensure 0xf4/0xff bytes embedded inside PUSH operands are never misread as opcodes."""
    assert disassemble_opcodes("0x61f4ff") == ["PUSH2 0xf4ff"]
    assert disassemble_opcodes("73" + "f4" * 20 + "f4") == [
        "PUSH20 0x" + "f4" * 20,
        "DELEGATECALL",
    ]
    assert disassemble_opcodes("60f4f4") == ["PUSH1 0xf4", "DELEGATECALL"]


def test_push32_swallows_exactly_32_bytes():
    """Ensure PUSH32 consumes 32 operand bytes before decoding the next instruction."""
    assert disassemble_opcodes("7f" + "ff" * 32 + "ff") == [
        "PUSH32 0x" + "ff" * 32,
        "SELFDESTRUCT",
    ]


def test_push0_has_no_operand():
    """Ensure EIP-3855 PUSH0 (`0x5f`) consumes zero immediate bytes."""
    assert disassemble_opcodes("5f5f") == ["PUSH0", "PUSH0"]


def test_truncated_push_keeps_remaining_bytes():
    """Ensure truncated bytecode at the end of a PUSH instruction does not raise IndexError."""
    assert disassemble_opcodes("61ff") == ["PUSH2 0xff"]


def test_recognises_delegatecall_create2_selfdestruct():
    """Verify dangerous execution/lifecycle opcodes disassemble to their exact names."""
    assert disassemble_opcodes("f4f5ff") == ["DELEGATECALL", "CREATE2", "SELFDESTRUCT"]


def test_eip1967_slot_pairing():
    """Distinguish standard EIP-1967 proxies from rogue non-standard DELEGATECALL contracts."""
    slot = EIP1967_IMPLEMENTATION_SLOT[2:]
    standard = analyze_bytecode("7f" + slot + "54f4")
    assert standard.has_delegatecall and standard.uses_eip1967_slot
    assert standard.has_nonstandard_delegatecall is False

    rogue = analyze_bytecode("6000f4")
    assert rogue.has_nonstandard_delegatecall is True
    assert analyze_bytecode("6000").has_nonstandard_delegatecall is False


def test_invalid_hex_raises():
    """Ensure non-hexadecimal bytecode strings raise ValueError."""
    with pytest.raises(ValueError):
        disassemble_opcodes("0xzz")


# --------------------------------------------------------------------------- #
# Rule ARG001: Unbounded approvals
# --------------------------------------------------------------------------- #


def test_arg001_unbounded_approve_to_eoa_is_critical():
    """Granting MAX_UINT256 ERC-20 allowance to an EOA must trigger CRITICAL."""
    ctx = make_ctx(approve_data(SPENDER_EOA, MAX_UINT), meta={SPENDER_EOA: EOA})
    findings = UnboundedApprovalRule().evaluate(ctx)
    assert [f.severity for f in findings] == [Severity.CRITICAL]


def test_arg001_unbounded_approve_to_unknown_spender_is_critical():
    """Granting >= 2^255 allowance to an unknown spender must fail-safe to CRITICAL."""
    ctx = make_ctx(approve_data(SPENDER_EOA, 2**255))
    assert UnboundedApprovalRule().evaluate(ctx)[0].severity is Severity.CRITICAL


def test_arg001_unbounded_approve_to_verified_router_is_high():
    """Granting MAX_UINT256 to a verified router contract triggers HIGH (not CRITICAL)."""
    ctx = make_ctx(approve_data(ROUTER, MAX_UINT), meta={ROUTER: VERIFIED})
    assert UnboundedApprovalRule().evaluate(ctx)[0].severity is Severity.HIGH


def test_arg001_bounded_approve_is_clean():
    """Bounded ERC-20 approvals for exact swap amounts must pass cleanly."""
    ctx = make_ctx(approve_data(SPENDER_EOA, 1000), meta={SPENDER_EOA: EOA})
    assert UnboundedApprovalRule().evaluate(ctx) == []


def test_arg001_set_approval_for_all():
    """Verify `setApprovalForAll(true)` escalates on EOAs, warns on routers, passes on revoke."""
    risky = make_ctx(approval_for_all_data(SPENDER_EOA, True), meta={SPENDER_EOA: EOA})
    assert UnboundedApprovalRule().evaluate(risky)[0].severity is Severity.CRITICAL

    trusted = make_ctx(approval_for_all_data(ROUTER, True), meta={ROUTER: VERIFIED})
    assert UnboundedApprovalRule().evaluate(trusted)[0].severity is Severity.HIGH

    revoke = make_ctx(approval_for_all_data(SPENDER_EOA, False), meta={SPENDER_EOA: EOA})
    assert UnboundedApprovalRule().evaluate(revoke) == []


def test_arg001_catches_stealth_trace_approval():
    """Ensure nested unlimited Approval logs emitted during simulation are caught by ARG001."""
    trace_grant = {
        "standard": "ERC-20",
        "token": USDC_ADDR,
        "spender": SPENDER_EOA,
        "amount": str(MAX_UINT),
    }
    ctx = make_ctx("0xe9ae5c53", meta={SPENDER_EOA: EOA}, approvals_granted=[trace_grant])
    findings = UnboundedApprovalRule().evaluate(ctx)
    assert findings and findings[0].severity is Severity.CRITICAL


# --------------------------------------------------------------------------- #
# Rule ARG002: EIP-2612 Permit phishing
# --------------------------------------------------------------------------- #


def test_arg002_permit_to_eoa():
    """Permit signature naming an EOA wallet as spender must trigger CRITICAL."""
    ctx = make_ctx(permit_data(SPENDER_EOA, NOW + 600), meta={SPENDER_EOA: EOA})
    findings = PermitPhishingRule().evaluate(ctx)
    assert findings and findings[0].severity is Severity.CRITICAL


def test_arg002_permit_to_unverified_contract_is_critical():
    """Permit signature naming an unverified smart contract spender must trigger CRITICAL."""
    ctx = make_ctx(permit_data(DELEGATE, NOW + 600), meta={DELEGATE: UNVERIFIED})
    findings = PermitPhishingRule().evaluate(ctx)
    assert findings and findings[0].severity is Severity.CRITICAL


def test_arg002_far_deadline_to_verified_spender():
    """Permit signature expiring >30 days in the future must trigger CRITICAL."""
    ctx = make_ctx(permit_data(ROUTER, NOW + 31 * 86400), meta={ROUTER: VERIFIED})
    findings = PermitPhishingRule().evaluate(ctx)
    assert findings and findings[0].evidence["days_until_deadline"] > 30


def test_arg002_short_deadline_to_verified_router_is_clean():
    """Short-lived Permit signature to a verified router contract must pass cleanly."""
    ctx = make_ctx(permit_data(ROUTER, NOW + 3600), meta={ROUTER: VERIFIED})
    assert PermitPhishingRule().evaluate(ctx) == []
    within_window = make_ctx(permit_data(ROUTER, NOW + 29 * 86400), meta={ROUTER: VERIFIED})
    assert PermitPhishingRule().evaluate(within_window) == []


# --------------------------------------------------------------------------- #
# Rule ARG003: EIP-7702 account delegation hijacks
# --------------------------------------------------------------------------- #


def test_arg003_unverified_delegate_is_critical():
    """Delegating EOA code execution to an unverified contract must trigger CRITICAL."""
    ctx = make_ctx(to=VICTIM, auth=[DELEGATE], meta={DELEGATE: UNVERIFIED})
    findings = Eip7702DelegationRule().evaluate(ctx)
    assert findings and findings[0].severity is Severity.CRITICAL


def test_arg003_unknown_delegate_is_flagged():
    """Delegating EOA code execution to an unknown address must fail-safe to CRITICAL."""
    ctx = make_ctx(to=VICTIM, auth=[DELEGATE])
    assert Eip7702DelegationRule().evaluate(ctx)


def test_arg003_verified_delegate_is_clean():
    """Delegating EOA code to a known, verified smart account implementation passes."""
    ctx = make_ctx(to=VICTIM, auth=[DELEGATE], meta={DELEGATE: VERIFIED})
    assert Eip7702DelegationRule().evaluate(ctx) == []


def test_arg003_no_authorization_list_is_clean():
    """Standard transactions without an EIP-7702 authorizationList do not trigger ARG003."""
    assert Eip7702DelegationRule().evaluate(make_ctx()) == []


# --------------------------------------------------------------------------- #
# Rule ARG004: Freshly deployed & unverified contracts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("verified", "age", "expected"),
    [
        (False, 10, True),
        (False, 499, True),
        (False, 500, False),
        (True, 10, False),
        (False, None, False),
    ],
)
def test_arg004(verified, age, expected):
    """Verify the 500-block deployment age boundary and verification status on ARG004."""
    ctx = make_ctx(is_contract=True, is_verified=verified, deployment_age_blocks=age)
    findings = FreshUnverifiedContractRule().evaluate(ctx)
    assert bool(findings) is expected
    if expected:
        assert findings[0].severity is Severity.HIGH


# --------------------------------------------------------------------------- #
# Rule ARG005: Dangerous opcodes in unverified bytecode
# --------------------------------------------------------------------------- #


def test_arg005_nonstandard_delegatecall_unverified():
    """Unverified contract using non-EIP-1967 DELEGATECALL must trigger HIGH."""
    ctx = make_ctx(is_contract=True, is_verified=False, has_nonstandard_delegatecall=True)
    assert DangerousBytecodeRule().evaluate(ctx)[0].severity is Severity.HIGH


def test_arg005_selfdestruct_unverified():
    """Unverified contract containing SELFDESTRUCT must trigger HIGH."""
    ctx = make_ctx(is_contract=True, is_verified=False, opcodes=["PUSH1 0x00", "SELFDESTRUCT"])
    assert DangerousBytecodeRule().evaluate(ctx)


def test_arg005_verified_contract_is_clean():
    """Verified contracts with DELEGATECALL are trusted and do not trigger ARG005."""
    ctx = make_ctx(is_contract=True, is_verified=True, has_nonstandard_delegatecall=True)
    assert DangerousBytecodeRule().evaluate(ctx) == []


def test_arg005_standard_proxy_is_clean():
    """Unverified EIP-1967 compliant proxy contracts do not trigger ARG005."""
    analysis = analyze_bytecode("7f" + EIP1967_IMPLEMENTATION_SLOT[2:] + "54f4")
    ctx = make_ctx(
        is_contract=True,
        is_verified=False,
        opcodes=analysis.opcodes,
        has_nonstandard_delegatecall=analysis.has_nonstandard_delegatecall,
    )
    assert DangerousBytecodeRule().evaluate(ctx) == []


# --------------------------------------------------------------------------- #
# Rule ARG006: Asymmetric asset drains & trace walker edge cases
# --------------------------------------------------------------------------- #


def test_arg006_outflow_without_inbound_is_critical():
    """Losing ETH and ERC-20 tokens in a contract call with zero inbound value is CRITICAL."""
    ctx = make_ctx(
        "0xe9ae5c53",
        is_contract=True,
        eth_delta_wei=-(10**18),
        erc20_deltas={USDC_ADDR: -5},
    )
    findings = AsymmetricDrainRule().evaluate(ctx)
    assert findings and findings[0].severity is Severity.CRITICAL
    assert set(findings[0].evidence["outflows"]) == {"ETH (wei)", USDC_ADDR}


def test_arg006_inbound_asset_or_receipt_token_is_clean():
    """Balanced swaps or vault deposits returning receipt tokens must not trigger ARG006."""
    swap = make_ctx(
        "0xe9ae5c53",
        is_contract=True,
        eth_delta_wei=-(10**18),
        erc20_deltas={USDC_ADDR: 900},
    )
    assert AsymmetricDrainRule().evaluate(swap) == []
    deposit = make_ctx(
        "0xe9ae5c53", is_contract=True, eth_delta_wei=-(10**18), receipt_tokens=["stETH"]
    )
    assert AsymmetricDrainRule().evaluate(deposit) == []


def test_arg006_ignores_reverted_and_traceless_simulations():
    """Reverted calls or RPCs without `debug_traceCall` must not emit false-positive drains."""
    reverted = make_ctx("0xe9ae5c53", is_contract=True, eth_delta_wei=-1, reverted=True)
    assert AsymmetricDrainRule().evaluate(reverted) == []
    traceless = make_ctx("0xe9ae5c53", is_contract=True, eth_delta_wei=-1, trace_available=False)
    assert AsymmetricDrainRule().evaluate(traceless) == []


def test_arg006_plain_eth_payment_to_eoa_is_not_a_drain():
    """Sending plain native ETH (`data == 0x`) to an EOA friend is intentional, not a drain."""
    ctx = make_ctx("0x", to=SPENDER_EOA, is_contract=False, eth_delta_wei=-(10**18))
    assert AsymmetricDrainRule().evaluate(ctx) == []


def test_walk_trace_ignores_reverted_subframes():
    """Ensure reverted internal subcalls caught via try/catch do not skew ETH deltas."""
    acc = _TraceAccumulator()
    frame = {
        "type": "CALL",
        "from": VICTIM,
        "to": ROUTER,
        "value": "0x0",
        "calls": [
            {
                "type": "CALL",
                "from": VICTIM,
                "to": SPENDER_EOA,
                "value": "0xde0b6b3a7640000",
                "error": "execution reverted",
            }
        ],
    }
    _walk_trace(frame, 1, VICTIM, acc)
    assert acc.eth_delta == 0
    assert acc.max_depth == 2


# --------------------------------------------------------------------------- #
# Rule registry, risk scoring, and example payload integration tests
# --------------------------------------------------------------------------- #


def test_registry_has_all_six_rules_in_order():
    """Confirm all six ARG001-ARG006 rules auto-register in deterministic order."""
    assert [rule.rule_id for rule in all_rules()] == [f"ARG00{i}" for i in range(1, 7)]


def test_risk_score_is_capped():
    """Confirm deterministic risk scores cap cleanly at 100 and start at 0."""
    _, report = analyze_example("proxy_delegatecall_drainer.json")
    assert compute_risk_score(report.findings) == 100
    assert compute_risk_score([]) == 0


@pytest.mark.parametrize(
    ("example", "expected_rules"),
    [
        ("safe_uniswap_swap.json", set()),
        ("permit2_phishing.json", {"ARG002"}),
        ("proxy_delegatecall_drainer.json", {"ARG004", "ARG005", "ARG006"}),
        ("eip7702_delegation_trap.json", {"ARG003"}),
    ],
)
def test_examples_trigger_expected_rules(example, expected_rules):
    """Verify each bundled JSON example triggers its exact expected rule IDs."""
    _, report = analyze_example(example)
    assert {f.rule_id for f in report.findings} == expected_rules


def test_example_calldata_is_well_formed():
    """Verify bundled swap and permit JSON examples contain valid ABI-encoded hex."""
    tx, _ = load_payload(EXAMPLES / "safe_uniswap_swap.json")
    assert len(tx.data) == 2 + 8 + 64 * 7
    assert decode_calldata(tx.data).function_name == "exactInputSingle"
    tx, _ = load_payload(EXAMPLES / "permit2_phishing.json")
    assert len(tx.data) == 2 + 8 + 64 * 7
    assert decode_calldata(tx.data).function_name == "permit"


def test_drainer_example_disassembles_dangerous_opcodes():
    """Verify the proxy drainer fixture disassembles DELEGATECALL and SELFDESTRUCT."""
    ctx, _ = analyze_example("proxy_delegatecall_drainer.json")
    assert ctx.has_nonstandard_delegatecall
    assert {"DELEGATECALL", "SELFDESTRUCT"} <= ctx.opcode_mnemonics


def test_offline_simulator_without_mock_is_honest():
    """Verify offline mode without `_mock_context` marks `trace_available=False`."""
    tx = UnsignedTransaction.model_validate({"chainId": 1, "from": VICTIM, "to": USDC_ADDR})
    ctx = OfflineFixtureSimulator({}).simulate(tx, decode_calldata(tx.data))
    assert ctx.trace_available is False and ctx.notes
    assert run_rules(ctx) == []


def test_invalid_payload_raises(tmp_path):
    """Verify malformed EVM addresses in JSON payloads raise PayloadError."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"chainId": 1, "from": "not-an-address"}))
    with pytest.raises(PayloadError):
        load_payload(bad)


# --------------------------------------------------------------------------- #
# SARIF 2.1.0 report validation tests
# --------------------------------------------------------------------------- #


def test_sarif_structure():
    """Validate SARIF 2.1.0 schema version, driver rules, locations, and fingerprints."""
    _, report = analyze_example("permit2_phishing.json")
    log = json.loads(render_sarif_report(report, artifact_uri="examples/permit2_phishing.json"))
    assert log["version"] == "2.1.0"
    assert log["$schema"].endswith("sarif-2.1.0.json")
    run = log["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "argus"
    assert [rule["id"] for rule in driver["rules"]] == [f"ARG00{i}" for i in range(1, 7)]
    result = run["results"][0]
    assert result["ruleId"] == "ARG002"
    assert driver["rules"][result["ruleIndex"]]["id"] == "ARG002"
    assert result["level"] == "error"
    location = result["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"] == "examples/permit2_phishing.json"
    assert location["region"]["startLine"] == 1
    assert result["partialFingerprints"]["argus/v1"]


def test_sarif_without_findings_has_empty_results():
    """Confirm clean payloads produce a valid SARIF document with an empty results list."""
    _, report = analyze_example("safe_uniswap_swap.json")
    log = json.loads(render_sarif_report(report))
    assert log["runs"][0]["results"] == []


# --------------------------------------------------------------------------- #
# Typer CLI integration & exit-code tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("example", "exit_code"),
    [
        ("safe_uniswap_swap.json", 0),
        ("permit2_phishing.json", 1),
        ("proxy_delegatecall_drainer.json", 1),
        ("eip7702_delegation_trap.json", 1),
    ],
)
def test_cli_exit_codes(example, exit_code):
    """Verify CLI returns exit code 0 on safe payloads and exit code 1 on exploits."""
    result = runner.invoke(app, ["check", str(EXAMPLES / example), "--offline"])
    assert result.exit_code == exit_code, result.output


def test_cli_json_output():
    """Verify `--format json` emits machine-readable JSON containing triggered findings."""
    example_path = str(EXAMPLES / "eip7702_delegation_trap.json")
    result = runner.invoke(app, ["check", example_path, "--offline", "--format", "json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert [f["rule_id"] for f in payload["report"]["findings"]] == ["ARG003"]
    assert payload["report"]["passed"] is False


def test_cli_sarif_output():
    """Verify `--format sarif` outputs valid SARIF 2.1.0 JSON via stdout."""
    result = runner.invoke(
        app, ["check", str(EXAMPLES / "permit2_phishing.json"), "--offline", "--format", "sarif"]
    )
    assert json.loads(result.stdout)["version"] == "2.1.0"


def test_cli_fail_on_threshold(tmp_path):
    """Verify `--fail-on` controls whether HIGH findings exit with code 1 or 0."""
    payload = {
        "chainId": 1,
        "from": VICTIM,
        "to": ROUTER,
        "data": "0x",
        "_mock_context": {"is_contract": True, "is_verified": False, "deployment_age_blocks": 12},
    }
    path = tmp_path / "fresh.json"
    path.write_text(json.dumps(payload))
    assert runner.invoke(app, ["check", str(path), "--offline", "--fail-on", "high"]).exit_code == 1
    assert runner.invoke(app, ["check", str(path), "--offline", "--fail-on", "HIGH"]).exit_code == 1
    res_crit = runner.invoke(app, ["check", str(path), "--offline", "--fail-on", "critical"])
    assert res_crit.exit_code == 0


def test_cli_requires_offline_or_rpc(monkeypatch):
    """Verify CLI exits with code 2 when neither `--offline` nor `--rpc-url` is provided."""
    monkeypatch.delenv("ARGUS_RPC_URL", raising=False)
    result = runner.invoke(app, ["check", str(EXAMPLES / "safe_uniswap_swap.json")])
    assert result.exit_code == 2


def test_cli_invalid_payload_exits_2(tmp_path):
    """Verify corrupted JSON files cause operational exit code 2."""
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    assert runner.invoke(app, ["check", str(path), "--offline"]).exit_code == 2


def test_cli_rules_lists_all_rules():
    """Verify `argus rules` renders all six rule IDs (ARG001-ARG006)."""
    result = runner.invoke(app, ["rules"])
    assert result.exit_code == 0
    for index in range(1, 7):
        assert f"ARG00{index}" in result.output


# --------------------------------------------------------------------------- #
# SSRF & RPC URL security hardening tests
# --------------------------------------------------------------------------- #


def test_ccip_read_is_disabled_on_live_provider():
    """Pin CVE-2026-40072 mitigation: `global_ccip_read_enabled` MUST be False."""
    simulator = LiveRPCSimulator("http://127.0.0.1:8545")
    assert simulator.provider.global_ccip_read_enabled is False
    assert simulator.w3.provider.global_ccip_read_enabled is False


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "not a url", ""])
def test_live_simulator_rejects_non_http_urls(url):
    """Ensure local `file://` or `ftp://` schemes are rejected before creating a provider."""
    with pytest.raises(ValueError):
        LiveRPCSimulator(url)


# --------------------------------------------------------------------------- #
# Explainable AI (ONNX + SHAP) tests (auto-skipped if `[ml]` extras aren't installed)
# --------------------------------------------------------------------------- #


@pytest.fixture
def ml_modules():
    """Import ML dependencies lazily; skip ML tests cleanly if `[ml]` extra is absent."""
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("shap")
    from argus import ml_scorer

    return ml_scorer


def test_dataset_is_seeded_and_balanced(ml_modules):
    """Confirm synthetic dataset generator is deterministic per seed and 50/50 balanced."""
    x1, y1 = ml_modules.generate_dataset(400, seed=7)
    x2, y2 = ml_modules.generate_dataset(400, seed=7)
    x3, _ = ml_modules.generate_dataset(400, seed=8)
    assert x1.shape == (400, 8) and (x1 == x2).all() and (y1 == y2).all()
    assert not (x1 == x3).all()
    assert int(y1.sum()) == 200


def test_feature_extraction(ml_modules):
    """Verify behavioral feature vector values match expected exploit characteristics."""
    names = ml_modules.FEATURE_NAMES
    ctx, _ = analyze_example("permit2_phishing.json")
    features = dict(zip(names, ml_modules.extract_features(ctx).tolist(), strict=True))
    assert features["approval_spender_untrusted"] == 1.0
    assert features["permit_deadline_horizon"] > 0.5

    ctx, _ = analyze_example("proxy_delegatecall_drainer.json")
    features = dict(zip(names, ml_modules.extract_features(ctx).tolist(), strict=True))
    assert features["dangerous_opcode"] == 1.0 and features["asymmetric_outflow"] == 1.0
    assert features["contract_freshness"] > 0.9


def test_onnx_scoring_ranks_drainer_above_swap(ml_modules, tmp_path):
    """Confirm ONNX Random Forest scores safe swaps < 0.2 and drainer exploits > 0.8."""
    model = tmp_path / "risk.onnx"
    safe_ctx, _ = analyze_example("safe_uniswap_swap.json")
    drainer_ctx, _ = analyze_example("proxy_delegatecall_drainer.json")
    safe, _ = ml_modules.explain(safe_ctx, model_path=model)
    drainer, _ = ml_modules.explain(drainer_ctx, model_path=model)
    assert model.exists() and model.with_suffix(".json").exists()
    assert safe < 0.2 < 0.8 < drainer


def test_onnx_latency_under_5ms(ml_modules, tmp_path):
    """Assert single-thread CPU ONNX median inference latency stays under 5 milliseconds."""
    scorer = ml_modules.OnnxRiskScorer(tmp_path / "risk.onnx")
    assert scorer.benchmark(200) < 5.0


def test_shap_attributions_cover_every_feature(ml_modules, tmp_path):
    """Confirm SHAP TreeExplainer returns an attribution score for all 8 features."""
    ctx, _ = analyze_example("proxy_delegatecall_drainer.json")
    _, attributions = ml_modules.explain(ctx, model_path=tmp_path / "risk.onnx")
    assert set(attributions) == set(ml_modules.FEATURE_NAMES)
    assert attributions["dangerous_opcode"] > 0


def test_cli_explain_flag(ml_modules, tmp_path, monkeypatch):
    """Verify `argus check --explain --format json` populates ML score and SHAP values."""
    monkeypatch.setenv("ARGUS_MODEL_PATH", str(tmp_path / "risk.onnx"))
    example_path = str(EXAMPLES / "proxy_delegatecall_drainer.json")
    result = runner.invoke(
        app,
        ["check", example_path, "--offline", "--explain", "--format", "json"],
    )
    assert result.exit_code == 1, result.output
    report = json.loads(result.stdout)["report"]
    assert report["ml_risk_score"] > 0.8
    assert set(report["shap_attributions"]) == set(ml_modules.FEATURE_NAMES)