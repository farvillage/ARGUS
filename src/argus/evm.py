"""EVM helpers: calldata decoding, pure-Python disassembly, and transaction simulators.

Design notes
------------
* No archived or unmaintained packages: decoding uses ``eth-abi``, disassembly is plain Python.
* ``OfflineFixtureSimulator`` is fully deterministic and needs no network or RPC key.
* ``LiveRPCSimulator`` hardens ``web3.py`` against ``OffchainLookup`` (EIP-3668 / CCIP-Read) SSRF.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from eth_abi import decode as abi_decode
from eth_abi.exceptions import DecodingError
from eth_utils import function_signature_to_4byte_selector, keccak
from pydantic import ValidationError
from web3 import Web3

from argus.models import (
    AddressInfo,
    DecodedCall,
    SimulationContext,
    UnsignedTransaction,
)


class PayloadError(ValueError):
    """Raised when a payload file cannot be read or is not a valid unsigned transaction."""


def load_payload(path: Path | str) -> tuple[UnsignedTransaction, dict[str, Any]]:
    """Load a JSON payload. Returns the parsed transaction and the raw JSON object."""
    file_path = Path(path)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PayloadError(f"cannot read {file_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PayloadError(f"{file_path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PayloadError(f"{file_path} must contain a JSON object")

    tx_obj = raw["transaction"] if isinstance(raw.get("transaction"), dict) else raw
    try:
        tx = UnsignedTransaction.model_validate(tx_obj)
    except ValidationError as exc:
        raise PayloadError(f"invalid unsigned transaction in {file_path}:\n{exc}") from exc
    return tx, raw


def _hex_to_bytes(value: str | bytes) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    text = value.strip()
    if text[:2].lower() == "0x":
        text = text[2:]
    if not text:
        return b""
    try:
        return bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"invalid hex string: {exc}") from exc


def _hex_to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in ("", "0x"):
        return 0
    return int(text, 16) if text.startswith("0x") else int(text, 10)


@dataclass(frozen=True)
class FunctionSpec:
    signature: str
    name: str
    abi_types: tuple[str, ...]
    arg_names: tuple[str, ...]
    struct_fields: tuple[str, ...] | None = None

    @property
    def selector(self) -> str:
        return "0x" + function_signature_to_4byte_selector(self.signature).hex()


KNOWN_FUNCTIONS: tuple[FunctionSpec, ...] = (
    FunctionSpec(
        "approve(address,uint256)",
        "approve",
        ("address", "uint256"),
        ("spender", "amount"),
    ),
    FunctionSpec(
        "setApprovalForAll(address,bool)",
        "setApprovalForAll",
        ("address", "bool"),
        ("operator", "approved"),
    ),
    FunctionSpec(
        "permit(address,address,uint256,uint256,uint8,bytes32,bytes32)",
        "permit",
        ("address", "address", "uint256", "uint256", "uint8", "bytes32", "bytes32"),
        ("owner", "spender", "value", "deadline", "v", "r", "s"),
    ),
    FunctionSpec(
        "transferFrom(address,address,uint256)",
        "transferFrom",
        ("address", "address", "uint256"),
        ("from", "to", "amount"),
    ),
    FunctionSpec(
        "exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))",
        "exactInputSingle",
        ("(address,address,uint24,address,uint256,uint256,uint256,uint160)",),
        ("params",),
        struct_fields=(
            "tokenIn",
            "tokenOut",
            "fee",
            "recipient",
            "deadline",
            "amountIn",
            "amountOutMinimum",
            "sqrtPriceLimitX96",
        ),
    ),
    FunctionSpec(
        "exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))",
        "exactInputSingle",
        ("(address,address,uint24,address,uint256,uint256,uint160)",),
        ("params",),
        struct_fields=(
            "tokenIn",
            "tokenOut",
            "fee",
            "recipient",
            "amountIn",
            "amountOutMinimum",
            "sqrtPriceLimitX96",
        ),
    ),
)

_SPEC_BY_SELECTOR: dict[str, FunctionSpec] = {spec.selector: spec for spec in KNOWN_FUNCTIONS}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return "0x" + bytes(value).hex()
    if isinstance(value, str) and value.startswith("0x") and len(value) == 42:
        return value.lower()
    return value


def decode_calldata(data: str) -> DecodedCall:
    """Decode calldata using ``eth-abi``. Unknown non-empty selectors are flagged."""
    raw = _hex_to_bytes(data)
    if not raw:
        return DecodedCall(
            selector="0x", function_name="<native transfer>", arguments={}, is_known_selector=True
        )
    selector = "0x" + raw[:4].hex()
    spec = _SPEC_BY_SELECTOR.get(selector) if len(raw) >= 4 else None
    if spec is None:
        return DecodedCall(
            selector=selector,
            function_name="<unknown>",
            arguments={"calldata_bytes": len(raw)},
            is_known_selector=False,
        )
    try:
        values = abi_decode(list(spec.abi_types), raw[4:])
    except (DecodingError, ValueError) as exc:
        return DecodedCall(
            selector=selector,
            function_name=spec.name,
            arguments={"decode_error": str(exc)},
            is_known_selector=True,
        )

    if spec.struct_fields is not None:
        arguments = {
            name: _normalize_value(item)
            for name, item in zip(spec.struct_fields, values[0], strict=True)
        }
    else:
        arguments = {
            name: _normalize_value(item) for name, item in zip(spec.arg_names, values, strict=True)
        }
    return DecodedCall(
        selector=selector, function_name=spec.name, arguments=arguments, is_known_selector=True
    )


_ADDRESS_ARG_NAMES = ("spender", "operator", "recipient", "to")


def addresses_of_interest(tx: UnsignedTransaction, decoded: DecodedCall) -> list[str]:
    """Addresses worth inspecting on-chain: target, counterparties, EIP-7702 delegates."""
    found: list[str] = []
    if tx.to_addr:
        found.append(tx.to_addr)
    for name in _ADDRESS_ARG_NAMES:
        candidate = decoded.arguments.get(name)
        if isinstance(candidate, str) and len(candidate) == 42:
            found.append(candidate.lower())
    found.extend(item.address for item in tx.authorization_list)
    return list(dict.fromkeys(found))


EIP1967_IMPLEMENTATION_SLOT = (
    "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
)


def _build_opcode_table() -> dict[int, str]:
    table: dict[int, str] = {}
    base = """
    00 STOP|01 ADD|02 MUL|03 SUB|04 DIV|05 SDIV|06 MOD|07 SMOD|08 ADDMOD|09 MULMOD|0a EXP
    0b SIGNEXTEND|10 LT|11 GT|12 SLT|13 SGT|14 EQ|15 ISZERO|16 AND|17 OR|18 XOR|19 NOT|1a BYTE
    1b SHL|1c SHR|1d SAR|20 KECCAK256|30 ADDRESS|31 BALANCE|32 ORIGIN|33 CALLER|34 CALLVALUE
    35 CALLDATALOAD|36 CALLDATASIZE|37 CALLDATACOPY|38 CODESIZE|39 CODECOPY|3a GASPRICE
    3b EXTCODESIZE|3c EXTCODECOPY|3d RETURNDATASIZE|3e RETURNDATACOPY|3f EXTCODEHASH
    40 BLOCKHASH|41 COINBASE|42 TIMESTAMP|43 NUMBER|44 PREVRANDAO|45 GASLIMIT|46 CHAINID
    47 SELFBALANCE|48 BASEFEE|49 BLOBHASH|4a BLOBBASEFEE|50 POP|51 MLOAD|52 MSTORE|53 MSTORE8
    54 SLOAD|55 SSTORE|56 JUMP|57 JUMPI|58 PC|59 MSIZE|5a GAS|5b JUMPDEST|5c TLOAD|5d TSTORE
    5e MCOPY|5f PUSH0|f0 CREATE|f1 CALL|f2 CALLCODE|f3 RETURN|f4 DELEGATECALL|f5 CREATE2
    fa STATICCALL|fd REVERT|fe INVALID|ff SELFDESTRUCT
    """
    for chunk in base.replace("\n", "|").split("|"):
        chunk = chunk.strip()
        if chunk:
            code, name = chunk.split()
            table[int(code, 16)] = name
    for i in range(32):
        table[0x60 + i] = f"PUSH{i + 1}"
    for i in range(16):
        table[0x80 + i] = f"DUP{i + 1}"
        table[0x90 + i] = f"SWAP{i + 1}"
    for i in range(5):
        table[0xA0 + i] = f"LOG{i}"
    return table


OPCODE_NAMES: dict[int, str] = _build_opcode_table()


def disassemble_opcodes(bytecode_hex: str) -> list[str]:
    """Disassemble EVM bytecode while skipping PUSH1..PUSH32 immediate operand bytes."""
    code = _hex_to_bytes(bytecode_hex)
    opcodes: list[str] = []
    pc = 0
    length = len(code)
    while pc < length:
        op = code[pc]
        name = OPCODE_NAMES.get(op, f"UNKNOWN_0x{op:02x}")
        if 0x60 <= op <= 0x7F:
            size = op - 0x5F
            operand = code[pc + 1 : pc + 1 + size]
            opcodes.append(f"{name} 0x{operand.hex()}" if operand else name)
            pc += 1 + size
        else:
            opcodes.append(name)
            pc += 1
    return opcodes


@dataclass(frozen=True)
class BytecodeAnalysis:
    opcodes: list[str]
    has_delegatecall: bool
    has_create2: bool
    has_selfdestruct: bool
    uses_eip1967_slot: bool
    has_nonstandard_delegatecall: bool


def analyze_bytecode(bytecode_hex: str) -> BytecodeAnalysis:
    """Disassemble and classify bytecode for non-standard DELEGATECALL and SELFDESTRUCT."""
    opcodes = disassemble_opcodes(bytecode_hex)
    names = {entry.split(" ", 1)[0] for entry in opcodes}
    uses_slot = f"PUSH32 {EIP1967_IMPLEMENTATION_SLOT}" in opcodes
    has_delegatecall = "DELEGATECALL" in names
    return BytecodeAnalysis(
        opcodes=opcodes,
        has_delegatecall=has_delegatecall,
        has_create2="CREATE2" in names,
        has_selfdestruct="SELFDESTRUCT" in names,
        uses_eip1967_slot=uses_slot,
        has_nonstandard_delegatecall=has_delegatecall and not uses_slot,
    )


class Simulator(Protocol):
    def simulate(self, tx: UnsignedTransaction, decoded: DecodedCall) -> SimulationContext: ...


class OfflineFixtureSimulator:
    """Deterministic simulator driven by a ``"_mock_context"`` object inside the payload JSON."""

    def __init__(self, fixture: Mapping[str, Any]) -> None:
        mock = fixture.get("_mock_context")
        self._mock: dict[str, Any] | None = dict(mock) if isinstance(mock, Mapping) else None

    def simulate(self, tx: UnsignedTransaction, decoded: DecodedCall) -> SimulationContext:
        if self._mock is None:
            return SimulationContext(
                tx=tx,
                decoded_call=decoded,
                trace_available=False,
                notes=["No `_mock_context` in payload: target metadata and state diff unknown."],
            )
        data = dict(self._mock)
        data.pop("tx", None)
        data.pop("decoded_call", None)
        bytecode = data.pop("runtime_bytecode", None)
        if bytecode:
            analysis = analyze_bytecode(bytecode)
            data.setdefault("opcodes", analysis.opcodes)
            data.setdefault("has_nonstandard_delegatecall", analysis.has_nonstandard_delegatecall)
        return SimulationContext(tx=tx, decoded_call=decoded, **data)


_TRANSFER_TOPIC = "0x" + keccak(text="Transfer(address,address,uint256)").hex()
_APPROVAL_TOPIC = "0x" + keccak(text="Approval(address,address,uint256)").hex()
_APPROVAL_FOR_ALL_TOPIC = "0x" + keccak(text="ApprovalForAll(address,address,bool)").hex()
_VALUE_FRAME_TYPES = {"CALL", "CREATE", "CREATE2", "SELFDESTRUCT"}


def _topic_address(topic: str) -> str:
    return "0x" + topic[-40:].lower()


@dataclass
class _TraceAccumulator:
    max_depth: int = 0
    eth_delta: int = 0
    erc20: dict[str, int] = field(default_factory=dict)
    approvals: list[dict[str, Any]] = field(default_factory=list)


def _apply_log(log: Mapping[str, Any], sender: str, acc: _TraceAccumulator) -> None:
    topics = [str(topic).lower() for topic in log.get("topics") or []]
    if not topics:
        return
    token = str(log.get("address", "")).lower()
    data_int = _hex_to_int(log.get("data"))
    head = topics[0]
    if head == _TRANSFER_TOPIC and len(topics) in (3, 4):
        amount = data_int if len(topics) == 3 else 1
        if _topic_address(topics[1]) == sender:
            acc.erc20[token] = acc.erc20.get(token, 0) - amount
        if _topic_address(topics[2]) == sender:
            acc.erc20[token] = acc.erc20.get(token, 0) + amount
    elif head == _APPROVAL_TOPIC and len(topics) in (3, 4):
        if _topic_address(topics[1]) == sender:
            acc.approvals.append(
                {
                    "standard": "ERC-20" if len(topics) == 3 else "ERC-721",
                    "token": token,
                    "spender": _topic_address(topics[2]),
                    "amount": str(data_int) if len(topics) == 3 else "token-id",
                }
            )
    elif head == _APPROVAL_FOR_ALL_TOPIC and len(topics) == 3:
        if _topic_address(topics[1]) == sender and data_int:
            acc.approvals.append(
                {
                    "standard": "ERC-721/1155",
                    "token": token,
                    "operator": _topic_address(topics[2]),
                    "approved": True,
                }
            )


def _walk_trace(
    frame: Mapping[str, Any], depth: int, sender: str, acc: _TraceAccumulator
) -> None:
    acc.max_depth = max(acc.max_depth, depth)
    if frame.get("error"):
        return
    source = str(frame.get("from", "")).lower()
    dest = str(frame.get("to", "")).lower()
    value = _hex_to_int(frame.get("value"))
    if value and frame.get("type") in _VALUE_FRAME_TYPES:
        if source == sender:
            acc.eth_delta -= value
        if dest == sender:
            acc.eth_delta += value
    for log in frame.get("logs") or []:
        _apply_log(log, sender, acc)
    for child in frame.get("calls") or []:
        _walk_trace(child, depth + 1, sender, acc)


class LiveRPCSimulator:
    """Simulates a transaction against a real node (read-only RPC calls only).

    ``provider.global_ccip_read_enabled`` is forced to ``False`` to block EIP-3668
    ``OffchainLookup`` SSRF (CVE-2026-40072 / CVE-2026-111171).
    """

    def __init__(
        self,
        rpc_url: str,
        *,
        timeout: float = 15.0,
        verified_addresses: Iterable[str] | None = None,
    ) -> None:
        parsed = urlparse(rpc_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("rpc_url must be an absolute http(s) URL")

        provider = Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout})
        provider.global_ccip_read_enabled = False
        if provider.global_ccip_read_enabled is not False:  # pragma: no cover
            raise RuntimeError("failed to disable CCIP-Read on the HTTP provider")
        self.provider = provider
        self.w3 = Web3(provider)

        allow: set[str] = set()
        env_value = os.environ.get("ARGUS_VERIFIED_ADDRESSES", "")
        for item in [*(verified_addresses or []), *env_value.split(",")]:
            item = item.strip().lower()
            if item:
                allow.add(item)
        self._verified = allow

    @staticmethod
    def _web3_params(tx: UnsignedTransaction) -> dict[str, Any]:
        params: dict[str, Any] = {
            "from": Web3.to_checksum_address(tx.from_addr),
            "value": tx.value,
            "data": tx.data,
        }
        if tx.to_addr:
            params["to"] = Web3.to_checksum_address(tx.to_addr)
        if tx.gas is not None:
            params["gas"] = tx.gas
        return params

    @staticmethod
    def _rpc_params(tx: UnsignedTransaction) -> dict[str, Any]:
        params: dict[str, Any] = {
            "from": Web3.to_checksum_address(tx.from_addr),
            "value": hex(tx.value),
            "data": tx.data,
        }
        if tx.to_addr:
            params["to"] = Web3.to_checksum_address(tx.to_addr)
        if tx.gas is not None:
            params["gas"] = hex(tx.gas)
        return params

    def _address_info(self, address: str) -> AddressInfo:
        code = bytes(self.w3.eth.get_code(Web3.to_checksum_address(address)))
        return AddressInfo(
            is_contract=len(code) > 0,
            is_verified=address.lower() in self._verified,
        )

    def simulate(self, tx: UnsignedTransaction, decoded: DecodedCall) -> SimulationContext:
        notes: list[str] = []
        latest = self.w3.eth.get_block("latest")
        block_timestamp = int(latest["timestamp"])

        metadata: dict[str, AddressInfo] = {}
        for address in addresses_of_interest(tx, decoded):
            metadata[address] = self._address_info(address)

        if tx.to_addr:
            target_info = metadata[tx.to_addr]
            code = bytes(self.w3.eth.get_code(Web3.to_checksum_address(tx.to_addr)))
            analysis = analyze_bytecode(code.hex())
        else:
            target_info = AddressInfo(is_contract=False, is_verified=False)
            analysis = analyze_bytecode(tx.data)
            notes.append("Contract creation: opcodes taken from init code in `data`.")

        if not self._verified:
            notes.append(
                "Source verification is unknown over RPC; set ARGUS_VERIFIED_ADDRESSES to "
                "mark known-good contracts as verified."
            )
        notes.append("deployment_age_blocks is unknown over plain RPC (needs archive/explorer).")

        reverted = False
        try:
            self.w3.eth.call(self._web3_params(tx), "latest", ccip_read_enabled=False)
        except Exception as exc:
            reverted = True
            notes.append(f"eth_call failed ({type(exc).__name__}): {str(exc)[:200]}")

        acc = _TraceAccumulator()
        trace_available = True
        try:
            response = self.provider.make_request(
                "debug_traceCall",
                [
                    self._rpc_params(tx),
                    "latest",
                    {"tracer": "callTracer", "tracerConfig": {"withLog": True}},
                ],
            )
            if "error" in response or "result" not in response:
                trace_available = False
                notes.append(f"debug_traceCall unavailable: {str(response.get('error'))[:200]}")
            else:
                _walk_trace(response["result"], 1, tx.from_addr, acc)
        except Exception as exc:
            trace_available = False
            notes.append(f"debug_traceCall failed ({type(exc).__name__}): {str(exc)[:200]}")

        return SimulationContext(
            tx=tx,
            decoded_call=decoded,
            is_contract=target_info.is_contract,
            is_verified=target_info.is_verified,
            deployment_age_blocks=None,
            opcodes=analysis.opcodes,
            has_nonstandard_delegatecall=analysis.has_nonstandard_delegatecall,
            address_metadata=metadata,
            block_timestamp=block_timestamp,
            eth_delta_wei=acc.eth_delta,
            erc20_deltas=acc.erc20,
            approvals_granted=acc.approvals,
            call_depth=acc.max_depth,
            reverted=reverted,
            trace_available=trace_available,
            notes=notes,
        )