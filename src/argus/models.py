"""Pydantic v2 data models shared by every argus component."""

import time
from enum import StrEnum
from typing import Annotated, Any

from eth_utils import is_address, to_normalized_address
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)


def _to_int(value: Any) -> int:
    """Convert integers, decimal strings, or 0x-prefixed hex strings into Python ints."""
    if isinstance(value, bool):
        raise ValueError("a boolean is not a valid integer quantity")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("0x", "-0x"):
            return 0
        if "0x" in text:
            return int(text, 16)
        return int(text, 10)
    raise ValueError(f"cannot interpret {type(value).__name__} as an integer")


def _normalize_address(value: Any) -> str:
    """Validate an EVM address (including EIP-55 checksum if mixed-case) and lowercase it."""
    if not isinstance(value, str) or not is_address(value):
        raise ValueError(f"invalid EVM address: {value!r}")
    return to_normalized_address(value)


def _normalize_hex(value: Any) -> str:
    """Normalize calldata or bytecode to a lowercase, even-length 0x hex string."""
    if not isinstance(value, str):
        raise ValueError("hex data must be a string")
    text = value.strip().lower()
    if text in ("", "0x"):
        return "0x"
    if not text.startswith("0x"):
        text = "0x" + text
    body = text[2:]
    if len(body) % 2 != 0:
        raise ValueError("hex data has an odd number of nibbles")
    bytes.fromhex(body)
    return "0x" + body


HexInt = Annotated[int, BeforeValidator(_to_int)]
Address = Annotated[str, BeforeValidator(_normalize_address)]
HexData = Annotated[str, BeforeValidator(_normalize_hex)]


class Severity(StrEnum):
    """Severity levels ordered from lowest impact to highest impact."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Return 0-based numeric rank (LOW=0, MEDIUM=1, HIGH=2, CRITICAL=3)."""
        return list(Severity).index(self)

    def meets(self, threshold: "Severity") -> bool:
        """Return True if this severity is greater than or equal to `threshold`."""
        return self.rank >= threshold.rank


class AuthorizationItem(BaseModel):
    """Single EIP-7702 authorization tuple delegating EOA code execution to `address`."""

    model_config = ConfigDict(extra="ignore")

    chain_id: HexInt = Field(validation_alias=AliasChoices("chainId", "chain_id"))
    address: Address
    nonce: HexInt = 0


class UnsignedTransaction(BaseModel):
    """Unsigned EVM transaction payload inspected prior to wallet signing."""

    model_config = ConfigDict(extra="ignore")

    chain_id: HexInt = Field(validation_alias=AliasChoices("chainId", "chain_id"))
    from_addr: Address = Field(validation_alias=AliasChoices("from", "from_addr"))
    to_addr: Address | None = Field(
        default=None, validation_alias=AliasChoices("to", "to_addr")
    )
    value: HexInt = 0
    data: HexData = Field(default="0x", validation_alias=AliasChoices("data", "input"))
    gas: HexInt | None = Field(
        default=None, validation_alias=AliasChoices("gas", "gasLimit")
    )
    authorization_list: list[AuthorizationItem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("authorizationList", "authorization_list"),
    )

    @property
    def is_contract_creation(self) -> bool:
        """Return True if this transaction deploys a new contract (`to` is null)."""
        return self.to_addr is None


class DecodedCall(BaseModel):
    """Structured result produced after decoding the 4-byte selector and ABI payload."""

    selector: str
    function_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    is_known_selector: bool


class AddressInfo(BaseModel):
    """On-chain code and source-verification status for a referenced EVM address."""

    is_contract: bool = False
    is_verified: bool = False


class SimulationContext(BaseModel):
    """Complete pre-sign inspection state combining static, bytecode, and trace data."""

    model_config = ConfigDict(extra="forbid")

    tx: UnsignedTransaction
    decoded_call: DecodedCall

    is_contract: bool = False
    is_verified: bool = False
    deployment_age_blocks: int | None = None
    opcodes: list[str] = Field(default_factory=list)
    has_nonstandard_delegatecall: bool = False

    address_metadata: dict[str, AddressInfo] = Field(default_factory=dict)
    block_timestamp: int | None = None

    eth_delta_wei: HexInt = 0
    erc20_deltas: dict[str, HexInt] = Field(default_factory=dict)
    approvals_granted: list[dict[str, Any]] = Field(default_factory=list)
    receipt_tokens: list[str] = Field(default_factory=list)
    call_depth: int = 0
    reverted: bool = False

    trace_available: bool = True
    notes: list[str] = Field(default_factory=list)

    @field_validator("address_metadata", mode="before")
    @classmethod
    def _lower_metadata_keys(cls, value: Any) -> Any:
        """Normalize all address keys in `address_metadata` to lowercase."""
        if isinstance(value, dict):
            return {str(k).lower(): v for k, v in value.items()}
        return value

    @field_validator("erc20_deltas", mode="before")
    @classmethod
    def _lower_delta_keys(cls, value: Any) -> Any:
        """Normalize all token contract addresses in `erc20_deltas` to lowercase."""
        if isinstance(value, dict):
            return {str(k).lower(): v for k, v in value.items()}
        return value

    def info(self, address: str | None) -> AddressInfo | None:
        """Look up AddressInfo for `address`, falling back to target fields if needed."""
        if not address:
            return None
        key = address.lower()
        if key in self.address_metadata:
            return self.address_metadata[key]
        if self.tx.to_addr and key == self.tx.to_addr:
            return AddressInfo(is_contract=self.is_contract, is_verified=self.is_verified)
        return None

    def is_eoa(self, address: str | None) -> bool:
        """Return True only when metadata confirms `address` has zero contract code."""
        found = self.info(address)
        return found is not None and not found.is_contract

    def is_unverified(self, address: str | None) -> bool:
        """Fail-safe check: return True if `address` is unknown, an EOA, or unverified."""
        found = self.info(address)
        return found is None or not found.is_verified

    @property
    def opcode_mnemonics(self) -> set[str]:
        """Extract unique opcode names without PUSH operands (e.g. {'PUSH1', 'SLOAD'})."""
        return {entry.split(" ", 1)[0] for entry in self.opcodes}

    @property
    def reference_time(self) -> int:
        """Return simulated block timestamp, or current wall-clock Unix time as fallback."""
        return self.block_timestamp if self.block_timestamp is not None else int(time.time())


class Finding(BaseModel):
    """Single security violation emitted by a deterministic ARGUS rule."""

    rule_id: str
    title: str
    severity: Severity
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    remediation: str


class LintReport(BaseModel):
    """Aggregated lint verdict combining deterministic findings and optional ML/SHAP output."""

    findings: list[Finding] = Field(default_factory=list)
    deterministic_risk_score: int = 0
    ml_risk_score: float | None = None
    shap_attributions: dict[str, float] | None = None
    passed: bool = True