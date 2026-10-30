"""Pydantic v2 data models shared by every argus component."""

# Standard library imports for timestamps, string enums, and type annotations
import time
from enum import StrEnum
from typing import Annotated, Any

# Ethereum address validation and normalization helpers from eth-utils
from eth_utils import is_address, to_normalized_address

# Pydantic v2 primitives for schema validation, field aliases, and coercion
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)

# --------------------------------------------------------------------------- #
# Coercion helpers (run before Pydantic's built-in type validation)
# --------------------------------------------------------------------------- #


def _to_int(value: Any) -> int:
    """Convert integers, decimal strings, or 0x-prefixed hex strings into Python ints."""
    # Reject booleans explicitly because `isinstance(True, int)` is True in Python
    if isinstance(value, bool):
        raise ValueError("a boolean is not a valid integer quantity")
    # Return plain Python integers directly without conversion
    if isinstance(value, int):
        return value
    # Parse string inputs formatted as JSON-RPC hex ("0x1a") or base-10 ("26")
    if isinstance(value, str):
        text = value.strip().lower()
        # Treat bare "0x" or "-0x" prefixes as zero
        if text in ("0x", "-0x"):
            return 0
        # Use base-16 parsing if a hex prefix is present
        if "0x" in text:
            return int(text, 16)
        # Otherwise parse as a standard base-10 decimal string
        return int(text, 10)
    # Reject any unsupported types (lists, dicts, floats, None)
    raise ValueError(f"cannot interpret {type(value).__name__} as an integer")


def _normalize_address(value: Any) -> str:
    """Validate an EVM address (including EIP-55 checksum if mixed-case) and lowercase it."""
    # Ensure input is a string and passes eth-utils address & checksum checks
    if not isinstance(value, str) or not is_address(value):
        raise ValueError(f"invalid EVM address: {value!r}")
    # Return canonical lowercase 0x-prefixed 40-hex-character address
    return to_normalized_address(value)


def _normalize_hex(value: Any) -> str:
    """Normalize calldata or bytecode to a lowercase, even-length 0x hex string."""
    # Require string input so arbitrary numbers or objects cannot slip through
    if not isinstance(value, str):
        raise ValueError("hex data must be a string")
    text = value.strip().lower()
    # Canonicalize empty strings or bare "0x" into empty calldata ("0x")
    if text in ("", "0x"):
        return "0x"
    # Ensure the hex string starts with "0x"
    if not text.startswith("0x"):
        text = "0x" + text
    # Strip "0x" temporarily to validate byte alignment and hex characters
    body = text[2:]
    if len(body) % 2 != 0:
        raise ValueError("hex data has an odd number of nibbles")
    # `bytes.fromhex` raises ValueError if any non-hex character is present
    bytes.fromhex(body)
    return "0x" + body


# Reusable Annotated types that attach our custom pre-validators to Pydantic fields
HexInt = Annotated[int, BeforeValidator(_to_int)]
Address = Annotated[str, BeforeValidator(_normalize_address)]
HexData = Annotated[str, BeforeValidator(_normalize_hex)]


# --------------------------------------------------------------------------- #
# Severity enumeration (uses Python 3.11+ StrEnum to satisfy Ruff UP042)
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Unsigned transaction & decoded calldata models
# --------------------------------------------------------------------------- #


class AuthorizationItem(BaseModel):
    """Single EIP-7702 authorization tuple delegating EOA code execution to `address`."""

    # Ignore extra signature fields like yParity/r/s that wallets may include
    model_config = ConfigDict(extra="ignore")

    # Chain ID where delegation is valid (0 means replayable across ALL EVM chains!)
    chain_id: HexInt = Field(validation_alias=AliasChoices("chainId", "chain_id"))
    # Smart contract implementation address receiving EOA execution control
    address: Address
    # Account nonce bound into the authorization tuple
    nonce: HexInt = 0


class UnsignedTransaction(BaseModel):
    """Unsigned EVM transaction payload inspected prior to wallet signing."""

    # Ignore non-security metadata fields (such as `_description` or `_mock_context`)
    model_config = ConfigDict(extra="ignore")

    # Target EVM network chain ID (accepts camelCase `chainId` or snake_case `chain_id`)
    chain_id: HexInt = Field(validation_alias=AliasChoices("chainId", "chain_id"))
    # Sender wallet address initiating the transaction
    from_addr: Address = Field(validation_alias=AliasChoices("from", "from_addr"))
    # Target contract/recipient address (`None` indicates contract deployment)
    to_addr: Address | None = Field(
        default=None, validation_alias=AliasChoices("to", "to_addr")
    )
    # Native ETH value sent with the call in wei
    value: HexInt = 0
    # Raw ABI-encoded hex calldata (accepts `data` or JSON-RPC `input` alias)
    data: HexData = Field(default="0x", validation_alias=AliasChoices("data", "input"))
    # Optional gas limit supplied by the caller
    gas: HexInt | None = Field(
        default=None, validation_alias=AliasChoices("gas", "gasLimit")
    )
    # Optional EIP-7702 type-4 transaction delegation list
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

    # 4-byte hex function selector (e.g. "0x095ea7b3"), or "0x" for empty calldata
    selector: str
    # Human-readable function name (e.g. "approve", "<native transfer>", "<unknown>")
    function_name: str
    # Dictionary mapping parameter names to decoded Python values
    arguments: dict[str, Any] = Field(default_factory=dict)
    # False when non-empty calldata uses an unrecognized 4-byte selector
    is_known_selector: bool


# --------------------------------------------------------------------------- #
# Simulation context (fed into deterministic rules and the ONNX/SHAP scorer)
# --------------------------------------------------------------------------- #


class AddressInfo(BaseModel):
    """On-chain code and source-verification status for a referenced EVM address."""

    # True if `eth_getCode` returned non-empty bytecode for this address
    is_contract: bool = False
    # True if source code is verified on an explorer or allowlisted by the user
    is_verified: bool = False


class SimulationContext(BaseModel):
    """Complete pre-sign inspection state combining static, bytecode, and trace data."""

    # Reject unknown keys in `_mock_context` so typos fail fast during testing
    model_config = ConfigDict(extra="forbid")

    # Parsed unsigned transaction and its ABI-decoded call arguments
    tx: UnsignedTransaction
    decoded_call: DecodedCall

    # Target contract telemetry
    is_contract: bool = False
    is_verified: bool = False
    deployment_age_blocks: int | None = None
    opcodes: list[str] = Field(default_factory=list)
    has_nonstandard_delegatecall: bool = False

    # Per-address lookup table (keys normalized to lowercase hex addresses)
    address_metadata: dict[str, AddressInfo] = Field(default_factory=dict)
    block_timestamp: int | None = None

    # Simulated balance changes and side effects from the sender's perspective
    eth_delta_wei: HexInt = 0
    erc20_deltas: dict[str, HexInt] = Field(default_factory=dict)
    approvals_granted: list[dict[str, Any]] = Field(default_factory=list)
    receipt_tokens: list[str] = Field(default_factory=list)
    call_depth: int = 0
    reverted: bool = False

    # Flag indicating whether a call trace was available (prevents false guesses)
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


# --------------------------------------------------------------------------- #
# Rule findings & final lint report models
# --------------------------------------------------------------------------- #


class Finding(BaseModel):
    """Single security violation emitted by a deterministic TXL rule."""

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