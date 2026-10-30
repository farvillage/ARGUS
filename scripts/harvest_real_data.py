#!/usr/bin/env python3
"""Harvest real Ethereum transactions into ARGUS 8-feature vectors (RPC + Etherscan V2 archive).

Uses public RPCs (`publicnode` + `drpc` archive) and automatically falls back to Etherscan V2's
`module=proxy` archive endpoints for older historical exploit blocks pruned under EIP-4444.

Usage:
    python3 scripts/harvest_real_data.py --seeds data/chronicle_seeds.json
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402
from eth_utils import keccak  # noqa: E402
from web3 import Web3  # noqa: E402

from argus.evm import addresses_of_interest, analyze_bytecode, decode_calldata  # noqa: E402
from argus.ml_scorer import FEATURE_NAMES, extract_features  # noqa: E402
from argus.models import AddressInfo, SimulationContext, UnsignedTransaction  # noqa: E402

ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"
DEFAULT_FREE_RPC = "https://ethereum-rpc.publicnode.com"
ARCHIVE_FALLBACK_RPC = "https://eth.drpc.org"

BENIGN_ROUTERS: tuple[str, ...] = (
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap SwapRouter02
    "0x66a9893cc07d91d95644aedd05d03f95e1dba8af",  # Uniswap UniversalRouter
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 SwapRouter
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH9
)

TRANSFER_TOPIC = "0x" + keccak(text="Transfer(address,address,uint256)").hex()
APPROVAL_TOPIC = "0x" + keccak(text="Approval(address,address,uint256)").hex()
APPROVAL_ALL_TOPIC = "0x" + keccak(text="ApprovalForAll(address,address,bool)").hex()


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from local `.env` using stdlib only."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


def _make_w3(rpc_url: str, timeout: float = 15.0) -> Web3:
    provider = Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout})
    provider.global_ccip_read_enabled = False
    return Web3(provider)


def _topic_addr(topic_hex: str) -> str:
    return "0x" + topic_hex[-40:].lower()


def _pad_topic_addr(address: str) -> str:
    return "0x" + address.lower().removeprefix("0x").rjust(64, "0")


def _hex_int(raw: Any) -> int:
    if raw is None:
        return 0
    if isinstance(raw, (bytes, bytearray)):
        return int.from_bytes(raw, "big") if raw else 0
    if isinstance(raw, int):
        return raw
    text = str(raw).strip().lower()
    if text in {"", "0x"}:
        return 0
    return int(text, 16) if text.startswith("0x") else int(text, 10)


def _hex_str(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return "0x" + bytes(raw).hex()
    text = str(raw or "0x").strip()
    return text if text.startswith("0x") else "0x" + text


class EtherscanV2Client:
    """Rate-limited (<3 req/s) Etherscan V2 client with archive JSON-RPC proxy fallback."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self._verified_cache: dict[str, bool] = {}
        self._creation_cache: dict[str, int | None] = {}
        self.tx_meta_cache: dict[str, dict[str, Any]] = {}
        self._last_call = 0.0

    def _get(self, params: dict[str, str | int]) -> Any:
        query = {"chainid": 1, **params, "apikey": self.api_key}
        for attempt in range(1, 4):
            elapsed = time.monotonic() - self._last_call
            if elapsed < 0.38:
                time.sleep(0.38 - elapsed)
            body: dict[str, Any] | None = None
            try:
                resp = self.session.get(ETHERSCAN_V2_URL, params=query, timeout=25.0)
                self._last_call = time.monotonic()
                resp.raise_for_status()
                body = resp.json()
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(1.5 * attempt)

            if body is not None:
                result = body.get("result")
                if isinstance(result, str) and "rate limit" in result.lower():
                    time.sleep(1.5 * attempt)
                else:
                    return result
        return None

    def is_verified(self, address: str) -> bool:
        key = address.lower()
        if key in self._verified_cache:
            return self._verified_cache[key]
        result = self._get({"module": "contract", "action": "getsourcecode", "address": key})
        verified = False
        if isinstance(result, list) and result and isinstance(result[0], dict):
            verified = bool(str(result[0].get("SourceCode", "")).strip())
        self._verified_cache[key] = verified
        return verified

    def creation_block(self, address: str) -> int | None:
        key = address.lower()
        if key in self._creation_cache:
            return self._creation_cache[key]
        result = self._get(
            {"module": "contract", "action": "getcontractcreation", "contractaddresses": key}
        )
        block_num: int | None = None
        if isinstance(result, list) and result and isinstance(result[0], dict):
            raw_blk = result[0].get("blockNumber")
            if raw_blk and str(raw_blk).isdigit():
                block_num = int(raw_blk)
        self._creation_cache[key] = block_num
        return block_num

    def tx_hashes_for_address(
        self,
        address: str,
        *,
        limit: int = 20,
        inbound_only: bool = False,
        require_calldata: bool = True,
    ) -> list[str]:
        result = self._get(
            {
                "module": "account",
                "action": "txlist",
                "address": address.lower(),
                "page": 1,
                "offset": max(limit * 3, 30),
                "sort": "desc",
            }
        )
        if not isinstance(result, list):
            return []
        hashes: list[str] = []
        target = address.lower()
        for item in result:
            if not isinstance(item, dict) or str(item.get("isError")) != "0":
                continue
            to_addr = str(item.get("to") or "").lower()
            if inbound_only and to_addr != target:
                continue
            calldata = str(item.get("input") or "0x").lower()
            if require_calldata and calldata in {"", "0x"} and to_addr:
                continue
            tx_h = str(item.get("hash") or "").lower()
            if len(tx_h) == 66 and tx_h.startswith("0x"):
                self.tx_meta_cache[tx_h] = item
                hashes.append(tx_h)
            if len(hashes) >= limit:
                break
        return hashes

    def victim_approval_tx_hashes(self, spender: str, *, limit: int = 12) -> list[str]:
        """Find real mainnet tx hashes where victims granted ERC-20 Approval to `spender`."""
        result = self._get(
            {
                "module": "logs",
                "action": "getLogs",
                "fromBlock": 0,
                "toBlock": "latest",
                "topic0": APPROVAL_TOPIC,
                "topic0_2_opr": "and",
                "topic2": _pad_topic_addr(spender),
                "page": 1,
                "offset": max(limit * 2, 15),
            }
        )
        if not isinstance(result, list):
            return []
        found: list[str] = []
        for entry in result:
            if not isinstance(entry, dict):
                continue
            tx_h = str(entry.get("transactionHash") or "").lower()
            if len(tx_h) == 66 and tx_h.startswith("0x") and tx_h not in found:
                found.append(tx_h)
            if len(found) >= limit:
                break
        return found

    def proxy_get_tx(self, tx_hash: str) -> dict[str, Any] | None:
        res = self._get(
            {"module": "proxy", "action": "eth_getTransactionByHash", "txhash": tx_hash}
        )
        return res if isinstance(res, dict) else None

    def proxy_get_receipt(self, tx_hash: str) -> dict[str, Any] | None:
        res = self._get(
            {"module": "proxy", "action": "eth_getTransactionReceipt", "txhash": tx_hash}
        )
        return res if isinstance(res, dict) else None

    def proxy_get_block_ts(self, block_num: int) -> int:
        res = self._get(
            {
                "module": "proxy",
                "action": "eth_getBlockByNumber",
                "tag": hex(block_num),
                "boolean": "false",
            }
        )
        if isinstance(res, dict) and res.get("timestamp"):
            return _hex_int(res["timestamp"])
        return int(time.time())

    def proxy_get_code(self, address: str) -> bytes:
        res = self._get(
            {"module": "proxy", "action": "eth_getCode", "address": address, "tag": "latest"}
        )
        if isinstance(res, str) and res.startswith("0x") and len(res) > 2:
            return bytes.fromhex(res[2:])
        return b""


def _fetch_tx_and_receipt(
    w3_list: list[Web3],
    explorer: EtherscanV2Client,
    tx_hash: str,
    block_ts_cache: dict[int, int],
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Fetch tx, receipt, block_number, and timestamp across RPCs + Etherscan archive proxy."""
    cached_meta = explorer.tx_meta_cache.get(tx_hash)
    raw_tx: dict[str, Any] | None = cached_meta
    receipt: dict[str, Any] | None = None

    for w3 in w3_list:
        try:
            if raw_tx is None:
                raw_tx = dict(w3.eth.get_transaction(tx_hash))
            if receipt is None:
                receipt = dict(w3.eth.get_transaction_receipt(tx_hash))
            if raw_tx is not None and receipt is not None:
                break
        except Exception as exc:
            _ = exc

    if raw_tx is None:
        raw_tx = explorer.proxy_get_tx(tx_hash)
    if receipt is None:
        receipt = explorer.proxy_get_receipt(tx_hash)

    if raw_tx is None or receipt is None:
        raise LookupError("not found on Ethereum Mainnet (likely L2/alt-chain tx)")

    block_num = _hex_int(receipt.get("blockNumber") or raw_tx.get("blockNumber"))
    if cached_meta and str(cached_meta.get("timeStamp", "")).isdigit():
        block_ts_cache[block_num] = int(cached_meta["timeStamp"])
    elif block_num not in block_ts_cache:
        fetched_ts: int | None = None
        for w3 in w3_list:
            try:
                blk = w3.eth.get_block(block_num)
                fetched_ts = int(blk["timestamp"])
                break
            except Exception as exc:
                _ = exc
        block_ts_cache[block_num] = (
            fetched_ts if fetched_ts is not None else explorer.proxy_get_block_ts(block_num)
        )

    return raw_tx, receipt, block_num, block_ts_cache[block_num]


def build_context_from_receipt(
    w3_list: list[Web3],
    explorer: EtherscanV2Client,
    tx_hash: str,
    code_cache: dict[str, bytes],
    block_ts_cache: dict[int, int],
) -> SimulationContext:
    """Reconstruct SimulationContext for a mined transaction (recent or historical archive)."""
    raw_tx, receipt, block_num, block_ts = _fetch_tx_and_receipt(
        w3_list, explorer, tx_hash, block_ts_cache
    )

    to_raw = raw_tx.get("to")
    to_clean = to_raw if (isinstance(to_raw, str) and len(to_raw) == 42) else None
    tx = UnsignedTransaction.model_validate(
        {
            "chainId": _hex_int(raw_tx.get("chainId")) or 1,
            "from": raw_tx["from"],
            "to": to_clean,
            "value": _hex_int(raw_tx.get("value")),
            "data": _hex_str(raw_tx.get("input", "0x")),
            "gas": _hex_int(raw_tx.get("gas")) or 21000,
            "authorizationList": raw_tx.get("authorizationList") or [],
        }
    )
    decoded = decode_calldata(tx.data)

    def get_code(addr: str) -> bytes:
        low = addr.lower()
        if low not in code_cache:
            checksummed = Web3.to_checksum_address(low)
            fetched: bytes | None = None
            for w3 in w3_list:
                try:
                    fetched = bytes(w3.eth.get_code(checksummed))
                    break
                except Exception as exc:
                    _ = exc
            code_cache[low] = fetched if fetched is not None else explorer.proxy_get_code(low)
        return code_cache[low]

    metadata: dict[str, AddressInfo] = {}
    for addr in addresses_of_interest(tx, decoded):
        code = get_code(addr)
        is_contract = len(code) > 0
        is_ver = explorer.is_verified(addr) if is_contract else False
        metadata[addr] = AddressInfo(is_contract=is_contract, is_verified=is_ver)

    age_blocks: int | None = None
    opcodes: list[str] = []
    nonstandard_dc = False
    target_contract = False
    target_verified = False

    if tx.to_addr:
        t_info = metadata[tx.to_addr]
        target_contract = t_info.is_contract
        target_verified = t_info.is_verified
        if target_contract:
            analysis = analyze_bytecode(get_code(tx.to_addr).hex())
            opcodes = analysis.opcodes
            nonstandard_dc = analysis.has_nonstandard_delegatecall
            created = explorer.creation_block(tx.to_addr)
            if created is not None:
                age_blocks = max(0, block_num - created)

    sender = tx.from_addr.lower()
    eth_delta = -tx.value if (tx.to_addr and tx.to_addr.lower() != sender) else 0
    erc20_deltas: dict[str, int] = {}
    approvals: list[dict[str, Any]] = []

    for log in receipt.get("logs") or []:
        raw_topics = log.get("topics") or []
        topics = [_hex_str(t).lower() for t in raw_topics]
        if not topics:
            continue
        token = str(log.get("address", "")).lower()
        amount_val = _hex_int(log.get("data"))
        head = topics[0]
        if head == TRANSFER_TOPIC and len(topics) in (3, 4):
            qty = amount_val if len(topics) == 3 else 1
            if _topic_addr(topics[1]) == sender:
                erc20_deltas[token] = erc20_deltas.get(token, 0) - qty
            if _topic_addr(topics[2]) == sender:
                erc20_deltas[token] = erc20_deltas.get(token, 0) + qty
        elif head == APPROVAL_TOPIC and len(topics) in (3, 4) and _topic_addr(topics[1]) == sender:
            approvals.append(
                {
                    "standard": "ERC-20" if len(topics) == 3 else "ERC-721",
                    "token": token,
                    "spender": _topic_addr(topics[2]),
                    "amount": str(amount_val) if len(topics) == 3 else "token-id",
                }
            )
        elif head == APPROVAL_ALL_TOPIC and len(topics) == 3:
            if _topic_addr(topics[1]) == sender and amount_val:
                approvals.append(
                    {
                        "standard": "ERC-721/1155",
                        "token": token,
                        "operator": _topic_addr(topics[2]),
                        "approved": True,
                    }
                )

    status_raw = receipt.get("status", 1)
    reverted = _hex_int(status_raw) == 0 if status_raw is not None else False
    return SimulationContext(
        tx=tx,
        decoded_call=decoded,
        is_contract=target_contract,
        is_verified=target_verified,
        deployment_age_blocks=age_blocks,
        opcodes=opcodes,
        has_nonstandard_delegatecall=nonstandard_dc,
        address_metadata=metadata,
        block_timestamp=block_ts,
        eth_delta_wei=eth_delta,
        erc20_deltas=erc20_deltas,
        approvals_granted=approvals,
        call_depth=1,
        reverted=reverted,
        trace_available=True,
    )


def _load_existing_rows(csv_path: Path) -> dict[str, dict[str, float | int | str]]:
    if not csv_path.exists():
        return {}
    cached: dict[str, dict[str, float | int | str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tx_h = str(row.get("tx_hash", "")).lower()
            if tx_h:
                parsed: dict[str, float | int | str] = {
                    name: float(row[name]) for name in FEATURE_NAMES
                }
                parsed["label"] = int(row["label"])
                parsed["tx_hash"] = tx_h
                cached[tx_h] = parsed
    return cached


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("data/chronicle_seeds.json"),
        help="Path to chronicle_seeds.json generated by parse_chronicle.py.",
    )
    parser.add_argument(
        "--rpc-url",
        default=os.environ.get("ARGUS_RPC_URL", DEFAULT_FREE_RPC),
        help="Primary Ethereum HTTP(S) RPC endpoint.",
    )
    parser.add_argument(
        "--etherscan-key",
        default=os.environ.get("ETHERSCAN_API_KEY", ""),
        help="Free Etherscan V2 API key (or set ETHERSCAN_API_KEY).",
    )
    parser.add_argument(
        "--max-malicious",
        type=int,
        default=90,
        help="Maximum malicious transactions to harvest.",
    )
    parser.add_argument(
        "--benign-ratio",
        type=float,
        default=2.5,
        help="Ratio of benign mainnet transactions per malicious sample.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing rows in --output and re-harvest from scratch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real_transactions.csv"),
        help="Destination CSV path.",
    )
    args = parser.parse_args()

    if not args.etherscan_key:
        print(
            "error: provide --etherscan-key or export ETHERSCAN_API_KEY (free tier).",
            file=sys.stderr,
        )
        return 2
    if not args.seeds.exists():
        print(
            f"error: {args.seeds} not found; run scripts/parse_chronicle.py first.",
            file=sys.stderr,
        )
        return 2

    existing = {} if args.fresh else _load_existing_rows(args.output)
    existing_benign = [r for r in existing.values() if int(r["label"]) == 0]
    existing_mal = [r for r in existing.values() if int(r["label"]) == 1]
    if existing:
        print(
            f"Reusing {len(existing_mal)} malicious + {len(existing_benign)} benign "
            f"cached rows from {args.output}"
        )

    seeds = json.loads(args.seeds.read_text(encoding="utf-8"))
    w3_list = [_make_w3(args.rpc_url, timeout=12.0)]
    if args.rpc_url != ARCHIVE_FALLBACK_RPC:
        w3_list.append(_make_w3(ARCHIVE_FALLBACK_RPC, timeout=15.0))
    explorer = EtherscanV2Client(args.etherscan_key)

    malicious_hashes: list[str] = [str(r["tx_hash"]) for r in existing_mal]
    for tx_h in seeds.get("malicious_tx_hashes", []):
        if tx_h not in malicious_hashes:
            malicious_hashes.append(tx_h)

    attackers: list[str] = list(seeds.get("attacker_addresses", []))
    for attacker in attackers:
        if len(malicious_hashes) >= args.max_malicious:
            break
        for tx_h in explorer.victim_approval_tx_hashes(attacker, limit=12):
            if tx_h not in malicious_hashes:
                malicious_hashes.append(tx_h)
            if len(malicious_hashes) >= args.max_malicious:
                break

    for attacker in attackers:
        if len(malicious_hashes) >= args.max_malicious:
            break
        txs = explorer.tx_hashes_for_address(
            attacker, limit=8, inbound_only=False, require_calldata=True
        )
        for tx_h in txs:
            if tx_h not in malicious_hashes:
                malicious_hashes.append(tx_h)
            if len(malicious_hashes) >= args.max_malicious:
                break

    malicious_hashes = malicious_hashes[: args.max_malicious]

    target_benign = max(30, int(max(len(malicious_hashes), 40) * args.benign_ratio))
    benign_hashes: list[str] = [
        str(r["tx_hash"]) for r in existing_benign if str(r["tx_hash"]) not in malicious_hashes
    ][:target_benign]

    if len(benign_hashes) < target_benign:
        needed = target_benign - len(benign_hashes)
        per_router = max(5, needed // len(BENIGN_ROUTERS) + 2)
        for router in BENIGN_ROUTERS:
            txs = explorer.tx_hashes_for_address(
                router, limit=per_router, inbound_only=True, require_calldata=False
            )
            for tx_h in txs:
                if tx_h not in malicious_hashes and tx_h not in benign_hashes:
                    benign_hashes.append(tx_h)
                if len(benign_hashes) >= target_benign:
                    break

    labeled_queue = [(h, 1) for h in malicious_hashes] + [(h, 0) for h in benign_hashes]
    print(
        f"Queued {len(malicious_hashes)} malicious + {len(benign_hashes)} benign "
        f"mainnet transactions..."
    )

    rows: list[dict[str, float | int | str]] = []
    code_cache: dict[str, bytes] = {}
    block_ts_cache: dict[int, int] = {}

    for idx, (tx_hash, label) in enumerate(labeled_queue, start=1):
        if tx_hash in existing:
            cached_row = dict(existing[tx_hash])
            cached_row["label"] = label
            rows.append(cached_row)
            continue
        try:
            ctx = build_context_from_receipt(w3_list, explorer, tx_hash, code_cache, block_ts_cache)
            feats = extract_features(ctx)
            row: dict[str, float | int | str] = dict(
                zip(FEATURE_NAMES, feats.tolist(), strict=True)
            )
            row["label"] = label
            row["tx_hash"] = tx_hash
            rows.append(row)
            if idx % 10 == 0 or idx == len(labeled_queue):
                mal_done = sum(1 for r in rows if int(r["label"]) == 1)
                ben_done = len(rows) - mal_done
                print(
                    f"  [{idx}/{len(labeled_queue)}] valid={len(rows)} "
                    f"(malicious={mal_done}, benign={ben_done})"
                )
        except Exception as exc:
            print(f"  warning: skipped {tx_hash[:14]}...: {exc}", file=sys.stderr)

    mal_total = sum(1 for r in rows if int(r["label"]) == 1)
    ben_total = len(rows) - mal_total
    if mal_total < 2 or ben_total < 2:
        print(
            f"error: insufficient class balance (malicious={mal_total}, benign={ben_total}).",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [*FEATURE_NAMES, "label", "tx_hash"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved {len(rows)} real transactions "
        f"(malicious={mal_total}, benign={ben_total}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())