#!/usr/bin/env python3
"""Parse DeFi-Hack-Chronicle JSON incidents into Ethereum Mainnet tx hashes and addresses.

Usage:
    python3 scripts/parse_chronicle.py --chronicle-dir /tmp/DeFi-Hack-Chronicle
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

TX_HASH_RE = re.compile(r"\b0x[a-fA-F0-9]{64}\b")
ADDRESS_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")

ETH_CHAIN_NAMES = {"ethereum", "eth", "mainnet", "ethereum mainnet", "1"}

# Publicly documented Etherscan Phish/Hack wallet-drainer addresses used as external
# ground-truth seeds to locate real mainnet victim ERC-20 approval/drainer transactions.
KNOWN_DRAINER_ADDRESSES: tuple[str, ...] = (
    "0xa5e4b451d0a3c3d05fc3a8076fda45952b8f4f83",  # Pink Drainer 0xa5e4 (Etherscan Phish/Hack)
    "0x67e5ae3e1ad16d4c020db518f2a9943d4f73d6ef",  # Fake_Phishing322880 (ScamSniffer)
)


def _is_eth_item(item: Any, default_eth: bool) -> bool:
    """Return True if a transaction/address entry belongs to Ethereum Mainnet."""
    if not isinstance(item, dict):
        return default_eth
    for key in ("chain", "blockchain", "network", "chainId"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower() in ETH_CHAIN_NAMES
        if isinstance(val, int):
            return val == 1
    url_hint = str(item.get("url") or item.get("explorer") or "").lower()
    if url_hint and "scan" in url_hint and "etherscan.io" not in url_hint:
        return False
    return default_eth


def _collect_eth_matches(
    entries: Any, pattern: re.Pattern[str], default_eth: bool
) -> list[str]:
    found: list[str] = []
    if isinstance(entries, list):
        for entry in entries:
            if _is_eth_item(entry, default_eth):
                found.extend(m.lower() for m in pattern.findall(json.dumps(entry)))
    elif entries is not None and default_eth:
        found.extend(m.lower() for m in pattern.findall(json.dumps(entries)))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chronicle-dir",
        type=Path,
        required=True,
        help="Path to local clone of DeFiHackLabs/DeFi-Hack-Chronicle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chronicle_seeds.json"),
        help="Output JSON manifest of Ethereum exploit tx hashes and attacker addresses.",
    )
    args = parser.parse_args()

    hacks_dir = args.chronicle_dir / "public" / "data" / "hacks"
    if not hacks_dir.is_dir():
        hacks_dir = args.chronicle_dir
    json_files = sorted(hacks_dir.glob("*.json"))
    if not json_files:
        print(f"error: no JSON incident files found in {hacks_dir}", file=sys.stderr)
        return 2

    tx_hashes: dict[str, str] = {}
    attacker_addrs: dict[str, str] = dict.fromkeys(KNOWN_DRAINER_ADDRESSES, "public-phish-hack")
    matched_events = 0

    for file_path in json_files:
        try:
            doc = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {file_path.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(doc, dict):
            continue

        ecosystem = str(doc.get("ecosystem", "")).lower()
        if ecosystem and ecosystem != "evm":
            continue

        chains = [str(c).strip().lower() for c in (doc.get("blockchain") or [])]
        if chains and not any(c in ETH_CHAIN_NAMES for c in chains):
            continue

        only_ethereum = chains in (["ethereum"], ["eth"])
        matched_events += 1
        event_id = str(doc.get("id") or file_path.stem)

        for tx_h in _collect_eth_matches(doc.get("transactions"), TX_HASH_RE, only_ethereum):
            tx_hashes.setdefault(tx_h, event_id)
        for addr in _collect_eth_matches(doc.get("attackers"), ADDRESS_RE, True):
            attacker_addrs.setdefault(addr, event_id)

    payload = {
        "matched_events": matched_events,
        "malicious_tx_hashes": sorted(tx_hashes),
        "attacker_addresses": sorted(attacker_addrs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Parsed {matched_events} Ethereum incidents -> "
        f"{len(tx_hashes)} tx hashes, {len(attacker_addrs)} attacker/drainer addresses "
        f"saved to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())