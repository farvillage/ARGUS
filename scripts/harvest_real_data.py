#!/usr/bin/env python3
"""Harvest real historical EVM transactions from an archive RPC node into a CSV dataset.

Usage:
    python3 scripts/harvest_real_data.py --rpc-url https://... [--output data/real_txs.csv]
"""

# Standard library imports sorted alphabetically to satisfy Ruff rule I001
import argparse
import csv
import os
import sys
from pathlib import Path

# Add the local `src` folder to sys.path so `argus` imports work without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# First-party ARGUS imports (`noqa: E402` allows importing after `sys.path.insert`)
from argus.evm import LiveRPCSimulator, decode_calldata  # noqa: E402
from argus.ml_scorer import FEATURE_NAMES, extract_features  # noqa: E402
from argus.models import UnsignedTransaction  # noqa: E402

# Sample list of (transaction_hash, binary_label) pairs to harvest from an archive node.
# Label 1 = confirmed exploit/drainer transaction; Label 0 = benign user transaction.
SAMPLE_SOURCES: list[tuple[str, int]] = [
    # Replace these placeholder hashes with real 66-character 0x mainnet tx hashes
    ("0x0000000000000000000000000000000000000000000000000000000000000001", 1),
    ("0x0000000000000000000000000000000000000000000000000000000000000002", 0),
]


def main() -> int:
    """Fetch historical transactions over RPC, extract ARGUS features, and save CSV."""
    # Configure CLI arguments for the archive node endpoint and output CSV path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rpc-url",
        default=os.environ.get("ARGUS_RPC_URL"),
        help="Archive HTTP(S) RPC endpoint (defaults to ARGUS_RPC_URL env var).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real_transactions.csv"),
        help="Destination CSV path for harvested feature vectors.",
    )
    args = parser.parse_args()

    # Require a valid RPC URL before attempting network requests
    if not args.rpc_url:
        print(
            "error: pass --rpc-url <URL> or set the ARGUS_RPC_URL environment variable.",
            file=sys.stderr,
        )
        return 2

    # Initialize the SSRF-hardened ARGUS live RPC simulator
    print(f"Connecting to archive RPC node: {args.rpc_url}")
    rpc = LiveRPCSimulator(args.rpc_url)

    # Accumulate extracted feature dictionaries here before writing to disk
    harvested_rows: list[dict[str, float | int | str]] = []

    # Iterate through each labeled transaction hash and simulate its execution
    for tx_hash, label in SAMPLE_SOURCES:
        print(f"Simulating transaction {tx_hash} (label={label})...")
        try:
            # Download the historical on-chain transaction fields by hash
            raw_tx = rpc.w3.eth.get_transaction(tx_hash)

            # Normalize RPC fields into ARGUS's strict UnsignedTransaction schema
            tx = UnsignedTransaction.model_validate(
                {
                    "chainId": raw_tx.get("chainId", 1),
                    "from": raw_tx["from"],
                    "to": raw_tx.get("to"),
                    "value": raw_tx.get("value", 0),
                    "data": raw_tx.get("input", "0x"),
                    "gas": raw_tx.get("gas"),
                }
            )

            # Decode the 4-byte selector and ABI arguments from transaction calldata
            decoded = decode_calldata(tx.data)

            # Run bytecode disassembly and callTracer simulation against the node
            ctx = rpc.simulate(tx, decoded)

            # Convert simulation telemetry into the 8-element float32 feature array
            features = extract_features(ctx)

            # Map each feature value to its canonical column name (strict=True per B905)
            row: dict[str, float | int | str] = dict(
                zip(FEATURE_NAMES, features.tolist(), strict=True)
            )
            # Append the ground-truth label and transaction hash for provenance
            row["label"] = label
            row["tx_hash"] = tx_hash
            harvested_rows.append(row)
        except Exception as exc:
            # Log individual RPC/tx failures without aborting the entire harvest run
            print(f"warning: skipped {tx_hash}: {exc}", file=sys.stderr)

    # Exit with error code 1 if no transactions could be fetched or simulated
    if not harvested_rows:
        print("error: zero transactions were harvested.", file=sys.stderr)
        return 1

    # Create parent directories (e.g. `data/`) if they do not exist yet
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write all harvested rows to a standard UTF-8 CSV file using Python's csv module
    columns = [*FEATURE_NAMES, "label", "tx_hash"]
    with args.output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(harvested_rows)

    print(f"Saved {len(harvested_rows)} labeled samples -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())