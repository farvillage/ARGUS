#!/usr/bin/env python3
"""Train the ARGUS risk model on synthetic or CSV data, evaluate it, and export to ONNX.

Usage:
    python3 scripts/train_model.py [--seed 1337] [--samples 6000] [--dataset PATH]
"""

# Standard library imports sorted alphabetically to satisfy Ruff rule I001
import argparse
import csv
import sys
from pathlib import Path

# Add local `src` directory to sys.path so `argus` resolves without global install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Third-party scientific & ONNX imports (`noqa: E402` allows post-sys.path imports)
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

# First-party ARGUS ML pipeline helpers sorted per Ruff isort conventions
from argus.ml_scorer import (  # noqa: E402
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    FEATURE_NAMES,
    OnnxRiskScorer,
    build_forest,
    default_model_path,
    export_onnx,
    generate_dataset,
    train_forest,
)


def load_csv_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 8 behavioral features and binary labels from a harvested CSV file."""
    feature_rows: list[list[float]] = []
    labels: list[int] = []

    # Read UTF-8 CSV rows using Python's built-in DictReader (no pandas required)
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # Extract the 8 features in exact FEATURE_NAMES order as floats
            feature_rows.append([float(row[name]) for name in FEATURE_NAMES])
            # Parse the ground-truth class label (0 = benign, 1 = malicious)
            labels.append(int(row["label"]))

    # Convert Python lists into contiguous NumPy arrays expected by scikit-learn
    x = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def main() -> int:
    """Train, evaluate, verify ONNX parity, and benchmark inference latency."""
    # Define command-line flags (all lines kept under 100 chars for Ruff E501)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible synthetic data and forest training.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of synthetic transactions to generate when no CSV is given.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for the exported .onnx model artifact.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional path to a real harvested CSV dataset file.",
    )
    args = parser.parse_args()

    # Resolve target ONNX file path (defaults to src/argus/data/risk_model.onnx)
    output = args.output or default_model_path()

    # Step 1: Load real CSV data if provided and present; otherwise synthesize data
    use_real_csv = args.dataset is not None and args.dataset.exists()
    if use_real_csv:
        print(f"Loading real-world CSV dataset from: {args.dataset}")
        x, y = load_csv_dataset(args.dataset)
    else:
        if args.dataset is not None:
            print(
                f"Warning: '{args.dataset}' not found. Falling back to synthetic data."
            )
        else:
            print("Using seeded synthetic dataset generator.")
        x, y = generate_dataset(n_samples=args.samples, seed=args.seed)

    # Print dataset dimensions and class balance summary
    joined_names = ", ".join(FEATURE_NAMES)
    print(f"dataset: {x.shape[0]} samples x {x.shape[1]} features ({joined_names})")
    print(f"malicious share: {y.mean():.1%}")

    # Step 2: Evaluate generalization metrics on a stratified 20% hold-out split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    evaluator = build_forest(args.seed).fit(x_train, y_train)
    predictions = evaluator.predict(x_test)
    probabilities = evaluator.predict_proba(x_test)[:, 1]
    print(
        "hold-out: "
        f"accuracy={accuracy_score(y_test, predictions):.4f} "
        f"precision={precision_score(y_test, predictions):.4f} "
        f"recall={recall_score(y_test, predictions):.4f} "
        f"roc_auc={roc_auc_score(y_test, probabilities):.4f}"
    )

    # Step 3: Fit the production Random Forest on the full dataset and export ONNX
    if use_real_csv:
        forest = build_forest(args.seed).fit(x, y)
    else:
        forest = train_forest(args.seed, args.samples)

    path = export_onnx(forest, output, seed=args.seed, n_samples=len(y))
    print(f"exported ONNX model -> {path}")

    # Step 4: Verify numerical parity between ONNX Runtime and scikit-learn
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    sample = x[: min(500, len(x))]
    onnx_probs = session.run(["probabilities"], {input_name: sample})[0][:, 1]
    sklearn_probs = forest.predict_proba(sample)[:, 1]
    max_diff = float(np.max(np.abs(onnx_probs - sklearn_probs)))
    print(f"ONNX vs sklearn max |diff| = {max_diff:.2e}")
    if max_diff > 1e-4:
        print("error: ONNX output diverges from scikit-learn", file=sys.stderr)
        return 1

    # Step 5: Benchmark median single-thread CPU inference latency (< 5ms target)
    latency = OnnxRiskScorer(path).benchmark(200)
    print(f"median inference latency = {latency:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())