#!/usr/bin/env python3
"""Benchmark RF, HistGBDT, and XGBoost on synthetic or real CSV data and export to ONNX.

Usage:
    python3 scripts/train_model.py [--dataset data/real_transactions.csv] [--model auto]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from argus.ml_scorer import (  # noqa: E402
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    FEATURE_NAMES,
    OnnxRiskScorer,
    build_estimator,
    default_model_path,
    export_onnx,
    generate_dataset,
    xgboost_available,
)


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


def load_csv_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 8 behavioral features and binary labels from a harvested CSV file."""
    feature_rows: list[list[float]] = []
    labels: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            feature_rows.append([float(row[name]) for name in FEATURE_NAMES])
            labels.append(int(row["label"]))
    return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def evaluate_cv(
    model_type: str, x: np.ndarray, y: np.ndarray, seed: int, n_splits: int
) -> dict[str, float]:
    """Compute out-of-fold Stratified K-Fold metrics for a candidate architecture."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y), dtype=np.float64)
    oof_preds = np.zeros(len(y), dtype=np.int64)

    for train_idx, val_idx in skf.split(x, y):
        y_tr = y[train_idx]
        pos = max(int(y_tr.sum()), 1)
        neg = max(len(y_tr) - pos, 1)
        clf = build_estimator(model_type, seed=seed, pos_weight=neg / pos)
        clf.fit(x[train_idx], y_tr)
        oof_probs[val_idx] = clf.predict_proba(x[val_idx])[:, 1]
        oof_preds[val_idx] = (oof_probs[val_idx] >= 0.5).astype(np.int64)

    return {
        "pr_auc": float(average_precision_score(y, oof_probs)),
        "roc_auc": float(roc_auc_score(y, oof_probs)),
        "f1": float(f1_score(y, oof_preds, zero_division=0)),
        "precision": float(precision_score(y, oof_preds, zero_division=0)),
        "recall": float(recall_score(y, oof_preds, zero_division=0)),
        "accuracy": float(accuracy_score(y, oof_preds)),
        "brier": float(brier_score_loss(y, oof_probs)),
    }


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Synthetic samples.")
    parser.add_argument("--output", type=Path, default=None, help="Destination .onnx path.")
    parser.add_argument("--dataset", type=Path, default=None, help="Real harvested CSV path.")
    parser.add_argument(
        "--model",
        choices=("auto", "rf", "hist_gbdt", "xgboost"),
        default="auto",
        help="Model architecture to export ('auto' picks highest CV PR-AUC).",
    )
    args = parser.parse_args()
    output = args.output or default_model_path()

    use_real_csv = args.dataset is not None and args.dataset.exists()
    if use_real_csv:
        print(f"Loading real-world CSV dataset: {args.dataset}")
        x, y = load_csv_dataset(args.dataset)
    else:
        if args.dataset is not None:
            print(f"Warning: '{args.dataset}' not found. Falling back to synthetic data.")
        else:
            print("Using seeded synthetic dataset generator.")
        x, y = generate_dataset(n_samples=args.samples, seed=args.seed)

    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    if pos_count < 2 or neg_count < 2:
        print(
            "error: dataset must contain at least 2 malicious and 2 benign rows.",
            file=sys.stderr,
        )
        return 2

    print(f"dataset: {x.shape[0]} samples x {x.shape[1]} features | malicious: {y.mean():.1%}")

    candidates = ["rf", "hist_gbdt"]
    if xgboost_available():
        candidates.append("xgboost")
    elif args.model == "xgboost":
        print(
            "error: install xgboost and onnxmltools (`pip install -e '.[train]'`).",
            file=sys.stderr,
        )
        return 2

    to_run = candidates if args.model == "auto" else [args.model]
    n_splits = min(5, pos_count, neg_count)
    print(f"\nStratified {n_splits}-Fold Cross-Validation Benchmark:")
    header = (
        f"{'Model':<12} {'PR-AUC':>8} {'ROC-AUC':>8} "
        f"{'F1':>8} {'Prec':>8} {'Rec':>8} {'Brier':>8}"
    )
    print(header)
    print("-" * len(header))

    scores: dict[str, dict[str, float]] = {}
    for name in to_run:
        metrics = evaluate_cv(name, x, y, args.seed, n_splits)
        scores[name] = metrics
        print(
            f"{name:<12} {metrics['pr_auc']:>8.4f} {metrics['roc_auc']:>8.4f} "
            f"{metrics['f1']:>8.4f} {metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} {metrics['brier']:>8.4f}"
        )

    winner = max(scores, key=lambda k: (scores[k]["pr_auc"], scores[k]["roc_auc"]))
    print(f"\nSelected architecture: {winner}")

    final_model = build_estimator(winner, seed=args.seed, pos_weight=neg_count / pos_count)
    final_model.fit(x, y)

    path = export_onnx(
        final_model,
        output,
        seed=args.seed,
        n_samples=len(y),
        model_type=winner,
        x_train=x if use_real_csv else None,
        y_train=y if use_real_csv else None,
    )
    print(f"Exported ONNX model -> {path}")

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    prob_name = "probabilities" if "probabilities" in out_names else out_names[-1]
    sample = x[: min(500, len(x))]
    onnx_probs = session.run([prob_name], {input_name: sample})[0][:, 1]
    ref_probs = final_model.predict_proba(sample)[:, 1]
    max_diff = float(np.max(np.abs(onnx_probs - ref_probs)))
    print(f"ONNX vs native max |diff| = {max_diff:.2e}")
    if max_diff > 1e-4:
        print("error: ONNX output diverges from native estimator", file=sys.stderr)
        return 1

    latency = OnnxRiskScorer(path).benchmark(200)
    print(f"Median single-thread CPU latency = {latency:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())