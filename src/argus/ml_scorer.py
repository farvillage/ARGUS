"""Explainable ML risk scorer: seeded synthetic data -> RandomForest -> ONNX -> SHAP."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from argus.models import Finding, SimulationContext

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES: tuple[str, ...] = (
    "unbounded_approval",
    "approval_spender_untrusted",
    "permit_deadline_horizon",
    "eip7702_untrusted_delegate",
    "target_unverified",
    "contract_freshness",
    "dangerous_opcode",
    "asymmetric_outflow",
)

DEFAULT_SEED = 1337
DEFAULT_SAMPLES = 6000
MODEL_FILENAME = "risk_model.onnx"

_UNLIMITED = 2**255
_PERMIT_HORIZON_SECONDS = 365 * 86400
_FRESHNESS_CAP_BLOCKS = 5000
_BINARY_FEATURES = (
    "unbounded_approval",
    "approval_spender_untrusted",
    "eip7702_untrusted_delegate",
    "target_unverified",
    "dangerous_opcode",
    "asymmetric_outflow",
)


def _has_asymmetric_outflow(ctx: SimulationContext) -> bool:
    if not ctx.trace_available or ctx.reverted:
        return False
    outflow = ctx.eth_delta_wei < 0 or any(delta < 0 for delta in ctx.erc20_deltas.values())
    if not outflow:
        return False
    inbound = (
        ctx.eth_delta_wei > 0
        or any(delta > 0 for delta in ctx.erc20_deltas.values())
        or bool(ctx.receipt_tokens)
    )
    if inbound:
        return False
    return not (ctx.tx.data == "0x" and not ctx.is_contract)


def extract_features(ctx: SimulationContext) -> np.ndarray:
    """Map a simulation context to the 8-dimensional feature vector (float32)."""
    call = ctx.decoded_call
    args = call.arguments

    unbounded = 0.0
    spender: str | None = None
    horizon = 0.0
    if call.function_name == "approve":
        spender = args.get("spender")
        amount = args.get("amount")
        if isinstance(amount, int) and amount >= _UNLIMITED:
            unbounded = 1.0
    elif call.function_name == "setApprovalForAll":
        spender = args.get("operator")
        if args.get("approved") is True:
            unbounded = 1.0
    elif call.function_name == "permit":
        spender = args.get("spender")
        deadline = args.get("deadline")
        if isinstance(deadline, int):
            raw_horizon = (deadline - ctx.reference_time) / _PERMIT_HORIZON_SECONDS
            horizon = min(max(raw_horizon, 0.0), 1.0)

    spender_untrusted = 1.0 if isinstance(spender, str) and ctx.is_unverified(spender) else 0.0

    untrusted_delegate = 0.0
    for item in ctx.tx.authorization_list:
        info = ctx.info(item.address)
        if info is None or not info.is_contract or not info.is_verified:
            untrusted_delegate = 1.0
            break

    target_unverified = 1.0 if ctx.is_contract and not ctx.is_verified else 0.0
    freshness = 0.0
    if ctx.deployment_age_blocks is not None:
        capped_age = min(max(ctx.deployment_age_blocks, 0), _FRESHNESS_CAP_BLOCKS)
        freshness = 1.0 - capped_age / float(_FRESHNESS_CAP_BLOCKS)

    has_danger = ctx.has_nonstandard_delegatecall or "SELFDESTRUCT" in ctx.opcode_mnemonics
    dangerous = 1.0 if has_danger else 0.0
    asymmetric = 1.0 if _has_asymmetric_outflow(ctx) else 0.0

    return np.array(
        [
            unbounded,
            spender_untrusted,
            horizon,
            untrusted_delegate,
            target_unverified,
            freshness,
            dangerous,
            asymmetric,
        ],
        dtype=np.float32,
    )


def _benign_sample(rng: np.random.Generator) -> list[float]:
    """Ordinary swaps/deposits: mostly clean, with realistic noise (max approvals to routers)."""
    row = dict.fromkeys(FEATURE_NAMES, 0.0)
    row["unbounded_approval"] = float(rng.random() < 0.08)
    row["approval_spender_untrusted"] = float(rng.random() < 0.02)
    row["permit_deadline_horizon"] = (
        float(rng.uniform(0.0, 0.02)) if rng.random() < 0.10 else 0.0
    )
    row["eip7702_untrusted_delegate"] = float(rng.random() < 0.01)
    row["target_unverified"] = float(rng.random() < 0.04)
    roll = rng.random()
    if roll < 0.70:
        row["contract_freshness"] = 0.0
    elif roll < 0.95:
        row["contract_freshness"] = float(rng.uniform(0.0, 0.2))
    else:
        row["contract_freshness"] = float(rng.uniform(0.2, 0.8))
    row["dangerous_opcode"] = float(rng.random() < 0.03)
    row["asymmetric_outflow"] = float(rng.random() < 0.01)
    return [row[name] for name in FEATURE_NAMES]


def _attacker_freshness(rng: np.random.Generator) -> float:
    return float(rng.uniform(0.0, 1.0)) if rng.random() < 0.5 else 0.0


def _malicious_sample(rng: np.random.Generator) -> list[float]:
    """Five drainer archetypes with stealth dropout so classes are not trivially separable."""
    row = dict.fromkeys(FEATURE_NAMES, 0.0)
    kind = int(rng.integers(0, 5))
    if kind == 0:
        row["unbounded_approval"] = 1.0
        row["approval_spender_untrusted"] = 1.0
        row["target_unverified"] = float(rng.random() < 0.3)
        row["contract_freshness"] = _attacker_freshness(rng)
    elif kind == 1:
        row["approval_spender_untrusted"] = float(rng.random() < 0.85)
        row["permit_deadline_horizon"] = float(rng.uniform(0.09, 1.0))
        row["contract_freshness"] = _attacker_freshness(rng)
    elif kind == 2:
        row["target_unverified"] = 1.0
        row["contract_freshness"] = float(rng.uniform(0.6, 1.0))
        row["dangerous_opcode"] = 1.0
        row["asymmetric_outflow"] = float(rng.random() < 0.8)
    elif kind == 3:
        row["eip7702_untrusted_delegate"] = 1.0
        row["target_unverified"] = float(rng.random() < 0.3)
        row["asymmetric_outflow"] = float(rng.random() < 0.4)
        row["contract_freshness"] = _attacker_freshness(rng)
    else:
        row["asymmetric_outflow"] = 1.0
        row["target_unverified"] = float(rng.random() < 0.7)
        row["contract_freshness"] = float(rng.uniform(0.0, 1.0))

    for name in _BINARY_FEATURES:
        if row[name] == 1.0 and rng.random() < 0.08:
            row[name] = 0.0
    return [row[name] for name in FEATURE_NAMES]


def generate_dataset(
    n_samples: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, class-balanced synthetic dataset: ``X`` (n, 8) float32, ``y`` in {0, 1}."""
    rng = np.random.default_rng(seed)
    n_malicious = n_samples // 2
    rows: list[list[float]] = []
    labels: list[int] = []
    for _ in range(n_samples - n_malicious):
        rows.append(_benign_sample(rng))
        labels.append(0)
    for _ in range(n_malicious):
        rows.append(_malicious_sample(rng))
        labels.append(1)
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order]


def build_forest(seed: int = DEFAULT_SEED) -> RandomForestClassifier:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )


def train_forest(
    seed: int = DEFAULT_SEED, n_samples: int = DEFAULT_SAMPLES
) -> RandomForestClassifier:
    """Fit the forest on the full seeded dataset (deterministic for a given sklearn version)."""
    x, y = generate_dataset(n_samples=n_samples, seed=seed)
    forest = build_forest(seed)
    forest.fit(x, y)
    return forest


def default_model_path() -> Path:
    override = os.environ.get("ARGUS_MODEL_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "data" / MODEL_FILENAME


def _meta_path(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def export_onnx(
    forest: RandomForestClassifier,
    path: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    n_samples: int = DEFAULT_SAMPLES,
) -> Path:
    """Convert the forest to ONNX (probabilities as a plain tensor) and write a JSON sidecar."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    target = Path(path)
    onnx_model = convert_sklearn(
        forest,
        initial_types=[("input", FloatTensorType([None, len(FEATURE_NAMES)]))],
        options={id(forest): {"zipmap": False}},
        target_opset={"": 15, "ai.onnx.ml": 3},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(onnx_model.SerializeToString())
    _meta_path(target).write_text(
        json.dumps(
            {"seed": seed, "n_samples": n_samples, "features": list(FEATURE_NAMES)}, indent=2
        ),
        encoding="utf-8",
    )
    return target


def ensure_model(model_path: Path | str | None = None) -> Path:
    """Return the ONNX path, training and exporting the default model on first use."""
    path = Path(model_path) if model_path else default_model_path()
    if path.exists():
        return path
    try:
        export_onnx(train_forest(), path, seed=DEFAULT_SEED, n_samples=DEFAULT_SAMPLES)
    except OSError as exc:
        raise RuntimeError(
            f"cannot write model to {path} ({exc}); set ARGUS_MODEL_PATH to a writable file"
        ) from exc
    return path


def _read_meta(model_path: Path) -> dict[str, Any]:
    meta = {"seed": DEFAULT_SEED, "n_samples": DEFAULT_SAMPLES}
    sidecar = _meta_path(model_path)
    if sidecar.exists():
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["seed"] = int(loaded.get("seed", DEFAULT_SEED))
        meta["n_samples"] = int(loaded.get("n_samples", DEFAULT_SAMPLES))
    return meta


class OnnxRiskScorer:
    """Single-thread CPU ONNX inference; call ``benchmark()`` for the median latency."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        import onnxruntime as ort

        self.model_path = ensure_model(model_path)
        self.meta = _read_meta(self.model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.predict_proba(np.zeros(len(FEATURE_NAMES), dtype=np.float32))

    def predict_proba(self, features: np.ndarray) -> float:
        batch = np.asarray(features, dtype=np.float32).reshape(1, -1)
        probabilities = self._session.run(["probabilities"], {self._input_name: batch})[0]
        return float(probabilities[0][1])

    def benchmark(self, runs: int = 200) -> float:
        sample = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        timings: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            self.predict_proba(sample)
            timings.append((time.perf_counter() - start) * 1000.0)
        timings.sort()
        return timings[len(timings) // 2]


@lru_cache(maxsize=4)
def _scorer_for(path_str: str) -> OnnxRiskScorer:
    return OnnxRiskScorer(path_str)


def get_scorer(model_path: Path | str | None = None) -> OnnxRiskScorer:
    path = Path(model_path) if model_path else default_model_path()
    return _scorer_for(str(path))


@lru_cache(maxsize=4)
def _explainer_for(seed: int, n_samples: int) -> Any:
    import shap

    return shap.TreeExplainer(train_forest(seed, n_samples))


def shap_attributions(
    features: np.ndarray, *, seed: int = DEFAULT_SEED, n_samples: int = DEFAULT_SAMPLES
) -> dict[str, float]:
    """SHAP values for the 'malicious' class (probability units); positive raises risk."""
    explainer = _explainer_for(seed, n_samples)
    batch = np.asarray(features, dtype=np.float32).reshape(1, -1)
    values = explainer.shap_values(batch, check_additivity=False)
    if isinstance(values, list):
        contribution = np.asarray(values[1])[0]
    else:
        array = np.asarray(values)
        contribution = array[0, :, 1] if array.ndim == 3 else array[0]
    return {
        name: round(float(v), 6)
        for name, v in zip(FEATURE_NAMES, contribution, strict=True)
    }


def explain(
    ctx: SimulationContext,
    findings: list[Finding] | None = None,
    model_path: Path | str | None = None,
) -> tuple[float, dict[str, float]]:
    """Return ``(malicious_probability, {feature: shap_value})``."""
    _ = findings
    features = extract_features(ctx)
    scorer = get_scorer(model_path)
    probability = scorer.predict_proba(features)
    attributions = shap_attributions(
        features, seed=scorer.meta["seed"], n_samples=scorer.meta["n_samples"]
    )
    return round(probability, 6), attributions