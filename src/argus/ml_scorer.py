"""Explainable ML risk scorer: synthetic or real CSV data -> RF/GBDT/XGBoost -> ONNX -> SHAP.

Imported lazily by the CLI only when ``--explain`` is passed. Requires the ``ml`` extra.

Design notes
------------
* Supports three tree architectures: ``rf`` (RandomForest), ``hist_gbdt`` (scikit-learn
  HistGradientBoosting), and ``xgboost`` (XGBClassifier, when installed).
* Features are derived strictly from ``SimulationContext``, never from deterministic rule
  verdicts, eliminating label leakage.
* Scoring uses ``onnxruntime`` on a single CPU thread (< 5 ms).
* SHAP attributions run in probability space with zero ``pickle``/``joblib`` deserialization:
  XGBoost trees are stored in safe ``.xgb.json`` format, and scikit-learn real-data models
  reconstruct deterministically from ``allow_pickle=False`` NumPy ``.npz`` arrays (or the
  recorded synthetic seed when no real dataset was used).
"""

from __future__ import annotations

import json
import os
import tempfile
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
SUPPORTED_MODELS: tuple[str, ...] = ("rf", "hist_gbdt", "xgboost")

_UNLIMITED = 2**255
_PERMIT_HORIZON_SECONDS = 365 * 86400
_FRESHNESS_CAP_BLOCKS = 5000
_SHAP_BG_ROWS = 128

_BINARY_FEATURES = (
    "unbounded_approval",
    "approval_spender_untrusted",
    "eip7702_untrusted_delegate",
    "target_unverified",
    "dangerous_opcode",
    "asymmetric_outflow",
)


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


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
            raw_h = (deadline - ctx.reference_time) / _PERMIT_HORIZON_SECONDS
            horizon = min(max(raw_h, 0.0), 1.0)

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
        capped = min(max(ctx.deployment_age_blocks, 0), _FRESHNESS_CAP_BLOCKS)
        freshness = 1.0 - capped / float(_FRESHNESS_CAP_BLOCKS)

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


# --------------------------------------------------------------------------- #
# Seeded synthetic dataset
# --------------------------------------------------------------------------- #


def _benign_sample(rng: np.random.Generator) -> list[float]:
    """Ordinary swaps/deposits: mostly clean, with realistic router approval noise."""
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
    """Five drainer archetypes with stealth dropout so classes are non-trivial."""
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
    """Deterministic, class-balanced synthetic dataset: ``X`` (n, 8) float32, ``y`` int64."""
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


# --------------------------------------------------------------------------- #
# Model builders (RF, HistGBDT, XGBoost) + ONNX export
# --------------------------------------------------------------------------- #


def xgboost_available() -> bool:
    """Return True when both ``xgboost`` and ``onnxmltools`` are importable."""
    try:
        import onnxmltools  # noqa: F401
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


def build_forest(seed: int = DEFAULT_SEED) -> RandomForestClassifier:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )


def build_estimator(
    model_type: str = "rf",
    seed: int = DEFAULT_SEED,
    pos_weight: float = 1.0,
) -> Any:
    """Construct an unfitted tree classifier for ``rf``, ``hist_gbdt``, or ``xgboost``."""
    if model_type == "rf":
        return build_forest(seed)
    if model_type == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=6,
            learning_rate=0.06,
            min_samples_leaf=8,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            min_child_weight=3,
            scale_pos_weight=max(pos_weight, 0.1),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"unsupported model_type {model_type!r}; choose from {SUPPORTED_MODELS}")


def train_forest(
    seed: int = DEFAULT_SEED, n_samples: int = DEFAULT_SAMPLES
) -> RandomForestClassifier:
    """Fit the default Random Forest on the full seeded synthetic dataset."""
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


def _xgb_sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(".xgb.json")


def _npz_sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(".npz")


def _detect_model_type(estimator: Any) -> str:
    cls_name = type(estimator).__name__
    if "XGB" in cls_name:
        return "xgboost"
    if "HistGradientBoosting" in cls_name:
        return "hist_gbdt"
    return "rf"


def _convert_estimator_to_onnx(estimator: Any, model_type: str) -> bytes:
    """Serialize a fitted tree classifier into an ONNX graph with tensor probabilities."""
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("input", FloatTensorType([None, len(FEATURE_NAMES)]))]
    if model_type == "xgboost":
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
        from skl2onnx.convert import may_switch_bases_classes_order
        from xgboost import XGBClassifier

        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        with may_switch_bases_classes_order(XGBClassifier):
            onnx_model = convert_sklearn(
                estimator,
                initial_types=initial_types,
                options={id(estimator): {"zipmap": False}},
                target_opset={"": 15, "ai.onnx.ml": 3},
            )
    else:
        onnx_model = convert_sklearn(
            estimator,
            initial_types=initial_types,
            options={id(estimator): {"zipmap": False}},
            target_opset={"": 15, "ai.onnx.ml": 3},
        )
    return onnx_model.SerializeToString()


def export_onnx(
    estimator: Any,
    path: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    n_samples: int = DEFAULT_SAMPLES,
    model_type: str | None = None,
    x_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
) -> Path:
    """Convert estimator to ONNX and write safe (zero-pickle) SHAP sidecar files."""
    target = Path(path)
    resolved_type = model_type or _detect_model_type(estimator)
    onnx_bytes = _convert_estimator_to_onnx(estimator, resolved_type)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(onnx_bytes)

    rng = np.random.default_rng(seed)
    if x_train is not None and len(x_train) > 0:
        bg_idx = rng.choice(len(x_train), size=min(_SHAP_BG_ROWS, len(x_train)), replace=False)
        bg_rows = np.asarray(x_train[bg_idx], dtype=np.float32)
    else:
        syn_x, _ = generate_dataset(n_samples=max(_SHAP_BG_ROWS, 256), seed=seed)
        bg_rows = syn_x[:_SHAP_BG_ROWS]

    xgb_path = _xgb_sidecar_path(target)
    npz_path = _npz_sidecar_path(target)
    has_real_npz = False

    if resolved_type == "xgboost":
        estimator.get_booster().save_model(str(xgb_path))
        if npz_path.exists():
            npz_path.unlink()
    else:
        if xgb_path.exists():
            xgb_path.unlink()
        if x_train is not None and y_train is not None:
            np.savez_compressed(
                npz_path,
                x=np.asarray(x_train, dtype=np.float32),
                y=np.asarray(y_train, dtype=np.int64),
            )
            has_real_npz = True
        elif npz_path.exists():
            npz_path.unlink()

    meta: dict[str, Any] = {
        "model_type": resolved_type,
        "seed": seed,
        "n_samples": n_samples,
        "has_real_data": resolved_type == "xgboost" or has_real_npz,
        "features": list(FEATURE_NAMES),
        "background": np.round(bg_rows, 6).tolist(),
    }
    _meta_path(target).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _scorer_for.cache_clear()
    _explainer_for_model.cache_clear()
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
    meta: dict[str, Any] = {
        "model_type": "rf",
        "seed": DEFAULT_SEED,
        "n_samples": DEFAULT_SAMPLES,
        "background": None,
    }
    sidecar = _meta_path(model_path)
    if sidecar.exists():
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["model_type"] = str(loaded.get("model_type", "rf"))
        meta["seed"] = int(loaded.get("seed", DEFAULT_SEED))
        meta["n_samples"] = int(loaded.get("n_samples", DEFAULT_SAMPLES))
        meta["background"] = loaded.get("background")
    return meta


# --------------------------------------------------------------------------- #
# ONNX runtime scorer
# --------------------------------------------------------------------------- #


class OnnxRiskScorer:
    """Single-thread CPU ONNX inference; call ``benchmark()`` for median latency."""

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
        out_names = [out.name for out in self._session.get_outputs()]
        self._prob_output = "probabilities" if "probabilities" in out_names else out_names[-1]
        self.predict_proba(np.zeros(len(FEATURE_NAMES), dtype=np.float32))

    def predict_proba(self, features: np.ndarray) -> float:
        """Probability that the transaction is malicious."""
        batch = np.asarray(features, dtype=np.float32).reshape(1, -1)
        probs = self._session.run([self._prob_output], {self._input_name: batch})[0]
        return float(probs[0][1])

    def benchmark(self, runs: int = 200) -> float:
        """Median latency in milliseconds over ``runs`` inferences."""
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


# --------------------------------------------------------------------------- #
# SHAP attribution (zero-pickle, supports XGBoost / HistGBDT / RandomForest)
# --------------------------------------------------------------------------- #


def _load_safe_xgb_booster(xgb_path: Path) -> Any:
    """Load an XGBoost Booster from JSON, sanitizing bracketed base_score if needed."""
    import xgboost as xgb

    raw_doc = json.loads(xgb_path.read_text(encoding="utf-8"))
    learner = raw_doc.get("learner", {})
    params = learner.get("learner_model_param", {})
    base_score = params.get("base_score")
    if isinstance(base_score, str) and base_score.startswith("[") and base_score.endswith("]"):
        params["base_score"] = base_score.strip("[]").split(",")[0].strip()
        with tempfile.TemporaryDirectory() as tmp_dir:
            clean_file = Path(tmp_dir) / "clean.xgb.json"
            clean_file.write_text(json.dumps(raw_doc), encoding="utf-8")
            booster = xgb.Booster()
            booster.load_model(str(clean_file))
            return booster

    booster = xgb.Booster()
    booster.load_model(str(xgb_path))
    return booster


@lru_cache(maxsize=4)
def _explainer_for(seed: int, n_samples: int) -> Any:
    import shap

    return shap.TreeExplainer(train_forest(seed, n_samples))


@lru_cache(maxsize=4)
def _explainer_for_model(path_str: str, mtime_ns: int) -> Any:
    """Construct and cache a probability-space SHAP TreeExplainer for ``path_str``."""
    import shap

    _ = mtime_ns
    model_path = Path(path_str)
    meta = _read_meta(model_path)
    model_type = meta["model_type"]
    seed = meta["seed"]
    n_samples = meta["n_samples"]

    bg_raw = meta.get("background")
    if isinstance(bg_raw, list) and bg_raw:
        background = np.asarray(bg_raw, dtype=np.float32)
    else:
        syn_x, _ = generate_dataset(n_samples=256, seed=seed)
        background = syn_x[:_SHAP_BG_ROWS]

    xgb_path = _xgb_sidecar_path(model_path)
    if model_type == "xgboost" and xgb_path.exists():
        try:
            import xgboost as xgb

            plain_booster = xgb.Booster()
            plain_booster.load_model(str(xgb_path))
            return shap.TreeExplainer(
                plain_booster,
                data=background,
                feature_perturbation="interventional",
                model_output="probability",
            )
        except ValueError:
            safe_booster = _load_safe_xgb_booster(xgb_path)
            return shap.TreeExplainer(
                safe_booster,
                data=background,
                feature_perturbation="interventional",
                model_output="probability",
            )

    npz_path = _npz_sidecar_path(model_path)
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as arrays:
            x_real = np.asarray(arrays["x"], dtype=np.float32)
            y_real = np.asarray(arrays["y"], dtype=np.int64)
        pos = max(int(y_real.sum()), 1)
        neg = max(len(y_real) - pos, 1)
        fitted = build_estimator(model_type, seed=seed, pos_weight=neg / pos).fit(x_real, y_real)
        if model_type == "hist_gbdt":
            return shap.TreeExplainer(
                fitted,
                data=background,
                feature_perturbation="interventional",
                model_output="probability",
            )
        return shap.TreeExplainer(fitted)

    if model_type == "hist_gbdt":
        x_syn, y_syn = generate_dataset(n_samples=n_samples, seed=seed)
        fitted_hgb = build_estimator("hist_gbdt", seed=seed).fit(x_syn, y_syn)
        return shap.TreeExplainer(
            fitted_hgb,
            data=background,
            feature_perturbation="interventional",
            model_output="probability",
        )

    return _explainer_for(seed, n_samples)


def _extract_shap_vector(values: Any) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[1])[0]
    array = np.asarray(values)
    if array.ndim == 3:
        return array[0, :, 1]
    return array[0]


def shap_attributions(
    features: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    n_samples: int = DEFAULT_SAMPLES,
    model_path: Path | str | None = None,
) -> dict[str, float]:
    """SHAP values for the 'malicious' class (probability units); positive raises risk."""
    if model_path is not None and Path(model_path).exists():
        resolved = Path(model_path)
        explainer = _explainer_for_model(str(resolved), resolved.stat().st_mtime_ns)
    else:
        explainer = _explainer_for(seed, n_samples)

    batch = np.asarray(features, dtype=np.float32).reshape(1, -1)
    values = explainer.shap_values(batch, check_additivity=False)
    contribution = _extract_shap_vector(values)
    return {
        name: round(float(v), 6)
        for name, v in zip(FEATURE_NAMES, contribution, strict=True)
    }


def explain(
    ctx: SimulationContext,
    findings: list[Finding] | None = None,
    model_path: Path | str | None = None,
) -> tuple[float, dict[str, float]]:
    """CLI contract: return ``(malicious_probability, {feature: shap_value})``."""
    _ = findings
    features = extract_features(ctx)
    scorer = get_scorer(model_path)
    probability = scorer.predict_proba(features)
    attributions = shap_attributions(
        features,
        seed=scorer.meta["seed"],
        n_samples=scorer.meta["n_samples"],
        model_path=scorer.model_path,
    )
    return round(probability, 6), attributions