from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .model import PhaseModel


OP_WORDS = {
    "add": "plus",
    "sub": "minus",
    "mul": "times",
}


@dataclass(frozen=True)
class ArithmeticProbeRow:
    prompt: str
    operation: str
    left: str
    right: str

    @property
    def pair(self) -> str:
        return f"{self.left},{self.right}"

    @property
    def result(self) -> str:
        return str(resolve_arithmetic_result(self.operation, self.left, self.right))


class NearestCentroidProbe:
    """Tiny nonparametric probe for basin separability."""

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.centroids: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> None:
        features = np.asarray(features, dtype=np.float32)
        labels = [str(item) for item in labels]
        unique = sorted(set(labels))
        centroids = []
        for label in unique:
            mask = np.asarray([item == label for item in labels], dtype=bool)
            centroids.append(np.mean(features[mask], axis=0))
        self.labels = unique
        self.centroids = np.asarray(centroids, dtype=np.float32)

    def predict(self, features: np.ndarray) -> list[str]:
        if self.centroids is None:
            raise ValueError("probe has not been fitted")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        distances = np.linalg.norm(features[:, None, :] - self.centroids[None, :, :], axis=2)
        indices = np.argmin(distances, axis=1)
        return [self.labels[int(index)] for index in indices]

    def to_dict(self) -> dict[str, Any]:
        if self.centroids is None:
            raise ValueError("probe has not been fitted")
        return {
            "labels": list(self.labels),
            "centroids": self.centroids.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NearestCentroidProbe:
        probe = cls()
        probe.labels = [str(item) for item in payload.get("labels", [])]
        probe.centroids = np.asarray(payload.get("centroids", []), dtype=np.float32)
        if probe.centroids.ndim != 2 or len(probe.labels) != int(probe.centroids.shape[0]):
            raise ValueError("invalid centroid probe payload")
        return probe


@dataclass
class ArithmeticFactorReadout:
    """Persistent structured-basin arithmetic readout."""

    config: dict[str, Any]
    probes: dict[str, NearestCentroidProbe]

    def _model(self) -> PhaseModel:
        return PhaseModel(
            grid_size=int(self.config["grid_size"]),
            basin_dim=int(self.config["basin_dim"]),
            hidden=int(self.config.get("hidden", 64)),
            seed=int(self.config.get("seed", 7)),
            backend=str(self.config.get("backend", "auto")),
            encoder_mode="structured",
            structured_result_hint=bool(self.config.get("structured_result_hint", False)),
            structured_feature_strength=float(self.config.get("structured_feature_strength", 2.0)),
            create_decoder=False,
        )

    def solve(self, prompt: str) -> dict[str, Any]:
        model = self._model()
        basin, prediction_error = model.encode_basin(
            prompt,
            steps_per_chunk=int(self.config["steps_per_chunk"]),
            reset=True,
        )
        feature = np.asarray(basin.center, dtype=np.float32).reshape(1, -1)
        decoded = {
            "operation": self.probes["operation"].predict(feature)[0],
            "left": self.probes["left"].predict(feature)[0],
            "right": self.probes["right"].predict(feature)[0],
        }
        try:
            result = resolve_arithmetic_result(decoded["operation"], decoded["left"], decoded["right"])
            status = "ok"
            error = ""
        except (TypeError, ValueError) as exc:
            result = None
            status = "error"
            error = str(exc)
        return {
            "status": status,
            "prompt": str(prompt),
            "decoded": decoded,
            "result": result,
            "answer": "" if result is None else str(result),
            "error": error,
            "readout": {
                "type": "arithmetic-factor-centroid",
                "ops": list(self.config.get("ops", [])),
                "min_value": int(self.config.get("min_value", 0)),
                "max_value": int(self.config.get("max_value", 0)),
            },
            "encoder_mode": "structured",
            "structured_result_hint": bool(self.config.get("structured_result_hint", False)),
            "structured_feature_strength": float(self.config.get("structured_feature_strength", 2.0)),
            "grid_size": int(self.config["grid_size"]),
            "basin_dim": int(self.config["basin_dim"]),
            "steps_per_chunk": int(self.config["steps_per_chunk"]),
            "prompt_prediction_error": float(prediction_error),
            "basin": {
                "x": int(basin.x),
                "y": int(basin.y),
                "coherence": float(basin.coherence),
                "gradient": float(basin.gradient),
                "energy": float(basin.energy),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "arithmetic-factor-centroid",
            "version": 1,
            "config": dict(self.config),
            "probes": {
                name: probe.to_dict()
                for name, probe in sorted(self.probes.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArithmeticFactorReadout:
        if payload.get("type") != "arithmetic-factor-centroid":
            raise ValueError("unsupported arithmetic readout type")
        probes = {
            str(name): NearestCentroidProbe.from_dict(value)
            for name, value in dict(payload.get("probes", {})).items()
        }
        for required in ("operation", "left", "right"):
            if required not in probes:
                raise ValueError(f"missing readout probe: {required}")
        return cls(config=dict(payload.get("config", {})), probes=probes)

    def save(self, out_dir: str | Path) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        readout_path = path / "readout.json"
        readout_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return readout_path

    @classmethod
    def load(cls, readout_dir: str | Path) -> ArithmeticFactorReadout:
        path = Path(readout_dir)
        readout_path = path / "readout.json" if path.is_dir() else path
        return cls.from_dict(json.loads(readout_path.read_text(encoding="utf-8")))


def make_arithmetic_probe_rows(
    *,
    max_value: int = 20,
    min_value: int = 0,
    ops: Sequence[str] = ("add", "mul"),
) -> list[ArithmeticProbeRow]:
    rows: list[ArithmeticProbeRow] = []
    for operation in ops:
        if operation not in OP_WORDS:
            continue
        word = OP_WORDS[operation]
        for left in range(int(min_value), int(max_value) + 1):
            for right in range(int(min_value), int(max_value) + 1):
                rows.append(
                    ArithmeticProbeRow(
                        prompt=f"{left} {word} {right}",
                        operation=operation,
                        left=str(left),
                        right=str(right),
                    )
                )
    return rows


def resolve_arithmetic_result(operation: str, left: str | int, right: str | int) -> int:
    """Resolve a decoded arithmetic tuple into an integer result."""

    op = str(operation)
    a = int(left)
    b = int(right)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    raise ValueError(f"unsupported arithmetic operation: {operation!r}")


def split_rows(
    rows: Sequence[ArithmeticProbeRow],
    *,
    train_fraction: float = 0.7,
    seed: int = 7,
) -> tuple[list[ArithmeticProbeRow], list[ArithmeticProbeRow]]:
    shuffled = list(rows)
    random.Random(int(seed)).shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1, round(len(shuffled) * float(train_fraction))))
    return shuffled[:cut], shuffled[cut:]


def encode_probe_rows(
    model: PhaseModel,
    rows: Sequence[ArithmeticProbeRow],
    *,
    steps_per_chunk: int = 12,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = []
    basin_rows = []
    for row in rows:
        basin, prediction_error = model.encode_basin(
            row.prompt,
            steps_per_chunk=steps_per_chunk,
            reset=True,
        )
        features.append(np.asarray(basin.center, dtype=np.float32))
        basin_rows.append({
            "prompt": row.prompt,
            "operation": row.operation,
            "left": row.left,
            "right": row.right,
            "x": int(basin.x),
            "y": int(basin.y),
            "coherence": float(basin.coherence),
            "gradient": float(basin.gradient),
            "energy": float(basin.energy),
            "prediction_error": float(prediction_error),
        })
    return np.asarray(features, dtype=np.float32), basin_rows


def _accuracy(predicted: Sequence[str], expected: Sequence[str]) -> float:
    expected = [str(item) for item in expected]
    predicted = [str(item) for item in predicted]
    if not expected:
        return 0.0
    return sum(int(p == e) for p, e in zip(predicted, expected)) / len(expected)


def _fit_eval_probe(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: Sequence[str],
    test_labels: Sequence[str],
) -> dict[str, Any]:
    probe = NearestCentroidProbe()
    probe.fit(train_features, train_labels)
    predictions = probe.predict(test_features)
    return {
        "accuracy": _accuracy(predictions, test_labels),
        "labels": len(probe.labels),
        "predictions": predictions,
        "examples": [
            {"expected": str(expected), "predicted": str(predicted)}
            for expected, predicted in list(zip(test_labels, predictions))[:12]
        ],
    }


def run_arithmetic_representation_probe(
    *,
    encoder_mode: str = "structured",
    max_value: int = 20,
    min_value: int = 0,
    ops: Sequence[str] = ("add", "mul"),
    grid_size: int = 64,
    basin_dim: int = 128,
    hidden: int = 64,
    steps_per_chunk: int = 12,
    seed: int = 7,
    backend: str = "auto",
    train_fraction: float = 0.7,
    structured_result_hint: bool = False,
    structured_feature_strength: float = 2.0,
) -> dict[str, Any]:
    rows = make_arithmetic_probe_rows(max_value=max_value, min_value=min_value, ops=ops)
    train_rows, test_rows = split_rows(rows, train_fraction=train_fraction, seed=seed)
    model = PhaseModel(
        grid_size=grid_size,
        basin_dim=basin_dim,
        hidden=hidden,
        seed=seed,
        backend=backend,
        encoder_mode=encoder_mode,
        structured_result_hint=structured_result_hint,
        structured_feature_strength=structured_feature_strength,
        create_decoder=False,
    )
    train_features, train_basin_rows = encode_probe_rows(
        model,
        train_rows,
        steps_per_chunk=steps_per_chunk,
    )
    test_features, test_basin_rows = encode_probe_rows(
        model,
        test_rows,
        steps_per_chunk=steps_per_chunk,
    )

    train_labels = {
        "operation": [row.operation for row in train_rows],
        "left": [row.left for row in train_rows],
        "right": [row.right for row in train_rows],
    }
    test_labels = {
        "operation": [row.operation for row in test_rows],
        "left": [row.left for row in test_rows],
        "right": [row.right for row in test_rows],
    }
    probes = {
        name: _fit_eval_probe(train_features, test_features, train_labels[name], test_labels[name])
        for name in train_labels
    }
    left_predictions = probes["left"]["predictions"]
    right_predictions = probes["right"]["predictions"]
    pair_expected = [row.pair for row in test_rows]
    pair_predictions = [
        f"{left},{right}"
        for left, right in zip(left_predictions, right_predictions)
    ]
    probes["pair"] = {
        "accuracy": _accuracy(pair_predictions, pair_expected),
        "labels": len(set(row.pair for row in rows)),
        "composed_from": ["left", "right"],
        "examples": [
            {"expected": str(expected), "predicted": str(predicted)}
            for expected, predicted in list(zip(pair_expected, pair_predictions))[:12]
        ],
    }
    for probe_result in probes.values():
        probe_result.pop("predictions", None)
    thresholds = {
        "operation": 0.95,
        "left": 0.95,
        "right": 0.95,
        "pair": 0.80,
    }
    passed = all(float(probes[name]["accuracy"]) >= threshold for name, threshold in thresholds.items())
    return {
        "encoder_mode": encoder_mode,
        "structured_result_hint": bool(structured_result_hint),
        "structured_feature_strength": float(structured_feature_strength),
        "grid_size": int(grid_size),
        "basin_dim": int(basin_dim),
        "steps_per_chunk": int(steps_per_chunk),
        "seed": int(seed),
        "ops": list(ops),
        "min_value": int(min_value),
        "max_value": int(max_value),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "probes": probes,
        "thresholds": thresholds,
        "passed_representation_gate": bool(passed),
        "mean_train_prediction_error": float(np.mean([row["prediction_error"] for row in train_basin_rows])),
        "mean_test_prediction_error": float(np.mean([row["prediction_error"] for row in test_basin_rows])),
    }


def run_arithmetic_result_readout_probe(
    *,
    encoder_mode: str = "structured",
    max_value: int = 20,
    min_value: int = 0,
    ops: Sequence[str] = ("add", "mul"),
    grid_size: int = 64,
    basin_dim: int = 128,
    hidden: int = 64,
    steps_per_chunk: int = 12,
    seed: int = 7,
    backend: str = "auto",
    train_fraction: float = 0.7,
    structured_result_hint: bool = False,
    structured_feature_strength: float = 2.0,
) -> dict[str, Any]:
    """Probe whether basin features can drive exact arithmetic readout.

    This is intentionally factorized: first decode operation and operands, then
    apply the operation. A direct result probe is reported alongside it as a
    control, because direct result labels can fail even when the basin contains
    all information needed to compute the answer.
    """

    rows = make_arithmetic_probe_rows(max_value=max_value, min_value=min_value, ops=ops)
    train_rows, test_rows = split_rows(rows, train_fraction=train_fraction, seed=seed)
    model = PhaseModel(
        grid_size=grid_size,
        basin_dim=basin_dim,
        hidden=hidden,
        seed=seed,
        backend=backend,
        encoder_mode=encoder_mode,
        structured_result_hint=structured_result_hint,
        structured_feature_strength=structured_feature_strength,
        create_decoder=False,
    )
    train_features, train_basin_rows = encode_probe_rows(
        model,
        train_rows,
        steps_per_chunk=steps_per_chunk,
    )
    test_features, test_basin_rows = encode_probe_rows(
        model,
        test_rows,
        steps_per_chunk=steps_per_chunk,
    )

    train_labels = {
        "operation": [row.operation for row in train_rows],
        "left": [row.left for row in train_rows],
        "right": [row.right for row in train_rows],
        "result": [row.result for row in train_rows],
    }
    test_labels = {
        "operation": [row.operation for row in test_rows],
        "left": [row.left for row in test_rows],
        "right": [row.right for row in test_rows],
        "result": [row.result for row in test_rows],
    }
    probes = {
        name: _fit_eval_probe(train_features, test_features, train_labels[name], test_labels[name])
        for name in ("operation", "left", "right", "result")
    }
    op_predictions = probes["operation"]["predictions"]
    left_predictions = probes["left"]["predictions"]
    right_predictions = probes["right"]["predictions"]
    resolved_predictions: list[str] = []
    factor_errors = 0
    for operation, left, right in zip(op_predictions, left_predictions, right_predictions):
        try:
            resolved_predictions.append(str(resolve_arithmetic_result(operation, left, right)))
        except (TypeError, ValueError):
            factor_errors += 1
            resolved_predictions.append("")
    factorized_examples = [
        {
            "prompt": row.prompt,
            "decoded_operation": str(operation),
            "decoded_left": str(left),
            "decoded_right": str(right),
            "expected": str(expected),
            "predicted": str(predicted),
        }
        for row, operation, left, right, expected, predicted in list(
            zip(
                test_rows,
                op_predictions,
                left_predictions,
                right_predictions,
                test_labels["result"],
                resolved_predictions,
            )
        )[:12]
    ]
    factorized_result = {
        "accuracy": _accuracy(resolved_predictions, test_labels["result"]),
        "factor_errors": int(factor_errors),
        "examples": factorized_examples,
    }
    direct_result = {
        key: value
        for key, value in probes["result"].items()
        if key != "predictions"
    }
    for probe_result in probes.values():
        probe_result.pop("predictions", None)
    thresholds = {
        "operation": 0.95,
        "left": 0.95,
        "right": 0.95,
        "factorized_result": 0.95,
    }
    passed = (
        float(probes["operation"]["accuracy"]) >= thresholds["operation"]
        and float(probes["left"]["accuracy"]) >= thresholds["left"]
        and float(probes["right"]["accuracy"]) >= thresholds["right"]
        and float(factorized_result["accuracy"]) >= thresholds["factorized_result"]
    )
    return {
        "encoder_mode": encoder_mode,
        "structured_result_hint": bool(structured_result_hint),
        "structured_feature_strength": float(structured_feature_strength),
        "grid_size": int(grid_size),
        "basin_dim": int(basin_dim),
        "steps_per_chunk": int(steps_per_chunk),
        "seed": int(seed),
        "ops": list(ops),
        "min_value": int(min_value),
        "max_value": int(max_value),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "factor_probes": {
            "operation": probes["operation"],
            "left": probes["left"],
            "right": probes["right"],
        },
        "direct_result_probe": direct_result,
        "factorized_result": factorized_result,
        "thresholds": thresholds,
        "passed_result_gate": bool(passed),
        "mean_train_prediction_error": float(np.mean([row["prediction_error"] for row in train_basin_rows])),
        "mean_test_prediction_error": float(np.mean([row["prediction_error"] for row in test_basin_rows])),
        "note": "Factorized readout computes from decoded basin factors; direct_result_probe is the non-factorized control.",
    }


def fit_arithmetic_factor_readout(
    *,
    max_value: int = 20,
    min_value: int = 0,
    ops: Sequence[str] = ("add", "mul"),
    grid_size: int = 64,
    basin_dim: int = 128,
    hidden: int = 64,
    steps_per_chunk: int = 12,
    seed: int = 7,
    backend: str = "auto",
    structured_result_hint: bool = False,
    structured_feature_strength: float = 2.0,
) -> tuple[ArithmeticFactorReadout, dict[str, Any]]:
    """Fit a persistent factor readout over the declared arithmetic domain."""

    rows = make_arithmetic_probe_rows(max_value=max_value, min_value=min_value, ops=ops)
    config = {
        "grid_size": int(grid_size),
        "basin_dim": int(basin_dim),
        "hidden": int(hidden),
        "steps_per_chunk": int(steps_per_chunk),
        "seed": int(seed),
        "backend": str(backend),
        "structured_result_hint": bool(structured_result_hint),
        "structured_feature_strength": float(structured_feature_strength),
        "ops": list(ops),
        "min_value": int(min_value),
        "max_value": int(max_value),
    }
    model = PhaseModel(
        grid_size=grid_size,
        basin_dim=basin_dim,
        hidden=hidden,
        seed=seed,
        backend=backend,
        encoder_mode="structured",
        structured_result_hint=structured_result_hint,
        structured_feature_strength=structured_feature_strength,
        create_decoder=False,
    )
    features, basin_rows = encode_probe_rows(
        model,
        rows,
        steps_per_chunk=steps_per_chunk,
    )
    probes: dict[str, NearestCentroidProbe] = {}
    for name, labels in {
        "operation": [row.operation for row in rows],
        "left": [row.left for row in rows],
        "right": [row.right for row in rows],
    }.items():
        probe = NearestCentroidProbe()
        probe.fit(features, labels)
        probes[name] = probe
    readout = ArithmeticFactorReadout(config=config, probes=probes)
    summary = {
        "type": "arithmetic-factor-centroid",
        "config": config,
        "rows": len(rows),
        "probe_labels": {
            name: len(probe.labels)
            for name, probe in sorted(probes.items())
        },
        "mean_fit_prediction_error": float(np.mean([row["prediction_error"] for row in basin_rows])),
    }
    return readout, summary


def fit_save_arithmetic_factor_readout(
    out_dir: str | Path,
    *,
    max_value: int = 20,
    min_value: int = 0,
    ops: Sequence[str] = ("add", "mul"),
    grid_size: int = 64,
    basin_dim: int = 128,
    hidden: int = 64,
    steps_per_chunk: int = 12,
    seed: int = 7,
    backend: str = "auto",
    train_fraction: float = 0.7,
    structured_result_hint: bool = False,
    structured_feature_strength: float = 2.0,
) -> dict[str, Any]:
    """Fit, evaluate, and save a reusable arithmetic factor readout."""

    ops = tuple(ops)
    evaluation = run_arithmetic_result_readout_probe(
        encoder_mode="structured",
        max_value=max_value,
        min_value=min_value,
        ops=ops,
        grid_size=grid_size,
        basin_dim=basin_dim,
        hidden=hidden,
        steps_per_chunk=steps_per_chunk,
        seed=seed,
        backend=backend,
        train_fraction=train_fraction,
        structured_result_hint=structured_result_hint,
        structured_feature_strength=structured_feature_strength,
    )
    readout, fit_summary = fit_arithmetic_factor_readout(
        max_value=max_value,
        min_value=min_value,
        ops=ops,
        grid_size=grid_size,
        basin_dim=basin_dim,
        hidden=hidden,
        steps_per_chunk=steps_per_chunk,
        seed=seed,
        backend=backend,
        structured_result_hint=structured_result_hint,
        structured_feature_strength=structured_feature_strength,
    )
    out_path = Path(out_dir)
    readout_path = readout.save(out_path)
    summary = {
        "status": "ok",
        "readout_path": str(readout_path),
        "fit": fit_summary,
        "evaluation": evaluation,
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def solve_arithmetic_with_factor_readout(
    prompt: str,
    *,
    max_value: int = 20,
    min_value: int = 0,
    ops: Sequence[str] = ("add", "mul"),
    grid_size: int = 64,
    basin_dim: int = 128,
    hidden: int = 64,
    steps_per_chunk: int = 12,
    seed: int = 7,
    backend: str = "auto",
    structured_result_hint: bool = False,
    structured_feature_strength: float = 2.0,
) -> dict[str, Any]:
    """Solve one arithmetic prompt through the structured basin factor readout."""

    readout, summary = fit_arithmetic_factor_readout(
        max_value=max_value,
        min_value=min_value,
        ops=ops,
        grid_size=grid_size,
        basin_dim=basin_dim,
        hidden=hidden,
        steps_per_chunk=steps_per_chunk,
        seed=seed,
        backend=backend,
        structured_result_hint=structured_result_hint,
        structured_feature_strength=structured_feature_strength,
    )
    payload = readout.solve(prompt)
    payload["readout_rows"] = int(summary["rows"])
    payload["mean_readout_prediction_error"] = float(summary["mean_fit_prediction_error"])
    return payload


def save_probe_result(result: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
