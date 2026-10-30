from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..probes import NearestCentroidProbe
from .base import DomainFitResult, DomainProbeResult, DomainSolveResult


JSON_FACTOR_NAMES = (
    "root_type",
    "key_signature",
    "key_count",
    "array_length",
    "depth",
    "scalar_kind",
    "syntax_ok",
)


def analyze_json_text(text: str) -> dict[str, Any]:
    value = str(text).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return {
            "syntax_ok": False,
            "error_type": "JSONDecodeError",
            "error": str(exc),
            "line": exc.lineno,
            "column": exc.colno,
            "source": value,
        }
    stats = _json_stats(parsed)
    return {
        "syntax_ok": True,
        "error_type": "",
        "root_type": _json_type(parsed),
        "top_level_keys": sorted(parsed.keys()) if isinstance(parsed, dict) else [],
        "key_count": len(parsed) if isinstance(parsed, dict) else 0,
        "array_length": len(parsed) if isinstance(parsed, list) else 0,
        "depth": stats["depth"],
        "scalar_count": stats["scalar_count"],
        "object_count": stats["object_count"],
        "array_count": stats["array_count"],
        "scalar_kind": _scalar_kind(parsed),
        "source": value,
    }


def _json_type(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _scalar_kind(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return "container"
    return _json_type(value)


def _json_stats(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        if not value:
            return {"depth": 1, "scalar_count": 0, "object_count": 1, "array_count": 0}
        child = [_json_stats(item) for item in value.values()]
        return {
            "depth": 1 + max(item["depth"] for item in child),
            "scalar_count": sum(item["scalar_count"] for item in child),
            "object_count": 1 + sum(item["object_count"] for item in child),
            "array_count": sum(item["array_count"] for item in child),
        }
    if isinstance(value, list):
        if not value:
            return {"depth": 1, "scalar_count": 0, "object_count": 0, "array_count": 1}
        child = [_json_stats(item) for item in value]
        return {
            "depth": 1 + max(item["depth"] for item in child),
            "scalar_count": sum(item["scalar_count"] for item in child),
            "object_count": sum(item["object_count"] for item in child),
            "array_count": 1 + sum(item["array_count"] for item in child),
        }
    return {"depth": 1, "scalar_count": 1, "object_count": 0, "array_count": 0}


@dataclass(frozen=True)
class JsonProbeRow:
    source: str
    root_type: str
    key_signature: str
    key_count: str
    array_length: str
    depth: str
    scalar_kind: str
    syntax_ok: str

    def labels(self) -> dict[str, str]:
        return {
            "root_type": self.root_type,
            "key_signature": self.key_signature,
            "key_count": self.key_count,
            "array_length": self.array_length,
            "depth": self.depth,
            "scalar_kind": self.scalar_kind,
            "syntax_ok": self.syntax_ok,
        }


def _row_from_analysis(source: str, analysis: dict[str, Any]) -> JsonProbeRow:
    keys = analysis.get("top_level_keys", []) or []
    key_signature = ",".join(str(item) for item in keys) if keys else "-"
    if not analysis.get("syntax_ok"):
        return JsonProbeRow(
            source=source,
            root_type="invalid",
            key_signature="invalid",
            key_count="0",
            array_length="0",
            depth="0",
            scalar_kind="invalid",
            syntax_ok="False",
        )
    return JsonProbeRow(
        source=source,
        root_type=str(analysis.get("root_type", "")),
        key_signature=key_signature,
        key_count=str(int(analysis.get("key_count", 0) or 0)),
        array_length=str(int(analysis.get("array_length", 0) or 0)),
        depth=str(int(analysis.get("depth", 0) or 0)),
        scalar_kind=str(analysis.get("scalar_kind", "")),
        syntax_ok="True",
    )


def make_json_probe_rows() -> list[JsonProbeRow]:
    sources = [
        '{"ok": true}',
        '{"ok": false}',
        '{"count": 1}',
        '{"count": 2}',
        '{"name": "Ada"}',
        '{"name": "Bob"}',
        '{"ok": true, "count": 2}',
        '{"count": 3, "ok": false}',
        "[1, 2, 3]",
        "[4, 5, 6]",
        "[true, false]",
        "[false, true]",
        '{"user": {"id": 1}, "tags": ["a"]}',
        '{"tags": ["b"], "user": {"id": 2}}',
        '"hello"',
        '"phase"',
        "42",
        "7",
        "true",
        "false",
        '{"ok": true',
        "[1, 2,]",
    ]
    return [_row_from_analysis(source, analyze_json_text(source)) for source in sources]


def json_feature_vector(analysis: dict[str, Any], feature_dim: int = 96) -> np.ndarray:
    dim = max(32, int(feature_dim))
    features = np.zeros(dim, dtype=np.float32)
    row = _row_from_analysis(str(analysis.get("source", "")), analysis)

    features[0] = 1.0 if analysis.get("syntax_ok") else -1.0
    features[1] = min(float(analysis.get("key_count", 0) or 0) / 8.0, 1.0)
    features[2] = min(float(analysis.get("array_length", 0) or 0) / 16.0, 1.0)
    features[3] = min(float(analysis.get("depth", 0) or 0) / 8.0, 1.0)
    features[4] = min(float(analysis.get("scalar_count", 0) or 0) / 32.0, 1.0)
    features[5] = min(float(analysis.get("object_count", 0) or 0) / 8.0, 1.0)
    features[6] = min(float(analysis.get("array_count", 0) or 0) / 8.0, 1.0)

    start = 8
    for prefix, width, labels, weight in (
        ("root", 16, [row.root_type], 3.0),
        ("keys", 24, [row.key_signature], 3.0),
        ("scalar", 12, [row.scalar_kind], 2.0),
        ("syntax", 8, [row.syntax_ok], 2.0),
        ("source", dim, [str(analysis.get("source", ""))[:48]], 0.5),
    ):
        if start >= dim:
            break
        usable_width = min(width, dim - start)
        for label in labels:
            _add_bucket(features, start, usable_width, f"{prefix}:{label}", weight)
        start += usable_width
    norm = float(np.linalg.norm(features))
    if norm > 0.0:
        features = features / norm
    return features


def _add_bucket(features: np.ndarray, start: int, width: int, label: str, weight: float) -> None:
    if width <= 0:
        return
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % int(width)
    features[start + index] += float(weight)


def _split_rows(rows: list[JsonProbeRow], train_fraction: float) -> tuple[list[JsonProbeRow], list[JsonProbeRow]]:
    grouped: dict[tuple[str, str], list[JsonProbeRow]] = {}
    for row in rows:
        grouped.setdefault((row.root_type, row.key_signature, row.array_length), []).append(row)
    train: list[JsonProbeRow] = []
    heldout: list[JsonProbeRow] = []
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) <= 1:
            train.extend(group)
            continue
        n_train = max(1, min(len(group) - 1, int(round(len(group) * train_fraction))))
        train.extend(group[:n_train])
        heldout.extend(group[n_train:])
    return train, heldout or train


def _features_for_rows(rows: list[JsonProbeRow], feature_dim: int) -> np.ndarray:
    return np.asarray(
        [json_feature_vector(analyze_json_text(row.source), feature_dim) for row in rows],
        dtype=np.float32,
    )


@dataclass
class JsonFactorReadout:
    config: dict[str, Any]
    probes: dict[str, NearestCentroidProbe]

    def predict_feature(self, feature: np.ndarray) -> dict[str, str]:
        vector = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        return {
            name: self.probes[name].predict(vector)[0]
            for name in JSON_FACTOR_NAMES
        }

    def solve(self, text: str) -> dict[str, Any]:
        analysis = analyze_json_text(text)
        feature = json_feature_vector(analysis, int(self.config["feature_dim"]))
        return {
            "status": "ok",
            "decoded": self.predict_feature(feature),
            "analysis": analysis,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "json-factor-centroid",
            "version": 1,
            "config": self.config,
            "probes": {
                name: probe.to_dict()
                for name, probe in sorted(self.probes.items())
            },
        }

    def save(self, out_dir: str | Path) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "readout.json").write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, artifact_dir: str | Path) -> JsonFactorReadout:
        payload = json.loads((Path(artifact_dir) / "readout.json").read_text(encoding="utf-8"))
        return cls(
            config=dict(payload.get("config", {})),
            probes={
                str(name): NearestCentroidProbe.from_dict(value)
                for name, value in payload.get("probes", {}).items()
            },
        )


def fit_json_factor_readout(
    *,
    feature_dim: int = 96,
    train_fraction: float = 0.65,
) -> tuple[JsonFactorReadout, dict[str, Any]]:
    rows = make_json_probe_rows()
    train_rows, heldout_rows = _split_rows(rows, train_fraction)
    train_features = _features_for_rows(train_rows, feature_dim)
    probes: dict[str, NearestCentroidProbe] = {}
    for name in JSON_FACTOR_NAMES:
        probe = NearestCentroidProbe()
        probe.fit(train_features, [row.labels()[name] for row in train_rows])
        probes[name] = probe
    readout = JsonFactorReadout(
        config={
            "feature_dim": int(feature_dim),
            "train_fraction": float(train_fraction),
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
        },
        probes=probes,
    )
    return readout, {
        "type": "json-factor-centroid",
        "fit": {
            "rows": len(rows),
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
        },
        "train_evaluation": evaluate_json_readout(readout, train_rows),
        "evaluation": evaluate_json_readout(readout, heldout_rows),
    }


def evaluate_json_readout(
    readout: JsonFactorReadout,
    rows: list[JsonProbeRow],
    *,
    max_examples: int = 8,
) -> dict[str, Any]:
    features = _features_for_rows(rows, int(readout.config["feature_dim"]))
    factor_metrics: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    for name in JSON_FACTOR_NAMES:
        expected = [row.labels()[name] for row in rows]
        observed = readout.probes[name].predict(features)
        accuracy = sum(int(a == b) for a, b in zip(expected, observed)) / max(1, len(expected))
        factor_metrics[name] = {
            "accuracy": float(accuracy),
            "labels": len(readout.probes[name].labels),
        }
    for row, feature in zip(rows[:max_examples], features[:max_examples]):
        decoded = readout.predict_feature(feature)
        examples.append({
            "input": row.source,
            "expected": row.labels(),
            "observed": decoded,
            "passed": all(decoded[name] == row.labels()[name] for name in JSON_FACTOR_NAMES),
        })
    mean_accuracy = float(np.mean([item["accuracy"] for item in factor_metrics.values()]))
    min_accuracy = float(min(item["accuracy"] for item in factor_metrics.values()))
    return {
        "rows": len(rows),
        "mean_accuracy": mean_accuracy,
        "min_accuracy": min_accuracy,
        "factor_metrics": factor_metrics,
        "examples": examples,
        "passed_factor_gate": min_accuracy >= 0.9,
    }


class JsonDomain:
    """JSON structure adapter with exact parse facts plus measured factor readout."""

    name = "json"

    def __init__(self, *, readout: JsonFactorReadout | None = None, feature_dim: int = 96) -> None:
        self.readout = readout
        self.feature_dim = int(feature_dim)

    @classmethod
    def load(cls, artifact_dir: str | Path) -> JsonDomain:
        readout = JsonFactorReadout.load(artifact_dir)
        return cls(readout=readout, feature_dim=int(readout.config.get("feature_dim", 96)))

    def fit(self, out_dir: str | Path) -> DomainFitResult:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        readout, summary = fit_json_factor_readout(feature_dim=self.feature_dim)
        readout.save(path)
        self.readout = readout
        (path / "manifest.json").write_text(
            json.dumps({
                "type": "json-factor-domain",
                "version": 1,
                "feature_dim": self.feature_dim,
                "summary": summary,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        evaluation = summary["evaluation"]
        return DomainFitResult(
            domain=self.name,
            status="ok",
            artifact_dir=str(path),
            metrics={
                "rows": summary["fit"]["rows"],
                "heldout_rows": summary["fit"]["heldout_rows"],
                "passed_factor_gate": bool(evaluation["passed_factor_gate"]),
                "mean_factor_accuracy": float(evaluation["mean_accuracy"]),
                "min_factor_accuracy": float(evaluation["min_accuracy"]),
            },
            notes=["JSON domain keeps exact parse facts and adds a persisted structural-factor centroid readout."],
        )

    def analyze(self, text: str) -> dict[str, Any]:
        return analyze_json_text(text)

    def probe(self) -> DomainProbeResult:
        examples = [
            ('{"ok": true}', {"syntax_ok": True, "root_type": "object", "key_count": 1}),
            ("[1, 2, 3]", {"syntax_ok": True, "root_type": "array", "array_length": 3}),
            ('{"ok": true', {"syntax_ok": False, "error_type": "JSONDecodeError"}),
        ]
        rows = []
        passed = 0
        for source, expected in examples:
            result = self.analyze(source)
            ok = result["syntax_ok"] == expected["syntax_ok"]
            for key, value in expected.items():
                ok = ok and result.get(key) == value
            passed += int(ok)
            rows.append({"input": source, "expected": expected, "observed": result, "passed": ok})
        exact_accuracy = passed / len(examples)
        readout = self.readout or fit_json_factor_readout(feature_dim=self.feature_dim)[0]
        factor_eval = evaluate_json_readout(readout, make_json_probe_rows())
        return DomainProbeResult(
            domain=self.name,
            passed=exact_accuracy == 1.0 and bool(factor_eval["passed_factor_gate"]),
            metrics={
                "exact_json_accuracy": exact_accuracy,
                "factor_mean_accuracy": float(factor_eval["mean_accuracy"]),
                "factor_min_accuracy": float(factor_eval["min_accuracy"]),
                "passed_factor_gate": bool(factor_eval["passed_factor_gate"]),
                "examples": len(examples),
            },
            examples=rows + factor_eval["examples"],
            notes=["JSON factors are parser-derived and measured with a persisted centroid readout."],
        )

    def solve(self, text: str) -> DomainSolveResult:
        data = self.analyze(text)
        if self.readout is not None:
            data = dict(data)
            data["factor_readout"] = self.readout.solve(text)["decoded"]
        answer = str(data.get("root_type", "invalid")) if data.get("syntax_ok") else f"json_error:{data.get('line')}"
        return DomainSolveResult(
            domain=self.name,
            status="ok",
            answer=answer,
            confidence=1.0,
            data=data,
            notes=["Decoded with Python json; factor_readout reports the persisted structural gate when loaded."],
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "json_factor_readout",
            "feature_dim": self.feature_dim,
            "has_readout": self.readout is not None,
        }
