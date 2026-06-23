from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..probes import NearestCentroidProbe
from .base import DomainFitResult, DomainProbeResult, DomainSolveResult


def extract_python_code(text: str) -> str:
    """Extract code from a fenced block or return the raw text."""

    value = str(text).strip()
    if "```" not in value:
        return value
    parts = value.split("```")
    for part in parts[1::2]:
        candidate = part
        if candidate.lstrip().startswith("python"):
            candidate = candidate.lstrip()[6:]
        candidate = candidate.strip("\n")
        if candidate.strip():
            return candidate
    return value


class _PythonFactVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: list[dict[str, Any]] = []
        self.classes: list[str] = []
        self.imports: list[str] = []
        self.calls: list[str] = []
        self.assignments: list[str] = []
        self.returns = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append({
            "name": node.name,
            "args": [arg.arg for arg in node.args.args],
            "returns": ast.unparse(node.returns) if node.returns is not None else "",
        })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        self.imports.extend(f"{module}.{alias.name}".strip(".") for alias in node.names)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(self._call_name(node.func))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assignments.extend(ast.unparse(target) for target in node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.assignments.append(ast.unparse(node.target))
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns += 1
        self.generic_visit(node)

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._call_name(node.value)}.{node.attr}"
        return ast.unparse(node)


def analyze_python_source(text: str) -> dict[str, Any]:
    code = extract_python_code(text)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "syntax_ok": False,
            "error_type": "SyntaxError",
            "error": str(exc),
            "line": exc.lineno,
            "offset": exc.offset,
            "code": code,
        }
    visitor = _PythonFactVisitor()
    visitor.visit(tree)
    return {
        "syntax_ok": True,
        "error_type": "",
        "functions": visitor.functions,
        "classes": visitor.classes,
        "imports": sorted(set(visitor.imports)),
        "calls": sorted(set(item for item in visitor.calls if item)),
        "assignments": sorted(set(visitor.assignments)),
        "return_count": visitor.returns,
        "node_count": sum(1 for _ in ast.walk(tree)),
        "code": code,
    }


@dataclass(frozen=True)
class CodeProbeRow:
    source: str
    kind: str
    primary: str
    arg_count: str
    return_count: str
    import_count: str
    call_count: str
    assignment_count: str
    syntax_ok: str

    def labels(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "primary": self.primary,
            "arg_count": self.arg_count,
            "return_count": self.return_count,
            "import_count": self.import_count,
            "call_count": self.call_count,
            "assignment_count": self.assignment_count,
            "syntax_ok": self.syntax_ok,
        }


CODE_FACTOR_NAMES = (
    "kind",
    "primary",
    "arg_count",
    "return_count",
    "import_count",
    "call_count",
    "assignment_count",
    "syntax_ok",
)


def _stable_bucket(label: str, width: int) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % max(1, int(width))


def _add_bucket(features: np.ndarray, start: int, width: int, label: str, weight: float) -> None:
    if width <= 0 or start >= features.size:
        return
    usable_width = min(width, features.size - start)
    features[start + _stable_bucket(label, usable_width)] += float(weight)


def _code_factor_from_analysis(source: str, analysis: dict[str, Any]) -> CodeProbeRow:
    syntax_ok = bool(analysis.get("syntax_ok", False))
    if not syntax_ok:
        kind = "syntax_error"
        primary = str(analysis.get("error_type", "SyntaxError"))
        arg_count = 0
    else:
        functions = list(analysis.get("functions", []))
        classes = list(analysis.get("classes", []))
        imports = list(analysis.get("imports", []))
        calls = list(analysis.get("calls", []))
        assignments = list(analysis.get("assignments", []))
        if functions:
            kind = "function"
            primary = str(functions[0].get("name", "function"))
            arg_count = len(functions[0].get("args", []))
        elif classes:
            kind = "class"
            primary = str(classes[0])
            arg_count = 0
        elif imports:
            kind = "import"
            primary = str(imports[0])
            arg_count = 0
        elif calls:
            kind = "call"
            primary = str(calls[0])
            arg_count = 0
        elif assignments:
            kind = "assignment"
            primary = str(assignments[0])
            arg_count = 0
        else:
            kind = "module"
            primary = "module"
            arg_count = 0
    return CodeProbeRow(
        source=source,
        kind=kind,
        primary=primary,
        arg_count=str(arg_count),
        return_count=str(int(analysis.get("return_count", 0) or 0)),
        import_count=str(len(analysis.get("imports", []) or [])),
        call_count=str(len(analysis.get("calls", []) or [])),
        assignment_count=str(len(analysis.get("assignments", []) or [])),
        syntax_ok=str(bool(analysis.get("syntax_ok", False))),
    )


def code_feature_vector(analysis: dict[str, Any], feature_dim: int = 128) -> np.ndarray:
    """AST-derived structured feature vector for a small code factor probe."""

    dim = max(32, int(feature_dim))
    features = np.zeros(dim, dtype=np.float32)
    row = _code_factor_from_analysis(str(analysis.get("code", "")), analysis)
    functions = [str(item.get("name", "")) for item in analysis.get("functions", [])]
    classes = [str(item) for item in analysis.get("classes", [])]
    imports = [str(item) for item in analysis.get("imports", [])]
    calls = [str(item) for item in analysis.get("calls", [])]
    assignments = [str(item) for item in analysis.get("assignments", [])]
    code = str(analysis.get("code", ""))

    features[0] = 1.0 if analysis.get("syntax_ok") else -1.0
    features[1] = min(float(analysis.get("node_count", 0) or 0) / 80.0, 1.0)
    features[2] = min(float(row.return_count) / 4.0, 1.0)
    features[3] = min(float(row.arg_count) / 8.0, 1.0)
    features[4] = min(float(row.import_count) / 4.0, 1.0)
    features[5] = min(float(row.call_count) / 8.0, 1.0)
    features[6] = min(float(row.assignment_count) / 4.0, 1.0)
    features[7] = min(float(max(1, code.count("\n") + 1)) / 16.0, 1.0)

    segments = (
        ("kind", 16, [row.kind], 3.0),
        ("primary", 32, [row.primary], 3.0),
        ("function", 16, functions, 1.5),
        ("class", 12, classes, 1.5),
        ("import", 16, imports, 1.5),
        ("call", 16, calls, 1.5),
        ("assignment", dim, assignments, 1.5),
    )
    start = 8
    for prefix, width, labels, weight in segments:
        if start >= dim:
            break
        usable_width = min(width, dim - start)
        for label in labels:
            if label:
                _add_bucket(features, start, usable_width, f"{prefix}:{label}", weight)
        start += usable_width
    norm = float(np.linalg.norm(features))
    if norm > 0.0:
        features = features / norm
    return features


def make_code_probe_rows() -> list[CodeProbeRow]:
    sources = [
        "def add(a, b):\n    return a + b",
        "def add(left, right):\n    return left + right",
        "def add(x, y):\n    return x + y",
        "def square(x):\n    return x * x",
        "def square(value):\n    return value * value",
        "class User:\n    pass",
        "class User:\n    pass\n",
        "class Order:\n    pass",
        "class Order:\n    pass\n",
        "import json",
        "import json\n",
        "from pathlib import Path",
        "from pathlib import Path\n",
        "value = 1",
        "value = 2",
        "total = 3",
        "total = 4",
        "print('hello')",
        "print(1 + 2)",
        "def broken(:\n    pass",
        "if True print('x')",
    ]
    return [
        _code_factor_from_analysis(source, analyze_python_source(source))
        for source in sources
    ]


def _split_rows_by_primary(rows: list[CodeProbeRow], train_fraction: float) -> tuple[list[CodeProbeRow], list[CodeProbeRow]]:
    grouped: dict[tuple[str, str], list[CodeProbeRow]] = {}
    for row in rows:
        grouped.setdefault((row.kind, row.primary), []).append(row)
    train: list[CodeProbeRow] = []
    heldout: list[CodeProbeRow] = []
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) <= 1:
            train.extend(group)
            continue
        n_train = max(1, min(len(group) - 1, int(round(len(group) * train_fraction))))
        train.extend(group[:n_train])
        heldout.extend(group[n_train:])
    return train, heldout or train


def _features_for_rows(rows: list[CodeProbeRow], feature_dim: int) -> np.ndarray:
    return np.asarray(
        [code_feature_vector(analyze_python_source(row.source), feature_dim) for row in rows],
        dtype=np.float32,
    )


def _evaluate_code_readout(
    readout: CodeFactorReadout,
    rows: list[CodeProbeRow],
    *,
    max_examples: int = 8,
) -> dict[str, Any]:
    features = _features_for_rows(rows, int(readout.config["feature_dim"]))
    factor_metrics: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    for name in CODE_FACTOR_NAMES:
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
            "passed": all(decoded[name] == row.labels()[name] for name in CODE_FACTOR_NAMES),
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


@dataclass
class CodeFactorReadout:
    """Persistent centroid readout over AST-derived code factors."""

    config: dict[str, Any]
    probes: dict[str, NearestCentroidProbe]

    def predict_feature(self, feature: np.ndarray) -> dict[str, str]:
        vector = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        return {
            name: self.probes[name].predict(vector)[0]
            for name in CODE_FACTOR_NAMES
        }

    def solve(self, text: str) -> dict[str, Any]:
        analysis = analyze_python_source(text)
        feature = code_feature_vector(analysis, int(self.config["feature_dim"]))
        decoded = self.predict_feature(feature)
        return {
            "status": "ok",
            "decoded": decoded,
            "analysis": analysis,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "code-factor-centroid",
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
    def load(cls, artifact_dir: str | Path) -> CodeFactorReadout:
        path = Path(artifact_dir)
        payload = json.loads((path / "readout.json").read_text(encoding="utf-8"))
        return cls(
            config=dict(payload.get("config", {})),
            probes={
                str(name): NearestCentroidProbe.from_dict(value)
                for name, value in payload.get("probes", {}).items()
            },
        )


def fit_code_factor_readout(
    *,
    feature_dim: int = 128,
    train_fraction: float = 0.65,
) -> tuple[CodeFactorReadout, dict[str, Any]]:
    rows = make_code_probe_rows()
    train_rows, heldout_rows = _split_rows_by_primary(rows, train_fraction)
    train_features = _features_for_rows(train_rows, feature_dim)
    probes: dict[str, NearestCentroidProbe] = {}
    for name in CODE_FACTOR_NAMES:
        probe = NearestCentroidProbe()
        probe.fit(train_features, [row.labels()[name] for row in train_rows])
        probes[name] = probe
    readout = CodeFactorReadout(
        config={
            "feature_dim": int(feature_dim),
            "train_fraction": float(train_fraction),
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
        },
        probes=probes,
    )
    return readout, {
        "type": "code-factor-centroid",
        "fit": {
            "rows": len(rows),
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
        },
        "train_evaluation": _evaluate_code_readout(readout, train_rows),
        "evaluation": _evaluate_code_readout(readout, heldout_rows),
    }


class CodeDomain:
    """Python structure adapter with exact facts plus measured factor readout."""

    name = "code"

    def __init__(self, *, readout: CodeFactorReadout | None = None, feature_dim: int = 128) -> None:
        self.readout = readout
        self.feature_dim = int(feature_dim)

    @classmethod
    def load(cls, artifact_dir: str | Path) -> CodeDomain:
        path = Path(artifact_dir)
        readout = CodeFactorReadout.load(path)
        return cls(readout=readout, feature_dim=int(readout.config.get("feature_dim", 128)))

    def fit(self, out_dir: str | Path) -> DomainFitResult:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        readout, summary = fit_code_factor_readout(feature_dim=self.feature_dim)
        readout.save(path)
        self.readout = readout
        evaluation = summary["evaluation"]
        (path / "manifest.json").write_text(
            json.dumps({
                "type": "code-factor-domain",
                "version": 1,
                "feature_dim": self.feature_dim,
                "summary": summary,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
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
            notes=["Code domain keeps exact AST facts and adds a persisted AST-factor centroid readout."],
        )

    def analyze(self, text: str) -> dict[str, Any]:
        return analyze_python_source(text)

    def probe(self) -> DomainProbeResult:
        examples = [
            ("def add(a, b):\n    return a + b", {"syntax_ok": True, "function": "add", "return_count": 1}),
            ("import json\nvalue = json.loads('{}')", {"syntax_ok": True, "import": "json", "assignment": "value"}),
            ("def broken(:\n    pass", {"syntax_ok": False, "error_type": "SyntaxError"}),
        ]
        rows = []
        passed = 0
        for source, expected in examples:
            result = self.analyze(source)
            ok = result["syntax_ok"] == expected["syntax_ok"]
            if "function" in expected:
                ok = ok and any(fn["name"] == expected["function"] for fn in result.get("functions", []))
            if "import" in expected:
                ok = ok and expected["import"] in result.get("imports", [])
            if "assignment" in expected:
                ok = ok and expected["assignment"] in result.get("assignments", [])
            if "return_count" in expected:
                ok = ok and result.get("return_count") == expected["return_count"]
            if "error_type" in expected:
                ok = ok and result.get("error_type") == expected["error_type"]
            passed += int(ok)
            rows.append({"input": source, "expected": expected, "observed": result, "passed": ok})
        exact_accuracy = passed / len(examples)
        if self.readout is None:
            readout, _summary = fit_code_factor_readout(feature_dim=self.feature_dim)
        else:
            readout = self.readout
        factor_eval = _evaluate_code_readout(readout, make_code_probe_rows())
        return DomainProbeResult(
            domain=self.name,
            passed=exact_accuracy == 1.0 and bool(factor_eval["passed_factor_gate"]),
            metrics={
                "exact_ast_accuracy": exact_accuracy,
                "factor_mean_accuracy": float(factor_eval["mean_accuracy"]),
                "factor_min_accuracy": float(factor_eval["min_accuracy"]),
                "passed_factor_gate": bool(factor_eval["passed_factor_gate"]),
                "examples": len(examples),
            },
            examples=rows + factor_eval["examples"],
            notes=["Code factors are AST-derived and measured with a persisted centroid readout; this is not a learned code model."],
        )

    def solve(self, text: str) -> DomainSolveResult:
        data = self.analyze(text)
        if self.readout is not None:
            data = dict(data)
            data["factor_readout"] = self.readout.solve(text)["decoded"]
        answer = "syntax_ok" if data.get("syntax_ok") else f"syntax_error:{data.get('line')}"
        return DomainSolveResult(
            domain=self.name,
            status="ok",
            answer=answer,
            confidence=1.0,
            data=data,
            notes=["Decoded with Python ast; factor_readout reports the persisted AST-factor gate when loaded."],
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "code_factor_readout",
            "feature_dim": self.feature_dim,
            "has_readout": self.readout is not None,
        }
