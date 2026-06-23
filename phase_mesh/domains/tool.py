from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..encoding_structured import parse_arithmetic
from .base import DomainFitResult, DomainProbeResult, DomainSolveResult
from .code import extract_python_code


class ToolDomain:
    """Deterministic routing adapter for early PhaseMesh actions."""

    name = "tool"

    def fit(self, out_dir: str | Path) -> DomainFitResult:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text(
            json.dumps({"type": "deterministic-tool-router", "version": 1}, indent=2) + "\n",
            encoding="utf-8",
        )
        return DomainFitResult(
            domain=self.name,
            status="ok",
            artifact_dir=str(path),
            metrics={"trainable": False},
            notes=["Tool routing is deterministic until enough traces exist for a learned gate."],
        )

    def route(self, text: str) -> dict[str, Any]:
        value = str(text).strip()
        lowered = value.lower()
        if parse_arithmetic(value) is not None:
            return {"domain": "arithmetic", "tool": "arithmetic-readout", "confidence": 0.98}
        if lowered.startswith(("remember ", "recall ")):
            return {"domain": "memory", "tool": "memory-atlas", "confidence": 0.95}
        if self._looks_like_json(value):
            return {"domain": "json", "tool": "json-parser", "confidence": 0.90}
        if self._looks_like_python(value):
            return {"domain": "code", "tool": "python-ast", "confidence": 0.90}
        if re.search(r"\b(http|https)://|\bsearch\b|\bdocs?\b", lowered):
            return {"domain": "search", "tool": "web-or-docs-search", "confidence": 0.75}
        if re.search(r"\b(run|shell|terminal|grep|rg|pytest|python3?)\b", lowered):
            return {"domain": "shell", "tool": "terminal", "confidence": 0.70}
        return {"domain": "tool", "tool": "planner", "confidence": 0.50}

    def probe(self) -> DomainProbeResult:
        examples = [
            ("8 plus 9", "arithmetic"),
            ("def add(a, b):\n    return a + b", "code"),
            ("remember project: PhaseMesh", "memory"),
            ('{"ok": true}', "json"),
        ]
        rows = []
        passed = 0
        for text, expected in examples:
            observed = self.route(text)
            ok = observed["domain"] == expected
            passed += int(ok)
            rows.append({"input": text, "expected": expected, "observed": observed, "passed": ok})
        accuracy = passed / len(examples)
        return DomainProbeResult(
            domain=self.name,
            passed=accuracy == 1.0,
            metrics={"accuracy": accuracy, "examples": len(examples)},
            examples=rows,
        )

    def solve(self, text: str) -> DomainSolveResult:
        route = self.route(text)
        return DomainSolveResult(
            domain=self.name,
            status="ok",
            answer=str(route["domain"]),
            confidence=float(route["confidence"]),
            data=route,
        )

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "kind": "deterministic_router"}

    def _looks_like_json(self, text: str) -> bool:
        try:
            json.loads(text)
            return True
        except Exception:
            return False

    def _looks_like_python(self, text: str) -> bool:
        code = extract_python_code(text)
        lowered = code.lower()
        return (
            "\n" in code
            and any(token in lowered for token in ("def ", "class ", "import ", "return ", "raise "))
        ) or lowered.startswith(("def ", "class ", "import "))
