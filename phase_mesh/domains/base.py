from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class DomainFitResult:
    domain: str
    status: str
    artifact_dir: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "status": self.status,
            "artifact_dir": self.artifact_dir,
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass
class DomainProbeResult:
    domain: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "passed": bool(self.passed),
            "metrics": self.metrics,
            "examples": self.examples,
            "notes": self.notes,
        }


@dataclass
class DomainSolveResult:
    domain: str
    status: str
    answer: str = ""
    confidence: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "status": self.status,
            "answer": self.answer,
            "confidence": float(self.confidence),
            "data": self.data,
            "notes": self.notes,
        }


class DomainAdapter(Protocol):
    """A verified skill surface for the PhaseMesh registry."""

    name: str

    def fit(self, out_dir: str | Path) -> DomainFitResult:
        ...

    def probe(self) -> DomainProbeResult:
        ...

    def solve(self, text: str) -> DomainSolveResult:
        ...

    def manifest(self) -> dict[str, Any]:
        ...
