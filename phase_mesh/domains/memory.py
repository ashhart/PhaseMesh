from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import DomainFitResult, DomainProbeResult, DomainSolveResult


FACT_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.+?)\s*$")


class MemoryDomain:
    """Small persistent fact atlas for PhaseMesh domains."""

    name = "memory"

    def __init__(self, facts: dict[str, list[str]] | None = None, artifact_dir: str | Path | None = None) -> None:
        self.facts = facts or {}
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    @classmethod
    def load(cls, artifact_dir: str | Path) -> MemoryDomain:
        path = Path(artifact_dir)
        atlas = path / "atlas.json"
        if not atlas.exists():
            return cls(artifact_dir=path)
        payload = json.loads(atlas.read_text(encoding="utf-8"))
        facts = {
            str(key): [str(item) for item in value]
            for key, value in dict(payload.get("facts", {})).items()
        }
        return cls(facts=facts, artifact_dir=path)

    def save(self, artifact_dir: str | Path | None = None) -> Path:
        path = Path(artifact_dir) if artifact_dir is not None else self.artifact_dir
        if path is None:
            raise ValueError("artifact_dir is required to save memory")
        path.mkdir(parents=True, exist_ok=True)
        atlas = path / "atlas.json"
        atlas.write_text(json.dumps({"facts": self.facts}, indent=2) + "\n", encoding="utf-8")
        self.artifact_dir = path
        return atlas

    def remember(self, subject: str, fact: str) -> None:
        key = self._normalize_subject(subject)
        value = str(fact).strip()
        if not value:
            return
        bucket = self.facts.setdefault(key, [])
        if value not in bucket:
            bucket.append(value)

    def recall(self, query: str) -> list[str]:
        key = self._normalize_subject(query)
        if key in self.facts:
            return list(self.facts[key])
        matches: list[str] = []
        for subject, facts in self.facts.items():
            if key and (key in subject or subject in key):
                matches.extend(facts)
        return matches

    def fit(self, out_dir: str | Path) -> DomainFitResult:
        path = Path(out_dir)
        if not self.facts:
            self.remember("phasemesh", "structured arithmetic is the first verified domain")
            self.remember("registry", "domains expose fit probe solve")
        self.save(path)
        return DomainFitResult(
            domain=self.name,
            status="ok",
            artifact_dir=str(path),
            metrics={"subjects": len(self.facts), "facts": sum(len(v) for v in self.facts.values())},
            notes=["Memory atlas stores explicit facts; no hidden vector database is used."],
        )

    def probe(self) -> DomainProbeResult:
        domain = MemoryDomain()
        domain.remember("project", "PhaseMesh")
        domain.remember("project", "domain registry")
        facts = domain.recall("project")
        passed = "PhaseMesh" in facts and "domain registry" in facts
        return DomainProbeResult(
            domain=self.name,
            passed=passed,
            metrics={"exact_recall": 1.0 if passed else 0.0},
            examples=[{"query": "project", "facts": facts, "passed": passed}],
            notes=["This gate proves exact persistent recall only."],
        )

    def solve(self, text: str) -> DomainSolveResult:
        value = str(text).strip()
        lowered = value.lower()
        if lowered.startswith("remember "):
            value = value[len("remember "):].strip()
            match = FACT_RE.match(value)
            if match:
                self.remember(match.group(1), match.group(2))
                if self.artifact_dir is not None:
                    self.save()
                return DomainSolveResult(
                    domain=self.name,
                    status="ok",
                    answer="remembered",
                    confidence=1.0,
                    data={"subject": self._normalize_subject(match.group(1)), "fact": match.group(2).strip()},
                )
        query = value
        if lowered.startswith("recall "):
            query = value[len("recall "):].strip()
        facts = self.recall(query)
        return DomainSolveResult(
            domain=self.name,
            status="ok" if facts else "not_found",
            answer="; ".join(facts),
            confidence=1.0 if facts else 0.0,
            data={"query": query, "facts": facts},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "explicit_fact_atlas",
            "subjects": len(self.facts),
            "facts": sum(len(v) for v in self.facts.values()),
        }

    def _normalize_subject(self, subject: str) -> str:
        return re.sub(r"\s+", " ", str(subject).strip().lower())
