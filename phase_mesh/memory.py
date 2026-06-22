from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .runtime_types import ResonanceLike


@dataclass(frozen=True)
class MemoryEntry:
    key: str
    value: str
    route: str
    signature: str
    dominant_sector: int
    sector_histogram: list[int]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "route": self.route,
            "signature": self.signature,
            "dominant_sector": self.dominant_sector,
            "sector_histogram": self.sector_histogram,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            key=str(data["key"]),
            value=str(data["value"]),
            route=str(data["route"]),
            signature=str(data["signature"]),
            dominant_sector=int(data["dominant_sector"]),
            sector_histogram=[int(value) for value in data["sector_histogram"]],
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass(frozen=True)
class RecallResult:
    found: bool
    key: str | None
    value: str | None
    score: float
    entry: MemoryEntry | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "key": self.key,
            "value": self.value,
            "score": self.score,
            "entry": None if self.entry is None else self.entry.to_dict(),
        }


class TopologicalMemory:
    """Associative memory over resonance basin traces."""

    def __init__(self, *, threshold: float = 0.92) -> None:
        self.threshold = threshold
        self.entries: list[MemoryEntry] = []

    def remember(self, key: str, value: str, run: ResonanceLike) -> MemoryEntry:
        decoded = run.decoded
        entry = MemoryEntry(
            key=key,
            value=value,
            route=decoded.route,
            signature=decoded.signature,
            dominant_sector=decoded.dominant_sector,
            sector_histogram=decoded.sector_histogram,
        )
        self.entries = [existing for existing in self.entries if existing.key != key]
        self.entries.append(entry)
        return entry

    def recall(self, run: ResonanceLike, *, key: str | None = None) -> RecallResult:
        if not self.entries:
            return RecallResult(found=False, key=None, value=None, score=0.0)

        query_histogram = np.asarray(run.decoded.sector_histogram, dtype=np.float64)
        best_entry: MemoryEntry | None = None
        best_score = -1.0
        for entry in self.entries:
            basin_score = histogram_similarity(
                query_histogram,
                np.asarray(entry.sector_histogram, dtype=np.float64),
            )
            key_score = key_similarity(key, entry.key) if key is not None else 0.0
            score = 0.68 * basin_score + 0.32 * key_score
            if run.decoded.dominant_sector == entry.dominant_sector:
                score += 0.035
            if run.decoded.route == entry.route:
                score += 0.015
            if key == entry.key:
                score += 0.08
            score = min(score, 1.0)
            if score > best_score:
                best_score = score
                best_entry = entry

        found = bool(best_entry is not None and best_score >= self.threshold)
        return RecallResult(
            found=found,
            key=None if best_entry is None else best_entry.key,
            value=None if not found or best_entry is None else best_entry.value,
            score=float(max(0.0, best_score)),
            entry=best_entry,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TopologicalMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        memory = cls(threshold=float(data.get("threshold", 0.92)))
        memory.entries = [MemoryEntry.from_dict(item) for item in data.get("entries", [])]
        return memory


def histogram_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def key_similarity(left: str | None, right: str) -> float:
    if left is None:
        return 0.0
    if left == right:
        return 1.0
    left_parts = set(left.lower().replace("-", "_").split("_"))
    right_parts = set(right.lower().replace("-", "_").split("_"))
    if not left_parts or not right_parts:
        return 0.0
    overlap = len(left_parts & right_parts)
    union = len(left_parts | right_parts)
    return overlap / union
