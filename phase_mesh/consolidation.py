from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .encoding import DecodedResonance
from .field import PhaseFieldMesh, gradient_components, smooth
from .memory import histogram_similarity


@dataclass(frozen=True)
class BasinSnapshot:
    basin_id: str
    route: str
    dominant_sector: int
    representative_histogram: list[float]
    count: int
    persistence: float
    mean_coherence: float
    mean_prediction_error: float
    first_seen: float
    last_seen: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "basin_id": self.basin_id,
            "route": self.route,
            "dominant_sector": self.dominant_sector,
            "representative_histogram": self.representative_histogram,
            "count": self.count,
            "persistence": self.persistence,
            "mean_coherence": self.mean_coherence,
            "mean_prediction_error": self.mean_prediction_error,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BasinSnapshot":
        return cls(
            basin_id=str(data["basin_id"]),
            route=str(data["route"]),
            dominant_sector=int(data["dominant_sector"]),
            representative_histogram=[float(item) for item in data["representative_histogram"]],
            count=int(data["count"]),
            persistence=float(data["persistence"]),
            mean_coherence=float(data["mean_coherence"]),
            mean_prediction_error=float(data["mean_prediction_error"]),
            first_seen=float(data["first_seen"]),
            last_seen=float(data["last_seen"]),
        )


class BasinTracker:
    """Tracks persistent attractor basins without labels or rewards."""

    def __init__(
        self,
        *,
        merge_threshold: float = 0.985,
        stable_count: int = 3,
        reinforcement_gain: float = 0.006,
        prune_rate: float = 0.0015,
    ) -> None:
        self.merge_threshold = merge_threshold
        self.stable_count = stable_count
        self.reinforcement_gain = reinforcement_gain
        self.prune_rate = prune_rate
        self.basins: list[BasinSnapshot] = []

    def discover(
        self,
        mesh: PhaseFieldMesh,
        decoded: DecodedResonance,
        *,
        coherence: float,
        prediction_error: float | None = None,
    ) -> BasinSnapshot:
        basin = self.observe(
            decoded,
            coherence=coherence,
            prediction_error=prediction_error,
        )
        self.apply_to_mesh(mesh, basin)
        return basin

    def observe(
        self,
        decoded: DecodedResonance,
        *,
        coherence: float,
        prediction_error: float | None = None,
    ) -> BasinSnapshot:
        prediction_error = 0.0 if prediction_error is None else prediction_error
        now = time.time()
        query_hist = np.asarray(decoded.sector_histogram, dtype=np.float64)
        best_index = -1
        best_score = -1.0

        for index, basin in enumerate(self.basins):
            if basin.dominant_sector != decoded.dominant_sector:
                continue
            score = histogram_similarity(
                query_hist,
                np.asarray(basin.representative_histogram, dtype=np.float64),
            )
            if decoded.route == basin.route:
                score += 0.01
            if score > best_score:
                best_index = index
                best_score = score

        if best_index == -1 or best_score < self.merge_threshold:
            basin = BasinSnapshot(
                basin_id=f"b{len(self.basins) + 1:04d}",
                route=decoded.route,
                dominant_sector=decoded.dominant_sector,
                representative_histogram=[float(item) for item in decoded.sector_histogram],
                count=1,
                persistence=self._persistence(1, coherence, prediction_error),
                mean_coherence=coherence,
                mean_prediction_error=prediction_error,
                first_seen=now,
                last_seen=now,
            )
            self.basins.append(basin)
            return basin

        old = self.basins[best_index]
        count = old.count + 1
        old_hist = np.asarray(old.representative_histogram, dtype=np.float64)
        merged_hist = (old_hist * old.count + query_hist) / count
        mean_coherence = (old.mean_coherence * old.count + coherence) / count
        mean_error = (old.mean_prediction_error * old.count + prediction_error) / count
        basin = BasinSnapshot(
            basin_id=old.basin_id,
            route=old.route,
            dominant_sector=old.dominant_sector,
            representative_histogram=[float(item) for item in merged_hist.tolist()],
            count=count,
            persistence=self._persistence(count, mean_coherence, mean_error),
            mean_coherence=mean_coherence,
            mean_prediction_error=mean_error,
            first_seen=old.first_seen,
            last_seen=now,
        )
        self.basins[best_index] = basin
        return basin

    def apply_to_mesh(self, mesh: PhaseFieldMesh, basin: BasinSnapshot) -> None:
        dx, dy = gradient_components(mesh.theta)
        basin_stability = np.exp(-(dx * dx + dy * dy))
        if basin.count >= self.stable_count:
            mesh.landscape = mesh.landscape + self.reinforcement_gain * basin.persistence * basin_stability * np.cos(mesh.theta)
        else:
            mesh.landscape = mesh.landscape * (1.0 - self.prune_rate)
        mesh.landscape = smooth(mesh.landscape, amount=0.05, backend=mesh.config.laplacian_backend)
        mesh.landscape = np.clip(mesh.landscape, -4.0, 4.0)

    def prune_transients(self, *, max_age_s: float = 86_400.0, min_persistence: float = 0.25) -> int:
        now = time.time()
        before = len(self.basins)
        self.basins = [
            basin
            for basin in self.basins
            if basin.persistence >= min_persistence or now - basin.last_seen <= max_age_s
        ]
        return before - len(self.basins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "merge_threshold": self.merge_threshold,
            "stable_count": self.stable_count,
            "reinforcement_gain": self.reinforcement_gain,
            "prune_rate": self.prune_rate,
            "basins": [basin.to_dict() for basin in self.basins],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BasinTracker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tracker = cls(
            merge_threshold=float(data.get("merge_threshold", 0.985)),
            stable_count=int(data.get("stable_count", 3)),
            reinforcement_gain=float(data.get("reinforcement_gain", 0.006)),
            prune_rate=float(data.get("prune_rate", 0.0015)),
        )
        tracker.basins = [BasinSnapshot.from_dict(item) for item in data.get("basins", [])]
        return tracker

    def _persistence(self, count: int, coherence: float, prediction_error: float) -> float:
        count_score = min(1.0, count / max(1, self.stable_count))
        error_score = max(0.0, 1.0 - prediction_error * 8.0)
        return float(np.clip(0.55 * count_score + 0.35 * coherence + 0.10 * error_score, 0.0, 1.0))

