from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..probes import (
    ArithmeticFactorReadout,
    fit_save_arithmetic_factor_readout,
    run_arithmetic_result_readout_probe,
    solve_arithmetic_with_factor_readout,
)
from .base import DomainFitResult, DomainProbeResult, DomainSolveResult


class ArithmeticDomain:
    """Structured arithmetic domain backed by the proven factor readout gate."""

    name = "arithmetic"

    def __init__(
        self,
        *,
        readout: ArithmeticFactorReadout | None = None,
        max_value: int = 20,
        min_value: int = 0,
        ops: Sequence[str] = ("add", "mul"),
        grid_size: int = 64,
        basin_dim: int = 128,
        hidden: int = 64,
        steps_per_chunk: int = 4,
        seed: int = 7,
        backend: str = "numpy",
        train_fraction: float = 0.7,
        structured_feature_strength: float = 2.0,
    ) -> None:
        self.readout = readout
        self.max_value = int(max_value)
        self.min_value = int(min_value)
        self.ops = tuple(str(item) for item in ops)
        self.grid_size = int(grid_size)
        self.basin_dim = int(basin_dim)
        self.hidden = int(hidden)
        self.steps_per_chunk = int(steps_per_chunk)
        self.seed = int(seed)
        self.backend = str(backend)
        self.train_fraction = float(train_fraction)
        self.structured_feature_strength = float(structured_feature_strength)

    @classmethod
    def load(cls, artifact_dir: str | Path) -> ArithmeticDomain:
        path = Path(artifact_dir)
        readout = ArithmeticFactorReadout.load(path)
        config = readout.config
        return cls(
            readout=readout,
            max_value=int(config.get("max_value", 20)),
            min_value=int(config.get("min_value", 0)),
            ops=tuple(config.get("ops", ("add", "mul"))),
            grid_size=int(config.get("grid_size", 64)),
            basin_dim=int(config.get("basin_dim", 128)),
            hidden=int(config.get("hidden", 64)),
            steps_per_chunk=int(config.get("steps_per_chunk", 4)),
            seed=int(config.get("seed", 7)),
            backend=str(config.get("backend", "numpy")),
            structured_feature_strength=float(config.get("structured_feature_strength", 2.0)),
        )

    def fit(self, out_dir: str | Path) -> DomainFitResult:
        path = Path(out_dir)
        summary = fit_save_arithmetic_factor_readout(
            path,
            max_value=self.max_value,
            min_value=self.min_value,
            ops=self.ops,
            grid_size=self.grid_size,
            basin_dim=self.basin_dim,
            hidden=self.hidden,
            steps_per_chunk=self.steps_per_chunk,
            seed=self.seed,
            backend=self.backend,
            train_fraction=self.train_fraction,
            structured_feature_strength=self.structured_feature_strength,
        )
        self.readout = ArithmeticFactorReadout.load(path)
        evaluation = summary.get("evaluation", {})
        return DomainFitResult(
            domain=self.name,
            status="ok",
            artifact_dir=str(path),
            metrics={
                "rows": summary.get("fit", {}).get("rows", 0),
                "passed_result_gate": bool(evaluation.get("passed_result_gate", False)),
                "factorized_result_accuracy": float(
                    evaluation.get("factorized_result", {}).get("accuracy", 0.0)
                ),
                "direct_result_accuracy": float(
                    evaluation.get("direct_result_probe", {}).get("accuracy", 0.0)
                ),
            },
            notes=["Structured arithmetic readout is factorized; direct result probe is reported as a control."],
        )

    def probe(self) -> DomainProbeResult:
        payload = run_arithmetic_result_readout_probe(
            encoder_mode="structured",
            max_value=self.max_value,
            min_value=self.min_value,
            ops=self.ops,
            grid_size=self.grid_size,
            basin_dim=self.basin_dim,
            hidden=self.hidden,
            steps_per_chunk=self.steps_per_chunk,
            seed=self.seed,
            backend=self.backend,
            train_fraction=self.train_fraction,
            structured_feature_strength=self.structured_feature_strength,
        )
        return DomainProbeResult(
            domain=self.name,
            passed=bool(payload.get("passed_result_gate", False)),
            metrics={
                "factorized_result_accuracy": float(payload.get("factorized_result", {}).get("accuracy", 0.0)),
                "direct_result_accuracy": float(payload.get("direct_result_probe", {}).get("accuracy", 0.0)),
                "operation_accuracy": float(payload.get("factor_probes", {}).get("operation", {}).get("accuracy", 0.0)),
                "left_accuracy": float(payload.get("factor_probes", {}).get("left", {}).get("accuracy", 0.0)),
                "right_accuracy": float(payload.get("factor_probes", {}).get("right", {}).get("accuracy", 0.0)),
            },
            examples=payload.get("factorized_result", {}).get("examples", []),
            notes=[str(payload.get("note", ""))],
        )

    def solve(self, text: str) -> DomainSolveResult:
        if self.readout is not None:
            payload = self.readout.solve(text)
        else:
            payload = solve_arithmetic_with_factor_readout(
                text,
                max_value=self.max_value,
                min_value=self.min_value,
                ops=self.ops,
                grid_size=self.grid_size,
                basin_dim=self.basin_dim,
                hidden=self.hidden,
                steps_per_chunk=self.steps_per_chunk,
                seed=self.seed,
                backend=self.backend,
                structured_feature_strength=self.structured_feature_strength,
            )
        return DomainSolveResult(
            domain=self.name,
            status=str(payload.get("status", "error")),
            answer=str(payload.get("answer", "")),
            confidence=1.0 if payload.get("status") == "ok" else 0.0,
            data=payload,
            notes=["Solved by decoding operation/operands from the structured basin, then applying an exact resolver."],
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "structured_factor_readout",
            "ops": list(self.ops),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "grid_size": self.grid_size,
            "basin_dim": self.basin_dim,
            "steps_per_chunk": self.steps_per_chunk,
            "has_readout": self.readout is not None,
        }
