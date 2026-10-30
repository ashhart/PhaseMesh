from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from .config import MeshConfig
from .consolidation import BasinSnapshot, BasinTracker
from .encoding import DecodedResonance, PhaseDecoder, TextPhaseEncoder
from .field import PhaseFieldMesh, ResonanceMetrics
from .memory import RecallResult, TopologicalMemory
from .verifier import VerifierResult, VerifierRouter


@dataclass(frozen=True)
class ResonanceRun:
    prompt: str
    decoded: DecodedResonance
    verifier: VerifierResult
    metrics: ResonanceMetrics
    history: list[ResonanceMetrics]
    basin: BasinSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "decoded": self.decoded.to_dict(),
            "verifier": self.verifier.to_dict(),
            "metrics": self.metrics.to_dict(),
            "basin": None if self.basin is None else self.basin.to_dict(),
            "history_tail": [item.to_dict() for item in self.history[-8:]],
        }


@dataclass(frozen=True)
class AdaptiveStep:
    step: int
    steps_used: int
    prediction_error: float
    damping: float
    temperature_scale: float
    metrics: ResonanceMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "steps_used": self.steps_used,
            "prediction_error": self.prediction_error,
            "damping": self.damping,
            "temperature_scale": self.temperature_scale,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class AdaptiveRun:
    prompt: str
    decoded: DecodedResonance
    verifier: VerifierResult
    metrics: ResonanceMetrics
    history: list[AdaptiveStep]
    max_budget: int
    steps_used: int
    exhausted: bool
    mean_prediction_error: float
    final_prediction_error: float
    verifier_checks: int = 0
    verifier_failures: int = 0
    basin: BasinSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "decoded": self.decoded.to_dict(),
            "verifier": self.verifier.to_dict(),
            "metrics": self.metrics.to_dict(),
            "max_budget": self.max_budget,
            "steps_used": self.steps_used,
            "budget_remaining": max(0, self.max_budget - self.steps_used),
            "exhausted": self.exhausted,
            "mean_prediction_error": self.mean_prediction_error,
            "final_prediction_error": self.final_prediction_error,
            "verifier_checks": self.verifier_checks,
            "verifier_failures": self.verifier_failures,
            "basin": None if self.basin is None else self.basin.to_dict(),
            "history_tail": [item.to_dict() for item in self.history[-8:]],
        }


class CognitiveMeshRuntime:
    """High-level runtime that wires encoding, field dynamics, decoding, and feedback."""

    def __init__(self, config: MeshConfig | None = None) -> None:
        self.config = config or MeshConfig()
        self.mesh = PhaseFieldMesh(self.config)
        self.encoder = TextPhaseEncoder(self.config.width, self.config.height)
        self.decoder = PhaseDecoder()
        self.verifier = VerifierRouter()
        self.memory = TopologicalMemory()
        self.basin_tracker = BasinTracker()

    def resonate(
        self,
        prompt: str,
        *,
        steps: int | None = None,
        expected: str | None = None,
        learn: bool = False,
    ) -> ResonanceRun:
        self.mesh.inject_text(prompt, self.encoder)
        history = self.mesh.run_until_resonance(max_steps=steps)
        metrics = history[-1] if history else self.mesh.metrics()
        decoded = self.decoder.decode(self.mesh.theta, coherence=metrics.coherence)
        verifier = self.verifier.verify(
            prompt,
            candidate=decoded.route,
            expected=expected,
            coherence=metrics.coherence,
        )
        basin = self.basin_tracker.discover(
            self.mesh,
            decoded,
            coherence=metrics.coherence,
        )
        if learn:
            self.mesh.apply_feedback(
                success=verifier.passed,
                message=verifier.message,
                encoder=self.encoder,
            )
            self.mesh.consolidate(cycles=8)
        return ResonanceRun(
            prompt=prompt,
            decoded=decoded,
            verifier=verifier,
            metrics=metrics,
            history=history,
            basin=basin,
        )

    def think(
        self,
        prompt: str,
        *,
        max_budget: int = 200,
        min_steps: int | None = None,
        temperature: float = 0.0,
        expected: str | None = None,
        learn: bool = False,
        verifier_control: bool = False,
    ) -> AdaptiveRun:
        self.mesh.inject_text(prompt, self.encoder)
        min_steps = self.config.min_steps if min_steps is None else min_steps
        current_damping = self.config.damping
        history: list[AdaptiveStep] = []
        prediction_errors: list[float] = []
        verifier_checks = 0
        verifier_failures = 0
        metrics = self.mesh.metrics()

        for steps_used in range(1, max(1, max_budget) + 1):
            predicted = self.mesh.predict_phase(damping=current_damping)
            metrics = self.mesh.step(damping=current_damping)
            prediction_error = self.mesh.observe_prediction(predicted)
            prediction_errors.append(prediction_error)
            current_damping = self.mesh.adaptive_damping(prediction_error)

            temperature_scale = 0.0
            if temperature > 0 and prediction_error > self.config.prediction_commit_error:
                temperature_scale = min(0.08, temperature * prediction_error * 0.18)
                self.mesh.inject_noise(scale=temperature_scale)

            history.append(
                AdaptiveStep(
                    step=metrics.step,
                    steps_used=steps_used,
                    prediction_error=prediction_error,
                    damping=current_damping,
                    temperature_scale=temperature_scale,
                    metrics=metrics,
                )
            )

            can_commit = (
                steps_used >= min_steps
                and metrics.resonant
                and prediction_error <= self.config.prediction_commit_error
            )
            if can_commit and verifier_control:
                decoded_probe = self.decoder.decode(self.mesh.theta, coherence=metrics.coherence)
                verifier_probe = self.verifier.verify(
                    prompt,
                    candidate=decoded_probe.route,
                    expected=expected,
                    coherence=metrics.coherence,
                )
                verifier_checks += 1
                if verifier_probe.passed:
                    break
                verifier_failures += 1
                if verifier_failures == 1:
                    self.mesh.apply_feedback(
                        success=False,
                        message=verifier_probe.message,
                        encoder=self.encoder,
                        learning_rate=0.025,
                    )
                if temperature > 0:
                    self.mesh.inject_noise(scale=min(0.08, max(0.01, temperature * 0.05)))
                continue
            if can_commit:
                break

        final_error = prediction_errors[-1] if prediction_errors else 0.0
        mean_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0
        steps_used = len(history)
        decoded = self.decoder.decode(self.mesh.theta, coherence=metrics.coherence)
        verifier = self.verifier.verify(
            prompt,
            candidate=decoded.route,
            expected=expected,
            coherence=metrics.coherence,
        )
        basin = self.basin_tracker.discover(
            self.mesh,
            decoded,
            coherence=metrics.coherence,
            prediction_error=final_error,
        )
        if learn:
            self.mesh.apply_feedback(
                success=verifier.passed,
                message=verifier.message,
                encoder=self.encoder,
            )
            self.mesh.consolidate(cycles=8)

        return AdaptiveRun(
            prompt=prompt,
            decoded=decoded,
            verifier=verifier,
            metrics=metrics,
            history=history,
            max_budget=max_budget,
            steps_used=steps_used,
            exhausted=steps_used >= max_budget
            and (
                not (metrics.resonant and final_error <= self.config.prediction_commit_error)
                or (verifier_control and not verifier.passed)
            ),
            mean_prediction_error=mean_error,
            final_prediction_error=final_error,
            verifier_checks=verifier_checks,
            verifier_failures=verifier_failures,
            basin=basin,
        )

    def think_stream(
        self,
        prompt: str,
        *,
        max_budget: int = 200,
        min_steps: int | None = None,
        temperature: float = 0.0,
        expected: str | None = None,
        learn: bool = False,
        verifier_control: bool = False,
        stream_interval: int = 4,
    ) -> Iterator[dict[str, Any]]:
        self.mesh.inject_text(prompt, self.encoder)
        min_steps = self.config.min_steps if min_steps is None else min_steps
        current_damping = self.config.damping
        history: list[AdaptiveStep] = []
        prediction_errors: list[float] = []
        verifier_checks = 0
        verifier_failures = 0
        metrics = self.mesh.metrics()

        yield {
            "event": "start",
            "prompt": prompt,
            "max_budget": max_budget,
            "min_steps": min_steps,
            "temperature": temperature,
            "metrics": metrics.to_dict(),
        }

        for steps_used in range(1, max(1, max_budget) + 1):
            predicted = self.mesh.predict_phase(damping=current_damping)
            metrics = self.mesh.step(damping=current_damping)
            prediction_error = self.mesh.observe_prediction(predicted)
            prediction_errors.append(prediction_error)
            current_damping = self.mesh.adaptive_damping(prediction_error)

            temperature_scale = 0.0
            if temperature > 0 and prediction_error > self.config.prediction_commit_error:
                temperature_scale = min(0.08, temperature * prediction_error * 0.18)
                self.mesh.inject_noise(scale=temperature_scale)

            step = AdaptiveStep(
                step=metrics.step,
                steps_used=steps_used,
                prediction_error=prediction_error,
                damping=current_damping,
                temperature_scale=temperature_scale,
                metrics=metrics,
            )
            history.append(step)

            can_commit = (
                steps_used >= min_steps
                and metrics.resonant
                and prediction_error <= self.config.prediction_commit_error
            )
            should_emit = (
                steps_used == 1
                or steps_used % max(1, stream_interval) == 0
                or can_commit
            )
            if should_emit:
                decoded_probe = self.decoder.decode(self.mesh.theta, coherence=metrics.coherence)
                yield {
                    "event": "step",
                    "step": step.to_dict(),
                    "decoded": decoded_probe.to_dict(),
                    "phase_sample": phase_sample(self.mesh.theta),
                }

            if can_commit and verifier_control:
                decoded_probe = self.decoder.decode(self.mesh.theta, coherence=metrics.coherence)
                verifier_probe = self.verifier.verify(
                    prompt,
                    candidate=decoded_probe.route,
                    expected=expected,
                    coherence=metrics.coherence,
                )
                verifier_checks += 1
                yield {
                    "event": "verifier",
                    "steps_used": steps_used,
                    "decoded": decoded_probe.to_dict(),
                    "verifier": verifier_probe.to_dict(),
                }
                if verifier_probe.passed:
                    break
                verifier_failures += 1
                if verifier_failures == 1:
                    self.mesh.apply_feedback(
                        success=False,
                        message=verifier_probe.message,
                        encoder=self.encoder,
                        learning_rate=0.025,
                    )
                if temperature > 0:
                    self.mesh.inject_noise(scale=min(0.08, max(0.01, temperature * 0.05)))
                continue
            if can_commit:
                break

        final_error = prediction_errors[-1] if prediction_errors else 0.0
        mean_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0
        steps_used = len(history)
        decoded = self.decoder.decode(self.mesh.theta, coherence=metrics.coherence)
        verifier = self.verifier.verify(
            prompt,
            candidate=decoded.route,
            expected=expected,
            coherence=metrics.coherence,
        )
        basin = self.basin_tracker.discover(
            self.mesh,
            decoded,
            coherence=metrics.coherence,
            prediction_error=final_error,
        )
        if learn:
            self.mesh.apply_feedback(
                success=verifier.passed,
                message=verifier.message,
                encoder=self.encoder,
            )
            self.mesh.consolidate(cycles=8)

        run = AdaptiveRun(
            prompt=prompt,
            decoded=decoded,
            verifier=verifier,
            metrics=metrics,
            history=history,
            max_budget=max_budget,
            steps_used=steps_used,
            exhausted=steps_used >= max_budget
            and (
                not (metrics.resonant and final_error <= self.config.prediction_commit_error)
                or (verifier_control and not verifier.passed)
            ),
            mean_prediction_error=mean_error,
            final_prediction_error=final_error,
            verifier_checks=verifier_checks,
            verifier_failures=verifier_failures,
            basin=basin,
        )
        yield {
            "event": "basin",
            "basin": basin.to_dict(),
            "decoded": decoded.to_dict(),
            "metrics": metrics.to_dict(),
        }
        yield {
            "event": "final",
            "run": run.to_dict(),
            "phase_sample": phase_sample(self.mesh.theta),
        }

    def discover_basins(self) -> dict[str, Any]:
        removed = self.basin_tracker.prune_transients()
        return {
            "removed": removed,
            "basins": [basin.to_dict() for basin in self.basin_tracker.basins],
        }

    def learn(
        self,
        prompt: str,
        *,
        expected: str | None = None,
        rounds: int = 4,
        steps: int | None = None,
    ) -> dict[str, Any]:
        traces: list[dict[str, Any]] = []
        for round_index in range(max(1, rounds)):
            run = self.resonate(prompt, steps=steps, expected=expected, learn=True)
            traces.append({"round": round_index + 1, **run.to_dict()})
        return {
            "prompt": prompt,
            "expected": expected,
            "rounds": len(traces),
            "final": traces[-1],
            "traces": traces,
        }

    def remember(
        self,
        key: str,
        value: str,
        *,
        steps: int | None = None,
    ) -> dict[str, Any]:
        run = self.resonate(key, steps=steps, learn=True)
        entry = self.memory.remember(key, value, run)
        self.mesh.consolidate(cycles=4)
        return {
            "key": key,
            "value": value,
            "entry": entry.to_dict(),
            "run": run.to_dict(),
        }

    def recall(self, key: str, *, steps: int | None = None) -> dict[str, Any]:
        run = self.resonate(key, steps=steps, learn=False)
        result: RecallResult = self.memory.recall(run, key=key)
        return {
            "key": key,
            "recall": result.to_dict(),
            "run": run.to_dict(),
        }



def phase_sample(theta: Any, *, sample_size: int = 32) -> dict[str, Any]:
    import numpy as np

    field = np.asarray(theta)
    height, width = field.shape
    y_idx = np.linspace(0, height - 1, min(sample_size, height)).astype(int)
    x_idx = np.linspace(0, width - 1, min(sample_size, width)).astype(int)
    sample = field[np.ix_(y_idx, x_idx)]
    normalized = ((sample + np.pi) / (2.0 * np.pi) * 255.0).clip(0, 255).astype(np.uint8)
    return {
        "width": int(normalized.shape[1]),
        "height": int(normalized.shape[0]),
        "values": normalized.ravel().tolist(),
    }
