from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeshConfig:
    """Configuration for a local 2D phase-field mesh."""

    width: int = 128
    height: int = 128
    dt: float = 0.055
    wave_speed: float = 0.62
    damping: float = 0.09
    nonlinear_gain: float = 0.07
    potential_scale: float = 0.018
    natural_frequency_noise: float = 0.002
    memory_decay: float = 0.0008
    memory_gain: float = 0.018
    laplacian_backend: str = "auto"
    phase_pin_strength: float = 0.0
    phase_pin_decay: float = 0.002
    phase_residual_carry: float = 0.08
    predictor_learning_rate: float = 0.065
    prediction_trace_decay: float = 0.004
    predictor_wave_speed_scale: float = 0.86
    predictor_nonlinear_scale: float = 0.75
    prediction_commit_error: float = 0.035
    adaptive_damping_alpha: float = 5.0
    adaptive_min_damping: float = 0.025
    adaptive_max_damping: float = 0.18
    resonance_coherence: float = 0.88
    resonance_gradient: float = 0.18
    resonance_delta: float = 0.0018
    min_steps: int = 40
    max_steps: int = 320
    seed: int = 7

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def validate(self) -> None:
        if self.width < 16 or self.height < 16:
            raise ValueError("Mesh width and height must be at least 16.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.wave_speed <= 0:
            raise ValueError("wave_speed must be positive.")
        if self.damping < 0:
            raise ValueError("damping cannot be negative.")
        if self.min_steps < 0 or self.max_steps <= 0:
            raise ValueError("step counts must be positive.")
        if self.laplacian_backend not in {"auto", "numpy", "scipy", "jax"}:
            raise ValueError("laplacian_backend must be one of: auto, numpy, scipy, jax.")
        if not 0.0 <= self.phase_pin_strength <= 1.0:
            raise ValueError("phase_pin_strength must be in [0, 1].")
        if not 0.0 <= self.phase_pin_decay <= 1.0:
            raise ValueError("phase_pin_decay must be in [0, 1].")
        if not 0.0 <= self.phase_residual_carry <= 1.0:
            raise ValueError("phase_residual_carry must be in [0, 1].")
        if self.predictor_learning_rate < 0:
            raise ValueError("predictor_learning_rate cannot be negative.")
        if self.prediction_commit_error <= 0:
            raise ValueError("prediction_commit_error must be positive.")
        if self.predictor_wave_speed_scale <= 0:
            raise ValueError("predictor_wave_speed_scale must be positive.")
        if self.predictor_nonlinear_scale <= 0:
            raise ValueError("predictor_nonlinear_scale must be positive.")
        if self.adaptive_min_damping <= 0 or self.adaptive_max_damping <= 0:
            raise ValueError("adaptive damping bounds must be positive.")
        if self.adaptive_min_damping > self.adaptive_max_damping:
            raise ValueError("adaptive_min_damping cannot exceed adaptive_max_damping.")
