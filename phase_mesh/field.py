from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import MeshConfig
from .encoding import TextPhaseEncoder, WavePacket

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - optional acceleration path
    ndimage = None


_JAX_LAPLACIAN = None
_JAX_NEIGHBOR_AVERAGE = None
_JAX_STEP = None
_JAX_SCAN = None


LAPLACIAN_KERNEL = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
NEIGHBOR_AVERAGE_KERNEL = np.array(
    [
        [0.0, 0.25, 0.0],
        [0.25, 0.0, 0.25],
        [0.0, 0.25, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ResonanceMetrics:
    step: int
    coherence: float
    gradient: float
    energy: float
    coherence_delta: float
    resonant: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "step": self.step,
            "coherence": self.coherence,
            "gradient": self.gradient,
            "energy": self.energy,
            "coherence_delta": self.coherence_delta,
            "resonant": self.resonant,
        }


@dataclass(frozen=True)
class BasinFeature:
    """Stable attractor readout used by the experimental model layer."""

    x: int
    y: int
    center: np.ndarray
    dominant_phase: float
    coherence: float
    gradient: float
    energy: float

    def to_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "x": self.x,
            "y": self.y,
            "center": [float(item) for item in self.center.tolist()],
            "dominant_phase": self.dominant_phase,
            "coherence": self.coherence,
            "gradient": self.gradient,
            "energy": self.energy,
        }


class PhaseFieldMesh:
    """Damped 2D phase field with a persistent potential landscape."""

    def __init__(self, config: MeshConfig | None = None) -> None:
        self.config = config or MeshConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.theta = np.zeros(self.config.shape, dtype=np.float64)
        self.velocity = np.zeros(self.config.shape, dtype=np.float64)
        self.omega = self.rng.normal(
            0.0,
            self.config.natural_frequency_noise,
            size=self.config.shape,
        )
        self.landscape = np.zeros(self.config.shape, dtype=np.float64)
        self.predictor_trace = np.zeros(self.config.shape, dtype=np.float64)
        self.pin_phase = np.zeros(self.config.shape, dtype=np.float64)
        self.pin_weights = np.zeros(self.config.shape, dtype=np.float64)
        self.step_index = 0
        self._last_coherence: float | None = None
        self._residual_theta = self.theta.copy()

    def reset_field(self) -> None:
        self.theta.fill(0.0)
        self.velocity.fill(0.0)
        self.pin_phase.fill(0.0)
        self.pin_weights.fill(0.0)
        self.step_index = 0
        self._last_coherence = None
        self._residual_theta = self.theta.copy()

    def inject_text(self, text: str, encoder: TextPhaseEncoder | None = None) -> list[WavePacket]:
        encoder = encoder or TextPhaseEncoder(self.config.width, self.config.height)
        packets = encoder.encode(text)
        for packet in packets:
            self.inject_packet(packet)
        return packets

    def inject_packet(self, packet: WavePacket, *, into_landscape: bool = False) -> None:
        target = self.landscape if into_landscape else self.theta
        velocity_target = None if into_landscape else self.velocity
        yy, xx = self._stamp_packet(packet, target, velocity_target)
        if not into_landscape:
            self.theta = wrap_phase(self.theta)
            self._pin_region(yy, xx, packet)

    def predict_phase(
        self,
        *,
        damping: float | None = None,
        external_force: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forecast the next phase with a learned residual world model.

        The predictor intentionally uses a smaller dynamics model than `step`:
        it sees the fixed natural landscape plus a learned residual trace, but
        not the full mutable potential landscape. The residual is updated from
        observed circular phase error after real steps.
        """

        damping = self.config.damping if damping is None else damping
        lap = laplacian(self.theta, backend=self.config.laplacian_backend)
        estimated_landscape_force = self.config.potential_scale * np.sin(self.omega) + self.predictor_trace
        force = (
            (self.config.wave_speed * self.config.predictor_wave_speed_scale) ** 2 * lap
            + estimated_landscape_force
            - (self.config.nonlinear_gain * self.config.predictor_nonlinear_scale) * np.sin(self.theta)
            - damping * self.velocity
        )
        if external_force is not None:
            force = force + external_force

        velocity_next = self.velocity + self.config.dt * force
        velocity_next = velocity_next * max(0.0, 1.0 - damping * self.config.dt)
        predicted = wrap_phase(self.theta + self.config.dt * velocity_next)
        predicted = circular_blend(predicted, self.theta, self.config.phase_residual_carry)
        return self._phase_pinned(predicted)

    def observe_prediction(self, predicted_theta: np.ndarray) -> float:
        """Update the predictor residual from the just-observed phase error."""

        error_field = circular_delta(self.theta, predicted_theta)
        phase_rate_scale = max(self.config.dt * 0.05, 1e-9)
        error = float(np.clip(np.sqrt(np.mean(error_field * error_field)) / phase_rate_scale, 0.0, 1.0))
        learning_rate = self.config.predictor_learning_rate
        self.predictor_trace = (
            (1.0 - self.config.prediction_trace_decay) * self.predictor_trace
            + learning_rate * error_field
        )
        self.predictor_trace = smooth(
            self.predictor_trace,
            amount=0.12,
            backend=self.config.laplacian_backend,
        )
        self.predictor_trace = np.clip(self.predictor_trace, -1.0, 1.0)
        return error

    def adaptive_damping(self, prediction_error: float) -> float:
        """Lower damping when uncertainty is high and raise it when it is low."""

        base = self.config.damping
        scaled = base / (1.0 + self.config.adaptive_damping_alpha * max(0.0, prediction_error))
        if prediction_error < self.config.prediction_commit_error * 0.5:
            scaled = base * 1.35
        return float(np.clip(scaled, self.config.adaptive_min_damping, self.config.adaptive_max_damping))

    def inject_noise(self, *, scale: float) -> None:
        if scale <= 0:
            return
        self.theta = wrap_phase(self.theta + self.rng.normal(0.0, scale, size=self.theta.shape))

    def step(
        self,
        external_force: np.ndarray | None = None,
        *,
        damping: float | None = None,
    ) -> ResonanceMetrics:
        damping = self.config.damping if damping is None else damping
        lap = laplacian(self.theta, backend=self.config.laplacian_backend)
        landscape_force = self.config.potential_scale * np.sin(self.omega + self.landscape)
        force = (
            (self.config.wave_speed**2) * lap
            + landscape_force
            - self.config.nonlinear_gain * np.sin(self.theta)
            - damping * self.velocity
        )
        if external_force is not None:
            force = force + external_force

        theta_prev = self.theta.copy()
        self.velocity = self.velocity + self.config.dt * force
        self.velocity = self.velocity * max(0.0, 1.0 - damping * self.config.dt)
        self.theta = wrap_phase(self.theta + self.config.dt * self.velocity)
        self.theta = circular_blend(self.theta, theta_prev, self.config.phase_residual_carry)
        self._apply_phase_pins()
        self.step_index += 1
        return self.metrics()

    def run_until_resonance(
        self,
        *,
        max_steps: int | None = None,
        min_steps: int | None = None,
    ) -> list[ResonanceMetrics]:
        max_steps = max_steps or self.config.max_steps
        min_steps = self.config.min_steps if min_steps is None else min_steps
        history: list[ResonanceMetrics] = []
        for _ in range(max_steps):
            metrics = self.step()
            history.append(metrics)
            if metrics.step >= min_steps and metrics.resonant:
                break
        return history

    def metrics(self) -> ResonanceMetrics:
        coherence = float(abs(np.mean(np.exp(1j * self.theta))))
        dx, dy = gradient_components(self.theta)
        gradient = float(np.mean(np.sqrt(dx * dx + dy * dy)))
        energy = float(0.5 * (np.mean(self.velocity * self.velocity) + np.mean(dx * dx + dy * dy)))
        delta = 0.0 if self._last_coherence is None else abs(coherence - self._last_coherence)
        self._last_coherence = coherence
        resonant = (
            self.step_index >= self.config.min_steps
            and coherence >= self.config.resonance_coherence
            and gradient <= self.config.resonance_gradient
            and delta <= self.config.resonance_delta
        )
        return ResonanceMetrics(
            step=self.step_index,
            coherence=coherence,
            gradient=gradient,
            energy=energy,
            coherence_delta=float(delta),
            resonant=bool(resonant),
        )

    def apply_feedback(
        self,
        *,
        success: bool,
        message: str,
        encoder: TextPhaseEncoder,
        learning_rate: float = 0.045,
    ) -> None:
        dx, dy = gradient_components(self.theta)
        basin_stability = np.exp(-(dx * dx + dy * dy))
        direction = 1.0 if success else -1.0
        self.landscape = (
            (1.0 - self.config.memory_decay) * self.landscape
            + direction * learning_rate * basin_stability * np.cos(self.theta)
        )

        for packet in encoder.encode_feedback(message, success=success):
            self.inject_packet(packet, into_landscape=True)

        self.landscape = smooth(
            self.landscape,
            amount=0.18,
            backend=self.config.laplacian_backend,
        )
        self.landscape = np.clip(self.landscape, -4.0, 4.0)

    def consolidate(self, *, cycles: int = 24) -> list[ResonanceMetrics]:
        history: list[ResonanceMetrics] = []
        for _ in range(max(0, cycles)):
            metrics = self.step()
            history.append(metrics)
            dx, dy = gradient_components(self.theta)
            basin_stability = np.exp(-(dx * dx + dy * dy))
            self.landscape = (
                (1.0 - self.config.memory_decay) * self.landscape
                + self.config.memory_gain * 0.12 * basin_stability * np.cos(self.theta)
            )
            self.landscape = smooth(
                self.landscape,
                amount=0.08,
                backend=self.config.laplacian_backend,
            )
            self.landscape = np.clip(self.landscape, -4.0, 4.0)
        return history

    def inject_residual(
        self,
        token: str | int,
        *,
        strength: float = 0.22,
        encoder: TextPhaseEncoder | None = None,
    ) -> None:
        """Inject a generated token back into the field without a KV cache.

        The previous phase state is blended into the current state, then a small
        deterministic packet for the token is stamped into the center-biased
        residual stream. This keeps generation state in phase space instead of
        storing per-token vectors.
        """

        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0.0:
            return
        if self._residual_theta.shape != self.theta.shape:
            self._residual_theta = self.theta.copy()
        self.theta = circular_blend(self.theta, self._residual_theta, self.config.phase_residual_carry)
        encoder = encoder or TextPhaseEncoder(
            self.config.width,
            self.config.height,
            max_packets=8,
            base_amplitude=0.45,
        )
        for packet in encoder.encode(f"residual:{token}", max_packets=4):
            scaled = replace(
                packet,
                amplitude=packet.amplitude * strength,
                radius=max(2, min(packet.radius, max(2, min(self.config.width, self.config.height) // 16))),
            )
            self.inject_packet(scaled)
        self.theta = wrap_phase(self.theta)
        self._residual_theta = self.theta.copy()

    def find_basin(self, *, feature_dim: int = 256, radius: int | None = None) -> BasinFeature:
        """Return a compact feature vector for the current stable attractor."""

        dx, dy = gradient_components(self.theta)
        gradient_field = np.sqrt(dx * dx + dy * dy)
        stability = np.exp(-gradient_field)
        activation = stability * (0.5 + 0.5 * np.abs(np.cos(self.theta))) + 0.15 * np.abs(self.landscape)
        y, x = np.unravel_index(int(np.argmax(activation)), activation.shape)
        center = self.basin_feature_vector(x=int(x), y=int(y), feature_dim=feature_dim, radius=radius)
        coherence = float(abs(np.mean(np.exp(1j * self.theta))))
        gradient = float(np.mean(gradient_field))
        energy = float(0.5 * (np.mean(self.velocity * self.velocity) + np.mean(dx * dx + dy * dy)))
        return BasinFeature(
            x=int(x),
            y=int(y),
            center=center,
            dominant_phase=float(self.theta[y, x]),
            coherence=coherence,
            gradient=gradient,
            energy=energy,
        )

    def basin_feature_vector(
        self,
        *,
        x: int,
        y: int,
        feature_dim: int = 256,
        radius: int | None = None,
    ) -> np.ndarray:
        """Pool local and global phase statistics into a fixed-size vector."""

        feature_dim = max(8, int(feature_dim))
        radius = max(2, int(radius or max(3, round(np.sqrt(feature_dim) / 3.0))))
        offsets = np.arange(-radius, radius + 1)
        yy = (int(y) + offsets) % self.config.height
        xx = (int(x) + offsets) % self.config.width
        region = np.ix_(yy, xx)
        theta_patch = self.theta[region]
        landscape_patch = self.landscape[region] / 4.0
        predictor_patch = self.predictor_trace[region]
        dx, dy = gradient_components(self.theta)
        gradient = np.sqrt(dx * dx + dy * dy)
        phase_hist, _ = np.histogram(
            (self.theta + np.pi) % (2.0 * np.pi),
            bins=16,
            range=(0.0, 2.0 * np.pi),
            density=False,
        )
        phase_hist = phase_hist.astype(np.float64) / max(1, self.theta.size)
        stats = np.array(
            [
                x / max(1, self.config.width - 1),
                y / max(1, self.config.height - 1),
                float(abs(np.mean(np.exp(1j * self.theta)))),
                float(np.mean(gradient)),
                float(np.mean(self.velocity * self.velocity)),
                float(np.mean(np.abs(self.landscape))),
                float(self.step_index / max(1, self.config.max_steps)),
                float(self.theta[int(y), int(x)] / np.pi),
            ],
            dtype=np.float64,
        )
        raw = np.concatenate(
            [
                np.sin(theta_patch).ravel(),
                np.cos(theta_patch).ravel(),
                landscape_patch.ravel(),
                predictor_patch.ravel(),
                phase_hist,
                stats,
            ]
        )
        if raw.size >= feature_dim:
            indices = np.linspace(0, raw.size - 1, feature_dim).astype(np.int64)
            vector = raw[indices]
        else:
            vector = np.zeros(feature_dim, dtype=np.float64)
            vector[: raw.size] = raw
        return np.asarray(vector, dtype=np.float32)

    def reinforce_basin(
        self,
        basin: BasinFeature,
        *,
        gain: float = 0.035,
        decay: float | None = None,
        sigma: float = 3.0,
    ) -> None:
        """Gradient-free topology carving around a persistent basin."""

        decay = self.config.memory_decay if decay is None else float(decay)
        kernel = self.gaussian_kernel(x=basin.x, y=basin.y, sigma=sigma)
        phase_alignment = np.cos(circular_delta(self.theta, basin.dominant_phase))
        update = kernel * phase_alignment
        self.landscape = (1.0 - decay) * self.landscape + gain * update
        self.omega = (1.0 - decay) * self.omega + (gain * 0.18) * update
        self.landscape = smooth(
            self.landscape,
            amount=0.08,
            backend=self.config.laplacian_backend,
        )
        self.omega = smooth(
            self.omega,
            amount=0.04,
            backend=self.config.laplacian_backend,
        )
        self.landscape = np.clip(self.landscape, -4.0, 4.0)
        self.omega = np.clip(self.omega, -1.0, 1.0)

    def gaussian_kernel(self, *, x: int, y: int, sigma: float = 3.0) -> np.ndarray:
        sigma = max(float(sigma), 1e-6)
        yy = np.arange(self.config.height)[:, None]
        xx = np.arange(self.config.width)[None, :]
        dx = np.minimum(np.abs(xx - int(x)), self.config.width - np.abs(xx - int(x)))
        dy = np.minimum(np.abs(yy - int(y)), self.config.height - np.abs(yy - int(y)))
        kernel = np.exp(-((dx * dx + dy * dy) / (2.0 * sigma * sigma)))
        return kernel / max(float(np.max(kernel)), 1e-12)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta=self.theta,
            velocity=self.velocity,
            omega=self.omega,
            landscape=self.landscape,
            predictor_trace=self.predictor_trace,
            pin_phase=self.pin_phase,
            pin_weights=self.pin_weights,
            step_index=np.array([self.step_index], dtype=np.int64),
            config_json=np.array([json.dumps(self.config.__dict__)]),
        )

    def save_quantized(self, path: str | Path) -> None:
        """Save a compact experimental state.

        Phase-like arrays are stored as int8 offsets; low-dynamic-range motion
        terms are stored as fp16. This is intentionally simple and inspectable.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta_i8=quantize_symmetric(self.theta, np.pi),
            landscape_i8=quantize_symmetric(self.landscape, 4.0),
            predictor_trace_i8=quantize_symmetric(self.predictor_trace, 1.0),
            pin_phase_i8=quantize_symmetric(self.pin_phase, np.pi),
            pin_weights_f16=self.pin_weights.astype(np.float16),
            velocity_f16=self.velocity.astype(np.float16),
            omega_f16=self.omega.astype(np.float16),
            step_index=np.array([self.step_index], dtype=np.int64),
            theta_scale=np.array([np.pi], dtype=np.float32),
            landscape_scale=np.array([4.0], dtype=np.float32),
            predictor_trace_scale=np.array([1.0], dtype=np.float32),
            pin_phase_scale=np.array([np.pi], dtype=np.float32),
            config_json=np.array([json.dumps(self.config.__dict__)]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PhaseFieldMesh":
        data = np.load(path, allow_pickle=False)
        config_data = json.loads(str(data["config_json"][0]))
        mesh = cls(MeshConfig(**config_data))
        mesh.theta = data["theta"]
        mesh.velocity = data["velocity"]
        mesh.omega = data["omega"]
        mesh.landscape = data["landscape"]
        if "predictor_trace" in data:
            mesh.predictor_trace = data["predictor_trace"]
        if "pin_phase" in data:
            mesh.pin_phase = data["pin_phase"]
        if "pin_weights" in data:
            mesh.pin_weights = data["pin_weights"]
        mesh.step_index = int(data["step_index"][0])
        mesh._last_coherence = None
        mesh._residual_theta = mesh.theta.copy()
        return mesh

    @classmethod
    def load_quantized(cls, path: str | Path) -> "PhaseFieldMesh":
        data = np.load(path, allow_pickle=False)
        config_data = json.loads(str(data["config_json"][0]))
        mesh = cls(MeshConfig(**config_data))
        mesh.theta = dequantize_symmetric(data["theta_i8"], float(data["theta_scale"][0]))
        mesh.landscape = dequantize_symmetric(
            data["landscape_i8"],
            float(data["landscape_scale"][0]),
        )
        if "predictor_trace_i8" in data:
            mesh.predictor_trace = dequantize_symmetric(
                data["predictor_trace_i8"],
                float(data["predictor_trace_scale"][0]),
            )
        if "pin_phase_i8" in data:
            mesh.pin_phase = dequantize_symmetric(
                data["pin_phase_i8"],
                float(data["pin_phase_scale"][0]),
            )
        if "pin_weights_f16" in data:
            mesh.pin_weights = data["pin_weights_f16"].astype(np.float64)
        mesh.velocity = data["velocity_f16"].astype(np.float64)
        mesh.omega = data["omega_f16"].astype(np.float64)
        mesh.step_index = int(data["step_index"][0])
        mesh._last_coherence = None
        mesh._residual_theta = mesh.theta.copy()
        return mesh

    def _stamp_packet(
        self,
        packet: WavePacket,
        target: np.ndarray,
        velocity_target: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        radius = max(1, int(packet.radius))
        offsets = np.arange(-radius, radius + 1)
        yy = (packet.y + offsets) % self.config.height
        xx = (packet.x + offsets) % self.config.width
        grid_y, grid_x = np.meshgrid(offsets, offsets, indexing="ij")
        dist = np.sqrt(grid_x * grid_x + grid_y * grid_y)
        sigma = max(1.0, radius / 2.2)
        envelope = np.exp(-(dist * dist) / (2.0 * sigma * sigma))
        phase = packet.phase + packet.frequency * dist / (radius + 1.0)
        patch = packet.amplitude * envelope * np.cos(phase)
        target[np.ix_(yy, xx)] += patch
        if velocity_target is not None:
            velocity_target[np.ix_(yy, xx)] += packet.amplitude * packet.frequency * envelope * np.sin(phase)
        return yy, xx

    def _pin_region(self, yy: np.ndarray, xx: np.ndarray, packet: WavePacket) -> None:
        strength = self.config.phase_pin_strength
        if strength <= 0.0 or not should_pin_packet(packet):
            return
        region = np.ix_(yy, xx)
        label_bonus = 1.25 if packet.label.startswith("ctx_") else 1.0
        local_strength = np.clip(strength * abs(packet.amplitude) * label_bonus, 0.0, 1.0)
        current = self.theta[region]
        existing_weight = self.pin_weights[region]
        next_weight = np.maximum(existing_weight, local_strength)
        self.pin_phase[region] = circular_blend(self.pin_phase[region], current, local_strength)
        self.pin_weights[region] = next_weight

    def _apply_phase_pins(self) -> None:
        if self.config.phase_pin_strength <= 0.0:
            return
        active = self.pin_weights > 1e-8
        if not np.any(active):
            return
        self.theta = self._phase_pinned(self.theta)
        weights = np.clip(self.pin_weights, 0.0, 1.0)
        self.velocity = self.velocity * (1.0 - 0.35 * weights)
        self.pin_weights = self.pin_weights * (1.0 - self.config.phase_pin_decay)

    def _phase_pinned(self, theta: np.ndarray) -> np.ndarray:
        if self.config.phase_pin_strength <= 0.0:
            return theta
        if not np.any(self.pin_weights > 1e-8):
            return theta
        return circular_blend(theta, self.pin_phase, np.clip(self.pin_weights, 0.0, 1.0))


class PhaseField:
    """Small compatibility wrapper for direct backend parity checks."""

    def __init__(
        self,
        size: int,
        *,
        backend: str = "auto",
        coupling: float = 0.62,
        damping: float = 0.09,
        dt: float = 0.055,
        omega: np.ndarray | None = None,
    ) -> None:
        self.size = size
        self.backend = backend
        self.coupling = coupling
        self.damping = damping
        self.dt = dt
        self.omega = np.zeros((size, size), dtype=np.float32) if omega is None else omega

    def step(self, theta):
        if self.backend == "jax":
            return step_jax(theta, self.omega, self.coupling, self.damping, self.dt)
        omega = np.asarray(self.omega, dtype=np.asarray(theta).dtype)
        lap = laplacian(np.asarray(theta), backend=self.backend)
        return np.asarray(theta) + (omega + self.coupling * lap - self.damping * np.sin(theta)) * self.dt

    def run(self, theta, *, steps: int):
        if self.backend == "jax":
            return run_jax_steps(theta, self.omega, self.coupling, self.damping, self.dt, steps)
        state = np.asarray(theta)
        for _ in range(max(0, steps)):
            state = self.step(state)
        return state


def wrap_phase(field: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * field))


def circular_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (a - b)))


def circular_blend(current: np.ndarray, target: np.ndarray, weight: float | np.ndarray) -> np.ndarray:
    return wrap_phase(current + np.asarray(weight) * circular_delta(target, current))


def should_pin_packet(packet: WavePacket) -> bool:
    if packet.label == "<global>":
        return False
    return any(character.isalnum() for character in packet.label)


def laplacian(field: np.ndarray, *, backend: str = "auto") -> np.ndarray:
    if use_jax_backend(backend):
        lap = jax_laplacian()(field)
        return np.asarray(lap, dtype=field.dtype)
    if use_scipy_backend(backend):
        return ndimage.convolve(field, LAPLACIAN_KERNEL, mode="wrap")  # type: ignore[union-attr]
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def gradient_components(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = circular_delta(np.roll(field, -1, axis=1), field)
    dy = circular_delta(np.roll(field, -1, axis=0), field)
    return dx, dy


def smooth(field: np.ndarray, *, amount: float, backend: str = "auto") -> np.ndarray:
    if use_jax_backend(backend):
        neighbor_average = np.asarray(jax_neighbor_average()(field), dtype=field.dtype)
        return (1.0 - amount) * field + amount * neighbor_average
    if use_scipy_backend(backend):
        neighbor_average = ndimage.convolve(field, NEIGHBOR_AVERAGE_KERNEL, mode="wrap")  # type: ignore[union-attr]
    else:
        neighbor_average = 0.25 * (
            np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
        )
    return (1.0 - amount) * field + amount * neighbor_average


def use_scipy_backend(backend: str) -> bool:
    if backend == "scipy":
        if ndimage is None:
            raise RuntimeError("SciPy backend requested but scipy.ndimage is unavailable.")
        return True
    return backend == "auto" and ndimage is not None


def use_jax_backend(backend: str) -> bool:
    return backend == "jax"


def jax_laplacian():
    global _JAX_LAPLACIAN
    if _JAX_LAPLACIAN is None:
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - optional backend path
            raise RuntimeError("JAX backend requested but jax is unavailable.") from exc

        @jax.jit
        def _compiled(field):
            return (
                jnp.roll(field, 1, axis=0)
                + jnp.roll(field, -1, axis=0)
                + jnp.roll(field, 1, axis=1)
                + jnp.roll(field, -1, axis=1)
                - 4.0 * field
            )

        _JAX_LAPLACIAN = _compiled
    return _JAX_LAPLACIAN


def jax_neighbor_average():
    global _JAX_NEIGHBOR_AVERAGE
    if _JAX_NEIGHBOR_AVERAGE is None:
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - optional backend path
            raise RuntimeError("JAX backend requested but jax is unavailable.") from exc

        @jax.jit
        def _compiled(field):
            return 0.25 * (
                jnp.roll(field, 1, axis=0)
                + jnp.roll(field, -1, axis=0)
                + jnp.roll(field, 1, axis=1)
                + jnp.roll(field, -1, axis=1)
            )

        _JAX_NEIGHBOR_AVERAGE = _compiled
    return _JAX_NEIGHBOR_AVERAGE


def jax_step_kernel():
    global _JAX_STEP
    if _JAX_STEP is None:
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - optional backend path
            raise RuntimeError("JAX backend requested but jax is unavailable.") from exc

        @jax.jit
        def _compiled(theta, omega, coupling, damping, dt):
            lap = (
                jnp.roll(theta, 1, axis=0)
                + jnp.roll(theta, -1, axis=0)
                + jnp.roll(theta, 1, axis=1)
                + jnp.roll(theta, -1, axis=1)
                - 4.0 * theta
            )
            dtheta = omega + coupling * lap - damping * jnp.sin(theta)
            return theta + dtheta * dt

        _JAX_STEP = _compiled
    return _JAX_STEP


def step_jax(theta, omega, coupling: float, damping: float, dt: float):
    return jax_step_kernel()(theta, omega, coupling, damping, dt)


def jax_scan_kernel():
    global _JAX_SCAN
    if _JAX_SCAN is None:
        try:
            import jax
        except Exception as exc:  # pragma: no cover - optional backend path
            raise RuntimeError("JAX backend requested but jax is unavailable.") from exc

        from functools import partial

        @partial(jax.jit, static_argnums=(5,))
        def _compiled(theta, omega, coupling, damping, dt, steps):
            def body(carry, _):
                return jax_step_kernel()(carry, omega, coupling, damping, dt), None

            return jax.lax.scan(body, theta, None, length=steps)[0]

        _JAX_SCAN = _compiled
    return _JAX_SCAN


def run_jax_steps(theta, omega, coupling: float, damping: float, dt: float, steps: int):
    return jax_scan_kernel()(theta, omega, coupling, damping, dt, int(steps))


def quantize_symmetric(field: np.ndarray, scale: float) -> np.ndarray:
    clipped = np.clip(field, -scale, scale)
    return np.rint(clipped / scale * 127.0).astype(np.int8)


def dequantize_symmetric(values: np.ndarray, scale: float) -> np.ndarray:
    return values.astype(np.float64) / 127.0 * scale
