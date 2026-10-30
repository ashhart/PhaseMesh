from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import MeshConfig
from .encoding import TOKEN_RE, TextPhaseEncoder
from .encoding_structured import StructuredPhaseEncoder, structured_arithmetic_feature_vector
from .field import BasinFeature, PhaseFieldMesh, smooth


TORCH_IMPORT_ERROR = (
    "The experimental model layer needs PyTorch. Install it with "
    "`pip install -e '.[model]'` or use the existing mesh/runtime APIs."
)


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(TORCH_IMPORT_ERROR) from exc
    return torch, nn, functional


class PhaseVocabulary:
    """Small deterministic token map for the experimental decoder head."""

    def __init__(self, tokens: Iterable[str] | None = None, *, max_size: int = 4096) -> None:
        self.max_size = int(max_size)
        if self.max_size < 4:
            raise ValueError("max_size must be at least 4.")
        self.token_to_idx: dict[str, int] = {}
        self.idx_to_token: list[str] = []
        for token in ("<unk>", "<eos>"):
            self.add(token)
        if tokens is not None:
            for token in tokens:
                self.add(token)

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def add(self, token: str) -> int:
        token = normalize_token(token)
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        if len(self.idx_to_token) >= self.max_size:
            return self.token_to_idx["<unk>"]
        index = len(self.idx_to_token)
        self.token_to_idx[token] = index
        self.idx_to_token.append(token)
        return index

    def encode_tokens(self, tokens: Sequence[str], *, add_new: bool = True) -> list[int]:
        indices: list[int] = []
        for token in tokens:
            normalized = normalize_token(token)
            if add_new:
                indices.append(self.add(normalized))
            else:
                indices.append(self.token_to_idx.get(normalized, self.token_to_idx["<unk>"]))
        return indices

    def decode_id(self, token_id: int) -> str:
        if 0 <= int(token_id) < len(self.idx_to_token):
            return self.idx_to_token[int(token_id)]
        return "<unk>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_size": self.max_size,
            "idx_to_token": self.idx_to_token,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseVocabulary:
        vocab = cls(max_size=int(data.get("max_size", 4096)))
        vocab.token_to_idx.clear()
        vocab.idx_to_token.clear()
        for token in data.get("idx_to_token", []):
            vocab.add(str(token))
        for token in ("<unk>", "<eos>"):
            if token not in vocab.token_to_idx:
                vocab.add(token)
        return vocab

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PhaseVocabulary:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class DecoderHead:
    """Tiny basin-to-token MLP.

    PyTorch is imported lazily so the core phase mesh remains lightweight for
    users who only want the field/runtime.
    """

    def __new__(
        cls,
        basin_dim: int = 256,
        hidden: int = 128,
        vocab_capacity: int = 4096,
    ):
        _torch, nn, _functional = _torch_modules()

        class _DecoderHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(int(basin_dim), int(hidden)),
                    nn.SiLU(),
                    nn.Linear(int(hidden), int(vocab_capacity), bias=False),
                )

            def forward(self, basin_center):
                return self.net(basin_center)

        return _DecoderHead()


class RerankerHead:
    """Candidate verifier over a basin state and a candidate token."""

    def __new__(
        cls,
        basin_dim: int = 256,
        hidden: int = 128,
        vocab_capacity: int = 4096,
    ):
        _torch, nn, _functional = _torch_modules()

        class _RerankerHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.candidate_embedding = nn.Embedding(int(vocab_capacity), int(basin_dim))
                self.net = nn.Sequential(
                    nn.Linear(int(basin_dim) * 4, int(hidden)),
                    nn.SiLU(),
                    nn.Linear(int(hidden), 1),
                )

            def forward(self, basin_center, candidate_ids, candidate_center=None):
                candidate_ids = candidate_ids.long().clamp(min=0, max=int(vocab_capacity) - 1)
                candidate_embedding = self.candidate_embedding(candidate_ids)
                if candidate_center is None:
                    candidate_center = candidate_embedding
                combined = _torch.cat(
                    [
                        basin_center,
                        candidate_center,
                        basin_center * candidate_center,
                        candidate_embedding,
                    ],
                    dim=-1,
                )
                return self.net(combined).squeeze(-1)

        return _RerankerHead()


class GateNetwork:
    """Tiny active-basin to resonance-slot selector."""

    def __new__(
        cls,
        basin_dim: int = 256,
        hidden: int = 128,
        num_slots: int = 256,
    ):
        _torch, nn, _functional = _torch_modules()

        class _GateNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(int(basin_dim), int(hidden)),
                    nn.SiLU(),
                    nn.Linear(int(hidden), int(num_slots)),
                )

            def forward(self, basin_center):
                features = basin_center.float()
                if features.ndim == 1:
                    features = features.view(1, -1)
                return self.net(features)

        return _GateNetwork()


class DeltaScorer:
    """Candidate scorer over the residual between active and candidate basins."""

    def __new__(
        cls,
        basin_dim: int = 256,
        hidden: int = 128,
    ):
        _torch, nn, _functional = _torch_modules()

        class _DeltaScorer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scorer = nn.Sequential(
                    nn.Linear(int(basin_dim) * 3, int(hidden)),
                    nn.SiLU(),
                    nn.Linear(int(hidden), 1),
                )

            def forward(self, active, candidate_proto):
                active_features = active.float()
                candidate_features = candidate_proto.float()
                if active_features.ndim == 1:
                    active_features = active_features.view(1, -1)
                if candidate_features.ndim == 1:
                    candidate_features = candidate_features.view(1, -1)
                if active_features.shape[0] == 1 and candidate_features.shape[0] > 1:
                    active_features = active_features.expand(candidate_features.shape[0], -1)
                elif candidate_features.shape[0] == 1 and active_features.shape[0] > 1:
                    candidate_features = candidate_features.expand(active_features.shape[0], -1)
                delta = active_features - candidate_features
                features = _torch.cat(
                    [
                        delta,
                        delta.abs(),
                        active_features * candidate_features,
                    ],
                    dim=-1,
                )
                return self.scorer(features).squeeze(-1)

        return _DeltaScorer()


class PrototypeReadout:
    """Distance-based prototype resolver for structurally anchored basins."""

    def __new__(
        cls,
        prototypes: dict[str, np.ndarray | Sequence[float]],
        *,
        prototype_target_ids: Sequence[int] | None = None,
        vocab_capacity: int = 4096,
        temperature: float = 0.1,
        direct_scale: float = 8.0,
    ):
        torch, nn, functional = _torch_modules()
        prototype_items = sorted(prototypes.items())
        if not prototype_items:
            raise ValueError("PrototypeReadout needs at least one structural prototype.")
        keys = [str(key) for key, _value in prototype_items]
        prototype_array = np.asarray([value for _key, value in prototype_items], dtype=np.float32)
        if prototype_array.ndim != 2:
            raise ValueError("Structural prototypes must be a 2D key x feature array.")
        target_ids = (
            np.asarray([int(item) for item in prototype_target_ids], dtype=np.int64)
            if prototype_target_ids is not None
            else np.full((len(keys),), -1, dtype=np.int64)
        )
        if target_ids.shape[0] != len(keys):
            raise ValueError("prototype_target_ids must match the number of prototypes.")

        class _PrototypeReadout(nn.Module):
            decoder_type = "prototype-readout"

            def __init__(self) -> None:
                super().__init__()
                self.prototype_keys = keys
                self.temperature = float(max(float(temperature), 1e-6))
                self.direct_scale = float(direct_scale)
                self.register_buffer("prototypes", torch.as_tensor(prototype_array, dtype=torch.float32))
                self.register_buffer("prototype_target_ids", torch.as_tensor(target_ids, dtype=torch.long))
                self.result_head = nn.Linear(len(keys), int(vocab_capacity), bias=True)
                nn.init.zeros_(self.result_head.weight)
                nn.init.zeros_(self.result_head.bias)

            def forward(self, basin_center):
                features = basin_center.float()
                if features.ndim == 1:
                    features = features.view(1, -1)
                if features.shape[-1] != self.prototypes.shape[-1]:
                    raise ValueError(
                        f"basin feature dim {features.shape[-1]} does not match prototype dim "
                        f"{self.prototypes.shape[-1]}"
                    )
                distances = torch.cdist(features, self.prototypes)
                weights = functional.softmax(-distances / self.temperature, dim=-1)
                logits = self.result_head(weights)
                valid = self.prototype_target_ids >= 0
                if bool(valid.any()):
                    target_ids = self.prototype_target_ids[valid].clamp(min=0, max=logits.shape[-1] - 1)
                    direct = torch.zeros_like(logits)
                    direct.scatter_add_(
                        1,
                        target_ids.view(1, -1).expand(features.shape[0], -1),
                        weights[:, valid] * self.direct_scale,
                    )
                    logits = logits + direct
                return logits

        return _PrototypeReadout()


@dataclass(frozen=True)
class PhaseObservation:
    prompt_tokens: list[str]
    target_token: str | None
    target_id: int | None
    steps_used: int
    mean_prediction_error: float
    final_prediction_error: float
    decoder_loss: float | None
    basin: dict[str, Any]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "target_token": self.target_token,
            "target_id": self.target_id,
            "steps_used": self.steps_used,
            "mean_prediction_error": self.mean_prediction_error,
            "final_prediction_error": self.final_prediction_error,
            "decoder_loss": self.decoder_loss,
            "basin": self.basin,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class GenerationStep:
    token: str
    token_id: int
    probability: float
    basin: dict[str, Any]
    prediction_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "token_id": self.token_id,
            "probability": self.probability,
            "basin": self.basin,
            "prediction_error": self.prediction_error,
        }


class PhaseModel:
    """Experimental self-supervised layer over `PhaseFieldMesh`.

    This is a prototype generator/trainer, not a pretrained language model. The
    field learns predictive residuals and topology without answer labels; the
    decoder head is the only backprop-trained component, mapping basin features
    to next-token logits.
    """

    def __init__(
        self,
        *,
        grid_size: int = 128,
        vocab: PhaseVocabulary | None = None,
        vocab_capacity: int = 4096,
        basin_dim: int = 256,
        hidden: int = 128,
        seed: int = 7,
        backend: str = "auto",
        pin_strength: float = 0.25,
        residual_carry: float = 0.08,
        learning_rate: float = 2e-4,
        num_slots: int = 0,
        encoder_mode: str = "text",
        structured_result_hint: bool = False,
        structured_feature_strength: float = 2.0,
        create_decoder: bool = True,
    ) -> None:
        self.config = MeshConfig(
            width=grid_size,
            height=grid_size,
            seed=seed,
            laplacian_backend=backend,
            phase_pin_strength=pin_strength,
            phase_residual_carry=residual_carry,
        )
        self.field = PhaseFieldMesh(self.config)
        self.encoder_mode = str(encoder_mode)
        self.structured_result_hint = bool(structured_result_hint)
        self.structured_feature_strength = float(structured_feature_strength)
        if self.encoder_mode == "structured":
            self.encoder = StructuredPhaseEncoder(
                grid_size,
                grid_size,
                include_result_hint=self.structured_result_hint,
            )
        elif self.encoder_mode == "text":
            self.encoder = TextPhaseEncoder(grid_size, grid_size)
        else:
            raise ValueError(f"unsupported encoder_mode: {encoder_mode}")
        self.vocab = vocab or PhaseVocabulary(max_size=vocab_capacity)
        self.basin_dim = int(basin_dim)
        self.hidden = int(hidden)
        self.vocab_capacity = int(vocab_capacity)
        self.learning_rate = float(learning_rate)
        self.decoder = None
        self.optimizer = None
        self.reranker = None
        self.reranker_optimizer = None
        self.gate_net = None
        self.gate_optimizer = None
        self.gate_trained_steps = 0
        self.delta_scorer = None
        self.delta_optimizer = None
        self.delta_trained_steps = 0
        self.structural_prototypes: dict[str, np.ndarray] = {}
        self.feature_omega = np.zeros(
            (self.field.config.height, self.field.config.width, self.basin_dim),
            dtype=np.float32,
        )
        self.num_slots = max(0, int(num_slots))
        self.feature_slots: np.ndarray | None = None
        self.feature_slot_overrides: dict[str, np.ndarray] = {}
        self.feature_slot_gates: dict[str, np.ndarray] = {}
        self._last_basin_cell: tuple[int, int] | None = None
        if self.num_slots > 0:
            self._ensure_feature_slots()
        if create_decoder:
            self._init_decoder()
            self._init_reranker()

    def _init_decoder(self) -> None:
        torch, _nn, _functional = _torch_modules()
        self.decoder = DecoderHead(
            basin_dim=self.basin_dim,
            hidden=self.hidden,
            vocab_capacity=self.vocab_capacity,
        )
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.learning_rate)

    def _init_reranker(self) -> None:
        torch, _nn, _functional = _torch_modules()
        self.reranker = RerankerHead(
            basin_dim=self.basin_dim,
            hidden=self.hidden,
            vocab_capacity=self.vocab_capacity,
        )
        self.reranker_optimizer = torch.optim.AdamW(self.reranker.parameters(), lr=self.learning_rate)

    def _init_gate(self) -> None:
        if self.num_slots <= 0:
            self.gate_net = None
            self.gate_optimizer = None
            return
        torch, _nn, _functional = _torch_modules()
        self.gate_net = GateNetwork(
            basin_dim=self.basin_dim,
            hidden=self.hidden,
            num_slots=self.num_slots,
        )
        self.gate_optimizer = torch.optim.AdamW(self.gate_net.parameters(), lr=self.learning_rate)

    def _init_delta_scorer(self) -> None:
        torch, _nn, _functional = _torch_modules()
        self.delta_scorer = DeltaScorer(basin_dim=self.basin_dim, hidden=self.hidden)
        self.delta_optimizer = torch.optim.AdamW(self.delta_scorer.parameters(), lr=self.learning_rate)

    def _ensure_feature_omega(self) -> None:
        expected = (self.field.config.height, self.field.config.width, self.basin_dim)
        if getattr(self, "feature_omega", None) is None or self.feature_omega.shape != expected:
            self.feature_omega = np.zeros(expected, dtype=np.float32)

    def _ensure_feature_slots(self) -> None:
        if self.num_slots <= 0:
            self.feature_slots = None
            return
        expected = (
            self.field.config.height,
            self.field.config.width,
            self.num_slots,
            self.basin_dim,
        )
        if getattr(self, "feature_slots", None) is None or self.feature_slots.shape != expected:
            self.feature_slots = np.zeros(expected, dtype=np.float16)

    def apply_structured_feature_overlay(self, basin: BasinFeature, text: str | Sequence[str]) -> BasinFeature:
        """Add typed arithmetic factors to the basin feature vector when enabled."""

        if self.encoder_mode != "structured" or self.structured_feature_strength <= 0.0:
            return basin
        text_value = text if isinstance(text, str) else " ".join(str(item) for item in text)
        overlay = structured_arithmetic_feature_vector(text_value, basin.center.shape[0])
        if not np.any(overlay):
            return basin
        center = (
            np.asarray(basin.center, dtype=np.float32)
            + float(self.structured_feature_strength) * overlay.astype(np.float32)
        ).astype(np.float32)
        return BasinFeature(
            x=basin.x,
            y=basin.y,
            center=center,
            dominant_phase=basin.dominant_phase,
            coherence=basin.coherence,
            gradient=basin.gradient,
            energy=basin.energy,
        )

    def get_slot_index(self, op_type: str, result_val: str | int | float) -> int:
        """Return a stable resonance slot for an operation/result pair."""

        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive to use resonance slots.")
        seed = f"{normalize_token(op_type)}:{normalize_token(str(result_val))}".encode("utf-8")
        digest = hashlib.blake2b(seed, digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.num_slots

    def get_slot_index_for_key(self, key: str) -> int:
        """Return the keyed slot index for structural keys like ``add:17``."""

        full_key = normalize_token(str(key))
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive to use resonance slots.")
        if ":" not in full_key:
            raise ValueError(f"Structural key must include an operation prefix: {key!r}")
        digest = hashlib.sha256(full_key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.num_slots

    @staticmethod
    def feature_slot_override_key(y: int, x: int, key: str) -> str:
        return f"{int(y)}:{int(x)}:{str(key)}"

    def apply_feature_omega(self, basin: BasinFeature) -> BasinFeature:
        """Add the local high-dimensional residual overlay to a basin readout."""

        self._ensure_feature_omega()
        y = int(basin.y) % self.field.config.height
        x = int(basin.x) % self.field.config.width
        self._last_basin_cell = (y, x)
        residual = np.asarray(
            self.feature_omega[y, x],
            dtype=np.float32,
        )
        if residual.shape != basin.center.shape:
            return basin
        return BasinFeature(
            x=basin.x,
            y=basin.y,
            center=(np.asarray(basin.center, dtype=np.float32) + residual).astype(np.float32),
            dominant_phase=basin.dominant_phase,
            coherence=basin.coherence,
            gradient=basin.gradient,
            energy=basin.energy,
        )

    def slot_augmented_center(
        self,
        basin_center: np.ndarray | Sequence[float],
        key: str,
        *,
        basin_cell: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Read a basin through the resonance slot associated with a candidate key."""

        center = np.asarray(basin_center, dtype=np.float32).copy()
        if self.num_slots <= 0:
            return center
        self._ensure_feature_slots()
        if self.feature_slots is None:
            return center
        cell = basin_cell if basin_cell is not None else self._last_basin_cell
        if cell is None:
            return center
        try:
            slot_idx = self.get_slot_index_for_key(str(key))
        except ValueError:
            return center
        y, x = int(cell[0]) % self.field.config.height, int(cell[1]) % self.field.config.width
        override_key = self.feature_slot_override_key(y, x, str(key))
        override = self.feature_slot_overrides.get(override_key)
        if override is not None:
            residual = np.asarray(override, dtype=np.float32)
            if residual.shape == center.shape:
                center += residual
                return center.astype(np.float32)
        if self.feature_slot_overrides:
            return center.astype(np.float32)
        residual = np.asarray(self.feature_slots[y, x, slot_idx], dtype=np.float32)
        if residual.shape == center.shape:
            center += residual
        return center.astype(np.float32)

    @staticmethod
    def _softmax_numpy(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        shifted = values - float(np.max(values))
        exp_values = np.exp(np.clip(shifted, -60.0, 60.0))
        denom = float(np.sum(exp_values))
        if denom <= 1e-12:
            return np.full_like(exp_values, 1.0 / max(1, exp_values.size), dtype=np.float64)
        return exp_values / denom

    @staticmethod
    def _row_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_arr = np.asarray(left, dtype=np.float64).reshape(1, -1)
        right_arr = np.asarray(right, dtype=np.float64)
        if right_arr.ndim == 1:
            right_arr = right_arr.reshape(1, -1)
        denom = np.linalg.norm(left_arr, axis=1) * np.linalg.norm(right_arr, axis=1)
        denom = np.maximum(denom, 1e-12)
        return np.sum(left_arr * right_arr, axis=1) / denom

    def read_resonant_slots(
        self,
        basin_center: np.ndarray | Sequence[float],
        *,
        basin_cell: tuple[int, int] | None = None,
        temperature: float = 10.0,
        top_k: int | None = None,
        min_gate_margin: float = 0.02,
    ) -> dict[str, Any]:
        """Return a prompt-gated slot readout for the active basin cell.

        This is intentionally candidate-independent: the active prompt chooses
        one shared residual mixture, then every prototype is scored against that
        same readout. That prevents the self-match loop where each candidate
        evaluates itself through its own slot.
        """

        center = np.asarray(basin_center, dtype=np.float32).copy()
        if self.num_slots <= 0:
            return {
                "center": center,
                "residual": np.zeros_like(center, dtype=np.float32),
                "slot_keys": [],
                "weights": [],
                "similarities": [],
                "mode": "disabled",
            }
        self._ensure_feature_slots()
        if self.feature_slots is None:
            return {
                "center": center,
                "residual": np.zeros_like(center, dtype=np.float32),
                "slot_keys": [],
                "weights": [],
                "similarities": [],
                "mode": "missing_slots",
            }
        cell = basin_cell if basin_cell is not None else self._last_basin_cell
        if cell is None:
            return {
                "center": center,
                "residual": np.zeros_like(center, dtype=np.float32),
                "slot_keys": [],
                "weights": [],
                "similarities": [],
                "mode": "missing_cell",
            }

        y, x = int(cell[0]) % self.field.config.height, int(cell[1]) % self.field.config.width
        prefix = f"{y}:{x}:"
        override_keys = sorted(key for key in self.feature_slot_overrides if key.startswith(prefix))
        if override_keys and self.gate_net is not None and self.gate_trained_steps > 0:
            torch, _nn, functional = _torch_modules()
            device = next(self.gate_net.parameters()).device
            features = torch.as_tensor(center, dtype=torch.float32, device=device).view(1, -1)
            with torch.no_grad():
                logits = self.gate_net(features)
                gate_probs = functional.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float64)
            slot_to_vectors: dict[int, list[np.ndarray]] = {}
            for key in override_keys:
                structural_key = str(key).split(":", 2)[2]
                try:
                    slot_index = self.get_slot_index_for_key(structural_key)
                except ValueError:
                    continue
                slot_to_vectors.setdefault(int(slot_index), []).append(
                    np.asarray(self.feature_slot_overrides[key], dtype=np.float32)
                )
            if not slot_to_vectors:
                return {
                    "center": center,
                    "residual": np.zeros_like(center, dtype=np.float32),
                    "slot_keys": [],
                    "weights": [],
                    "similarities": [],
                    "gate_margin": 0.0,
                    "mode": "learned-gate-empty",
                }
            ordered_slots = sorted(slot_to_vectors)
            if top_k is not None and int(top_k) > 0 and int(top_k) < len(ordered_slots):
                ordered_slots = sorted(
                    ordered_slots,
                    key=lambda slot: float(gate_probs[int(slot)]),
                    reverse=True,
                )[: int(top_k)]
            residuals = np.asarray(
                [np.mean(slot_to_vectors[slot], axis=0) for slot in ordered_slots],
                dtype=np.float32,
            )
            raw_weights = np.asarray([gate_probs[int(slot)] for slot in ordered_slots], dtype=np.float64)
            if gate_probs.size > 1:
                ordered_gate = np.sort(gate_probs)[::-1]
                gate_margin = float(ordered_gate[0] - ordered_gate[1])
            elif gate_probs.size == 1:
                gate_margin = float(gate_probs[0])
            else:
                gate_margin = 0.0
            top_slot = int(np.argmax(gate_probs)) if gate_probs.size else -1
            if top_slot not in slot_to_vectors or gate_margin < float(min_gate_margin):
                return {
                    "center": center,
                    "residual": np.zeros_like(center, dtype=np.float32),
                    "slot_keys": [f"{y}:{x}:slot:{slot}" for slot in ordered_slots],
                    "weights": [float(item) for item in raw_weights.tolist()],
                    "similarities": [float(item) for item in raw_weights.tolist()],
                    "gate_margin": gate_margin,
                    "top_slot": top_slot,
                    "mode": "learned-gate-low-confidence",
                }
            weight_sum = float(np.sum(raw_weights))
            if weight_sum <= 1e-12:
                weights = np.full_like(raw_weights, 1.0 / max(1, raw_weights.size), dtype=np.float64)
            else:
                weights = raw_weights / weight_sum
            residual = np.sum(residuals * weights.reshape(-1, 1), axis=0).astype(np.float32)
            return {
                "center": (center + residual).astype(np.float32),
                "residual": residual,
                "slot_keys": [f"{y}:{x}:slot:{slot}" for slot in ordered_slots],
                "weights": [float(item) for item in weights.tolist()],
                "similarities": [float(item) for item in raw_weights.tolist()],
                "gate_margin": gate_margin,
                "top_slot": top_slot,
                "mode": "learned-gate",
            }

        if override_keys:
            residuals = np.asarray(
                [self.feature_slot_overrides[key] for key in override_keys],
                dtype=np.float32,
            )
            gate_vectors = np.asarray(
                [
                    self.feature_slot_gates.get(key, self.feature_slot_overrides[key])
                    for key in override_keys
                ],
                dtype=np.float32,
            )
            slot_keys = override_keys
            mode = "exact-gated"
        else:
            residuals = np.asarray(self.feature_slots[y, x], dtype=np.float32)
            gate_vectors = residuals
            slot_keys = [f"{y}:{x}:slot:{slot}" for slot in range(residuals.shape[0])]
            mode = "dense-gated"

        if residuals.ndim != 2 or residuals.shape[1] != center.shape[0]:
            return {
                "center": center,
                "residual": np.zeros_like(center, dtype=np.float32),
                "slot_keys": [],
                "weights": [],
                "similarities": [],
                "mode": "shape_mismatch",
            }

        similarities = self._row_cosine_similarity(center, gate_vectors)
        if top_k is not None and int(top_k) > 0 and int(top_k) < similarities.shape[0]:
            keep = np.argsort(similarities)[-int(top_k):]
            keep = keep[np.argsort(similarities[keep])[::-1]]
            residuals = residuals[keep]
            similarities = similarities[keep]
            slot_keys = [slot_keys[int(index)] for index in keep]
        weights = self._softmax_numpy(similarities * float(temperature)).astype(np.float32)
        if weights.size > 1:
            ordered_weights = np.sort(weights)[::-1]
            gate_margin = float(ordered_weights[0] - ordered_weights[1])
        elif weights.size == 1:
            gate_margin = float(weights[0])
        else:
            gate_margin = 0.0
        if weights.size == 0 or gate_margin < float(min_gate_margin):
            return {
                "center": center,
                "residual": np.zeros_like(center, dtype=np.float32),
                "slot_keys": slot_keys,
                "weights": [float(item) for item in weights.tolist()],
                "similarities": [float(item) for item in similarities.tolist()],
                "gate_margin": gate_margin,
                "mode": f"{mode}-low-confidence",
            }
        residual = np.sum(residuals * weights.reshape(-1, 1), axis=0).astype(np.float32)
        return {
            "center": (center + residual).astype(np.float32),
            "residual": residual,
            "slot_keys": slot_keys,
            "weights": [float(item) for item in weights.tolist()],
            "similarities": [float(item) for item in similarities.tolist()],
            "gate_margin": gate_margin,
            "mode": mode,
        }

    def use_prototype_readout(self, *, temperature: float = 0.1, direct_scale: float = 8.0) -> None:
        """Replace the generic MLP decoder with a fixed-prototype resolver."""

        torch, _nn, _functional = _torch_modules()
        prototype_target_ids = [
            self.vocab.add(prototype_target_from_key(key))
            for key in sorted(self.structural_prototypes)
        ]
        self.decoder = PrototypeReadout(
            self.structural_prototypes,
            prototype_target_ids=prototype_target_ids,
            vocab_capacity=self.vocab_capacity,
            temperature=temperature,
            direct_scale=direct_scale,
        )
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.learning_rate)

    def reset_optimizer(self, learning_rate: float | None = None) -> None:
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        if self.decoder is None:
            self._init_decoder()
        if self.reranker is None:
            self._init_reranker()
        if self.num_slots > 0 and self.gate_net is None:
            self._init_gate()
        if self.delta_scorer is None:
            self._init_delta_scorer()
        torch, _nn, _functional = _torch_modules()
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.learning_rate)
        self.reranker_optimizer = torch.optim.AdamW(self.reranker.parameters(), lr=self.learning_rate)
        if self.gate_net is not None:
            self.gate_optimizer = torch.optim.AdamW(self.gate_net.parameters(), lr=self.learning_rate)
        if self.delta_scorer is not None:
            self.delta_optimizer = torch.optim.AdamW(self.delta_scorer.parameters(), lr=self.learning_rate)

    def tokenize(self, text_or_tokens: str | Sequence[str]) -> list[str]:
        if isinstance(text_or_tokens, str):
            tokens = TOKEN_RE.findall(text_or_tokens.lower())
        else:
            tokens = [normalize_token(item) for item in text_or_tokens]
        return tokens or ["<empty>"]

    def observe_text(
        self,
        text_or_tokens: str | Sequence[str],
        *,
        steps_per_chunk: int = 20,
        train_decoder: bool = True,
        train_topology: bool = True,
        freeze_omega: bool = False,
        reset: bool = True,
        reinforce_gain: float = 0.035,
    ) -> PhaseObservation:
        tokens = self.tokenize(text_or_tokens)
        prompt_tokens = tokens[:-1] if len(tokens) > 1 else tokens
        target_token = tokens[-1] if len(tokens) > 1 else None
        token_ids = self.vocab.encode_tokens(tokens, add_new=True)
        omega_before = self.field.omega.copy() if freeze_omega else None
        if reset:
            self.field.reset_field()
        self.field.inject_text(" ".join(prompt_tokens), self.encoder)

        prediction_errors: list[float] = []
        metrics = self.field.metrics()
        for _ in range(max(1, int(steps_per_chunk))):
            predicted = self.field.predict_phase()
            metrics = self.field.step()
            prediction_errors.append(self.field.observe_prediction(predicted))

        basin = self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim))
        target_id: int | None = None
        decoder_loss: float | None = None
        if target_token is not None:
            target_id = token_ids[-1]
        if train_topology:
            self.field.reinforce_basin(basin, gain=reinforce_gain)
        if omega_before is not None:
            self.field.omega = omega_before
        if train_decoder and target_id is not None:
            decoder_loss = self.train_decoder_step(basin.center, target_id)

        return PhaseObservation(
            prompt_tokens=prompt_tokens,
            target_token=target_token,
            target_id=target_id,
            steps_used=max(1, int(steps_per_chunk)),
            mean_prediction_error=float(sum(prediction_errors) / max(1, len(prediction_errors))),
            final_prediction_error=float(prediction_errors[-1] if prediction_errors else 0.0),
            decoder_loss=decoder_loss,
            basin=basin.to_dict(),
            metrics=metrics.to_dict(),
        )

    def train_decoder_step(self, basin_center: np.ndarray, target_id: int, *, mode: str = "next-token") -> float:
        if self.decoder is None or self.optimizer is None:
            self._init_decoder()
        if target_id >= self.vocab_capacity:
            raise ValueError("target_id exceeds decoder vocab_capacity.")
        torch, _nn, _functional = _torch_modules()
        device = next(self.decoder.parameters()).device
        features = torch.as_tensor(basin_center, dtype=torch.float32, device=device).view(1, -1)
        target = torch.tensor([int(target_id)], dtype=torch.long, device=device)
        self.optimizer.zero_grad()
        logits = self.decoder(features)
        loss = decoder_training_loss(logits, target, self.vocab, mode=mode)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def train_decoder_batch(
        self,
        basin_centers: Sequence[Sequence[float] | np.ndarray],
        target_ids: Sequence[int],
        *,
        mode: str = "next-token",
    ) -> float:
        if self.decoder is None or self.optimizer is None:
            self._init_decoder()
        if len(basin_centers) != len(target_ids):
            raise ValueError("basin_centers and target_ids must have the same length.")
        if not basin_centers:
            return 0.0
        max_target = max(int(item) for item in target_ids)
        if max_target >= self.vocab_capacity:
            raise ValueError("target_id exceeds decoder vocab_capacity.")
        torch, _nn, _functional = _torch_modules()
        device = next(self.decoder.parameters()).device
        features = torch.as_tensor(np.asarray(basin_centers, dtype=np.float32), dtype=torch.float32, device=device)
        targets = torch.as_tensor([int(item) for item in target_ids], dtype=torch.long, device=device)
        self.optimizer.zero_grad()
        logits = self.decoder(features)
        loss = decoder_training_loss(logits, targets, self.vocab, mode=mode)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def decoder_loss(self, basin_center: np.ndarray | Sequence[float], target_id: int) -> float:
        if self.decoder is None:
            self._init_decoder()
        torch, _nn, functional = _torch_modules()
        device = next(self.decoder.parameters()).device
        with torch.no_grad():
            features = torch.as_tensor(basin_center, dtype=torch.float32, device=device).view(1, -1)
            target = torch.tensor([int(target_id)], dtype=torch.long, device=device)
            logits = self.decoder(features)
            loss = functional.cross_entropy(logits, target)
        return float(loss.detach().cpu().item())

    def train_gate_contrastive(
        self,
        basin_center: np.ndarray | Sequence[float],
        *,
        correct_key: str,
        wrong_keys: Sequence[str] = (),
        margin: float = 0.25,
    ) -> dict[str, float | int | bool]:
        """Train the prompt-conditioned slot selector with contrastive pressure."""

        if self.num_slots <= 0:
            return {
                "used": False,
                "reason": "slots_disabled",
                "loss": float("nan"),
                "correct_probability": 0.0,
                "gate_margin": 0.0,
                "top_slot": -1,
                "correct_slot": -1,
                "top_match": False,
            }
        if self.gate_net is None or self.gate_optimizer is None:
            self._init_gate()
        torch, _nn, functional = _torch_modules()
        device = next(self.gate_net.parameters()).device
        correct_slot = self.get_slot_index_for_key(correct_key)
        wrong_slots = sorted({
            self.get_slot_index_for_key(wrong)
            for wrong in wrong_keys
            if str(wrong) != str(correct_key) and ":" in str(wrong)
        })

        features = torch.as_tensor(basin_center, dtype=torch.float32, device=device).view(1, -1)
        target = torch.tensor([int(correct_slot)], dtype=torch.long, device=device)
        self.gate_optimizer.zero_grad()
        logits = self.gate_net(features)
        ce_loss = functional.cross_entropy(logits, target)
        probs = functional.softmax(logits, dim=-1)
        correct_prob = probs[:, int(correct_slot)]
        margin_loss = torch.zeros((), dtype=torch.float32, device=device)
        if wrong_slots:
            wrong_tensor = torch.tensor(wrong_slots, dtype=torch.long, device=device)
            wrong_prob = probs.index_select(1, wrong_tensor).mean(dim=-1)
            margin_loss = functional.relu(float(margin) - correct_prob + wrong_prob).mean()
        loss = ce_loss + margin_loss
        loss.backward()
        self.gate_optimizer.step()
        self.gate_trained_steps += 1

        with torch.no_grad():
            updated_logits = self.gate_net(features)
            updated_probs = functional.softmax(updated_logits, dim=-1)[0]
            top_values, top_indices = torch.topk(updated_probs, k=min(2, int(self.num_slots)))
            top_slot = int(top_indices[0].detach().cpu().item())
            if top_values.numel() > 1:
                gate_margin = float((top_values[0] - top_values[1]).detach().cpu().item())
            else:
                gate_margin = float(top_values[0].detach().cpu().item())
            correct_probability = float(updated_probs[int(correct_slot)].detach().cpu().item())
        return {
            "used": True,
            "loss": float(loss.detach().cpu().item()),
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "margin_loss": float(margin_loss.detach().cpu().item()),
            "correct_probability": correct_probability,
            "gate_margin": gate_margin,
            "top_slot": top_slot,
            "correct_slot": int(correct_slot),
            "top_match": bool(top_slot == int(correct_slot)),
        }

    def train_delta_contrastive(
        self,
        basin_center: np.ndarray | Sequence[float],
        target_proto: np.ndarray | Sequence[float],
        wrong_protos: Sequence[np.ndarray | Sequence[float]] = (),
        *,
        margin: float = 1.0,
    ) -> dict[str, float | int | bool]:
        """Train a scorer over active-minus-candidate residuals."""

        if self.delta_scorer is None or self.delta_optimizer is None:
            self._init_delta_scorer()
        torch, _nn, functional = _torch_modules()
        device = next(self.delta_scorer.parameters()).device
        active = torch.as_tensor(basin_center, dtype=torch.float32, device=device).view(1, -1)
        target = torch.as_tensor(target_proto, dtype=torch.float32, device=device).view(1, -1)
        wrong_list = [
            torch.as_tensor(item, dtype=torch.float32, device=device).view(1, -1)
            for item in wrong_protos
        ]

        self.delta_optimizer.zero_grad()
        target_score = self.delta_scorer(active, target).view(1)
        if wrong_list:
            wrong_tensor = torch.cat(wrong_list, dim=0)
            wrong_scores = self.delta_scorer(active.expand(wrong_tensor.shape[0], -1), wrong_tensor)
            hardest_wrong = wrong_scores.max().view(1)
            mean_wrong = wrong_scores.mean().view(1)
            loss = functional.relu(float(margin) - target_score + wrong_scores).mean()
            pre_top_match = bool((target_score > hardest_wrong).detach().cpu().item())
        else:
            hardest_wrong = torch.zeros((1,), dtype=torch.float32, device=device)
            mean_wrong = torch.zeros((1,), dtype=torch.float32, device=device)
            loss = -target_score.mean()
            pre_top_match = True
        loss.backward()
        self.delta_optimizer.step()
        self.delta_trained_steps += 1

        with torch.no_grad():
            updated_target_score = self.delta_scorer(active, target).view(1)
            if wrong_list:
                wrong_tensor = torch.cat(wrong_list, dim=0)
                updated_wrong_scores = self.delta_scorer(
                    active.expand(wrong_tensor.shape[0], -1),
                    wrong_tensor,
                )
                updated_hardest_wrong = updated_wrong_scores.max().view(1)
                updated_mean_wrong = updated_wrong_scores.mean().view(1)
                updated_top_match = bool(
                    (updated_target_score > updated_hardest_wrong).detach().cpu().item()
                )
            else:
                updated_hardest_wrong = torch.zeros((1,), dtype=torch.float32, device=device)
                updated_mean_wrong = torch.zeros((1,), dtype=torch.float32, device=device)
                updated_top_match = True
            delta_margin = updated_target_score - updated_hardest_wrong
        return {
            "used": True,
            "loss": float(loss.detach().cpu().item()),
            "target_score": float(updated_target_score.detach().cpu().item()),
            "wrong_score": float(updated_mean_wrong.detach().cpu().item()),
            "hardest_wrong_score": float(updated_hardest_wrong.detach().cpu().item()),
            "delta_margin": float(delta_margin.detach().cpu().item()),
            "top_match": updated_top_match,
            "wrong_count": int(len(wrong_list)),
            "pre_target_score": float(target_score.detach().cpu().item()),
            "pre_wrong_score": float(mean_wrong.detach().cpu().item()),
            "pre_hardest_wrong_score": float(hardest_wrong.detach().cpu().item()),
            "pre_top_match": pre_top_match,
        }

    def score_delta_candidates(
        self,
        basin_center: np.ndarray | Sequence[float],
        prototypes: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Score candidate prototypes with the trained delta scorer."""

        if self.delta_scorer is None or self.delta_trained_steps <= 0:
            raise ValueError("delta scorer has not been trained")
        torch, _nn, _functional = _torch_modules()
        device = next(self.delta_scorer.parameters()).device
        active = torch.as_tensor(basin_center, dtype=torch.float32, device=device).view(1, -1)
        candidates = torch.as_tensor(
            np.asarray(prototypes, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        if candidates.ndim == 1:
            candidates = candidates.view(1, -1)
        with torch.no_grad():
            scores = self.delta_scorer(active.expand(candidates.shape[0], -1), candidates)
        return scores.detach().cpu().numpy().astype(np.float64)

    def _snapshot_field_state(self) -> dict[str, Any]:
        return {
            "theta": self.field.theta.copy(),
            "velocity": self.field.velocity.copy(),
            "omega": self.field.omega.copy(),
            "landscape": self.field.landscape.copy(),
            "predictor_trace": self.field.predictor_trace.copy(),
            "pin_phase": self.field.pin_phase.copy(),
            "pin_weights": self.field.pin_weights.copy(),
            "step_index": int(self.field.step_index),
            "last_coherence": self.field._last_coherence,
            "residual_theta": self.field._residual_theta.copy(),
        }

    def _restore_field_state(self, state: dict[str, Any]) -> None:
        self.field.theta = np.asarray(state["theta"], dtype=np.float64).copy()
        self.field.velocity = np.asarray(state["velocity"], dtype=np.float64).copy()
        self.field.omega = np.asarray(state["omega"], dtype=np.float64).copy()
        self.field.landscape = np.asarray(state["landscape"], dtype=np.float64).copy()
        self.field.predictor_trace = np.asarray(state["predictor_trace"], dtype=np.float64).copy()
        self.field.pin_phase = np.asarray(state["pin_phase"], dtype=np.float64).copy()
        self.field.pin_weights = np.asarray(state["pin_weights"], dtype=np.float64).copy()
        self.field.step_index = int(state["step_index"])
        self.field._last_coherence = state["last_coherence"]
        self.field._residual_theta = np.asarray(state["residual_theta"], dtype=np.float64).copy()

    def score_joint_stability(
        self,
        prompt: str,
        candidate: str | int | float,
        *,
        steps_per_chunk: int = 20,
        settle_tail: int = 6,
    ) -> dict[str, Any]:
        """Score a prompt/candidate pair by how stably the joint field settles."""

        snapshot = self._snapshot_field_state()
        try:
            self.field.reset_field()
            self.field.inject_text(f"{prompt} answer {candidate}", self.encoder)
            centers: list[np.ndarray] = []
            coherences: list[float] = []
            gradients: list[float] = []
            energies: list[float] = []
            prediction_errors: list[float] = []
            total_steps = max(1, int(steps_per_chunk))
            tail = max(1, min(int(settle_tail), total_steps))
            for index in range(total_steps):
                predicted = self.field.predict_phase()
                metrics = self.field.step()
                prediction_error = self.field.observe_prediction(predicted)
                if index >= total_steps - tail:
                    basin = self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim))
                    centers.append(np.asarray(basin.center, dtype=np.float32))
                    coherences.append(float(metrics.coherence))
                    gradients.append(float(metrics.gradient))
                    energies.append(float(metrics.energy))
                    prediction_errors.append(float(prediction_error))
            final_basin = self.apply_structured_feature_overlay(
                self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim)),
                prompt,
            )
            center_matrix = np.asarray(centers, dtype=np.float32)
            if center_matrix.shape[0] > 1:
                center_variance = float(np.mean(np.var(center_matrix, axis=0)))
                center_drift = float(
                    np.mean(np.linalg.norm(np.diff(center_matrix, axis=0), ord=2, axis=1))
                )
            else:
                center_variance = 0.0
                center_drift = 0.0
            mean_coherence = float(np.mean(coherences)) if coherences else 0.0
            mean_gradient = float(np.mean(gradients)) if gradients else float(final_basin.gradient)
            mean_energy = float(np.mean(energies)) if energies else float(final_basin.energy)
            mean_prediction_error = float(np.mean(prediction_errors)) if prediction_errors else 0.0
            stability_penalty = (
                12.0 * center_variance
                + 0.5 * center_drift
                + mean_gradient
                + 0.25 * mean_energy
                + mean_prediction_error
            )
            score = float((0.25 + mean_coherence) / (1.0 + stability_penalty))
            return {
                "score": score,
                "candidate": str(candidate),
                "center_variance": center_variance,
                "center_drift": center_drift,
                "coherence": mean_coherence,
                "gradient": mean_gradient,
                "energy": mean_energy,
                "prediction_error": mean_prediction_error,
                "basin": final_basin,
            }
        finally:
            self._restore_field_state(snapshot)

    def score_joint_candidates(
        self,
        prompt: str,
        candidate_keys: Sequence[str],
        *,
        steps_per_chunk: int = 20,
        settle_tail: int = 6,
    ) -> dict[str, Any]:
        """Rank operation/result keys by prompt+candidate joint stability."""

        rows = []
        for key in dict.fromkeys(str(item) for item in candidate_keys):
            if ":" not in key:
                continue
            candidate = key.split(":", 1)[1]
            metrics = self.score_joint_stability(
                prompt,
                candidate,
                steps_per_chunk=steps_per_chunk,
                settle_tail=settle_tail,
            )
            rows.append({
                "key": key,
                "candidate": candidate,
                **{name: value for name, value in metrics.items() if name != "basin"},
            })
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        best = rows[0] if rows else None
        second_score = float(rows[1]["score"]) if len(rows) > 1 else float("-inf")
        margin = (float(best["score"]) - second_score) if best is not None else 0.0
        return {
            "best_key": str(best["key"]) if best is not None else "",
            "best_score": float(best["score"]) if best is not None else 0.0,
            "second_score": second_score if np.isfinite(second_score) else 0.0,
            "margin": float(margin if np.isfinite(margin) else 0.0),
            "scores": rows,
        }

    def encode_basin(
        self,
        text_or_tokens: str | Sequence[str],
        *,
        steps_per_chunk: int = 20,
        anneal: bool = False,
        anneal_steps: int = 30,
        anneal_temperature: float = 0.012,
        reset: bool = True,
        attractor: np.ndarray | Sequence[float] | None = None,
        attractor_key: str | None = None,
        coupling: float = 0.0,
    ) -> tuple[BasinFeature, float]:
        """Inject a full prompt and return its stable basin feature."""

        tokens = self.tokenize(text_or_tokens)
        raw_text = text_or_tokens if isinstance(text_or_tokens, str) else " ".join(str(item) for item in text_or_tokens)
        if reset:
            self.field.reset_field()
        self.field.inject_text(" ".join(tokens), self.encoder)
        prediction_error = 0.0
        total_steps = max(1, int(steps_per_chunk))
        if anneal:
            total_steps = max(total_steps, int(anneal_steps))
        for index in range(total_steps):
            if anneal and anneal_temperature > 0.0:
                progress = index / max(1, total_steps - 1)
                self.field.inject_noise(scale=float(anneal_temperature) * (1.0 - progress))
            external_force = None
            if attractor is not None and float(coupling) > 0.0:
                current_basin = self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim))
                external_force = self.guided_external_force(
                    current_basin,
                    attractor,
                    correct_key=attractor_key,
                    coupling=coupling,
                )
            predicted = self.field.predict_phase(external_force=external_force)
            self.field.step(external_force=external_force)
            prediction_error = self.field.observe_prediction(predicted)
        basin = self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim))
        return self.apply_structured_feature_overlay(basin, raw_text), float(prediction_error)

    def guided_external_force(
        self,
        basin: BasinFeature,
        attractor: np.ndarray | Sequence[float],
        *,
        correct_key: str | None = None,
        coupling: float = 0.30,
        sigma: float = 3.0,
    ) -> np.ndarray:
        """Project feature-space teacher error into a 2D external force field."""

        active = np.asarray(basin.center, dtype=np.float32)
        target = np.asarray(attractor, dtype=np.float32)
        if active.shape != target.shape:
            return np.zeros(self.field.config.shape, dtype=np.float64)
        delta = target - active
        delta_l2 = float(np.linalg.norm(delta, ord=2))
        if delta_l2 <= 1e-9:
            return np.zeros(self.field.config.shape, dtype=np.float64)
        pressure = float(np.tanh(delta_l2))
        signed_force = float(np.tanh(np.mean(delta) * 8.0))
        if abs(signed_force) < 0.05:
            signed_force = 0.05 if float(np.mean(target)) >= float(np.mean(active)) else -0.05

        active_kernel = self.field.gaussian_kernel(x=basin.x, y=basin.y, sigma=sigma)
        force = 0.45 * active_kernel
        if correct_key:
            anchor_x, anchor_y = self.prototype_anchor_coordinate(correct_key)
            force = force + 0.55 * self.field.gaussian_kernel(
                x=anchor_x,
                y=anchor_y,
                sigma=sigma * 1.25,
            )
        return np.asarray(float(coupling) * pressure * signed_force * force, dtype=np.float64)

    def project_teacher_to_phase(
        self,
        teacher_vec: np.ndarray | Sequence[float],
        *,
        patch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a basin feature vector into local phase/amplitude geometry."""

        teacher = np.asarray(teacher_vec, dtype=np.float32).ravel()
        if teacher.size == 0:
            return np.zeros((1, 1), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)
        side = int(patch_size or max(3, min(15, round(np.sqrt(max(9, teacher.size // 2))))))
        side = max(3, side if side % 2 == 1 else side + 1)
        needed = side * side
        tiled = np.resize(teacher, needed).reshape(side, side).astype(np.float64)
        centered = tiled - float(np.mean(tiled))
        scale = float(np.std(centered))
        if scale <= 1e-8:
            scale = float(np.max(np.abs(centered))) or 1.0
        normalized = np.clip(centered / scale, -3.0, 3.0)
        phase_patch = np.arctan2(normalized, np.ones_like(normalized))
        amp_patch = np.abs(normalized)
        amp_max = float(np.max(amp_patch))
        if amp_max > 1e-8:
            amp_patch = amp_patch / amp_max
        yy, xx = np.indices((side, side))
        center = (side - 1) / 2.0
        envelope = np.exp(-(((xx - center) ** 2 + (yy - center) ** 2) / (2.0 * (side / 3.0) ** 2)))
        return phase_patch.astype(np.float64), (amp_patch * envelope).astype(np.float64)

    def project_delta_to_phase(
        self,
        teacher_vec: np.ndarray | Sequence[float],
        target_proto: np.ndarray | Sequence[float],
        *,
        patch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project the result-specific teacher-to-target delta into phase geometry."""

        teacher = np.asarray(teacher_vec, dtype=np.float32).ravel()
        target = np.asarray(target_proto, dtype=np.float32).ravel()
        if teacher.shape != target.shape:
            return np.zeros((1, 1), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)
        return self.project_teacher_to_phase(target - teacher, patch_size=patch_size)

    def _stamp_phase_patch(
        self,
        target: np.ndarray,
        *,
        x: int,
        y: int,
        patch_phase: np.ndarray,
        patch_amp: np.ndarray,
        strength: float,
        current_theta: np.ndarray | None = None,
    ) -> None:
        patch_phase = np.asarray(patch_phase, dtype=np.float64)
        patch_amp = np.asarray(patch_amp, dtype=np.float64)
        side_y, side_x = patch_phase.shape
        off_y = np.arange(-(side_y // 2), side_y // 2 + 1)
        off_x = np.arange(-(side_x // 2), side_x // 2 + 1)
        yy = (int(y) + off_y[:side_y]) % self.field.config.height
        xx = (int(x) + off_x[:side_x]) % self.field.config.width
        region = np.ix_(yy, xx)
        if current_theta is None:
            patch = patch_amp * np.sin(patch_phase)
        else:
            patch = patch_amp * np.sin(patch_phase - current_theta[region])
        target[region] += float(strength) * patch

    def phase_geometry_external_force(
        self,
        basin: BasinFeature,
        patch_phase: np.ndarray,
        patch_amp: np.ndarray,
        *,
        correct_key: str | None = None,
        coupling: float = 0.30,
    ) -> np.ndarray:
        """Turn a teacher phase patch into a full-grid external force."""

        force = np.zeros(self.field.config.shape, dtype=np.float64)
        self._stamp_phase_patch(
            force,
            x=basin.x,
            y=basin.y,
            patch_phase=patch_phase,
            patch_amp=patch_amp,
            strength=0.55 * float(coupling),
            current_theta=self.field.theta,
        )
        if correct_key:
            anchor_x, anchor_y = self.prototype_anchor_coordinate(correct_key)
            self._stamp_phase_patch(
                force,
                x=anchor_x,
                y=anchor_y,
                patch_phase=patch_phase,
                patch_amp=patch_amp,
                strength=0.45 * float(coupling),
                current_theta=self.field.theta,
            )
        return force

    def encode_basin_with_phase_geometry(
        self,
        text_or_tokens: str | Sequence[str],
        teacher: np.ndarray | Sequence[float],
        *,
        correct_key: str | None = None,
        steps_per_chunk: int = 20,
        coupling: float = 0.30,
        patch_size: int | None = None,
        reset: bool = True,
    ) -> tuple[BasinFeature, float, dict[str, float | int]]:
        """Inject text and steer evolution with a projected teacher phase patch."""

        tokens = self.tokenize(text_or_tokens)
        if reset:
            self.field.reset_field()
        self.field.inject_text(" ".join(tokens), self.encoder)
        patch_phase, patch_amp = self.project_teacher_to_phase(teacher, patch_size=patch_size)
        prediction_error = 0.0
        total_steps = max(1, int(steps_per_chunk))
        for _ in range(total_steps):
            current_basin = self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim))
            external_force = self.phase_geometry_external_force(
                current_basin,
                patch_phase,
                patch_amp,
                correct_key=correct_key,
                coupling=coupling,
            )
            predicted = self.field.predict_phase(external_force=external_force)
            self.field.step(external_force=external_force)
            prediction_error = self.field.observe_prediction(predicted)
        basin = self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim))
        info = {
            "patch_height": int(patch_phase.shape[0]),
            "patch_width": int(patch_phase.shape[1]),
            "patch_amp_mean": float(np.mean(patch_amp)),
            "patch_amp_max": float(np.max(patch_amp)),
        }
        return basin, float(prediction_error), info

    def encode_basin_with_delta_geometry(
        self,
        text_or_tokens: str | Sequence[str],
        teacher: np.ndarray | Sequence[float],
        target_proto: np.ndarray | Sequence[float],
        *,
        correct_key: str | None = None,
        steps_per_chunk: int = 20,
        coupling: float = 0.30,
        patch_size: int | None = None,
        reset: bool = True,
    ) -> tuple[BasinFeature, float, dict[str, float | int]]:
        """Inject text and steer evolution with a result-specific delta phase patch."""

        tokens = self.tokenize(text_or_tokens)
        if reset:
            self.field.reset_field()
        self.field.inject_text(" ".join(tokens), self.encoder)
        patch_phase, patch_amp = self.project_delta_to_phase(
            teacher,
            target_proto,
            patch_size=patch_size,
        )
        prediction_error = 0.0
        total_steps = max(1, int(steps_per_chunk))
        for _ in range(total_steps):
            current_basin = self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim))
            external_force = self.phase_geometry_external_force(
                current_basin,
                patch_phase,
                patch_amp,
                correct_key=correct_key,
                coupling=coupling,
            )
            predicted = self.field.predict_phase(external_force=external_force)
            self.field.step(external_force=external_force)
            prediction_error = self.field.observe_prediction(predicted)
        basin = self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim))
        delta = np.asarray(target_proto, dtype=np.float32).ravel() - np.asarray(teacher, dtype=np.float32).ravel()
        info = {
            "patch_height": int(patch_phase.shape[0]),
            "patch_width": int(patch_phase.shape[1]),
            "patch_amp_mean": float(np.mean(patch_amp)),
            "patch_amp_max": float(np.max(patch_amp)),
            "delta_l2": float(np.linalg.norm(delta, ord=2)) if delta.size else 0.0,
        }
        return basin, float(prediction_error), info

    def inject_teacher_phase_geometry(
        self,
        basin: BasinFeature,
        teacher: np.ndarray | Sequence[float],
        *,
        correct_key: str | None = None,
        strength: float = 0.05,
        patch_size: int | None = None,
    ) -> dict[str, float | int | str | None]:
        """Persist a teacher phase patch into omega/landscape around active and result anchors."""

        patch_phase, patch_amp = self.project_teacher_to_phase(teacher, patch_size=patch_size)
        self._stamp_phase_patch(
            self.field.landscape,
            x=basin.x,
            y=basin.y,
            patch_phase=patch_phase,
            patch_amp=patch_amp,
            strength=0.70 * float(strength),
            current_theta=None,
        )
        self._stamp_phase_patch(
            self.field.omega,
            x=basin.x,
            y=basin.y,
            patch_phase=patch_phase,
            patch_amp=patch_amp,
            strength=0.18 * float(strength),
            current_theta=None,
        )
        anchor_x: int | None = None
        anchor_y: int | None = None
        if correct_key:
            anchor_x, anchor_y = self.prototype_anchor_coordinate(correct_key)
            self._stamp_phase_patch(
                self.field.landscape,
                x=anchor_x,
                y=anchor_y,
                patch_phase=patch_phase,
                patch_amp=patch_amp,
                strength=0.50 * float(strength),
                current_theta=None,
            )
            self._stamp_phase_patch(
                self.field.omega,
                x=anchor_x,
                y=anchor_y,
                patch_phase=patch_phase,
                patch_amp=patch_amp,
                strength=0.14 * float(strength),
                current_theta=None,
            )
        self.field.landscape = smooth(
            self.field.landscape,
            amount=0.08,
            backend=self.field.config.laplacian_backend,
        )
        self.field.omega = smooth(
            self.field.omega,
            amount=0.04,
            backend=self.field.config.laplacian_backend,
        )
        self.field.landscape = np.clip(self.field.landscape, -4.0, 4.0)
        self.field.omega = np.clip(self.field.omega, -1.0, 1.0)
        return {
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "patch_height": int(patch_phase.shape[0]),
            "patch_width": int(patch_phase.shape[1]),
            "patch_amp_mean": float(np.mean(patch_amp)),
            "patch_amp_max": float(np.max(patch_amp)),
        }

    def inject_delta_phase_geometry(
        self,
        basin: BasinFeature,
        teacher: np.ndarray | Sequence[float],
        target_proto: np.ndarray | Sequence[float],
        *,
        correct_key: str | None = None,
        strength: float = 0.05,
        patch_size: int | None = None,
    ) -> dict[str, float | int | str | None]:
        """Persist a result-specific delta phase patch into omega/landscape."""

        patch_phase, patch_amp = self.project_delta_to_phase(
            teacher,
            target_proto,
            patch_size=patch_size,
        )
        self._stamp_phase_patch(
            self.field.landscape,
            x=basin.x,
            y=basin.y,
            patch_phase=patch_phase,
            patch_amp=patch_amp,
            strength=0.75 * float(strength),
            current_theta=None,
        )
        self._stamp_phase_patch(
            self.field.omega,
            x=basin.x,
            y=basin.y,
            patch_phase=patch_phase,
            patch_amp=patch_amp,
            strength=0.20 * float(strength),
            current_theta=None,
        )
        anchor_x: int | None = None
        anchor_y: int | None = None
        if correct_key:
            anchor_x, anchor_y = self.prototype_anchor_coordinate(correct_key)
            self._stamp_phase_patch(
                self.field.landscape,
                x=anchor_x,
                y=anchor_y,
                patch_phase=patch_phase,
                patch_amp=patch_amp,
                strength=0.65 * float(strength),
                current_theta=None,
            )
            self._stamp_phase_patch(
                self.field.omega,
                x=anchor_x,
                y=anchor_y,
                patch_phase=patch_phase,
                patch_amp=patch_amp,
                strength=0.18 * float(strength),
                current_theta=None,
            )
        self.field.landscape = smooth(
            self.field.landscape,
            amount=0.08,
            backend=self.field.config.laplacian_backend,
        )
        self.field.omega = smooth(
            self.field.omega,
            amount=0.04,
            backend=self.field.config.laplacian_backend,
        )
        self.field.landscape = np.clip(self.field.landscape, -4.0, 4.0)
        self.field.omega = np.clip(self.field.omega, -1.0, 1.0)
        delta = np.asarray(target_proto, dtype=np.float32).ravel() - np.asarray(teacher, dtype=np.float32).ravel()
        return {
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "patch_height": int(patch_phase.shape[0]),
            "patch_width": int(patch_phase.shape[1]),
            "patch_amp_mean": float(np.mean(patch_amp)),
            "patch_amp_max": float(np.max(patch_amp)),
            "delta_l2": float(np.linalg.norm(delta, ord=2)) if delta.size else 0.0,
        }

    def reinforce_equivalence(
        self,
        basin_a: BasinFeature,
        basin_b: BasinFeature,
        *,
        gain: float = 0.025,
        sigma: float = 3.5,
    ) -> float:
        """Bridge equivalent attractors in the topology with a shared basin update."""

        alignment = cosine_similarity(basin_a.center, basin_b.center)
        width = self.field.config.width
        height = self.field.config.height
        mid_x = circular_midpoint(basin_a.x, basin_b.x, width)
        mid_y = circular_midpoint(basin_a.y, basin_b.y, height)
        kernel_a = self.field.gaussian_kernel(x=basin_a.x, y=basin_a.y, sigma=sigma)
        kernel_b = self.field.gaussian_kernel(x=basin_b.x, y=basin_b.y, sigma=sigma)
        kernel_mid = self.field.gaussian_kernel(x=mid_x, y=mid_y, sigma=sigma * 1.2)
        update = (0.35 * kernel_a) + (0.35 * kernel_b) + (0.30 * kernel_mid)
        strength = gain * (1.0 + max(0.0, 1.0 - alignment))
        decay = self.field.config.memory_decay
        self.field.landscape = (1.0 - decay) * self.field.landscape + strength * update
        self.field.omega = (1.0 - decay) * self.field.omega + (strength * 0.18) * update
        self.field.landscape = smooth(
            self.field.landscape,
            amount=0.08,
            backend=self.field.config.laplacian_backend,
        )
        self.field.omega = smooth(
            self.field.omega,
            amount=0.04,
            backend=self.field.config.laplacian_backend,
        )
        self.field.landscape = np.clip(self.field.landscape, -4.0, 4.0)
        self.field.omega = np.clip(self.field.omega, -1.0, 1.0)
        return alignment

    def prototype_anchor_coordinate(self, prototype_key: str) -> tuple[int, int]:
        """Map a structural prototype key to a stable mesh coordinate."""

        digest = hashlib.sha256(str(prototype_key).encode("utf-8")).digest()
        x = int.from_bytes(digest[:4], "big") % max(1, self.field.config.width)
        y = int.from_bytes(digest[4:8], "big") % max(1, self.field.config.height)
        return x, y

    def operation_anchor_coordinate(self, op_type: str) -> tuple[int, int]:
        """Map an operation family to a stable mesh coordinate."""

        return self.prototype_anchor_coordinate(f"operation:{normalize_token(op_type)}")

    def operation_prototype_state(self, op_type: str) -> np.ndarray | None:
        """Return the mean prototype feature for an operation family."""

        prefix = f"{normalize_token(op_type)}:"
        vectors = [
            np.asarray(value, dtype=np.float32)
            for key, value in self.structural_prototypes.items()
            if str(key).startswith(prefix)
        ]
        if not vectors:
            return None
        return np.mean(vectors, axis=0).astype(np.float32)

    def carve_computation_manifold(
        self,
        *,
        correct_key: str,
        op_state: np.ndarray | Sequence[float] | None = None,
        result_state: np.ndarray | Sequence[float] | None = None,
        wrong_keys: Sequence[str] = (),
        strength: float = 0.05,
        sigma: float = 2.2,
    ) -> dict[str, float | int | str | bool]:
        """Pre-condition the 2D landscape with an operation-to-result road.

        This deliberately avoids the active basin. The road is carved between a
        stable operation anchor and the deterministic result anchor, while the
        high-dimensional prototype delta only controls pressure/sign.
        """

        key = str(correct_key)
        if ":" not in key:
            return {
                "used": False,
                "correct_key": key,
                "reason": "malformed_correct_key",
                "path_steps": 0,
            }
        op_type, _target = key.split(":", 1)
        if key not in self.structural_prototypes and result_state is None:
            return {
                "used": False,
                "correct_key": key,
                "reason": "missing_correct_prototype",
                "path_steps": 0,
            }

        op_vector = (
            np.asarray(op_state, dtype=np.float32)
            if op_state is not None
            else self.operation_prototype_state(op_type)
        )
        target_vector = (
            np.asarray(result_state, dtype=np.float32)
            if result_state is not None
            else np.asarray(self.structural_prototypes[key], dtype=np.float32)
        )
        if op_vector is None:
            return {
                "used": False,
                "correct_key": key,
                "reason": "missing_operation_prototype",
                "path_steps": 0,
            }
        if op_vector.shape != target_vector.shape:
            return {
                "used": False,
                "correct_key": key,
                "reason": "state_shape_mismatch",
                "path_steps": 0,
            }

        op_x, op_y = self.operation_anchor_coordinate(op_type)
        target_x, target_y = self.prototype_anchor_coordinate(key)
        width = max(1, self.field.config.width)
        height = max(1, self.field.config.height)

        def signed_toroidal_delta(start: int, end: int, size: int) -> float:
            raw = (int(end) - int(start)) % int(size)
            if raw > size / 2.0:
                raw -= size
            return float(raw)

        dx = signed_toroidal_delta(op_x, target_x, width)
        dy = signed_toroidal_delta(op_y, target_y, height)
        distance = float(np.sqrt(dx * dx + dy * dy))
        path_steps = max(3, min(max(width, height), int(np.ceil(distance * 1.5)) + 1))

        road = np.zeros(self.field.config.shape, dtype=np.float64)
        for t in np.linspace(0.0, 1.0, path_steps):
            x = int(round((op_x + dx * float(t)) % width))
            y = int(round((op_y + dy * float(t)) % height))
            road = np.maximum(road, self.field.gaussian_kernel(x=x, y=y, sigma=sigma))
        op_kernel = self.field.gaussian_kernel(x=op_x, y=op_y, sigma=sigma * 1.15)
        target_kernel = self.field.gaussian_kernel(x=target_x, y=target_y, sigma=sigma * 1.35)
        wrong_kernel = np.zeros_like(road)
        for wrong_key in wrong_keys:
            wrong = str(wrong_key)
            if wrong == key or wrong not in self.structural_prototypes:
                continue
            wrong_x, wrong_y = self.prototype_anchor_coordinate(wrong)
            wrong_kernel = np.maximum(
                wrong_kernel,
                self.field.gaussian_kernel(x=wrong_x, y=wrong_y, sigma=sigma),
            )

        delta = target_vector - op_vector
        delta_l2 = float(np.linalg.norm(delta, ord=2))
        pressure = float(np.tanh(delta_l2))
        signed_force = 1.0 if float(np.mean(delta)) >= 0.0 else -1.0
        update = (0.60 * road) + (0.30 * target_kernel) + (0.10 * op_kernel) - (0.20 * wrong_kernel)
        gain = float(strength) * max(0.05, pressure)
        decay = self.field.config.memory_decay
        self.field.landscape = (1.0 - decay) * self.field.landscape + gain * update
        self.field.omega = (1.0 - decay) * self.field.omega + (gain * 0.04 * signed_force) * update
        self.field.predictor_trace = (
            (1.0 - self.field.config.prediction_trace_decay) * self.field.predictor_trace
            + (gain * 0.025 * signed_force) * road
        )
        self.field.landscape = smooth(
            self.field.landscape,
            amount=0.05,
            backend=self.field.config.laplacian_backend,
        )
        self.field.omega = smooth(
            self.field.omega,
            amount=0.025,
            backend=self.field.config.laplacian_backend,
        )
        self.field.predictor_trace = smooth(
            self.field.predictor_trace,
            amount=0.025,
            backend=self.field.config.laplacian_backend,
        )
        self.field.landscape = np.clip(self.field.landscape, -4.0, 4.0)
        self.field.omega = np.clip(self.field.omega, -1.0, 1.0)
        self.field.predictor_trace = np.clip(self.field.predictor_trace, -1.0, 1.0)
        return {
            "used": True,
            "correct_key": key,
            "op_type": op_type,
            "op_x": int(op_x),
            "op_y": int(op_y),
            "target_x": int(target_x),
            "target_y": int(target_y),
            "path_steps": int(path_steps),
            "delta_l2": delta_l2,
            "pressure": pressure,
            "gain": gain,
            "wrong_count": int(len(wrong_keys)),
            "global_landscape_updated": True,
            "prototype_updated": False,
        }

    def reinforce_structural_anchor(
        self,
        basin_a: BasinFeature,
        basin_b: BasinFeature,
        *,
        prototype_key: str,
        gain: float = 0.05,
        sigma: float = 3.0,
        prototype_alpha: float = 0.10,
    ) -> dict[str, float | int | str]:
        """Hard-collapse equivalent basin features toward a persistent prototype.

        The decoder cannot train the substrate directly, so this keeps an EMA
        feature prototype and carves a deterministic attractor into the field's
        landscape/omega for that operation-result key.
        """

        key = str(prototype_key)
        center_a = np.asarray(basin_a.center, dtype=np.float32)
        center_b = np.asarray(basin_b.center, dtype=np.float32)
        batch_mean = ((center_a + center_b) * 0.5).astype(np.float32)
        if key not in self.structural_prototypes:
            self.structural_prototypes[key] = batch_mean.copy()
        else:
            alpha = float(np.clip(prototype_alpha, 0.0, 1.0))
            self.structural_prototypes[key] = (
                alpha * batch_mean + (1.0 - alpha) * self.structural_prototypes[key]
            ).astype(np.float32)

        prototype = self.structural_prototypes[key]
        diff = center_a - center_b
        feature_mse = float(np.mean(diff * diff))
        feature_l2 = float(np.linalg.norm(diff, ord=2))
        prototype_l2 = float(
            0.5
            * (
                np.linalg.norm(center_a - prototype, ord=2)
                + np.linalg.norm(center_b - prototype, ord=2)
            )
        )
        alignment = cosine_similarity(center_a, center_b)
        anchor_x, anchor_y = self.prototype_anchor_coordinate(key)
        anchor_kernel = self.field.gaussian_kernel(x=anchor_x, y=anchor_y, sigma=sigma * 1.35)
        kernel_a = self.field.gaussian_kernel(x=basin_a.x, y=basin_a.y, sigma=sigma)
        kernel_b = self.field.gaussian_kernel(x=basin_b.x, y=basin_b.y, sigma=sigma)
        old_kernel = np.maximum(kernel_a, kernel_b)
        collapse_pressure = min(4.0, max(0.0, feature_l2))
        strength = float(gain) * (1.0 + 0.35 * collapse_pressure)

        # Make the canonical anchor the easiest attractor, while still leaving a
        # weak bridge at the observed basins so equivalent forms can migrate.
        update = (1.35 * anchor_kernel) + (0.08 * old_kernel)
        if feature_l2 > 0.2:
            update -= 0.04 * old_kernel
        decay = self.field.config.memory_decay
        self.field.landscape = (1.0 - decay) * self.field.landscape + strength * update
        self.field.omega = (1.0 - decay) * self.field.omega + (strength * 0.20) * update
        self.field.landscape = smooth(
            self.field.landscape,
            amount=0.10,
            backend=self.field.config.laplacian_backend,
        )
        self.field.omega = smooth(
            self.field.omega,
            amount=0.05,
            backend=self.field.config.laplacian_backend,
        )
        self.field.landscape = np.clip(self.field.landscape, -4.0, 4.0)
        self.field.omega = np.clip(self.field.omega, -1.0, 1.0)
        return {
            "prototype_key": key,
            "feature_mse": feature_mse,
            "feature_l2": feature_l2,
            "prototype_l2": prototype_l2,
            "alignment": alignment,
            "anchor_x": int(anchor_x),
            "anchor_y": int(anchor_y),
            "strength": strength,
        }

    def nearest_structural_prototype(
        self,
        basin_center: np.ndarray | Sequence[float],
        *,
        operation: str | None = None,
        k: int = 1,
        basin_cell: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest saved structural prototypes by Euclidean distance.

        When resonance slots are enabled, the active prompt chooses one shared
        slot readout. All candidates are scored against that same prompt-gated
        center, so wrong candidates cannot self-match through their own slots.
        """

        if not self.structural_prototypes:
            return []
        center = np.asarray(basin_center, dtype=np.float32)
        candidates: list[tuple[str, np.ndarray]] = []
        operation_prefix = f"{operation}:" if operation else None
        for key, prototype in self.structural_prototypes.items():
            if operation_prefix is not None and not str(key).startswith(operation_prefix):
                continue
            candidates.append((str(key), np.asarray(prototype, dtype=np.float32)))
        if not candidates:
            return []
        keys = [key for key, _prototype in candidates]
        prototypes = np.asarray([prototype for _key, prototype in candidates], dtype=np.float32)
        distances = np.linalg.norm(prototypes - center.reshape(1, -1), ord=2, axis=1)
        scores: np.ndarray | None = None
        if self.delta_scorer is not None and self.delta_trained_steps > 0:
            scores = self.score_delta_candidates(center, prototypes)
            order = np.argsort(-scores)[: max(1, int(k))]
        elif self.num_slots > 0:
            readout = self.read_resonant_slots(center, basin_cell=basin_cell)
            readout_center = np.asarray(readout["center"], dtype=np.float32)
            distances = np.linalg.norm(prototypes - readout_center.reshape(1, -1), ord=2, axis=1)
            order = np.argsort(distances)[: max(1, int(k))]
        else:
            order = np.argsort(distances)[: max(1, int(k))]
        return [
            {
                "key": keys[int(index)],
                "target": prototype_target_from_key(keys[int(index)]),
                "distance": float(distances[int(index)]),
                "score": float(scores[int(index)]) if scores is not None else None,
                "ranker": "delta-scorer" if scores is not None else "distance",
                "rank": int(rank),
            }
            for rank, index in enumerate(order, start=1)
        ]

    def reinforce_result_target(
        self,
        basin: BasinFeature,
        *,
        correct_key: str,
        wrong_keys: Sequence[str] = (),
        attract_gain: float = 0.10,
        repulsion_strength: float = 0.40,
        topology_gain: float = 0.025,
        sigma: float = 3.0,
        margin: float = 0.20,
    ) -> dict[str, Any]:
        """Pull the correct result prototype toward an active basin and repel nearby wrong prototypes."""

        key = str(correct_key)
        if key not in self.structural_prototypes:
            return {
                "used": False,
                "correct_key": key,
                "reason": "missing_correct_prototype",
            }
        active = np.asarray(basin.center, dtype=np.float32)
        operation = key.split(":", 1)[0] if ":" in key else None
        nearest_before = self.nearest_structural_prototype(active, operation=operation, k=1)
        correct_before = np.asarray(self.structural_prototypes[key], dtype=np.float32)
        correct_distance_before = float(np.linalg.norm(active - correct_before, ord=2))

        attract = float(np.clip(attract_gain, 0.0, 1.0))
        self.structural_prototypes[key] = (
            correct_before + attract * (active - correct_before)
        ).astype(np.float32)

        wrong_updates = 0
        wrong_distances_before: list[float] = []
        for wrong_key in wrong_keys:
            wrong = str(wrong_key)
            if wrong == key or wrong not in self.structural_prototypes:
                continue
            wrong_vector = np.asarray(self.structural_prototypes[wrong], dtype=np.float32)
            delta = wrong_vector - active
            distance = float(np.linalg.norm(delta, ord=2))
            wrong_distances_before.append(distance)
            pressure = max(0.0, float(margin) - distance)
            if pressure <= 0.0:
                continue
            direction = delta / max(distance, 1e-6)
            self.structural_prototypes[wrong] = (
                wrong_vector + (float(repulsion_strength) * attract * pressure) * direction
            ).astype(np.float32)
            wrong_updates += 1

        correct_after = np.asarray(self.structural_prototypes[key], dtype=np.float32)
        correct_distance_after = float(np.linalg.norm(active - correct_after, ord=2))

        correct_x, correct_y = self.prototype_anchor_coordinate(key)
        active_kernel = self.field.gaussian_kernel(x=basin.x, y=basin.y, sigma=sigma)
        correct_kernel = self.field.gaussian_kernel(x=correct_x, y=correct_y, sigma=sigma * 1.25)
        wrong_kernel = np.zeros_like(active_kernel)
        for wrong_key in wrong_keys:
            wrong = str(wrong_key)
            if wrong == key or wrong not in self.structural_prototypes:
                continue
            wrong_x, wrong_y = self.prototype_anchor_coordinate(wrong)
            wrong_kernel = np.maximum(
                wrong_kernel,
                self.field.gaussian_kernel(x=wrong_x, y=wrong_y, sigma=sigma),
            )

        topo = float(topology_gain)
        repel = float(repulsion_strength)
        update = (0.75 * correct_kernel) + (0.30 * active_kernel) - (0.35 * repel * wrong_kernel)
        decay = self.field.config.memory_decay
        self.field.landscape = (1.0 - decay) * self.field.landscape + topo * update
        self.field.omega = (1.0 - decay) * self.field.omega + (topo * 0.20) * update
        self.field.landscape = smooth(
            self.field.landscape,
            amount=0.08,
            backend=self.field.config.laplacian_backend,
        )
        self.field.omega = smooth(
            self.field.omega,
            amount=0.04,
            backend=self.field.config.laplacian_backend,
        )
        self.field.landscape = np.clip(self.field.landscape, -4.0, 4.0)
        self.field.omega = np.clip(self.field.omega, -1.0, 1.0)

        nearest_after = self.nearest_structural_prototype(active, operation=operation, k=1)
        return {
            "used": True,
            "correct_key": key,
            "nearest_before": nearest_before[0]["key"] if nearest_before else None,
            "nearest_after": nearest_after[0]["key"] if nearest_after else None,
            "target_match_before": bool(nearest_before and nearest_before[0]["key"] == key),
            "target_match_after": bool(nearest_after and nearest_after[0]["key"] == key),
            "correct_distance_before": correct_distance_before,
            "correct_distance_after": correct_distance_after,
            "wrong_distance_min_before": min(wrong_distances_before) if wrong_distances_before else None,
            "wrong_updates": wrong_updates,
        }

    def update_landscape_toward(
        self,
        basin: BasinFeature,
        target_state: np.ndarray | Sequence[float],
        *,
        strength: float = 0.05,
        correct_key: str | None = None,
        update_prototype: bool = True,
        sigma: float = 3.0,
    ) -> dict[str, float | int | str | None]:
        """Carve a durable local valley from an active basin toward a teacher state.

        `PhaseFieldMesh.omega` is a 2D potential field, while basin states are
        feature vectors. The update therefore projects the feature-space delta
        into a scalar pressure and applies it around the active basin plus the
        deterministic result anchor when one is available.
        """

        active = np.asarray(basin.center, dtype=np.float32)
        target = np.asarray(target_state, dtype=np.float32)
        if active.shape != target.shape:
            return {
                "used": False,
                "reason": "target_shape_mismatch",
                "delta_l2": float("nan"),
                "delta_mean": float("nan"),
            }
        delta = target - active
        delta_l2 = float(np.linalg.norm(delta, ord=2))
        delta_mean = float(np.mean(delta))
        pressure = float(np.tanh(delta_l2))
        active_kernel = self.field.gaussian_kernel(x=basin.x, y=basin.y, sigma=sigma)
        update = 0.70 * active_kernel
        anchor_x: int | None = None
        anchor_y: int | None = None
        if correct_key:
            anchor_x, anchor_y = self.prototype_anchor_coordinate(correct_key)
            update = update + 0.30 * self.field.gaussian_kernel(
                x=anchor_x,
                y=anchor_y,
                sigma=sigma * 1.25,
            )

        gain = float(strength) * max(0.05, pressure)
        signed_force = 1.0 if delta_mean >= 0.0 else -1.0
        decay = self.field.config.memory_decay
        self.field.landscape = (1.0 - decay) * self.field.landscape + gain * update
        self.field.omega = (1.0 - decay) * self.field.omega + (gain * 0.20 * signed_force) * update
        self.field.predictor_trace = (
            (1.0 - self.field.config.prediction_trace_decay) * self.field.predictor_trace
            + (gain * 0.08 * signed_force) * active_kernel
        )
        self.field.landscape = smooth(
            self.field.landscape,
            amount=0.08,
            backend=self.field.config.laplacian_backend,
        )
        self.field.omega = smooth(
            self.field.omega,
            amount=0.04,
            backend=self.field.config.laplacian_backend,
        )
        self.field.predictor_trace = smooth(
            self.field.predictor_trace,
            amount=0.04,
            backend=self.field.config.laplacian_backend,
        )
        self.field.landscape = np.clip(self.field.landscape, -4.0, 4.0)
        self.field.omega = np.clip(self.field.omega, -1.0, 1.0)
        self.field.predictor_trace = np.clip(self.field.predictor_trace, -1.0, 1.0)

        if update_prototype and correct_key and correct_key in self.structural_prototypes:
            prototype = np.asarray(self.structural_prototypes[correct_key], dtype=np.float32)
            proto_gain = float(np.clip(strength, 0.0, 1.0))
            self.structural_prototypes[correct_key] = (
                (1.0 - proto_gain) * prototype + proto_gain * active
            ).astype(np.float32)

        return {
            "used": True,
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "delta_l2": delta_l2,
            "delta_mean": delta_mean,
            "pressure": pressure,
            "gain": gain,
            "prototype_updated": bool(update_prototype and correct_key and correct_key in self.structural_prototypes),
        }

    def carve_residual_tunnel(
        self,
        basin: BasinFeature,
        target_state: np.ndarray | Sequence[float],
        *,
        correct_key: str,
        strength: float = 0.05,
        sigma: float = 2.0,
    ) -> dict[str, float | int | str | bool]:
        """Carve a low-energy 2D path from an active basin to a frozen target anchor.

        The field's persistent landscape is 2D, while result prototypes are
        feature vectors. This keeps the target vector immutable and uses the
        full feature residual only to set tunnel pressure and sign.
        """

        key = str(correct_key)
        if key not in self.structural_prototypes:
            return {
                "used": False,
                "correct_key": key,
                "reason": "missing_correct_prototype",
                "residual_l2": float("nan"),
                "path_steps": 0,
            }
        active = np.asarray(basin.center, dtype=np.float32)
        target = np.asarray(target_state, dtype=np.float32)
        if active.shape != target.shape:
            return {
                "used": False,
                "correct_key": key,
                "reason": "target_shape_mismatch",
                "residual_l2": float("nan"),
                "path_steps": 0,
            }

        residual = target - active
        residual_l2 = float(np.linalg.norm(residual, ord=2))
        residual_mse = float(np.mean(residual * residual))
        pressure = float(np.tanh(residual_l2))
        signed_force = 1.0 if float(np.mean(residual)) >= 0.0 else -1.0
        anchor_x, anchor_y = self.prototype_anchor_coordinate(key)
        width = max(1, self.field.config.width)
        height = max(1, self.field.config.height)

        def signed_toroidal_delta(start: int, end: int, size: int) -> float:
            raw = (int(end) - int(start)) % int(size)
            if raw > size / 2.0:
                raw -= size
            return float(raw)

        dx = signed_toroidal_delta(basin.x, anchor_x, width)
        dy = signed_toroidal_delta(basin.y, anchor_y, height)
        distance = float(np.sqrt(dx * dx + dy * dy))
        path_steps = max(3, min(max(width, height), int(np.ceil(distance * 2.0)) + 1))
        tunnel = np.zeros(self.field.config.shape, dtype=np.float64)
        path_points: list[tuple[int, int, float]] = []
        for t in np.linspace(0.0, 1.0, path_steps):
            x = int(round((basin.x + dx * float(t)) % width))
            y = int(round((basin.y + dy * float(t)) % height))
            weight = 0.5 + 0.5 * float(t)
            path_points.append((x, y, weight))
            tunnel = np.maximum(tunnel, self.field.gaussian_kernel(x=x, y=y, sigma=sigma))
        target_kernel = self.field.gaussian_kernel(x=anchor_x, y=anchor_y, sigma=sigma * 1.4)
        active_kernel = self.field.gaussian_kernel(x=basin.x, y=basin.y, sigma=sigma * 0.9)
        update = (0.70 * tunnel) + (0.25 * target_kernel) + (0.05 * active_kernel)
        gain = float(strength) * max(0.05, pressure)
        self._ensure_feature_omega()
        feature_gain = float(strength) * max(0.05, pressure)
        for x, y, weight in path_points:
            self.feature_omega[y % height, x % width, :] += (feature_gain * weight) * residual
        self.feature_omega[anchor_y % height, anchor_x % width, :] += (feature_gain * 0.5) * target
        self.feature_omega = np.clip(self.feature_omega, -2.0, 2.0)
        decay = self.field.config.memory_decay
        self.field.landscape = (1.0 - decay) * self.field.landscape + gain * update
        self.field.omega = (1.0 - decay) * self.field.omega + (gain * 0.20 * signed_force) * update
        self.field.predictor_trace = (
            (1.0 - self.field.config.prediction_trace_decay) * self.field.predictor_trace
            + (gain * 0.10 * signed_force) * tunnel
        )
        self.field.landscape = smooth(
            self.field.landscape,
            amount=0.06,
            backend=self.field.config.laplacian_backend,
        )
        self.field.omega = smooth(
            self.field.omega,
            amount=0.03,
            backend=self.field.config.laplacian_backend,
        )
        self.field.predictor_trace = smooth(
            self.field.predictor_trace,
            amount=0.03,
            backend=self.field.config.laplacian_backend,
        )
        self.field.landscape = np.clip(self.field.landscape, -4.0, 4.0)
        self.field.omega = np.clip(self.field.omega, -1.0, 1.0)
        self.field.predictor_trace = np.clip(self.field.predictor_trace, -1.0, 1.0)
        return {
            "used": True,
            "correct_key": key,
            "anchor_x": int(anchor_x),
            "anchor_y": int(anchor_y),
            "path_steps": int(path_steps),
            "residual_l2": residual_l2,
            "residual_mse": residual_mse,
            "pressure": pressure,
            "gain": gain,
            "feature_gain": feature_gain,
            "feature_omega_l2": float(np.linalg.norm(self.feature_omega[anchor_y % height, anchor_x % width], ord=2)),
            "prototype_updated": False,
        }

    def carve_push_pull(
        self,
        basin: BasinFeature,
        target_state: np.ndarray | Sequence[float],
        wrong_targets: Sequence[np.ndarray | Sequence[float]] = (),
        *,
        correct_key: str,
        wrong_keys: Sequence[str] = (),
        strength: float = 0.05,
        wrong_strength: float = 0.5,
        sigma: float = 1.6,
    ) -> dict[str, float | int | str | bool]:
        """Pull a local feature attractor toward target and away from wrong results.

        ``omega`` is the 2D wave landscape, so the high-dimensional result
        residual lives in ``feature_omega`` at mesh cells. This keeps structural
        prototypes frozen while making the active cell read out as a cleaner
        target-localized basin on future encodes.
        """

        key = str(correct_key)
        if key not in self.structural_prototypes:
            return {
                "used": False,
                "correct_key": key,
                "reason": "missing_correct_prototype",
                "pull_l2": float("nan"),
                "push_l2": float("nan"),
                "update_l2": float("nan"),
                "wrong_count": 0,
            }
        active = np.asarray(basin.center, dtype=np.float32)
        target = np.asarray(target_state, dtype=np.float32)
        if active.shape != target.shape:
            return {
                "used": False,
                "correct_key": key,
                "reason": "target_shape_mismatch",
                "pull_l2": float("nan"),
                "push_l2": float("nan"),
                "update_l2": float("nan"),
                "wrong_count": 0,
            }

        wrong_vectors: list[np.ndarray] = []
        for wrong in wrong_targets:
            wrong_vector = np.asarray(wrong, dtype=np.float32)
            if wrong_vector.shape == active.shape:
                wrong_vectors.append(wrong_vector)

        pull = target - active
        if wrong_vectors:
            push = np.mean([active - wrong for wrong in wrong_vectors], axis=0).astype(np.float32)
        else:
            push = np.zeros_like(pull, dtype=np.float32)
        push_scale = float(wrong_strength)
        update_vec = (pull + push_scale * push).astype(np.float32)
        pull_l2 = float(np.linalg.norm(pull, ord=2))
        push_l2 = float(np.linalg.norm(push, ord=2))
        update_l2 = float(np.linalg.norm(update_vec, ord=2))
        pressure = max(0.05, float(np.tanh(update_l2)))
        gain = float(strength) * pressure

        self._ensure_feature_omega()
        height, width = self.field.config.shape
        active_kernel = self.field.gaussian_kernel(x=basin.x, y=basin.y, sigma=sigma)
        target_x, target_y = self.prototype_anchor_coordinate(key)
        target_kernel = self.field.gaussian_kernel(x=target_x, y=target_y, sigma=sigma * 1.25)
        local_update = (0.80 * active_kernel) + (0.20 * target_kernel)
        self.feature_omega += (gain * local_update[..., np.newaxis] * update_vec.reshape(1, 1, -1)).astype(
            np.float32
        )
        self.feature_omega[target_y % height, target_x % width, :] += (
            gain * 0.20 * (target - self.feature_omega[target_y % height, target_x % width, :])
        ).astype(np.float32)
        self.feature_omega = np.clip(self.feature_omega, -2.5, 2.5)

        wrong_kernel = np.zeros(self.field.config.shape, dtype=np.float64)
        for wrong_key in wrong_keys:
            wrong = str(wrong_key)
            if wrong == key or wrong not in self.structural_prototypes:
                continue
            wrong_x, wrong_y = self.prototype_anchor_coordinate(wrong)
            wrong_kernel = np.maximum(
                wrong_kernel,
                self.field.gaussian_kernel(x=wrong_x, y=wrong_y, sigma=sigma),
            )
        landscape_update = (0.70 * target_kernel) + (0.30 * active_kernel) - (
            0.40 * push_scale * wrong_kernel
        )
        decay = self.field.config.memory_decay
        self.field.landscape = (1.0 - decay) * self.field.landscape + (gain * 0.35) * landscape_update
        self.field.omega = (1.0 - decay) * self.field.omega + (gain * 0.08) * landscape_update
        self.field.predictor_trace = (
            (1.0 - self.field.config.prediction_trace_decay) * self.field.predictor_trace
            + (gain * 0.05) * landscape_update
        )
        self.field.landscape = smooth(
            self.field.landscape,
            amount=0.04,
            backend=self.field.config.laplacian_backend,
        )
        self.field.omega = smooth(
            self.field.omega,
            amount=0.02,
            backend=self.field.config.laplacian_backend,
        )
        self.field.predictor_trace = smooth(
            self.field.predictor_trace,
            amount=0.02,
            backend=self.field.config.laplacian_backend,
        )
        self.field.landscape = np.clip(self.field.landscape, -4.0, 4.0)
        self.field.omega = np.clip(self.field.omega, -1.0, 1.0)
        self.field.predictor_trace = np.clip(self.field.predictor_trace, -1.0, 1.0)
        return {
            "used": True,
            "correct_key": key,
            "anchor_x": int(target_x),
            "anchor_y": int(target_y),
            "pull_l2": pull_l2,
            "push_l2": push_l2,
            "update_l2": update_l2,
            "pressure": pressure,
            "gain": gain,
            "wrong_count": int(len(wrong_vectors)),
            "feature_omega_l2": float(
                np.linalg.norm(self.feature_omega[target_y % height, target_x % width], ord=2)
            ),
            "prototype_updated": False,
        }

    def carve_sparse_tunnel(
        self,
        basin: BasinFeature,
        target_state: np.ndarray | Sequence[float],
        wrong_targets: Sequence[np.ndarray | Sequence[float]] = (),
        *,
        correct_key: str,
        strength: float = 0.05,
        wrong_strength: float = 0.5,
    ) -> dict[str, float | int | str | bool]:
        """Update only the active basin cell's feature residual.

        This preserves the global 2D phase landscape and prototype ordering:
        no ``omega``/``landscape`` updates, no smoothing, and no writes to
        neighboring cells or prototype anchor cells.
        """

        key = str(correct_key)
        if key not in self.structural_prototypes:
            return {
                "used": False,
                "correct_key": key,
                "reason": "missing_correct_prototype",
                "pull_l2": float("nan"),
                "push_l2": float("nan"),
                "update_l2": float("nan"),
                "wrong_count": 0,
            }
        active = np.asarray(basin.center, dtype=np.float32)
        target = np.asarray(target_state, dtype=np.float32)
        if active.shape != target.shape:
            return {
                "used": False,
                "correct_key": key,
                "reason": "target_shape_mismatch",
                "pull_l2": float("nan"),
                "push_l2": float("nan"),
                "update_l2": float("nan"),
                "wrong_count": 0,
            }

        wrong_vectors: list[np.ndarray] = []
        for wrong in wrong_targets:
            wrong_vector = np.asarray(wrong, dtype=np.float32)
            if wrong_vector.shape == active.shape:
                wrong_vectors.append(wrong_vector)

        pull = target - active
        if wrong_vectors:
            push = np.mean([active - wrong for wrong in wrong_vectors], axis=0).astype(np.float32)
        else:
            push = np.zeros_like(pull, dtype=np.float32)
        update_vec = (pull + float(wrong_strength) * push).astype(np.float32)
        pull_l2 = float(np.linalg.norm(pull, ord=2))
        push_l2 = float(np.linalg.norm(push, ord=2))
        update_l2 = float(np.linalg.norm(update_vec, ord=2))

        self._ensure_feature_omega()
        height, width = self.field.config.shape
        y = int(basin.y) % height
        x = int(basin.x) % width
        gain = float(strength)
        self.feature_omega[y, x, :] += (gain * update_vec).astype(np.float32)
        self.feature_omega[y, x, :] = np.clip(self.feature_omega[y, x, :], -2.5, 2.5)
        return {
            "used": True,
            "correct_key": key,
            "cell_x": int(x),
            "cell_y": int(y),
            "pull_l2": pull_l2,
            "push_l2": push_l2,
            "update_l2": update_l2,
            "gain": gain,
            "wrong_count": int(len(wrong_vectors)),
            "feature_omega_l2": float(np.linalg.norm(self.feature_omega[y, x], ord=2)),
            "prototype_updated": False,
            "global_landscape_updated": False,
        }

    def carve_resonance_slot(
        self,
        basin: BasinFeature,
        target_state: np.ndarray | Sequence[float],
        wrong_targets: Sequence[np.ndarray | Sequence[float]] = (),
        *,
        correct_key: str,
        strength: float = 0.05,
        wrong_strength: float = 0.5,
    ) -> dict[str, float | int | str | bool]:
        """Update only the active cell's keyed resonance slot.

        The sparse tunnel path still aliases when many answers land in the same
        cell. This method adds a second address: the result key determines a
        slot inside the active cell, so ``add:17`` and ``add:20`` can coexist.
        """

        key = str(correct_key)
        if self.num_slots <= 0:
            return {
                "used": False,
                "correct_key": key,
                "reason": "slots_disabled",
                "pull_l2": float("nan"),
                "push_l2": float("nan"),
                "update_l2": float("nan"),
                "wrong_count": 0,
            }
        if key not in self.structural_prototypes:
            return {
                "used": False,
                "correct_key": key,
                "reason": "missing_correct_prototype",
                "pull_l2": float("nan"),
                "push_l2": float("nan"),
                "update_l2": float("nan"),
                "wrong_count": 0,
            }
        active = np.asarray(basin.center, dtype=np.float32)
        target = np.asarray(target_state, dtype=np.float32)
        if active.shape != target.shape:
            return {
                "used": False,
                "correct_key": key,
                "reason": "target_shape_mismatch",
                "pull_l2": float("nan"),
                "push_l2": float("nan"),
                "update_l2": float("nan"),
                "wrong_count": 0,
            }

        wrong_vectors: list[np.ndarray] = []
        for wrong in wrong_targets:
            wrong_vector = np.asarray(wrong, dtype=np.float32)
            if wrong_vector.shape == active.shape:
                wrong_vectors.append(wrong_vector)

        self._ensure_feature_slots()
        if self.feature_slots is None:
            return {
                "used": False,
                "correct_key": key,
                "reason": "slot_allocation_failed",
                "pull_l2": float("nan"),
                "push_l2": float("nan"),
                "update_l2": float("nan"),
                "wrong_count": 0,
            }

        height, width = self.field.config.shape
        y = int(basin.y) % height
        x = int(basin.x) % width
        slot_idx = self.get_slot_index_for_key(key)
        override_key = self.feature_slot_override_key(y, x, key)
        current_slot = np.asarray(
            self.feature_slot_overrides.get(override_key, self.feature_slots[y, x, slot_idx]),
            dtype=np.float32,
        )
        slot_center = (active + current_slot).astype(np.float32)

        pull = target - slot_center
        if wrong_vectors:
            push = np.mean([slot_center - wrong for wrong in wrong_vectors], axis=0).astype(np.float32)
        else:
            push = np.zeros_like(pull, dtype=np.float32)
        update_vec = (pull + float(wrong_strength) * push).astype(np.float32)
        pull_l2 = float(np.linalg.norm(pull, ord=2))
        push_l2 = float(np.linalg.norm(push, ord=2))
        update_l2 = float(np.linalg.norm(update_vec, ord=2))

        gain = float(strength)
        next_slot = current_slot + (gain * update_vec).astype(np.float32)
        next_slot = np.clip(
            next_slot,
            -2.5,
            2.5,
        )
        self.feature_slot_overrides[override_key] = next_slot.astype(np.float32)
        current_gate = np.asarray(self.feature_slot_gates.get(override_key, active), dtype=np.float32)
        gate_alpha = min(1.0, max(0.05, gain))
        self.feature_slot_gates[override_key] = (
            (1.0 - gate_alpha) * current_gate + gate_alpha * active
        ).astype(np.float32)
        self.feature_slots[y, x, slot_idx, :] = next_slot.astype(np.float32)
        return {
            "used": True,
            "correct_key": key,
            "cell_x": int(x),
            "cell_y": int(y),
            "slot_idx": int(slot_idx),
            "slot_key": override_key,
            "pull_l2": pull_l2,
            "push_l2": push_l2,
            "update_l2": update_l2,
            "gain": gain,
            "wrong_count": int(len(wrong_vectors)),
            "feature_slot_l2": float(np.linalg.norm(next_slot, ord=2)),
            "prototype_updated": False,
            "global_landscape_updated": False,
            "collision_safe": True,
        }

    def distill_computational_teacher(
        self,
        basin: BasinFeature,
        teacher: np.ndarray | Sequence[float],
        *,
        correct_key: str,
        wrong_keys: Sequence[str] = (),
        distill_gain: float = 0.10,
        repulsion_strength: float = 0.25,
        topology_gain: float = 0.025,
        sigma: float = 3.0,
        margin: float = 0.20,
    ) -> dict[str, Any]:
        """Distill an active basin toward a computed teacher feature vector."""

        key = str(correct_key)
        if key not in self.structural_prototypes:
            return {
                "used": False,
                "correct_key": key,
                "reason": "missing_correct_prototype",
            }
        active = np.asarray(basin.center, dtype=np.float32)
        teacher_vector = np.asarray(teacher, dtype=np.float32)
        if teacher_vector.shape != active.shape:
            return {
                "used": False,
                "correct_key": key,
                "reason": "teacher_shape_mismatch",
            }

        operation = key.split(":", 1)[0] if ":" in key else None
        nearest_before = self.nearest_structural_prototype(active, operation=operation, k=1)
        correct_before = np.asarray(self.structural_prototypes[key], dtype=np.float32)
        active_teacher_distance = float(np.linalg.norm(active - teacher_vector, ord=2))
        correct_distance_before = float(np.linalg.norm(active - correct_before, ord=2))
        teacher_correct_distance_before = float(np.linalg.norm(teacher_vector - correct_before, ord=2))
        landscape_metrics = self.update_landscape_toward(
            basin,
            teacher_vector,
            strength=topology_gain,
            correct_key=key,
            sigma=sigma,
        )
        correct_before = np.asarray(self.structural_prototypes[key], dtype=np.float32)

        # The active basin cannot be edited directly, so the durable state we can
        # move is the result prototype plus the local potential around the active
        # coordinate. Bias toward the teacher, but keep a small pull toward the
        # observed active basin so nearest-prototype readout can improve.
        gain = float(np.clip(distill_gain, 0.0, 1.0))
        target_vector = ((0.70 * teacher_vector) + (0.30 * active)).astype(np.float32)
        self.structural_prototypes[key] = (
            correct_before + gain * (target_vector - correct_before)
        ).astype(np.float32)

        wrong_updates = 0
        wrong_distances_before: list[float] = []
        for wrong_key in wrong_keys:
            wrong = str(wrong_key)
            if wrong == key or wrong not in self.structural_prototypes:
                continue
            wrong_vector = np.asarray(self.structural_prototypes[wrong], dtype=np.float32)
            delta = wrong_vector - teacher_vector
            distance = float(np.linalg.norm(delta, ord=2))
            wrong_distances_before.append(distance)
            pressure = max(0.0, float(margin) - distance)
            if pressure <= 0.0:
                continue
            direction = delta / max(distance, 1e-6)
            self.structural_prototypes[wrong] = (
                wrong_vector + (float(repulsion_strength) * gain * pressure) * direction
            ).astype(np.float32)
            wrong_updates += 1

        correct_after = np.asarray(self.structural_prototypes[key], dtype=np.float32)
        correct_distance_after = float(np.linalg.norm(active - correct_after, ord=2))
        teacher_correct_distance_after = float(np.linalg.norm(teacher_vector - correct_after, ord=2))

        nearest_after = self.nearest_structural_prototype(active, operation=operation, k=1)
        return {
            "used": True,
            "correct_key": key,
            "nearest_before": nearest_before[0]["key"] if nearest_before else None,
            "nearest_after": nearest_after[0]["key"] if nearest_after else None,
            "target_match_before": bool(nearest_before and nearest_before[0]["key"] == key),
            "target_match_after": bool(nearest_after and nearest_after[0]["key"] == key),
            "active_teacher_distance": active_teacher_distance,
            "correct_distance_before": correct_distance_before,
            "correct_distance_after": correct_distance_after,
            "teacher_correct_distance_before": teacher_correct_distance_before,
            "teacher_correct_distance_after": teacher_correct_distance_after,
            "wrong_distance_min_before": min(wrong_distances_before) if wrong_distances_before else None,
            "wrong_updates": wrong_updates,
            "landscape": landscape_metrics,
        }

    def train_structural_batch(
        self,
        basin_a_centers: Sequence[Sequence[float] | np.ndarray],
        basin_b_centers: Sequence[Sequence[float] | np.ndarray],
        target_ids: Sequence[int],
        *,
        structural_weight: float = 0.5,
    ) -> dict[str, float]:
        if self.decoder is None or self.optimizer is None:
            self._init_decoder()
        if len(basin_a_centers) != len(basin_b_centers) or len(basin_a_centers) != len(target_ids):
            raise ValueError("basin_a_centers, basin_b_centers, and target_ids must have the same length.")
        if not target_ids:
            return {"loss": 0.0, "ce_loss": 0.0, "structural_loss": 0.0, "alignment": 0.0}
        max_target = max(int(item) for item in target_ids)
        if max_target >= self.vocab_capacity:
            raise ValueError("target_id exceeds decoder vocab_capacity.")
        torch, _nn, functional = _torch_modules()
        features_a = torch.as_tensor(np.asarray(basin_a_centers, dtype=np.float32), dtype=torch.float32)
        features_b = torch.as_tensor(np.asarray(basin_b_centers, dtype=np.float32), dtype=torch.float32)
        targets = torch.as_tensor([int(item) for item in target_ids], dtype=torch.long)
        features = torch.cat([features_a, features_b], dim=0)
        doubled_targets = torch.cat([targets, targets], dim=0)
        self.optimizer.zero_grad()
        logits = self.decoder(features)
        ce_loss = functional.cross_entropy(logits, doubled_targets)
        similarity = functional.cosine_similarity(features_a, features_b, dim=-1)
        structural_loss = functional.mse_loss(features_a, features_b)
        alignment = similarity.mean()
        feature_l2 = torch.linalg.vector_norm(features_a - features_b, ord=2, dim=-1).mean()
        loss = ce_loss + float(structural_weight) * structural_loss
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu().item()),
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "structural_loss": float(structural_loss.detach().cpu().item()),
            "alignment": float(alignment.detach().cpu().item()),
            "feature_l2": float(feature_l2.detach().cpu().item()),
        }

    def train_reranker_batch(
        self,
        basin_centers: Sequence[Sequence[float] | np.ndarray],
        candidate_ids: Sequence[int],
        labels: Sequence[int | float],
        candidate_basin_centers: Sequence[Sequence[float] | np.ndarray] | None = None,
    ) -> dict[str, float]:
        if self.reranker is None or self.reranker_optimizer is None:
            self._init_reranker()
        if len(basin_centers) != len(candidate_ids) or len(candidate_ids) != len(labels):
            raise ValueError("basin_centers, candidate_ids, and labels must have the same length.")
        if candidate_basin_centers is not None and len(candidate_basin_centers) != len(candidate_ids):
            raise ValueError("candidate_basin_centers and candidate_ids must have the same length.")
        if not basin_centers:
            return {"loss": 0.0, "accuracy": 0.0, "margin_loss": 0.0}
        max_candidate = max(int(item) for item in candidate_ids)
        if max_candidate >= self.vocab_capacity:
            raise ValueError("candidate_id exceeds reranker vocab_capacity.")
        torch, _nn, functional = _torch_modules()
        features = torch.as_tensor(np.asarray(basin_centers, dtype=np.float32), dtype=torch.float32)
        candidates = torch.as_tensor([int(item) for item in candidate_ids], dtype=torch.long)
        label_tensor = torch.as_tensor([float(item) for item in labels], dtype=torch.float32)
        candidate_features = None
        if candidate_basin_centers is not None:
            candidate_features = torch.as_tensor(
                np.asarray(candidate_basin_centers, dtype=np.float32),
                dtype=torch.float32,
            )
        self.reranker_optimizer.zero_grad()
        scores = self.reranker(features, candidates, candidate_features)
        bce_loss = functional.binary_cross_entropy_with_logits(scores, label_tensor)
        pos_scores = scores[label_tensor > 0.5]
        neg_scores = scores[label_tensor <= 0.5]
        margin_loss = torch.tensor(0.0, dtype=torch.float32)
        if int(pos_scores.numel()) > 0 and int(neg_scores.numel()) > 0:
            pairs = min(int(pos_scores.numel()), int(neg_scores.numel()))
            margin_loss = functional.relu(1.0 - pos_scores[:pairs] + neg_scores[:pairs]).mean()
        loss = bce_loss + margin_loss
        loss.backward()
        self.reranker_optimizer.step()
        with torch.no_grad():
            predictions = (scores.sigmoid() >= 0.5).float()
            accuracy = float((predictions == label_tensor).float().mean().detach().cpu().item())
        return {
            "loss": float(loss.detach().cpu().item()),
            "accuracy": accuracy,
            "margin_loss": float(margin_loss.detach().cpu().item()),
        }

    def score_candidates(
        self,
        prompt: str | Sequence[str],
        candidates: Sequence[str],
        *,
        steps_per_chunk: int = 20,
        anneal: bool = False,
        anneal_steps: int = 30,
    ) -> list[dict[str, Any]]:
        if self.reranker is None:
            self._init_reranker()
        torch, _nn, _functional = _torch_modules()
        candidate_tokens = [normalize_token(item) for item in candidates]
        if not candidate_tokens:
            return []
        basin, prediction_error = self.encode_basin(
            prompt,
            steps_per_chunk=steps_per_chunk,
            anneal=anneal,
            anneal_steps=anneal_steps,
            reset=True,
        )
        candidate_ids = self.vocab.encode_tokens(candidate_tokens, add_new=False)
        features = torch.as_tensor(np.repeat(basin.center.reshape(1, -1), len(candidate_ids), axis=0), dtype=torch.float32)
        candidate_tensor = torch.as_tensor(candidate_ids, dtype=torch.long)
        prompt_text = " ".join(self.tokenize(prompt))
        candidate_centers = []
        for token in candidate_tokens:
            candidate_basin, _candidate_error = self.encode_basin(
                f"{prompt_text} {token}",
                steps_per_chunk=steps_per_chunk,
                anneal=anneal,
                anneal_steps=anneal_steps,
                reset=True,
            )
            candidate_centers.append(candidate_basin.center)
        candidate_features = torch.as_tensor(np.asarray(candidate_centers, dtype=np.float32), dtype=torch.float32)
        with torch.no_grad():
            scores = self.reranker(features, candidate_tensor, candidate_features)
            probabilities = scores.sigmoid()
        ranked = []
        for token, token_id, score, probability in zip(candidate_tokens, candidate_ids, scores, probabilities, strict=False):
            ranked.append(
                {
                    "candidate": token,
                    "token_id": int(token_id),
                    "score": float(score.detach().cpu().item()),
                    "probability": float(probability.detach().cpu().item()),
                    "prediction_error": prediction_error,
                    "basin": basin.to_dict(),
                }
            )
        return sorted(ranked, key=lambda item: item["score"], reverse=True)

    def top_decoder_candidates(
        self,
        prompt: str | Sequence[str],
        *,
        k: int = 5,
        steps_per_chunk: int = 20,
        anneal: bool = False,
        anneal_steps: int = 30,
    ) -> list[dict[str, Any]]:
        if self.decoder is None:
            self._init_decoder()
        torch, _nn, _functional = _torch_modules()
        basin, prediction_error = self.encode_basin(
            prompt,
            steps_per_chunk=steps_per_chunk,
            anneal=anneal,
            anneal_steps=anneal_steps,
            reset=True,
        )
        features = torch.as_tensor(basin.center, dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            logits = self.decoder(features)[0][: len(self.vocab)]
            values, indices = torch.topk(logits, min(max(1, int(k)), len(self.vocab)))
        return [
            {
                "candidate": self.vocab.decode_id(int(index.detach().cpu().item())),
                "token_id": int(index.detach().cpu().item()),
                "decoder_logit": float(value.detach().cpu().item()),
                "prediction_error": prediction_error,
                "basin": basin.to_dict(),
            }
            for value, index in zip(values, indices, strict=False)
        ]

    def rerank(
        self,
        prompt: str | Sequence[str],
        *,
        candidates: Sequence[str] | None = None,
        k: int = 5,
        steps_per_chunk: int = 20,
        anneal: bool = False,
        anneal_steps: int = 30,
    ) -> dict[str, Any]:
        if candidates is None:
            generated = self.top_decoder_candidates(
                prompt,
                k=k,
                steps_per_chunk=steps_per_chunk,
                anneal=anneal,
                anneal_steps=anneal_steps,
            )
            candidates = [item["candidate"] for item in generated]
        scored = self.score_candidates(
            prompt,
            candidates,
            steps_per_chunk=steps_per_chunk,
            anneal=anneal,
            anneal_steps=anneal_steps,
        )
        return {
            "best": scored[0] if scored else None,
            "candidates": scored,
        }

    def generate(
        self,
        prompt: str | Sequence[str],
        *,
        max_tokens: int = 32,
        steps_per_token: int = 15,
        temperature: float = 0.8,
        top_k: int = 16,
        top_p: float = 1.0,
        temperature_decay: float = 1.0,
        min_temperature: float = 0.05,
        repeat_penalty: float = 1.0,
        repeat_window: int = 10,
        anneal: bool = False,
        anneal_steps: int = 30,
        reset: bool = True,
    ) -> list[str]:
        return [step.token for step in self.generate_steps(
            prompt,
            max_tokens=max_tokens,
            steps_per_token=steps_per_token,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            temperature_decay=temperature_decay,
            min_temperature=min_temperature,
            repeat_penalty=repeat_penalty,
            repeat_window=repeat_window,
            anneal=anneal,
            anneal_steps=anneal_steps,
            reset=reset,
        )]

    def generate_steps(
        self,
        prompt: str | Sequence[str],
        *,
        max_tokens: int = 32,
        steps_per_token: int = 15,
        temperature: float = 0.8,
        top_k: int = 16,
        top_p: float = 1.0,
        temperature_decay: float = 1.0,
        min_temperature: float = 0.05,
        repeat_penalty: float = 1.0,
        repeat_window: int = 10,
        anneal: bool = False,
        anneal_steps: int = 30,
        reset: bool = True,
    ) -> list[GenerationStep]:
        if self.decoder is None:
            self._init_decoder()
        torch, _nn, functional = _torch_modules()
        if len(self.vocab) <= 2:
            raise ValueError("Vocabulary has no learned tokens. Train first or load a saved model.")
        if reset:
            self.field.reset_field()
            self.field.inject_text(" ".join(self.tokenize(prompt)), self.encoder)

        steps: list[GenerationStep] = []
        generated_ids: list[int] = []
        for step_index in range(max(0, int(max_tokens))):
            prediction_error = 0.0
            total_steps = max(1, int(steps_per_token))
            if anneal:
                total_steps = max(total_steps, int(anneal_steps))
            for substep in range(total_steps):
                if anneal:
                    progress = substep / max(1, total_steps - 1)
                    self.field.inject_noise(scale=0.012 * (1.0 - progress))
                predicted = self.field.predict_phase()
                self.field.step()
                prediction_error = self.field.observe_prediction(predicted)

            basin = self.apply_feature_omega(self.field.find_basin(feature_dim=self.basin_dim))
            features = torch.as_tensor(basin.center, dtype=torch.float32).view(1, -1)
            logits = self.decoder(features)[0][: len(self.vocab)]
            logits = apply_repetition_penalty(
                logits,
                generated_ids[-max(0, int(repeat_window)) :],
                penalty=repeat_penalty,
            )
            effective_temperature = max(float(min_temperature), float(temperature) * (float(temperature_decay) ** step_index))
            token_id, probability = sample_logits(
                logits,
                temperature=effective_temperature,
                top_k=top_k,
                top_p=top_p,
            )
            token = self.vocab.decode_id(token_id)
            generated_ids.append(token_id)
            steps.append(
                GenerationStep(
                    token=token,
                    token_id=token_id,
                    probability=probability,
                    basin=basin.to_dict(),
                    prediction_error=float(prediction_error),
                )
            )
            self.field.inject_residual(token, encoder=self.encoder)
        return steps

    def evaluate_texts(
        self,
        text_iter: Iterable[str],
        *,
        steps_per_chunk: int = 20,
        max_chunks: int | None = None,
        reset_between_chunks: bool = True,
    ) -> dict[str, Any]:
        losses: list[float] = []
        prediction_errors: list[float] = []
        for index, text in enumerate(text_iter, start=1):
            if max_chunks is not None and index > max_chunks:
                break
            observation = self.observe_text(
                text,
                steps_per_chunk=steps_per_chunk,
                train_decoder=False,
                train_topology=False,
                freeze_omega=True,
                reset=reset_between_chunks,
            )
            prediction_errors.append(observation.mean_prediction_error)
            if observation.target_id is not None:
                center = np.asarray(observation.basin["center"], dtype=np.float32)
                losses.append(self.decoder_loss(center, observation.target_id))
        mean_loss = sum(losses) / len(losses) if losses else None
        return {
            "chunks": len(prediction_errors),
            "scored_chunks": len(losses),
            "mean_decoder_loss": mean_loss,
            "perplexity": perplexity_from_loss(mean_loss),
            "mean_prediction_error": sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0,
        }

    def save(self, out_dir: str | Path) -> dict[str, str]:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        topology = output / "topology.q8.npz"
        vocab_path = output / "vocab.json"
        config_path = output / "model_config.json"
        prototypes_path = output / "prototypes.json"
        feature_omega_path = output / "feature_omega.npz"
        feature_slots_path = output / "feature_slots.npz"
        feature_slot_overrides_path = output / "feature_slot_overrides.npz"
        feature_slot_gates_path = output / "feature_slot_gates.npz"
        decoder_path = output / "decoder.pt"
        reranker_path = output / "reranker.pt"
        gate_path = output / "gate.pt"
        delta_path = output / "delta_scorer.pt"
        field_config = self.field.config
        self.field.save_quantized(topology)
        self.vocab.save(vocab_path)
        self._ensure_feature_omega()
        np.savez_compressed(feature_omega_path, feature_omega=self.feature_omega.astype(np.float16))
        self._ensure_feature_slots()
        if self.feature_slots is not None:
            np.savez_compressed(feature_slots_path, feature_slots=self.feature_slots.astype(np.float16))
        if self.feature_slot_overrides:
            sorted_keys = sorted(self.feature_slot_overrides)
            key_width = max(1, max(len(key) for key in sorted_keys))
            override_keys = np.asarray(sorted_keys, dtype=f"U{key_width}")
            override_values = np.asarray(
                [self.feature_slot_overrides[str(key)] for key in override_keys],
                dtype=np.float16,
            )
            np.savez_compressed(
                feature_slot_overrides_path,
                keys=override_keys,
                values=override_values,
            )
        if self.feature_slot_gates:
            sorted_gate_keys = sorted(self.feature_slot_gates)
            gate_key_width = max(1, max(len(key) for key in sorted_gate_keys))
            gate_keys = np.asarray(sorted_gate_keys, dtype=f"U{gate_key_width}")
            gate_values = np.asarray(
                [self.feature_slot_gates[str(key)] for key in gate_keys],
                dtype=np.float16,
            )
            np.savez_compressed(
                feature_slot_gates_path,
                keys=gate_keys,
                values=gate_values,
            )
        config_path.write_text(
            json.dumps(
                {
                    "grid_size": field_config.width,
                    "basin_dim": self.basin_dim,
                    "hidden": self.hidden,
                    "vocab_capacity": self.vocab_capacity,
                    "seed": field_config.seed,
                    "backend": field_config.laplacian_backend,
                    "pin_strength": field_config.phase_pin_strength,
                    "residual_carry": field_config.phase_residual_carry,
                    "num_slots": self.num_slots,
                    "encoder_mode": self.encoder_mode,
                    "structured_result_hint": self.structured_result_hint,
                    "structured_feature_strength": self.structured_feature_strength,
                    "gate_trained_steps": self.gate_trained_steps,
                    "delta_trained_steps": self.delta_trained_steps,
                    "decoder_type": getattr(self.decoder, "decoder_type", "mlp") if self.decoder is not None else "none",
                    "readout_temperature": float(getattr(self.decoder, "temperature", 0.1))
                    if getattr(self.decoder, "decoder_type", "mlp") == "prototype-readout"
                    else None,
                    "readout_direct_scale": float(getattr(self.decoder, "direct_scale", 8.0))
                    if getattr(self.decoder, "decoder_type", "mlp") == "prototype-readout"
                    else None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if self.decoder is not None:
            torch, _nn, _functional = _torch_modules()
            torch.save(self.decoder.state_dict(), decoder_path)
        if self.reranker is not None:
            torch, _nn, _functional = _torch_modules()
            torch.save(self.reranker.state_dict(), reranker_path)
        if self.gate_net is not None:
            torch, _nn, _functional = _torch_modules()
            torch.save(self.gate_net.state_dict(), gate_path)
        if self.delta_scorer is not None:
            torch, _nn, _functional = _torch_modules()
            torch.save(self.delta_scorer.state_dict(), delta_path)
        prototypes_path.write_text(
            json.dumps(
                {
                    key: [float(item) for item in value.tolist()]
                    for key, value in sorted(self.structural_prototypes.items())
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "topology": str(topology),
            "vocab": str(vocab_path),
            "config": str(config_path),
            "decoder": str(decoder_path),
            "reranker": str(reranker_path),
            "gate": str(gate_path),
            "delta_scorer": str(delta_path),
            "prototypes": str(prototypes_path),
            "feature_omega": str(feature_omega_path),
            "feature_slots": str(feature_slots_path) if self.feature_slots is not None else "",
            "feature_slot_overrides": str(feature_slot_overrides_path) if self.feature_slot_overrides else "",
            "feature_slot_gates": str(feature_slot_gates_path) if self.feature_slot_gates else "",
        }

    @classmethod
    def load(cls, model_dir: str | Path, *, load_decoder: bool = True) -> PhaseModel:
        path = Path(model_dir)
        config = json.loads((path / "model_config.json").read_text(encoding="utf-8"))
        vocab = PhaseVocabulary.load(path / "vocab.json")
        decoder_type = str(config.get("decoder_type", "mlp"))
        model = cls(
            grid_size=int(config["grid_size"]),
            vocab=vocab,
            vocab_capacity=int(config["vocab_capacity"]),
            basin_dim=int(config["basin_dim"]),
            hidden=int(config["hidden"]),
            seed=int(config.get("seed", 7)),
            backend=str(config.get("backend", "auto")),
            pin_strength=float(config.get("pin_strength", 0.25)),
            residual_carry=float(config.get("residual_carry", 0.08)),
            num_slots=int(config.get("num_slots", 0)),
            encoder_mode=str(config.get("encoder_mode", "text")),
            structured_result_hint=bool(config.get("structured_result_hint", False)),
            structured_feature_strength=float(config.get("structured_feature_strength", 2.0)),
            create_decoder=load_decoder and decoder_type != "prototype-readout",
        )
        model.gate_trained_steps = int(config.get("gate_trained_steps", 0))
        model.delta_trained_steps = int(config.get("delta_trained_steps", 0))
        topology = path / "topology.q8.npz"
        if topology.exists():
            model.field = PhaseFieldMesh.load_quantized(topology)
            model.config = model.field.config
        prototypes_path = path / "prototypes.json"
        if prototypes_path.exists():
            prototypes = json.loads(prototypes_path.read_text(encoding="utf-8"))
            model.structural_prototypes = {
                str(key): np.asarray(value, dtype=np.float32)
                for key, value in prototypes.items()
            }
        feature_omega_path = path / "feature_omega.npz"
        if feature_omega_path.exists():
            data = np.load(feature_omega_path, allow_pickle=False)
            if "feature_omega" in data:
                model.feature_omega = data["feature_omega"].astype(np.float32)
                model._ensure_feature_omega()
        feature_slots_path = path / "feature_slots.npz"
        if feature_slots_path.exists():
            data = np.load(feature_slots_path, allow_pickle=False)
            if "feature_slots" in data:
                loaded_slots = data["feature_slots"].astype(np.float16)
                if loaded_slots.ndim == 4:
                    model.num_slots = int(loaded_slots.shape[2])
                    model.feature_slots = loaded_slots
                    model._ensure_feature_slots()
        feature_slot_overrides_path = path / "feature_slot_overrides.npz"
        if feature_slot_overrides_path.exists():
            data = np.load(feature_slot_overrides_path, allow_pickle=False)
            if "keys" in data and "values" in data:
                keys = [str(key) for key in data["keys"].tolist()]
                values = data["values"].astype(np.float32)
                model.feature_slot_overrides = {
                    key: np.asarray(value, dtype=np.float32)
                    for key, value in zip(keys, values)
                    if np.asarray(value).shape == (model.basin_dim,)
                }
        feature_slot_gates_path = path / "feature_slot_gates.npz"
        if feature_slot_gates_path.exists():
            data = np.load(feature_slot_gates_path, allow_pickle=False)
            if "keys" in data and "values" in data:
                keys = [str(key) for key in data["keys"].tolist()]
                values = data["values"].astype(np.float32)
                model.feature_slot_gates = {
                    key: np.asarray(value, dtype=np.float32)
                    for key, value in zip(keys, values)
                    if np.asarray(value).shape == (model.basin_dim,)
                }
        decoder_path = path / "decoder.pt"
        if load_decoder and decoder_path.exists():
            torch, _nn, _functional = _torch_modules()
            if decoder_type == "prototype-readout":
                model.use_prototype_readout(
                    temperature=float(config.get("readout_temperature") or 0.1),
                    direct_scale=float(config.get("readout_direct_scale") or 8.0),
                )
            elif model.decoder is None:
                model._init_decoder()
            model.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
            model.decoder.eval()
        reranker_path = path / "reranker.pt"
        if load_decoder and reranker_path.exists():
            torch, _nn, _functional = _torch_modules()
            if model.reranker is None:
                model._init_reranker()
            model.reranker.load_state_dict(torch.load(reranker_path, map_location="cpu"))
            model.reranker.eval()
        gate_path = path / "gate.pt"
        if load_decoder and gate_path.exists() and model.num_slots > 0:
            torch, _nn, _functional = _torch_modules()
            if model.gate_net is None:
                model._init_gate()
            model.gate_net.load_state_dict(torch.load(gate_path, map_location="cpu"))
            model.gate_net.eval()
        delta_path = path / "delta_scorer.pt"
        if load_decoder and delta_path.exists():
            torch, _nn, _functional = _torch_modules()
            if model.delta_scorer is None:
                model._init_delta_scorer()
            try:
                model.delta_scorer.load_state_dict(torch.load(delta_path, map_location="cpu"))
                model.delta_scorer.eval()
            except RuntimeError:
                model.delta_trained_steps = 0
                model._init_delta_scorer()
        return model


def normalize_token(token: str) -> str:
    token = str(token).strip().lower()
    return token if token else "<empty>"


def perplexity_from_loss(loss: float | None) -> float | None:
    if loss is None:
        return None
    return float(math.exp(min(20.0, max(0.0, loss))))


def cosine_similarity(left: np.ndarray | Sequence[float], right: np.ndarray | Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_arr, right_arr) / denominator)


def circular_midpoint(left: int, right: int, size: int) -> int:
    size = max(1, int(size))
    left = int(left) % size
    right = int(right) % size
    delta = ((right - left + size // 2) % size) - size // 2
    return int((left + delta / 2.0) % size)


def structural_feature_l2(left: np.ndarray | Sequence[float], right: np.ndarray | Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(left_arr - right_arr, ord=2))


def structural_feature_mse(left: np.ndarray | Sequence[float], right: np.ndarray | Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    diff = left_arr - right_arr
    return float(np.mean(diff * diff))


def infer_operation_type(*texts: str) -> str:
    joined = " ".join(str(text).lower() for text in texts)
    tokens = set(TOKEN_RE.findall(joined))
    if tokens & {"plus", "add", "sum"} or "+" in joined:
        return "add"
    if tokens & {"times", "multiply", "product"} or "*" in joined:
        return "mul"
    if tokens & {"minus", "subtract", "difference"} or "-" in joined:
        return "sub"
    if tokens & {"divide", "divided", "quotient"} or "/" in joined:
        return "div"
    if tokens & {"greater", "less", "compare", "is"} or ">" in joined or "<" in joined:
        return "compare"
    return "other"


def structural_prototype_key(seq_a: str, seq_b: str, target: str) -> str:
    operation = infer_operation_type(seq_a, seq_b)
    return f"{operation}:{normalize_token(target)}"


def prototype_target_from_key(prototype_key: str) -> str:
    key = str(prototype_key)
    if ":" not in key:
        return normalize_token(key)
    return normalize_token(key.rsplit(":", 1)[1])


def hard_negative_ids(target_id: int, vocab: PhaseVocabulary, *, k: int = 12) -> list[int]:
    target_id = int(target_id)
    negatives: list[int] = []
    seen = {target_id}
    token = vocab.decode_id(target_id)
    try:
        number = int(token)
    except ValueError:
        number = None
    if number is not None:
        for offset in (-2, -1, 1, 2, -5, 5, 10, -10):
            candidate = str(number + offset)
            candidate_id = vocab.token_to_idx.get(candidate)
            if candidate_id is not None and candidate_id not in seen:
                negatives.append(candidate_id)
                seen.add(candidate_id)
    for token_text in ("yes", "no", "answer", "question", "plus", "minus", "times", "greater", "is"):
        candidate_id = vocab.token_to_idx.get(token_text)
        if candidate_id is not None and candidate_id not in seen:
            negatives.append(candidate_id)
            seen.add(candidate_id)
        if len(negatives) >= k:
            return negatives
    for candidate_id in range(max(0, target_id - 8), min(len(vocab), target_id + 9)):
        if candidate_id not in seen:
            negatives.append(candidate_id)
            seen.add(candidate_id)
        if len(negatives) >= k:
            return negatives
    for token_text in ("0", "1", "2", "3", "5", "10", "17", "20", "30", "42"):
        candidate_id = vocab.token_to_idx.get(token_text)
        if candidate_id is not None and candidate_id not in seen:
            negatives.append(candidate_id)
            seen.add(candidate_id)
        if len(negatives) >= k:
            return negatives
    return negatives


def decoder_training_loss(logits: Any, targets: Any, vocab: PhaseVocabulary, *, mode: str = "next-token"):
    _torch, _nn, functional = _torch_modules()
    mode = str(mode)
    if mode == "next-token":
        return functional.cross_entropy(logits, targets)
    if mode != "contrastive":
        raise ValueError(f"unsupported decoder train mode: {mode}")

    log_probs = functional.log_softmax(logits, dim=-1)
    correct_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    negative_rows = []
    for row_index, target_id in enumerate(targets.detach().cpu().tolist()):
        negatives = hard_negative_ids(int(target_id), vocab, k=4)
        if negatives:
            negative_rows.append(log_probs[row_index, negatives].mean())
        else:
            negative_rows.append(log_probs[row_index].mean())
    neg_log_probs = _torch.stack(negative_rows)
    margin_loss = functional.softplus(neg_log_probs - correct_log_probs + 1.0).mean()
    return functional.cross_entropy(logits, targets) + margin_loss


def apply_repetition_penalty(logits: Any, recent_token_ids: Sequence[int], *, penalty: float = 1.0):
    penalty = float(penalty)
    if penalty <= 1.0 or not recent_token_ids:
        return logits
    adjusted = logits.clone()
    vocab_size = int(adjusted.shape[-1])
    for token_id in set(int(item) for item in recent_token_ids):
        if token_id < 0 or token_id >= vocab_size:
            continue
        if adjusted[token_id] > 0:
            adjusted[token_id] = adjusted[token_id] / penalty
        else:
            adjusted[token_id] = adjusted[token_id] * penalty
    return adjusted


def sample_logits(logits: Any, *, temperature: float = 0.8, top_k: int = 16, top_p: float = 1.0) -> tuple[int, float]:
    torch, _nn, functional = _torch_modules()
    if float(temperature) <= 0.0:
        token_id = int(torch.argmax(logits).detach().cpu().item())
        return token_id, 1.0
    scaled = logits / max(float(temperature), 1e-6)
    keep = min(max(1, int(top_k)), int(scaled.shape[-1]))
    values, indices = torch.topk(scaled, keep)
    sorted_values, sorted_order = torch.sort(values, descending=True)
    sorted_indices = indices[sorted_order]
    probs = functional.softmax(sorted_values, dim=-1)
    top_p = float(top_p)
    if 0.0 < top_p < 1.0:
        cumulative = torch.cumsum(probs, dim=-1)
        keep_mask = cumulative <= top_p
        keep_mask[0] = True
        probs = probs[keep_mask]
        sorted_indices = sorted_indices[keep_mask]
        probs = probs / probs.sum()
    sampled_offset = torch.multinomial(probs, 1).item()
    token_id = int(sorted_indices[sampled_offset].detach().cpu().item())
    probability = float(probs[sampled_offset].detach().cpu().item())
    return token_id, probability
