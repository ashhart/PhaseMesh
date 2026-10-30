from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .weight_pour import load_weight_manifest


TOKENIZER_IMPORT_ERROR = (
    "Phase weight readout requires a tokenizer. Install `transformers` or pass an injected tokenizer in code."
)


@dataclass
class PhaseWeightReadoutConfig:
    context_tokens: int = 48
    phase_mix: float = 0.08
    repeat_penalty: float = 1.12
    seed: int = 7


class PhaseWeightReader:
    """Prompt-conditioned readout over a poured PhaseMesh checkpoint artifact.

    This does not run the original transformer layers. It reads the PhaseMesh
    artifact created by `weight-pour`: token-row phase signatures plus the
    global checkpoint phase bank. The output is a native PhaseMesh nearest-phase
    token readout.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        tokenizer: Any | None = None,
        config: PhaseWeightReadoutConfig | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.manifest = load_weight_manifest(self.artifact_dir)
        self.config = config or PhaseWeightReadoutConfig(
            seed=int(self.manifest.get("config", {}).get("seed", 7))
        )
        self.tokenizer = tokenizer or self._load_tokenizer()
        self.token_signatures = self._load_token_signatures()
        self.global_signature = self._load_global_signature(int(self.token_signatures.shape[1]))

    def rank_tokens(self, prompt: str, *, top_k: int = 20, recent_ids: list[int] | None = None) -> list[dict[str, Any]]:
        prompt_ids = self._encode(prompt)
        active = self._active_signature(prompt_ids)
        scores = self._score_ids(active)
        for token_id in self._special_ids():
            if 0 <= token_id < scores.size:
                scores[token_id] = -1.0e12
        recent = Counter(recent_ids or [])
        for token_id, count in recent.items():
            if 0 <= token_id < scores.size and self.config.repeat_penalty > 1.0:
                scores[token_id] -= math.log(float(self.config.repeat_penalty)) * (1.0 + float(count))
        rows: list[dict[str, Any]] = []
        for token_id in np.argsort(scores)[::-1]:
            token = self._decode_one(int(token_id))
            if not _usable_token(token):
                continue
            rows.append({"id": int(token_id), "token": token, "score": float(scores[int(token_id)])})
            if len(rows) >= max(1, int(top_k)):
                break
        return rows

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 32,
        top_k: int = 24,
        temperature: float = 0.0,
        seed: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        generated_ids: list[int] = []
        rng = np.random.default_rng(self.config.seed if seed is None else int(seed))
        text = str(prompt)
        trace: list[dict[str, Any]] = []
        for _ in range(max(0, int(max_tokens))):
            ranks = self.rank_tokens(text, top_k=top_k, recent_ids=generated_ids[-24:])
            if not ranks:
                break
            chosen = self._sample(ranks, temperature=temperature, rng=rng)
            generated_ids.append(int(chosen["id"]))
            completion = self._decode(generated_ids)
            text = str(prompt) + completion
            trace.append({"token": chosen["token"], "id": chosen["id"], "score": chosen["score"]})
            if chosen["token"].strip() in {".", "!", "?", "\n"} and len(generated_ids) >= 6:
                break
        completion = self._decode(generated_ids)
        return {
            "prompt": prompt,
            "completion": completion,
            "text": str(prompt) + completion,
            "tokens": trace,
            "artifact": self.summary(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "type": "phase-weight-reader",
            "source": self.manifest.get("source"),
            "artifact_dir": str(self.artifact_dir),
            "checkpoint_values": int(self.manifest.get("elements_seen", 0)),
            "tensors": int(self.manifest.get("tensors", 0)),
            "token_signatures": list(self.token_signatures.shape),
            "phase_bank_norm": float(self.manifest.get("phase_bank_norm", 0.0)),
        }

    def _load_tokenizer(self) -> Any:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(TOKENIZER_IMPORT_ERROR) from exc
        metadata = self.artifact_dir / "teacher_metadata"
        if not metadata.exists():
            raise FileNotFoundError(f"No teacher_metadata tokenizer folder in {self.artifact_dir}")
        return AutoTokenizer.from_pretrained(str(metadata), local_files_only=True)

    def _load_token_signatures(self) -> np.ndarray:
        files = list(self.manifest.get("token_signature_files") or [])
        if not files:
            raise FileNotFoundError(f"No token signature files in {self.artifact_dir}")
        preferred = files[0]
        for name in files:
            lowered = str(name).lower()
            if "embed_tokens" in lowered or "wte" in lowered:
                preferred = name
                break
        payload = np.load(self.artifact_dir / preferred)
        signatures = (payload["real"] + 1j * payload["imag"]).astype(np.complex64)
        norms = np.linalg.norm(signatures, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (signatures / norms).astype(np.complex64)

    def _load_global_signature(self, token_cells: int) -> np.ndarray:
        bank_name = str(self.manifest.get("phase_bank", "phase_weight_bank.npz"))
        payload = np.load(self.artifact_dir / bank_name)
        bank = (payload["real"] + 1j * payload["imag"]).astype(np.complex64)
        if bank.size == token_cells:
            compressed = bank
        else:
            usable = (bank.size // token_cells) * token_cells
            if usable == 0:
                compressed = np.resize(bank, token_cells)
            else:
                compressed = bank[:usable].reshape(token_cells, -1).mean(axis=1)
        norm = float(np.linalg.norm(compressed))
        if norm > 0.0:
            compressed = compressed / norm
        return compressed.astype(np.complex64)

    def _active_signature(self, token_ids: list[int]) -> np.ndarray:
        width = int(self.token_signatures.shape[1])
        active = np.zeros(width, dtype=np.complex64)
        usable = [token_id for token_id in token_ids[-max(1, int(self.config.context_tokens)) :] if 0 <= token_id < self.token_signatures.shape[0]]
        for position, token_id in enumerate(usable):
            active += self.token_signatures[int(token_id)] * _position_phasor(position, width, int(self.config.seed))
        if not usable:
            active += self.global_signature
        active = active + float(self.config.phase_mix) * self.global_signature
        norm = float(np.linalg.norm(active))
        if norm > 0.0:
            active = active / norm
        return active.astype(np.complex64)

    def _score_ids(self, active: np.ndarray) -> np.ndarray:
        return np.real(self.token_signatures @ np.conj(active)).astype(np.float32)

    def _encode(self, text: str) -> list[int]:
        try:
            return [int(item) for item in self.tokenizer.encode(str(text), add_special_tokens=False)]
        except TypeError:
            return [int(item) for item in self.tokenizer.encode(str(text))]

    def _decode(self, token_ids: list[int]) -> str:
        try:
            return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        except TypeError:
            return str(self.tokenizer.decode(token_ids))

    def _decode_one(self, token_id: int) -> str:
        return self._decode([int(token_id)])

    def _special_ids(self) -> set[int]:
        values = set()
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            value = getattr(self.tokenizer, attr, None)
            if value is not None:
                values.add(int(value))
        ids = getattr(self.tokenizer, "all_special_ids", None)
        if ids:
            values.update(int(item) for item in ids)
        return values

    def _sample(self, ranks: list[dict[str, Any]], *, temperature: float, rng: np.random.Generator) -> dict[str, Any]:
        if temperature <= 0.0:
            return ranks[0]
        scores = np.asarray([row["score"] for row in ranks], dtype=np.float64)
        scores = scores / max(1e-6, float(temperature))
        scores -= float(np.max(scores))
        probs = np.exp(scores)
        total = float(np.sum(probs))
        if total <= 0.0 or not np.isfinite(total):
            return ranks[0]
        probs /= total
        return ranks[int(rng.choice(np.arange(len(ranks)), p=probs))]


def _position_phasor(position: int, cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(f"{int(seed)}:readout-pos:{int(position)}"))
    angles = rng.uniform(0.0, 2.0 * math.pi, int(cells))
    return np.exp(1j * angles).astype(np.complex64)


def _stable_seed(text: str) -> int:
    import hashlib

    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "big")


def _usable_token(token: str) -> bool:
    value = str(token)
    stripped = value.strip()
    if not stripped:
        return False
    if "�" in value:
        return False
    if stripped.startswith("<") and stripped.endswith(">"):
        return False
    if stripped.startswith("<|"):
        return False
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in value):
        return False
    return True
