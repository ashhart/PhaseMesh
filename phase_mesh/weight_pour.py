from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


WEIGHT_POUR_IMPORT_ERROR = (
    "Checkpoint weight pouring requires optional Hugging Face checkpoint dependencies. "
    "Install them with `pip install huggingface_hub safetensors torch` or `pip install -e '.[distill]'`."
)


@dataclass
class PhaseWeightPourConfig:
    phase_cells: int = 65536
    token_cells: int = 64
    seed: int = 7
    chunk_size: int = 1_000_000
    include_token_signatures: bool = True
    max_elements_per_tensor: int | None = None


def pour_arrays_to_phase(
    arrays: Mapping[str, np.ndarray],
    *,
    out_dir: str | Path,
    config: PhaseWeightPourConfig | None = None,
    source: str = "arrays",
) -> dict[str, Any]:
    """Pour in-memory tensors into a PhaseMesh weight artifact.

    This is the testable core used by the Hugging Face checkpoint importer.
    Every numeric value contributes to the global phase bank unless a
    max-elements cap is explicitly supplied for smoke tests.
    """

    cfg = config or PhaseWeightPourConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    phase_bank = np.zeros(int(cfg.phase_cells), dtype=np.complex64)
    stats: list[dict[str, Any]] = []
    token_signature_files: list[str] = []
    for name, array in arrays.items():
        stat = _pour_tensor(name, np.asarray(array), phase_bank, config=cfg)
        stats.append(stat)
        if cfg.include_token_signatures and _looks_like_token_matrix(name, np.asarray(array)):
            token_signature_files.append(_write_token_signatures(out, name, np.asarray(array), cfg))
    return _write_artifact(
        out,
        source=source,
        config=cfg,
        phase_bank=phase_bank,
        tensor_stats=stats,
        token_signature_files=token_signature_files,
        copied_files=[],
    )


def pour_hf_checkpoint_to_phase(
    *,
    teacher_model: str,
    out_dir: str | Path,
    config: PhaseWeightPourConfig | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Stream a Hugging Face causal-LM checkpoint into PhaseMesh phase banks."""

    cfg = config or PhaseWeightPourConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = _resolve_checkpoint_path(teacher_model, revision=revision, local_files_only=local_files_only)
    phase_bank = np.zeros(int(cfg.phase_cells), dtype=np.complex64)
    stats: list[dict[str, Any]] = []
    token_signature_files: list[str] = []
    for name, array, info in _iter_checkpoint_tensors(model_path):
        stat = _pour_tensor(name, np.asarray(array), phase_bank, config=cfg, extra=info)
        stats.append(stat)
        if cfg.include_token_signatures and _looks_like_token_matrix(name, np.asarray(array)):
            token_signature_files.append(_write_token_signatures(out, name, np.asarray(array), cfg))
    copied = _copy_checkpoint_metadata(model_path, out)
    return _write_artifact(
        out,
        source=str(teacher_model),
        config=cfg,
        phase_bank=phase_bank,
        tensor_stats=stats,
        token_signature_files=token_signature_files,
        copied_files=copied,
        checkpoint_path=str(model_path),
    )


def load_weight_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))


def _pour_tensor(
    name: str,
    array: np.ndarray,
    phase_bank: np.ndarray,
    *,
    config: PhaseWeightPourConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(array)
    if not np.issubdtype(values.dtype, np.number):
        return {
            "name": name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "elements_seen": 0,
            "skipped": "non-numeric",
        }
    flat = values.reshape(-1)
    limit = flat.size
    if config.max_elements_per_tensor is not None:
        limit = min(limit, int(config.max_elements_per_tensor))
    salt = _stable_int(f"{int(config.seed)}:{name}")
    seen = 0
    total = 0.0
    sq_total = 0.0
    abs_total = 0.0
    min_value = math.inf
    max_value = -math.inf
    checksum_a = 0
    checksum_b = 0
    chunk_size = max(1, int(config.chunk_size))
    for offset in range(0, limit, chunk_size):
        raw = flat[offset : min(limit, offset + chunk_size)]
        chunk = np.asarray(raw, dtype=np.float32)
        if chunk.size == 0:
            continue
        _add_chunk_to_phase_bank(phase_bank, chunk, offset=offset, salt=salt)
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            total += float(np.sum(finite, dtype=np.float64))
            sq_total += float(np.sum(finite * finite, dtype=np.float64))
            abs_total += float(np.sum(np.abs(finite), dtype=np.float64))
            min_value = min(min_value, float(np.min(finite)))
            max_value = max(max_value, float(np.max(finite)))
        checksum_a = int((checksum_a + int(np.sum(np.abs(chunk) * 1000003.0, dtype=np.float64))) % (2**63 - 1))
        checksum_b = int((checksum_b + int(np.sum(chunk * np.arange(1, chunk.size + 1, dtype=np.float32), dtype=np.float64))) % (2**63 - 1))
        seen += int(chunk.size)
    mean = total / max(1, seen)
    variance = max(0.0, (sq_total / max(1, seen)) - mean * mean)
    payload: dict[str, Any] = {
        "name": name,
        "group": _tensor_group(name),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "elements": int(flat.size),
        "elements_seen": int(seen),
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "l1": float(abs_total),
        "l2": float(math.sqrt(max(0.0, sq_total))),
        "min": None if min_value is math.inf else float(min_value),
        "max": None if max_value == -math.inf else float(max_value),
        "checksum": f"{checksum_a:016x}{checksum_b % (2**63 - 1):016x}",
    }
    if extra:
        payload.update(extra)
    return payload


def _add_chunk_to_phase_bank(phase_bank: np.ndarray, chunk: np.ndarray, *, offset: int, salt: int) -> None:
    cells_count = int(phase_bank.size)
    indices = np.arange(int(offset), int(offset) + int(chunk.size), dtype=np.uint64)
    mixed = _mix_uint64(indices + np.uint64(salt))
    cells = np.asarray(mixed % np.uint64(cells_count), dtype=np.int64)
    angle_seed = _mix_uint64(mixed ^ np.uint64(0x9E3779B97F4A7C15))
    angles = (np.asarray(angle_seed >> np.uint64(11), dtype=np.float64) / float(1 << 53)) * (2.0 * math.pi)
    real = np.bincount(cells, weights=np.asarray(chunk * np.cos(angles), dtype=np.float64), minlength=cells_count)
    imag = np.bincount(cells, weights=np.asarray(chunk * np.sin(angles), dtype=np.float64), minlength=cells_count)
    phase_bank.real += real.astype(np.float32)
    phase_bank.imag += imag.astype(np.float32)


def _write_token_signatures(out: Path, name: str, array: np.ndarray, config: PhaseWeightPourConfig) -> str:
    matrix = np.asarray(array)
    rows = int(matrix.shape[0])
    token_cells = int(config.token_cells)
    real = np.zeros((rows, token_cells), dtype=np.float32)
    imag = np.zeros((rows, token_cells), dtype=np.float32)
    limit_cols = int(matrix.shape[1])
    if config.max_elements_per_tensor is not None:
        rows = min(rows, max(1, int(config.max_elements_per_tensor) // max(1, limit_cols)))
    safe_name = _safe_name(name)
    for row_index in range(rows):
        row_bank = np.zeros(token_cells, dtype=np.complex64)
        row = np.asarray(matrix[row_index], dtype=np.float32).reshape(-1)
        _add_chunk_to_phase_bank(row_bank, row, offset=0, salt=_stable_int(f"{config.seed}:{name}:{row_index}"))
        norm = float(np.linalg.norm(row_bank))
        if norm > 0.0:
            row_bank = row_bank / norm
        real[row_index] = row_bank.real
        imag[row_index] = row_bank.imag
    path = out / f"token_signatures_{safe_name}.npz"
    np.savez_compressed(path, real=real, imag=imag, tensor=name)
    return str(path.name)


def _write_artifact(
    out: Path,
    *,
    source: str,
    config: PhaseWeightPourConfig,
    phase_bank: np.ndarray,
    tensor_stats: list[dict[str, Any]],
    token_signature_files: list[str],
    copied_files: list[str],
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    np.savez_compressed(out / "phase_weight_bank.npz", real=phase_bank.real, imag=phase_bank.imag)
    stats_path = out / "tensor_stats.jsonl"
    stats_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in tensor_stats), encoding="utf-8")
    total_seen = sum(int(row.get("elements_seen", 0)) for row in tensor_stats)
    manifest = {
        "status": "ok",
        "type": "phase-mesh-weight-pour",
        "source": source,
        "checkpoint_path": checkpoint_path,
        "config": asdict(config),
        "tensors": len(tensor_stats),
        "elements_seen": int(total_seen),
        "phase_bank": "phase_weight_bank.npz",
        "phase_bank_norm": float(np.linalg.norm(phase_bank)),
        "tensor_stats": "tensor_stats.jsonl",
        "token_signature_files": token_signature_files,
        "copied_files": copied_files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _resolve_checkpoint_path(teacher_model: str, *, revision: str | None, local_files_only: bool) -> Path:
    path = Path(teacher_model)
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(WEIGHT_POUR_IMPORT_ERROR) from exc
    return Path(
        snapshot_download(
            repo_id=teacher_model,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=[
                "*.safetensors",
                "*.bin",
                "config.json",
                "generation_config.json",
                "tokenizer*",
                "vocab*",
                "merges.txt",
                "*.model",
                "*.json",
            ],
        )
    )


def _iter_checkpoint_tensors(model_path: Path) -> Iterable[tuple[str, np.ndarray, dict[str, Any]]]:
    safetensors = sorted(model_path.glob("*.safetensors"))
    if safetensors:
        try:
            from safetensors import safe_open
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(WEIGHT_POUR_IMPORT_ERROR) from exc
        for shard in safetensors:
            try:
                handle = safe_open(shard, framework="pt", device="cpu")
            except Exception:
                handle = safe_open(shard, framework="np")
            with handle as opened:
                for name in opened.keys():
                    tensor = opened.get_tensor(name)
                    if hasattr(tensor, "detach"):
                        original_dtype = str(tensor.dtype)
                        tensor = tensor.detach().cpu()
                        if "bfloat16" in original_dtype or "float16" in original_dtype:
                            tensor = tensor.float()
                        tensor = tensor.numpy()
                        yield name, np.asarray(tensor), {"shard": shard.name, "original_dtype": original_dtype}
                    else:
                        yield name, np.asarray(tensor), {"shard": shard.name}
        return

    bins = sorted(model_path.glob("pytorch_model*.bin"))
    if not bins:
        raise FileNotFoundError(f"No safetensors or pytorch_model*.bin files found in {model_path}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(WEIGHT_POUR_IMPORT_ERROR) from exc
    for shard in bins:
        state = torch.load(shard, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        for name, tensor in state.items():
            if hasattr(tensor, "detach"):
                original_dtype = str(tensor.dtype)
                tensor = tensor.detach().cpu()
                if "bfloat16" in original_dtype or "float16" in original_dtype:
                    tensor = tensor.float()
                tensor = tensor.numpy()
                yield name, np.asarray(tensor), {"shard": shard.name, "original_dtype": original_dtype}
            else:
                yield name, np.asarray(tensor), {"shard": shard.name}


def _copy_checkpoint_metadata(model_path: Path, out: Path) -> list[str]:
    copied: list[str] = []
    metadata_dir = out / "teacher_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "*.model",
    ]
    for pattern in patterns:
        for src in model_path.glob(pattern):
            if src.is_file():
                dest = metadata_dir / src.name
                shutil.copy2(src, dest)
                copied.append(str(Path("teacher_metadata") / src.name))
    return sorted(set(copied))


def _looks_like_token_matrix(name: str, array: np.ndarray) -> bool:
    lowered = name.lower()
    return array.ndim == 2 and (
        "embed_tokens.weight" in lowered
        or lowered.endswith("wte.weight")
        or lowered.endswith("lm_head.weight")
    )


def _tensor_group(name: str) -> str:
    match = re.search(r"(?:layers|h|blocks)\.(\d+)", name)
    if match:
        return f"layer.{match.group(1)}"
    lowered = name.lower()
    if "embed" in lowered:
        return "embedding"
    if "lm_head" in lowered:
        return "lm_head"
    return name.split(".")[0]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:160]


def _stable_int(text: str) -> int:
    import hashlib

    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _mix_uint64(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))
