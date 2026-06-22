from __future__ import annotations

import json
import os
import resource
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from phase_mesh import CognitiveMeshRuntime, MeshConfig

T = TypeVar("T")


def make_runtime(
    *,
    size: int,
    steps: int,
    seed: int,
    backend: str = "auto",
    pin_strength: float = 0.0,
    residual_carry: float = 0.08,
) -> CognitiveMeshRuntime:
    return CognitiveMeshRuntime(
        MeshConfig(
            width=size,
            height=size,
            max_steps=steps,
            seed=seed,
            laplacian_backend=backend,
            phase_pin_strength=pin_strength,
            phase_residual_carry=residual_carry,
        )
    )


def timed(func: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = func()
    return result, time.perf_counter() - start


@dataclass(frozen=True)
class Measurement:
    elapsed_s: float
    rss_start_bytes: int | None
    rss_end_bytes: int | None
    rss_peak_bytes: int | None
    max_rss_bytes: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "elapsed_s": self.elapsed_s,
            "rss_start_bytes": self.rss_start_bytes,
            "rss_end_bytes": self.rss_end_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
            "max_rss_bytes": self.max_rss_bytes,
        }


class MemorySampler:
    """Samples process RSS while a benchmark call runs.

    psutil is optional. Without it we still report OS max RSS, but live peak
    samples are unavailable on some platforms.
    """

    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []
        self.start_rss = current_rss_bytes()

    def __enter__(self) -> MemorySampler:
        if self.start_rss is not None:
            self.samples.append(self.start_rss)
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        end = current_rss_bytes()
        if end is not None:
            self.samples.append(end)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            rss = current_rss_bytes()
            if rss is not None:
                self.samples.append(rss)

    @property
    def end_rss(self) -> int | None:
        return self.samples[-1] if self.samples else current_rss_bytes()

    @property
    def peak_rss(self) -> int | None:
        return max(self.samples) if self.samples else None


def measured(func: Callable[[], T]) -> tuple[T, Measurement]:
    start = time.perf_counter()
    with MemorySampler() as sampler:
        result = func()
    elapsed = time.perf_counter() - start
    return result, Measurement(
        elapsed_s=elapsed,
        rss_start_bytes=sampler.start_rss,
        rss_end_bytes=sampler.end_rss,
        rss_peak_bytes=sampler.peak_rss,
        max_rss_bytes=max_rss_bytes(),
    )


def count_mesh_flops(grid_size: int, steps_used: int, neighbors: int = 4) -> int:
    """Count deterministic phase-kernel operations for a square mesh."""

    return int(steps_used * (grid_size**2 * neighbors * 8))


def count_verifier_flops(prompt: str, target_len: int = 8) -> int:
    """Count cheap verifier work used by benchmark scoring/control paths.

    This is a deterministic harness counter for AST parsing and simple local
    evaluation, not a CPU profiler trace.
    """

    token_count = max(1, int(target_len), len(str(prompt).split()))
    return int(120 + token_count * 24)


def count_llama_flops(model: Any, input_len: int) -> int:
    """Count parameter MACs touched by a transformer forward sequence.

    This is an auditable model-size counter, not a CUDA profiler trace. It is
    stricter than a hand-entered reference number because it walks the actual
    loaded model parameters and multiplies weight counts by sequence length.
    """

    macs = 0
    for parameter in model.parameters():
        macs += int(parameter.numel()) * int(input_len)
    return macs


def count_flops(model: Any, input_ids: Any) -> int:
    """Backward-compatible alias for transformer MAC counting."""

    return count_llama_flops(model, int(input_ids.shape[1]))


def current_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value * 1024)


def write_result(out_dir: str | Path, name: str, payload: dict[str, Any]) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"{name}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def append_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return output


def summarize_numbers(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def summarize_distribution(values: list[float]) -> dict[str, float]:
    summary = summarize_numbers(values)
    if not values:
        return {**summary, "median": 0.0}
    return {**summary, "median": float(statistics.median(values))}
