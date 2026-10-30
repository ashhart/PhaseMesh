"""Peak memory and device sync for CUDA, Apple MPS and CPU, so benches report memory off-CUDA too.

CUDA has an exact allocator high-water mark. MPS and CPU do not, so for those a background thread samples
the MPS tensor allocator or the process RSS (CPU, needs psutil) every millisecond. The MPS driver counter is
not used: Metal hands out memory from large heaps, too coarse to show a KV cache. Sampled peaks can miss a
spike shorter than the interval. On MPS that includes short-lived temporaries such as the PhaseSSM FFT kernel:
the CPU thread allocates and frees them as it enqueues work, faster than the sampler, so MPS numbers are good
for long-lived buffers (KV cache, recurrent state) and not for prefill temporaries. Every result carries the
method that produced it:

* ``cuda_max_allocated``: exact, ``torch.cuda.max_memory_allocated``.
* ``mps_allocated_sampled``: ``torch.mps.current_allocated_memory``, sampled.
* ``cpu_rss_sampled``: process resident set size, sampled (includes the Python process itself).
* ``unavailable``: CPU without psutil; peak is ``None``.

``peak_gb`` is the high-water mark during the block; ``added_gb`` is that minus the reading on entry,
which is what a growing KV cache or activation buffer shows up as.

    with PeakMemory(device) as mem:
        model(x)
    mem.result()  # {"peak_gb": ..., "added_gb": ..., "mem_method": ...}
"""
from __future__ import annotations

import threading
from typing import Callable

import torch

SAMPLE_INTERVAL_S = 0.001
GB = 1e9


def device_type(device: str | torch.device) -> str:
    return torch.device(device).type


def sync(device: str | torch.device) -> None:
    """Wait for queued kernels so wall-clock timing covers the work."""
    kind = device_type(device)
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cpu_reader() -> Callable[[], int] | None:
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process()
    return lambda: process.memory_info().rss


class PeakMemory:
    """Context manager measuring peak memory on one device during the block."""

    def __init__(self, device: str | torch.device, interval_s: float = SAMPLE_INTERVAL_S):
        self.kind = device_type(device)
        self.interval_s = interval_s
        self._reader: Callable[[], int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline = 0
        self.peak: int | None = None
        if self.kind == "cuda":
            self.method = "cuda_max_allocated"
        elif self.kind == "mps":
            self.method, self._reader = "mps_allocated_sampled", torch.mps.current_allocated_memory
        else:
            self._reader = _cpu_reader()
            self.method = "cpu_rss_sampled" if self._reader else "unavailable"

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak or 0, self._reader())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PeakMemory":
        if self.kind == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.baseline = torch.cuda.memory_allocated()
        elif self._reader is not None:
            sync(self.kind)
            self.baseline = self.peak = self._reader()
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        sync(self.kind)
        if self.kind == "cuda":
            self.peak = torch.cuda.max_memory_allocated()
        elif self._thread is not None:
            self._stop.set()
            self._thread.join()
            self.peak = max(self.peak or 0, self._reader())

    def result(self) -> dict:
        if self.peak is None:
            return {"peak_gb": None, "added_gb": None, "mem_method": self.method}
        return {"peak_gb": self.peak / GB, "added_gb": (self.peak - self.baseline) / GB, "mem_method": self.method}
