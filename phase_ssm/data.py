"""Byte-level corpus loading for the bake-off (text8 by default).

text8 is the canonical small-scale char-LM benchmark: ~100MB of cleaned
Wikipedia. Byte-level + a held-out split gives a clean bits-per-byte number with
no tokenizer confound, so PhaseSSM and the transformer are compared on identical
footing.
"""
from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"


def get_text8(root: str | Path = "data") -> bytes:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "text8"
    if not raw.exists():
        zp = root / "text8.zip"
        if not zp.exists():
            print(f"[data] downloading {TEXT8_URL} ...")
            urllib.request.urlretrieve(TEXT8_URL, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
    return raw.read_bytes()


class ByteData:
    """Holds train/val/test byte arrays and yields random contiguous windows."""

    def __init__(self, data: bytes, *, val_frac: float = 0.05, test_frac: float = 0.05):
        arr = np.frombuffer(data, dtype=np.uint8)
        n = len(arr)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        self.train = torch.from_numpy(arr[: n - n_val - n_test].copy()).long()
        self.val = torch.from_numpy(arr[n - n_val - n_test : n - n_test].copy()).long()
        self.test = torch.from_numpy(arr[n - n_test :].copy()).long()
        self.vocab_size = 256

    def get_batch(self, split: str, batch_size: int, seq_len: int, device: str):
        src = getattr(self, split)
        ix = torch.randint(0, len(src) - seq_len - 1, (batch_size,))
        x = torch.stack([src[i : i + seq_len] for i in ix])
        y = torch.stack([src[i + 1 : i + seq_len + 1] for i in ix])
        if device.startswith("cuda"):
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def eval_bpc(self, model, split: str, batch_size: int, seq_len: int, device: str, iters: int = 50) -> float:
        """Bits-per-byte on a split (mean CE in nats / ln2)."""
        model.eval()
        losses = []
        for _ in range(iters):
            x, y = self.get_batch(split, batch_size, seq_len, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        return float(np.mean(losses) / np.log(2))


def stats(data: ByteData) -> str:
    return (f"train={len(data.train):,}  val={len(data.val):,}  test={len(data.test):,} bytes "
            f"vocab={data.vocab_size}")
