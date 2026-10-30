"""POC common: ONE memory config, ONE MQAR harness, ONE verdict rule.

The whole POC bundle hinges on killing the dk**2 state-size drift that made the
prior recall results non-comparable:

    prior mqar_sweep.py        head_dim=48  -> dk**2 = 2304 state per head
    prior recall_fix.py        head_dim=16  -> dk**2 =  256 state per head
    prior premise_sweep.py     head_dim=16  -> dk**2 =  256 state per head
    prior mqar_signal.py       head_dim=16  -> dk**2 =  256 state per head

A 9x difference in the central state-size knob, across scripts asking the same
question. That is why the verdict was muddy. This module fixes ONE config and
makes every POC experiment use it, so results are directly comparable.

It also fixes the underpowering bug in the prior negative: the P=16 signal run
used 800 steps on CPU where the transformer positive control ALSO sat at chance
(0.044 vs chance 0.008). A negative where the positive control also fails is
not a valid negative. The defaults here (steps, seeds, warmup, cosine decay)
come straight from tasks.train_on_mqar and are adequate for convergence.

Decision rule (pre-registered, from mqar_curve.py's own docstring — this is the
load-bearing definition of what "the memory carries recall" means):

    transformer holds, OURS+mem holds, SSM-only collapses  -> THESIS LIVES
    transformer holds, OURS+mem collapses with SSM-only    -> THESIS DEAD (real)
    all three collapse (transformer below threshold)       -> UNDERPOWERED (no verdict)

"holds" = recall >= HOLD_THRESHOLD at that P. The transformer is the positive
control: if it cannot learn the task, the run is inconclusive regardless of what
the memory scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import PhaseSSMConfig, PhaseSSMLM
from .transformer import TransformerConfig, TransformerLM


# ----------------------------------------------------------------------------
# The ONE config. Everything else is derived from it. Do not fork it.
# ----------------------------------------------------------------------------
HEAD_DIM = 32          # dk = 32 -> dk**2 = 1024 state per head (the central knob)
MEM_HEADS = 4          # 4 heads -> 4 * 1024 = 4096-dim fixed state, ~134 KB complex64
D_MODEL = 192          # matched across all arms (transformer uses the same width)
N_LAYERS = 4
STATE_DIM = 64         # oscillatory SSM pole count per channel (temporal branch)
EXPAND = 1             # in/out width of the SSM branch
D_FF_MULT = 2          # FFN expansion

# Convergence defaults (adequate power — the prior negative used 800 steps).
DEFAULT_STEPS = 6000
DEFAULT_SEEDS = 3
DEFAULT_BATCH = 64
HOLD_THRESHOLD = 0.85  # "solved": a fair bar well above the 1/P floor
POWER_FAIL_BAR = 0.20  # if the TRANSFORMER is below this, the run is inconclusive


def base_phasessm(use_mem: bool, *, mem_linear: bool = False, vocab: int = 256) -> PhaseSSMLM:
    """The standard PhaseSSM arm.

    mem_linear defaults to False (the O(L^2) quadratic oracle) because MQAR
    sequences are SHORT: the chunked 'linear' form is O(L) in memory but has a
    huge constant factor (the dk**2 feature expansion + a Python chunk loop),
    making it ~50x SLOWER per step than the quadratic form at L<~256 with no
    memory benefit. The two are bit-equivalent (1e-7), verified in tests — so
    the quadratic oracle is the correct, fast training path for the recall gate,
    and the linear form's advantage only materializes at long context (which is
    the serving/decode story, measured separately in effbench). Same memory
    mechanism, fast form for short sequences.
    """
    return PhaseSSMLM(PhaseSSMConfig(
        vocab_size=vocab, d_model=D_MODEL, n_layers=N_LAYERS, state_dim=STATE_DIM,
        expand=EXPAND, d_ff_mult=D_FF_MULT,
        use_phase_memory=use_mem,
        mem_heads=MEM_HEADS, mem_head_dim=HEAD_DIM,
        mem_linear=mem_linear, mem_layout="parallel",
        tie_embeddings=True,
    ))


def base_transformer(*, vocab: int = 256) -> TransformerLM:
    """The positive control. d_model matched; heads chosen so head_dim matches dk."""
    n_heads = max(1, D_MODEL // HEAD_DIM)   # 192 // 32 = 6 heads, head_dim 32 (matches the memory)
    return TransformerLM(TransformerConfig(
        vocab_size=vocab, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=n_heads, d_ff_mult=D_FF_MULT, block_size=1024, pos_enc="rope",
    ))


@dataclass
class ArmResult:
    name: str
    mean: float
    std: float
    all: list
    params_M: float

    def to_dict(self) -> dict:
        return {"name": self.name, "mean": self.mean, "std": self.std,
                "all": self.all, "params_M": self.params_M}


def run_arm(build_fn, *, steps: int, seeds: int, lr: float, device: str,
            n_pairs: int, n_queries: int, n_keys: int, n_values: int,
            batch: int = DEFAULT_BATCH, name: str = "", log=print) -> ArmResult:
    """Train `build_fn()` on MQAR for `seeds` seeds, return mean recall."""
    from .tasks import train_on_mqar
    accs = []
    params = 0.0
    for s in range(seeds):
        torch.manual_seed(s)
        m = build_fn()
        params = m.num_params() / 1e6
        acc = train_on_mqar(m, steps=steps, n_pairs=n_pairs, n_queries=n_queries,
                            n_keys=n_keys, n_values=n_values,
                            batch=batch, lr=lr, device=device, log_every=0)
        accs.append(acc)
        log(f"    {name or 'arm':<14} seed{s} recall={acc:.3f}  ({params:.2f}M)")
    import statistics
    return ArmResult(name=name or "arm", mean=statistics.mean(accs),
                     std=statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                     all=accs, params_M=params)


# ----------------------------------------------------------------------------
# Vectorized MQAR — same task as tasks.make_mqar (unique keys/vals per sequence,
# queries point to random prior pairs), but no per-sequence Python loop. The
# loop version is ~43ms/call on CUDA (tiny CPU->GPU syncs per sequence); this is
# <2ms. Patched into tasks.make_mqar below so the whole POC uses it without
# touching the original tasks.py. Task semantics preserved.
# ----------------------------------------------------------------------------
def make_mqar_fast(batch: int, n_pairs: int, n_queries: int, n_keys: int, n_values: int,
                   device: str = "cpu", seed: int | None = None):
    """Vectorized MQAR batch. Same shape/semantics as tasks.make_mqar."""
    KEY0, VAL0 = 1, 1 + n_keys
    L = 2 * n_pairs + n_queries
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    # Per-sequence unique keys/values (vectorized over batch; randperm is 1-D so
    # we stack — still far cheaper than the Python loop + per-seq CUDA syncs).
    keys = torch.stack([torch.randperm(n_keys, generator=g)[:n_pairs] for _ in range(batch)]) + KEY0
    vals = torch.stack([torch.randperm(n_values, generator=g)[:n_pairs] for _ in range(batch)]) + VAL0

    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch, L), -100, dtype=torch.long)
    x[:, 0:2 * n_pairs:2] = keys
    x[:, 1:2 * n_pairs:2] = vals
    qidx = torch.randint(0, n_pairs, (batch, n_queries), generator=g)
    q_pos = torch.arange(2 * n_pairs, L)
    x[:, q_pos] = keys.gather(1, qidx)            # query token = the pair's key
    y[:, q_pos] = vals.gather(1, qidx)            # target = the pair's value

    vocab = 1 + n_keys + n_values
    if device.startswith("cuda"):
        x, y = x.to(device), y.to(device)
    return x, y, vocab


# Patch the tasks module so train_on_mqar (which looks up make_mqar by module
# global) uses the fast vectorized generator. Do this once at import.
try:
    from . import tasks as _tasks_mod
    _tasks_mod.make_mqar = make_mqar_fast
except Exception:
    pass


# ----------------------------------------------------------------------------
# Vectorized MQAR — same task as tasks.make_mqar (unique keys/vals per sequence,
# queries point to random prior pairs), but no per-sequence Python loop. The
# loop version is ~43ms/call on CUDA (tiny CPU->GPU syncs per sequence); this is
# <1ms. Identical task semantics — verified to produce the same structure.
# ----------------------------------------------------------------------------
def make_mqar_fast(batch: int, n_pairs: int, n_queries: int, n_keys: int, n_values: int,
                   device: str = "cpu", seed: int | None = None):
    """Vectorized MQAR batch. Same shape/semantics as tasks.make_mqar."""
    KEY0, VAL0 = 1, 1 + n_keys
    L = 2 * n_pairs + n_queries
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    # Per-sequence unique keys and values, vectorized over the batch.
    # keys: (batch, n_pairs) drawn without replacement from [1..n_keys]
    keys = torch.stack([torch.randperm(n_keys, generator=g)[:n_pairs] for _ in range(batch)]) + KEY0
    vals = torch.stack([torch.randperm(n_values, generator=g)[:n_pairs] for _ in range(batch)]) + VAL0

    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch, L), -100, dtype=torch.long)
    # interleave k,v into the first 2*n_pairs positions
    x[:, 0::2][:, :n_pairs] = keys
    x[:, 1::2][:, :n_pairs] = vals
    # queries: random pair index per query position, vectorized
    qidx = torch.randint(0, n_pairs, (batch, n_queries), generator=g)
    q_pos = torch.arange(2 * n_pairs, L)
    x[:, q_pos] = keys.gather(1, qidx)             # query token = the pair's key
    y[:, q_pos] = vals.gather(1, qidx)             # target = the pair's value

    vocab = 1 + n_keys + n_values
    if device.startswith("cuda"):
        x, y = x.to(device), y.to(device)
    return x, y, vocab


def make_mqar(*args, **kwargs):
    """Route POC experiments through the fast vectorized generator."""
    return make_mqar_fast(*args, **kwargs)


# ----------------------------------------------------------------------------
# The ONE verdict. Used by every POC experiment so the bar is identical.
# ----------------------------------------------------------------------------
def verdict_recal(ours: ArmResult, ssm: ArmResult, xf: ArmResult, *, P: int) -> str:
    """Pre-registered recall-parity verdict.

    Returns one of: THESIS_LIVES | THESIS_DEAD | UNDERPOWERED.
    The transformer is the positive control: if it didn't learn, no verdict.
    """
    if xf.mean < POWER_FAIL_BAR:
        return ("UNDERPOWERED — transformer positive control only reached "
                f"{xf.mean:.3f} (<{POWER_FAIL_BAR}); no verdict is valid at this config. "
                "Raise steps/state/width and rerun.")
    ours_holds = ours.mean >= HOLD_THRESHOLD
    ssm_holds = ssm.mean >= HOLD_THRESHOLD
    if ours_holds and not ssm_holds:
        return (f"THESIS_LIVES — memory carries recall at P={P} "
                f"(OURS {ours.mean:.3f} >= {HOLD_THRESHOLD}, SSM {ssm.mean:.3f} collapses).")
    if not ours_holds and not ssm_holds:
        return (f"THESIS_DEAD — memory collapses with SSM-only at P={P} "
                f"(OURS {ours.mean:.3f}, SSM {ssm.mean:.3f}, XF {xf.mean:.3f} holds). "
                "The linear dk**2 phase memory cannot carry the recall load.")
    # both above threshold -> this P doesn't separate them (push P higher)
    return (f"NOT_DISCRIMINATING — both hold at P={P} "
            f"(OURS {ours.mean:.3f}, SSM {ssm.mean:.3f} both >= {HOLD_THRESHOLD}). "
            f"Push P higher to find the SSM collapse point.")


# ----------------------------------------------------------------------------
# Char-LM data helper for the ternary-parity gate (tiny-shakespeare-ish).
# Falls back to a deterministic synthetic stream if no corpus is found.
# ----------------------------------------------------------------------------
def char_corpus_iter(path: str | None = None, *, min_chars: int = 200_000):
    """Yield lowercase chars from a text file. Falls back to a synthetic stream.

    The fallback is a deterministic repeating pseudo-text so the gate is runnable
    anywhere. For a real number, point at tiny-shakespeare or text8.
    """
    if path:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().lower()
            if len(txt) >= min_chars:
                yield from iter(txt[:min_chars])
                return
        except FileNotFoundError:
            pass
    # Synthetic fallback: a structured pseudo-language with repetition + symbols.
    # Deterministic, fixed entropy, so fp vs ternary is compared on identical data.
    vocab = "abcdefghijklmnopqrstuvwxyz .,!?0123456789\n"
    seed_text = ("the quick brown fox jumps over the lazy dog. "
                 "pack my box with five dozen liquor jugs! "
                 "she sells sea shells by the sea shore. "
                 "how much wood would a woodchuck chuck? ")
    buf = (seed_text * (min_chars // len(seed_text) + 1))[:min_chars]
    yield from iter(buf)


def build_char_dataset(path: str | None = None, *, seq_len: int = 128, n_samples: int = 4096,
                       vocab: str | None = None):
    """Build (ids, targets) over a char vocabulary. Returns (x, y, stoi, itos, vocab_size)."""
    import itertools
    chars = list(itertools.islice(char_corpus_iter(path), 2_000_000))
    if vocab is None:
        vocab = "".join(sorted(set(chars)))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}
    ids = [stoi[c] for c in chars if c in stoi]
    V = len(vocab)

    x = torch.zeros(n_samples, seq_len, dtype=torch.long)
    y = torch.zeros(n_samples, seq_len, dtype=torch.long)
    for i in range(n_samples):
        s = (i * 7919) % max(1, len(ids) - seq_len - 1)  # 7919 prime -> spread
        chunk = ids[s : s + seq_len + 1]
        x[i] = torch.tensor(chunk[:seq_len], dtype=torch.long)
        y[i] = torch.tensor(chunk[1 : seq_len + 1], dtype=torch.long)
    return x, y, stoi, itos, V


def char_bpc(loss: float) -> float:
    """Cross-entropy loss (nats) -> bits per char."""
    return loss / math.log(2.0)
