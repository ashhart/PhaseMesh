"""Iterated recall read — the novel mechanism for the post-attention POC.

THESIS
  A fixed-size recurrent state has bounded CAPACITY (number of associations it
  can store) but its SELECTION PRECISION (picking the right stored value among
  collisions) can be sharpened by ITERATING THE READ. A single read of a linear
  associative state is a one-shot linear projection — approximate under load.
  Iterating the read with a learned query-refinement, supervised at every step
  (anytime), buys precision with compute WITHOUT growing the state.

  This is a trade transformers structurally cannot make: their KV cache is
  already exact, so iterating buys them nothing. Our fixed state is approximate,
  so iterating buys us precision. Test-time compute on the RECALL axis — the
  exact axis where fixed-state models lose to attention.

  This is the adaptive-depth idea from serial_depth_report.md REDIRECTED: that
  result showed input-dependent serial depth helps state-TRACKING (S5) but does
  NOT transfer to language (bounded composition depth). The pivot is to apply
  adaptive depth to RECALL SELECTION, which is unbounded and task-dependent.
  Right tool, redirected to the right axis.

MECHANISM
  WRITE once (build associative state S via the linear phase memory). State is
  FIXED after the write pass. Then ITERATE THE READ: refine the query by a
  residual update conditioned on the previous read, re-read against the SAME
  frozen S, repeat for n_refine steps. Every read is supervised (anytime_loss).

  The load-bearing distinction: we iterate the READ, not the WRITE. Iterating
  the write is a DeltaNet settle (known). Iterating the read against a frozen
  state is the novel move. Keep them separate.

WHAT WOULD KILL IT (adaptive_recall_gate.py measures this)
  - refinement converges to read_0 (no gain) -> mechanism is a no-op
  - gain does not scale with load -> not a real recall mechanism
  Gold pattern: single-read drops as P grows; iterated read holds; gap widens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_memory import phase_memory_chunked, phase_memory_quadratic
from .model import RMSNorm


class IteratedReadMemory(nn.Module):
    """Phase-binding memory with an iterated, anytime-supervised READ.

    The write (state S) is built once and FROZEN during refinement. The read is
    iterated: refine query -> re-read -> repeat. n_refine=1 == single-read
    baseline (identical math to PhaseBindingMemory with linear=True).
    """

    def __init__(self, d_model: int, n_heads: int = 4, head_dim: int = 32,
                 n_refine: int = 3, halt: bool = False, chunk: int = 64, linear: bool = False):
        super().__init__()
        self.h = n_heads
        self.dk = head_dim
        self.dv = head_dim
        self.n_refine = max(1, n_refine)
        self.halt = halt
        self.chunk = chunk
        # linear=False (quadratic oracle) is the fast path for the short MQAR
        # sequences in the recall gate; the chunked form is bit-equivalent but
        # ~50x slower at L<~256 due to its constant factor. See poc_common note.
        self.linear = linear

        # Write path (identical to PhaseBindingMemory).
        self.to_q0 = nn.Linear(d_model, n_heads * head_dim)
        self.to_k = nn.Linear(d_model, n_heads * head_dim)
        self.to_v = nn.Linear(d_model, n_heads * self.dv)
        self.out = nn.Linear(n_heads * self.dv, d_model)
        self.decay_logit = nn.Parameter(torch.full((n_heads,), 4.0))   # gamma ~ 0.98

        # Refinement: query phase update conditioned on (q, prev_read).
        self.refine_q = nn.Linear(head_dim + self.dv, head_dim)
        self.refine_norm = RMSNorm(head_dim)

        if halt:
            self.halt_head = nn.Linear(self.dv, 1)

    def _read(self, q_phi, k_phi, v, gamma):
        """One-shot read for all positions against the (implicit) frozen state.

        Quadratic (O(L^2)) by default — fast for the short MQAR sequences. The
        chunked linear form is bit-equivalent but ~50x slower at short L.
        """
        if self.linear:
            return phase_memory_chunked(q_phi, k_phi, v, gamma, chunk=self.chunk)
        return phase_memory_quadratic(q_phi, k_phi, v, gamma)

    def forward(self, x: torch.Tensor) -> dict:
        """Returns {'reads_raw': (K,B,h,L,dv) read at each refine step k}.

        Each read is a valid output; the LM supervises all of them (anytime).
        """
        B, L, _ = x.shape
        q0 = self.to_q0(x).float().view(B, L, self.h, self.dk).permute(0, 2, 1, 3)
        k = self.to_k(x).float().view(B, L, self.h, self.dk).permute(0, 2, 1, 3)
        v = self.to_v(x).float().view(B, L, self.h, self.dv).permute(0, 2, 1, 3)
        gamma = torch.sigmoid(self.decay_logit)

        o = self._read(q0, k, v, gamma)                  # read at step 0
        reads = [o]
        q_cur = q0
        for _ in range(self.n_refine - 1):
            qf = torch.cat([q_cur, o], dim=-1)           # (B,h,L,dk+dv)
            dq = self.refine_norm(self.refine_q(qf))
            q_cur = q_cur + dq
            o = self._read(q_cur, k, v, gamma)           # re-read SAME frozen state
            reads.append(o)
        return {"reads_raw": torch.stack(reads, dim=0)}  # (K,B,h,L,dv)

    def project(self, read_raw: torch.Tensor) -> torch.Tensor:
        """Project a (B,h,L,dv) read to (B,L,d_model)."""
        B, h, L, dv = read_raw.shape
        return self.out(read_raw.permute(0, 2, 1, 3).reshape(B, L, h * dv))


class IteratedReadLM(nn.Module):
    """Minimal LM wrapping IteratedReadMemory. Isolates the recall mechanism.

    Architecture: embed -> N x (RMSNorm -> mem -> +resid) -> RMSNorm -> head.
    Supervises the LAST layer's reads with anytime_loss (geometric weight to the
    final step). n_refine=1 == the single-read baseline (no anytime needed).
    """

    def __init__(self, vocab: int, d_model: int = 192, n_layers: int = 2,
                 n_heads: int = 4, head_dim: int = 32, n_refine: int = 3,
                 halt: bool = False, tie_embeddings: bool = True,
                 anytime_decay: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.n_refine = n_refine
        self.anytime_decay = anytime_decay
        self.embed = nn.Embedding(vocab, d_model)
        self.norms_in = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.mems = nn.ModuleList([
            IteratedReadMemory(d_model, n_heads, head_dim, n_refine, halt)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._last_n_refine = n_refine

    def _logits_from_read(self, read_raw: torch.Tensor, mem: IteratedReadMemory) -> torch.Tensor:
        """Project a raw read to vocab logits: read -> mem.out -> norm_f -> head."""
        h = mem.project(read_raw)              # (B,L,d_model)
        return self.lm_head(self.norm_f(h))

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(ids)
        last_reads = None
        last_mem = None
        for norm, mem in zip(self.norms_in, self.mems):
            out = mem(norm(x))
            x = x + mem.project(out["reads_raw"][-1])     # residual: final read
            last_reads = out["reads_raw"]
            last_mem = mem
        logits = self.lm_head(self.norm_f(x))
        if targets is None:
            return logits, None

        if self.n_refine <= 1:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
            return logits, loss

        # Anytime: supervise every refine step of the last layer.
        K = last_reads.shape[0]
        w = self.anytime_decay ** torch.arange(K - 1, -1, -1, device=last_reads.device).float()
        w = w / w.sum()
        loss = 0.0
        for k in range(K):
            lk = self._logits_from_read(last_reads[k], last_mem)
            ce = F.cross_entropy(lk.reshape(-1, lk.size(-1)),
                                 targets.reshape(-1), ignore_index=-100)
            loss = loss + w[k] * ce
        return logits, loss

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p)); total += p.numel()
        return total


if __name__ == "__main__":
    torch.manual_seed(0)
    V = 64
    # n_refine=3
    m = IteratedReadLM(vocab=V, d_model=128, n_layers=2, n_heads=4, head_dim=32, n_refine=3)
    ids = torch.randint(0, V, (2, 16)); y = torch.randint(0, V, (2, 16))
    logits, loss = m(ids, y)
    print(f"n_refine=3: logits {tuple(logits.shape)} loss {loss.item():.3f} "
          f"params {m.num_params()/1e3:.1f}K")
    # n_refine=1 (single-read baseline)
    m1 = IteratedReadLM(vocab=V, d_model=128, n_layers=2, n_heads=4, head_dim=32, n_refine=1)
    logits1, loss1 = m1(ids, y)
    print(f"n_refine=1: logits {tuple(logits1.shape)} loss {loss1.item():.3f} "
          f"params {m1.num_params()/1e3:.1f}K")
    print("OK")
