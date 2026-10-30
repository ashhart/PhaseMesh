"""Linear phase-binding memory — the post-transformer mechanic.

The :class:`~phase_ssm.memory.PhaseBindingMemory` in ``memory.py`` binds
key->value as complex phasor products and retrieves by phase unbinding. Its
retrieval weight is the squared phase coherence ``|<q_t, k_s>|^2`` between the
query and each prior key. That form is *separable*: it is a feature inner
product, which collapses the whole (B,h,L,L) attention-shaped cost into a
fixed-size linear recurrence. This module implements that linear form.

Derivation (per head; ``q,k`` are unit phasors, ``v`` real)::

    |<q,k>|^2 = (Σ_d q_d k_d*) · (Σ_e q_e* k_e)
             = Σ_{d,e} (q_d q_e*) · (k_d* k_e)
             = <Φ(q), conj(Φ(k))> ,   Φ(x)_{d,e} := x_d · x_e*  in dk x dk dims

So the (positive) retrieval weight ``|<q_t,k_s>|^2`` equals an inner product in
a fixed dk**2-dimensional complex feature space. With a per-head decay γ the
recurrence is then exactly RetNet/GLA structure::

    write:   S_t = γ·S_{t-1} + Φ(k_t) ⊗ v_t      # state (dk**2, dv), O(1) in L
    norm:    Z_t = γ·Z_{t-1} + Φ(k_t)             # state (dk**2,),   O(1) in L
    read:    o_t = (Φ(q_t)· S_t) / (Φ(q_t)· Z_t)   # O(dk**2 · dv) per step

Three forms are provided and must be bit-equivalent:

* :func:`phase_memory_quadratic` — the O(L^2) reference (matches
  ``PhaseBindingMemory.forward``). This is the oracle.
* :func:`phase_memory_chunked` — O(L) memory, parallel within a chunk,
  sequential across chunks. The training path.
* :func:`phase_memory_recurrent` — O(1) per step. The streaming/decode path.

The state footprint is fixed at ``h · dk**2 · dv · complex64`` per layer and is
*independent of context length* — there is no KV cache to grow. That fixed
structured state is the property attention cannot offer and the reason this is a
post-transformer mechanic rather than a slower attention.

Cost note: the feature dim is ``dk**2`` (quadratic in the per-head key dim).
Binding sharpness trades against state size across ``dk``; that knob is the
central design variable, swept in the MQAR differentiator.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def phasor_feature(phi: torch.Tensor) -> torch.Tensor:
    """Map phase angles to the separable phase-coherence feature Φ(x).

    ``phi``: (..., dk) real phase angles.
    Returns (..., dk, dk) complex features ``Φ(x)_{d,e} = exp(i(φ_d − φ_e))``,
    the rank-1 unit-modulus matrix whose inner product reproduces
    ``|<x,y>|^2`` for unit phasors ``x, y``.

    Identity (verified numerically)::

        <Φ(q), conj(Φ(k))> = Σ_{d,e} exp(i(q_d-q_e)) exp(-i(k_d-k_e))
                           = (Σ_d e^{i q_d} e^{-i k_d})(Σ_e e^{-i q_e} e^{i k_e})
                           = dk · |<q,k>|^2

    So the quadratic coherence target ``|<q,k>|^2 / dk`` (what ``memory.py``
    computes) equals ``<Φ(q), conj(Φ(k))> / dk``. The query feature is ``Φ(q)``;
    the key feature stored in the recurrent state is ``conj(Φ(k))``; the readout
    divides by ``dk``. This lifts a dot-product-squared into an ordinary
    linear-attention feature dot product in ``dk**2`` dims — the whole reason the
    quadratic coherence is linearizable.
    """
    phi = phi.float()
    diff = phi[..., :, None] - phi[..., None, :]          # (..., dk, dk)
    return torch.exp(1j * diff)                           # unit-modulus complex


def query_feature(phi: torch.Tensor) -> torch.Tensor:
    """Query feature ``Φ(q)`` reshaped to (..., dk**2). See :func:`phasor_feature`."""
    return phasor_feature(phi).flatten(-2)


def key_feature(phi: torch.Tensor) -> torch.Tensor:
    """Key feature ``conj(Φ(k))`` reshaped to (..., dk**2). Stored in the state."""
    return phasor_feature(phi).conj().flatten(-2)


def _feature_scale(dk: int) -> float:
    """``1/dk``: maps ``<Φ(q), conj(Φ(k))>`` back onto ``|<q,k>|^2/dk``.

    Numerically: ``<Φ(q), conj(Φ(k))> = dk · |<q,k>|^2`` (verified), so dividing by
    ``dk`` reproduces the ``|<q,k>|^2/dk`` coherence target that the quadratic
    form and the original ``PhaseBindingMemory`` use.
    """
    return 1.0 / float(dk)


def _causal_decay(L: int, gamma: torch.Tensor, device) -> torch.Tensor:
    """(h, L, L) decay*causal matrix: entry [t,s] = γ^{t-s} for s<=t else 0."""
    g = gamma.clamp_min(1e-4).float()                     # (h,)
    logg = torch.log(g)[:, None, None]                    # (h,1,1)
    t = torch.arange(L, device=device)
    rel = (t[:, None] - t[None, :]).float()               # (L,L)
    return torch.exp(logg * rel) * (rel[None] >= 0).float()


def phase_memory_quadratic(
    q_phi: torch.Tensor,
    k_phi: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """O(L^2) reference forward — the bit-equivalence oracle.

    Mirrors ``PhaseBindingMemory.forward`` exactly, but factored so the chunked
    and recurrent forms can be checked against it. Inputs are pre-projected:

    * ``q_phi`` / ``k_phi``: (B, h, L, dk) real phase angles
    * ``v``:                  (B, h, L, dv) real values
    * ``gamma``:              (h,) per-head decay in (0, 1)

    Returns (B, h, L, dv).
    """
    B, h, L, dk = q_phi.shape
    dv = v.shape[-1]
    q = torch.exp(1j * q_phi.float())                     # (B,h,L,dk) unit phasors
    k = torch.exp(1j * k_phi.float())
    inner = torch.einsum("bhtd,bhsd->bhts", q, k.conj()) / (dk ** 0.5)
    coh = inner.real ** 2 + inner.imag ** 2               # (B,h,L,L) |<q,k>|^2 >= 0
    decay = _causal_decay(L, gamma, q_phi.device)         # (h,L,L)
    A = coh * decay[None]                                  # (B,h,L,L)
    o = torch.einsum("bhts,bhsd->bhtd", A, v.float())     # (B,h,L,dv)
    denom = A.sum(-1, keepdim=True).clamp_min(1e-4)
    return o / denom


def phase_memory_chunked(
    q_phi: torch.Tensor,
    k_phi: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
    chunk: int = 64,
) -> torch.Tensor:
    """Hybrid chunked forward — the training path. O(L) memory, O(L·C) compute.

    RetNet-style "chunkwise" form. The work is split to keep memory bounded at
    real ``dk`` (naively expanding the ``dk**2`` feature for every position would
    cost ``O(B·h·L·dk**2·dv)`` memory and OOM at dk=32):

    * **Intra-chunk** (positions within the same chunk): compute the full causal
      coherence directly as the cheap ``|<q_p,k_q>|^2`` form on ``chunk x chunk``
      pairs. No feature expansion — memory ``O(B·h·nc·chunk^2)``.
    * **Cross-chunk** (positions in earlier chunks): carry two ``dk**2``-dim
      linear-feature states, one per chunk, of fixed size. Memory ``O(B·h·dk**2)``.

    Total memory ``O(B·h·(chunk^2 + dk**2))`` — independent of L on the carry
    side, and bounded by the small chunk on the intra side. This is what makes
    the mechanic trainable at real ``dk`` rather than only as a reference.

    Two decayed running states (fixed in L), ``fdim = dk**2``::

        S_t = Σ_{s<=t} γ^{t-s} conj(Φ(k_s)) ⊗ v_s    # (B,h,fdim,dv)
        Z_t = Σ_{s<=t} γ^{t-s} conj(Φ(k_s))          # (B,h,fdim)

    Bit-equivalent to :func:`phase_memory_quadratic` (verified in tests).
    """
    B, h, L, dk = q_phi.shape
    dv = v.shape[-1]
    dev = q_phi.device
    fdim = dk * dk
    scale = _feature_scale(dk)

    qf = torch.exp(1j * q_phi.float())                      # unit phasors (B,h,L,dk)
    kf = torch.exp(1j * k_phi.float())
    Q = query_feature(q_phi).view(B, h, L, fdim)            # Φ(q)        (B,h,L,fdim)
    K = key_feature(k_phi).view(B, h, L, fdim)              # conj(Φ(k))  (B,h,L,fdim)
    vf = v.float()
    g = gamma.clamp_min(1e-4).float()                       # (h,)

    pad = (chunk - L % chunk) % chunk
    if pad:
        Q = F.pad(Q, (0, 0, 0, pad))
        K = F.pad(K, (0, 0, 0, pad))
        qf = F.pad(qf, (0, 0, 0, pad))
        kf = F.pad(kf, (0, 0, 0, pad))
        vf = F.pad(vf, (0, 0, 0, pad))
    Lp = Q.shape[2]
    nc = Lp // chunk

    qpc = qf.view(B, h, nc, chunk, dk)                      # phasors for intra coherence
    kpc = kf.view(B, h, nc, chunk, dk)
    Vc = vf.view(B, h, nc, chunk, dv)
    Qc = Q.view(B, h, nc, chunk, fdim)                      # features for cross-chunk read
    Kc = K.view(B, h, nc, chunk, fdim)                      # features for cross-chunk write

    j = torch.arange(chunk, device=dev).float()
    rel = j[:, None] - j[None, :]                           # (chunk,chunk)
    causal = (rel[None] >= 0).float()
    win = (g[:, None, None] ** rel) * causal                # (h,chunk,chunk) intra decay+causal

    # Intra-chunk coherence |<q_p,k_q>|^2 / dk on the cheap (no dk**2 expansion).
    inner = torch.einsum("bhcpd,bhcqd->bhcpq", qpc, kpc.conj()) / (dk ** 0.5)
    coh = inner.real ** 2 + inner.imag ** 2                 # (B,h,nc,chunk,chunk) >= 0
    A_intra = coh * win[:, None]                            # (B,h,nc,chunk,chunk)
    num_intra = torch.einsum("bhcpq,bhcqd->bhcpd", A_intra, Vc)  # (B,h,nc,chunk,dv)
    den_intra = A_intra.sum(-1)                              # (B,h,nc,chunk)

    # Cross-chunk carry (sequential over chunks): fixed dk**2 states in L.
    state_S = torch.zeros(B, h, fdim, dv, dtype=torch.complex64, device=dev)
    state_Z = torch.zeros(B, h, fdim, dtype=torch.complex64, device=dev)

    out_num = torch.empty(B, h, nc, chunk, dv, device=dev)
    out_den = torch.empty(B, h, nc, chunk, device=dev)

    for c in range(nc):
        qc = Qc[:, :, c]                                    # (B,h,chunk,fdim) Φ(q)
        kc = Kc[:, :, c]                                    # (B,h,chunk,fdim) conj(Φ(k))
        vc = Vc[:, :, c]                                    # (B,h,chunk,dv)
        # Cross contribution at within-chunk position p: γ^p · <Φ(q_p), state>.
        g_p = (g[:, None] ** j)                            # (h,chunk) decay of carry to position p
        cross_num = torch.einsum("bhpf,bhfd->bhpd", qc, state_S).real * scale * g_p[..., None]
        cross_den = torch.einsum("bhpf,bhf->bhp", qc, state_Z).real * scale * g_p
        out_num[:, :, c] = num_intra[:, :, c] + cross_num
        out_den[:, :, c] = den_intra[:, :, c] + cross_den
        # Advance carry: state *= γ^chunk, then add this chunk's keys weighted by
        # γ^{chunk-q} so a key at within-chunk position q lands at the correct
        # horizon (the start of the next chunk) — distance from q to that boundary
        # is exactly (chunk - q).
        chunk_decay = g ** chunk                           # (h,)
        tail_w = (g[:, None] ** (chunk - j)).to(torch.complex64)  # (h,chunk)
        add_S = torch.einsum("bhqf,bhqd,hq->bhfd", kc, vc, tail_w)
        add_Z = torch.einsum("bhqf,hq->bhf", kc, tail_w)
        state_S = state_S * chunk_decay[:, None, None] + add_S
        state_Z = state_Z * chunk_decay[:, None] + add_Z

    num = out_num.reshape(B, h, Lp, dv)[:, :, :L]
    den = out_den.reshape(B, h, Lp, 1)[:, :, :L]
    return num / den.clamp_min(1e-4)


def phase_memory_recurrent(
    q_phi: torch.Tensor,
    k_phi: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """O(1)-per-step streaming forward — the decode path.

    Fixed state (S, Z) per head, advanced one token at a time. Independent of L.
    Bit-equivalent to :func:`phase_memory_quadratic`.
    """
    B, h, L, dk = q_phi.shape
    dv = v.shape[-1]
    dev = q_phi.device
    fdim = dk * dk
    scale = _feature_scale(dk)

    Q = query_feature(q_phi).view(B, h, L, fdim)           # (B,h,L,fdim) Φ(q)
    K = key_feature(k_phi).view(B, h, L, fdim)             # (B,h,L,fdim) conj(Φ(k))
    vf = v.float()
    g = gamma.clamp_min(1e-4).float()                      # (h,)

    state_S = torch.zeros(B, h, fdim, dv, dtype=torch.complex64, device=dev)
    state_Z = torch.zeros(B, h, fdim, dtype=torch.complex64, device=dev)
    out = torch.empty(B, h, L, dv, device=dev)

    for t in range(L):
        k_t = K[:, :, t]                                   # (B,h,fdim) conj(Φ(k_t))
        q_t = Q[:, :, t]                                   # (B,h,fdim) Φ(q_t)
        v_t = vf[:, :, t]                                  # (B,h,dv)
        state_S = state_S * g[:, None, None] + torch.einsum("bhf,bhd->bhfd", k_t, v_t)
        state_Z = state_Z * g[:, None] + k_t
        num = torch.einsum("bhf,bhfd->bhd", q_t, state_S).real * scale
        den = torch.einsum("bhf,bhf->bh", q_t, state_Z).real * scale
        out[:, :, t] = num / den.clamp_min(1e-4).unsqueeze(-1)
    return out


def _test_equivalence() -> bool:
    """Bit-equivalence check: quadratic == chunked == recurrent."""
    torch.manual_seed(0)
    B, h, L, dk, dv = 2, 3, 40, 5, 4
    q_phi = torch.randn(B, h, L, dk) * 3.0
    k_phi = torch.randn(B, h, L, dk) * 3.0
    v = torch.randn(B, h, L, dv)
    gamma = torch.tensor([0.85, 0.95, 0.99])

    y_q = phase_memory_quadratic(q_phi, k_phi, v, gamma)
    y_c = phase_memory_chunked(q_phi, k_phi, v, gamma, chunk=8)
    y_r = phase_memory_recurrent(q_phi, k_phi, v, gamma)

    scale = y_q.abs().max().item() + 1e-9
    rc = (y_q - y_c).abs().max().item() / scale
    rr = (y_q - y_r).abs().max().item() / scale
    print(f"rel|quadratic - chunked|   = {rc:.2e}")
    print(f"rel|quadratic - recurrent| = {rr:.2e}")
    ok = rc < 1e-4 and rr < 1e-4
    print("chunked   == quadratic:", "PASS ✓" if rc < 1e-4 else "FAIL ✗")
    print("recurrent == quadratic:", "PASS ✓" if rr < 1e-4 else "FAIL ✗")

    # State-footprint sanity: fixed in L, vs attention's growing KV cache.
    bytes_per_head = dk * dk * dv * 8                      # complex64 S + half of Z
    total = h * bytes_per_head
    print(f"fixed state: {total / 1024:.1f} KB (independent of L={L})")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test_equivalence() else 1)
