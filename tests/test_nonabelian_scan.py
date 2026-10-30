"""Rung A unit tests: the algebra must be right before the run means anything.

1. Parallel scan == sequential recurrence (full + diagonal), non-power-of-2 L.
2. Householder products express the S5 generators EXACTLY (constructive
   existence: transposition = 1 reflection, 5-cycle = 4 reflections).
3. End-to-end: analytic keys/betas per token -> scan prefix states equal the
   MUL-table running composition through the permutation representation.
4. Model smoke: finite loss, gradients flow, scan/seq paths agree in the LM.
"""
import torch
import torch.nn.functional as F

from phase_ssm.nonabelian_scan import (
    NonAbelianScanLM, affine_scan, affine_seq, householder_products)
from phase_ssm.statetrack import GENS, IDENTITY, MUL, PERMS, C, S


def _rand_affine(B=2, L=7, H=3, d=6, n_h=4, diag=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    if diag:
        A = torch.tanh(torch.randn(B, L, H, d, generator=g))
    else:
        keys = F.normalize(torch.randn(B, L, H, n_h, d, generator=g), dim=-1)
        betas = 2.0 * torch.sigmoid(torch.randn(B, L, H, n_h, generator=g))
        A = householder_products(keys, betas)
    b = torch.randn(B, L, H, d, generator=g)
    h0 = torch.randn(H, d, generator=g)
    return A, b, h0


def test_scan_matches_sequential_full():
    A, b, h0 = _rand_affine(diag=False)
    assert torch.allclose(affine_scan(A, b, h0), affine_seq(A, b, h0), atol=1e-5)


def test_scan_matches_sequential_diag():
    A, b, h0 = _rand_affine(diag=True)
    assert torch.allclose(affine_scan(A, b, h0), affine_seq(A, b, h0), atol=1e-5)


def _perm_matrix(idx):
    p = PERMS[idx]
    P = torch.zeros(5, 5)
    for i in range(5):
        P[p[i], i] = 1.0          # P e_i = e_{p(i)}; then P(a then b) = P(b) @ P(a)
    return P


def _e(i, j):
    k = torch.zeros(5)
    k[i], k[j] = 1.0, -1.0
    return k / k.norm()


def _gen_factors(tok):
    """Analytic Householder decomposition of each generator token (n_h=4, d=5).

    transposition (01): one reflection H(e0-e1), rest identity (beta=0).
    5-cycle: P_C = H(0,4) @ H(0,3) @ H(0,2) @ H(0,1) — factor 1 applied first.
    identity: all beta=0.
    """
    keys = torch.stack([_e(0, 1), _e(0, 2), _e(0, 3), _e(0, 4)])
    if tok == S:
        betas = torch.tensor([2.0, 0.0, 0.0, 0.0])
    elif tok == C:
        betas = torch.tensor([2.0, 2.0, 2.0, 2.0])
    else:
        betas = torch.zeros(4)
    return keys, betas


def test_householder_expresses_s5_generators():
    for tok in (S, C, IDENTITY):
        keys, betas = _gen_factors(tok)
        A = householder_products(keys, betas)
        assert torch.allclose(A, _perm_matrix(tok), atol=1e-6), f"token {tok}"


def test_scan_tracks_group_composition():
    torch.manual_seed(0)
    L = 33                        # non-power-of-2 on purpose
    word = [GENS[i] for i in torch.randint(0, len(GENS), (L,)).tolist()]
    keys = torch.stack([_gen_factors(t)[0] for t in word])[None, :, None]   # (1,L,1,4,5)
    betas = torch.stack([_gen_factors(t)[1] for t in word])[None, :, None]  # (1,L,1,4)
    A = householder_products(keys, betas)
    b = torch.zeros(1, L, 1, 5)
    h0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])[None]
    h = affine_scan(A, b, h0)
    state = IDENTITY
    for t in range(L):
        state = MUL[state, word[t]].item()
        want = _perm_matrix(state) @ h0[0]
        assert torch.allclose(h[0, t, 0], want, atol=1e-4), f"t={t}"


def test_model_smoke_and_path_agreement():
    torch.manual_seed(0)
    ids = torch.randint(0, 150, (2, 9))
    tgt = torch.randint(0, 150, (2, 9))
    for transition in ("householder", "diag"):
        torch.manual_seed(0)
        m = NonAbelianScanLM(150, d=32, layers=2, H=4, d_h=8, n_h=2, transition=transition)
        logits, loss = m(ids, tgt)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in m.parameters() if p.requires_grad)
        for blk in m.blocks:
            blk.scan = False
        logits_seq, _ = m(ids)
        assert torch.allclose(logits.detach(), logits_seq.detach(), atol=1e-4)
