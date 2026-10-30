"""Rung B unit tests: the task must be right before any separation claim."""
import torch

from phase_ssm.conditional_composition import (
    FIX0, INV, DreamScanLM, make_conditional_batch)
from phase_ssm.statetrack import IDENTITY, MUL, PERMS


def test_inverse_table():
    for j in range(120):
        assert MUL[j, INV[j]].item() == IDENTITY
        assert MUL[INV[j], j].item() == IDENTITY


def test_fix0_predicate():
    for j in range(120):
        assert FIX0[j].item() == (1 if PERMS[j][0] == 0 else 0)


def test_generator_matches_bruteforce():
    torch.manual_seed(0)
    B, L = 8, 40
    x, tgt = make_conditional_batch(B, L, 0.5, 30, "cpu")
    for b in range(B):
        s = IDENTITY
        for t in range(L):
            tok = x[b, t].item()
            if tok < 120:                                   # generator token
                eff = tok if FIX0[s].item() == 1 else INV[tok].item()
                s = MUL[s, eff].item()
            assert tgt[b, t].item() == s, f"b={b} t={t}"


def test_mode_actually_flips():
    """The task must exercise both branches — g and g^-1 applications."""
    torch.manual_seed(0)
    x, _ = make_conditional_batch(64, 64, 0.5, 30, "cpu")
    # replay and count branches
    both = [0, 0]
    for b in range(64):
        s = IDENTITY
        for t in range(64):
            tok = x[b, t].item()
            if tok < 120:
                m = FIX0[s].item()
                both[m] += 1
                eff = tok if m == 1 else INV[tok].item()
                s = MUL[s, eff].item()
    assert both[0] > 100 and both[1] > 100, both


def test_dreamscan_smoke():
    torch.manual_seed(0)
    m = DreamScanLM(150, d=32, H=4, d_h=8, n_h=2, S_train=3)
    ids = torch.randint(0, 150, (2, 9))
    tgt = torch.randint(0, 150, (2, 9))
    _, loss = m(ids, tgt)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    logits1, _ = m(ids, sweeps=1)
    logits8, _ = m(ids, sweeps=8)
    assert logits1.shape == logits8.shape
    assert not torch.allclose(logits1, logits8)             # sweeps must do something
