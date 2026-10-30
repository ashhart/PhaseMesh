"""Brute-force / constructive tests for the nested-inversion task + damped scan.

No training. The generator tests are the assay: Rung B died on a generator that
was secretly regular (telescoping theorem), so the highest-value checks here are
(a) the reducer matches an independent stack-based reference, and (b) inversion
does real work while its absence collapses to flat (regular) composition.
"""
import torch

from phase_ssm.nested_inversion import (
    INV, OPEN, CLOSE, make_nested_batch, DampedDreamScanLM,
)
from phase_ssm.statetrack import PERMS, IDX, MUL, IDENTITY, GENS

NG = 120


def _inv_perm(p):
    qi = [0] * 5
    for i in range(5):
        qi[p[i]] = i
    return tuple(qi)


def _reduce(seq, invert):
    """Independent pure-Python stack reducer. seq is a list of token ints."""
    active = IDENTITY
    stack = []
    for tok in seq:
        if tok == OPEN:
            stack.append(active)
            active = IDENTITY
        elif tok == CLOSE:
            pop = stack.pop()
            ai = IDX[_inv_perm(PERMS[active])] if invert else active
            active = int(MUL[pop][ai])
        elif tok < NG:
            active = int(MUL[active][tok])
        # distractor (tok >= NG + 2) leaves active unchanged
    assert not stack, "unbalanced brackets"
    return active


def test_inverse_table():
    """INV[j] is the index of PERMS[j]^{-1}: MUL[j, INV[j]] == IDENTITY both ways."""
    for j in range(NG):
        assert int(MUL[j][INV[j]]) == IDENTITY
        assert int(MUL[INV[j]][j]) == IDENTITY


def test_reducer_matches_bruteforce():
    """The batch generator's targets match an independent stack reducer at every position."""
    torch.manual_seed(0)
    for invert in (True, False):
        for D_max in (1, 2, 4):
            x, tgt = make_nested_batch(8, 40, 0.5, 20, D_max, invert, "cpu")
            for b in range(8):
                seq = x[b].tolist()
                # replay incrementally; tgt[b,t] must equal the reducer's active after seq[:t+1]
                active = IDENTITY
                stack = []
                for t, tok in enumerate(seq):
                    if tok == OPEN:
                        stack.append(active); active = IDENTITY
                    elif tok == CLOSE:
                        pop = stack.pop()
                        ai = int(INV[active]) if invert else active
                        active = int(MUL[pop][ai])
                    elif tok < NG:
                        active = int(MUL[active][tok])
                    assert int(tgt[b][t]) == active, \
                        f"mismatch invert={invert} D_max={D_max} b={b} t={t}"


def test_inversion_changes_value():
    """Inversion does real work: same bracket structure, invert True vs False differ.

    With the SAME random seed, the two generators emit identical token streams
    (the emission draws don't see `invert`), so any target difference is purely
    the inversion semantics. Across a batch we expect many positions to differ."""
    torch.manual_seed(1)
    x_inv, t_inv = make_nested_batch(16, 48, 0.5, 20, 4, True, "cpu")
    torch.manual_seed(1)
    x_flat, t_flat = make_nested_batch(16, 48, 0.5, 20, 4, False, "cpu")
    assert torch.equal(x_inv, x_flat), "same seed should yield identical token streams"
    # there must be closes (else the task is degenerate); and inversion must
    # change at least some targets (it does real work whenever a close follows a
    # non-identity enclosed product).
    assert (x_inv == CLOSE).sum().item() > 0, "no close-brackets emitted"
    diff = (t_inv != t_flat).sum().item()
    assert diff > 0, "inversion changed no targets — inversion is a no-op (bug)"


def test_non_inverting_equals_flat_composition():
    """Void-check at the data level: without inversion, brackets are a no-op.

    By associativity, a fully-reduced non-inverting sequence equals the plain
    left-to-right MUL-fold over its generator tokens (OPEN/CLOSE only regroup;
    they never reorder). Two consequences, both asserted: (a) every sequence is
    BALANCED (final depth 0) — without this the final target isn't the full
    reduction and the task is malformed; (b) at every DEPTH-0 position the target
    == the flat fold of generators seen so far (at depth>0 the active product
    legitimately resets on OPEN). If either fails, the generator is not
    collapsing to regular composition and the void-check is meaningless."""
    torch.manual_seed(2)
    x, tgt = make_nested_batch(12, 50, 0.5, 20, 4, False, "cpu")
    for b in range(12):
        toks = x[b].tolist()
        # (a) balance: final depth must be 0
        d = 0
        for tok in toks:
            d += (tok == OPEN) - (tok == CLOSE)
            assert d >= 0, f"depth went negative at b={b}"
        assert d == 0, f"unbalanced sequence at b={b}: final depth {d}"
        # (b) at every depth-0 position, target == flat fold of gens seen so far
        flat = IDENTITY
        d = 0
        for t, tok in enumerate(toks):
            if tok == OPEN:
                d += 1
            elif tok == CLOSE:
                d -= 1
            elif tok < NG:
                flat = int(MUL[flat][tok])
            if d == 0:
                assert int(tgt[b][t]) == flat, \
                    f"non-inverting depth-0 target != flat fold at b={b} t={t}"
        # the full flat fold equals the final (depth-0) target
        flat_all = IDENTITY
        for tok in toks:
            if tok < NG:
                flat_all = int(MUL[flat_all][tok])
        assert int(tgt[b][-1]) == flat_all, \
            f"non-inverting FINAL target != flat fold at b={b}"


def test_balanced_across_seeds_lengths_and_depths():
    """Every emitted sequence is balanced (final depth 0) and never negative.

    Balance is a generator invariant independent of inversion semantics; the
    close-forcing logic must hold across seeds, lengths, depths, and both
    invert settings. A failure here means the task is malformed (the scored
    last-token target is a partial reduction, not the full stack resolution)."""
    for seed in range(8):
        for invert in (True, False):
            for D_max in (1, 2, 4, 8):
                for L in (16, 32, 48, 50):
                    torch.manual_seed(seed)
                    x, _ = make_nested_batch(16, L, 0.5, 20, D_max, invert, "cpu")
                    for b in range(16):
                        d = 0
                        for tok in x[b].tolist():
                            d += (tok == OPEN) - (tok == CLOSE)
                            assert d >= 0, f"neg depth seed={seed} invert={invert} D={D_max} L={L} b={b}"
                            assert d <= D_max, f"exceeds D_max seed={seed} D={D_max} L={L} b={b}"
                        assert d == 0, f"unbalanced seed={seed} invert={invert} D={D_max} L={L} b={b}"


def test_depth_control():
    """Generated sequences respect D_max; D_max=1 forbids nesting, D_max>=2 permits it."""
    torch.manual_seed(3)
    # D_max=1: every close immediately follows the matching open (depth never > 1,
    # and there is never an open at depth 1, so no nested opens) -> max depth <= 1.
    x1, _ = make_nested_batch(32, 40, 0.5, 20, 1, True, "cpu")
    for b in range(32):
        d = 0
        for tok in x1[b].tolist():
            if tok == OPEN:
                d += 1
            elif tok == CLOSE:
                d -= 1
            assert d <= 1, f"D_max=1 violated at b={b}: depth reached {d}"
            assert d >= 0, f"D_max=1 unbalanced at b={b}"
    # D_max>=2: across enough sequences we observe nesting (depth 2 at some point).
    x2, _ = make_nested_batch(64, 64, 0.5, 20, 4, True, "cpu")
    max_d = 0
    for b in range(64):
        d = 0
        for tok in x2[b].tolist():
            d += (tok == OPEN) - (tok == CLOSE)
            max_d = max(max_d, d)
    assert max_d >= 2, f"D_max=4 never produced nesting (max depth {max_d})"


def test_damped_dreamscan_smoke():
    """Finite loss + grads flow; sweeps change output; sweeps=16 does not diverge.

    The last assertion is the Rung-B divergence fix made static: undamped
    iteration to S=16 after S=4 training collapsed to 0.19-0.43; the damped
    contraction must keep S=16 finite (no NaN/inf) at any sweeps setting."""
    torch.manual_seed(4)
    m = DampedDreamScanLM(150, d=32, H=4, d_h=8, n_h=2, S_train=4)
    ids = torch.randint(0, 150, (2, 9))
    tgt = torch.randint(0, 150, (2, 9))
    logits, loss = m(ids, tgt)
    assert torch.isfinite(loss).item(), "loss not finite"
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
               if p.requires_grad), "missing/non-finite grad"
    # sweeps must change the output (the iteration does something)
    with torch.no_grad():
        z1 = m(ids, sweeps=1)[0]
        z8 = m(ids, sweeps=8)[0]
        assert z1.shape == z8.shape
        assert not torch.allclose(z1, z8), "sweeps=1 and sweeps=8 identical"
        z16 = m(ids, sweeps=16)[0]   # the Rung-B divergence point — must stay finite
        assert torch.isfinite(z16).all(), "sweeps=16 diverged (damping failed)"
