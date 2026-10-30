import torch
import torch.nn as nn

from phase_ssm.dream import DreamConfig, DeltaMemory, DreamBlock, DreamLM


def test_delta_write_recalls():
    # orthonormal keys -> exact recall after writing (k_i -> v_i)
    B, h, dk, dv = 1, 1, 4, 4
    M = torch.zeros(B, h, dv, dk)
    eye = torch.eye(dk)
    k1, k2 = eye[0].view(1, 1, dk), eye[1].view(1, 1, dk)
    v1 = torch.tensor([1., 0., 0., 0.]).view(1, 1, dv)
    v2 = torch.tensor([0., 1., 0., 0.]).view(1, 1, dv)
    beta = torch.ones(1, 1, 1)
    M = DeltaMemory.delta_write(M, k1, v1, beta)
    M = DeltaMemory.delta_write(M, k2, v2, beta)
    o1 = torch.einsum("bhvd,bhd->bhv", M, k1)
    assert torch.allclose(o1, v1, atol=1e-5)


def test_delta_write_no_interference():
    # writing a second orthogonal binding must NOT corrupt the first
    # (exactly the additive-memory failure DREAM is built to fix)
    B, h, dk, dv = 1, 1, 4, 4
    eye = torch.eye(dk)
    k1, k2 = eye[0].view(1, 1, dk), eye[1].view(1, 1, dk)
    v1, v2 = eye[2].view(1, 1, dv), eye[3].view(1, 1, dv)
    beta = torch.ones(1, 1, 1)
    M = torch.zeros(B, h, dv, dk)
    M = DeltaMemory.delta_write(M, k1, v1, beta)
    before = torch.einsum("bhvd,bhd->bhv", M, k1).clone()
    M = DeltaMemory.delta_write(M, k2, v2, beta)
    after = torch.einsum("bhvd,bhd->bhv", M, k1)
    assert torch.allclose(before, after, atol=1e-5)


def test_delta_memory_forward_shape():
    cfg = DreamConfig(d_model=32, mem_heads=2, mem_head_dim=8, inner_iters=2)
    mem = DeltaMemory(cfg)
    x = torch.randn(3, 7, 32)
    o = mem(x)
    assert o.shape == (3, 7, 32)
    assert torch.isfinite(o).all()


def test_linearize_control_removes_nonlinearity():
    lin = DreamBlock(DreamConfig(linearize=True))
    assert isinstance(lin.nl, nn.Linear)            # no SiLU in the control
    nonlin = DreamBlock(DreamConfig(linearize=False))
    assert any(isinstance(m, nn.SiLU) for m in nonlin.nl.modules())


def test_dream_lm_runs_with_knobs():
    cfg = DreamConfig(vocab_size=40, d_model=32, n_layers=2,
                      outer_iters=2, inner_iters=2, mem_heads=2, mem_head_dim=8)
    m = DreamLM(cfg)
    ids = torch.randint(0, 40, (2, 9))
    tgt = torch.randint(0, 40, (2, 9))
    logits, loss = m(ids, tgt)
    assert logits.shape == (2, 9, 40)
    assert torch.isfinite(loss) and loss.item() > 0
    assert m.num_params() > 0
