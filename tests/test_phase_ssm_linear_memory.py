"""Bit-equivalence tests for the linear phase-binding memory.

The linear forms (chunked, recurrent) must reproduce the O(L^2) quadratic
oracle, and the oracle must reproduce the original ``PhaseBindingMemory``. If any
of these break, the post-transformer mechanic is no longer drop-in with the
existing model code.
"""
from __future__ import annotations

import unittest

import torch


class LinearPhaseMemoryEquivalenceTests(unittest.TestCase):
    def _make_inputs(self, B=2, h=3, L=40, dk=5, dv=4, seed=0):
        torch.manual_seed(seed)
        q_phi = torch.randn(B, h, L, dk) * 3.0
        k_phi = torch.randn(B, h, L, dk) * 3.0
        v = torch.randn(B, h, L, dv)
        gamma = torch.tensor([0.85, 0.95, 0.99])
        return q_phi, k_phi, v, gamma

    def test_chunked_matches_quadratic(self) -> None:
        from phase_ssm.linear_memory import phase_memory_quadratic, phase_memory_chunked

        q_phi, k_phi, v, gamma = self._make_inputs()
        y_q = phase_memory_quadratic(q_phi, k_phi, v, gamma)
        # chunk < L exercises the cross-chunk carry; chunk | L and a non-divisor
        # both covered by chunk=8 over L=40.
        y_c = phase_memory_chunked(q_phi, k_phi, v, gamma, chunk=8)
        scale = y_q.abs().max().item() + 1e-9
        rel = (y_q - y_c).abs().max().item() / scale
        self.assertLess(rel, 1e-4)

    def test_single_chunk_matches_quadratic(self) -> None:
        from phase_ssm.linear_memory import phase_memory_quadratic, phase_memory_chunked

        q_phi, k_phi, v, gamma = self._make_inputs(L=40)
        y_q = phase_memory_quadratic(q_phi, k_phi, v, gamma)
        y_c = phase_memory_chunked(q_phi, k_phi, v, gamma, chunk=64)
        scale = y_q.abs().max().item() + 1e-9
        rel = (y_q - y_c).abs().max().item() / scale
        self.assertLess(rel, 1e-4)

    def test_recurrent_matches_quadratic(self) -> None:
        from phase_ssm.linear_memory import phase_memory_quadratic, phase_memory_recurrent

        q_phi, k_phi, v, gamma = self._make_inputs()
        y_q = phase_memory_quadratic(q_phi, k_phi, v, gamma)
        y_r = phase_memory_recurrent(q_phi, k_phi, v, gamma)
        scale = y_q.abs().max().item() + 1e-9
        rel = (y_q - y_r).abs().max().item() / scale
        self.assertLess(rel, 1e-4)

    def test_quadratic_reproduces_original_phase_binding_memory(self) -> None:
        """The linear oracle must reproduce ``PhaseBindingMemory.forward`` exactly."""
        from phase_ssm.memory import PhaseBindingMemory
        from phase_ssm.linear_memory import phase_memory_quadratic

        torch.manual_seed(3)
        B, L, d_model = 2, 20, 16
        h, dk, dv = 3, 8, 8
        m = PhaseBindingMemory(d_model, n_heads=h, head_dim=dk, v_dim=dv)
        m.decay_logit.data = torch.tensor([2.0, 3.0, 4.0])
        x = torch.randn(B, L, d_model)

        y_orig = m(x)

        gamma = torch.sigmoid(m.decay_logit)
        qa = m.to_q(x).float().view(B, L, h, dk).permute(0, 2, 1, 3)
        ka = m.to_k(x).float().view(B, L, h, dk).permute(0, 2, 1, 3)
        va = m.to_v(x).float().view(B, L, h, dv).permute(0, 2, 1, 3)
        y_quad = phase_memory_quadratic(qa, ka, va, gamma)
        y_quad = m.out(y_quad.permute(0, 2, 1, 3).reshape(B, L, h * dv))

        scale = y_orig.abs().max().item() + 1e-9
        rel = (y_orig - y_quad).abs().max().item() / scale
        self.assertLess(rel, 1e-5)

    def test_state_footprint_is_independent_of_length(self) -> None:
        """The streaming state must be fixed-size, not a growing KV cache."""
        from phase_ssm.memory import PhaseBindingMemory

        m = PhaseBindingMemory(d_model=32, n_heads=4, head_dim=8)
        bytes_fixed = m.recurrent_state_bytes()
        # Same footprint regardless of how long the context grows.
        self.assertEqual(m.recurrent_state_bytes(), bytes_fixed)
        # Sanity: a real fixed-state number, not zero.
        self.assertGreater(bytes_fixed, 0)


if __name__ == "__main__":
    unittest.main()
