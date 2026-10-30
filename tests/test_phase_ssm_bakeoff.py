"""Tests for the matched-transformer baseline and the bake-off verdict logic.

The transformer is the comparison oracle for the whole post-transformer claim,
so its positional-encoding wiring and the bake-off's verdict/refusal logic both
need explicit coverage — a silently weak baseline flatters a win, and a verdict
computed from mismatched params is meaningless.
"""
from __future__ import annotations

import unittest

import torch


class TransformerBaselineTests(unittest.TestCase):
    def test_rope_is_default_and_applied_to_qk(self) -> None:
        """The honest baseline uses RoPE on Q,K in attention (no learned pos emb)."""
        from phase_ssm.transformer import TransformerConfig, TransformerLM

        cfg = TransformerConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4,
                                d_ff_mult=2, block_size=32)  # default pos_enc="rope"
        m = TransformerLM(cfg)
        self.assertTrue(m.blocks[0].attn.use_rope)
        self.assertIsNone(m.pos)  # RoPE adds no learned position embedding

    def test_learned_pos_enc_is_opt_in(self) -> None:
        from phase_ssm.transformer import TransformerConfig, TransformerLM

        cfg = TransformerConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4,
                                d_ff_mult=2, block_size=32, pos_enc="learned")
        m = TransformerLM(cfg)
        self.assertFalse(m.blocks[0].attn.use_rope)
        self.assertIsNotNone(m.pos)  # learned absolute embedding allocated

    def test_rope_changes_output(self) -> None:
        """RoPE must not be a no-op: same weights, different pos_enc -> different logits."""
        from phase_ssm.transformer import TransformerConfig, TransformerLM

        torch.manual_seed(0)
        ids = torch.randint(0, 32, (1, 8))
        base = dict(vocab_size=32, d_model=16, n_layers=1, n_heads=4, d_ff_mult=2, block_size=32)
        m_rope = TransformerLM(TransformerConfig(**base, pos_enc="rope"))
        m_learned = TransformerLM(TransformerConfig(**base, pos_enc="learned"))
        # Copy all shared weights so only the positional path differs.
        shared = {k: v for k, v in m_rope.state_dict().items()
                  if k in m_learned.state_dict() and m_learned.state_dict()[k].shape == v.shape}
        m_learned.load_state_dict({**m_learned.state_dict(), **shared}, strict=False)
        with torch.no_grad():
            y_rope, _ = m_rope(ids)
            y_learned, _ = m_learned(ids)
        self.assertFalse(torch.allclose(y_rope, y_learned, atol=1e-5),
                         "RoPE and learned pos enc should produce different outputs")

    def test_transformer_forward_backward(self) -> None:
        from phase_ssm.transformer import TransformerConfig, TransformerLM

        cfg = TransformerConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=4,
                                d_ff_mult=2, block_size=32, pos_enc="rope")
        m = TransformerLM(cfg)
        ids = torch.randint(0, 32, (2, 16))
        tgt = torch.randint(0, 32, (2, 16))
        _, loss = m(ids, tgt)
        loss.backward()
        self.assertGreater(loss.item(), 0)


class BakeoffVerdictTests(unittest.TestCase):
    def _payload(self, ssm_bpc, xf_bpc, ssm_params=100, xf_params=100, tol=1.05):
        return {
            "date": "2026-06-24T00:00:00Z", "train_bytes": 1, "val_bytes": 1,
            "param_tol": tol, "bpc_deadband": 0.02,
            "models": [
                {"name": "phasessm", "params": ssm_params, "tokens": 1000, "steps": 10,
                 "best_val_bpc": ssm_bpc, "final_val_bpc": ssm_bpc, "total_s": 1},
                {"name": "transformer", "params": xf_params, "tokens": 1000, "steps": 10,
                 "best_val_bpc": xf_bpc, "final_val_bpc": xf_bpc, "total_s": 1},
            ],
        }

    def test_report_marks_ssm_win(self) -> None:
        from phase_ssm.bakeoff import render_report
        report = render_report(self._payload(ssm_bpc=1.10, xf_bpc=1.20))
        self.assertIn("SSM wins", report)

    def test_report_marks_transformer_win(self) -> None:
        from phase_ssm.bakeoff import render_report
        report = render_report(self._payload(ssm_bpc=1.30, xf_bpc=1.20))
        self.assertIn("Transformer wins", report)

    def test_report_marks_tie_within_deadband(self) -> None:
        from phase_ssm.bakeoff import render_report
        report = render_report(self._payload(ssm_bpc=1.20, xf_bpc=1.21))
        self.assertIn("TIE", report)

    def test_report_refuses_verdict_when_params_drift_past_tolerance(self) -> None:
        from phase_ssm.bakeoff import render_report
        report = render_report(self._payload(ssm_bpc=1.10, xf_bpc=1.20,
                                             ssm_params=200, xf_params=100, tol=1.05))
        self.assertIn("NO VERDICT", report)

    def test_matched_steps_equalizes_total_tokens(self) -> None:
        from phase_ssm.bakeoff import matched_steps
        from types import SimpleNamespace
        args = SimpleNamespace(tokens=512_000_000, batch=64, seq=1024)
        s, x = matched_steps(args, ssm_params=1, xf_params=1)
        # Both models get the same number of steps (hence the same token budget).
        self.assertEqual(s, x)
        # Actual tokens are steps*batch*seq, within one step of the requested budget.
        actual = s * args.batch * args.seq
        self.assertLessEqual(actual, args.tokens)
        self.assertGreater(actual, args.tokens - args.batch * args.seq)


if __name__ == "__main__":
    unittest.main()
