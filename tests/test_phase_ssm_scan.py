from __future__ import annotations

import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch


class PhaseSSMRealPairScanTests(unittest.TestCase):
    def test_real_pair_chunked_scan_matches_complex_chunked_scan(self) -> None:
        from phase_ssm.model import OscillatorySSM
        from phase_ssm.recurrent import ssm_chunked, ssm_chunked_real

        torch.manual_seed(7)
        ssm = OscillatorySSM(channels=12, state_dim=8, dt_min=1e-3, dt_max=1e-1)
        u = torch.randn(2, 77, 12)

        complex_y = ssm_chunked(ssm, u, chunk=16)
        real_y = ssm_chunked_real(ssm, u, chunk=16)
        scale = complex_y.abs().max().item() + 1e-9
        rel = (complex_y - real_y).abs().max().item() / scale

        self.assertLess(rel, 1e-5)

    def test_real_pair_recurrent_matches_conv_oracle(self) -> None:
        from phase_ssm.model import OscillatorySSM
        from phase_ssm.recurrent import ssm_recurrent_real

        torch.manual_seed(11)
        ssm = OscillatorySSM(channels=10, state_dim=6, dt_min=1e-3, dt_max=1e-1)
        u = torch.randn(1, 64, 10)

        conv_y = ssm(u)
        real_y = ssm_recurrent_real(ssm, u)
        scale = conv_y.abs().max().item() + 1e-9
        rel = (conv_y - real_y).abs().max().item() / scale

        self.assertLess(rel, 1e-4)

    def test_real_chunked_backend_matches_fft_backend(self) -> None:
        from phase_ssm.model import OscillatorySSM

        torch.manual_seed(13)
        fft = OscillatorySSM(channels=6, state_dim=5, dt_min=1e-3, dt_max=1e-1, backend="fft")
        real = OscillatorySSM(
            channels=6,
            state_dim=5,
            dt_min=1e-3,
            dt_max=1e-1,
            backend="real_chunked",
            chunk=8,
        )
        real.load_state_dict(fft.state_dict())
        u = torch.randn(2, 31, 6)

        fft_y = fft(u)
        real_y = real(u)
        scale = fft_y.abs().max().item() + 1e-9
        rel = (fft_y - real_y).abs().max().item() / scale

        self.assertLess(rel, 1e-4)

    def test_oscillatory_ssm_auto_backend_routes_by_length(self) -> None:
        from phase_ssm.model import OscillatorySSM

        ssm = OscillatorySSM(
            channels=4,
            state_dim=3,
            dt_min=1e-3,
            dt_max=1e-1,
            backend="auto",
            auto_threshold=16,
        )

        self.assertEqual(ssm.backend_for_length(16), "fft")
        self.assertEqual(ssm.backend_for_length(17), "fixed_triton")

    def test_fixed_kernel_backend_matches_fft_input_gradient(self) -> None:
        from phase_ssm.model import OscillatorySSM

        torch.manual_seed(19)
        fft = OscillatorySSM(channels=5, state_dim=4, dt_min=1e-3, dt_max=1e-1, backend="fft")
        fixed = OscillatorySSM(
            channels=5,
            state_dim=4,
            dt_min=1e-3,
            dt_max=1e-1,
            backend="fixed_triton",
            chunk=8,
        )
        fixed.load_state_dict(fft.state_dict())
        for param in fft.parameters():
            param.requires_grad_(False)
        for param in fixed.parameters():
            param.requires_grad_(False)

        u_fft = torch.randn(2, 23, 5, requires_grad=True)
        u_fixed = u_fft.detach().clone().requires_grad_(True)
        target = torch.randn(2, 23, 5)

        fft_loss = (fft(u_fft) * target).sum()
        fixed_loss = (fixed(u_fixed) * target).sum()
        fft_loss.backward()
        fixed_loss.backward()

        y_scale = abs(float(fft_loss.detach())) + 1e-9
        y_rel = abs(float(fft_loss.detach() - fixed_loss.detach())) / y_scale
        g_scale = u_fft.grad.abs().max().item() + 1e-9
        g_rel = (u_fft.grad - u_fixed.grad).abs().max().item() / g_scale

        self.assertLess(y_rel, 1e-4)
        self.assertLess(g_rel, 1e-4)

    def test_phase_ssm_block_ablation_configs_train(self) -> None:
        from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM

        for cfg in [
            PhaseSSMConfig(vocab_size=32, d_model=16, n_layers=1, state_dim=4, use_mixer=False),
            PhaseSSMConfig(vocab_size=32, d_model=16, n_layers=1, state_dim=4, use_ffn=False),
            PhaseSSMConfig(vocab_size=32, d_model=16, n_layers=1, state_dim=4, use_gate=False),
            PhaseSSMConfig(vocab_size=32, d_model=16, n_layers=1, state_dim=4, use_phase_memory=True, mem_heads=2, mem_head_dim=8),
        ]:
            model = PhaseSSMLM(cfg)
            ids = torch.randint(0, 32, (2, 12))
            logits, loss = model(ids, ids)
            self.assertEqual(logits.shape, (2, 12, 32))
            self.assertIsNotNone(loss)
            loss.backward()

    def test_train_build_model_exposes_phase_memory_flags(self) -> None:
        from phase_ssm.train import build_model

        args = SimpleNamespace(
            model="phasessm",
            d_model=16,
            n_layers=1,
            state_dim=4,
            expand=1,
            d_ff_mult=1,
            short_conv=0,
            ssm_backend="fft",
            ssm_chunk=8,
            ssm_auto_threshold=128,
            dropout=0.0,
            use_phase_memory=True,
            mem_heads=2,
            mem_head_dim=8,
            mem_linear=True,
            mem_chunk=32,
            mem_layout="parallel",
        )

        model, cfg = build_model(args, 32)

        self.assertTrue(cfg["use_phase_memory"])
        self.assertEqual(cfg["mem_heads"], 2)
        self.assertTrue(cfg["mem_linear"])
        self.assertEqual(cfg["mem_layout"], "parallel")
        self.assertIsNotNone(model.blocks[0].memory)

    def test_phase_block_parallel_layout_shares_norm(self) -> None:
        from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM

        cfg = PhaseSSMConfig(vocab_size=32, d_model=16, n_layers=1, state_dim=4,
                             use_phase_memory=True, mem_heads=2, mem_head_dim=8,
                             mem_layout="parallel")
        model = PhaseSSMLM(cfg)
        block = model.blocks[0]
        # Parallel layout: both branches present, memory has no private norm.
        self.assertIsNotNone(block.mixer)
        self.assertIsNotNone(block.memory)
        self.assertIsNone(block.norm_m)         # shares norm1 with the mixer

    def test_phase_block_sequential_layout_gives_memory_own_norm(self) -> None:
        from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM

        cfg = PhaseSSMConfig(vocab_size=32, d_model=16, n_layers=1, state_dim=4,
                             use_phase_memory=True, mem_heads=2, mem_head_dim=8,
                             mem_layout="sequential")
        model = PhaseSSMLM(cfg)
        block = model.blocks[0]
        # Sequential layout: memory gets its own norm between mixer and ffn.
        self.assertIsNotNone(block.memory)
        self.assertIsNotNone(block.norm_m)

    def test_phase_block_linear_and_quadratic_memory_produce_identical_output(self) -> None:
        """The linear (O(L)) memory must reproduce the O(L^2) form in the full model."""
        from phase_ssm.model import PhaseSSMConfig, PhaseSSMLM

        torch.manual_seed(0)
        base = dict(vocab_size=32, d_model=16, n_layers=2, state_dim=4,
                    use_phase_memory=True, mem_heads=2, mem_head_dim=8, mem_layout="parallel")
        m_lin = PhaseSSMLM(PhaseSSMConfig(**base, mem_linear=True))
        m_quad = PhaseSSMLM(PhaseSSMConfig(**base, mem_linear=False))
        m_quad.load_state_dict(m_lin.state_dict())

        ids = torch.randint(0, 32, (2, 16))
        with torch.no_grad():
            y_lin, _ = m_lin(ids)
            y_quad, _ = m_quad(ids)
        scale = y_lin.abs().max().item() + 1e-9
        rel = (y_lin - y_quad).abs().max().item() / scale
        self.assertLess(rel, 1e-4)

    def test_effbench_auto_backend_routes_short_to_fft_and_long_to_triton(self) -> None:
        from phase_ssm.effbench import SSMStack

        stack = SSMStack(8, 4, 1, backend="auto", auto_threshold=128)

        self.assertEqual(stack.backend_for_length(64), "fft")
        self.assertEqual(stack.backend_for_length(128), "fft")
        self.assertEqual(stack.backend_for_length(129), "triton")

    def test_effbench_auto_threshold_zero_uses_triton(self) -> None:
        from phase_ssm.effbench import SSMStack

        stack = SSMStack(8, 4, 1, backend="auto", auto_threshold=0)

        self.assertEqual(stack.backend_for_length(64), "triton")
        self.assertEqual(stack.backend_for_length(65536), "triton")

    def test_shortbench_recommends_measured_auto_threshold(self) -> None:
        from phase_ssm.shortbench import recommend_threshold, render_markdown

        rows = [
            {
                "length": 1024,
                "backend": "fft",
                "ssm_tok_s": 150_000.0,
                "attn_tok_s": 300_000.0,
                "ssm_mem_gb": 0.1,
                "attn_mem_gb": 0.2,
                "ratio": 0.5,
            },
            {
                "length": 1024,
                "backend": "triton",
                "ssm_tok_s": 100_000.0,
                "attn_tok_s": 300_000.0,
                "ssm_mem_gb": 0.1,
                "attn_mem_gb": 0.2,
                "ratio": 0.333,
            },
            {
                "length": 4096,
                "backend": "fft",
                "ssm_tok_s": 420_000.0,
                "attn_tok_s": 400_000.0,
                "ssm_mem_gb": 0.1,
                "attn_mem_gb": 0.2,
                "ratio": 1.05,
            },
            {
                "length": 4096,
                "backend": "triton",
                "ssm_tok_s": 250_000.0,
                "attn_tok_s": 400_000.0,
                "ssm_mem_gb": 0.1,
                "attn_mem_gb": 0.2,
                "ratio": 0.625,
            },
            {
                "length": 65536,
                "backend": "fft",
                "ssm_tok_s": 180_000.0,
                "attn_tok_s": 150_000.0,
                "ssm_mem_gb": 0.1,
                "attn_mem_gb": 0.2,
                "ratio": 1.2,
            },
            {
                "length": 65536,
                "backend": "triton",
                "ssm_tok_s": 220_000.0,
                "attn_tok_s": 150_000.0,
                "ssm_mem_gb": 0.1,
                "attn_mem_gb": 0.2,
                "ratio": 1.466,
            },
        ]

        recommendation = recommend_threshold(rows)
        self.assertEqual(recommendation["recommended_auto_threshold"], 4096)
        self.assertEqual(recommendation["first_best_ssm_win_vs_attention"], 4096)
        self.assertEqual(recommendation["best_by_length"][-1]["best_backend"], "triton")

        markdown = render_markdown({"rows": rows, "recommendation": recommendation})
        self.assertIn("recommended_auto_threshold", markdown)
        self.assertIn("| 4096 | `fft`", markdown)
        self.assertIn("| 65536 | `triton`", markdown)

    def test_train_rejects_diagnostic_backend_without_explicit_override(self) -> None:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "phase_ssm.train",
                "--model",
                "phasessm",
                "--ssm-backend",
                "skip",
                "--out",
                "/tmp/phase-ssm-should-not-run",
                "--steps",
                "0",
            ],
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(run.returncode, 0)
        self.assertIn("diagnostic", run.stderr + run.stdout)

    def test_train_rejects_auto_backend_when_sequence_crosses_threshold(self) -> None:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "phase_ssm.train",
                "--model",
                "phasessm",
                "--ssm-backend",
                "auto",
                "--ssm-auto-threshold",
                "16",
                "--seq",
                "17",
                "--out",
                "/tmp/phase-ssm-auto-should-not-run",
                "--steps",
                "0",
            ],
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(run.returncode, 0)
        self.assertIn("frozen recurrent path", run.stderr + run.stdout)

    def test_triton_backend_fails_loudly_without_cuda_triton(self) -> None:
        from phase_ssm.model import OscillatorySSM
        from phase_ssm.triton_scan import (
            ssm_chunked_scan_triton,
            ssm_recurrent_triton,
            ssm_step_triton,
            triton_available,
        )

        if triton_available():
            self.skipTest("CUDA+Triton host exercises this through the efficiency bench")

        ssm = OscillatorySSM(channels=4, state_dim=4, dt_min=1e-3, dt_max=1e-1)
        u = torch.randn(1, 8, 4)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ssm_recurrent_triton(ssm, u)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ssm_chunked_scan_triton(ssm, u)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ssm_step_triton(ssm, u[:, 0], torch.zeros(1, 4, 4), torch.zeros(1, 4, 4))

    def test_triton_chunked_scan_matches_recurrent_when_available(self) -> None:
        from phase_ssm.model import OscillatorySSM
        from phase_ssm.triton_scan import ssm_chunked_scan_triton, ssm_recurrent_triton, triton_available

        if not triton_available():
            self.skipTest("CUDA+Triton required")

        torch.manual_seed(17)
        ssm = OscillatorySSM(channels=8, state_dim=8, dt_min=1e-3, dt_max=1e-1).cuda()
        u = torch.randn(1, 65, 8, device="cuda")
        recurrent = ssm_recurrent_triton(ssm, u)
        chunked = ssm_chunked_scan_triton(ssm, u, chunk=16, block_n=4)
        torch.cuda.synchronize()

        scale = recurrent.abs().max().item() + 1e-9
        rel = (recurrent - chunked).abs().max().item() / scale
        self.assertLess(rel, 1e-5)

    def test_triton_step_matches_recurrent_tail_when_available(self) -> None:
        from phase_ssm.model import OscillatorySSM
        from phase_ssm.triton_scan import ssm_recurrent_triton, ssm_step_triton, triton_available

        if not triton_available():
            self.skipTest("CUDA+Triton required")

        torch.manual_seed(23)
        ssm = OscillatorySSM(channels=8, state_dim=8, dt_min=1e-3, dt_max=1e-1).cuda()
        prefix = torch.randn(1, 17, 8, device="cuda")
        nxt = torch.randn(1, 8, device="cuda")

        full = ssm_recurrent_triton(ssm, torch.cat([prefix, nxt[:, None]], dim=1))
        _, state_r, state_i = ssm_recurrent_triton(ssm, prefix, return_state=True)
        stepped = ssm_step_triton(ssm, nxt, state_r, state_i)
        torch.cuda.synchronize()

        scale = full[:, -1].abs().max().item() + 1e-9
        rel = (full[:, -1] - stepped).abs().max().item() / scale
        self.assertLess(rel, 1e-5)


if __name__ == "__main__":
    unittest.main()
