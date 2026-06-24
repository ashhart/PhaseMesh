from __future__ import annotations

import subprocess
import sys
import unittest

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
        ]:
            model = PhaseSSMLM(cfg)
            ids = torch.randint(0, 32, (2, 12))
            logits, loss = model(ids, ids)
            self.assertEqual(logits.shape, (2, 12, 32))
            self.assertIsNotNone(loss)
            loss.backward()

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
