from __future__ import annotations

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
