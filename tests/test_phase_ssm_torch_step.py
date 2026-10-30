from __future__ import annotations

import unittest

import torch

from phase_ssm.model import OscillatorySSM
from phase_ssm.recurrent import ssm_final_state_real, ssm_recurrent_real, ssm_step_real


class TorchStepTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.ssm = OscillatorySSM(24, 8, 1e-3, 1e-1)
        self.u = torch.randn(2, 137, 24)  # not a multiple of the chunk, to test the tail

    def _stepped(self):
        state_r = torch.zeros(2, 24, 8)
        state_i = torch.zeros_like(state_r)
        ys = [ssm_step_real(self.ssm, self.u[:, t], state_r, state_i) for t in range(self.u.shape[1])]
        return torch.stack(ys, dim=1), state_r, state_i

    def test_step_matches_reference_recurrence(self) -> None:
        y_steps, _, _ = self._stepped()
        y_ref = ssm_recurrent_real(self.ssm, self.u)
        torch.testing.assert_close(y_steps, y_ref, rtol=1e-5, atol=1e-5)

    def test_final_state_matches_stepping(self) -> None:
        _, state_r, state_i = self._stepped()
        h_r, h_i = ssm_final_state_real(self.ssm, self.u, chunk=32)
        torch.testing.assert_close(h_r, state_r, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(h_i, state_i, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
