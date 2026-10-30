from __future__ import annotations

import importlib.util
import unittest

import torch

from phase_ssm.devmem import PeakMemory, sync

HAS_PSUTIL = importlib.util.find_spec("psutil") is not None


class DevMemTests(unittest.TestCase):
    def test_sync_is_a_no_op_on_cpu(self) -> None:
        sync("cpu")

    @unittest.skipUnless(HAS_PSUTIL, "psutil is needed for CPU memory sampling")
    def test_cpu_sampler_sees_a_large_allocation(self) -> None:
        with PeakMemory("cpu") as mem:
            block = torch.ones(64 * 1024 * 1024, dtype=torch.float32)  # 256 MiB, touched
            del block
        result = mem.result()
        self.assertEqual(result["mem_method"], "cpu_rss_sampled")
        self.assertGreater(result["added_gb"], 0.2)
        self.assertGreaterEqual(result["peak_gb"], result["added_gb"])

    @unittest.skipIf(HAS_PSUTIL, "only meaningful without psutil")
    def test_cpu_without_psutil_reports_unavailable(self) -> None:
        with PeakMemory("cpu") as mem:
            pass
        self.assertEqual(mem.result(), {"peak_gb": None, "added_gb": None, "mem_method": "unavailable"})

    @unittest.skipUnless(torch.backends.mps.is_available(), "needs Apple MPS")
    def test_mps_sampler_sees_a_large_allocation(self) -> None:
        with PeakMemory("mps") as mem:
            block = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device="mps")
            sync("mps")
            del block
        result = mem.result()
        self.assertEqual(result["mem_method"], "mps_allocated_sampled")
        self.assertGreater(result["added_gb"], 0.2)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_peak_is_exact(self) -> None:
        with PeakMemory("cuda") as mem:
            block = torch.ones(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
            del block
        self.assertGreaterEqual(mem.result()["added_gb"], 0.268)


if __name__ == "__main__":
    unittest.main()
