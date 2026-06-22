from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from phase_mesh import CognitiveMeshRuntime, MeshConfig, PhaseFieldMesh, TextPhaseEncoder
from phase_mesh.field import laplacian
from phase_mesh.verifier import VerifierRouter


class EncoderTests(unittest.TestCase):
    def test_encoder_is_deterministic_and_bounded(self) -> None:
        encoder = TextPhaseEncoder(32, 32, max_packets=8)
        first = encoder.encode("alpha beta 17 * 19")
        second = encoder.encode("alpha beta 17 * 19")
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 8)
        for packet in first:
            self.assertGreaterEqual(packet.x, 0)
            self.assertLess(packet.x, 32)
            self.assertGreaterEqual(packet.y, 0)
            self.assertLess(packet.y, 32)


class MeshTests(unittest.TestCase):
    def test_mesh_reaches_finite_metrics(self) -> None:
        config = MeshConfig(width=32, height=32, max_steps=80, min_steps=8)
        runtime = CognitiveMeshRuntime(config)
        run = runtime.resonate("check 3 * 7 = 21")
        self.assertTrue(np.isfinite(run.metrics.coherence))
        self.assertTrue(np.isfinite(run.metrics.gradient))
        self.assertGreaterEqual(run.metrics.coherence, 0.0)
        self.assertLessEqual(run.metrics.coherence, 1.0)
        self.assertGreater(len(run.history), 0)

    def test_feedback_updates_landscape(self) -> None:
        config = MeshConfig(width=32, height=32, max_steps=60, min_steps=8)
        mesh = PhaseFieldMesh(config)
        encoder = TextPhaseEncoder(32, 32)
        mesh.inject_text("17 * 19 = 323", encoder)
        mesh.run_until_resonance(max_steps=20, min_steps=4)
        before = mesh.landscape.copy()
        mesh.apply_feedback(success=True, message="verified arithmetic", encoder=encoder)
        self.assertGreater(float(np.mean(np.abs(mesh.landscape - before))), 0.0)

    def test_laplacian_backends_match(self) -> None:
        field = np.arange(64, dtype=np.float64).reshape(8, 8)
        np.testing.assert_allclose(
            laplacian(field, backend="numpy"),
            laplacian(field, backend="scipy"),
        )

    def test_jax_laplacian_matches_when_available(self) -> None:
        try:
            import jax  # noqa: F401
        except Exception:
            self.skipTest("JAX is not installed")
        field = np.arange(64, dtype=np.float64).reshape(8, 8)
        np.testing.assert_allclose(
            laplacian(field, backend="numpy"),
            laplacian(field, backend="jax"),
            atol=1e-5,
        )

    def test_phase_field_jax_wrapper_when_available(self) -> None:
        try:
            import jax.numpy as jnp
        except Exception:
            self.skipTest("JAX is not installed")
        from phase_mesh.field import PhaseField

        stepped = PhaseField(16, backend="jax").step(jnp.zeros((16, 16)))
        self.assertEqual(tuple(stepped.shape), (16, 16))

    def test_quantized_state_round_trip(self) -> None:
        config = MeshConfig(width=32, height=32, max_steps=20, min_steps=4)
        runtime = CognitiveMeshRuntime(config)
        runtime.resonate("check 5 * 6 = 30", learn=True)
        path = "/tmp/phase_mesh_quantized_state.npz"
        runtime.mesh.save_quantized(path)
        loaded = PhaseFieldMesh.load_quantized(path)
        self.assertEqual(loaded.theta.shape, runtime.mesh.theta.shape)
        self.assertEqual(loaded.step_index, runtime.mesh.step_index)
        self.assertLess(float(np.max(np.abs(loaded.theta - runtime.mesh.theta))), 0.04)

    def test_topological_memory_recalls_exact_key(self) -> None:
        config = MeshConfig(width=32, height=32, max_steps=60, min_steps=8)
        runtime = CognitiveMeshRuntime(config)
        runtime.remember("mesh_fact_alpha", "value_123", steps=40)
        recalled = runtime.recall("mesh_fact_alpha", steps=40)
        self.assertTrue(recalled["recall"]["found"])
        self.assertEqual(recalled["recall"]["value"], "value_123")

    def test_predictor_observes_phase_error(self) -> None:
        config = MeshConfig(width=32, height=32, max_steps=40, min_steps=4)
        mesh = PhaseFieldMesh(config)
        encoder = TextPhaseEncoder(32, 32)
        mesh.inject_text("predict 17 * 19", encoder)
        predicted = mesh.predict_phase()
        mesh.step()
        error = mesh.observe_prediction(predicted)
        self.assertGreaterEqual(error, 0.0)
        self.assertGreater(float(np.mean(np.abs(mesh.predictor_trace))), 0.0)

    def test_phase_pinning_reduces_context_gradient(self) -> None:
        unpinned = CognitiveMeshRuntime(
            MeshConfig(width=32, height=32, max_steps=80, min_steps=8, phase_pin_strength=0.0)
        )
        pinned = CognitiveMeshRuntime(
            MeshConfig(width=32, height=32, max_steps=80, min_steps=8, phase_pin_strength=0.25)
        )
        prompt = " ".join(f"ctx_{index:03d}" for index in range(24)) + "\nquery: preserve ctx_005"
        unpinned_run = unpinned.resonate(prompt)
        pinned_run = pinned.resonate(prompt)
        self.assertLess(pinned_run.metrics.gradient, unpinned_run.metrics.gradient)
        self.assertGreater(float(np.mean(pinned.mesh.pin_weights)), 0.0)

    def test_adaptive_think_respects_budget_and_tracks_basin(self) -> None:
        config = MeshConfig(width=32, height=32, max_steps=80, min_steps=6)
        runtime = CognitiveMeshRuntime(config)
        run = runtime.think("check 3 * 7 = 21", max_budget=30, expected=None)
        self.assertGreater(run.steps_used, 0)
        self.assertLessEqual(run.steps_used, 30)
        self.assertGreaterEqual(run.mean_prediction_error, 0.0)
        self.assertIsNotNone(run.basin)

    def test_basin_tracker_persists_repeated_attractor(self) -> None:
        config = MeshConfig(width=32, height=32, max_steps=80, min_steps=6)
        runtime = CognitiveMeshRuntime(config)
        for _ in range(3):
            runtime.think("repeatable basin prompt", max_budget=40)
        basins = runtime.discover_basins()["basins"]
        self.assertTrue(any(item["count"] >= 2 for item in basins))


class VerifierTests(unittest.TestCase):
    def test_arithmetic_equation_passes(self) -> None:
        result = VerifierRouter().verify("verify 17 * 19 = 323")
        self.assertTrue(result.passed)
        self.assertEqual(result.checker, "arithmetic-equation")

    def test_arithmetic_equation_fails(self) -> None:
        result = VerifierRouter().verify("verify 17 * 19 = 322")
        self.assertFalse(result.passed)
        self.assertEqual(result.checker, "arithmetic-equation")

    def test_python_prompt_is_checked_when_candidate_is_route(self) -> None:
        result = VerifierRouter().verify("def add(a, b):\n    return a + b", candidate="route")
        self.assertTrue(result.passed)
        self.assertEqual(result.checker, "python-compile")


class ServiceTests(unittest.TestCase):
    def test_service_imports(self) -> None:
        from phase_mesh.service import app

        self.assertEqual(app.title, "Phase-Field Cognitive Mesh")

    def test_think_endpoint(self) -> None:
        from fastapi.testclient import TestClient
        from phase_mesh.service import app

        with TestClient(app) as client:
            response = client.post(
                "/think",
                json={"text": "check 2 * 3 = 6", "max_budget": 20},
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("steps_used", response.json())

    def test_demo_endpoint(self) -> None:
        from fastapi.testclient import TestClient
        from phase_mesh.service import app

        with TestClient(app) as client:
            response = client.get("/demo")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Phase Mesh", response.text)
        self.assertIn("/think/stream", response.text)

    def test_stream_endpoint_emits_final_event(self) -> None:
        from fastapi.testclient import TestClient
        from phase_mesh.service import app

        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/think/stream",
                json={
                    "text": "check 2 * 3 = 6",
                    "max_budget": 20,
                    "stream_interval": 10,
                },
            ) as response:
                text = "".join(response.iter_text())
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: final", text)

    def test_state_save_endpoint(self) -> None:
        from fastapi.testclient import TestClient
        from phase_mesh.service import app

        with TestClient(app) as client:
            response = client.post("/state/save")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "saved")


class FrontierCompareTests(unittest.TestCase):
    def test_flop_counters(self) -> None:
        from bench.common import count_mesh_flops, count_verifier_flops

        self.assertEqual(count_mesh_flops(8, 2), 4096)
        self.assertGreater(count_verifier_flops("Calculate: 11 * 13"), 0)

    def test_mesh_scoring_uses_decoded_output_not_prompt_truth(self) -> None:
        from bench.frontier_compare import ComparisonTask, score_mesh_decoded_output
        from phase_mesh.encoding import DecodedResonance

        task = ComparisonTask(
            id="math",
            suite="unit",
            kind="arithmetic",
            prompt="Calculate: 11 * 13\nReturn only the final number.",
            expected="143",
        )
        decoded = DecodedResonance(
            route="calculate",
            signature="abcdef123456",
            dominant_sector=1,
            confidence=0.8,
            sector_histogram=[0, 1],
        )
        passed, score_mode, score = score_mesh_decoded_output(task, decoded)
        self.assertFalse(passed)
        self.assertEqual(score_mode, "mesh-decoded-numeric-exact-match")
        self.assertEqual(score["numbers_found"], [])

    def test_frontier_compare_mesh_smoke(self) -> None:
        from bench.frontier_compare import run

        with TemporaryDirectory() as tmp:
            payload = run(
                out=tmp,
                math_count=1,
                context_tokens=[16],
                size=32,
                steps=50,
                max_budget=30,
                baseline="none",
            )
            out_path = Path(tmp)
            self.assertEqual(payload["task_count"], 2)
            self.assertEqual(payload["models"]["baseline"]["status"], "not_requested")
            self.assertEqual(payload["aggregates"]["phase_mesh"]["ok_records"], 2)
            self.assertIn("verifier_flops", payload["aggregates"]["phase_mesh"])
            self.assertTrue((out_path / "queries.jsonl").exists())
            self.assertTrue((out_path / "results.json").exists())
            self.assertTrue((out_path / "summary.md").exists())
            self.assertTrue((out_path / "phase_mesh_topology.q8.npz").exists())


if __name__ == "__main__":
    unittest.main()
