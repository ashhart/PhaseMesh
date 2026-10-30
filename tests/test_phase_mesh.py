from __future__ import annotations

import json
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

    def test_structured_arithmetic_encoder_preserves_factors_without_result_leak(self) -> None:
        from phase_mesh.encoding_structured import StructuredPhaseEncoder, parse_arithmetic

        parsed = parse_arithmetic("question: 8 plus 9")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.operation, "add")
        self.assertEqual(parsed.left, 8)
        self.assertEqual(parsed.right, 9)

        encoder = StructuredPhaseEncoder(32, 32)
        labels = [packet.label for packet in encoder.encode("8 plus 9")]
        self.assertIn("a:8", labels)
        self.assertIn("op:add", labels)
        self.assertIn("b:9", labels)
        self.assertNotIn("r:17", labels)

    def test_structured_arithmetic_encoder_result_hint_is_explicit_ablation(self) -> None:
        from phase_mesh.encoding_structured import StructuredPhaseEncoder

        encoder = StructuredPhaseEncoder(32, 32, include_result_hint=True)
        labels = [packet.label for packet in encoder.encode("8 plus 9")]
        self.assertIn("r:17", labels)

    def test_structured_arithmetic_feature_vector_does_not_depend_on_result_text(self) -> None:
        from phase_mesh.encoding_structured import structured_arithmetic_feature_vector

        base = structured_arithmetic_feature_vector("8 plus 9", 32)
        with_answer = structured_arithmetic_feature_vector("8 plus 9 answer 17", 32)
        different_right = structured_arithmetic_feature_vector("8 plus 10", 32)
        np.testing.assert_allclose(base, with_answer)
        self.assertGreater(float(np.linalg.norm(base - different_right)), 0.1)

    def test_arithmetic_representation_probe_smoke(self) -> None:
        from phase_mesh.probes import run_arithmetic_representation_probe

        result = run_arithmetic_representation_probe(
            encoder_mode="structured",
            max_value=2,
            ops=("add",),
            grid_size=16,
            basin_dim=16,
            hidden=8,
            steps_per_chunk=1,
            seed=3,
            backend="numpy",
        )
        self.assertEqual(result["encoder_mode"], "structured")
        self.assertIn("operation", result["probes"])
        self.assertIn("left", result["probes"])
        self.assertIn("right", result["probes"])
        self.assertIn("passed_representation_gate", result)

    def test_arithmetic_result_readout_probe_smoke(self) -> None:
        from phase_mesh.probes import run_arithmetic_result_readout_probe

        result = run_arithmetic_result_readout_probe(
            encoder_mode="structured",
            max_value=2,
            ops=("add", "mul"),
            grid_size=16,
            basin_dim=24,
            hidden=8,
            steps_per_chunk=1,
            seed=3,
            backend="numpy",
        )
        self.assertEqual(result["encoder_mode"], "structured")
        self.assertIn("factor_probes", result)
        self.assertIn("direct_result_probe", result)
        self.assertIn("factorized_result", result)
        self.assertIn("passed_result_gate", result)

    def test_solve_arithmetic_with_factor_readout_smoke(self) -> None:
        from phase_mesh.probes import solve_arithmetic_with_factor_readout

        result = solve_arithmetic_with_factor_readout(
            "2 times 3",
            max_value=3,
            ops=("add", "mul"),
            grid_size=16,
            basin_dim=24,
            hidden=8,
            steps_per_chunk=1,
            seed=3,
            backend="numpy",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["decoded"]["operation"], "mul")
        self.assertEqual(result["decoded"]["left"], "2")
        self.assertEqual(result["decoded"]["right"], "3")
        self.assertEqual(result["answer"], "6")

    def test_arithmetic_factor_readout_round_trip(self) -> None:
        from phase_mesh.probes import ArithmeticFactorReadout, fit_arithmetic_factor_readout

        readout, summary = fit_arithmetic_factor_readout(
            max_value=3,
            ops=("add", "mul"),
            grid_size=16,
            basin_dim=24,
            hidden=8,
            steps_per_chunk=1,
            seed=3,
            backend="numpy",
        )
        self.assertEqual(summary["rows"], 32)
        with TemporaryDirectory() as tmp:
            readout.save(tmp)
            loaded = ArithmeticFactorReadout.load(tmp)
            result = loaded.solve("2 times 3")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], "6")


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

    def test_basin_feature_and_residual_injection_update_field(self) -> None:
        config = MeshConfig(width=32, height=32, max_steps=40, min_steps=4, phase_pin_strength=0.25)
        mesh = PhaseFieldMesh(config)
        encoder = TextPhaseEncoder(32, 32)
        mesh.inject_text("alpha beta gamma", encoder)
        mesh.run_until_resonance(max_steps=8, min_steps=2)
        basin = mesh.find_basin(feature_dim=32)
        self.assertEqual(basin.center.shape, (32,))
        before = mesh.theta.copy()
        mesh.inject_residual("delta", encoder=encoder)
        self.assertGreater(float(np.mean(np.abs(mesh.theta - before))), 0.0)
        landscape_before = mesh.landscape.copy()
        mesh.reinforce_basin(basin, gain=0.02)
        self.assertGreater(float(np.mean(np.abs(mesh.landscape - landscape_before))), 0.0)


class PhaseModelTests(unittest.TestCase):
    def test_phase_model_train_and_generate_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=32,
            hidden=16,
            backend="numpy",
            create_decoder=True,
        )
        observation = model.observe_text("alpha beta gamma", steps_per_chunk=2)
        self.assertEqual(observation.target_token, "gamma")
        self.assertIsNotNone(observation.decoder_loss)
        steps = model.generate_steps("alpha beta", max_tokens=2, steps_per_token=2, top_k=4, top_p=0.9)
        self.assertEqual(len(steps), 2)
        self.assertTrue(all(step.token for step in steps))

    def test_repetition_penalty_pushes_recent_tokens_down(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import apply_repetition_penalty

        logits = torch.tensor([4.0, -1.0, 2.0])
        adjusted = apply_repetition_penalty(logits, [0, 1], penalty=2.0)
        self.assertLess(float(adjusted[0]), float(logits[0]))
        self.assertLess(float(adjusted[1]), float(logits[1]))
        self.assertEqual(float(adjusted[2]), float(logits[2]))

    def test_greedy_sampling_when_temperature_is_zero(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import sample_logits

        token_id, probability = sample_logits(torch.tensor([0.1, 3.0, 2.9]), temperature=0.0)
        self.assertEqual(token_id, 1)
        self.assertEqual(probability, 1.0)

    def test_contrastive_decoder_batch_training_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        target_id = model.vocab.add("17")
        model.vocab.add("16")
        model.vocab.add("18")
        loss = model.train_decoder_batch([[0.1] * 8, [0.2] * 8], [target_id, target_id], mode="contrastive")
        self.assertTrue(np.isfinite(loss))

    def test_reranker_batch_training_scores_positive_candidate(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            learning_rate=1e-2,
            create_decoder=True,
        )
        prompt = "question 8 plus 9 answer"
        positive_id = model.vocab.add("17")
        negative_id = model.vocab.add("16")
        metrics = {"loss": 0.0}
        for _ in range(40):
            basin, _prediction_error = model.encode_basin(prompt, steps_per_chunk=1)
            metrics = model.train_reranker_batch(
                [basin.center, basin.center],
                [positive_id, negative_id],
                [1, 0],
            )
        ranked = model.score_candidates(prompt, ["17", "16"], steps_per_chunk=1)

        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertEqual(ranked[0]["candidate"], "17")

    def test_structural_batch_training_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        basin_a, _error_a = model.encode_basin("8 plus 9", steps_per_chunk=1)
        basin_b, _error_b = model.encode_basin("9 plus 8", steps_per_chunk=1)
        target_id = model.vocab.add("17")
        alignment = model.reinforce_equivalence(basin_a, basin_b)
        metrics = model.train_structural_batch([basin_a.center], [basin_b.center], [target_id])

        self.assertGreaterEqual(alignment, -1.0)
        self.assertLessEqual(alignment, 1.0)
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertIn("structural_loss", metrics)
        self.assertIn("feature_l2", metrics)

    def test_structural_anchor_carves_and_persists_prototype(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel, structural_prototype_key

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        basin_a, _error_a = model.encode_basin("8 plus 9", steps_per_chunk=1)
        basin_b, _error_b = model.encode_basin("9 plus 8", steps_per_chunk=1)
        key = structural_prototype_key("8 plus 9", "9 plus 8", "17")
        metrics = model.reinforce_structural_anchor(basin_a, basin_b, prototype_key=key)

        self.assertEqual(key, "add:17")
        self.assertIn(key, model.structural_prototypes)
        self.assertEqual(model.structural_prototypes[key].shape, (8,))
        self.assertGreaterEqual(metrics["feature_l2"], 0.0)
        with TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = PhaseModel.load(tmp)

        self.assertIn(key, loaded.structural_prototypes)
        np.testing.assert_allclose(
            loaded.structural_prototypes[key],
            model.structural_prototypes[key],
        )

    def test_annealed_generate_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        model.vocab.add("17")
        steps = model.generate_steps("question 8 plus 9 answer", max_tokens=1, steps_per_token=1, anneal=True, anneal_steps=2)
        self.assertEqual(len(steps), 1)

    def test_phase_model_batch_training_and_eval(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel
        from phase_mesh.trainer import stream_evaluate, stream_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=32,
            hidden=16,
            backend="numpy",
            create_decoder=True,
        )
        with TemporaryDirectory() as tmp:
            summary = stream_train(
                model,
                ["alpha beta gamma", "def add return sum", "phase mesh topology"],
                steps_per_chunk=2,
                batch_size=2,
                out_dir=tmp,
                save_interval=0,
                train_decoder=True,
                freeze_omega=True,
                consolidate_interval=2,
                consolidate_cycles=1,
            )
            self.assertEqual(summary["chunks_seen"], 3)
            self.assertIsNotNone(summary["mean_decoder_loss"])
            metrics = model.evaluate_texts(["alpha beta gamma"], steps_per_chunk=2)
            self.assertEqual(metrics["scored_chunks"], 1)
            self.assertIsNotNone(metrics["perplexity"])
            window_metrics = stream_evaluate(
                model,
                ["alpha beta gamma delta"],
                steps_per_chunk=2,
                context_tokens=2,
                windows_per_chunk=3,
            )
            self.assertEqual(window_metrics["chunks"], 1)
            self.assertEqual(window_metrics["observations"], 3)
            self.assertEqual(window_metrics["scored_windows"], 3)
            self.assertIsNotNone(window_metrics["perplexity"])

    def test_training_windows_sample_next_token_contexts(self) -> None:
        from phase_mesh.model import PhaseModel
        from phase_mesh.trainer import iter_training_windows, stream_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=32,
            hidden=16,
            backend="numpy",
            create_decoder=False,
        )
        windows = list(
            iter_training_windows(
                model,
                "alpha beta gamma delta epsilon",
                context_tokens=2,
                max_windows=3,
                stride=1,
            )
        )
        self.assertEqual(windows, [["alpha", "beta"], ["beta", "gamma", "delta"], ["gamma", "delta", "epsilon"]])
        with TemporaryDirectory() as tmp:
            summary = stream_train(
                model,
                ["alpha beta gamma delta epsilon"],
                steps_per_chunk=1,
                context_tokens=2,
                windows_per_chunk=3,
                out_dir=tmp,
                save_interval=0,
                train_decoder=False,
                train_topology=False,
            )
        self.assertEqual(summary["chunks_seen"], 1)
        self.assertEqual(summary["observations"], 3)

    def test_math_qa_corpus_rows_keep_question_and_answer_together(self) -> None:
        from scripts.gen_math_qa import qa_lines

        rows = qa_lines(1, seed=1)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all("question" in row and "answer" in row for row in rows))

    def test_ranking_data_generator_emits_positive_and_negative_rows(self) -> None:
        from scripts.gen_ranking_data import ranking_rows

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "qa.txt"
            path.write_text("question 8 plus 9 answer 17\n", encoding="utf-8")
            rows = ranking_rows(path, k_neg=2, seed=1)

        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(1 for row in rows if row["label"] == 1), 1)
        self.assertEqual(sum(1 for row in rows if row["label"] == 0), 2)

    def test_structural_rows_expand_to_ranking_pairs(self) -> None:
        from phase_mesh.trainer import ranking_rows_from_any_row

        rows = list(ranking_rows_from_any_row({"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}))

        self.assertEqual(sum(1 for row in rows if row["label"] == 1), 2)
        self.assertGreaterEqual(sum(1 for row in rows if row["label"] == 0), 2)
        self.assertTrue(any(row["prompt"] == "question: 8 plus 9\nanswer:" for row in rows))
        self.assertTrue(any(row["prompt"] == "question: 9 plus 8\nanswer:" for row in rows))
        self.assertTrue(any(row["candidate"] == "17" and row["label"] == 1 for row in rows))

    def test_equivalence_data_generator_emits_commutative_rows(self) -> None:
        from scripts.gen_equiv_data import equivalence_rows

        rows = equivalence_rows(1, seed=1)

        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(all({"seq_a", "seq_b", "target"} <= set(row) for row in rows))
        self.assertTrue(any("plus" in row["seq_a"] and "plus" in row["seq_b"] for row in rows))

    def test_structural_stream_train_and_eval_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel
        from phase_mesh.trainer import stream_structural_evaluate, stream_structural_train

        rows = [
            {"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"},
            {"seq_a": "3 times 5", "seq_b": "5 times 3", "target": "15"},
        ]
        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        with TemporaryDirectory() as tmp:
            summary = stream_structural_train(
                model,
                rows,
                steps_per_chunk=1,
                batch_size=2,
                out_dir=tmp,
                save_interval=0,
            )
            metrics = stream_structural_evaluate(model, rows, steps_per_chunk=1)

        self.assertEqual(summary["rows_seen"], 2)
        self.assertIsNotNone(summary["mean_alignment"])
        self.assertEqual(metrics["rows"], 2)
        self.assertIn("mean_feature_l2", metrics)

    def test_structural_anchor_stream_train_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel
        from phase_mesh.trainer import stream_structural_train

        rows = [
            {"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"},
            {"seq_a": "3 times 5", "seq_b": "5 times 3", "target": "15"},
        ]
        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        with TemporaryDirectory() as tmp:
            summary = stream_structural_train(
                model,
                rows,
                steps_per_chunk=1,
                batch_size=2,
                out_dir=tmp,
                save_interval=0,
                anchor=True,
                freeze_decoder=True,
            )

        self.assertEqual(summary["mode"], "structural-anchor")
        self.assertEqual(summary["batches"], 0)
        self.assertEqual(summary["rows_seen"], 2)
        self.assertGreater(summary["prototype_count"], 0)
        self.assertIsNotNone(summary["mean_feature_mse"])

    def test_ranking_stream_can_sync_decoder_from_structural_rows(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel
        from phase_mesh.trainer import iter_ranking_jsonl, stream_ranking_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        with TemporaryDirectory() as tmp:
            data = Path(tmp) / "equiv.jsonl"
            data.write_text(
                json.dumps({"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}) + "\n",
                encoding="utf-8",
            )
            summary = stream_ranking_train(
                model,
                iter_ranking_jsonl(data),
                steps_per_chunk=1,
                batch_size=2,
                out_dir=tmp,
                save_interval=0,
                max_rows=4,
                train_decoder=True,
            )

        self.assertTrue(summary["decoder_synced"])
        self.assertIsNotNone(summary["mean_decoder_loss"])

    def test_prototype_decoder_stream_train_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel, structural_prototype_key
        from phase_mesh.trainer import stream_prototype_decoder_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        key = structural_prototype_key("8 plus 9", "9 plus 8", "17")
        model.structural_prototypes[key] = np.ones(8, dtype=np.float32) * 0.25
        rows = [{"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}]
        with TemporaryDirectory() as tmp:
            summary = stream_prototype_decoder_train(
                model,
                rows,
                batch_size=1,
                out_dir=tmp,
                save_interval=0,
                max_rows=1,
            )

        self.assertEqual(summary["rows_seen"], 1)
        self.assertEqual(summary["rows_used"], 1)
        self.assertEqual(summary["rows_skipped"], 0)
        self.assertIsNotNone(summary["mean_decoder_loss"])

    def test_structural_repulsion_stream_train_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel, structural_prototype_key
        from phase_mesh.trainer import repulsion_rows_from_any_row, stream_structural_repulsion_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        key = structural_prototype_key("8 plus 9", "9 plus 8", "17")
        model.structural_prototypes[key] = np.ones(8, dtype=np.float32) * 0.4
        model.structural_prototypes["add:16"] = np.ones(8, dtype=np.float32) * 0.1
        rows = list(repulsion_rows_from_any_row({"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}))
        with TemporaryDirectory() as tmp:
            summary = stream_structural_repulsion_train(
                model,
                rows,
                steps_per_chunk=1,
                out_dir=tmp,
                save_interval=0,
                max_rows=1,
            )

        self.assertEqual(summary["mode"], "structural-repulsion")
        self.assertEqual(summary["rows_used"], 1)
        self.assertIsNotNone(summary["nearest_prototype_target_accuracy_after"])

    def test_computational_distillation_stream_train_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel, structural_prototype_key
        from phase_mesh.trainer import stream_computational_distillation_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        key = structural_prototype_key("8 plus 9", "9 plus 8", "17")
        model.structural_prototypes[key] = np.ones(8, dtype=np.float32) * 0.4
        rows = [{"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}]
        with TemporaryDirectory() as tmp:
            summary = stream_computational_distillation_train(
                model,
                rows,
                steps_per_chunk=1,
                out_dir=tmp,
                save_interval=0,
                max_rows=1,
            )

        self.assertEqual(summary["mode"], "computational-distillation")
        self.assertGreaterEqual(summary["rows_used"], 1)
        self.assertIsNotNone(summary["mean_active_teacher_distance"])

    def test_guided_evolution_stream_train_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel, structural_prototype_key
        from phase_mesh.trainer import stream_guided_evolution_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        key = structural_prototype_key("8 plus 9", "9 plus 8", "17")
        model.structural_prototypes[key] = np.ones(8, dtype=np.float32) * 0.4
        rows = [{"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}]
        with TemporaryDirectory() as tmp:
            summary = stream_guided_evolution_train(
                model,
                rows,
                steps_per_chunk=1,
                out_dir=tmp,
                save_interval=0,
                max_rows=1,
                coupling=0.2,
            )

        self.assertEqual(summary["mode"], "guided-evolution")
        self.assertGreaterEqual(summary["rows_used"], 1)
        self.assertIsNotNone(summary["mean_guided_active_teacher_mse"])

    def test_phase_geometry_stream_train_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel, structural_prototype_key
        from phase_mesh.trainer import stream_phase_geometry_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        key = structural_prototype_key("8 plus 9", "9 plus 8", "17")
        model.structural_prototypes[key] = np.ones(8, dtype=np.float32) * 0.4
        rows = [{"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}]
        with TemporaryDirectory() as tmp:
            summary = stream_phase_geometry_train(
                model,
                rows,
                steps_per_chunk=1,
                out_dir=tmp,
                save_interval=0,
                max_rows=1,
                coupling=0.2,
                patch_size=3,
            )

        self.assertEqual(summary["mode"], "phase-geometry")
        self.assertGreaterEqual(summary["rows_used"], 1)
        self.assertIsNotNone(summary["mean_phase_geometry_active_teacher_mse"])

    def test_delta_geometry_stream_train_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel, structural_prototype_key
        from phase_mesh.trainer import stream_delta_geometry_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        key = structural_prototype_key("8 plus 9", "9 plus 8", "17")
        model.structural_prototypes[key] = np.ones(8, dtype=np.float32) * 0.4
        rows = [{"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}]
        with TemporaryDirectory() as tmp:
            summary = stream_delta_geometry_train(
                model,
                rows,
                steps_per_chunk=1,
                out_dir=tmp,
                save_interval=0,
                max_rows=1,
                coupling=0.2,
                patch_size=3,
            )

        self.assertEqual(summary["mode"], "delta-geometry")
        self.assertGreaterEqual(summary["rows_used"], 1)
        self.assertIsNotNone(summary["mean_delta_active_target_distance"])

    def test_frozen_delta_geometry_leaves_target_prototype_unchanged(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel, structural_prototype_key
        from phase_mesh.trainer import stream_delta_geometry_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        key = structural_prototype_key("8 plus 9", "9 plus 8", "17")
        original = np.ones(8, dtype=np.float32) * 0.4
        model.structural_prototypes[key] = original.copy()
        rows = [{"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}]
        with TemporaryDirectory() as tmp:
            summary = stream_delta_geometry_train(
                model,
                rows,
                steps_per_chunk=1,
                out_dir=tmp,
                save_interval=0,
                max_rows=1,
                coupling=0.2,
                patch_size=3,
                freeze_targets=True,
            )

        self.assertEqual(summary["mode"], "delta-geometry-frozen")
        self.assertTrue(summary["targets_frozen"])
        np.testing.assert_allclose(model.structural_prototypes[key], original)

    def test_residual_tunnel_stream_train_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel, structural_prototype_key
        from phase_mesh.trainer import stream_residual_tunnel_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        key = structural_prototype_key("8 plus 9", "9 plus 8", "17")
        original = np.ones(8, dtype=np.float32) * 0.4
        model.structural_prototypes[key] = original.copy()
        rows = [{"seq_a": "8 plus 9", "seq_b": "9 plus 8", "target": "17"}]
        with TemporaryDirectory() as tmp:
            summary = stream_residual_tunnel_train(
                model,
                rows,
                steps_per_chunk=1,
                out_dir=tmp,
                save_interval=0,
                max_rows=1,
                tunnel_strength=0.05,
            )

        self.assertEqual(summary["mode"], "residual-tunnel")
        self.assertTrue(summary["targets_frozen"])
        self.assertGreaterEqual(summary["rows_used"], 1)
        self.assertIsNotNone(summary["mean_tunnel_active_target_distance_after"])
        np.testing.assert_allclose(model.structural_prototypes[key], original)

    def test_push_pull_stream_train_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel
        from phase_mesh.trainer import stream_push_pull_train

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=True,
        )
        correct_key = "add:17"
        wrong_key = "add:16"
        original = np.ones(8, dtype=np.float32) * 0.4
        model.structural_prototypes[correct_key] = original.copy()
        model.structural_prototypes[wrong_key] = -original.copy()
        before_feature = model.feature_omega.copy()
        rows = [
            {
                "prompt": "8 plus 9",
                "correct_key": correct_key,
                "target": "17",
                "wrong_keys": [wrong_key],
            }
        ]
        with TemporaryDirectory() as tmp:
            summary = stream_push_pull_train(
                model,
                rows,
                steps_per_chunk=1,
                out_dir=tmp,
                save_interval=0,
                max_rows=1,
                push_pull_strength=0.05,
                wrong_strength=0.5,
            )

        self.assertEqual(summary["mode"], "push-pull")
        self.assertTrue(summary["targets_frozen"])
        self.assertEqual(summary["rows_used"], 1)
        self.assertIsNotNone(summary["mean_push_pull_active_target_distance_after"])
        self.assertGreater(np.linalg.norm(model.feature_omega - before_feature), 0.0)
        np.testing.assert_allclose(model.structural_prototypes[correct_key], original)

    def test_sparse_tunnel_updates_only_active_feature_cell(self) -> None:
        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=False,
        )
        correct_key = "add:17"
        wrong_key = "add:16"
        model.structural_prototypes[correct_key] = np.ones(8, dtype=np.float32) * 0.4
        model.structural_prototypes[wrong_key] = -np.ones(8, dtype=np.float32) * 0.2
        basin, _error = model.encode_basin("8 plus 9", steps_per_chunk=1, reset=True)
        before_feature = model.feature_omega.copy()
        before_landscape = model.field.landscape.copy()
        before_omega = model.field.omega.copy()

        metrics = model.carve_sparse_tunnel(
            basin,
            model.structural_prototypes[correct_key],
            [model.structural_prototypes[wrong_key]],
            correct_key=correct_key,
            strength=0.05,
            wrong_strength=0.5,
        )

        changed = np.argwhere(np.linalg.norm(model.feature_omega - before_feature, axis=2) > 0.0)
        self.assertTrue(metrics["used"])
        self.assertEqual(changed.shape[0], 1)
        self.assertEqual((int(changed[0][0]), int(changed[0][1])), (basin.y, basin.x))
        np.testing.assert_allclose(model.field.landscape, before_landscape)
        np.testing.assert_allclose(model.field.omega, before_omega)
        self.assertFalse(metrics["global_landscape_updated"])

    def test_resonance_slot_updates_only_keyed_active_slot(self) -> None:
        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            num_slots=8,
            create_decoder=False,
        )
        basin, _error = model.encode_basin("8 plus 9", steps_per_chunk=1, reset=True)
        correct_key = "add:17"
        wrong_key = "add:16"
        correct = basin.center.copy()
        correct[0] += 1.0
        wrong = basin.center.copy()
        wrong[0] += 0.1
        model.structural_prototypes[correct_key] = correct.astype(np.float32)
        model.structural_prototypes[wrong_key] = wrong.astype(np.float32)

        before_feature = model.feature_omega.copy()
        self.assertIsNotNone(model.feature_slots)
        before_slots = model.feature_slots.copy()
        before_landscape = model.field.landscape.copy()
        before_omega = model.field.omega.copy()
        slot_idx = model.get_slot_index_for_key(correct_key)

        nearest_before = model.nearest_structural_prototype(basin.center, operation="add", k=1)
        self.assertEqual(nearest_before[0]["key"], wrong_key)

        metrics = model.carve_resonance_slot(
            basin,
            model.structural_prototypes[correct_key],
            [model.structural_prototypes[wrong_key]],
            correct_key=correct_key,
            strength=1.0,
            wrong_strength=0.0,
        )

        self.assertTrue(metrics["used"])
        self.assertEqual(metrics["slot_idx"], slot_idx)
        changed = np.argwhere(np.linalg.norm(model.feature_slots - before_slots, axis=3) > 0.0)
        self.assertEqual(changed.shape[0], 1)
        self.assertEqual((int(changed[0][0]), int(changed[0][1]), int(changed[0][2])), (basin.y, basin.x, slot_idx))
        np.testing.assert_allclose(model.feature_omega, before_feature)
        np.testing.assert_allclose(model.field.landscape, before_landscape)
        np.testing.assert_allclose(model.field.omega, before_omega)
        self.assertFalse(metrics["global_landscape_updated"])
        self.assertIn(metrics["slot_key"], model.feature_slot_gates)

        nearest_after = model.nearest_structural_prototype(basin.center, operation="add", k=1)
        self.assertEqual(nearest_after[0]["key"], correct_key)

        with TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = PhaseModel.load(tmp, load_decoder=False)
        self.assertEqual(loaded.num_slots, model.num_slots)
        self.assertTrue(loaded.feature_slot_overrides)
        self.assertTrue(loaded.feature_slot_gates)
        loaded._last_basin_cell = (basin.y, basin.x)
        loaded_after = loaded.nearest_structural_prototype(basin.center, operation="add", k=1)
        self.assertEqual(loaded_after[0]["key"], correct_key)

    def test_prompt_gated_slots_prevent_candidate_self_match(self) -> None:
        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            num_slots=8,
            create_decoder=False,
        )
        basin, _error = model.encode_basin("8 plus 9", steps_per_chunk=1, reset=True)
        active = basin.center.astype(np.float32)
        correct_key = "add:17"
        wrong_key = "add:20"
        correct = active.copy()
        correct[0] += 1.0
        wrong = active.copy()
        wrong[1] += 1.0
        model.structural_prototypes[correct_key] = correct.astype(np.float32)
        model.structural_prototypes[wrong_key] = wrong.astype(np.float32)
        y, x = int(basin.y), int(basin.x)
        correct_slot_key = model.feature_slot_override_key(y, x, correct_key)
        wrong_slot_key = model.feature_slot_override_key(y, x, wrong_key)

        # Both candidate residuals can perfectly self-match their own prototype.
        model.feature_slot_overrides[correct_slot_key] = (correct - active).astype(np.float32)
        model.feature_slot_overrides[wrong_slot_key] = (wrong - active).astype(np.float32)
        # Only the correct slot has a gate signature matching this prompt.
        model.feature_slot_gates[correct_slot_key] = active.copy()
        model.feature_slot_gates[wrong_slot_key] = (-active).astype(np.float32)
        model._last_basin_cell = (y, x)

        nearest = model.nearest_structural_prototype(active, operation="add", k=1)
        self.assertEqual(nearest[0]["key"], correct_key)
        readout = model.read_resonant_slots(active)
        self.assertEqual(readout["mode"], "exact-gated")
        self.assertGreater(max(readout["weights"]), 0.5)

    def test_contrastive_gate_training_selects_correct_slot(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=16,
            backend="numpy",
            num_slots=8,
            create_decoder=False,
        )
        basin, _error = model.encode_basin("8 plus 9", steps_per_chunk=1, reset=True)
        correct_key = "add:17"
        wrong_keys = ["add:16", "add:18", "add:20"]
        model.learning_rate = 5e-2
        model._init_gate()

        first = model.train_gate_contrastive(
            basin.center,
            correct_key=correct_key,
            wrong_keys=wrong_keys,
        )
        latest = first
        for _ in range(80):
            latest = model.train_gate_contrastive(
                basin.center,
                correct_key=correct_key,
                wrong_keys=wrong_keys,
            )

        self.assertGreater(latest["correct_probability"], first["correct_probability"])
        self.assertTrue(latest["top_match"])
        self.assertEqual(latest["correct_slot"], model.get_slot_index_for_key(correct_key))

    def test_delta_contrastive_training_scores_correct_candidate(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=16,
            backend="numpy",
            create_decoder=False,
        )
        active = np.zeros(8, dtype=np.float32)
        target = np.asarray([0.05, 0.02, 0.01, 0.0, -0.01, 0.0, 0.01, -0.02], dtype=np.float32)
        wrongs = [
            np.asarray([0.8, -0.3, 0.2, 0.6, -0.5, 0.4, 0.1, -0.7], dtype=np.float32),
            np.asarray([-0.6, 0.7, -0.4, 0.3, 0.5, -0.2, 0.8, 0.1], dtype=np.float32),
        ]
        model.learning_rate = 5e-2
        model._init_delta_scorer()

        first = model.train_delta_contrastive(active, target, wrongs, margin=0.5)
        latest = first
        for _ in range(80):
            latest = model.train_delta_contrastive(active, target, wrongs, margin=0.5)

        self.assertGreater(latest["delta_margin"], first["delta_margin"])
        self.assertTrue(latest["top_match"])
        scores = model.score_delta_candidates(active, np.asarray([target, *wrongs], dtype=np.float32))
        self.assertEqual(int(np.argmax(scores)), 0)

    def test_joint_stability_scoring_restores_field_state(self) -> None:
        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=False,
        )
        model.field.inject_text("persistent state", model.encoder)
        for _ in range(3):
            model.field.step()
        theta_before = model.field.theta.copy()
        velocity_before = model.field.velocity.copy()
        step_before = model.field.step_index

        scored = model.score_joint_candidates(
            "question: 8 plus 9",
            ["add:17", "add:20"],
            steps_per_chunk=3,
            settle_tail=2,
        )

        self.assertIn(scored["best_key"], {"add:17", "add:20"})
        self.assertEqual(len(scored["scores"]), 2)
        self.assertEqual(model.field.step_index, step_before)
        self.assertTrue(np.allclose(model.field.theta, theta_before))
        self.assertTrue(np.allclose(model.field.velocity, velocity_before))

    def test_computation_manifold_updates_global_road_without_prototype_drift(self) -> None:
        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=8,
            hidden=8,
            backend="numpy",
            create_decoder=False,
        )
        correct_key = "add:17"
        wrong_key = "add:16"
        correct = np.ones(8, dtype=np.float32) * 0.4
        wrong = -np.ones(8, dtype=np.float32) * 0.2
        model.structural_prototypes[correct_key] = correct.copy()
        model.structural_prototypes[wrong_key] = wrong.copy()
        before_landscape = model.field.landscape.copy()
        before_omega = model.field.omega.copy()
        before_correct = model.structural_prototypes[correct_key].copy()

        metrics = model.carve_computation_manifold(
            correct_key=correct_key,
            wrong_keys=[wrong_key],
            strength=0.05,
        )

        self.assertTrue(metrics["used"])
        self.assertTrue(metrics["global_landscape_updated"])
        self.assertFalse(metrics["prototype_updated"])
        self.assertGreater(np.linalg.norm(model.field.landscape - before_landscape), 0.0)
        self.assertGreater(np.linalg.norm(model.field.omega - before_omega), 0.0)
        np.testing.assert_allclose(model.structural_prototypes[correct_key], before_correct)

    def test_phase_model_topology_only_without_decoder(self) -> None:
        from phase_mesh.model import PhaseModel

        model = PhaseModel(
            grid_size=16,
            vocab_capacity=64,
            basin_dim=32,
            backend="numpy",
            create_decoder=False,
        )
        observation = model.observe_text(
            "topology only stream",
            steps_per_chunk=2,
            train_decoder=False,
        )
        self.assertIsNone(observation.decoder_loss)
        self.assertGreaterEqual(observation.mean_prediction_error, 0.0)

    def test_basin_entropy_counts_records(self) -> None:
        from phase_mesh.trainer import basin_entropy, basin_target_mutual_info

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            rows = [
                {"basin": {"x": 1, "y": 1}, "target_token": "a"},
                {"basin": {"x": 1, "y": 1}, "target_token": "a"},
                {"basin": {"x": 2, "y": 2}, "target_token": "b"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            entropy = basin_entropy(path, bins=4)
            mutual_info = basin_target_mutual_info(path, bins=4)

        self.assertEqual(entropy["records"], 3)
        self.assertEqual(entropy["unique_basins"], 2)
        self.assertGreater(entropy["normalized_entropy"], 0.0)
        self.assertEqual(mutual_info["records"], 3)
        self.assertGreater(mutual_info["normalized_mutual_info"], 0.0)

    def test_ablation_runner_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from bench.ablations import run

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "train.txt"
            heldout = base / "heldout.txt"
            data.write_text(
                "\n".join(
                    [
                        "question 1 plus 2 answer 3",
                        "question 2 plus 2 answer 4",
                        "question 3 plus 2 answer 5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            heldout.write_text("question 4 plus 2 answer 6\n", encoding="utf-8")
            payload = run(
                data=data,
                heldout=heldout,
                out=base / "ablations",
                modes=("full", "static-topology"),
                chunks=2,
                eval_chunks=1,
                size=16,
                basin_dim=8,
                hidden=8,
                vocab_capacity=64,
                steps_per_chunk=1,
                batch_size=2,
                context_tokens=3,
                windows_per_chunk=1,
                seed=3,
                backend="numpy",
            )

        self.assertEqual(payload["suite"], "phase_mesh_ablation_matrix")
        self.assertIn("full", payload["results"])
        self.assertIn("static-topology", payload["comparisons"])
        self.assertIn("basin_target_mutual_info", payload["results"]["full"])
        self.assertIsNotNone(payload["results"]["full"]["heldout"]["perplexity"])


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


class RegistryDomainTests(unittest.TestCase):
    def test_code_domain_extracts_python_facts(self) -> None:
        from phase_mesh.domains import CodeDomain

        result = CodeDomain().solve("def add(a, b):\n    return a + b")
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.data["syntax_ok"])
        self.assertEqual(result.data["functions"][0]["name"], "add")
        self.assertIn("a", result.data["functions"][0]["args"])

    def test_code_domain_factor_readout_round_trips(self) -> None:
        from phase_mesh.domains import CodeDomain

        with TemporaryDirectory() as tmp:
            domain = CodeDomain()
            fit = domain.fit(tmp)
            loaded = CodeDomain.load(tmp)
            probe = loaded.probe()
            solved = loaded.solve("def add(x, y):\n    return x + y")

        self.assertTrue(fit.metrics["passed_factor_gate"])
        self.assertTrue(probe.passed)
        self.assertIn("factor_readout", solved.data)
        self.assertEqual(solved.data["factor_readout"]["kind"], "function")
        self.assertEqual(solved.data["factor_readout"]["primary"], "add")

    def test_json_domain_factor_readout_round_trips(self) -> None:
        from phase_mesh.domains import JsonDomain

        with TemporaryDirectory() as tmp:
            domain = JsonDomain()
            fit = domain.fit(tmp)
            loaded = JsonDomain.load(tmp)
            probe = loaded.probe()
            solved = loaded.solve('{"ok": true}')

        self.assertTrue(fit.metrics["passed_factor_gate"])
        self.assertTrue(probe.passed)
        self.assertEqual(solved.answer, "object")
        self.assertIn("factor_readout", solved.data)
        self.assertEqual(solved.data["factor_readout"]["root_type"], "object")
        self.assertEqual(solved.data["factor_readout"]["key_signature"], "ok")

    def test_tool_domain_routes_core_domains(self) -> None:
        from phase_mesh.domains import ToolDomain

        tool = ToolDomain()
        self.assertEqual(tool.solve("8 plus 9").answer, "arithmetic")
        self.assertEqual(tool.solve("remember project: PhaseMesh").answer, "memory")
        self.assertEqual(tool.solve("def add(a, b):\n    return a + b").answer, "code")
        self.assertEqual(tool.solve('{"ok": true}').answer, "json")

    def test_memory_domain_persists_facts(self) -> None:
        from phase_mesh.domains import MemoryDomain

        with TemporaryDirectory() as tmp:
            memory = MemoryDomain(artifact_dir=tmp)
            remembered = memory.solve("remember project: PhaseMesh")
            loaded = MemoryDomain.load(tmp)
            recalled = loaded.solve("recall project")
        self.assertEqual(remembered.status, "ok")
        self.assertIn("PhaseMesh", recalled.answer)

    def test_phase_mesh_registry_fit_probe_and_solve(self) -> None:
        from phase_mesh.domains import ArithmeticDomain, CodeDomain, JsonDomain, MemoryDomain, ToolDomain
        from phase_mesh.registry import PhaseMeshRegistry, render_domain_report

        with TemporaryDirectory() as tmp:
            registry = PhaseMeshRegistry(
                domains={
                    "arithmetic": ArithmeticDomain(
                        max_value=3,
                        grid_size=16,
                        basin_dim=24,
                        hidden=8,
                        steps_per_chunk=1,
                        backend="numpy",
                    ),
                    "code": CodeDomain(),
                    "json": JsonDomain(),
                    "memory": MemoryDomain(),
                    "tool": ToolDomain(),
                }
            )
            fit = registry.fit(tmp, domains=["arithmetic", "code", "json", "memory", "tool"])
            loaded = PhaseMeshRegistry.load(tmp)
            probe = loaded.probe(domains=["code", "json", "memory", "tool"])

            report = render_domain_report(probe, artifact_dir=tmp)
            solved = loaded.solve("2 times 3")
            solved_json = loaded.solve('{"ok": true}')
        self.assertEqual(fit["type"], "phase-mesh-domain-registry")
        self.assertTrue(probe["passed"])
        self.assertIn("| Domain | Gate | Key Metrics |", report)
        self.assertIn("Status: **PASS**", report)
        self.assertEqual(solved["domain"], "arithmetic")
        self.assertEqual(solved["answer"], "6")
        self.assertEqual(solved_json["domain"], "json")
        self.assertEqual(solved_json["answer"], "object")

    def test_lab_demo_html_renders_gates_and_limits(self) -> None:
        from phase_mesh.lab_demo import render_lab_demo_html

        html = render_lab_demo_html({
            "status": "pass",
            "elapsed_s": 1.2,
            "artifact_sizes": {"registry_bytes": 2048},
            "config": {"size": 16, "context_tokens": [16]},
            "probe": {
                "passed": True,
                "domains": {
                    "json": {
                        "passed": True,
                        "metrics": {
                            "factor_mean_accuracy": 1.0,
                            "exact_json_accuracy": 1.0,
                        },
                    }
                },
            },
            "context_sweep": [
                {
                    "token_count": 16,
                    "gradient": 0.01,
                    "coherence": 0.99,
                    "elapsed_s": 0.1,
                    "passed": True,
                }
            ],
            "controls": {
                "context_pin_off": [
                    {
                        "token_count": 16,
                        "gradient": 0.12,
                        "passed": False,
                    }
                ],
                "context_pin_on": [
                    {
                        "token_count": 16,
                        "gradient": 0.01,
                        "passed": True,
                    }
                ],
                "gradient_reduction": 12.0,
                "separation_passed": True,
            },
            "solves": [
                {
                    "label": "json",
                    "prompt": "{\"ok\": true}",
                    "result": {
                        "answer": "object",
                        "result": {"data": {"factor_readout": {"root_type": "object"}}},
                    },
                }
            ],
            "claims": ["This is not a general LLM claim."],
        })

        self.assertIn("PhaseMesh Lab Demo", html)
        self.assertIn("Domain Gates", html)
        self.assertIn("Pinning Ablation", html)
        self.assertIn("12.0x", html)
        self.assertIn("This is not a general LLM claim.", html)

    def test_phase_accio_sketch_retrieves_bound_candidate(self) -> None:
        from phase_mesh.phase_accio import PhaseAccioSketch

        sketch = PhaseAccioSketch(grid_size=32, slots_per_symbol=10, pin_strength=0.25, filler_noise=0.0)
        sketch.ingest("the audit memo linked k_abc123def0 beside marker v_0123456789abcdef after review")
        ranked = sketch.rank("k_abc123def0", ["v_fedcba9876543210", "v_0123456789abcdef", "v_1111111111111111"])

        self.assertEqual(ranked[0]["candidate"], "v_0123456789abcdef")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])

        pin_off = PhaseAccioSketch(grid_size=32, slots_per_symbol=10, pin_strength=0.0, filler_noise=0.0)
        pin_off.ingest("the audit memo linked k_abc123def0 beside marker v_0123456789abcdef after review")
        self.assertTrue(all(abs(item["score"]) < 1e-8 for item in pin_off.rank("k_abc123def0", ["v_fedcba9876543210", "v_0123456789abcdef"])))

    def test_phase_accio_artifact_has_pin_ablation(self) -> None:
        from phase_mesh.phase_accio import run_phase_accio

        with TemporaryDirectory() as tmp:
            payload = run_phase_accio(
                out_dir=tmp,
                context_tokens=256,
                needles=12,
                candidates=6,
                seeds=2,
                grid_size=32,
                slots_per_symbol=10,
                pin_strength=0.25,
                filler_noise=0.0,
                context_style="natural",
            )
            out_path = Path(tmp)
            self.assertTrue((out_path / "summary.json").exists())
            self.assertTrue((out_path / "summary.md").exists())
            self.assertTrue((out_path / "index.html").exists())
            html = (out_path / "index.html").read_text(encoding="utf-8")

        self.assertEqual(payload["status"], "pass")
        self.assertGreaterEqual(payload["summary"]["pin_on"]["accuracy"], 0.9)
        self.assertLessEqual(payload["summary"]["pin_off"]["accuracy"], 0.35)
        self.assertLessEqual(payload["summary"]["scrambled"]["accuracy"], 0.5)
        self.assertEqual(payload["summary"]["hash_map"]["accuracy"], 1.0)
        self.assertIn("collapse_series", payload)
        self.assertIn("PhaseAccio", html)

    def test_phase_advantage_artifact_has_corruption_controls(self) -> None:
        from phase_mesh.phase_advantage import run_phase_advantage

        with TemporaryDirectory() as tmp:
            payload = run_phase_advantage(
                out_dir=tmp,
                seed=3,
                items=80,
                key_length=8,
                vocab_size=400,
                candidates=8,
                trials=30,
                memory_size=1024,
                slots=3,
            )
            out_path = Path(tmp)
            self.assertTrue((out_path / "summary.json").exists())
            self.assertTrue((out_path / "summary.md").exists())
            self.assertTrue((out_path / "index.html").exists())
            html = (out_path / "index.html").read_text(encoding="utf-8")

        self.assertEqual(payload["type"], "phase-mesh-advantage-probes")
        self.assertIn("corruption_curve", payload)
        self.assertIn("capacity_curve", payload)
        self.assertIn("segmentation", payload)
        self.assertEqual(payload["corruption_curve"]["by_rate"]["0.30"]["exact_hash"]["accuracy"], 0.0)
        self.assertLessEqual(payload["corruption_curve"]["by_rate"]["0.30"]["whole_key_phase"]["accuracy"], 0.4)
        self.assertGreaterEqual(payload["segmentation"]["coupled"]["pair_accuracy"], 0.8)
        self.assertIn("Corrupted-Key Pattern Completion", html)

    def test_phase_advantage_docs_writes_natural_dashboard(self) -> None:
        from phase_mesh.phase_advantage_docs import run_phase_advantage_docs

        with TemporaryDirectory() as tmp:
            payload = run_phase_advantage_docs(
                out_dir=tmp,
                context_tokens=4096,
                records=48,
                candidates=6,
                trials=24,
                corruption_rates=[0.0, 0.3],
                phase_cells=1024,
                slots=3,
                seed=5,
                skip_architecture=True,
            )
            out_path = Path(tmp)
            self.assertTrue((out_path / "summary.json").exists())
            self.assertTrue((out_path / "summary.md").exists())
            self.assertTrue((out_path / "index.html").exists())
            self.assertTrue((out_path / "context_sample.txt").exists())
            html = (out_path / "index.html").read_text(encoding="utf-8")

        self.assertEqual(payload["type"], "phase-mesh-natural-document-advantage")
        self.assertEqual(payload["context"]["actual_tokens"], 4096)
        self.assertIn("bm25", payload["baselines"])
        self.assertIn("vector_faiss", payload["baselines"])
        self.assertEqual(payload["corruption_curve"]["by_rate"]["0.30"]["exact_hash"]["accuracy"], 0.0)
        self.assertIn("Control Collapse Live", html)

    def test_phase_binding_hard_writes_role_binding_dashboard(self) -> None:
        from phase_mesh.phase_binding_hard import run_phase_binding_hard

        with TemporaryDirectory() as tmp:
            payload = run_phase_binding_hard(
                out_dir=tmp,
                records=48,
                candidates=6,
                trials=24,
                corruption_rates=[0.0, 0.3],
                phase_cells=1024,
                slots=3,
                context_tokens=4096,
                seed=5,
            )
            out_path = Path(tmp)
            self.assertTrue((out_path / "summary.json").exists())
            self.assertTrue((out_path / "summary.md").exists())
            self.assertTrue((out_path / "index.html").exists())
            html = (out_path / "index.html").read_text(encoding="utf-8")

        self.assertEqual(payload["type"], "phase-mesh-adversarial-role-binding")
        self.assertEqual(payload["context"]["actual_tokens"], 4096)
        self.assertIn("role_phase", payload["baselines"])
        self.assertGreater(payload["curve"]["by_rate"]["0.30"]["role_phase"]["accuracy"], 0.5)
        self.assertGreater(payload["curve"]["role_vs_bm25_at_30"], 1.0)
        self.assertIn("Hard Role-Binding", html)

    def test_phase_binding_recoverable_signature_hits_exact_ceiling(self) -> None:
        from phase_mesh.phase_binding_hard import run_phase_binding_hard

        with TemporaryDirectory() as tmp:
            payload = run_phase_binding_hard(
                out_dir=tmp,
                records=48,
                candidates=6,
                trials=24,
                corruption_rates=[0.0, 0.3],
                phase_cells=2048,
                slots=4,
                context_tokens=4096,
                seed=5,
                corruption_mode="recoverable-signature",
                ecc_readout=True,
            )

        row = payload["curve"]["by_rate"]["0.30"]
        self.assertEqual(payload["config"]["corruption_mode"], "recoverable-signature")
        self.assertTrue(payload["config"]["ecc_readout"])
        self.assertEqual(row["role_phase"]["accuracy"], 1.0)
        self.assertEqual(row["exact_hash"]["accuracy"], 0.0)
        self.assertEqual(row["exact_bloom"]["accuracy"], 0.0)

    def test_phase_binding_ecc_signature_hits_forced_answer_ceiling(self) -> None:
        from phase_mesh.phase_binding_hard import run_phase_binding_hard

        with TemporaryDirectory() as tmp:
            payload = run_phase_binding_hard(
                out_dir=tmp,
                records=48,
                candidates=6,
                trials=24,
                corruption_rates=[0.0, 0.3],
                phase_cells=2048,
                slots=4,
                context_tokens=4096,
                seed=17,
                corruption_mode="ecc-signature",
                ecc_readout=True,
            )

        row = payload["curve"]["by_rate"]["0.30"]
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["config"]["corruption_mode"], "ecc-signature")
        self.assertTrue(payload["config"]["ecc_readout"])
        self.assertFalse(payload["config"]["safe_abstain"])
        self.assertEqual(row["role_phase"]["accuracy"], 1.0)
        self.assertEqual(row["exact_hash"]["accuracy"], 0.0)
        self.assertEqual(row["exact_bloom"]["accuracy"], 0.0)

    def test_phase_binding_safe_abstain_reports_no_wrong_decisions(self) -> None:
        from phase_mesh.phase_binding_hard import run_phase_binding_hard

        with TemporaryDirectory() as tmp:
            payload = run_phase_binding_hard(
                out_dir=tmp,
                records=48,
                candidates=6,
                trials=24,
                corruption_rates=[0.0, 0.3],
                phase_cells=2048,
                slots=4,
                context_tokens=4096,
                seed=5,
                corruption_mode="arbitrary",
                ecc_readout=True,
                safe_abstain=True,
                abstain_margin=0.008,
            )

        safe = payload["curve"]["safe_decision_by_rate"]["0.30"]["role_phase"]
        row = payload["curve"]["by_rate"]["0.30"]
        self.assertTrue(payload["config"]["safe_abstain"])
        self.assertEqual(safe["no_wrong_rate"], 1.0)
        self.assertEqual(safe["wrong_answered"], 0)
        self.assertGreaterEqual(safe["coverage"], 0.95)
        self.assertEqual(row["exact_hash"]["accuracy"], 0.0)
        self.assertEqual(row["exact_bloom"]["accuracy"], 0.0)

    def test_learnable_core_probe_writes_gradient_artifact(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("PyTorch is not installed")

        from phase_mesh.learnable_core import run_learnable_core_probe

        with TemporaryDirectory() as tmp:
            payload = run_learnable_core_probe(
                out_dir=tmp,
                sequence_length=8,
                train_size=64,
                test_size=32,
                epochs=1,
                batch_size=16,
                oscillators=8,
                hidden=8,
                seed=1,
            )
            out_path = Path(tmp)
            self.assertTrue((out_path / "summary.json").exists())
            self.assertTrue((out_path / "summary.md").exists())

        gradients = payload["results"]["learned_phase"]["gradient_probe"]
        self.assertGreater(gradients["token_phase"], 0.0)
        self.assertGreater(gradients["omega"], 0.0)
        self.assertGreater(gradients["coupling"], 0.0)
        self.assertEqual(payload["type"], "phase-mesh-learnable-core-probe")


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
