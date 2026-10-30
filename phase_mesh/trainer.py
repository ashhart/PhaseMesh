from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .model import (
    PhaseModel,
    cosine_similarity,
    infer_operation_type,
    perplexity_from_loss,
    structural_feature_l2,
    structural_feature_mse,
    structural_prototype_key,
)


def iter_text_file(path: str | Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield stripped


def iter_ranking_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield prompt/candidate/label rows from JSONL or a JSON array."""

    source = Path(path).read_text(encoding="utf-8").strip()
    if not source:
        return
    if source.startswith("["):
        rows = json.loads(source)
        for row in rows:
            yield from ranking_rows_from_any_row(row)
        return
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        yield from ranking_rows_from_any_row(json.loads(stripped))


def iter_structural_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield seq_a/seq_b/target equivalence triples from JSONL or a JSON array."""

    source = Path(path).read_text(encoding="utf-8").strip()
    if not source:
        return
    if source.startswith("["):
        rows = json.loads(source)
        for row in rows:
            yield normalize_structural_row(row)
        return
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        yield normalize_structural_row(json.loads(stripped))


def iter_repulsion_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield prompt/correct/wrong prototype rows from JSONL or a JSON array."""

    source = Path(path).read_text(encoding="utf-8").strip()
    if not source:
        return
    if source.startswith("["):
        rows = json.loads(source)
        for row in rows:
            yield from repulsion_rows_from_any_row(row)
        return
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        yield from repulsion_rows_from_any_row(json.loads(stripped))


def normalize_ranking_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"]).strip()
    candidate = str(row["candidate"]).strip()
    label = int(row["label"])
    if label not in {0, 1}:
        raise ValueError("ranking labels must be 0 or 1")
    if not prompt or not candidate:
        raise ValueError("ranking rows need non-empty prompt and candidate")
    return {"prompt": prompt, "candidate": candidate, "label": label}


def ranking_rows_from_any_row(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Expand native ranking rows or structural equivalence rows into ranking rows."""

    if {"prompt", "candidate", "label"} <= set(row):
        yield normalize_ranking_row(row)
        return
    if {"seq_a", "seq_b", "target"} <= set(row):
        normalized = normalize_structural_row(row)
        prompts = [
            f"question: {normalized['seq_a']}\nanswer:",
            f"question: {normalized['seq_b']}\nanswer:",
        ]
        target = normalized["target"]
        for prompt in prompts:
            yield {"prompt": prompt, "candidate": target, "label": 1}
            for candidate in ranking_negative_candidates(target):
                yield {"prompt": prompt, "candidate": candidate, "label": 0}
        return
    raise ValueError("ranking rows need prompt/candidate/label or seq_a/seq_b/target")


def ranking_negative_candidates(answer: str, *, k_neg: int = 12) -> list[str]:
    answer = str(answer).strip().lower()
    if answer in {"yes", "no"}:
        return ["no" if answer == "yes" else "yes"]
    try:
        value = int(answer)
    except ValueError:
        return []

    negatives: list[str] = []
    seen = {answer}
    common = ("10", "20", "0", "1", "2", "3", "5", "7", "8", "9", "12", "15", "16", "17", "18", "19", "22", "27", "30", "42")
    for candidate in common:
        if candidate not in seen:
            negatives.append(candidate)
            seen.add(candidate)
        if len(negatives) >= max(0, int(k_neg)):
            return negatives
    for offset in (-10, -5, -2, -1, 1, 2, 5, 10):
        candidate = str(value + offset)
        if candidate not in seen:
            negatives.append(candidate)
            seen.add(candidate)
        if len(negatives) >= max(0, int(k_neg)):
            break
    return negatives


def normalize_structural_row(row: dict[str, Any]) -> dict[str, str]:
    seq_a = str(row["seq_a"]).strip()
    seq_b = str(row["seq_b"]).strip()
    target = str(row["target"]).strip()
    if not seq_a or not seq_b or not target:
        raise ValueError("structural rows need non-empty seq_a, seq_b, and target")
    return {"seq_a": seq_a, "seq_b": seq_b, "target": target}


def normalize_repulsion_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"]).strip()
    correct_key = str(row["correct_key"]).strip()
    target = str(row.get("target", correct_key.rsplit(":", 1)[-1])).strip()
    wrong_keys = [str(item).strip() for item in row.get("wrong_keys", []) if str(item).strip()]
    if not prompt or not correct_key or not target:
        raise ValueError("repulsion rows need non-empty prompt, correct_key, and target")
    return {
        "prompt": prompt,
        "correct_key": correct_key,
        "target": target,
        "wrong_keys": [item for item in wrong_keys if item != correct_key],
    }


def repulsion_rows_from_any_row(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Expand native repulsion rows or structural triples into target-localization rows."""

    if {"prompt", "correct_key"} <= set(row):
        yield normalize_repulsion_row(row)
        return
    if {"seq_a", "seq_b", "target"} <= set(row):
        normalized = normalize_structural_row(row)
        correct_key = structural_prototype_key(
            normalized["seq_a"],
            normalized["seq_b"],
            normalized["target"],
        )
        wrong_keys = repulsion_wrong_keys(correct_key, normalized["target"])
        for prompt in (normalized["seq_a"], normalized["seq_b"]):
            yield {
                "prompt": prompt,
                "correct_key": correct_key,
                "target": normalized["target"],
                "wrong_keys": wrong_keys,
            }
        return
    raise ValueError("repulsion rows need prompt/correct_key or seq_a/seq_b/target")


def repulsion_wrong_keys(correct_key: str, target: str, *, k_neg: int = 12) -> list[str]:
    operation = str(correct_key).split(":", 1)[0] if ":" in str(correct_key) else "other"
    target_text = str(target).strip().lower()
    if target_text in {"yes", "no"}:
        return [f"{operation}:{'no' if target_text == 'yes' else 'yes'}"]
    try:
        value = int(target_text)
    except ValueError:
        return []

    wrongs: list[str] = []
    seen = {target_text}
    for offset in (-2, -1, 1, 2, -5, 5, -10, 10):
        candidate = str(value + offset)
        if candidate not in seen:
            wrongs.append(f"{operation}:{candidate}")
            seen.add(candidate)
        if len(wrongs) >= max(0, int(k_neg)):
            return wrongs
    for candidate in ("0", "1", "2", "3", "5", "7", "8", "9", "10", "16", "18", "20", "30", "42", "100"):
        if candidate not in seen:
            wrongs.append(f"{operation}:{candidate}")
            seen.add(candidate)
        if len(wrongs) >= max(0, int(k_neg)):
            break
    return wrongs


def parse_computation_prompt(prompt: str) -> dict[str, Any] | None:
    operation = infer_operation_type(prompt)
    numbers = re.findall(r"-?\d+", str(prompt))
    if operation in {"add", "sub", "mul", "div", "compare"} and len(numbers) >= 2:
        operator_token = {
            "add": "plus",
            "sub": "minus",
            "mul": "times",
            "div": "divide",
            "compare": "compare",
        }[operation]
        return {
            "operation": operation,
            "operands": numbers[:2],
            "operator_token": operator_token,
        }
    return None


def computational_teacher_basin(
    model: PhaseModel,
    prompt: str,
    correct_key: str,
    *,
    steps_per_chunk: int = 20,
    result_weight: float = 2.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    parsed = parse_computation_prompt(prompt)
    if parsed is None:
        return None, {"reason": "unparsed_prompt"}
    vectors: list[np.ndarray] = []
    component_errors: list[float] = []
    component_steps = max(2, min(max(1, int(steps_per_chunk)), 8))
    for component in [*parsed["operands"], parsed["operator_token"]]:
        basin, prediction_error = model.encode_basin(
            str(component),
            steps_per_chunk=component_steps,
            reset=True,
        )
        vectors.append(np.asarray(basin.center, dtype=np.float32))
        component_errors.append(float(prediction_error))
    if not vectors:
        return None, {"reason": "no_teacher_components", **parsed}
    teacher = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    if correct_key in model.structural_prototypes:
        result = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
        weight = max(0.0, float(result_weight))
        teacher = ((teacher + (weight * result)) / (1.0 + weight)).astype(np.float32)
    return teacher.astype(np.float32), {
        **parsed,
        "component_prediction_error": sum(component_errors) / len(component_errors),
        "result_weight": float(result_weight),
    }


def basin_toroidal_distance(basin_a: Any, basin_b: Any, *, width: int, height: int) -> float:
    dx = abs(int(basin_a.x) - int(basin_b.x))
    dy = abs(int(basin_a.y) - int(basin_b.y))
    dx = min(dx, max(0, int(width) - dx))
    dy = min(dy, max(0, int(height) - dy))
    return float(np.sqrt((dx * dx) + (dy * dy)))


ABLATION_MODES = {"full", "no-interference", "random-walk", "static-topology", "uniform-init"}


@dataclass(frozen=True)
class AblationSnapshot:
    wave_speed: float
    nonlinear_gain: float
    potential_scale: float
    memory_decay: float
    memory_gain: float
    natural_frequency_noise: float
    omega: np.ndarray
    landscape: np.ndarray
    predictor_trace: np.ndarray


def apply_ablation_mode(model: PhaseModel, mode: str) -> AblationSnapshot:
    mode = str(mode)
    if mode not in ABLATION_MODES:
        raise ValueError(f"unsupported ablation mode: {mode}")
    field = model.field
    snapshot = AblationSnapshot(
        wave_speed=field.config.wave_speed,
        nonlinear_gain=field.config.nonlinear_gain,
        potential_scale=field.config.potential_scale,
        memory_decay=field.config.memory_decay,
        memory_gain=field.config.memory_gain,
        natural_frequency_noise=field.config.natural_frequency_noise,
        omega=field.omega.copy(),
        landscape=field.landscape.copy(),
        predictor_trace=field.predictor_trace.copy(),
    )
    if mode == "full":
        return snapshot
    if mode == "no-interference":
        field.config = field.config.__class__(
            **{**field.config.__dict__, "wave_speed": 1e-9, "nonlinear_gain": 0.0}
        )
    elif mode == "random-walk":
        field.config = field.config.__class__(
            **{
                **field.config.__dict__,
                "wave_speed": 1e-9,
                "nonlinear_gain": 0.0,
                "potential_scale": 0.0,
                "memory_gain": 0.0,
            }
        )
        field.omega.fill(0.0)
        field.landscape.fill(0.0)
    elif mode == "static-topology":
        field.config = field.config.__class__(
            **{**field.config.__dict__, "memory_decay": 0.0, "memory_gain": 0.0}
        )
    elif mode == "uniform-init":
        field.config = field.config.__class__(
            **{**field.config.__dict__, "natural_frequency_noise": 0.0}
        )
        field.omega.fill(0.0)
        field.landscape.fill(0.0)
    return snapshot


def restore_ablation_snapshot(model: PhaseModel, snapshot: AblationSnapshot) -> None:
    field = model.field
    field.config = field.config.__class__(
        **{
            **field.config.__dict__,
            "wave_speed": snapshot.wave_speed,
            "nonlinear_gain": snapshot.nonlinear_gain,
            "potential_scale": snapshot.potential_scale,
            "memory_decay": snapshot.memory_decay,
            "memory_gain": snapshot.memory_gain,
            "natural_frequency_noise": snapshot.natural_frequency_noise,
        }
    )
    field.omega = snapshot.omega.copy()
    field.landscape = snapshot.landscape.copy()
    field.predictor_trace = snapshot.predictor_trace.copy()


def basin_entropy(records_path: str | Path, *, bins: int = 16) -> dict[str, float | int]:
    bins = max(2, int(bins))
    path = Path(records_path)
    if not path.exists():
        return {"entropy": 0.0, "normalized_entropy": 0.0, "unique_basins": 0, "records": 0}
    counts: dict[tuple[int, int], int] = {}
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            basin = record.get("basin")
            if not isinstance(basin, dict):
                continue
            x = int(basin.get("x", 0))
            y = int(basin.get("y", 0))
            key = (x % bins, y % bins)
            counts[key] = counts.get(key, 0) + 1
            total += 1
    if total == 0:
        return {"entropy": 0.0, "normalized_entropy": 0.0, "unique_basins": 0, "records": 0}
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / float(total)
    entropy = float(-(probabilities * np.log2(np.maximum(probabilities, 1e-12))).sum())
    return {
        "entropy": entropy,
        "normalized_entropy": float(entropy / np.log2(bins * bins)),
        "unique_basins": len(counts),
        "records": total,
    }


def basin_target_mutual_info(records_path: str | Path, *, bins: int = 16) -> dict[str, float | int]:
    """Estimate how much basin bucket identity says about the target token."""

    bins = max(2, int(bins))
    path = Path(records_path)
    if not path.exists():
        return {
            "mutual_info": 0.0,
            "normalized_mutual_info": 0.0,
            "basin_entropy": 0.0,
            "target_entropy": 0.0,
            "unique_basins": 0,
            "unique_targets": 0,
            "records": 0,
        }

    joint_counts: dict[tuple[tuple[int, int], str], int] = {}
    basin_counts: dict[tuple[int, int], int] = {}
    target_counts: dict[str, int] = {}
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            basin = record.get("basin")
            target = record.get("target_token")
            if not isinstance(basin, dict) or target is None:
                continue
            bucket = (int(basin.get("x", 0)) % bins, int(basin.get("y", 0)) % bins)
            target_key = str(target)
            joint_counts[(bucket, target_key)] = joint_counts.get((bucket, target_key), 0) + 1
            basin_counts[bucket] = basin_counts.get(bucket, 0) + 1
            target_counts[target_key] = target_counts.get(target_key, 0) + 1
            total += 1

    if total == 0:
        return {
            "mutual_info": 0.0,
            "normalized_mutual_info": 0.0,
            "basin_entropy": 0.0,
            "target_entropy": 0.0,
            "unique_basins": 0,
            "unique_targets": 0,
            "records": 0,
        }

    total_f = float(total)
    mutual_info = 0.0
    for (bucket, target), count in joint_counts.items():
        p_joint = count / total_f
        p_basin = basin_counts[bucket] / total_f
        p_target = target_counts[target] / total_f
        mutual_info += p_joint * np.log2(p_joint / max(p_basin * p_target, 1e-12))

    basin_probs = np.asarray(list(basin_counts.values()), dtype=np.float64) / total_f
    target_probs = np.asarray(list(target_counts.values()), dtype=np.float64) / total_f
    basin_h = float(-(basin_probs * np.log2(np.maximum(basin_probs, 1e-12))).sum())
    target_h = float(-(target_probs * np.log2(np.maximum(target_probs, 1e-12))).sum())
    normalizer = max(1e-12, min(basin_h, target_h))
    return {
        "mutual_info": float(mutual_info),
        "normalized_mutual_info": float(mutual_info / normalizer),
        "basin_entropy": basin_h,
        "target_entropy": target_h,
        "unique_basins": len(basin_counts),
        "unique_targets": len(target_counts),
        "records": total,
    }


def stream_train(
    model: PhaseModel,
    text_iter: Iterable[str],
    *,
    steps_per_chunk: int = 20,
    batch_size: int = 1,
    context_tokens: int = 8,
    windows_per_chunk: int = 4,
    window_stride: int = 1,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/phase-model",
    max_chunks: int | None = None,
    reset_between_chunks: bool = True,
    train_decoder: bool = True,
    train_topology: bool = True,
    freeze_omega: bool = False,
    train_mode: str = "next-token",
    ablation_mode: str = "full",
    restore_after_train: bool = False,
    consolidate_interval: int = 0,
    consolidate_cycles: int = 8,
) -> dict[str, Any]:
    """Stream text through the field and carve topology without answer labels."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "train_records.jsonl"
    summary: dict[str, Any] = {
        "chunks_seen": 0,
        "observations": 0,
        "mean_prediction_error": 0.0,
        "mean_decoder_loss": None,
        "prediction_error_first_window": None,
        "prediction_error_recent_window": None,
        "decoder_perplexity": None,
        "latest_checkpoint": None,
        "ablation_mode": ablation_mode,
    }
    prediction_errors: list[float] = []
    decoder_losses: list[float] = []
    pending_features: list[list[float]] = []
    pending_targets: list[int] = []

    def flush_decoder_batch() -> float | None:
        if not train_decoder or not pending_features:
            return None
        loss = model.train_decoder_batch(pending_features, pending_targets, mode=train_mode)
        decoder_losses.append(loss)
        pending_features.clear()
        pending_targets.clear()
        return loss

    snapshot = apply_ablation_mode(model, ablation_mode)
    try:
        with records_path.open("a", encoding="utf-8") as records:
            for index, chunk in enumerate(text_iter, start=1):
                if max_chunks is not None and index > max_chunks:
                    break
                for window_index, window_tokens in enumerate(
                    iter_training_windows(
                        model,
                        chunk,
                        context_tokens=context_tokens,
                        max_windows=windows_per_chunk,
                        stride=window_stride,
                    ),
                    start=1,
                ):
                    if ablation_mode == "random-walk":
                        model.field.inject_noise(scale=0.012)
                    freeze_this_observation = freeze_omega or ablation_mode in {"random-walk", "static-topology"}
                    train_topology_this_observation = train_topology and ablation_mode not in {"random-walk", "static-topology"}
                    if ablation_mode == "uniform-init":
                        model.field.omega.fill(0.0)
                    observation = model.observe_text(
                        window_tokens,
                        steps_per_chunk=steps_per_chunk,
                        train_decoder=False,
                        train_topology=train_topology_this_observation,
                        freeze_omega=freeze_this_observation,
                        reset=reset_between_chunks,
                    )
                    prediction_errors.append(observation.mean_prediction_error)
                    if observation.decoder_loss is not None:
                        decoder_losses.append(observation.decoder_loss)
                    if train_decoder and batch_size > 1 and observation.target_id is not None:
                        pending_features.append([float(item) for item in observation.basin["center"]])
                        pending_targets.append(int(observation.target_id))
                        if len(pending_features) >= batch_size:
                            observation_loss = flush_decoder_batch()
                            if observation_loss is not None:
                                records.write(json.dumps({
                                    "chunk_index": index,
                                    "window_index": window_index,
                                    "batch_size": batch_size,
                                    "decoder_loss": observation_loss,
                                    "decoder_perplexity": perplexity_from_loss(observation_loss),
                                    "record_type": "decoder_batch",
                                    "train_mode": train_mode,
                                    "ablation_mode": ablation_mode,
                                }, sort_keys=True) + "\n")
                    elif train_decoder and observation.target_id is not None:
                        observation_loss = model.train_decoder_batch(
                            [[float(item) for item in observation.basin["center"]]],
                            [int(observation.target_id)],
                            mode=train_mode,
                        )
                        decoder_losses.append(observation_loss)
                        observation = observation.__class__(
                            prompt_tokens=observation.prompt_tokens,
                            target_token=observation.target_token,
                            target_id=observation.target_id,
                            steps_used=observation.steps_used,
                            mean_prediction_error=observation.mean_prediction_error,
                            final_prediction_error=observation.final_prediction_error,
                            decoder_loss=observation_loss,
                            basin=observation.basin,
                            metrics=observation.metrics,
                        )
                    record = observation.to_dict()
                    record["chunk_index"] = index
                    record["window_index"] = window_index
                    record["decoder_perplexity"] = perplexity_from_loss(observation.decoder_loss)
                    record["train_mode"] = train_mode
                    record["ablation_mode"] = ablation_mode
                    records.write(json.dumps(record, sort_keys=True) + "\n")
                    summary["observations"] += 1
                if consolidate_interval > 0 and index % consolidate_interval == 0 and ablation_mode != "static-topology":
                    model.field.consolidate(cycles=consolidate_cycles)
                summary["chunks_seen"] = index
                if prediction_errors:
                    summary["mean_prediction_error"] = sum(prediction_errors) / len(prediction_errors)
                    first_window = prediction_errors[: min(len(prediction_errors), 100)]
                    recent_window = prediction_errors[-min(len(prediction_errors), 100) :]
                    summary["prediction_error_first_window"] = sum(first_window) / len(first_window)
                    summary["prediction_error_recent_window"] = sum(recent_window) / len(recent_window)
                if decoder_losses:
                    summary["mean_decoder_loss"] = sum(decoder_losses) / len(decoder_losses)
                    summary["decoder_perplexity"] = perplexity_from_loss(summary["mean_decoder_loss"])

                if save_interval > 0 and index % save_interval == 0:
                    checkpoint = output / f"checkpoint-{index:08d}"
                    model.save(checkpoint)
                    summary["latest_checkpoint"] = str(checkpoint)
                    summary["basin_entropy"] = basin_entropy(records_path)
                    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        tail_loss = flush_decoder_batch()
        if tail_loss is not None:
            summary["mean_decoder_loss"] = sum(decoder_losses) / len(decoder_losses)
            summary["decoder_perplexity"] = perplexity_from_loss(summary["mean_decoder_loss"])
        summary["basin_entropy"] = basin_entropy(records_path)
        final_dir = output / "final"
        model.save(final_dir)
        summary["latest_checkpoint"] = str(final_dir)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        if restore_after_train:
            restore_ablation_snapshot(model, snapshot)


def stream_ranking_train(
    model: PhaseModel,
    ranking_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    batch_size: int = 64,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/reranker-model",
    max_rows: int | None = None,
    train_decoder: bool = False,
    decoder_mode: str = "next-token",
) -> dict[str, Any]:
    """Train the basin/candidate verifier head from prompt/candidate labels."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "ranking_records.jsonl"
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "batches": 0,
        "mean_loss": None,
        "mean_accuracy": None,
        "mean_decoder_loss": None,
        "decoder_synced": bool(train_decoder),
        "latest_checkpoint": None,
    }
    pending_features: list[list[float]] = []
    pending_candidate_features: list[list[float]] = []
    pending_candidates: list[int] = []
    pending_labels: list[int] = []
    pending_decoder_features: list[list[float]] = []
    pending_decoder_targets: list[int] = []
    losses: list[float] = []
    accuracies: list[float] = []
    decoder_losses: list[float] = []

    def flush_batch() -> dict[str, float] | None:
        if not pending_features:
            return None
        metrics = model.train_reranker_batch(
            pending_features,
            pending_candidates,
            pending_labels,
            candidate_basin_centers=pending_candidate_features,
        )
        losses.append(metrics["loss"])
        accuracies.append(metrics["accuracy"])
        pending_features.clear()
        pending_candidate_features.clear()
        pending_candidates.clear()
        pending_labels.clear()
        summary["batches"] += 1
        if train_decoder and pending_decoder_features:
            decoder_loss = model.train_decoder_batch(
                pending_decoder_features,
                pending_decoder_targets,
                mode=decoder_mode,
            )
            decoder_losses.append(decoder_loss)
            pending_decoder_features.clear()
            pending_decoder_targets.clear()
            summary["mean_decoder_loss"] = sum(decoder_losses) / len(decoder_losses)
        summary["mean_loss"] = sum(losses) / len(losses)
        summary["mean_accuracy"] = sum(accuracies) / len(accuracies)
        return metrics

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(ranking_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_ranking_row(row)
            basin, prediction_error = model.encode_basin(
                normalized["prompt"],
                steps_per_chunk=steps_per_chunk,
                reset=True,
            )
            candidate_basin, candidate_prediction_error = model.encode_basin(
                f"{normalized['prompt']} {normalized['candidate']}",
                steps_per_chunk=steps_per_chunk,
                reset=True,
            )
            candidate_id = model.vocab.add(normalized["candidate"])
            pending_features.append([float(item) for item in basin.center])
            pending_candidate_features.append([float(item) for item in candidate_basin.center])
            pending_candidates.append(int(candidate_id))
            pending_labels.append(int(normalized["label"]))
            if train_decoder and int(normalized["label"]) == 1:
                pending_decoder_features.append([float(item) for item in basin.center])
                pending_decoder_targets.append(int(candidate_id))
            summary["rows_seen"] = index
            record: dict[str, Any] = {
                "row_index": index,
                "prompt": normalized["prompt"],
                "candidate": normalized["candidate"],
                "candidate_id": int(candidate_id),
                "label": int(normalized["label"]),
                "prediction_error": float(prediction_error),
                "candidate_prediction_error": float(candidate_prediction_error),
                "basin": basin.to_dict(),
                "candidate_basin": candidate_basin.to_dict(),
            }
            if len(pending_features) >= max(1, int(batch_size)):
                metrics = flush_batch()
                record["batch_metrics"] = metrics
            records.write(json.dumps(record, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    flush_batch()
    if train_decoder and pending_decoder_features:
        decoder_loss = model.train_decoder_batch(
            pending_decoder_features,
            pending_decoder_targets,
            mode=decoder_mode,
        )
        decoder_losses.append(decoder_loss)
        summary["mean_decoder_loss"] = sum(decoder_losses) / len(decoder_losses)
        pending_decoder_features.clear()
        pending_decoder_targets.clear()
    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_structural_train(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    batch_size: int = 64,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/geometry-model",
    max_rows: int | None = None,
    structural_weight: float = 0.5,
    topology_gain: float = 0.025,
    anchor: bool = False,
    freeze_decoder: bool = False,
    prototype_alpha: float = 0.10,
) -> dict[str, Any]:
    """Train equivalent expressions to share basin geometry plus decoder target."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / ("structural_anchor_records.jsonl" if anchor else "structural_records.jsonl")
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "batches": 0,
        "mode": "structural-anchor" if anchor else "structural",
        "decoder_frozen": bool(freeze_decoder),
        "mean_alignment": None,
        "alignment_first_window": None,
        "alignment_recent_window": None,
        "mean_feature_l2": None,
        "mean_basin_distance": None,
        "mean_feature_mse": None,
        "same_basin_rate": None,
        "same_basin_radius": 0.2,
        "mean_coordinate_distance": None,
        "same_coordinate_rate": None,
        "mean_loss": None,
        "mean_ce_loss": None,
        "mean_structural_loss": None,
        "prototype_count": 0,
        "latest_checkpoint": None,
    }
    pending_a: list[list[float]] = []
    pending_b: list[list[float]] = []
    pending_targets: list[int] = []
    alignments: list[float] = []
    feature_l2s: list[float] = []
    feature_mses: list[float] = []
    coordinate_distances: list[float] = []
    losses: list[float] = []
    ce_losses: list[float] = []
    structural_losses: list[float] = []

    def flush_batch() -> dict[str, float] | None:
        if freeze_decoder or not pending_a:
            pending_a.clear()
            pending_b.clear()
            pending_targets.clear()
            return None
        metrics = model.train_structural_batch(
            pending_a,
            pending_b,
            pending_targets,
            structural_weight=structural_weight,
        )
        losses.append(metrics["loss"])
        ce_losses.append(metrics["ce_loss"])
        structural_losses.append(metrics["structural_loss"])
        pending_a.clear()
        pending_b.clear()
        pending_targets.clear()
        summary["batches"] += 1
        summary["mean_loss"] = sum(losses) / len(losses)
        summary["mean_ce_loss"] = sum(ce_losses) / len(ce_losses)
        summary["mean_structural_loss"] = sum(structural_losses) / len(structural_losses)
        return metrics

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(structural_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_structural_row(row)
            basin_a, prediction_error_a = model.encode_basin(
                normalized["seq_a"],
                steps_per_chunk=steps_per_chunk,
                reset=True,
            )
            basin_b, prediction_error_b = model.encode_basin(
                normalized["seq_b"],
                steps_per_chunk=steps_per_chunk,
                reset=True,
            )
            if anchor:
                prototype_key = structural_prototype_key(
                    normalized["seq_a"],
                    normalized["seq_b"],
                    normalized["target"],
                )
                anchor_metrics = model.reinforce_structural_anchor(
                    basin_a,
                    basin_b,
                    prototype_key=prototype_key,
                    gain=topology_gain,
                    prototype_alpha=prototype_alpha,
                )
                alignment = float(anchor_metrics["alignment"])
            else:
                prototype_key = ""
                anchor_metrics = None
                alignment = model.reinforce_equivalence(basin_a, basin_b, gain=topology_gain)
            feature_l2 = structural_feature_l2(basin_a.center, basin_b.center)
            feature_mse = structural_feature_mse(basin_a.center, basin_b.center)
            coordinate_distance = basin_toroidal_distance(
                basin_a,
                basin_b,
                width=model.field.config.width,
                height=model.field.config.height,
            )
            target_id = model.vocab.add(normalized["target"])
            pending_a.append([float(item) for item in basin_a.center])
            pending_b.append([float(item) for item in basin_b.center])
            pending_targets.append(int(target_id))
            alignments.append(float(alignment))
            feature_l2s.append(float(feature_l2))
            feature_mses.append(float(feature_mse))
            coordinate_distances.append(float(coordinate_distance))
            summary["rows_seen"] = index
            summary["mean_alignment"] = sum(alignments) / len(alignments)
            summary["mean_feature_l2"] = sum(feature_l2s) / len(feature_l2s)
            summary["mean_basin_distance"] = summary["mean_feature_l2"]
            summary["mean_feature_mse"] = sum(feature_mses) / len(feature_mses)
            summary["same_basin_rate"] = sum(1 for item in feature_l2s if item < 0.2) / len(feature_l2s)
            summary["mean_coordinate_distance"] = sum(coordinate_distances) / len(coordinate_distances)
            summary["same_coordinate_rate"] = (
                sum(1 for item in coordinate_distances if item == 0.0) / len(coordinate_distances)
            )
            summary["prototype_count"] = len(model.structural_prototypes)
            first_window = alignments[: min(len(alignments), 100)]
            recent_window = alignments[-min(len(alignments), 100) :]
            summary["alignment_first_window"] = sum(first_window) / len(first_window)
            summary["alignment_recent_window"] = sum(recent_window) / len(recent_window)
            record: dict[str, Any] = {
                "row_index": index,
                "seq_a": normalized["seq_a"],
                "seq_b": normalized["seq_b"],
                "target": normalized["target"],
                "target_id": int(target_id),
                "alignment": float(alignment),
                "feature_l2": float(feature_l2),
                "basin_distance": float(feature_l2),
                "feature_mse": float(feature_mse),
                "coordinate_distance": float(coordinate_distance),
                "same_basin": bool(feature_l2 < 0.2),
                "same_coordinate": bool(coordinate_distance == 0.0),
                "prediction_error_a": float(prediction_error_a),
                "prediction_error_b": float(prediction_error_b),
                "basin_a": basin_a.to_dict(),
                "basin_b": basin_b.to_dict(),
            }
            if anchor_metrics is not None:
                record["prototype_key"] = prototype_key
                record["anchor_metrics"] = anchor_metrics
            if len(pending_a) >= max(1, int(batch_size)):
                metrics = flush_batch()
                if metrics is not None:
                    record["batch_metrics"] = metrics
            records.write(json.dumps(record, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    flush_batch()
    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_prototype_decoder_train(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    batch_size: int = 64,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/injected-model",
    max_rows: int | None = None,
    train_mode: str = "next-token",
    use_prototype_readout: bool = True,
    readout_temperature: float = 0.1,
    readout_direct_scale: float = 8.0,
) -> dict[str, Any]:
    """Train decoder directly on saved structural prototype vectors."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if use_prototype_readout:
        model.use_prototype_readout(
            temperature=readout_temperature,
            direct_scale=readout_direct_scale,
        )
    records_path = output / "prototype_decoder_records.jsonl"
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "batches": 0,
        "mean_decoder_loss": None,
        "decoder_perplexity": None,
        "prototype_count": len(model.structural_prototypes),
        "decoder_type": getattr(model.decoder, "decoder_type", "mlp") if model.decoder is not None else "none",
        "latest_checkpoint": None,
    }
    pending_features: list[list[float]] = []
    pending_targets: list[int] = []
    losses: list[float] = []

    def flush_batch() -> float | None:
        if not pending_features:
            return None
        loss = model.train_decoder_batch(
            pending_features,
            pending_targets,
            mode=train_mode,
        )
        losses.append(loss)
        pending_features.clear()
        pending_targets.clear()
        summary["batches"] += 1
        summary["mean_decoder_loss"] = sum(losses) / len(losses)
        summary["decoder_perplexity"] = perplexity_from_loss(summary["mean_decoder_loss"])
        return loss

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(structural_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_structural_row(row)
            prototype_key = structural_prototype_key(
                normalized["seq_a"],
                normalized["seq_b"],
                normalized["target"],
            )
            summary["rows_seen"] = index
            target_id = model.vocab.add(normalized["target"])
            prototype = model.structural_prototypes.get(prototype_key)
            record: dict[str, Any] = {
                "row_index": index,
                "prototype_key": prototype_key,
                "target": normalized["target"],
                "target_id": int(target_id),
                "used": prototype is not None,
            }
            if prototype is None:
                summary["rows_skipped"] += 1
                records.write(json.dumps(record, sort_keys=True) + "\n")
                continue
            pending_features.append([float(item) for item in np.asarray(prototype, dtype=np.float32)])
            pending_targets.append(int(target_id))
            summary["rows_used"] += 1
            if len(pending_features) >= max(1, int(batch_size)):
                loss = flush_batch()
                if loss is not None:
                    record["decoder_loss"] = loss
                    record["decoder_perplexity"] = perplexity_from_loss(loss)
            records.write(json.dumps(record, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    flush_batch()
    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_structural_repulsion_train(
    model: PhaseModel,
    repulsion_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/targeted-model",
    max_rows: int | None = None,
    attract_gain: float = 0.10,
    repulsion_strength: float = 0.40,
    topology_gain: float = 0.025,
) -> dict[str, Any]:
    """Localize active expression basins toward correct result prototypes."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "structural_repulsion_records.jsonl"
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "mode": "structural-repulsion",
        "nearest_prototype_target_accuracy_before": None,
        "nearest_prototype_target_accuracy_after": None,
        "mean_correct_distance_before": None,
        "mean_correct_distance_after": None,
        "mean_wrong_distance_min_before": None,
        "prototype_count": len(model.structural_prototypes),
        "latest_checkpoint": None,
    }
    matches_before = 0
    matches_after = 0
    used = 0
    correct_before_distances: list[float] = []
    correct_after_distances: list[float] = []
    wrong_min_distances: list[float] = []

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(repulsion_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_repulsion_row(row)
            summary["rows_seen"] = index
            if normalized["correct_key"] not in model.structural_prototypes:
                summary["rows_skipped"] += 1
                records.write(json.dumps({
                    "row_index": index,
                    **normalized,
                    "used": False,
                    "reason": "missing_correct_prototype",
                }, sort_keys=True) + "\n")
                continue

            basin, prediction_error = model.encode_basin(
                normalized["prompt"],
                steps_per_chunk=steps_per_chunk,
                reset=True,
            )
            metrics = model.reinforce_result_target(
                basin,
                correct_key=normalized["correct_key"],
                wrong_keys=normalized["wrong_keys"],
                attract_gain=attract_gain,
                repulsion_strength=repulsion_strength,
                topology_gain=topology_gain,
            )
            if not metrics.get("used"):
                summary["rows_skipped"] += 1
                continue
            used += 1
            summary["rows_used"] = used
            matches_before += int(bool(metrics["target_match_before"]))
            matches_after += int(bool(metrics["target_match_after"]))
            correct_before_distances.append(float(metrics["correct_distance_before"]))
            correct_after_distances.append(float(metrics["correct_distance_after"]))
            if metrics.get("wrong_distance_min_before") is not None:
                wrong_min_distances.append(float(metrics["wrong_distance_min_before"]))
            summary["nearest_prototype_target_accuracy_before"] = matches_before / used
            summary["nearest_prototype_target_accuracy_after"] = matches_after / used
            summary["mean_correct_distance_before"] = sum(correct_before_distances) / len(correct_before_distances)
            summary["mean_correct_distance_after"] = sum(correct_after_distances) / len(correct_after_distances)
            summary["mean_wrong_distance_min_before"] = (
                sum(wrong_min_distances) / len(wrong_min_distances) if wrong_min_distances else None
            )
            summary["prototype_count"] = len(model.structural_prototypes)

            records.write(json.dumps({
                "row_index": index,
                **normalized,
                "used": True,
                "prediction_error": float(prediction_error),
                "basin": basin.to_dict(),
                "metrics": metrics,
            }, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_computational_distillation_train(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/distilled-model",
    max_rows: int | None = None,
    distill_gain: float = 0.10,
    repulsion_strength: float = 0.25,
    topology_gain: float = 0.025,
    result_weight: float = 2.0,
) -> dict[str, Any]:
    """Distill raw expression basins toward clean operand/operator teacher vectors."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "computational_distillation_records.jsonl"
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "mode": "computational-distillation",
        "nearest_prototype_target_accuracy_before": None,
        "nearest_prototype_target_accuracy_after": None,
        "mean_active_teacher_distance": None,
        "mean_correct_distance_before": None,
        "mean_correct_distance_after": None,
        "mean_teacher_correct_distance_before": None,
        "mean_teacher_correct_distance_after": None,
        "prototype_count": len(model.structural_prototypes),
        "latest_checkpoint": None,
    }
    matches_before = 0
    matches_after = 0
    used = 0
    active_teacher_distances: list[float] = []
    correct_before_distances: list[float] = []
    correct_after_distances: list[float] = []
    teacher_correct_before_distances: list[float] = []
    teacher_correct_after_distances: list[float] = []

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(structural_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_structural_row(row)
            summary["rows_seen"] = index
            correct_key = structural_prototype_key(
                normalized["seq_a"],
                normalized["seq_b"],
                normalized["target"],
            )
            if correct_key not in model.structural_prototypes:
                summary["rows_skipped"] += 1
                records.write(json.dumps({
                    "row_index": index,
                    **normalized,
                    "correct_key": correct_key,
                    "used": False,
                    "reason": "missing_correct_prototype",
                }, sort_keys=True) + "\n")
                continue
            wrong_keys = repulsion_wrong_keys(correct_key, normalized["target"])
            row_records: list[dict[str, Any]] = []
            for prompt in (normalized["seq_a"], normalized["seq_b"]):
                teacher, teacher_info = computational_teacher_basin(
                    model,
                    prompt,
                    correct_key,
                    steps_per_chunk=steps_per_chunk,
                    result_weight=result_weight,
                )
                if teacher is None:
                    summary["rows_skipped"] += 1
                    row_records.append({
                        "prompt": prompt,
                        "used": False,
                        **teacher_info,
                    })
                    continue
                basin, prediction_error = model.encode_basin(
                    prompt,
                    steps_per_chunk=steps_per_chunk,
                    reset=True,
                )
                metrics = model.distill_computational_teacher(
                    basin,
                    teacher,
                    correct_key=correct_key,
                    wrong_keys=wrong_keys,
                    distill_gain=distill_gain,
                    repulsion_strength=repulsion_strength,
                    topology_gain=topology_gain,
                )
                if not metrics.get("used"):
                    summary["rows_skipped"] += 1
                    row_records.append({
                        "prompt": prompt,
                        "used": False,
                        "metrics": metrics,
                    })
                    continue
                used += 1
                matches_before += int(bool(metrics["target_match_before"]))
                matches_after += int(bool(metrics["target_match_after"]))
                active_teacher_distances.append(float(metrics["active_teacher_distance"]))
                correct_before_distances.append(float(metrics["correct_distance_before"]))
                correct_after_distances.append(float(metrics["correct_distance_after"]))
                teacher_correct_before_distances.append(float(metrics["teacher_correct_distance_before"]))
                teacher_correct_after_distances.append(float(metrics["teacher_correct_distance_after"]))
                row_records.append({
                    "prompt": prompt,
                    "used": True,
                    "prediction_error": float(prediction_error),
                    "teacher_info": teacher_info,
                    "metrics": metrics,
                    "basin": basin.to_dict(),
                })

            summary["rows_used"] = used
            if used:
                summary["nearest_prototype_target_accuracy_before"] = matches_before / used
                summary["nearest_prototype_target_accuracy_after"] = matches_after / used
                summary["mean_active_teacher_distance"] = sum(active_teacher_distances) / len(active_teacher_distances)
                summary["mean_correct_distance_before"] = sum(correct_before_distances) / len(correct_before_distances)
                summary["mean_correct_distance_after"] = sum(correct_after_distances) / len(correct_after_distances)
                summary["mean_teacher_correct_distance_before"] = (
                    sum(teacher_correct_before_distances) / len(teacher_correct_before_distances)
                )
                summary["mean_teacher_correct_distance_after"] = (
                    sum(teacher_correct_after_distances) / len(teacher_correct_after_distances)
                )
            summary["prototype_count"] = len(model.structural_prototypes)
            records.write(json.dumps({
                "row_index": index,
                **normalized,
                "correct_key": correct_key,
                "prompt_records": row_records,
            }, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_computational_distillation_evaluate(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_rows: int | None = None,
    result_weight: float = 2.0,
) -> dict[str, Any]:
    rows = 0
    used = 0
    correct = 0
    teacher_closer_than_active = 0
    active_target_distances: list[float] = []
    teacher_target_distances: list[float] = []
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(structural_iter, start=1):
        if max_rows is not None and index > max_rows:
            break
        normalized = normalize_structural_row(row)
        correct_key = structural_prototype_key(
            normalized["seq_a"],
            normalized["seq_b"],
            normalized["target"],
        )
        if correct_key not in model.structural_prototypes:
            continue
        for prompt in (normalized["seq_a"], normalized["seq_b"]):
            rows += 1
            teacher, teacher_info = computational_teacher_basin(
                model,
                prompt,
                correct_key,
                steps_per_chunk=steps_per_chunk,
                result_weight=result_weight,
            )
            if teacher is None:
                continue
            basin, prediction_error = model.encode_basin(
                prompt,
                steps_per_chunk=steps_per_chunk,
                reset=True,
            )
            operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
            nearest = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
            target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
            active_distance = structural_feature_l2(basin.center, target)
            teacher_distance = structural_feature_l2(teacher, target)
            used += 1
            match = bool(nearest and nearest[0]["key"] == correct_key)
            correct += int(match)
            teacher_closer_than_active += int(teacher_distance < active_distance)
            active_target_distances.append(active_distance)
            teacher_target_distances.append(teacher_distance)
            if len(examples) < 5:
                examples.append(
                    {
                        "prompt": prompt,
                        "correct_key": correct_key,
                        "nearest_key": nearest[0]["key"] if nearest else None,
                        "active_target_distance": active_distance,
                        "teacher_target_distance": teacher_distance,
                        "teacher_info": teacher_info,
                        "prediction_error": float(prediction_error),
                        "correct": match,
                    }
                )
    return {
        "rows": rows,
        "used_rows": used,
        "nearest_prototype_target_accuracy": correct / used if used else None,
        "teacher_closer_than_active_rate": teacher_closer_than_active / used if used else None,
        "mean_active_target_distance": sum(active_target_distances) / len(active_target_distances) if active_target_distances else None,
        "mean_teacher_target_distance": sum(teacher_target_distances) / len(teacher_target_distances) if teacher_target_distances else None,
        "examples": examples,
    }


def stream_guided_evolution_train(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/guided-model",
    max_rows: int | None = None,
    coupling: float = 0.30,
    success_mse: float = 0.10,
    distill_gain: float = 0.10,
    repulsion_strength: float = 0.25,
    topology_gain: float = 0.025,
    result_weight: float = 2.0,
) -> dict[str, Any]:
    """Steer phase evolution toward a teacher while the basin is forming."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "guided_evolution_records.jsonl"
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "mode": "guided-evolution",
        "guided_nearest_prototype_target_accuracy_before_update": None,
        "guided_nearest_prototype_target_accuracy_after_update": None,
        "mean_guided_active_teacher_l2": None,
        "mean_guided_active_teacher_mse": None,
        "mean_guided_active_target_distance": None,
        "updates_applied": 0,
        "prototype_count": len(model.structural_prototypes),
        "latest_checkpoint": None,
    }
    used = 0
    before_correct = 0
    after_correct = 0
    updates_applied = 0
    teacher_l2s: list[float] = []
    teacher_mses: list[float] = []
    target_distances: list[float] = []

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(structural_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_structural_row(row)
            summary["rows_seen"] = index
            correct_key = structural_prototype_key(
                normalized["seq_a"],
                normalized["seq_b"],
                normalized["target"],
            )
            if correct_key not in model.structural_prototypes:
                summary["rows_skipped"] += 1
                continue
            wrong_keys = repulsion_wrong_keys(correct_key, normalized["target"])
            row_records: list[dict[str, Any]] = []
            for prompt in (normalized["seq_a"], normalized["seq_b"]):
                teacher, teacher_info = computational_teacher_basin(
                    model,
                    prompt,
                    correct_key,
                    steps_per_chunk=steps_per_chunk,
                    result_weight=result_weight,
                )
                if teacher is None:
                    summary["rows_skipped"] += 1
                    row_records.append({"prompt": prompt, "used": False, **teacher_info})
                    continue
                basin, prediction_error = model.encode_basin(
                    prompt,
                    steps_per_chunk=steps_per_chunk,
                    reset=True,
                    attractor=teacher,
                    attractor_key=correct_key,
                    coupling=coupling,
                )
                operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
                nearest_before = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
                active = np.asarray(basin.center, dtype=np.float32)
                teacher_arr = np.asarray(teacher, dtype=np.float32)
                target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
                teacher_l2 = structural_feature_l2(active, teacher_arr)
                teacher_mse = structural_feature_mse(active, teacher_arr)
                target_distance = structural_feature_l2(active, target)
                metrics: dict[str, Any] | None = None
                if teacher_mse < float(success_mse):
                    metrics = model.distill_computational_teacher(
                        basin,
                        teacher_arr,
                        correct_key=correct_key,
                        wrong_keys=wrong_keys,
                        distill_gain=distill_gain,
                        repulsion_strength=repulsion_strength,
                        topology_gain=topology_gain,
                    )
                    updates_applied += int(bool(metrics.get("used")))
                nearest_after = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
                used += 1
                before_correct += int(bool(nearest_before and nearest_before[0]["key"] == correct_key))
                after_correct += int(bool(nearest_after and nearest_after[0]["key"] == correct_key))
                teacher_l2s.append(float(teacher_l2))
                teacher_mses.append(float(teacher_mse))
                target_distances.append(float(target_distance))
                row_records.append(
                    {
                        "prompt": prompt,
                        "used": True,
                        "prediction_error": float(prediction_error),
                        "teacher_info": teacher_info,
                        "nearest_before": nearest_before[0] if nearest_before else None,
                        "nearest_after": nearest_after[0] if nearest_after else None,
                        "guided_active_teacher_l2": float(teacher_l2),
                        "guided_active_teacher_mse": float(teacher_mse),
                        "guided_active_target_distance": float(target_distance),
                        "updated": metrics is not None,
                        "metrics": metrics,
                        "basin": basin.to_dict(),
                    }
                )

            summary["rows_used"] = used
            if used:
                summary["guided_nearest_prototype_target_accuracy_before_update"] = before_correct / used
                summary["guided_nearest_prototype_target_accuracy_after_update"] = after_correct / used
                summary["mean_guided_active_teacher_l2"] = sum(teacher_l2s) / len(teacher_l2s)
                summary["mean_guided_active_teacher_mse"] = sum(teacher_mses) / len(teacher_mses)
                summary["mean_guided_active_target_distance"] = sum(target_distances) / len(target_distances)
            summary["updates_applied"] = updates_applied
            summary["prototype_count"] = len(model.structural_prototypes)
            records.write(json.dumps({
                "row_index": index,
                **normalized,
                "correct_key": correct_key,
                "prompt_records": row_records,
            }, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_guided_evolution_evaluate(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_rows: int | None = None,
    coupling: float = 0.30,
    result_weight: float = 2.0,
) -> dict[str, Any]:
    rows = 0
    used = 0
    correct = 0
    teacher_l2s: list[float] = []
    teacher_mses: list[float] = []
    target_distances: list[float] = []
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(structural_iter, start=1):
        if max_rows is not None and index > max_rows:
            break
        normalized = normalize_structural_row(row)
        correct_key = structural_prototype_key(
            normalized["seq_a"],
            normalized["seq_b"],
            normalized["target"],
        )
        if correct_key not in model.structural_prototypes:
            continue
        for prompt in (normalized["seq_a"], normalized["seq_b"]):
            rows += 1
            teacher, teacher_info = computational_teacher_basin(
                model,
                prompt,
                correct_key,
                steps_per_chunk=steps_per_chunk,
                result_weight=result_weight,
            )
            if teacher is None:
                continue
            basin, prediction_error = model.encode_basin(
                prompt,
                steps_per_chunk=steps_per_chunk,
                reset=True,
                attractor=teacher,
                attractor_key=correct_key,
                coupling=coupling,
            )
            operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
            nearest = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
            active = np.asarray(basin.center, dtype=np.float32)
            teacher_arr = np.asarray(teacher, dtype=np.float32)
            target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
            teacher_l2 = structural_feature_l2(active, teacher_arr)
            teacher_mse = structural_feature_mse(active, teacher_arr)
            target_distance = structural_feature_l2(active, target)
            match = bool(nearest and nearest[0]["key"] == correct_key)
            used += 1
            correct += int(match)
            teacher_l2s.append(float(teacher_l2))
            teacher_mses.append(float(teacher_mse))
            target_distances.append(float(target_distance))
            if len(examples) < 5:
                examples.append(
                    {
                        "prompt": prompt,
                        "correct_key": correct_key,
                        "nearest_key": nearest[0]["key"] if nearest else None,
                        "nearest_distance": nearest[0]["distance"] if nearest else None,
                        "guided_active_teacher_l2": float(teacher_l2),
                        "guided_active_teacher_mse": float(teacher_mse),
                        "guided_active_target_distance": float(target_distance),
                        "teacher_info": teacher_info,
                        "prediction_error": float(prediction_error),
                        "correct": match,
                    }
                )
    return {
        "rows": rows,
        "used_rows": used,
        "guided_nearest_prototype_target_accuracy": correct / used if used else None,
        "mean_guided_active_teacher_l2": sum(teacher_l2s) / len(teacher_l2s) if teacher_l2s else None,
        "mean_guided_active_teacher_mse": sum(teacher_mses) / len(teacher_mses) if teacher_mses else None,
        "mean_guided_active_target_distance": sum(target_distances) / len(target_distances) if target_distances else None,
        "examples": examples,
    }


def stream_phase_geometry_train(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/geometry-model",
    max_rows: int | None = None,
    coupling: float = 0.30,
    success_mse: float = 0.10,
    geometry_strength: float = 0.05,
    patch_size: int | None = None,
    distill_gain: float = 0.10,
    repulsion_strength: float = 0.25,
    topology_gain: float = 0.025,
    result_weight: float = 2.0,
) -> dict[str, Any]:
    """Steer phase evolution with teacher-shaped phase patches."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "phase_geometry_records.jsonl"
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "mode": "phase-geometry",
        "phase_geometry_nearest_prototype_target_accuracy_before_update": None,
        "phase_geometry_nearest_prototype_target_accuracy_after_update": None,
        "mean_phase_geometry_active_teacher_l2": None,
        "mean_phase_geometry_active_teacher_mse": None,
        "mean_phase_geometry_active_target_distance": None,
        "mean_patch_amp": None,
        "updates_applied": 0,
        "prototype_count": len(model.structural_prototypes),
        "latest_checkpoint": None,
    }
    used = 0
    before_correct = 0
    after_correct = 0
    updates_applied = 0
    teacher_l2s: list[float] = []
    teacher_mses: list[float] = []
    target_distances: list[float] = []
    patch_amps: list[float] = []

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(structural_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_structural_row(row)
            summary["rows_seen"] = index
            correct_key = structural_prototype_key(
                normalized["seq_a"],
                normalized["seq_b"],
                normalized["target"],
            )
            if correct_key not in model.structural_prototypes:
                summary["rows_skipped"] += 1
                continue
            wrong_keys = repulsion_wrong_keys(correct_key, normalized["target"])
            row_records: list[dict[str, Any]] = []
            for prompt in (normalized["seq_a"], normalized["seq_b"]):
                teacher, teacher_info = computational_teacher_basin(
                    model,
                    prompt,
                    correct_key,
                    steps_per_chunk=steps_per_chunk,
                    result_weight=result_weight,
                )
                if teacher is None:
                    summary["rows_skipped"] += 1
                    row_records.append({"prompt": prompt, "used": False, **teacher_info})
                    continue
                basin, prediction_error, patch_info = model.encode_basin_with_phase_geometry(
                    prompt,
                    teacher,
                    correct_key=correct_key,
                    steps_per_chunk=steps_per_chunk,
                    coupling=coupling,
                    patch_size=patch_size,
                    reset=True,
                )
                operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
                nearest_before = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
                active = np.asarray(basin.center, dtype=np.float32)
                teacher_arr = np.asarray(teacher, dtype=np.float32)
                target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
                teacher_l2 = structural_feature_l2(active, teacher_arr)
                teacher_mse = structural_feature_mse(active, teacher_arr)
                target_distance = structural_feature_l2(active, target)
                geometry_metrics: dict[str, Any] | None = None
                distill_metrics: dict[str, Any] | None = None
                if teacher_mse < float(success_mse):
                    geometry_metrics = model.inject_teacher_phase_geometry(
                        basin,
                        teacher_arr,
                        correct_key=correct_key,
                        strength=geometry_strength,
                        patch_size=patch_size,
                    )
                    distill_metrics = model.distill_computational_teacher(
                        basin,
                        teacher_arr,
                        correct_key=correct_key,
                        wrong_keys=wrong_keys,
                        distill_gain=distill_gain,
                        repulsion_strength=repulsion_strength,
                        topology_gain=topology_gain,
                    )
                    updates_applied += int(bool(distill_metrics.get("used")))
                nearest_after = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
                used += 1
                before_correct += int(bool(nearest_before and nearest_before[0]["key"] == correct_key))
                after_correct += int(bool(nearest_after and nearest_after[0]["key"] == correct_key))
                teacher_l2s.append(float(teacher_l2))
                teacher_mses.append(float(teacher_mse))
                target_distances.append(float(target_distance))
                patch_amps.append(float(patch_info.get("patch_amp_mean", 0.0)))
                row_records.append(
                    {
                        "prompt": prompt,
                        "used": True,
                        "prediction_error": float(prediction_error),
                        "teacher_info": teacher_info,
                        "patch_info": patch_info,
                        "nearest_before": nearest_before[0] if nearest_before else None,
                        "nearest_after": nearest_after[0] if nearest_after else None,
                        "phase_geometry_active_teacher_l2": float(teacher_l2),
                        "phase_geometry_active_teacher_mse": float(teacher_mse),
                        "phase_geometry_active_target_distance": float(target_distance),
                        "updated": geometry_metrics is not None,
                        "geometry_metrics": geometry_metrics,
                        "distill_metrics": distill_metrics,
                        "basin": basin.to_dict(),
                    }
                )

            summary["rows_used"] = used
            if used:
                summary["phase_geometry_nearest_prototype_target_accuracy_before_update"] = before_correct / used
                summary["phase_geometry_nearest_prototype_target_accuracy_after_update"] = after_correct / used
                summary["mean_phase_geometry_active_teacher_l2"] = sum(teacher_l2s) / len(teacher_l2s)
                summary["mean_phase_geometry_active_teacher_mse"] = sum(teacher_mses) / len(teacher_mses)
                summary["mean_phase_geometry_active_target_distance"] = sum(target_distances) / len(target_distances)
                summary["mean_patch_amp"] = sum(patch_amps) / len(patch_amps)
            summary["updates_applied"] = updates_applied
            summary["prototype_count"] = len(model.structural_prototypes)
            records.write(json.dumps({
                "row_index": index,
                **normalized,
                "correct_key": correct_key,
                "prompt_records": row_records,
            }, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_phase_geometry_evaluate(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_rows: int | None = None,
    coupling: float = 0.30,
    patch_size: int | None = None,
    result_weight: float = 2.0,
) -> dict[str, Any]:
    rows = 0
    used = 0
    correct = 0
    teacher_l2s: list[float] = []
    teacher_mses: list[float] = []
    target_distances: list[float] = []
    patch_amps: list[float] = []
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(structural_iter, start=1):
        if max_rows is not None and index > max_rows:
            break
        normalized = normalize_structural_row(row)
        correct_key = structural_prototype_key(
            normalized["seq_a"],
            normalized["seq_b"],
            normalized["target"],
        )
        if correct_key not in model.structural_prototypes:
            continue
        for prompt in (normalized["seq_a"], normalized["seq_b"]):
            rows += 1
            teacher, teacher_info = computational_teacher_basin(
                model,
                prompt,
                correct_key,
                steps_per_chunk=steps_per_chunk,
                result_weight=result_weight,
            )
            if teacher is None:
                continue
            basin, prediction_error, patch_info = model.encode_basin_with_phase_geometry(
                prompt,
                teacher,
                correct_key=correct_key,
                steps_per_chunk=steps_per_chunk,
                coupling=coupling,
                patch_size=patch_size,
                reset=True,
            )
            operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
            nearest = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
            active = np.asarray(basin.center, dtype=np.float32)
            teacher_arr = np.asarray(teacher, dtype=np.float32)
            target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
            teacher_l2 = structural_feature_l2(active, teacher_arr)
            teacher_mse = structural_feature_mse(active, teacher_arr)
            target_distance = structural_feature_l2(active, target)
            match = bool(nearest and nearest[0]["key"] == correct_key)
            used += 1
            correct += int(match)
            teacher_l2s.append(float(teacher_l2))
            teacher_mses.append(float(teacher_mse))
            target_distances.append(float(target_distance))
            patch_amps.append(float(patch_info.get("patch_amp_mean", 0.0)))
            if len(examples) < 5:
                examples.append(
                    {
                        "prompt": prompt,
                        "correct_key": correct_key,
                        "nearest_key": nearest[0]["key"] if nearest else None,
                        "nearest_distance": nearest[0]["distance"] if nearest else None,
                        "phase_geometry_active_teacher_l2": float(teacher_l2),
                        "phase_geometry_active_teacher_mse": float(teacher_mse),
                        "phase_geometry_active_target_distance": float(target_distance),
                        "teacher_info": teacher_info,
                        "patch_info": patch_info,
                        "prediction_error": float(prediction_error),
                        "correct": match,
                    }
                )
    return {
        "rows": rows,
        "used_rows": used,
        "phase_geometry_nearest_prototype_target_accuracy": correct / used if used else None,
        "mean_phase_geometry_active_teacher_l2": sum(teacher_l2s) / len(teacher_l2s) if teacher_l2s else None,
        "mean_phase_geometry_active_teacher_mse": sum(teacher_mses) / len(teacher_mses) if teacher_mses else None,
        "mean_phase_geometry_active_target_distance": sum(target_distances) / len(target_distances) if target_distances else None,
        "mean_patch_amp": sum(patch_amps) / len(patch_amps) if patch_amps else None,
        "examples": examples,
    }


def stream_delta_geometry_train(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/delta-model",
    max_rows: int | None = None,
    coupling: float = 0.30,
    success_distance: float = 1.0,
    geometry_strength: float = 0.05,
    patch_size: int | None = None,
    topology_gain: float = 0.025,
    result_weight: float = 2.0,
    freeze_targets: bool = False,
) -> dict[str, Any]:
    """Steer phase evolution with the result-specific target minus teacher delta."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "delta_geometry_records.jsonl"
    mode_name = "delta-geometry-frozen" if freeze_targets else "delta-geometry"
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "mode": mode_name,
        "targets_frozen": bool(freeze_targets),
        "delta_nearest_prototype_target_accuracy_before_update": None,
        "delta_nearest_prototype_target_accuracy_after_update": None,
        "mean_delta_active_teacher_l2": None,
        "mean_delta_active_teacher_mse": None,
        "mean_delta_active_target_distance": None,
        "mean_target_distance_after_update": None,
        "mean_delta_l2": None,
        "mean_patch_amp": None,
        "updates_applied": 0,
        "prototype_count": len(model.structural_prototypes),
        "latest_checkpoint": None,
    }
    used = 0
    before_correct = 0
    after_correct = 0
    updates_applied = 0
    teacher_l2s: list[float] = []
    teacher_mses: list[float] = []
    target_distances: list[float] = []
    target_distances_after: list[float] = []
    delta_l2s: list[float] = []
    patch_amps: list[float] = []

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(structural_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_structural_row(row)
            summary["rows_seen"] = index
            correct_key = structural_prototype_key(
                normalized["seq_a"],
                normalized["seq_b"],
                normalized["target"],
            )
            if correct_key not in model.structural_prototypes:
                summary["rows_skipped"] += 1
                continue
            row_records: list[dict[str, Any]] = []
            for prompt in (normalized["seq_a"], normalized["seq_b"]):
                teacher, teacher_info = computational_teacher_basin(
                    model,
                    prompt,
                    correct_key,
                    steps_per_chunk=steps_per_chunk,
                    result_weight=result_weight,
                )
                if teacher is None:
                    summary["rows_skipped"] += 1
                    row_records.append({"prompt": prompt, "used": False, **teacher_info})
                    continue
                target_before = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
                basin, prediction_error, patch_info = model.encode_basin_with_delta_geometry(
                    prompt,
                    teacher,
                    target_before,
                    correct_key=correct_key,
                    steps_per_chunk=steps_per_chunk,
                    coupling=coupling,
                    patch_size=patch_size,
                    reset=True,
                )
                operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
                nearest_before = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
                active = np.asarray(basin.center, dtype=np.float32)
                teacher_arr = np.asarray(teacher, dtype=np.float32)
                teacher_l2 = structural_feature_l2(active, teacher_arr)
                teacher_mse = structural_feature_mse(active, teacher_arr)
                target_distance = structural_feature_l2(active, target_before)
                geometry_metrics: dict[str, Any] | None = None
                landscape_metrics: dict[str, Any] | None = None
                if target_distance < float(success_distance):
                    geometry_metrics = model.inject_delta_phase_geometry(
                        basin,
                        teacher_arr,
                        target_before,
                        correct_key=correct_key,
                        strength=geometry_strength,
                        patch_size=patch_size,
                    )
                    landscape_metrics = model.update_landscape_toward(
                        basin,
                        target_before,
                        strength=topology_gain,
                        correct_key=correct_key,
                        update_prototype=not freeze_targets,
                    )
                    updates_applied += int(bool(landscape_metrics.get("used")))
                target_after = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
                nearest_after = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
                target_distance_after = structural_feature_l2(active, target_after)
                used += 1
                before_correct += int(bool(nearest_before and nearest_before[0]["key"] == correct_key))
                after_correct += int(bool(nearest_after and nearest_after[0]["key"] == correct_key))
                teacher_l2s.append(float(teacher_l2))
                teacher_mses.append(float(teacher_mse))
                target_distances.append(float(target_distance))
                target_distances_after.append(float(target_distance_after))
                delta_l2s.append(float(patch_info.get("delta_l2", 0.0)))
                patch_amps.append(float(patch_info.get("patch_amp_mean", 0.0)))
                row_records.append(
                    {
                        "prompt": prompt,
                        "used": True,
                        "prediction_error": float(prediction_error),
                        "teacher_info": teacher_info,
                        "patch_info": patch_info,
                        "nearest_before": nearest_before[0] if nearest_before else None,
                        "nearest_after": nearest_after[0] if nearest_after else None,
                        "delta_active_teacher_l2": float(teacher_l2),
                        "delta_active_teacher_mse": float(teacher_mse),
                        "delta_active_target_distance": float(target_distance),
                        "delta_active_target_distance_after_update": float(target_distance_after),
                        "updated": geometry_metrics is not None,
                        "geometry_metrics": geometry_metrics,
                        "landscape_metrics": landscape_metrics,
                        "basin": basin.to_dict(),
                    }
                )

            summary["rows_used"] = used
            if used:
                summary["delta_nearest_prototype_target_accuracy_before_update"] = before_correct / used
                summary["delta_nearest_prototype_target_accuracy_after_update"] = after_correct / used
                summary["mean_delta_active_teacher_l2"] = sum(teacher_l2s) / len(teacher_l2s)
                summary["mean_delta_active_teacher_mse"] = sum(teacher_mses) / len(teacher_mses)
                summary["mean_delta_active_target_distance"] = sum(target_distances) / len(target_distances)
                summary["mean_target_distance_after_update"] = (
                    sum(target_distances_after) / len(target_distances_after)
                )
                summary["mean_delta_l2"] = sum(delta_l2s) / len(delta_l2s)
                summary["mean_patch_amp"] = sum(patch_amps) / len(patch_amps)
            summary["updates_applied"] = updates_applied
            summary["prototype_count"] = len(model.structural_prototypes)
            records.write(json.dumps({
                "row_index": index,
                **normalized,
                "correct_key": correct_key,
                "prompt_records": row_records,
            }, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_delta_geometry_evaluate(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_rows: int | None = None,
    coupling: float = 0.30,
    patch_size: int | None = None,
    result_weight: float = 2.0,
) -> dict[str, Any]:
    rows = 0
    used = 0
    correct = 0
    teacher_l2s: list[float] = []
    teacher_mses: list[float] = []
    target_distances: list[float] = []
    delta_l2s: list[float] = []
    patch_amps: list[float] = []
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(structural_iter, start=1):
        if max_rows is not None and index > max_rows:
            break
        normalized = normalize_structural_row(row)
        correct_key = structural_prototype_key(
            normalized["seq_a"],
            normalized["seq_b"],
            normalized["target"],
        )
        if correct_key not in model.structural_prototypes:
            continue
        for prompt in (normalized["seq_a"], normalized["seq_b"]):
            rows += 1
            teacher, teacher_info = computational_teacher_basin(
                model,
                prompt,
                correct_key,
                steps_per_chunk=steps_per_chunk,
                result_weight=result_weight,
            )
            if teacher is None:
                continue
            target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
            basin, prediction_error, patch_info = model.encode_basin_with_delta_geometry(
                prompt,
                teacher,
                target,
                correct_key=correct_key,
                steps_per_chunk=steps_per_chunk,
                coupling=coupling,
                patch_size=patch_size,
                reset=True,
            )
            operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
            nearest = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
            active = np.asarray(basin.center, dtype=np.float32)
            teacher_arr = np.asarray(teacher, dtype=np.float32)
            teacher_l2 = structural_feature_l2(active, teacher_arr)
            teacher_mse = structural_feature_mse(active, teacher_arr)
            target_distance = structural_feature_l2(active, target)
            match = bool(nearest and nearest[0]["key"] == correct_key)
            used += 1
            correct += int(match)
            teacher_l2s.append(float(teacher_l2))
            teacher_mses.append(float(teacher_mse))
            target_distances.append(float(target_distance))
            delta_l2s.append(float(patch_info.get("delta_l2", 0.0)))
            patch_amps.append(float(patch_info.get("patch_amp_mean", 0.0)))
            if len(examples) < 5:
                examples.append(
                    {
                        "prompt": prompt,
                        "correct_key": correct_key,
                        "nearest_key": nearest[0]["key"] if nearest else None,
                        "nearest_distance": nearest[0]["distance"] if nearest else None,
                        "delta_active_teacher_l2": float(teacher_l2),
                        "delta_active_teacher_mse": float(teacher_mse),
                        "delta_active_target_distance": float(target_distance),
                        "teacher_info": teacher_info,
                        "patch_info": patch_info,
                        "prediction_error": float(prediction_error),
                        "correct": match,
                    }
                )
    return {
        "rows": rows,
        "used_rows": used,
        "delta_nearest_prototype_target_accuracy": correct / used if used else None,
        "mean_delta_active_teacher_l2": sum(teacher_l2s) / len(teacher_l2s) if teacher_l2s else None,
        "mean_delta_active_teacher_mse": sum(teacher_mses) / len(teacher_mses) if teacher_mses else None,
        "mean_delta_active_target_distance": sum(target_distances) / len(target_distances) if target_distances else None,
        "mean_delta_l2": sum(delta_l2s) / len(delta_l2s) if delta_l2s else None,
        "mean_patch_amp": sum(patch_amps) / len(patch_amps) if patch_amps else None,
        "examples": examples,
    }


def stream_residual_tunnel_train(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/tunnel-model",
    max_rows: int | None = None,
    tunnel_strength: float = 0.05,
) -> dict[str, Any]:
    """Carve residual landscape tunnels toward frozen result prototypes."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "residual_tunnel_records.jsonl"
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "mode": "residual-tunnel",
        "targets_frozen": True,
        "tunnel_nearest_target_accuracy_before": None,
        "tunnel_nearest_target_accuracy_after": None,
        "mean_tunnel_active_target_distance_before": None,
        "mean_tunnel_active_target_distance_after": None,
        "mean_residual_l2": None,
        "mean_path_steps": None,
        "updates_applied": 0,
        "prototype_count": len(model.structural_prototypes),
        "latest_checkpoint": None,
    }
    used = 0
    before_correct = 0
    after_correct = 0
    updates_applied = 0
    before_distances: list[float] = []
    after_distances: list[float] = []
    residual_l2s: list[float] = []
    path_steps: list[float] = []

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(structural_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_structural_row(row)
            summary["rows_seen"] = index
            correct_key = structural_prototype_key(
                normalized["seq_a"],
                normalized["seq_b"],
                normalized["target"],
            )
            if correct_key not in model.structural_prototypes:
                summary["rows_skipped"] += 1
                continue
            row_records: list[dict[str, Any]] = []
            for prompt in (normalized["seq_a"], normalized["seq_b"]):
                target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
                target_snapshot = target.copy()
                basin_before, prediction_error_before = model.encode_basin(
                    prompt,
                    steps_per_chunk=steps_per_chunk,
                    reset=True,
                )
                operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
                nearest_before = model.nearest_structural_prototype(
                    basin_before.center,
                    operation=operation,
                    k=1,
                )
                distance_before = structural_feature_l2(basin_before.center, target)
                tunnel_metrics = model.carve_residual_tunnel(
                    basin_before,
                    target,
                    correct_key=correct_key,
                    strength=tunnel_strength,
                )
                updates_applied += int(bool(tunnel_metrics.get("used")))

                basin_after, prediction_error_after = model.encode_basin(
                    prompt,
                    steps_per_chunk=steps_per_chunk,
                    reset=True,
                )
                nearest_after = model.nearest_structural_prototype(
                    basin_after.center,
                    operation=operation,
                    k=1,
                )
                distance_after = structural_feature_l2(basin_after.center, target)
                used += 1
                before_correct += int(bool(nearest_before and nearest_before[0]["key"] == correct_key))
                after_correct += int(bool(nearest_after and nearest_after[0]["key"] == correct_key))
                before_distances.append(float(distance_before))
                after_distances.append(float(distance_after))
                residual_l2s.append(float(tunnel_metrics.get("residual_l2", distance_before)))
                path_steps.append(float(tunnel_metrics.get("path_steps", 0)))
                if not np.allclose(model.structural_prototypes[correct_key], target_snapshot):
                    raise RuntimeError(f"frozen target prototype changed for {correct_key}")
                row_records.append(
                    {
                        "prompt": prompt,
                        "used": True,
                        "prediction_error_before": float(prediction_error_before),
                        "prediction_error_after": float(prediction_error_after),
                        "nearest_before": nearest_before[0] if nearest_before else None,
                        "nearest_after": nearest_after[0] if nearest_after else None,
                        "active_target_distance_before": float(distance_before),
                        "active_target_distance_after": float(distance_after),
                        "tunnel_metrics": tunnel_metrics,
                        "basin_before": basin_before.to_dict(),
                        "basin_after": basin_after.to_dict(),
                    }
                )

            summary["rows_used"] = used
            if used:
                summary["tunnel_nearest_target_accuracy_before"] = before_correct / used
                summary["tunnel_nearest_target_accuracy_after"] = after_correct / used
                summary["mean_tunnel_active_target_distance_before"] = (
                    sum(before_distances) / len(before_distances)
                )
                summary["mean_tunnel_active_target_distance_after"] = (
                    sum(after_distances) / len(after_distances)
                )
                summary["mean_residual_l2"] = sum(residual_l2s) / len(residual_l2s)
                summary["mean_path_steps"] = sum(path_steps) / len(path_steps)
            summary["updates_applied"] = updates_applied
            summary["prototype_count"] = len(model.structural_prototypes)
            records.write(json.dumps({
                "row_index": index,
                **normalized,
                "correct_key": correct_key,
                "prompt_records": row_records,
            }, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_residual_tunnel_evaluate(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_rows: int | None = None,
) -> dict[str, Any]:
    rows = 0
    used = 0
    correct = 0
    distances: list[float] = []
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(structural_iter, start=1):
        if max_rows is not None and index > max_rows:
            break
        normalized = normalize_structural_row(row)
        correct_key = structural_prototype_key(
            normalized["seq_a"],
            normalized["seq_b"],
            normalized["target"],
        )
        if correct_key not in model.structural_prototypes:
            continue
        for prompt in (normalized["seq_a"], normalized["seq_b"]):
            rows += 1
            target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
            basin, prediction_error = model.encode_basin(
                prompt,
                steps_per_chunk=steps_per_chunk,
                reset=True,
            )
            operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
            nearest = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
            distance = structural_feature_l2(basin.center, target)
            match = bool(nearest and nearest[0]["key"] == correct_key)
            used += 1
            correct += int(match)
            distances.append(float(distance))
            if len(examples) < 5:
                examples.append(
                    {
                        "prompt": prompt,
                        "correct_key": correct_key,
                        "nearest_key": nearest[0]["key"] if nearest else None,
                        "nearest_distance": nearest[0]["distance"] if nearest else None,
                        "active_target_distance": float(distance),
                        "prediction_error": float(prediction_error),
                        "correct": match,
                    }
                )
    return {
        "rows": rows,
        "used_rows": used,
        "tunnel_nearest_prototype_target_accuracy": correct / used if used else None,
        "mean_tunnel_active_target_distance": sum(distances) / len(distances) if distances else None,
        "examples": examples,
    }


def stream_push_pull_train(
    model: PhaseModel,
    repulsion_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    save_interval: int = 10_000,
    out_dir: str | Path = "runs/push-pull-model",
    max_rows: int | None = None,
    push_pull_strength: float = 0.05,
    wrong_strength: float = 0.5,
) -> dict[str, Any]:
    """Carve local feature basins by pulling to target and pushing from wrong results."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "push_pull_records.jsonl"
    summary: dict[str, Any] = {
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "mode": "push-pull",
        "targets_frozen": True,
        "push_pull_nearest_target_accuracy_before": None,
        "push_pull_nearest_target_accuracy_after": None,
        "mean_push_pull_active_target_distance_before": None,
        "mean_push_pull_active_target_distance_after": None,
        "mean_pull_l2": None,
        "mean_push_l2": None,
        "mean_update_l2": None,
        "mean_wrong_count": None,
        "updates_applied": 0,
        "prototype_count": len(model.structural_prototypes),
        "latest_checkpoint": None,
    }
    used = 0
    before_correct = 0
    after_correct = 0
    updates_applied = 0
    before_distances: list[float] = []
    after_distances: list[float] = []
    pull_l2s: list[float] = []
    push_l2s: list[float] = []
    update_l2s: list[float] = []
    wrong_counts: list[float] = []

    with records_path.open("w", encoding="utf-8") as records:
        for index, row in enumerate(repulsion_iter, start=1):
            if max_rows is not None and index > max_rows:
                break
            normalized = normalize_repulsion_row(row)
            summary["rows_seen"] = index
            correct_key = normalized["correct_key"]
            if correct_key not in model.structural_prototypes:
                summary["rows_skipped"] += 1
                continue
            target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
            target_snapshot = target.copy()
            wrong_keys = [
                key for key in normalized["wrong_keys"]
                if key != correct_key and key in model.structural_prototypes
            ]
            wrong_targets = [
                np.asarray(model.structural_prototypes[key], dtype=np.float32)
                for key in wrong_keys
            ]
            basin_before, prediction_error_before = model.encode_basin(
                normalized["prompt"],
                steps_per_chunk=steps_per_chunk,
                reset=True,
            )
            operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
            nearest_before = model.nearest_structural_prototype(
                basin_before.center,
                operation=operation,
                k=1,
            )
            distance_before = structural_feature_l2(basin_before.center, target)
            push_pull_metrics = model.carve_push_pull(
                basin_before,
                target,
                wrong_targets,
                correct_key=correct_key,
                wrong_keys=wrong_keys,
                strength=push_pull_strength,
                wrong_strength=wrong_strength,
            )
            updates_applied += int(bool(push_pull_metrics.get("used")))

            basin_after, prediction_error_after = model.encode_basin(
                normalized["prompt"],
                steps_per_chunk=steps_per_chunk,
                reset=True,
            )
            nearest_after = model.nearest_structural_prototype(
                basin_after.center,
                operation=operation,
                k=1,
            )
            distance_after = structural_feature_l2(basin_after.center, target)
            used += 1
            before_correct += int(bool(nearest_before and nearest_before[0]["key"] == correct_key))
            after_correct += int(bool(nearest_after and nearest_after[0]["key"] == correct_key))
            before_distances.append(float(distance_before))
            after_distances.append(float(distance_after))
            pull_l2s.append(float(push_pull_metrics.get("pull_l2", distance_before)))
            push_l2s.append(float(push_pull_metrics.get("push_l2", 0.0)))
            update_l2s.append(float(push_pull_metrics.get("update_l2", 0.0)))
            wrong_counts.append(float(push_pull_metrics.get("wrong_count", len(wrong_targets))))
            if not np.allclose(model.structural_prototypes[correct_key], target_snapshot):
                raise RuntimeError(f"frozen target prototype changed for {correct_key}")

            summary["rows_used"] = used
            if used:
                summary["push_pull_nearest_target_accuracy_before"] = before_correct / used
                summary["push_pull_nearest_target_accuracy_after"] = after_correct / used
                summary["mean_push_pull_active_target_distance_before"] = (
                    sum(before_distances) / len(before_distances)
                )
                summary["mean_push_pull_active_target_distance_after"] = (
                    sum(after_distances) / len(after_distances)
                )
                summary["mean_pull_l2"] = sum(pull_l2s) / len(pull_l2s)
                summary["mean_push_l2"] = sum(push_l2s) / len(push_l2s)
                summary["mean_update_l2"] = sum(update_l2s) / len(update_l2s)
                summary["mean_wrong_count"] = sum(wrong_counts) / len(wrong_counts)
            summary["updates_applied"] = updates_applied
            summary["prototype_count"] = len(model.structural_prototypes)
            records.write(json.dumps({
                "row_index": index,
                **normalized,
                "used": True,
                "prediction_error_before": float(prediction_error_before),
                "prediction_error_after": float(prediction_error_after),
                "nearest_before": nearest_before[0] if nearest_before else None,
                "nearest_after": nearest_after[0] if nearest_after else None,
                "active_target_distance_before": float(distance_before),
                "active_target_distance_after": float(distance_after),
                "push_pull_metrics": push_pull_metrics,
                "basin_before": basin_before.to_dict(),
                "basin_after": basin_after.to_dict(),
            }, sort_keys=True) + "\n")
            if save_interval > 0 and index % save_interval == 0:
                checkpoint = output / f"checkpoint-{index:08d}"
                model.save(checkpoint)
                summary["latest_checkpoint"] = str(checkpoint)
                (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    final_dir = output / "final"
    model.save(final_dir)
    summary["latest_checkpoint"] = str(final_dir)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def stream_push_pull_evaluate(
    model: PhaseModel,
    repulsion_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_rows: int | None = None,
) -> dict[str, Any]:
    rows = 0
    used = 0
    correct = 0
    operation_correct = 0
    nearest_distances: list[float] = []
    target_distances: list[float] = []
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(repulsion_iter, start=1):
        if max_rows is not None and index > max_rows:
            break
        normalized = normalize_repulsion_row(row)
        rows += 1
        correct_key = normalized["correct_key"]
        if correct_key not in model.structural_prototypes:
            continue
        target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
        basin, prediction_error = model.encode_basin(
            normalized["prompt"],
            steps_per_chunk=steps_per_chunk,
            reset=True,
        )
        operation = correct_key.split(":", 1)[0] if ":" in correct_key else None
        nearest_global = model.nearest_structural_prototype(basin.center, k=1)
        nearest_operation = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
        best = nearest_operation[0] if nearest_operation else None
        target_distance = structural_feature_l2(basin.center, target)
        used += 1
        match = bool(best and best["key"] == correct_key)
        correct += int(match)
        operation_correct += int(
            bool(nearest_global and nearest_global[0]["key"].split(":", 1)[0] == operation)
        )
        if best is not None:
            nearest_distances.append(float(best["distance"]))
        target_distances.append(float(target_distance))
        if len(examples) < 5:
            examples.append(
                {
                    "prompt": normalized["prompt"],
                    "correct_key": correct_key,
                    "nearest_key": best["key"] if best else None,
                    "nearest_distance": best["distance"] if best else None,
                    "active_target_distance": float(target_distance),
                    "global_nearest_key": nearest_global[0]["key"] if nearest_global else None,
                    "prediction_error": float(prediction_error),
                    "correct": match,
                }
            )
    return {
        "rows": rows,
        "used_rows": used,
        "push_pull_nearest_prototype_target_accuracy": correct / used if used else None,
        "global_operation_accuracy": operation_correct / used if used else None,
        "mean_nearest_distance": sum(nearest_distances) / len(nearest_distances) if nearest_distances else None,
        "mean_push_pull_active_target_distance": sum(target_distances) / len(target_distances) if target_distances else None,
        "examples": examples,
    }


def stream_repulsion_evaluate(
    model: PhaseModel,
    repulsion_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_rows: int | None = None,
) -> dict[str, Any]:
    rows = 0
    used = 0
    correct = 0
    operation_correct = 0
    distances: list[float] = []
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(repulsion_iter, start=1):
        if max_rows is not None and index > max_rows:
            break
        normalized = normalize_repulsion_row(row)
        rows += 1
        if normalized["correct_key"] not in model.structural_prototypes:
            continue
        basin, prediction_error = model.encode_basin(
            normalized["prompt"],
            steps_per_chunk=steps_per_chunk,
            reset=True,
        )
        operation = normalized["correct_key"].split(":", 1)[0] if ":" in normalized["correct_key"] else None
        nearest_global = model.nearest_structural_prototype(basin.center, k=1)
        nearest_operation = model.nearest_structural_prototype(basin.center, operation=operation, k=1)
        best = nearest_operation[0] if nearest_operation else None
        used += 1
        match = bool(best and best["key"] == normalized["correct_key"])
        correct += int(match)
        operation_correct += int(
            bool(nearest_global and nearest_global[0]["key"].split(":", 1)[0] == operation)
        )
        if best is not None:
            distances.append(float(best["distance"]))
        if len(examples) < 5:
            examples.append(
                {
                    "prompt": normalized["prompt"],
                    "correct_key": normalized["correct_key"],
                    "nearest_key": best["key"] if best else None,
                    "nearest_distance": best["distance"] if best else None,
                    "global_nearest_key": nearest_global[0]["key"] if nearest_global else None,
                    "prediction_error": float(prediction_error),
                    "correct": match,
                }
            )
    return {
        "rows": rows,
        "used_rows": used,
        "nearest_prototype_target_accuracy": correct / used if used else None,
        "global_operation_accuracy": operation_correct / used if used else None,
        "mean_nearest_distance": sum(distances) / len(distances) if distances else None,
        "examples": examples,
    }


def stream_structural_evaluate(
    model: PhaseModel,
    structural_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_rows: int | None = None,
) -> dict[str, Any]:
    rows = 0
    alignments: list[float] = []
    feature_l2s: list[float] = []
    feature_mses: list[float] = []
    coordinate_distances: list[float] = []
    decoder_correct = 0
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(structural_iter, start=1):
        if max_rows is not None and index > max_rows:
            break
        normalized = normalize_structural_row(row)
        basin_a, _prediction_error_a = model.encode_basin(
            normalized["seq_a"],
            steps_per_chunk=steps_per_chunk,
            reset=True,
        )
        basin_b, _prediction_error_b = model.encode_basin(
            normalized["seq_b"],
            steps_per_chunk=steps_per_chunk,
            reset=True,
        )
        alignment = cosine_similarity(basin_a.center, basin_b.center)
        feature_l2 = structural_feature_l2(basin_a.center, basin_b.center)
        feature_mse = structural_feature_mse(basin_a.center, basin_b.center)
        coordinate_distance = basin_toroidal_distance(
            basin_a,
            basin_b,
            width=model.field.config.width,
            height=model.field.config.height,
        )
        target_id = model.vocab.encode_tokens([normalized["target"]], add_new=False)[0]
        top_candidate = model.top_decoder_candidates(
            normalized["seq_a"],
            k=1,
            steps_per_chunk=steps_per_chunk,
        )
        predicted = top_candidate[0]["candidate"] if top_candidate else ""
        rows += 1
        alignments.append(float(alignment))
        feature_l2s.append(float(feature_l2))
        feature_mses.append(float(feature_mse))
        coordinate_distances.append(float(coordinate_distance))
        decoder_correct += int(int(top_candidate[0]["token_id"]) == int(target_id)) if top_candidate else 0
        if len(examples) < 5:
            examples.append(
                {
                    "seq_a": normalized["seq_a"],
                    "seq_b": normalized["seq_b"],
                    "target": normalized["target"],
                    "predicted": predicted,
                    "alignment": float(alignment),
                    "feature_l2": float(feature_l2),
                    "basin_distance": float(feature_l2),
                    "feature_mse": float(feature_mse),
                    "coordinate_distance": float(coordinate_distance),
                }
            )
    return {
        "rows": rows,
        "mean_alignment": sum(alignments) / len(alignments) if alignments else None,
        "mean_feature_l2": sum(feature_l2s) / len(feature_l2s) if feature_l2s else None,
        "mean_basin_distance": sum(feature_l2s) / len(feature_l2s) if feature_l2s else None,
        "mean_feature_mse": sum(feature_mses) / len(feature_mses) if feature_mses else None,
        "same_basin_radius": 0.2,
        "same_basin_rate": sum(1 for item in feature_l2s if item < 0.2) / len(feature_l2s) if feature_l2s else None,
        "mean_coordinate_distance": sum(coordinate_distances) / len(coordinate_distances) if coordinate_distances else None,
        "same_coordinate_rate": (
            sum(1 for item in coordinate_distances if item == 0.0) / len(coordinate_distances)
            if coordinate_distances
            else None
        ),
        "decoder_top1_accuracy": decoder_correct / rows if rows else None,
        "examples": examples,
    }


def stream_ranking_evaluate(
    model: PhaseModel,
    ranking_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_rows: int | None = None,
) -> dict[str, Any]:
    rows = 0
    correct = 0
    positives = 0
    positive_correct = 0
    scores: list[float] = []
    for index, row in enumerate(ranking_iter, start=1):
        if max_rows is not None and index > max_rows:
            break
        normalized = normalize_ranking_row(row)
        scored = model.score_candidates(
            normalized["prompt"],
            [normalized["candidate"]],
            steps_per_chunk=steps_per_chunk,
        )
        score = float(scored[0]["score"]) if scored else 0.0
        prediction = 1 if score >= 0.0 else 0
        label = int(normalized["label"])
        rows += 1
        correct += int(prediction == label)
        positives += int(label == 1)
        positive_correct += int(label == 1 and prediction == 1)
        scores.append(score)
    return {
        "rows": rows,
        "accuracy": correct / rows if rows else None,
        "positive_accuracy": positive_correct / positives if positives else None,
        "mean_score": sum(scores) / len(scores) if scores else None,
    }


def stream_ranking_group_evaluate(
    model: PhaseModel,
    ranking_iter: Iterable[dict[str, Any]],
    *,
    steps_per_chunk: int = 20,
    max_groups: int | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in ranking_iter:
        normalized = normalize_ranking_row(row)
        groups.setdefault(normalized["prompt"], []).append(normalized)

    scored_groups = 0
    top1_correct = 0
    margins: list[float] = []
    examples: list[dict[str, Any]] = []
    for prompt, rows in groups.items():
        positives = {row["candidate"] for row in rows if int(row["label"]) == 1}
        negatives = {row["candidate"] for row in rows if int(row["label"]) == 0}
        if not positives or not negatives:
            continue
        candidates = sorted(positives | negatives)
        ranked = model.score_candidates(prompt, candidates, steps_per_chunk=steps_per_chunk)
        if not ranked:
            continue
        best = ranked[0]
        positive_scores = [item["score"] for item in ranked if item["candidate"] in positives]
        negative_scores = [item["score"] for item in ranked if item["candidate"] in negatives]
        if positive_scores and negative_scores:
            margins.append(max(positive_scores) - max(negative_scores))
        correct = best["candidate"] in positives
        scored_groups += 1
        top1_correct += int(correct)
        if len(examples) < 5:
            examples.append(
                {
                    "prompt": prompt,
                    "best": best["candidate"],
                    "positives": sorted(positives),
                    "top3": [
                        {"candidate": item["candidate"], "score": item["score"]}
                        for item in ranked[:3]
                    ],
                    "correct": correct,
                }
            )
        if max_groups is not None and scored_groups >= max_groups:
            break

    return {
        "groups": scored_groups,
        "top1_accuracy": top1_correct / scored_groups if scored_groups else None,
        "mean_positive_margin": sum(margins) / len(margins) if margins else None,
        "examples": examples,
    }


def stream_evaluate(
    model: PhaseModel,
    text_iter: Iterable[str],
    *,
    steps_per_chunk: int = 20,
    max_chunks: int | None = None,
    context_tokens: int = 8,
    windows_per_chunk: int = 4,
    window_stride: int = 1,
) -> dict[str, Any]:
    """Evaluate held-out text using the same next-token windows as training."""

    losses: list[float] = []
    prediction_errors: list[float] = []
    observations = 0
    chunks_seen = 0
    for index, text in enumerate(text_iter, start=1):
        if max_chunks is not None and index > max_chunks:
            break
        chunks_seen = index
        for window_tokens in iter_training_windows(
            model,
            text,
            context_tokens=context_tokens,
            max_windows=windows_per_chunk,
            stride=window_stride,
        ):
            observation = model.observe_text(
                window_tokens,
                steps_per_chunk=steps_per_chunk,
                train_decoder=False,
                train_topology=False,
                freeze_omega=True,
                reset=True,
            )
            observations += 1
            prediction_errors.append(observation.mean_prediction_error)
            if observation.target_id is not None:
                center = np.asarray(observation.basin["center"], dtype=np.float32)
                losses.append(model.decoder_loss(center, observation.target_id))

    mean_loss = sum(losses) / len(losses) if losses else None
    return {
        "chunks": chunks_seen,
        "observations": observations,
        "scored_windows": len(losses),
        "mean_decoder_loss": mean_loss,
        "perplexity": perplexity_from_loss(mean_loss),
        "mean_prediction_error": sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0,
    }


def iter_training_windows(
    model: PhaseModel,
    chunk: str,
    *,
    context_tokens: int = 8,
    max_windows: int = 4,
    stride: int = 1,
) -> Iterator[list[str]]:
    tokens = model.tokenize(chunk)
    if len(tokens) <= 1:
        yield tokens
        return

    stride = max(1, int(stride))
    target_positions = list(range(1, len(tokens), stride))
    max_windows = max(1, int(max_windows))
    if len(target_positions) > max_windows:
        if max_windows == 1:
            target_positions = [target_positions[-1]]
        else:
            last = len(target_positions) - 1
            selected = sorted({round(i * last / (max_windows - 1)) for i in range(max_windows)})
            target_positions = [target_positions[item] for item in selected]

    context_tokens = max(1, int(context_tokens))
    for target_index in target_positions:
        start = max(0, target_index - context_tokens)
        yield tokens[start : target_index + 1]
