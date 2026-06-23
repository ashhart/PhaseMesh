#!/usr/bin/env python3
"""Decoder backprop and push-pull topology carving at full scale.

Strategy:
1. Initialize prototypes by encoding each (operation, result) pair once
2. Train in one of four modes:
   - manifold: operation-anchor -> result-anchor road building, no active-basin writes
   - topology: sparse/keyed feature residuals only, decoder frozen
   - decoder: decoder backprop only, topology frozen
   - combined: old falsified baseline, both updates at once
3. Train at grid_size=128, basin_dim=256 (4x the capacity of the 32x64 model)
4. Focus on add/mul rows with clear numeric answers

Run on GPU box:
    cd ~/phase-mesh && source .venv/bin/activate
    python3 scripts/train_combined.py \
        --data runs/repulsion_data.jsonl \
        --out runs/combined-model \
        --max-rows 8000 \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_mesh.model import PhaseModel
from phase_mesh.trainer import iter_repulsion_jsonl, normalize_repulsion_row


def select_torch_device(requested="auto"):
    """Pick the decoder training device. Field evolution is still backend-dependent."""

    import torch

    requested = str(requested).lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_decoder_device(model, device):
    """Move trainable torch heads to device and rebuild optimizers."""

    if getattr(model, "num_slots", 0) > 0 and getattr(model, "gate_net", None) is None:
        model._init_gate()
    if model.decoder is not None:
        model.decoder.to(device)
    if model.reranker is not None:
        model.reranker.to(device)
    if getattr(model, "gate_net", None) is not None:
        model.gate_net.to(device)
    if getattr(model, "delta_scorer", None) is None:
        model._init_delta_scorer()
    if getattr(model, "delta_scorer", None) is not None:
        model.delta_scorer.to(device)
    model.reset_optimizer(model.learning_rate)
    return model


def set_decoder_trainable(model, trainable: bool) -> None:
    """Toggle gradients for the decoder/reranker heads."""

    for head in (model.decoder, model.reranker):
        if head is None:
            continue
        for parameter in head.parameters():
            parameter.requires_grad = bool(trainable)


def filter_by_op(rows_iter, ops):
    """Yield only rows for specified operations."""
    for row in rows_iter:
        normalized = normalize_repulsion_row(row)
        op = normalized["correct_key"].split(":", 1)[0]
        if op in ops:
            yield normalized


def build_prototypes(model, data_path, ops, steps_per_chunk=20):
    """Initialize structural prototypes by encoding canonical expressions.

    For each unique (op, result) pair, create a deterministic canonical prompt
    and encode it once to establish the prototype basin.
    """
    print("Building prototypes...", flush=True)
    seen = set()
    for row in iter_repulsion_jsonl(data_path):
        normalized = normalize_repulsion_row(row)
        key = normalized["correct_key"]
        op = key.split(":", 1)[0]
        if op not in ops or key in seen:
            continue
        seen.add(key)

        # Encode the prompt to get a basin, then use it as prototype
        basin, _ = model.encode_basin(normalized["prompt"], steps_per_chunk=steps_per_chunk, reset=True)
        model.structural_prototypes[key] = basin.center.copy()

    print(f"  Built {len(model.structural_prototypes)} prototypes for ops: {ops}", flush=True)
    return model


def sample_global_negative_keys(model, correct_key, hard_wrong_keys, *, operation=None, limit=64):
    """Sample operation-local negatives so scorer training matches global ranking."""

    if limit is None or int(limit) == 0:
        return list(dict.fromkeys(str(key) for key in hard_wrong_keys if str(key) != str(correct_key)))
    op_prefix = f"{operation}:" if operation else None
    pool = [
        str(key)
        for key in model.structural_prototypes
        if str(key) != str(correct_key)
        and (op_prefix is None or str(key).startswith(op_prefix))
    ]
    hard = [str(key) for key in hard_wrong_keys if str(key) != str(correct_key) and str(key) in model.structural_prototypes]
    remaining = [key for key in pool if key not in set(hard)]
    if int(limit) < 0 or int(limit) >= len(remaining):
        sampled = remaining
    else:
        sampled = random.sample(remaining, k=int(limit))
    return list(dict.fromkeys([*hard, *sampled]))


def evaluate_nearest_accuracy(model, data_path, ops, max_eval=500, steps_per_chunk=20):
    """Evaluate nearest-prototype target accuracy on a sample of rows."""
    correct = 0
    total = 0
    distances = []
    errors = []

    for i, row in enumerate(iter_repulsion_jsonl(data_path)):
        if i >= max_eval:
            break
        normalized = normalize_repulsion_row(row)
        op = normalized["correct_key"].split(":", 1)[0]
        if op not in ops:
            continue

        basin, pred_error = model.encode_basin(
            normalized["prompt"], steps_per_chunk=steps_per_chunk, reset=True
        )
        correct_key = normalized["correct_key"]
        target = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
        readout_center = model.read_resonant_slots(basin.center)["center"]
        distance = float(np.linalg.norm(readout_center - target, ord=2))
        distances.append(distance)

        nearest = model.nearest_structural_prototype(basin.center, operation=op, k=1)
        match = nearest and nearest[0]["key"] == correct_key
        if match:
            correct += 1
        total += 1
        errors.append(pred_error)

    if total == 0:
        return {"accuracy": 0.0, "mean_distance": 0.0, "total": 0}
    return {
        "accuracy": correct / total,
        "mean_distance": sum(distances) / len(distances),
        "mean_prediction_error": sum(errors) / len(errors),
        "total": total,
        "correct": correct,
    }


def evaluate_generation(model, prompts, steps_per_chunk=20):
    """Try generating from the decoder and report top-3 tokens."""
    results = []
    for prompt in prompts:
        basin, pred_error = model.encode_basin(prompt, steps_per_chunk=steps_per_chunk, reset=True)
        if model.decoder is None:
            results.append({"prompt": prompt, "error": "no decoder"})
            continue

        import torch
        device = next(model.decoder.parameters()).device
        features = torch.as_tensor(basin.center, dtype=torch.float32, device=device).view(1, -1)
        with torch.no_grad():
            logits = model.decoder(features)
            if logits.shape[-1] > len(model.vocab):
                logits[:, len(model.vocab):] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            top_ids = torch.topk(probs[0], k=3).indices.tolist()
            top_tokens = [model.vocab.decode_id(tid) for tid in top_ids]
            top_probs = torch.topk(probs[0], k=3).values.tolist()

        results.append({
            "prompt": prompt,
            "top_3": list(zip(top_tokens, [float(p) for p in top_probs])),
            "prediction_error": pred_error,
        })
    return results


def train_combined(
    model,
    data_path,
    *,
    phase="combined",
    ops=("add", "mul"),
    steps_per_chunk=20,
    batch_size=32,
    lr=5e-4,
    push_pull_strength=0.05,
    wrong_strength=0.5,
    topology_update="auto",
    max_rows=None,
    max_steps=None,
    epochs=1,
    save_interval=1000,
    out_dir="runs/combined-model",
    device="cpu",
    initial_eval=None,
    gate_carve_threshold=0.5,
    delta_carve_margin=0.1,
    delta_global_negatives=64,
    joint_carve_margin=0.02,
    joint_global_negatives=12,
    joint_settle_tail=6,
):
    """Train with phase-gated topology carving and/or decoder backprop."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "combined_records.jsonl"

    # Reinit optimizer with new learning rate
    model.reset_optimizer(lr)
    train_decoder = phase in {"decoder", "combined"}
    train_topology = phase in {"manifold", "topology", "combined"}
    set_decoder_trainable(model, train_decoder)
    resolved_topology_update = topology_update
    if phase == "manifold":
        resolved_topology_update = "manifold"
    elif resolved_topology_update == "auto":
        if phase == "topology" and getattr(model, "num_slots", 0) <= 0:
            resolved_topology_update = "delta-score"
        else:
            resolved_topology_update = "resonance" if getattr(model, "num_slots", 0) > 0 else "sparse"

    summary = {
        "mode": "sequential",
        "phase": phase,
        "grid_size": model.config.width,
        "basin_dim": model.basin_dim,
        "ops": list(ops),
        "prototype_count": len(model.structural_prototypes),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "push_pull_strength": push_pull_strength,
        "wrong_strength": wrong_strength,
        "topology_update": resolved_topology_update,
        "num_slots": getattr(model, "num_slots", 0),
        "decoder_device": str(device),
        "train_decoder": train_decoder,
        "train_topology": train_topology,
        "max_steps": max_steps,
        "eval_before": initial_eval,
        "gate_carve_threshold": gate_carve_threshold,
        "delta_carve_margin": delta_carve_margin,
        "delta_global_negatives": delta_global_negatives,
        "joint_carve_margin": joint_carve_margin,
        "joint_global_negatives": joint_global_negatives,
        "joint_settle_tail": joint_settle_tail,
    }
    total_updates = 0
    manifold_seen_keys: set[str] = set()

    with records_path.open("w", encoding="utf-8") as records:
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===", flush=True)
            start = time.time()

            # Load and shuffle data
            rows = list(filter_by_op(iter_repulsion_jsonl(data_path), ops))
            if max_rows:
                rows = rows[:max_rows]
            random.shuffle(rows)
            print(f"  Training on {len(rows)} rows", flush=True)

            # Accumulate batch
            batch_centers = []
            batch_targets = []
            epoch_losses = []
            epoch_pull_l2s = []
            epoch_push_l2s = []
            epoch_gate_losses = []
            epoch_gate_margins = []
            epoch_gate_probs = []
            epoch_gate_top_matches = []
            epoch_delta_losses = []
            epoch_delta_scores = []
            epoch_delta_wrong_scores = []
            epoch_delta_margins = []
            epoch_delta_top_matches = []
            epoch_joint_scores = []
            epoch_joint_margins = []
            epoch_joint_top_matches = []
            used = 0
            skipped = 0
            carved = 0

            for idx, normalized in enumerate(rows):
                if max_steps is not None and total_updates >= int(max_steps):
                    break
                correct_key = normalized["correct_key"]
                op = correct_key.split(":", 1)[0]
                target_proto = np.asarray(model.structural_prototypes[correct_key], dtype=np.float32)
                wrong_keys = [k for k in normalized.get("wrong_keys", [])
                              if k != correct_key and k in model.structural_prototypes]
                delta_wrong_keys = sample_global_negative_keys(
                    model,
                    correct_key,
                    wrong_keys,
                    operation=op,
                    limit=delta_global_negatives,
                )
                joint_candidate_keys = [correct_key, *sample_global_negative_keys(
                    model,
                    correct_key,
                    wrong_keys,
                    operation=op,
                    limit=joint_global_negatives,
                )]

                if phase == "manifold":
                    if correct_key in manifold_seen_keys:
                        skipped += 1
                        metrics = {"used": False, "reason": "duplicate_manifold_key"}
                    else:
                        metrics = model.carve_computation_manifold(
                            correct_key=correct_key,
                            result_state=target_proto,
                            wrong_keys=wrong_keys,
                            strength=push_pull_strength,
                        )
                        pull_l2 = float(metrics.get("delta_l2", 0.0)) if metrics.get("used") else 0.0
                        if metrics.get("used"):
                            manifold_seen_keys.add(correct_key)
                            epoch_pull_l2s.append(pull_l2)
                            used += 1
                            total_updates += 1
                        else:
                            skipped += 1

                    if (idx + 1) % 500 == 0:
                        mean_pull = sum(epoch_pull_l2s) / len(epoch_pull_l2s) if epoch_pull_l2s else 0
                        print(
                            f"  Row {idx + 1}/{len(rows)}: "
                            f"loss=n/a pull_l2={mean_pull:.3f} update={resolved_topology_update} "
                            f"used={used} skipped={skipped} total={total_updates}",
                            flush=True,
                        )
                        records.write(json.dumps({
                            "row": idx + 1,
                            "epoch": epoch + 1,
                            "phase": phase,
                            "topology_update": resolved_topology_update,
                            "mean_loss": None,
                            "mean_pull_l2": mean_pull,
                            "used": used,
                            "skipped": skipped,
                            "total_updates": total_updates,
                        }) + "\n")
                        records.flush()

                    if save_interval > 0 and (idx + 1) % save_interval == 0:
                        ckpt = output / f"checkpoint-{epoch:03d}-{idx + 1:07d}"
                        model.save(ckpt)
                        print(f"  Saved checkpoint: {ckpt}", flush=True)
                    continue

                # Encode prompt
                basin, pred_error = model.encode_basin(
                    normalized["prompt"], steps_per_chunk=steps_per_chunk, reset=True
                )

                # Get target token ID
                target_token = normalized.get("target", correct_key.split(":", 1)[1])
                target_id = model.vocab.token_to_idx.get(target_token)
                if target_id is None:
                    target_id = model.vocab.add(target_token)

                batch_centers.append(basin.center.tolist())
                batch_targets.append(target_id)

                wrong_targets = [np.asarray(model.structural_prototypes[k], dtype=np.float32)
                                 for k in wrong_keys]
                delta_wrong_targets = [np.asarray(model.structural_prototypes[k], dtype=np.float32)
                                       for k in delta_wrong_keys]

                pull_l2 = float(np.linalg.norm(target_proto - basin.center, ord=2))
                metrics = {"used": True}
                if train_topology:
                    if resolved_topology_update == "joint-stability":
                        joint_metrics = model.score_joint_candidates(
                            normalized["prompt"],
                            joint_candidate_keys,
                            steps_per_chunk=steps_per_chunk,
                            settle_tail=joint_settle_tail,
                        )
                        joint_top_match = joint_metrics.get("best_key") == correct_key
                        epoch_joint_scores.append(float(joint_metrics.get("best_score", 0.0)))
                        epoch_joint_margins.append(float(joint_metrics.get("margin", 0.0)))
                        epoch_joint_top_matches.append(1.0 if joint_top_match else 0.0)
                        if joint_top_match and float(joint_metrics.get("margin", 0.0)) >= float(joint_carve_margin):
                            if getattr(model, "num_slots", 0) > 0:
                                metrics = model.carve_resonance_slot(
                                    basin,
                                    target_proto,
                                    wrong_targets,
                                    correct_key=correct_key,
                                    strength=push_pull_strength,
                                    wrong_strength=wrong_strength,
                                )
                            else:
                                metrics = model.carve_sparse_tunnel(
                                    basin,
                                    target_proto,
                                    wrong_targets,
                                    correct_key=correct_key,
                                    strength=push_pull_strength,
                                    wrong_strength=wrong_strength,
                                )
                            metrics["joint_carved"] = True
                            metrics["joint_best_key"] = joint_metrics.get("best_key")
                            metrics["joint_margin"] = joint_metrics.get("margin", 0.0)
                            carved += 1
                        else:
                            metrics = {
                                "used": True,
                                "correct_key": correct_key,
                                "joint_carved": False,
                                "joint_best_key": joint_metrics.get("best_key"),
                                "joint_margin": joint_metrics.get("margin", 0.0),
                                "reason": "joint_below_margin_or_wrong_top1",
                            }
                    elif resolved_topology_update == "delta-score":
                        delta_metrics = model.train_delta_contrastive(
                            basin.center,
                            target_proto,
                            delta_wrong_targets,
                        )
                        if delta_metrics.get("used"):
                            epoch_delta_losses.append(float(delta_metrics["loss"]))
                            epoch_delta_scores.append(float(delta_metrics["target_score"]))
                            epoch_delta_wrong_scores.append(float(delta_metrics["wrong_score"]))
                            epoch_delta_margins.append(float(delta_metrics["delta_margin"]))
                            epoch_delta_top_matches.append(1.0 if delta_metrics.get("top_match") else 0.0)
                        if (
                            bool(delta_metrics.get("top_match"))
                            and float(delta_metrics.get("delta_margin", 0.0)) >= float(delta_carve_margin)
                        ):
                            metrics = model.carve_sparse_tunnel(
                                basin,
                                target_proto,
                                wrong_targets,
                                correct_key=correct_key,
                                strength=push_pull_strength,
                                wrong_strength=wrong_strength,
                            )
                            metrics["delta_carved"] = True
                            carved += 1
                        else:
                            metrics = {
                                "used": True,
                                "correct_key": correct_key,
                                "delta_carved": False,
                                "reason": "delta_below_threshold",
                            }
                    elif resolved_topology_update == "resonance":
                        gate_metrics = model.train_gate_contrastive(
                            basin.center,
                            correct_key=correct_key,
                            wrong_keys=wrong_keys,
                        )
                        if gate_metrics.get("used"):
                            epoch_gate_losses.append(float(gate_metrics["loss"]))
                            epoch_gate_margins.append(float(gate_metrics["gate_margin"]))
                            epoch_gate_probs.append(float(gate_metrics["correct_probability"]))
                            epoch_gate_top_matches.append(1.0 if gate_metrics.get("top_match") else 0.0)
                        if float(gate_metrics.get("correct_probability", 0.0)) >= float(gate_carve_threshold):
                            metrics = model.carve_resonance_slot(
                                basin,
                                target_proto,
                                wrong_targets,
                                correct_key=correct_key,
                                strength=push_pull_strength,
                                wrong_strength=wrong_strength,
                            )
                            metrics["gate_carved"] = True
                            carved += 1
                        else:
                            metrics = {
                                "used": True,
                                "correct_key": correct_key,
                                "gate_carved": False,
                                "reason": "gate_below_threshold",
                            }
                    elif resolved_topology_update == "sparse":
                        metrics = model.carve_sparse_tunnel(
                            basin,
                            target_proto,
                            wrong_targets,
                            correct_key=correct_key,
                            strength=push_pull_strength,
                            wrong_strength=wrong_strength,
                        )
                    else:
                        metrics = model.carve_push_pull(
                            basin, target_proto, wrong_targets,
                            correct_key=correct_key,
                            wrong_keys=wrong_keys,
                            strength=push_pull_strength,
                            wrong_strength=wrong_strength,
                        )

                if metrics.get("used"):
                    epoch_pull_l2s.append(pull_l2)
                    if wrong_targets:
                        push_l2 = float(np.mean([
                            np.linalg.norm(basin.center - wt, ord=2)
                            for wt in wrong_targets
                        ]))
                        epoch_push_l2s.append(push_l2)
                    used += 1
                else:
                    skipped += 1

                # Flush batch
                if train_decoder and len(batch_centers) >= batch_size:
                    loss = model.train_decoder_batch(batch_centers, batch_targets)
                    epoch_losses.append(loss)
                    batch_centers.clear()
                    batch_targets.clear()
                elif not train_decoder:
                    batch_centers.clear()
                    batch_targets.clear()
                total_updates += 1

                # Log every 500 rows
                if (idx + 1) % 500 == 0:
                    mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
                    mean_pull = sum(epoch_pull_l2s) / len(epoch_pull_l2s) if epoch_pull_l2s else 0
                    mean_gate_loss = sum(epoch_gate_losses) / len(epoch_gate_losses) if epoch_gate_losses else None
                    mean_gate_margin = sum(epoch_gate_margins) / len(epoch_gate_margins) if epoch_gate_margins else None
                    mean_gate_prob = sum(epoch_gate_probs) / len(epoch_gate_probs) if epoch_gate_probs else None
                    gate_top1 = sum(epoch_gate_top_matches) / len(epoch_gate_top_matches) if epoch_gate_top_matches else None
                    mean_delta_loss = sum(epoch_delta_losses) / len(epoch_delta_losses) if epoch_delta_losses else None
                    mean_delta_score = sum(epoch_delta_scores) / len(epoch_delta_scores) if epoch_delta_scores else None
                    mean_delta_wrong = sum(epoch_delta_wrong_scores) / len(epoch_delta_wrong_scores) if epoch_delta_wrong_scores else None
                    mean_delta_margin = sum(epoch_delta_margins) / len(epoch_delta_margins) if epoch_delta_margins else None
                    delta_top1 = sum(epoch_delta_top_matches) / len(epoch_delta_top_matches) if epoch_delta_top_matches else None
                    mean_joint_score = sum(epoch_joint_scores) / len(epoch_joint_scores) if epoch_joint_scores else None
                    mean_joint_margin = sum(epoch_joint_margins) / len(epoch_joint_margins) if epoch_joint_margins else None
                    joint_top1 = sum(epoch_joint_top_matches) / len(epoch_joint_top_matches) if epoch_joint_top_matches else None
                    print(
                        f"  Row {idx + 1}/{len(rows)}: "
                        f"loss={mean_loss if mean_loss is not None else 'n/a'} "
                        f"pull_l2={mean_pull:.3f} update={resolved_topology_update} "
                        f"gate_loss={mean_gate_loss if mean_gate_loss is not None else 'n/a'} "
                        f"gate_prob={mean_gate_prob if mean_gate_prob is not None else 'n/a'} "
                        f"gate_margin={mean_gate_margin if mean_gate_margin is not None else 'n/a'} "
                        f"gate_top1={gate_top1 if gate_top1 is not None else 'n/a'} "
                        f"delta_loss={mean_delta_loss if mean_delta_loss is not None else 'n/a'} "
                        f"delta_score={mean_delta_score if mean_delta_score is not None else 'n/a'} "
                        f"delta_wrong={mean_delta_wrong if mean_delta_wrong is not None else 'n/a'} "
                        f"delta_margin={mean_delta_margin if mean_delta_margin is not None else 'n/a'} "
                        f"delta_top1={delta_top1 if delta_top1 is not None else 'n/a'} "
                        f"joint_score={mean_joint_score if mean_joint_score is not None else 'n/a'} "
                        f"joint_margin={mean_joint_margin if mean_joint_margin is not None else 'n/a'} "
                        f"joint_top1={joint_top1 if joint_top1 is not None else 'n/a'} "
                        f"carved={carved} used={used} skipped={skipped} "
                        f"total={total_updates}",
                        flush=True,
                    )
                    records.write(json.dumps({
                        "row": idx + 1,
                        "epoch": epoch + 1,
                        "phase": phase,
                        "topology_update": resolved_topology_update,
                        "mean_loss": mean_loss,
                        "mean_pull_l2": mean_pull,
                        "mean_gate_loss": mean_gate_loss,
                        "mean_gate_probability": mean_gate_prob,
                        "mean_gate_margin": mean_gate_margin,
                        "gate_top1": gate_top1,
                        "mean_delta_loss": mean_delta_loss,
                        "mean_delta_score": mean_delta_score,
                        "mean_delta_wrong_score": mean_delta_wrong,
                        "mean_delta_margin": mean_delta_margin,
                        "delta_top1": delta_top1,
                        "mean_joint_score": mean_joint_score,
                        "mean_joint_margin": mean_joint_margin,
                        "joint_top1": joint_top1,
                        "carved": carved,
                        "used": used,
                        "skipped": skipped,
                        "total_updates": total_updates,
                    }) + "\n")
                    records.flush()

                # Save checkpoint
                if save_interval > 0 and (idx + 1) % save_interval == 0:
                    ckpt = output / f"checkpoint-{epoch:03d}-{idx + 1:07d}"
                    model.save(ckpt)
                    print(f"  Saved checkpoint: {ckpt}", flush=True)

            # Flush remaining batch
            if train_decoder and batch_centers:
                loss = model.train_decoder_batch(batch_centers, batch_targets)
                epoch_losses.append(loss)

            elapsed = time.time() - start
            mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
            mean_pull = sum(epoch_pull_l2s) / len(epoch_pull_l2s) if epoch_pull_l2s else 0
            mean_push = sum(epoch_push_l2s) / len(epoch_push_l2s) if epoch_push_l2s else 0
            mean_gate_loss = sum(epoch_gate_losses) / len(epoch_gate_losses) if epoch_gate_losses else None
            mean_gate_margin = sum(epoch_gate_margins) / len(epoch_gate_margins) if epoch_gate_margins else None
            mean_gate_prob = sum(epoch_gate_probs) / len(epoch_gate_probs) if epoch_gate_probs else None
            gate_top1 = sum(epoch_gate_top_matches) / len(epoch_gate_top_matches) if epoch_gate_top_matches else None
            mean_delta_loss = sum(epoch_delta_losses) / len(epoch_delta_losses) if epoch_delta_losses else None
            mean_delta_score = sum(epoch_delta_scores) / len(epoch_delta_scores) if epoch_delta_scores else None
            mean_delta_wrong = sum(epoch_delta_wrong_scores) / len(epoch_delta_wrong_scores) if epoch_delta_wrong_scores else None
            mean_delta_margin = sum(epoch_delta_margins) / len(epoch_delta_margins) if epoch_delta_margins else None
            delta_top1 = sum(epoch_delta_top_matches) / len(epoch_delta_top_matches) if epoch_delta_top_matches else None
            mean_joint_score = sum(epoch_joint_scores) / len(epoch_joint_scores) if epoch_joint_scores else None
            mean_joint_margin = sum(epoch_joint_margins) / len(epoch_joint_margins) if epoch_joint_margins else None
            joint_top1 = sum(epoch_joint_top_matches) / len(epoch_joint_top_matches) if epoch_joint_top_matches else None

            summary["epoch"] = epoch + 1
            summary["mean_loss"] = mean_loss
            summary["mean_pull_l2"] = mean_pull
            summary["mean_push_l2"] = mean_push
            summary["mean_gate_loss"] = mean_gate_loss
            summary["mean_gate_probability"] = mean_gate_prob
            summary["mean_gate_margin"] = mean_gate_margin
            summary["gate_top1"] = gate_top1
            summary["mean_delta_loss"] = mean_delta_loss
            summary["mean_delta_score"] = mean_delta_score
            summary["mean_delta_wrong_score"] = mean_delta_wrong
            summary["mean_delta_margin"] = mean_delta_margin
            summary["delta_top1"] = delta_top1
            summary["mean_joint_score"] = mean_joint_score
            summary["mean_joint_margin"] = mean_joint_margin
            summary["joint_top1"] = joint_top1
            summary["carved"] = carved
            summary["used"] = used
            summary["skipped"] = skipped
            summary["elapsed_s"] = elapsed
            summary["rows_used"] = used
            summary["rows_skipped"] = skipped
            summary["total_updates"] = total_updates
            summary["manifold_unique_keys"] = len(manifold_seen_keys)

            print(f"  Epoch done: loss={mean_loss if mean_loss is not None else 'n/a'} "
                  f"pull={mean_pull:.3f} push={mean_push:.3f} "
                  f"gate_loss={mean_gate_loss if mean_gate_loss is not None else 'n/a'} "
                  f"gate_prob={mean_gate_prob if mean_gate_prob is not None else 'n/a'} "
                  f"gate_margin={mean_gate_margin if mean_gate_margin is not None else 'n/a'} "
                  f"gate_top1={gate_top1 if gate_top1 is not None else 'n/a'} "
                  f"delta_loss={mean_delta_loss if mean_delta_loss is not None else 'n/a'} "
                  f"delta_score={mean_delta_score if mean_delta_score is not None else 'n/a'} "
                  f"delta_wrong={mean_delta_wrong if mean_delta_wrong is not None else 'n/a'} "
                  f"delta_margin={mean_delta_margin if mean_delta_margin is not None else 'n/a'} "
                  f"delta_top1={delta_top1 if delta_top1 is not None else 'n/a'} "
                  f"joint_score={mean_joint_score if mean_joint_score is not None else 'n/a'} "
                  f"joint_margin={mean_joint_margin if mean_joint_margin is not None else 'n/a'} "
                  f"joint_top1={joint_top1 if joint_top1 is not None else 'n/a'} "
                  f"carved={carved} used={used} skipped={skipped} time={elapsed:.0f}s", flush=True)

            # Evaluate after each epoch
            eval_result = evaluate_nearest_accuracy(
                model, data_path, ops, max_eval=500, steps_per_chunk=steps_per_chunk
            )
            summary[f"eval_epoch_{epoch + 1}"] = eval_result
            print(f"  Eval: accuracy={eval_result['accuracy']:.3f} "
                  f"distance={eval_result['mean_distance']:.3f} "
                  f"({eval_result['correct']}/{eval_result['total']})", flush=True)

            # Save final
            final_dir = output / "final"
            model.save(final_dir)
            summary["latest_checkpoint"] = str(final_dir)
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"  Saved final model: {final_dir}", flush=True)
            if max_steps is not None and total_updates >= int(max_steps):
                break

    # Write summary
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Generate some examples
    test_prompts = [
        "8 plus 9",
        "9 plus 8",
        "15 plus 82",
        "7 times 3",
        "17 times 4",
    ]
    gen_results = evaluate_generation(model, test_prompts, steps_per_chunk=steps_per_chunk)
    (output / "generation_examples.json").write_text(json.dumps(gen_results, indent=2) + "\n")
    print("\n=== Generation Examples ===")
    for ex in gen_results:
        print(f"  {ex['prompt']} -> {ex.get('top_3', ex.get('error', 'N/A'))}", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Combined decoder + push-pull training")
    parser.add_argument("--data", type=Path, default=Path("runs/repulsion_data.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/combined-model"))
    parser.add_argument("--load-model-dir", type=Path, default=None)
    parser.add_argument("--phase", choices=["manifold", "topology", "decoder", "combined"], default="combined")
    parser.add_argument("--max-rows", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=None, help="Total row updates across epochs.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--basin-dim", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steps-per-chunk", type=int, default=20)
    parser.add_argument("--push-pull-strength", type=float, default=0.05)
    parser.add_argument("--wrong-strength", type=float, default=0.5)
    parser.add_argument(
        "--topology-update",
        choices=["auto", "sparse", "push-pull", "resonance", "delta-score", "joint-stability"],
        default="auto",
        help="Topology update path. auto uses delta-score for slot-free topology, resonance with slots.",
    )
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ops", type=str, default="add,mul", help="Comma-separated operations to train")
    parser.add_argument("--device", default="auto", help="Torch decoder device: auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"], help="Phase-field backend")
    parser.add_argument("--encoder-mode", choices=["text", "structured"], default="text")
    parser.add_argument("--structured-result-hint", action="store_true", help="Leak result hints for upper-bound ablations only.")
    parser.add_argument("--structured-feature-strength", type=float, default=2.0)
    parser.add_argument("--num-slots", type=int, default=0, help="Keyed resonance slots per active grid cell.")
    parser.add_argument(
        "--gate-carve-threshold",
        type=float,
        default=0.5,
        help="Only carve a resonance slot when the learned gate assigns this probability to the correct slot.",
    )
    parser.add_argument(
        "--delta-carve-margin",
        type=float,
        default=0.1,
        help="Only carve topology when the delta scorer ranks the correct candidate above negatives by this margin.",
    )
    parser.add_argument(
        "--delta-global-negatives",
        type=int,
        default=64,
        help="Sample this many operation-local global negatives for delta training; -1 uses all.",
    )
    parser.add_argument(
        "--joint-carve-margin",
        type=float,
        default=0.02,
        help="Only carve when joint-stability selects the correct candidate by this score margin.",
    )
    parser.add_argument(
        "--joint-global-negatives",
        type=int,
        default=12,
        help="Sample this many operation-local global negatives for joint-stability scoring; -1 uses all.",
    )
    parser.add_argument(
        "--joint-settle-tail",
        type=int,
        default=6,
        help="Number of final settle steps used to estimate joint basin stability.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ops = tuple(args.ops.split(","))
    device = select_torch_device(args.device)
    print(f"Training on ops: {ops}", flush=True)
    print(f"Requested grid: {args.grid_size}x{args.grid_size}, basin dim: {args.basin_dim}", flush=True)
    print(f"Decoder device: {device}; field backend: {args.backend}", flush=True)

    if args.load_model_dir is not None:
        print(f"Loading model: {args.load_model_dir}", flush=True)
        model = PhaseModel.load(args.load_model_dir)
        model.learning_rate = args.lr
    else:
        model = PhaseModel(
            grid_size=args.grid_size,
            basin_dim=args.basin_dim,
            hidden=args.hidden,
            vocab_capacity=4096,
            seed=args.seed,
            backend=args.backend,
            pin_strength=0.25,
            residual_carry=0.08,
            learning_rate=args.lr,
            num_slots=args.num_slots,
            encoder_mode=args.encoder_mode,
            structured_result_hint=args.structured_result_hint,
            structured_feature_strength=args.structured_feature_strength,
            create_decoder=True,
        )
    if args.num_slots > 0 and getattr(model, "num_slots", 0) != args.num_slots:
        print(f"Allocating resonance slots: {args.num_slots}", flush=True)
        model.num_slots = int(args.num_slots)
        model._ensure_feature_slots()
    model = configure_decoder_device(model, device)
    print(
        f"Active model: {model.config.width}x{model.config.height}, "
        f"basin dim: {model.basin_dim}, slots: {getattr(model, 'num_slots', 0)}, "
        f"encoder: {getattr(model, 'encoder_mode', 'text')}",
        flush=True,
    )

    if not model.structural_prototypes:
        model = build_prototypes(model, args.data, ops, steps_per_chunk=args.steps_per_chunk)
    else:
        print(f"Using {len(model.structural_prototypes)} loaded prototypes", flush=True)

    # Evaluate before training
    print("\n=== Pre-training Evaluation ===", flush=True)
    eval_before = evaluate_nearest_accuracy(
        model, args.data, ops, max_eval=500, steps_per_chunk=args.steps_per_chunk
    )
    print(f"  Accuracy: {eval_before['accuracy']:.3f} "
          f"Distance: {eval_before['mean_distance']:.3f} "
          f"({eval_before['correct']}/{eval_before['total']})", flush=True)

    # Train
    summary = train_combined(
        model,
        args.data,
        phase=args.phase,
        ops=ops,
        steps_per_chunk=args.steps_per_chunk,
        batch_size=args.batch_size,
        lr=args.lr,
        push_pull_strength=args.push_pull_strength,
        wrong_strength=args.wrong_strength,
        topology_update=args.topology_update,
        max_rows=args.max_rows,
        max_steps=args.max_steps,
        epochs=args.epochs,
        save_interval=args.save_interval,
        out_dir=args.out,
        device=device,
        initial_eval=eval_before,
        gate_carve_threshold=args.gate_carve_threshold,
        delta_carve_margin=args.delta_carve_margin,
        delta_global_negatives=args.delta_global_negatives,
        joint_carve_margin=args.joint_carve_margin,
        joint_global_negatives=args.joint_global_negatives,
        joint_settle_tail=args.joint_settle_tail,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nFinal summary: {json.dumps(summary, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
