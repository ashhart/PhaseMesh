from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from phase_mesh.model import PhaseModel
from phase_mesh.trainer import (
    ABLATION_MODES,
    basin_entropy,
    basin_target_mutual_info,
    iter_text_file,
    stream_evaluate,
    stream_train,
)


DEFAULT_MODES = ("full", "no-interference", "random-walk", "static-topology", "uniform-init")


def loss_drop_rate(first: float | None, recent: float | None) -> float | None:
    if first is None or recent is None or first == 0:
        return None
    return float((first - recent) / abs(first))


def run_mode(
    *,
    mode: str,
    data: Path,
    heldout: Path | None,
    out: Path,
    chunks: int,
    eval_chunks: int,
    size: int,
    basin_dim: int,
    hidden: int,
    vocab_capacity: int,
    steps_per_chunk: int,
    batch_size: int,
    context_tokens: int,
    windows_per_chunk: int,
    lr: float,
    seed: int,
    backend: str,
    pin_strength: float,
    residual_carry: float,
) -> dict[str, Any]:
    mode_out = out / mode
    if mode_out.exists():
        shutil.rmtree(mode_out)
    model = PhaseModel(
        grid_size=size,
        vocab_capacity=vocab_capacity,
        basin_dim=basin_dim,
        hidden=hidden,
        seed=seed,
        backend=backend,
        pin_strength=pin_strength,
        residual_carry=residual_carry,
        learning_rate=lr,
        create_decoder=True,
    )
    summary = stream_train(
        model,
        iter_text_file(data),
        steps_per_chunk=steps_per_chunk,
        batch_size=batch_size,
        context_tokens=context_tokens,
        windows_per_chunk=windows_per_chunk,
        window_stride=1,
        save_interval=0,
        out_dir=mode_out,
        max_chunks=chunks,
        train_decoder=True,
        train_topology=True,
        freeze_omega=False,
        train_mode="next-token",
        ablation_mode=mode,
    )
    entropy = basin_entropy(mode_out / "train_records.jsonl")
    mutual_info = basin_target_mutual_info(mode_out / "train_records.jsonl")
    evaluation = None
    if heldout is not None:
        eval_model = PhaseModel.load(mode_out / "final")
        evaluation = stream_evaluate(
            eval_model,
            iter_text_file(heldout),
            steps_per_chunk=steps_per_chunk,
            max_chunks=eval_chunks,
            context_tokens=context_tokens,
            windows_per_chunk=windows_per_chunk,
            window_stride=1,
        )
    return {
        "mode": mode,
        "train": summary,
        "final_model_dir": str(mode_out / "final"),
        "basin_entropy": entropy,
        "basin_target_mutual_info": mutual_info,
        "prediction_error_drop_rate": loss_drop_rate(
            summary.get("prediction_error_first_window"),
            summary.get("prediction_error_recent_window"),
        ),
        "heldout": evaluation,
    }


def run(
    *,
    data: str | Path,
    heldout: str | Path | None = None,
    out: str | Path = "runs/ablation-results",
    modes: list[str] | tuple[str, ...] = DEFAULT_MODES,
    chunks: int = 800,
    eval_chunks: int = 200,
    size: int = 32,
    basin_dim: int = 64,
    hidden: int = 64,
    vocab_capacity: int = 4096,
    steps_per_chunk: int = 12,
    batch_size: int = 32,
    context_tokens: int = 8,
    windows_per_chunk: int = 1,
    lr: float = 5e-4,
    seed: int = 42,
    backend: str = "numpy",
    pin_strength: float = 0.3,
    residual_carry: float = 0.05,
) -> dict[str, Any]:
    data_path = Path(data)
    heldout_path = Path(heldout) if heldout is not None else None
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    invalid = sorted(set(modes) - ABLATION_MODES)
    if invalid:
        raise ValueError(f"unsupported ablation mode(s): {', '.join(invalid)}")
    results = [
        run_mode(
            mode=mode,
            data=data_path,
            heldout=heldout_path,
            out=out_path,
            chunks=chunks,
            eval_chunks=eval_chunks,
            size=size,
            basin_dim=basin_dim,
            hidden=hidden,
            vocab_capacity=vocab_capacity,
            steps_per_chunk=steps_per_chunk,
            batch_size=batch_size,
            context_tokens=context_tokens,
            windows_per_chunk=windows_per_chunk,
            lr=lr,
            seed=seed,
            backend=backend,
            pin_strength=pin_strength,
            residual_carry=residual_carry,
        )
        for mode in modes
    ]
    by_mode = {record["mode"]: record for record in results}
    full = by_mode.get("full")
    comparisons: dict[str, Any] = {}
    if full is not None:
        full_entropy = float(full["basin_entropy"]["normalized_entropy"])
        full_mi = float(full["basin_target_mutual_info"]["normalized_mutual_info"])
        full_perplexity = (full.get("heldout") or {}).get("perplexity")
        for mode, record in by_mode.items():
            if mode == "full":
                continue
            entropy = float(record["basin_entropy"]["normalized_entropy"])
            mi = float(record["basin_target_mutual_info"]["normalized_mutual_info"])
            perplexity = (record.get("heldout") or {}).get("perplexity")
            comparisons[mode] = {
                "entropy_delta_vs_full": entropy - full_entropy,
                "normalized_mutual_info_delta_vs_full": mi - full_mi,
                "heldout_perplexity_delta_vs_full": (
                    None if full_perplexity is None or perplexity is None else float(perplexity - full_perplexity)
                ),
            }
    payload = {
        "suite": "phase_mesh_ablation_matrix",
        "data": str(data_path),
        "heldout": str(heldout_path) if heldout_path is not None else None,
        "seed": seed,
        "settings": {
            "chunks": chunks,
            "eval_chunks": eval_chunks,
            "size": size,
            "basin_dim": basin_dim,
            "hidden": hidden,
            "vocab_capacity": vocab_capacity,
            "steps_per_chunk": steps_per_chunk,
            "batch_size": batch_size,
            "context_tokens": context_tokens,
            "windows_per_chunk": windows_per_chunk,
            "lr": lr,
            "backend": backend,
            "pin_strength": pin_strength,
            "residual_carry": residual_carry,
        },
        "results": by_mode,
        "comparisons": comparisons,
    }
    (out_path / "ablation-results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/ablation-results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PhaseMesh wave/topology ablations.")
    parser.add_argument("data", type=Path)
    parser.add_argument("--heldout", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/ablation-results"))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--chunks", type=int, default=800)
    parser.add_argument("--eval-chunks", type=int, default=200)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--basin-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--vocab-capacity", type=int, default=4096)
    parser.add_argument("--steps-per-chunk", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-tokens", type=int, default=8)
    parser.add_argument("--windows-per-chunk", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="numpy", choices=["auto", "numpy", "scipy", "jax"])
    parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.3)
    parser.add_argument("--residual-carry", type=float, default=0.05)
    args = parser.parse_args(argv)
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    payload = run(
        data=args.data,
        heldout=args.heldout,
        out=args.out,
        modes=modes,
        chunks=args.chunks,
        eval_chunks=args.eval_chunks,
        size=args.size,
        basin_dim=args.basin_dim,
        hidden=args.hidden,
        vocab_capacity=args.vocab_capacity,
        steps_per_chunk=args.steps_per_chunk,
        batch_size=args.batch_size,
        context_tokens=args.context_tokens,
        windows_per_chunk=args.windows_per_chunk,
        lr=args.lr,
        seed=args.seed,
        backend=args.backend,
        pin_strength=args.pin_strength,
        residual_carry=args.residual_carry,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
