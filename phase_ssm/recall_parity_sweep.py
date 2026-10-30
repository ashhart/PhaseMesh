"""EXPERIMENT 1 (THE GATE) — recall parity: does the linear dk**2 phase memory
carry recall load over SSM-only, at high pair-count, where a transformer holds?

This is the single highest-EV experiment in the PhaseMesh portfolio. It is the
go/no-go gate for the whole "post-attention" thesis.

WHY THIS SUPERSEDES PRIOR RESULTS
  - prior mqar_sweep.py   : head_dim=48 (dk**2=2304), never run at power
  - prior mqar_signal.py  : head_dim=16, 800 steps CPU, transformer ALSO at chance
                            -> a negative where the positive control failed: invalid
  - prior premise_sweep.py: head_dim=16, written but NEVER RUN (no result file)

This script fixes all three: ONE config (HEAD_DIM=32 from poc_common), adequate
steps, multi-seed, and a transformer positive control with the same head_dim.

DECISION RULE (pre-registered, in poc_common.verdict_recal):
    XF holds, OURS holds, SSM collapses -> THESIS_LIVES   (build the POC)
    XF holds, OURS collapses w/ SSM      -> THESIS_DEAD   (fold the line, ship product)
    XF below POWER_FAIL_BAR             -> UNDERPOWERED  (no verdict; raise power)

The pair sweep P in {16, 32, 64, 128} finds the SSM's collapse point. At low P
every competent model solves it (the prior P=4 "both perfect" result). The
differentiator only appears once SSM capacity is exceeded.

Run on the GB10:
    python -m phase_ssm.recall_parity_sweep --steps 6000 --seeds 3 --device cuda
Quick smoke on CPU:
    python -m phase_ssm.recall_parity_sweep --smoke
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .poc_common import (
    HEAD_DIM, MEM_HEADS, D_MODEL, N_LAYERS, STATE_DIM,
    DEFAULT_STEPS, DEFAULT_SEEDS, HOLD_THRESHOLD,
    base_phasessm, base_transformer, run_arm, verdict_recal,
)


def run_P(P: int, *, steps: int, seeds: int, device: str, log=print) -> dict:
    # Scale keys/values with P so the task stays hard (more distractors at high P).
    NK = NV = max(256, 4 * P)
    vocab = 1 + NK + NV
    n_queries = max(8, P // 2)
    log(f"\n=== P={P}  (NK=NV={NK}, Q={n_queries}, chance={1/NV:.4f}, 1/P={1/P:.3f}) ===")

    def mk_ours():
        return base_phasessm(use_mem=True, mem_linear=True, vocab=vocab)
    def mk_ssm():
        return base_phasessm(use_mem=False, vocab=vocab)
    def mk_xf():
        return base_transformer(vocab=vocab)

    # transformer needs a lower LR for induction-head formation (justified by
    # the LR-sensitivity note in the original mqar_sweep.py).
    ours = run_arm(mk_ours, steps=steps, seeds=seeds, lr=1e-3, device=device,
                   n_pairs=P, n_queries=n_queries, n_keys=NK, n_values=NV, name="OURS+mem", log=log)
    ssm = run_arm(mk_ssm, steps=steps, seeds=seeds, lr=1e-3, device=device,
                  n_pairs=P, n_queries=n_queries, n_keys=NK, n_values=NV, name="SSM-only", log=log)
    xf = run_arm(mk_xf, steps=steps, seeds=seeds, lr=3e-4, device=device,
                 n_pairs=P, n_queries=n_queries, n_keys=NK, n_values=NV, name="Transformer", log=log)

    gap = ours.mean - ssm.mean
    verdict = verdict_recal(ours, ssm, xf, P=P)
    log(f"  >>> P={P}: SSM {ssm.mean:.3f} | OURS {ours.mean:.3f} | XF {xf.mean:.3f}"
        f"  (OURS-SSM {gap:+.3f}, hold={HOLD_THRESHOLD})")
    log(f"  VERDICT: {verdict}")
    return {
        "P": P, "NK": NK, "NV": NV, "Q": n_queries, "chance": 1 / NV, "1_over_P": 1 / P,
        "OURS+mem": ours.to_dict(), "SSM-only": ssm.to_dict(), "Transformer": xf.to_dict(),
        "gap_ours_minus_ssm": gap, "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/poc_recall_parity.json")
    ap.add_argument("--markdown", default="results/poc_recall_parity.md")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to prove the script works")
    args = ap.parse_args()

    if args.smoke:
        args.pairs = [8]
        args.steps = 15
        args.seeds = 1
        args.out = "results/poc_recall_smoke.json"
        args.markdown = "results/poc_recall_smoke.md"
        # shrink everything for a fast CPU code-path check (NOT a real result)
        import phase_ssm.poc_common as _pc
        _pc.D_MODEL = 48
        _pc.N_LAYERS = 1
        _pc.MEM_HEADS = 2
        _pc.HEAD_DIM = 16
        _pc.STATE_DIM = 16
        # keep real thresholds so the verdict logic is exercised honestly;
        # at 15 steps nothing will hold, so the correct smoke verdict is UNDERPOWERED.
        print("[SMOKE] tiny run to verify the pipeline. NOT a real result.")

    print(f"RECALL PARITY SWEEP (THE GATE)")
    print(f"  config: d={D_MODEL} L={N_LAYERS} state={STATE_DIM} "
          f"mem_heads={MEM_HEADS} dk={HEAD_DIM} (dk**2={HEAD_DIM**2})")
    print(f"  steps={args.steps} seeds={args.seeds} device={args.device} hold={HOLD_THRESHOLD}")

    t0 = time.time()
    results = {}
    for P in args.pairs:
        results[str(P)] = run_P(P, steps=args.steps, seeds=args.seeds, device=args.device)
        json.dump(results, open(args.out, "w"), indent=2)   # checkpoint after each P

    # Overall verdict = the highest P where the transformer held.
    overall = "UNDERPOWERED — transformer never held at any tested P; raise power."
    for P in args.pairs:
        r = results[str(P)]
        if "THESIS_LIVES" in r["verdict"]:
            overall = f"THESIS_LIVES at P={P}: " + r["verdict"]
            break
        if "THESIS_DEAD" in r["verdict"]:
            overall = f"THESIS_DEAD at P={P}: " + r["verdict"]
            break

    print(f"\n{'='*70}")
    print(f"OVERALL VERDICT: {overall}")
    print(f"(elapsed {time.time()-t0:.0f}s)")
    print(f"{'='*70}")

    # Markdown table
    lines = [
        "# POC Experiment 1 — Recall Parity (THE GATE)",
        "",
        f"Config: d_model={D_MODEL}, n_layers={N_LAYERS}, mem dk={HEAD_DIM} "
        f"(dk**2={HEAD_DIM**2} state/head), steps={args.steps}, seeds={args.seeds}, "
        f"device={args.device}. Hold threshold={HOLD_THRESHOLD}.",
        "",
        "| P | chance | SSM-only | OURS+mem | Transformer | gap (O-S) | verdict |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for P in args.pairs:
        r = results[str(P)]
        v = r["verdict"].split("—")[0].strip()
        lines.append(f"| {P} | {r['chance']:.4f} | {r['SSM-only']['mean']:.3f} | "
                     f"{r['OURS+mem']['mean']:.3f} | {r['Transformer']['mean']:.3f} | "
                     f"{r['gap_ours_minus_ssm']:+.3f} | {v} |")
    lines += ["", f"**OVERALL: {overall}**"]
    open(args.markdown, "w").write("\n".join(lines))
    print(f"wrote {args.out} and {args.markdown}")


if __name__ == "__main__":
    main()
