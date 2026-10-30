"""EXPERIMENT 2 — adaptive recall depth: does iterating the READ beat a single read,
and does the gain SCALE WITH LOAD?

This is the genuinely novel piece of the POC. It stands as a result on its own,
REGARDLESS of how Experiment 1 (recall parity) lands:
  - if Exp 1 LIVES: iterated read is the capability amplifier on top of a working
    memory -> the headline mechanism of the post-attention architecture.
  - if Exp 1 DIES: a gain here still means fixed-state recall can be sharpened
    with test-time compute on the recall axis -> a distinct, publishable finding.

ARMS
  single-read : IteratedReadLM(n_refine=1) — the baseline (== standard linear
                phase memory, one-shot read).
  iterated-K  : IteratedReadLM(n_refine=K) — same params/memory, K read iters.
  transformer : the positive control (exact recall, gives the ceiling).

  single-read and iterated-K have IDENTICAL state size (the state is frozen
  during iteration) and ~identical params (only the small refine_q layer differs).
  So any accuracy gap is attributable to READ DEPTH, not capacity. That is the
  clean comparison.

GOLD PATTERN (pre-registered)
  As P grows, single-read drops, iterated-K holds above it, and the GAP WIDENS
  with P. That scaling-with-load is the differentiator: it shows iteration buys
  PRECISION where the fixed state loses it, which is exactly the recall axis.

KILL CONDITIONS
  - iterated-K converges to single-read at all P  -> mechanism is a no-op
  - gain is flat / shrinks as P grows               -> not a recall mechanism

Run on the GB10:
    python -m phase_ssm.adaptive_recall_gate --steps 6000 --seeds 3 --device cuda
Smoke:
    python -m phase_ssm.adaptive_recall_gate --smoke
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .iterated_memory import IteratedReadLM
from . import poc_common as pc
from .poc_common import DEFAULT_STEPS, DEFAULT_SEEDS, base_transformer, run_arm


_SMOKE_NLAYERS = None  # set by --smoke to shrink layer count for fast CPU code-path check

def build_iterated(vocab: int, n_refine: int, *, halt: bool = False, n_layers: int = 2):
    # Read config via the module object so --smoke patches propagate (binding the
    # names directly with `from ... import` would freeze them at import time).
    if _SMOKE_NLAYERS is not None:
        n_layers = _SMOKE_NLAYERS
    return IteratedReadLM(
        vocab=vocab, d_model=pc.D_MODEL, n_layers=n_layers,
        n_heads=pc.MEM_HEADS, head_dim=pc.HEAD_DIM, n_refine=n_refine, halt=halt,
        tie_embeddings=True,
    )


def run_P(P: int, *, refine_steps: list[int], steps: int, seeds: int, device: str,
          log=print) -> dict:
    NK = NV = max(256, 4 * P)
    vocab = 1 + NK + NV
    n_queries = max(8, P // 2)
    log(f"\n=== P={P}  (NK=NV={NK}, Q={n_queries}, chance={1/NV:.4f}, 1/P={1/P:.3f}) ===")

    arms = {}
    # single-read baseline (n_refine=1) — MUST always be present
    if 1 not in refine_steps:
        refine_steps = [1] + refine_steps
    for K in refine_steps:
        arm = run_arm(lambda K=K: build_iterated(vocab, K),
                      steps=steps, seeds=seeds, lr=1e-3, device=device,
                      n_pairs=P, n_queries=n_queries, n_keys=NK, n_values=NV,
                      name=f"iterated-K{K}", log=log)
        arms[f"iterated-K{K}"] = arm

    xf = run_arm(lambda: base_transformer(vocab=vocab),
                 steps=steps, seeds=seeds, lr=3e-4, device=device,
                 n_pairs=P, n_queries=n_queries, n_keys=NK, n_values=NV,
                 name="Transformer", log=log)
    arms["Transformer"] = xf

    single = arms["iterated-K1"].mean
    best_iter = max((arms[f"iterated-K{K}"].mean for K in refine_steps if K > 1), default=0.0)
    gap = best_iter - single
    log(f"  >>> P={P}: single {single:.3f} | best-iter {best_iter:.3f} (gap {gap:+.3f}) | "
        f"XF {xf.mean:.3f}")
    return {
        "P": P, "NK": NK, "NV": NV, "Q": n_queries, "chance": 1 / NV, "1_over_P": 1 / P,
        "arms": {k: v.to_dict() for k, v in arms.items()},
        "single_read": single, "best_iterated": best_iter, "gap": gap,
        "transformer_holds": xf.mean >= pc.HOLD_THRESHOLD,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--refine", type=int, nargs="+", default=[1, 2, 4],
                    help="read-refinement depths to compare (1 = single-read baseline)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/poc_adaptive_recall.json")
    ap.add_argument("--markdown", default="results/poc_adaptive_recall.md")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.pairs = [8]
        args.refine = [1, 2]
        args.steps = 8
        args.seeds = 1
        args.out = "results/poc_adaptive_smoke.json"
        args.markdown = "results/poc_adaptive_smoke.md"
        import phase_ssm.poc_common as _pc
        _pc.D_MODEL = 32
        _pc.MEM_HEADS = 1
        _pc.HEAD_DIM = 8
        _pc.STATE_DIM = 8
        _pc.HOLD_THRESHOLD = 0.0
        # single layer for the smoke (iterated read cost is K * n_layers * chunked read)
        global _SMOKE_NLAYERS
        _SMOKE_NLAYERS = 1
        print("[SMOKE] tiny run to verify the pipeline. NOT a real result.")

    print("ADAPTIVE RECALL DEPTH (EXPERIMENT 2 — the novel mechanism)")
    print(f"  config: d={pc.D_MODEL} L=2 (isolated recall) mem dk={pc.HEAD_DIM} "
          f"(dk**2={pc.HEAD_DIM**2})")
    print(f"  refine depths: {args.refine}  steps={args.steps} seeds={args.seeds} device={args.device}")

    t0 = time.time()
    results = {}
    for P in args.pairs:
        results[str(P)] = run_P(P, refine_steps=args.refine, steps=args.steps,
                                seeds=args.seeds, device=args.device)
        json.dump(results, open(args.out, "w"), indent=2)

    # Overall: does the gap WIDEN with P? (the gold pattern)
    Ps = sorted(args.pairs)
    gaps = [results[str(P)]["gap"] for P in Ps]
    # need the transformer to have held somewhere for the gap to be meaningful
    xf_held_any = any(results[str(P)]["transformer_holds"] for P in Ps)
    # also require the single-read to have learned somewhere — otherwise a 0/0
    # gap is underpowered, not evidence of "no improvement"
    single_learned_any = any(results[str(P)]["single_read"] > 1.5 * results[str(P)]["chance"]
                             for P in Ps)
    widens = len(gaps) >= 2 and gaps[-1] > gaps[0] and all(gaps[i] <= gaps[i + 1] + 0.05
                                                           for i in range(len(gaps) - 1))
    positive_mean = sum(gaps) / len(gaps) if gaps else 0.0

    print(f"\n{'='*70}")
    print("gap (best-iterated - single-read) by P:")
    for P, g in zip(Ps, gaps):
        held = "XF-held" if results[str(P)]["transformer_holds"] else "XF-failed"
        print(f"  P={P:<4} gap {g:+.3f}   ({held})")
    if not xf_held_any or not single_learned_any:
        print("VERDICT: UNDERPOWERED — transformer never held and/or single-read never learned "
              "above chance; no valid read-depth comparison. Raise power.")
    elif widens and positive_mean > 0.05:
        print("VERDICT: GOLD — iterated read beats single-read AND the gap WIDENS with load.")
    elif positive_mean > 0.05:
        print("VERDICT: PARTIAL — iterated read helps, but the gain does not cleanly scale with P.")
    else:
        print("VERDICT: DEAD — iterating the read does not beat a single read.")
    print(f"(elapsed {time.time()-t0:.0f}s)")
    print(f"{'='*70}")

    lines = [
        "# POC Experiment 2 — Adaptive Recall Depth (the novel mechanism)",
        "",
        f"Single-read vs iterated-read, identical state size (frozen during iteration). "
        f"dk={pc.HEAD_DIM} (dk**2={pc.HEAD_DIM**2}), steps={args.steps}, seeds={args.seeds}.",
        "",
        "Gold pattern: single-read drops as P grows, iterated holds, gap WIDENS.",
        "",
        "| P | chance | single-read | best-iterated | gap | Transformer held? |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for P in Ps:
        r = results[str(P)]
        lines.append(f"| {P} | {r['chance']:.4f} | {r['single_read']:.3f} | "
                     f"{r['best_iterated']:.3f} | {r['gap']:+.3f} | "
                     f"{'yes' if r['transformer_holds'] else 'no'} |")
    open(args.markdown, "w").write("\n".join(lines))
    print(f"wrote {args.out} and {args.markdown}")


if __name__ == "__main__":
    main()
