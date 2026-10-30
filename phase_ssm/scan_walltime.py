"""Rung A cost leg: measured training-step wall-clock vs sequence length.

Settling trains through L SEQUENTIAL cell applications (BPTT depth L, activation
memory O(L) copies of the sequence); the non-abelian scan trains through
ceil(log2 L) rounds of batched matmuls (BPTT depth O(log L)). At toy L both are
launch-overhead-dominated; this measures where the curves separate, and records
settling's OOM wall as an honest datapoint (O(L) BPTT memory is part of the
sequential cost, not an implementation accident).

Run: python -m phase_ssm.scan_walltime --Ls 32 128 512 1024 2048 --json results/rungA_walltime.json
"""
from __future__ import annotations
import argparse, json, time
import torch

from .equilibrium import SettlingLM
from .nonabelian_scan import NonAbelianScanLM, pick_device


def step_time(model, B, L, vocab, dev, settling, reps=5, warmup=2):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, vocab, (B, L), device=dev)
    tgt = torch.randint(0, vocab, (B, L), device=dev)
    times = []
    for i in range(warmup + reps):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _, loss = model(ids, tgt, iters=L) if settling else model(ids, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.time() - t0)
    return sorted(times)[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", type=int, nargs="+", default=[32, 128, 512, 1024, 2048])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=150)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = pick_device(args.device)
    print(f"WALLTIME | fwd+bwd+step median (s) | B={args.batch} d={args.d} dev={dev}", flush=True)
    print(f"{'L':>6} | {'scan (log-depth)':>17} | {'settling (iters=L)':>19} | {'ratio':>7}", flush=True)
    rows = {}
    for L in args.Ls:
        torch.manual_seed(0)
        scan_t = step_time(NonAbelianScanLM(args.vocab, d=args.d), args.batch, L,
                           args.vocab, dev, settling=False)
        try:
            torch.manual_seed(0)
            sett_t = step_time(SettlingLM(args.vocab, d=args.d, iters=L), args.batch, L,
                               args.vocab, dev, settling=True)
            ratio = f"{sett_t / scan_t:>6.1f}x"
            sett_s = f"{sett_t:>19.3f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            sett_t, ratio, sett_s = None, "   OOM", f"{'OOM (BPTT depth L)':>19}"
        rows[L] = {"scan": scan_t, "settling": sett_t}
        print(f"{L:>6} | {scan_t:>17.3f} | {sett_s} | {ratio}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": dev, "batch": args.batch, "d": args.d, "rows": rows}, f, indent=2)
        print(f"wrote {args.json}", flush=True)
    print("WALLTIME DONE", flush=True)


if __name__ == "__main__":
    main()
