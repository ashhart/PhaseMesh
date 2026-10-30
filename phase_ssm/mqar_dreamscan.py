"""MQAR gate for DreamScan — the go/no-go differentiator.

MQAR (multi-query associative recall) is the canonical task that breaks pure
SSMs: bind P key->value pairs, then probe with Q keys and require the right
values back. A pure diagonal SSM collapses on it; an attention model aces it.
TARGET_ARCH.md makes MQAR the go/no-go gate for whether the recall memory can
carry the recall load a pure mixer can't.

This probe runs TWO arms through identical MQAR:
  * pure-scan DreamScan (--no-use-memory)        -- expected to FAIL (pure SSM)
  * composition DreamScan (scan || phase memory)  -- expected to PASS

If pure-scan fails and the composition passes, the recall memory is doing its
job: attention-class retrieval at fixed state, the thing the scan alone lacks.
If the composition ALSO fails, the memory isn't carrying recall and the
architecture's recall half is broken -- the whole composition thesis is in
trouble. Either outcome is decisive and cheap (minutes, not hours).

Run: python -m phase_ssm.mqar_dreamscan --steps 4000
"""
from __future__ import annotations
import argparse, time
import torch
from .tasks import make_mqar, recall_accuracy
from .dreamscan_lm import DreamScanLM


def train_mqar(model, x, y, steps, lr=1e-3, device="cpu"):
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    for step in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0 or step == steps - 1:
            acc = recall_accuracy(model, x, y)
            print(f"    step {step:4d}: loss {loss.item():.3f}  recall_acc {acc:.3f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=16, help="key->value pairs to bind")
    ap.add_argument("--queries", type=int, default=8, help="probe keys")
    ap.add_argument("--n-keys", type=int, default=32)
    ap.add_argument("--n-values", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available()
           else "cpu") if args.device == "auto" else args.device

    x, y, vocab = make_mqar(args.batch, args.pairs, args.queries, args.n_keys, args.n_values, device=dev)
    L = x.shape[1]
    # chance accuracy on the query positions: 1/n_values
    chance = 1.0 / args.n_values
    print(f"MQAR gate | pairs={args.pairs} queries={args.queries} L={L} vocab={vocab} "
          f"dev={dev} | chance {chance:.3f}", flush=True)
    print(f"  task: bind {args.pairs} (k,v) pairs, recall {args.queries} of them. "
          f"A model that can't do exact in-context recall collapses here.", flush=True)

    arms = {
        "pure-scan (no mem)": (False, "expected to FAIL -- pure SSM, no recall mechanism"),
        "composition (scan+mem)": (True, "expected to PASS -- phase memory is attention-class recall"),
    }
    results = {}
    for name, (use_mem, expectation) in arms.items():
        print(f"\n  === {name} === ({expectation})", flush=True)
        seed_accs = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            m = DreamScanLM(vocab=vocab, d_model=args.d_model, n_layer=1,
                            scan_heads=8, scan_head_dim=args.d_model // 8, scan_n_h=4,
                            mem_heads=4, mem_head_dim=16, iters=args.iters,
                            use_memory=use_mem).to(dev)
            t0 = time.time()
            train_mqar(m, x, y, args.steps, device=dev)
            acc = recall_accuracy(m, x, y)
            seed_accs.append(acc)
            print(f"    seed {s}: final recall_acc {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        import numpy as np
        mean, std = float(np.mean(seed_accs)), float(np.std(seed_accs))
        results[name] = (mean, std, seed_accs)
        print(f"  {name}: {mean:.3f} ± {std:.3f}", flush=True)

    print(f"\nMQAR GATE VERDICT (chance {chance:.3f}):", flush=True)
    pm = results["pure-scan (no mem)"][0]
    cm = results["composition (scan+mem)"][0]
    if pm < 0.3 and cm > 0.7:
        print(f"  PASS — pure-scan fails ({pm:.3f}), composition passes ({cm:.3f}). "
              f"Recall memory carries the load the scan alone can't.", flush=True)
    elif cm < 0.3:
        print(f"  FAIL — composition also collapses ({cm:.3f}). The recall memory is NOT "
              f"doing its job; the composition thesis is in trouble.", flush=True)
    elif pm > 0.7:
        print(f"  INCONCLUSIVE — pure-scan ALSO passes ({pm:.3f}). MQAR at pairs={args.pairs} "
              f"is too easy; re-run with more pairs to separate.", flush=True)
    else:
        print(f"  PARTIAL — composition {cm:.3f} > pure-scan {pm:.3f} but neither clearly "
              f"passes/fails. Inconclusive at this difficulty.", flush=True)
    print("MQAR GATE DONE", flush=True)


if __name__ == "__main__":
    main()
