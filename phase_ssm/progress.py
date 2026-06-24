"""Print bake-off training progress from the JSON run logs."""
import json
import os
import sys

dirs = sys.argv[1:] or ["runs/text8_ssm", "runs/text8_xf"]
for d in dirs:
    p = os.path.join(d, "log.json")
    if not os.path.exists(p):
        print(f"{d} -> no log yet")
        continue
    j = json.load(open(p))
    h = j.get("history", [])
    best = j.get("best_val_bpc")
    head = "{} ({}, {:.2f}M params) evals={} best_val_bpc={}".format(
        d, j["model"], j["params"] / 1e6, len(h), f"{best:.4f}" if best else "running")
    print("=== " + head + " ===")
    for e in h[-8:]:
        print("  step {:6d}  train_bpc {:.4f}  val_bpc {:.4f}  {:.0f}k tok/s  {:.0f}s".format(
            e["step"], e["train_bpc"], e["val_bpc"], e["tok_per_s"] / 1e3, e["elapsed_s"]))
