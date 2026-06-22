#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-runs/frontier-compare}"

python3 -m bench.frontier_compare \
  --out "$OUT" \
  --pin 0.25 \
  --size 64 \
  --steps 180 \
  --math-count 20 \
  --context-tokens 512 2048 8192 \
  "$@"
