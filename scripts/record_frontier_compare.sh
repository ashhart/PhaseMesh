#!/usr/bin/env bash
set -euo pipefail

interval="${1:-0.5}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
summary="$root/artifacts/frontier-honest/summary.md"
results_json="$root/artifacts/frontier-honest/results.json"
topology="$root/artifacts/frontier-honest/phase_mesh_topology.q8.npz"
queries_jsonl="$root/artifacts/frontier-honest/queries.jsonl"

while true; do
  clear
  date
  echo
  cat "$summary"
  echo
  echo "Artifacts"
  ls -lh "$topology" "$results_json" "$queries_jsonl"
  echo
  echo "Recording helper: press Ctrl-C to stop."
  sleep "$interval"
done
