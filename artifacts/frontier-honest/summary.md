# Frontier Comparison Summary

This file is generated from measured local runs. Skipped baselines are not treated as proof.

## Lead Table

| Metric | Baseline | Phase-Mesh | Ratio |
| --- | ---: | ---: | ---: |
| Disk Size | not_measured | 63.6 KB | not_measured |
| Peak RAM | not_measured | 61.6 MB | not_measured |
| Context Scaling | not_requested | flat gradient 0.0075 | not_applicable |
| Median Latency | not_measured | 200.2 ms | not_measured |
| Task Pass Rate | not_measured | 13.0% | not_measured |
| FLOPs / Query | not_measured | 2.1e+07 | not_measured |

## Configuration

- Mesh size: 128x128
- Mesh pin strength: 0.25
- Mesh max budget: 180
- Baseline: none (not_requested)
- Query log: `runs/frontier-honest/queries.jsonl`

## Aggregates

### phase_mesh

- Pass rate: 0.130 over 23 scored records
- Mean latency: 2.140908 s
- Peak RSS mean: 63916121 bytes
- FLOPs mean: 20972104
- Mean adaptive steps: 40.00

## Audit Notes

- Mesh context rows measure phase-gradient retention; HF context rows measure text recall.
- HF FLOPs are parameter MAC counts over the generated sequence when the baseline loads.
- Mesh arithmetic rows are scored against decoded mesh output. The prompt verifier is logged as a diagnostic/control signal, not as the grade.
- Mesh FLOPs are deterministic kernel-operation counters plus verifier counters where verifier work is used.
- q8 state is quantized compact state, not mathematically lossless compression.
