# Frontier Comparison

Reproducible wrapper around the phase-field mesh benchmark results.

The harness does three local measurements:

- context-retention gradient with `--pin 0.25`
- adaptive predictive-compute spread between an easy arithmetic prompt and a harder contradiction/context prompt
- full precision and Q8 topology/state footprint

It also emits conservative estimates for FLOPs/query and RAM-vs-context curves. Reference Llama-3-8B INT4 values are comparison anchors, not locally measured model runs.

## Run

```bash
python3 benchmark/frontier_comparison/run_comparison.py \
  --out benchmark/frontier_comparison/out
```

Then generate the one-page PDF:

```bash
python3 benchmark/frontier_comparison/make_report.py \
  --metrics benchmark/frontier_comparison/out/frontier_metrics.json \
  --out benchmark/frontier_comparison/adaptive_phase_field_reasoning.pdf
```

Open `analysis.ipynb` for a notebook view of the same plots.

Build GSM8K and LongBench task JSONL for the broader harness:

```bash
python3 benchmark/frontier_comparison/build_external_tasks.py \
  --gsm8k-count 8 \
  --longbench-count 4 \
  --out benchmark/frontier_comparison/out/external_tasks.jsonl

python3 -m bench.frontier_compare \
  --input-jsonl benchmark/frontier_comparison/out/external_tasks.jsonl \
  --pin 0.25 \
  --out runs/frontier-compare-external
```

## Interpreting The Numbers

`prediction_accuracy_proxy` is `1 - mean_prediction_error` from the mesh predictor. It is useful for testing whether adaptive compute scales with difficulty, but it is not the same thing as benchmark answer accuracy on GSM8K, LongBench, HumanEval, or MMLU.

`estimated_flops_per_query` is a transparent field-update estimate:

```text
steps_used * width * height * flops_per_cell_step
```

The default `flops_per_cell_step` is intentionally configurable from the CLI.
