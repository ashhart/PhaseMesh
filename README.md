# Phase-Mesh: Compact Phase-Field Cognitive Substrate

A 128x128 damped phase field with predictive residual dynamics, adaptive compute, and verifier-guided basin reinforcement. The corrected `frontier-honest` artifact shows flat synthetic context-gradient retention, 23/23 row completion, a 63.6 KB q8 topology, and an auditable comparison harness.

This is not a trained LLM and it does not replace Qwen. It is a runnable research harness for a different inference substrate: text is injected as phase-modulated wave packets, a 2D field evolves under damped wave dynamics, resonance is detected from coherence and phase gradients, and a persistent potential landscape is updated through verifier feedback and consolidation.

The parts implemented here are intentionally small and inspectable:

- deterministic text-to-wave encoding
- 2D phase field with neighbor-update dynamics
- resonance capture and sector decoding
- persistent topological landscape updates
- verifier feedback for arithmetic, JSON, and code-shape checks
- consolidation cycles that smooth and reinforce stable basins
- CLI, optional FastAPI service, image output, and tests

## Quick Start

Install from a clone:

```bash
git clone https://github.com/ashhart/PhaseMesh.git
cd PhaseMesh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the mesh:

```bash
python3 -m phase_mesh run "check 17 * 19 = 323"
python3 -m phase_mesh think "check 17 * 19 = 323" --expect 323 --verifier-control --pin 0.25
python3 -m phase_mesh bench --trials 50 --out runs/my-bench
```

Run a custom prompt:

```bash
python3 -m phase_mesh run "calculate whether 17 * 19 = 323 and route the answer"
```

Run predictive adaptive compute:

```bash
python3 -m phase_mesh think "check 17 * 19 = 323" --max-budget 120 --temperature 0.2
```

Train the experimental basin-to-token model layer:

```bash
pip install -e '.[model]'
python3 scripts/prepare_corpus.py path/to/data.jsonl \
  --out runs/corpus.txt \
  --max-lines 100000 \
  --max-tokens 128

python3 -m phase_mesh model-train runs/corpus.txt \
  --out runs/phase-model \
  --max-steps 200000 \
  --steps-per-chunk 30 \
  --batch-size 32 \
  --context-tokens 8 \
  --windows-per-chunk 8 \
  --window-stride 1 \
  --lr 2e-4 \
  --freeze-omega \
  --consolidate-interval 5000

python3 -m phase_mesh model-eval path/to/heldout.txt \
  --model-dir runs/phase-model/final \
  --chunks 1000 \
  --context-tokens 8 \
  --windows-per-chunk 8

python3 -m phase_mesh generate "write a python function" \
  --model-dir runs/phase-model/final \
  --max-len 32 \
  --temp 0.8 \
  --top-k 50 \
  --top-p 0.95
```

Run with salient-token phase pinning:

```bash
python3 -m phase_mesh run "ctx_000 ctx_001 ctx_002 query ctx_000" --pin 0.25
```

Return a compact tool dispatch decision:

```bash
python3 -m phase_mesh route "debug this python function: def add(a,b): return a+b"
```

Run a few learning rounds with verifier feedback:

```bash
python3 -m phase_mesh learn "17 * 19" --expect 323 --rounds 4 --out runs/learn
```

Run the benchmark suite:

```bash
python3 -m phase_mesh bench --trials 50 --facts 10 --math-count 50 --out runs/bench
python3 -m phase_mesh bench --pin 0.25 --size 64 --out runs/bench-pinned
```

Start the local HTTP service:

```bash
python3 -m phase_mesh serve --host 127.0.0.1 --port 8765
```

Then call:

```bash
curl -s http://127.0.0.1:8765/resonate \
  -H 'content-type: application/json' \
  -d '{"text":"check 12 * 12 = 144","steps":220}' | python3 -m json.tool
```

For longer calls, queue a background job:

```bash
curl -s "http://127.0.0.1:8765/think?text=check%2017%20*%2019%20=%20323&max_budget=120&temperature=0.2" \
  | python3 -m json.tool

curl -s http://127.0.0.1:8765/jobs/resonate \
  -H 'content-type: application/json' \
  -d '{"text":"check 17 * 19 = 323","steps":220,"learn":true}' | python3 -m json.tool
```

## How It Maps To The Spec

`phase_mesh.field.PhaseFieldMesh` is the core engine. It keeps:

- `theta`: wrapped phase field in radians
- `velocity`: second-order wave velocity
- `omega`: seeded natural-frequency landscape
- `landscape`: persistent learned topology

`phase_mesh.encoding.TextPhaseEncoder` maps local text into wave packets using stable hashes. It is intentionally cheap and deterministic so experiments are repeatable.

`phase_mesh.verifier.VerifierRouter` is the feedback bottleneck. It can validate simple arithmetic, parse JSON, compile Python snippets, and otherwise fall back to coherence-based checks.

`phase_mesh.memory.TopologicalMemory` stores learned resonance traces and recalls them by basin similarity plus a query-key boundary anchor. This is intentionally small and inspectable.

`phase_mesh.runtime.CognitiveMeshRuntime` wires everything together for one-shot resonance and iterative learning.

`CognitiveMeshRuntime.think()` uses predictive coding for test-time compute scaling: the field forecasts its next phase, observes circular phase error after the real step, updates a residual predictor trace, adapts damping, and stops when both resonance and prediction error are low.

Phase pinning is optional. `--pin 0.25` writes decaying phase anchors for salient packets and blends a small residual of the previous phase into each step. It preserves important wave structure without storing token vectors or a KV cache.

## Experimental Model Layer

`phase_mesh.model.PhaseModel` turns the substrate into a small self-supervised training loop:

- stream text into the field as wave packets
- predict the next phase and update the residual predictor from circular phase error
- extract a stable basin feature vector from the settled field
- reinforce the topology around that basin without answer labels
- optionally train a tiny PyTorch decoder head from basin features to next-token logits
- generate by repeatedly settling the field, sampling from the decoder head, and injecting the sampled token back as residual phase

The field/topology update is gradient-free. The decoder head is a conventional optional PyTorch MLP, so generation quality depends on actual training data and should not be described as a pretrained language model.

The default Laplacian backend is `auto`: SciPy convolution when available, NumPy `roll` otherwise. You can force either path:

```bash
python3 -m phase_mesh run "17 * 19 = 323" --backend scipy
python3 -m phase_mesh run "17 * 19 = 323" --backend numpy
```

Runs saved with `--out` now write both full precision state and quantized state:

- `*.npz`: full precision field, velocity, natural landscape, learned landscape
- `*.q8.npz`: int8 phase/landscape plus fp16 velocity/omega

## Benchmarks

The `bench/` package writes JSON artifacts into `runs/bench/`.

The current public artifact is intentionally conservative. It scores arithmetic against decoded mesh output, not prompt truth, and counts local verifier work when the verifier is used for control or diagnostics.

| Metric | Phase-Mesh | What It Measures |
| --- | ---: | --- |
| Completed rows | 23/23 | No runtime errors or OOM on the synthetic suite |
| Decoded arithmetic pass | 0/20 | Current decoder emits route/signature fields, not numeric answers |
| Context rows passed | 3/3 | Phase-gradient retention below 0.05 |
| Mean context gradient | 0.0075 | Flat retention across 512, 2048, and 8192 synthetic tokens |
| Median latency | 200.2 ms | Field resonance, decode, and verifier diagnostics |
| Peak RSS | 61.6 MB | Field, verifier, and process buffer |
| FLOPs/query median | 21.0M + 672 verifier ops | Stencil kernel ops plus local verifier counter |
| Phase topology | 63.6 KB | q8 compact topology artifact |

Audit artifacts:

- `artifacts/frontier-honest/summary.md`
- `artifacts/frontier-honest/results.json`
- `artifacts/frontier-honest/queries.jsonl`
- `artifacts/frontier-honest/phase_mesh_topology.q8.npz`

FLOPs use deterministic harness counters. Mesh FLOPs include neighbor-update ops x steps_used plus verifier counters where local verifier work is used. The prompt verifier is logged as a control/diagnostic signal; pass rate grades decoded mesh output.

To create your own mesh/topology, see `docs/CREATE_YOUR_OWN.md`. For the clean GitHub release checklist, see `docs/PUBLISHING.md`.

```bash
python3 -m bench.test_stability --trials 50
python3 -m bench.test_correction
python3 -m bench.test_context
python3 -m bench.test_context --pin 0.25
python3 -m bench.test_memory --facts 10
python3 -m bench.test_adaptive
python3 -m bench.train_math --count 50
python3 -m bench.compare
python3 -m bench.frontier_compare --pin 0.25 --out runs/frontier-compare
python3 benchmark/frontier_comparison/run_comparison.py --out benchmark/frontier_comparison/out
python3 -m bench --trials 50 --facts 10 --math-count 50
```

Current experiments:

- phase-lock stability under randomized packet phase offsets
- verifier correction on wrong arithmetic claims
- context-retention gradient after long prompt injection
- topology memory recall after consolidation
- predictive adaptive compute on easy vs hard prompts
- generated arithmetic learning rounds
- full precision vs q8 disk footprint
- report-ready frontier comparison plots and PDF source

## Frontier Comparison Harness

`bench.frontier_compare` is the reproducible side-by-side harness for the bigger claim: phase topology vs a local transformer baseline under the same task list. By default it runs mesh-only and writes only measured values:

```bash
python3 -m bench.frontier_compare \
  --pin 0.25 \
  --size 64 \
  --steps 180 \
  --math-count 20 \
  --context-tokens 512 2048 8192 \
  --out runs/frontier-compare
```

Or use the wrapper:

```bash
bench/run_comparison.sh
```

To run a real local Hugging Face baseline, install the optional dependencies and request the baseline explicitly:

```bash
pip install -e '.[frontier]'
python3 -m bench.frontier_compare \
  --baseline hf \
  --baseline-model meta-llama/Meta-Llama-3-8B-Instruct \
  --baseline-load-in-4bit \
  --pin 0.25 \
  --out runs/frontier-compare-llama
```

The harness writes:

- `queries.jsonl`: one audit row per model per task
- `results.json`: aggregate accuracy, latency, RSS, estimated FLOPs, and context rows
- `summary.md`: one-page technical summary
- `phase_mesh_topology.q8.npz`: compact quantized mesh state

If the transformer model or dependencies are unavailable, the baseline is recorded as skipped instead of being approximated. Mesh arithmetic rows are scored against decoded mesh output; context rows measure phase-gradient retention for the mesh and expected-token text recall for transformer baselines.

For the compact report bundle, run:

```bash
python3 benchmark/frontier_comparison/run_comparison.py --out benchmark/frontier_comparison/out
python3 benchmark/frontier_comparison/make_report.py \
  --metrics benchmark/frontier_comparison/out/frontier_metrics.json \
  --out benchmark/frontier_comparison/adaptive_phase_field_reasoning.pdf
```

To build GSM8K and LongBench JSONL tasks for `bench.frontier_compare`, run:

```bash
python3 benchmark/frontier_comparison/build_external_tasks.py \
  --gsm8k-count 8 \
  --longbench-count 4 \
  --out benchmark/frontier_comparison/out/external_tasks.jsonl
```

## Realistic Expectation

The useful thing here is not language generation. The useful thing is a small, always-on experimental substrate for routing, verification, personal-memory topology, and phase-based adaptation. A transformer can still sit beside it for fluent text; this mesh can decide, verify, route, and accumulate local structure without a KV cache or a fine-tune.
