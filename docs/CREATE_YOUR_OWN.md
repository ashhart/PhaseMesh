# Create Your Own Phase Mesh

This project is a small research harness, not a pretrained language model. The point is to let you create a local phase-field substrate, inject your own text as wave packets, let the field settle, optionally apply verifier feedback, and save the resulting compact topology.

## 1. Install

```bash
git clone https://github.com/ashhart/PhaseMesh.git
cd PhaseMesh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e '.[bench]'
```

## 2. Run A First Mesh

```bash
phase-mesh think \
  "check whether 17 * 19 = 323 and stabilize the answer" \
  --expect 323 \
  --verifier-control \
  --pin 0.25 \
  --out runs/my-first-mesh
```

This writes:

- `runs/my-first-mesh/think.json`: decoded route, verifier result, resonance metrics, adaptive steps
- `runs/my-first-mesh/think.npz`: full precision state
- `runs/my-first-mesh/think.q8.npz`: compact quantized topology/state
- `runs/my-first-mesh/think.png`: phase-field image

## 3. Learn From Your Own Examples

Use `learn` when you have an expected answer or validation target:

```bash
phase-mesh learn "17 * 19" --expect 323 --rounds 4 --pin 0.25 --out runs/math-17x19
phase-mesh learn '{"ok": true}' --expect '{"ok": true}' --rounds 3 --out runs/json-ok
phase-mesh learn "def add(a, b): return a + b" --rounds 3 --out runs/code-add
```

Verifier feedback reinforces successful basins and destabilizes failed ones. This is local online adaptation, not backpropagation or fine-tuning.

## 4. Route Local Workflows

```bash
echo "debug this python function: def add(a,b): return a+b" | xargs phase-mesh route
```

Or call directly:

```bash
phase-mesh route "debug this python function: def add(a,b): return a+b"
```

The route output is JSON with a route name, suggested tool, phase signature, confidence, resonance status, and verifier result.

## 5. Run The Service

```bash
phase-mesh serve --host 127.0.0.1 --port 8765 --pin 0.25
```

Then:

```bash
curl -s "http://127.0.0.1:8765/think?text=check%2017%20*%2019%20=%20323&max_budget=120&temperature=0.2" \
  | python3 -m json.tool
```

The service persists state by default under `runs/service-state/`.

## 6. Benchmark Your Mesh

```bash
phase-mesh bench --trials 50 --facts 10 --math-count 50 --pin 0.25 --out runs/bench-local
```

For the transformer comparison harness:

```bash
python3 -m bench.frontier_compare \
  --pin 0.25 \
  --size 128 \
  --steps 180 \
  --math-count 20 \
  --context-tokens 512 2048 8192 \
  --out runs/frontier-compare-local
```

To add a Hugging Face baseline on a CUDA machine:

```bash
pip install -e '.[frontier]'
python3 -m bench.frontier_compare \
  --baseline hf \
  --baseline-model Qwen/Qwen2.5-7B-Instruct \
  --baseline-quant int4 \
  --pin 0.25 \
  --size 128 \
  --steps 180 \
  --out runs/frontier-compare-qwen-local
```

## 7. Share A Topology

Share the compact `.q8.npz` file plus the JSON summary that produced it. Avoid publishing private prompts, API keys, or logs from personal notes.

Suggested share bundle:

```text
my-topology/
  README.md
  topology.q8.npz
  summary.json
  queries.sample.jsonl
```

Include:

- mesh size and backend
- pin strength and residual carry
- task list or sample prompts
- pass/fail criteria
- whether results are synthetic, private, or external benchmark data

## Boundaries

This harness measures phase dynamics, decoded mesh output, verifier-guided control, context-gradient retention, and routing behavior. It does not generate fluent language like a transformer and it should not be described as a drop-in LLM replacement.
