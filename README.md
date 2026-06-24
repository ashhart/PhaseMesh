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

Train the compact PhaseMesh language model:

```bash
python3 -m phase_mesh lm-train runs/corpus.txt \
  --out runs/phase-lm \
  --order 4 \
  --phase-cells 2048 \
  --vocab-size 4096

python3 -m phase_mesh lm-generate "phase mesh" \
  --model-dir runs/phase-lm \
  --max-tokens 40 \
  --temperature 0.7

python3 -m phase_mesh lm-eval runs/heldout.txt --model-dir runs/phase-lm
```

This is the first actual trainable language-model core in the repo: it learns next-token continuations from a corpus, stores ordered context/next-token bindings in a complex phase memory, saves/loads the model, and generates token streams. It is small and early, but it is no longer just routing or a shell.

Train and talk to the PhaseSSM engine:

```bash
python3 -m phase_ssm.train \
  --model phasessm \
  --out runs/ssm-small \
  --steps 10000 \
  --seq 512 \
  --batch 32

python3 -m phase_ssm.chat \
  --checkpoint runs/ssm-small \
  --max-tokens 120 \
  --temperature 0.8 \
  "PhaseMesh is"
```

PhaseSSM is the trainable damped-oscillator backbone. Its strongest current result is long-context efficiency: fixed-state decode and flat prefill throughput through 1M tokens. See `results/final_scorecard.md`, `results/effbench_long_ctx.md`, and `results/decode_mixed_bench.md`.

Pour a small transformer into PhaseMesh:

```bash
pip install -e '.[distill]'

python3 -m phase_mesh lm-distill \
  --teacher-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompts examples/distill_prompts.txt \
  --out runs/phase-lm-distill-smollm2 \
  --samples-per-prompt 2 \
  --max-new-tokens 96 \
  --phase-cells 8192 \
  --vocab-size 8192 \
  --chat-template auto

python3 -m phase_mesh lm-generate "PhaseMesh is" \
  --model-dir runs/phase-lm-distill-smollm2/phase_lm \
  --max-tokens 80 \
  --temperature 0

python3 -m phase_mesh llm-shell "PhaseMesh is" \
  --state-dir runs/llm-shell \
  --language-model-dir runs/phase-lm-distill-smollm2/phase_lm
```

The distillation command writes `teacher_corpus.txt`, `teacher_samples.jsonl`, `summary.json`, and a saved `phase_lm/` model. This is behavior distillation through generated traces, not weight conversion.

Pour the actual checkpoint weights into a PhaseMesh artifact:

```bash
python3 -m phase_mesh weight-pour \
  --teacher-model Qwen/Qwen3-4B-Instruct-2507 \
  --out runs/qwen3-4b-weight-pour \
  --phase-cells 262144 \
  --token-cells 128
```

`weight-pour` walks every tensor in the Hugging Face checkpoint and folds every numeric value into `phase_weight_bank.npz`. It also writes `tensor_stats.jsonl`, `manifest.json`, copied teacher tokenizer/config metadata, and optional token-row phase signatures for `embed_tokens` and `lm_head`. For a quick plumbing check, add `--max-elements-per-tensor 100000`; omit it for the full pour.

Try the current local assistant shell:

```bash
python3 -m phase_mesh llm-shell "What is 12 * 60 * 5?"
python3 -m phase_mesh llm-shell "Find the bug in this Python function: def add(a, b): return a - b"
python3 -m phase_mesh llm-shell "Write Python code for a function named add that returns a + b."
```

The shell is currently the practical interface for coding/reasoning. It routes to verified narrow PhaseMesh organs and returns traced answers quickly. The raw checkpoint readout is available for experiments:

```bash
python3 -m phase_mesh weight-readout \
  "write a python function that adds two numbers" \
  --artifact-dir runs/qwen3-4b-weight-pour \
  --rank-only
```

`weight-readout` is not Qwen inference. It is an experimental nearest-phase reader over the poured checkpoint signatures, useful for probing the artifact but not yet the quality path for coding help.

Keep pouring Qwen behavior and soft next-token distributions into the PhaseMesh LM:

```bash
python3 -m phase_mesh lm-pour \
  --teacher-model Qwen/Qwen3-4B-Instruct-2507 \
  --prompts examples/distill_prompts.txt \
  --out runs/qwen3-4b-lm-pour \
  --rounds 25 \
  --samples-per-prompt 4 \
  --max-new-tokens 128 \
  --soft-top-k 16 \
  --phase-cells 32768 \
  --vocab-size 16384 \
  --chat-template auto \
  --device cuda
```

`lm-pour` resumes `phase_lm/` if it exists, appends teacher completions, injects top-k teacher probabilities as weighted PhaseMesh context bindings, snapshots each round, and writes `pour_events.jsonl` plus `summary.json`.

Build and use the prompt-conditioned PhaseMesh LM:

```bash
python3 scripts/gen_phasechat_coverage.py

python3 scripts/pour_phasechat_teacher.py \
  --teacher-model Qwen/Qwen3-4B-Instruct-2507 \
  --prompts examples/qwen_coding_reasoning_prompts_beast.txt \
  --out runs/qwen3-4b-phasechat-teacher-beast \
  --batch-size 4 \
  --max-new-tokens 96 \
  --device cuda

python3 scripts/gen_phasechat_augments.py

python3 scripts/pour_phasechat_teacher.py \
  --teacher-model Qwen/Qwen3-4B-Instruct-2507 \
  --prompts examples/qwen_coding_reasoning_augments_beast.jsonl \
  --out runs/qwen3-4b-phasechat-augment-beast \
  --batch-size 4 \
  --max-new-tokens 80 \
  --device cuda

mkdir -p runs/qwen3-4b-phasechat-teacher-beast-plus

cat runs/qwen3-4b-phasechat-teacher-beast/teacher_samples.jsonl \
  runs/qwen3-4b-phasechat-augment-beast/teacher_samples.jsonl \
  > runs/qwen3-4b-phasechat-teacher-beast-plus/teacher_samples.jsonl

python3 -m phase_mesh lm-chat-build \
  runs/qwen3-4b-phasechat-teacher-beast-plus/teacher_samples.jsonl \
  --out runs/qwen3-4b-phase-chat-beast-plus \
  --signature-cells 16384 \
  --retrieval-threshold 0.14 \
  --topic-coverage-threshold 0.66

python3 -m phase_mesh lm-chat \
  "Write a Python function that returns the maximum value in a list without using max()" \
  --model-dir runs/qwen3-4b-phase-chat-beast-plus \
  --no-fallback

python3 -m phase_mesh llm-shell \
  "Explain how to debug a Python TypeError." \
  --chat-model-dir runs/qwen3-4b-phase-chat-beast-plus

python3 scripts/eval_phasechat_coverage.py \
  examples/qwen_coding_reasoning_eval_beast.jsonl \
  --model-dir runs/qwen3-4b-phase-chat-beast-plus \
  --out runs/qwen3-4b-phase-chat-beast-plus/coverage_eval_beast_final.json
```

This is the current practical PhaseMesh LM path: Qwen traces are poured into prompt-phase signatures, then PhaseMesh retrieves the matching answer. The confidence gate can abstain quickly instead of bluffing when a prompt is outside the poured manifold.

Current measured local artifact:

| Artifact | Rows | Median hot-path latency | Eval |
| --- | ---: | ---: | ---: |
| `runs/qwen3-4b-phase-chat-beast-plus` | 180 Qwen traces | ~2.18 ms | 65/65 on `qwen_coding_reasoning_eval_beast.jsonl` |

Published artifact: [Vontra/PhaseMesh-Qwen3-4B](https://huggingface.co/Vontra/PhaseMesh-Qwen3-4B)

The comparable live Qwen3-4B CUDA run on PGX took ~4,966 ms median per 96-token answer after model load. PhaseChat is a fast specialist over poured behavior, not a raw checkpoint clone.

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

Build a small multi-domain PhaseMesh registry artifact:

```bash
python3 -m phase_mesh domain-fit --out runs/phase-mesh-registry
python3 -m phase_mesh domain-probe --registry-dir runs/phase-mesh-registry
python3 -m phase_mesh domain-report --registry-dir runs/phase-mesh-registry
python3 -m phase_mesh domain-solve "8 plus 9" --registry-dir runs/phase-mesh-registry
python3 -m phase_mesh domain-solve "def add(a, b): return a + b" --registry-dir runs/phase-mesh-registry
python3 -m phase_mesh domain-solve '{"ok": true}' --registry-dir runs/phase-mesh-registry
```

The current registry layer is a composable adapter scaffold inside PhaseMesh. Arithmetic uses a measured structured-basin readout. Code keeps exact AST facts and adds a persisted AST-factor readout with its own gate metrics. JSON keeps exact parser facts and adds a persisted structural-factor readout. Memory and tool routing are exact starter domains with probes, ready to be replaced or augmented with learned basin gates.

Run the composed six-organ shell:

```bash
python3 -m phase_mesh llm-shell "Remember: PhaseMesh project codename is Azure Compass."
python3 -m phase_mesh llm-shell "What codename did I ask you to remember for the PhaseMesh project?"
python3 -m phase_mesh llm-shell "What is 7 * 6?"
python3 -m phase_mesh llm-shell "Write Python code for a function named add_one that returns n + 1." --json
```

`llm-shell` is the current integrated ladder rung. It wires memory/retrieval, role binding, narrow reasoning adapters, surface generation, learning, and control into one traced runtime. It is still an executive shell over verified PhaseMesh organs, not an open-ended general LLM.

Build the reviewer-facing local demo artifact:

```bash
python3 -m phase_mesh lab-demo --out runs/lab-demo
open runs/lab-demo/index.html
```

The lab demo writes `summary.json`, `summary.md`, and a self-contained `index.html` with domain gates, solve traces, context-gradient rows, artifact size, and explicit limits.

Build the PhaseAccio retrieval proof:

```bash
python3 -m phase_mesh phase-accio \
  --context 1048576 \
  --needles 100 \
  --candidates 8 \
  --seeds 3 \
  --out runs/phase-accio
open runs/phase-accio/index.html
```

PhaseAccio is a candidate-conditioned long-context retrieval artifact. It hides synthetic key/value identifiers inside natural-note filler text, binds nearby key/value identifiers by token-window proximity, scores candidate values with a fixed-size phase-binding sketch, and reports pin-on, pin-off, scrambled, hash-map, Bloom-filter, and random-candidate rows in the same run. This is a structured retrieval demo, not a general LLM claim.

Current local PhaseAccio run (`1048576` context tokens, `100` hidden records, `8` candidates, `3` seeds):

| Variant | Accuracy | Rows | Mean Rank | Mean Margin | Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| pin_on | 1.000 | 300 | 1.00 | 0.0371 | 128.0 KB |
| pin_off | 0.087 | 300 | 4.64 | 0.0036 | 128.0 KB |
| scrambled | 0.127 | 300 | 4.60 | 0.0032 | 128.0 KB |
| hash_map | 1.000 | 300 | 1.00 | 1.0000 | 12.3 KB |
| bloom_filter | 1.000 | 300 | 1.00 | 0.7663 | 2.0 KB |
| random_candidate | 0.123 | 300 | 4.43 | 0.1099 | 0 B |

The hash-map and Bloom rows are intentionally included as classical associative-memory baselines. PhaseAccio is not claiming to beat exact maps on this synthetic task; it is the auditable phase-binding/control-collapse demo.

Run the PhaseMesh advantage probes:

```bash
python3 -m phase_mesh phase-advantage --out runs/phase-advantage
```

This is the sharper associative-memory discriminator. It stores synthetic multi-token associations, corrupts 20-40% of each query key, and asks whether the fixed-size phase memory can complete the pattern from partial overlap. Exact hash maps and exact Bloom filters are included as controls: they have no fuzzy path, so a single corrupted token breaks exact lookup. The same run includes a whole-key phase ablation, fixed-byte capacity rolloff, and a phase-synchrony segmentation sanity check.

The pass gate is intentionally narrow:

- at 30% key corruption, distributed phase completion must stay above 50%
- exact hash and exact Bloom controls must stay at 0% under corrupted keys
- the whole-key phase ablation must collapse under corruption
- phase-coupled segmentation must separate objects while no-coupling collapses

This still is not a general LLM claim. It is a pattern-completion and binding probe for the substrate.

Current local PhaseMesh advantage run (`800` associations, `12` tokens/key, `32` candidates, `300` trials, `32 KB` phase memory):

| Corruption | Phase Completion | Whole-Key Phase | Exact Hash | Exact Bloom |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 0.820 | 0.853 | 1.000 | 1.000 |
| 10% | 0.750 | 0.023 | 0.000 | 0.000 |
| 20% | 0.683 | 0.023 | 0.000 | 0.000 |
| 30% | 0.513 | 0.033 | 0.000 | 0.000 |
| 40% | 0.400 | 0.020 | 0.000 | 0.000 |

Phase-synchrony segmentation in the same artifact:

| Variant | Pair Accuracy | Within Similarity | Across Similarity |
| --- | ---: | ---: | ---: |
| coupled | 1.000 | 1.000 | 0.522 |
| no_coupling | 0.470 | -0.168 | -0.004 |

The fixed-byte capacity curve is included in `runs/phase-advantage/summary.md`. On clean exact-key lookups, Bloom remains exact at this byte budget; the capacity curve is therefore a load/rolloff diagnostic, not a Bloom-beating result.

Run the natural-document PhaseMesh advantage rung:

```bash
python3 -m phase_mesh phase-advantage-docs \
  --context 1048576 \
  --records 500 \
  --candidates 16 \
  --trials 240 \
  --corruption 0.0 0.1 0.2 0.3 0.4 0.5 \
  --out runs/phase-advantage-docs
open runs/phase-advantage-docs/index.html
```

This is the next discriminator after synthetic corrupted keys. It generates natural noisy paragraphs, stores semantic associations, corrupts/paraphrases queries by token drop, replacement, synonym substitution, and typo noise, then compares PhaseMesh against exact hash, exact Bloom, BM25, vector/FAISS-if-installed, n-gram hash, and random-reservoir baselines. It also writes a 1M-token context artifact and a live dashboard where the corruption slider shows exact-control collapse.

This rung is deliberately harder to hand-wave, but still narrow: it tests corrupted natural-document retrieval and control collapse, not open-ended reasoning.

Current local natural-document run (`1,048,576` context tokens, `500` records, `16` candidates, `240` trials, `64 KB` PhaseMesh memory):

| Corruption | PhaseMesh | Whole-Key | Hash | Bloom | BM25 | Vector/FAISS | N-gram | Reservoir |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.983 | 0.075 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.925 |
| 10% | 0.946 | 0.092 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.850 |
| 20% | 0.829 | 0.067 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.721 |
| 30% | 0.688 | 0.054 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.679 |
| 40% | 0.512 | 0.071 | 0.000 | 0.000 | 0.996 | 0.983 | 0.996 | 0.658 |
| 50% | 0.421 | 0.042 | 0.000 | 0.000 | 0.983 | 0.938 | 0.992 | 0.483 |

Memory in the same run: PhaseMesh `65,536` bytes; exact hash `91,154`; exact Bloom `224,000`; BM25 `307,887`; vector/FAISS `288,502`; n-gram hash `16,384,000`; random reservoir `256,000`. BM25, vector, and n-gram remain stronger on this generated task. The PhaseMesh result is the smaller fuzzy attractor path plus control collapse, not a universal retrieval win.

Run the hard role-binding benchmark:

```bash
python3 -m phase_mesh phase-binding-hard \
  --records 500 \
  --candidates 16 \
  --trials 240 \
  --context 1048576 \
  --phase-cells 32768 \
  --slots 8 \
  --out runs/phase-binding-hard
open runs/phase-binding-hard/index.html
```

This is the stricter relation-binding rung. Each lexical decoy contains the same surface words but swaps which object belongs to which clause. Bag-of-words retrieval sees the right words in the wrong relation. The role-bound PhaseMesh readout gets role features; bag-phase and whole-phase ablations deliberately remove that binding.

Current local hard role-binding run (`1,048,576` context tokens, `500` records, `16` candidates, `240` trials, `256 KB` role-phase memory, arbitrary corruption):

| Corruption | Role Phase | Bag Phase | Whole Phase | Hash | Bloom | BM25 | Vector | N-gram | Reservoir |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.996 | 0.479 | 0.050 | 1.000 | 1.000 | 0.500 | 0.508 | 0.517 | 0.075 |
| 10% | 0.988 | 0.487 | 0.083 | 0.004 | 0.004 | 0.450 | 0.483 | 0.475 | 0.096 |
| 20% | 0.996 | 0.479 | 0.067 | 0.008 | 0.008 | 0.483 | 0.475 | 0.475 | 0.092 |
| 30% | 0.979 | 0.438 | 0.054 | 0.000 | 0.000 | 0.492 | 0.446 | 0.487 | 0.083 |
| 40% | 0.912 | 0.450 | 0.046 | 0.000 | 0.000 | 0.508 | 0.450 | 0.512 | 0.042 |
| 50% | 0.892 | 0.400 | 0.071 | 0.000 | 0.000 | 0.421 | 0.354 | 0.450 | 0.083 |

Memory in the same run: role phase `262,144` bytes; exact hash `79,698`; exact Bloom `224,000`; BM25 `227,570`; vector `288,806`; n-gram hash `16,384,000`; random reservoir `256,000`. This is the strongest current PhaseMesh demo because the win is attributable to role binding: role phase beats its own bag/whole ablations and the lexical baselines on swapped-role decoys.

For the forced-answer 100% ECC demo, run:

```bash
python3 -m phase_mesh phase-binding-hard \
  --records 500 \
  --candidates 16 \
  --trials 240 \
  --context 1048576 \
  --phase-cells 32768 \
  --slots 8 \
  --corruption-mode ecc-signature \
  --ecc-readout \
  --out runs/phase-binding-hard-ecc-100
open runs/phase-binding-hard-ecc-100/index.html
```

ECC-signature mode repeats marked role fields before corruption, then applies the same arbitrary token drop/replacement/synonym/typo noise. This is forced-answer accuracy: no abstain path, no safe-decision scoring.

Current local ECC forced-answer run:

| Corruption | Role Phase | Bag Phase | Whole Phase | Hash | Bloom | BM25 | Vector | N-gram | Reservoir |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 1.000 | 0.496 | 0.092 | 1.000 | 1.000 | 0.500 | 0.479 | 0.508 | 0.079 |
| 10% | 1.000 | 0.471 | 0.062 | 0.000 | 0.000 | 0.479 | 0.500 | 0.475 | 0.083 |
| 20% | 1.000 | 0.396 | 0.046 | 0.000 | 0.000 | 0.492 | 0.492 | 0.487 | 0.075 |
| 30% | 1.000 | 0.358 | 0.062 | 0.000 | 0.000 | 0.521 | 0.500 | 0.508 | 0.075 |
| 40% | 1.000 | 0.375 | 0.071 | 0.000 | 0.000 | 0.521 | 0.483 | 0.508 | 0.058 |
| 50% | 0.996 | 0.388 | 0.079 | 0.000 | 0.000 | 0.525 | 0.467 | 0.529 | 0.054 |

For the 100% recoverable-signal demo, run:

```bash
python3 -m phase_mesh phase-binding-hard \
  --records 500 \
  --candidates 16 \
  --trials 240 \
  --context 1048576 \
  --phase-cells 32768 \
  --slots 8 \
  --corruption-mode recoverable-signature \
  --ecc-readout \
  --out runs/phase-binding-hard-100
open runs/phase-binding-hard-100/index.html
```

Recoverable-signature mode preserves the role-bearing actor/action/object/place tokens and injects distractor noise that breaks exact string lookup. It is not the arbitrary corruption curve above; it is the ECC-style ceiling test for whether PhaseMesh completes the pattern when the disambiguating signal is still present.

Current local recoverable-signature run:

| Corruption | Role Phase | Bag Phase | Whole Phase | Hash | Bloom | BM25 | Vector | N-gram | Reservoir |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 1.000 | 0.479 | 0.050 | 1.000 | 1.000 | 0.500 | 0.508 | 0.517 | 0.075 |
| 10% | 1.000 | 0.467 | 0.050 | 0.000 | 0.000 | 0.479 | 0.454 | 0.467 | 0.079 |
| 20% | 1.000 | 0.479 | 0.042 | 0.000 | 0.000 | 0.542 | 0.483 | 0.537 | 0.054 |
| 30% | 1.000 | 0.454 | 0.058 | 0.000 | 0.000 | 0.542 | 0.458 | 0.525 | 0.062 |
| 40% | 1.000 | 0.508 | 0.071 | 0.000 | 0.000 | 0.533 | 0.529 | 0.521 | 0.062 |
| 50% | 1.000 | 0.496 | 0.071 | 0.000 | 0.000 | 0.479 | 0.492 | 0.475 | 0.071 |

Run the learnable-core falsifier:

```bash
python3 -m phase_mesh learnable-core --out runs/learnable-core
```

This is the architecture bar PhaseAccio does not clear by itself: gradients must flow into the oscillator core, and the trained core must beat a frozen reservoir/readout baseline.

Current local learnable-core run:

| Model | Test Accuracy | Train Accuracy | Trainable Params | Core Delta |
| --- | ---: | ---: | ---: | ---: |
| learned_phase | 0.822 | 0.845 | 5412 | 24.2538 |
| frozen_phase | 0.495 | 0.500 | 4290 | 0.0000 |
| bag_mlp | 0.567 | 0.572 | 322 | 0.0000 |

The task is first-bit memory through same-symbol binary noise. This is still a toy memory probe, not an LLM benchmark, but it directly tests the missing ingredient: end-to-end credit assignment through a differentiable oscillator core.

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
