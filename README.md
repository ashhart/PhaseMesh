![PhaseMesh banner with luminous oscillating wavefronts](assets/phasemesh-banner.png)

# PhaseMesh

PhaseMesh is an experimental codebase for phase-based sequence processing and memory. Its trainable language-model component, **PhaseSSM**, uses damped complex oscillators to carry information through a recurrent state. An optional phase-binding memory adds content-based retrieval.

This source release is a starting point for small-model architecture experiments: inspect the implementation, run the tests, reproduce a training run, and compare against the included transformer.

## What is here

- `phase_ssm/`: PhaseSSM, phase-binding memory, recurrent inference, a transformer baseline, training code, and architecture experiments.
- `phase_mesh/`: the separate phase-field runtime, role-binding retrieval, verifier-guided demos, and model experiments.
- `tests/`: numerical equivalence and behavior checks.
- `bench/` and `benchmark/`: task and efficiency comparisons.
- `results/`: retained small research reports and measurements, including negative results.
- `docs/`: architecture and usage notes.

## Recorded results and limits

The [131.7M-parameter scout](results/ssm_130m_scout.md) completed 5,000 steps on text8, reaching a best validation score of 1.2568 bits per byte. This is a historical recorded result, not a new reproduction from this source release. The trained checkpoint is not included.

The [scorecard](results/final_scorecard.md) records long-context efficiency and role-binding experiments. Its speed and memory comparisons are mixer-level synthetic benchmarks, not quality-matched full language-model comparisons.

The matched 130M transformer quality comparison remains open. The small scout is not an instruction-tuned assistant. Later recall, depth, and state-tracking experiments are separate research tracks; their results do not establish general language-model superiority.

## Install and test

Use a Python 3.12 or newer virtual environment. PyTorch is included by the `model` extra; Triton is installed only on Linux. CPU reference paths are available for correctness checks.

```bash
git clone https://github.com/ashhart/PhaseMesh.git
cd PhaseMesh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model,dev]'
python -m pytest -q
```

Check the linear phase-memory derivation against its quadratic reference:

```bash
python -m phase_ssm.linear_memory
```

## Train a small model

The trainer downloads text8 on first use. Both architectures use the same byte-level data loader and training loop; the example below trains a small PhaseSSM on CPU. Adjust model size and token budget for a substantive experiment.

```bash
python -m phase_ssm.train \
  --model phasessm --out runs/ssm-small \
  --device cpu --dtype float32 \
  --d-model 64 --n-layers 2 --state-dim 16 \
  --seq 64 --batch 4 --steps 100 \
  --eval-interval 50 --eval-iters 2 --warmup 10

python -m phase_ssm.chat \
  --checkpoint runs/ssm-small --max-tokens 80 \
  "the "
```

This short run checks the pipeline; it does not reproduce the 130M scout. Use the FFT backend for quality training. The `fixed_triton` and `skip` training backends are diagnostic and can degrade quality.

## Compare with a transformer

```bash
python -m phase_ssm.bakeoff --help
python -m phase_ssm.bakeoff --out runs/bakeoff
```

Review the configuration before starting a full run: the defaults are a substantial training job. The comparison uses identical data splits and token budgets, checks parameter-count tolerance, and supports separate peak learning rates for the two architectures. Equal tokens and similar parameter counts do not guarantee equal FLOPs or wall time; report those costs separately. The bake-off enables the optional phase-memory branch, so it is a different configuration from a plain SSM scout.

A useful first collaboration is to freeze the configurations and evaluation criteria, reproduce a small run, then report held-out bits per byte, parameter counts, training tokens, elapsed time, and peak memory for each model.

## Phase-field demos

```bash
python -m phase_mesh run "check 17 * 19 = 323"
```

See [the phase-field guide](docs/CREATE_YOUR_OWN.md) for the other runtime. Verifier feedback and specialist retrieval are distinct from learned general reasoning.

## Contributing

Keep numerical changes covered by reference comparisons, retain negative results, and state which model, data, and hardware produced a result. See [the contribution guide](.github/CONTRIBUTING.md).

## Licence and data

Original code is MIT-licensed. See [third-party notices](THIRD_PARTY.md) for the vendored baseline and external data. Downloaded data, checkpoints, generated runs, local environments, temporary files, and historical development Git metadata are excluded from this fresh source release.
