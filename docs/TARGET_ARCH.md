# PhaseSSM architecture

PhaseSSM combines an oscillatory state-space sequence mixer with a feed-forward block. Each channel carries a complex recurrent state whose pole is parameterized by frequency, damping, and a learned time step:

```text
h[t] = a * h[t-1] + u[t]
a = exp(dt * (-damping + i * frequency))
y[t] = 2 * real(sum(w * h[t])) + D * u[t]
```

The implementation is in `phase_ssm/model.py`; recurrent inference is in `phase_ssm/recurrent.py`. The fixed-size sequence state is independent of input length during recurrent execution. Training intermediates and total process memory have different costs.

## Optional phase-binding memory

`--use-phase-memory` adds the branch implemented in `memory.py` and `linear_memory.py`. It binds key/value information using complex phase features and reads with squared phase coherence.

The quadratic reference, chunked training form, and recurrent inference form are compared by `tests/test_phase_ssm_linear_memory.py`. Floating-point agreement is numerical, not a claim of universal bitwise identity.

The `parallel` layout reads the same normalized input through the SSM and memory branches before the feed-forward block. The `sequential` layout keeps separate residual sublayers for ablation.

```bash
python -m phase_ssm.train --model phasessm \
  --use-phase-memory --mem-linear --mem-layout parallel \
  --out runs/phasessm-memory
```

## Research questions

The open questions are whether fixed-state recall remains useful under higher loads, whether the model reaches competitive language quality, and what memory and compute it needs to do so. The included MQAR and architecture probes address narrower tasks.

`phase_ssm/bakeoff.py` compares a phase-memory-enabled model with a transformer at similar parameter counts and equal token budgets. It does not make the architectures FLOP-equivalent. The recorded 131.7M scout and the synthetic mixer-efficiency rows are separate evidence, not a completed matched-quality comparison.

Start with `results/final_scorecard.md`, `results/ssm_130m_scout.md`, and the retained negative results before interpreting the broader architecture experiments.
