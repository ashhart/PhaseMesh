# Contributing

Thanks for helping make PhaseMesh easier to inspect, reproduce, and extend.

## How To Extend

- `phase_mesh/field.py`: Laplacian kernels, phase pinning, residual predictor, adaptive damping.
- `phase_mesh/runtime.py`: `think()` loop, resonance detection, budget control, verifier gating.
- `phase_mesh/service.py`: FastAPI endpoints, streaming, state persistence, job queue behavior.
- `bench/`: FLOP counters, comparison harnesses, metrics, and reproducible task suites.
- `examples/`: small, runnable scripts that teach one concept at a time.

## Ground Rules

- Keep claims tied to measured artifacts.
- Do not describe this as a drop-in LLM replacement.
- Keep local/private prompts out of committed run logs.
- Prefer small JSON/Markdown audit snapshots over large generated folders.
- Add or update tests for behavior changes in the runtime, field dynamics, verifier, memory, or benchmark counters.

## Local Checks

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q phase_mesh bench benchmark tests examples
python3 -m phase_mesh think "check 17 * 19 = 323" --expect 323 --verifier-control --pin 0.25
```

## Benchmark Notes

Transformer comparison rows and phase-mesh rows do not measure identical capabilities. Mesh arithmetic rows must be scored against decoded mesh output; prompt verifier results are diagnostics/control signals, not answer grades. Keep that boundary visible in any new reports.
