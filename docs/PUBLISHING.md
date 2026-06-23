# Publishing Plan

Publish this as a clean open research harness, not as a model checkpoint.

## Recommended Repository Shape

Include:

- `phase_mesh/`: runtime, field, memory, verifier, service, CLI
- `phase_mesh/model.py` and `phase_mesh/trainer.py`: experimental self-supervised model layer
- `bench/`: benchmark harnesses and FLOP/RSS counters
- `benchmark/`: report-generation scripts and small derived outputs
- `tests/`: unit tests
- `docs/`: creator and publishing notes
- `artifacts/frontier-honest/`: corrected audit snapshot
- `README.md`, `pyproject.toml`, `.gitignore`

Do not include:

- `LLM-V2/`: separate experiment tree, 1.3 GB with checkpoints
- `runs/`: local run outputs, copied prompts, temporary artifacts
- `benchmark/frontier_comparison/out/`: generated external-task outputs that may contain copied benchmark text
- `.pytest_cache/`, `__pycache__/`, `.DS_Store`, virtualenvs
- any Hugging Face or API tokens

## Defensible Public Claim

Use this:

> A compact phase-field substrate showing flat synthetic context-gradient retention, a 63.6 KB q8 topology, verifier-guided control, and an auditable benchmark harness. Arithmetic rows are scored against decoded mesh output, not prompt truth.

Avoid these:

- "beats transformers"
- "new scaling law"
- "replacement for Qwen/Llama"
- "general reasoning model"
- "10Kx FLOP drop"
- "100% pass rate"

## First Release Checklist

1. Choose a license before public release. MIT is simple; Apache-2.0 adds explicit patent language.
2. Initialize a fresh git repository from this cleaned tree.
3. Run:

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q phase_mesh bench benchmark tests
phase-mesh think "check 17 * 19 = 323" --expect 323 --verifier-control --pin 0.25
```

4. Confirm credential hygiene with your normal local scanner before pushing.

5. Confirm large folders are excluded:

```bash
du -sh .
find . -maxdepth 2 -type f -size +5M -print
```

6. Publish the repo.
7. Open the README from a fresh clone and run the Quick Start exactly as written.

## GitHub Commands

After choosing the repo name and license:

```bash
git init
git add README.md pyproject.toml LICENSE .gitignore phase_mesh bench benchmark tests docs artifacts scripts examples .github
git status --short
git commit -m "Initial phase-field cognitive mesh release"
gh repo create PhaseMesh --public --source . --remote origin --push
```

If `gh` is not authenticated, create the GitHub repo in the browser, then:

```bash
git remote add origin https://github.com/ashhart/PhaseMesh.git
git branch -M main
git push -u origin main
```

## Access Model

People should be able to:

1. clone the repo,
2. install it with `pip install -e .`,
3. run `phase-mesh think`,
4. save their own `.q8.npz` topology,
5. run `phase-mesh bench`,
6. publish their own topology bundle separately.
