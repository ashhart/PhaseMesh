# PhaseMesh North Star

PhaseMesh is the app and research direction for this repository. The north star is an embodied computer-world model: a system that observes a real computer workspace, maintains compact state about what it has seen, predicts likely effects of bounded actions, assigns work to specialist agent roles, and learns from verified episodes.

PhaseMesh is not currently a general LLM. Today it is a small, inspectable phase-field and PhaseSSM research harness with narrow language, routing, memory, and verification surfaces. Any claim about broad assistant quality, general autonomy, or replacement-level language modeling needs to be earned by held-out evaluation, episode replay, and human-inspectable traces.

## End Goal

Build PhaseMesh into a local embodied computer-world model with a multi-agent control loop:

- `observer`: reads files, terminal output, browser/app state, screenshots, and git state into compact observations.
- `memory`: retrieves prior episodes, persistent user/project facts, and learned state-action patterns.
- `reasoner`: routes through PhaseMesh organs, PhaseSSM checkpoints, verifiers, and domain adapters.
- `planner`: proposes bounded next actions with predicted state changes.
- `critic`: rejects unsafe, ungrounded, irreversible, or overbroad actions before execution.
- `executor`: performs approved computer actions with scoped permissions and rollback-aware logging.
- `recorder`: writes observation, action, prediction, result, verifier outcome, and critique into an episode store.
- `trainer`: distills verified episodes into improved retrieval, prediction, routing, and PhaseSSM behavior.

The goal is not a chatbot wrapped around tools. The goal is a model of the computer environment itself: state, action, consequence, uncertainty, and role-coordinated control.

## Current Baseline

- `phase_mesh llm-shell` is the practical interface for narrow coding, arithmetic, JSON, memory, routing, and generation behavior. It composes verified organs and returns traces, but it is still a deterministic executive shell rather than an open-ended language model.
- PhaseSSM is the trainable damped-oscillator backbone. Its strongest current result is long-context efficiency and fixed-state decode behavior, not broad chat quality.
- PhaseMesh LM and chat artifacts can learn or pour behavior from corpora and teacher traces, but this is behavior distillation and associative phase memory, not native frontier-model inference.
- The current agent-loop direction can support read-only planning and episode capture. The next accepted rung is optional policy-gated read-only/test execution with prediction-vs-result scoring, not autonomous editing or arbitrary tool use.

## Next Accepted Agent-Loop Rung

The next accepted capability is deliberately narrow: PhaseMesh may optionally execute only policy-approved read-only inspection commands and tests, then score what happened against the prediction it made before execution.

Required behavior:

- Execution is opt-in for an episode and remains off when policy or user scope does not allow it.
- The policy admits only bounded inspection/test commands, such as workspace status, diffs, file listing/search, or project tests.
- The planner predicts the command class, expected return-code class, expected file-change set, expected output or test-result shape, and risk before execution.
- The executor records the policy decision, exact command, cwd, elapsed time, return code, output summary, and post-command dirty-worktree check.
- The recorder writes prediction-vs-result scoring for each executed, blocked, or failed command.
- Failed predictions are preserved as training signal and must not be hidden behind task-success wording.

Out of scope for this rung:

- File edits, arbitrary shell commands, browser/app control, git mutations, commits, pushes, network writes, production changes, emails, payments, or other external side effects.
- Claims that read-only/test grounding equals general autonomy, robust planning, or repository modification ability.

## Milestones

### 1. Make The Shell Auditable

Deliverable:

- Stable `llm-shell` manifest for organs, loaded models, state paths, and claim boundaries.
- Uniform trace schema across memory, routing, reasoning, generation, verifier, and learning steps.
- Golden prompts for arithmetic, code-shape checks, JSON validation, memory write/read, and refusal boundaries.

Acceptance checks:

- Running the golden prompt set produces deterministic routes and machine-readable traces.
- Every answer includes enough trace data to explain which organ produced it.
- The doc and CLI both state that this is not a general LLM.

### 2. Turn PhaseSSM Into A Quality Baseline

Deliverable:

- Reproducible PhaseSSM training runs against matched transformer baselines on the same byte/token data.
- Held-out bpc/perplexity tables alongside speed and memory tables.
- A talkable checkpoint path with saved config, tokenizer/data notes, and generation examples.

Acceptance checks:

- PhaseSSM quality is reported against a matched baseline, not only against itself.
- Long-context efficiency claims stay separate from language-quality claims.
- Any "better" claim includes dataset, model size, training budget, backend, and exact metric.

### 3. Add Episode Observation

Deliverable:

- Read-only workspace observer that records prompt, file sample, git state, terminal output, tool metadata, and optional visual state.
- Episode JSONL schema for observation, role outputs, proposed action, prediction, result, verifier outcome, and elapsed time.
- Redaction rules for secrets, personal data, and large/private files.

Acceptance checks:

- Episode records can be replayed without executing actions.
- Dirty worktrees, missing files, command failures, and permission limits are represented explicitly.
- Secret-shaped values are redacted before persistence.

### 4. Add Multi-Agent Role Coordination

Deliverable:

- First-class role outputs for observer, memory, reasoner, planner, critic, executor, recorder, and trainer.
- A coordinator that requires each role to write structured state, not only prose.
- Role-level scoring for usefulness, uncertainty, and verifier agreement.

Acceptance checks:

- Plans cite observations and memory hits they depend on.
- Critic output can block an executor action.
- Recorder persists both successful and blocked episodes for future training.

### 5. Predict Before Acting

Deliverable:

- Next-state prediction for each proposed action: expected files changed, commands run, outputs observed, tests affected, and risk level.
- Prediction-vs-result scoring after each episode, including policy decision, return-code class, file-change expectation, output/test-result class, and risk.
- Retrieval over similar past episodes to improve prediction and planning.

Acceptance checks:

- At least one benchmark suite measures prediction accuracy before execution quality.
- Every executed or blocked read-only/test action has a prediction record before the result is known.
- The score is reported even when policy blocks the command, the command fails, or the prediction is wrong.
- Failed predictions are saved as training signal, not hidden.
- The planner can choose inspect-only actions when predicted uncertainty is high.

### 6. Execute Scoped Computer Actions

Deliverable:

- Optional permission-gated executor for approved read-only inspection commands and tests as the next accepted rung.
- Action ledger with exact command/input, cwd, policy decision, prediction, result, changed-file check, outputs, scoring, and rollback notes when applicable.
- Policy gates for destructive commands, secrets, networked writes, and irreversible external actions.
- Later, separately accepted rungs may add safe edits, browser/app actions, and git operations after their own gates and replayable checks.

Acceptance checks:

- Execution is opt-in and blocked unless the command matches the approved read-only/test policy.
- The executor never writes outside the approved scope for an episode; if a test creates files, the episode flags and scores that result explicitly.
- Tests or verification commands are attached to each future write episode where feasible.
- Human approval boundaries are explicit for commits, pushes, emails, payments, production changes, and external side effects.

### 7. Train From Verified Episodes

Deliverable:

- Episode dataset builder that converts verified traces into supervised examples for retrieval, routing, prediction, critique, and generation.
- PhaseSSM or PhaseMesh LM training jobs that use episodes without leaking secrets.
- Evaluation split that tests unseen repos, unseen failure modes, and delayed consequences.

Acceptance checks:

- Training and evaluation splits are documented and reproducible.
- Improvements are shown on held-out episodes, not only on replays.
- Model updates preserve or improve safety gates and refusal boundaries.

### 8. Close The Agentic Loop

Deliverable:

- Observe, remember, reason, plan, critique, act, verify, record, and train in a bounded loop.
- Episode replay UI or report that makes failures inspectable.
- Benchmarks for task completion, prediction accuracy, verifier agreement, rollback rate, and human intervention rate.

Acceptance checks:

- PhaseMesh completes bounded computer tasks end to end in clean and dirty worktrees.
- It can explain what it observed, what it predicted, what changed, and why it stopped.
- Generality claims remain limited to the evaluated task families.

## Claim Boundaries

- PhaseMesh is the app name. Do not rename the direction around alternate product branding.
- PhaseMesh is not currently a general LLM, a frontier chatbot, or a replacement for Qwen/Claude/GPT-style inference.
- PhaseSSM efficiency does not imply assistant quality.
- Teacher-poured behavior does not imply native reasoning ability.
- A read-only agent loop does not imply autonomous execution.
- Optional policy-gated read-only/test execution does not imply write capability, arbitrary shell access, or autonomous repository changes.
- Prediction-vs-result scoring is an episode-quality metric, not proof of broad task completion or general planning ability.
- Passing synthetic tasks does not imply robust performance on real user workspaces.
- Episode replay success does not imply transfer to unseen repos unless the held-out split proves it.
- Safety claims require blocked-action examples, secret-redaction tests, and irreversible-action gates.

## North-Star Acceptance Suite

PhaseMesh can claim progress toward the embodied computer-world model only when these checks are green:

- `shell_trace`: golden `llm-shell` prompts emit stable routes, organs, answers, and trace schemas.
- `ssm_quality`: PhaseSSM quality is measured against a matched transformer baseline on held-out data.
- `episode_schema`: observation, role output, prediction, action, result, verifier, and critique fields are present and replayable.
- `policy_gated_execution`: optional execution runs only approved read-only/test commands and records blocked attempts.
- `prediction_score`: next-state predictions are recorded before execution and scored before action success is counted.
- `critic_gate`: unsafe or unsupported actions are blocked in recorded examples.
- `executor_scope`: file writes and commands stay inside the approved workspace and action policy.
- `verification`: task completion includes tests, verifier output, or explicit reason why no check was possible.
- `held_out_transfer`: claimed capabilities hold on unseen projects or tasks, with failures reported.
- `human_boundary`: commits, pushes, emails, production changes, and other external side effects require explicit approval.
- `honest_status`: docs and CLI output continue to say what is measured, what is inferred, and what is not claimed.

The useful future version of PhaseMesh is not defined by sounding fluent. It is defined by whether it can build, test, remember, predict, act, and learn inside a computer environment while keeping its claims smaller than its evidence.
