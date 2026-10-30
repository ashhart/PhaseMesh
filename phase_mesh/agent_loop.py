from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .llm_shell import PhaseMeshLLMShell


ROLE_NAMES = (
    "observer",
    "memory",
    "reasoner",
    "planner",
    "critic",
    "executor",
    "recorder",
    "trainer",
)


@dataclass
class AgentRole:
    name: str
    purpose: str
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentEpisode:
    status: str
    prompt: str
    answer: str
    observation: dict[str, Any]
    agents: dict[str, dict[str, Any]]
    plan: dict[str, Any]
    prediction: dict[str, Any]
    action_result: dict[str, Any]
    prediction_score: dict[str, Any]
    trace: list[dict[str, Any]]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhaseMeshAgentLoop:
    """Embodied PhaseMesh loop over a local computer workspace.

    The loop is intentionally conservative. It observes workspace state,
    routes through the existing PhaseMeshLLMShell organs, proposes a next
    action, critiques the risk, and records an episode for future world-model
    training. It does not execute write actions.
    """

    episode_name = "episodes.jsonl"

    def __init__(
        self,
        *,
        workspace: str | Path | None = None,
        state_dir: str | Path | None = None,
        language_model_dir: str | Path | None = None,
        chat_model_dir: str | Path | None = None,
        weight_artifact_dir: str | Path | None = None,
        shell: PhaseMeshLLMShell | None = None,
        execute_readonly: bool = False,
        command_timeout: float = 10.0,
    ) -> None:
        self.workspace = Path(workspace or Path.cwd()).resolve()
        self.state_dir = Path(state_dir) if state_dir is not None else Path("runs/agent-loop")
        self.execute_readonly = bool(execute_readonly)
        self.command_timeout = float(command_timeout)
        self.shell = shell or PhaseMeshLLMShell.load(
            self.state_dir / "shell",
            language_model_dir=language_model_dir,
            chat_model_dir=chat_model_dir,
            weight_artifact_dir=weight_artifact_dir,
        )

    @property
    def episode_path(self) -> Path:
        return self.state_dir / self.episode_name

    def observe(self, prompt: str, *, max_files: int = 40) -> dict[str, Any]:
        files = self._workspace_files(max_files=max_files)
        git_status = self._run_readonly(["git", "status", "--short", "--branch"])
        return {
            "type": "computer-workspace",
            "workspace": str(self.workspace),
            "prompt": prompt,
            "files_sample": files,
            "file_count_sampled": len(files),
            "git_status": git_status,
            "episode_count": self._episode_count(),
        }

    def run(self, prompt: str) -> dict[str, Any]:
        started = time.perf_counter()
        text = str(prompt).strip()
        observation = self.observe(text)
        shell_result = self.shell.run(text)
        plan = self._plan(text, observation, shell_result)
        prediction = self._predict_next_observation(observation, plan, shell_result)
        critic = self._critic(plan, observation)
        action_result = self._execute_plan(plan, critic)
        prediction_score = self._score_prediction(prediction, action_result)

        agents = {
            "observer": AgentRole(
                name="observer",
                purpose="turn terminal/files/git state into compact observations",
                output={
                    "workspace": observation["workspace"],
                    "files_sampled": observation["file_count_sampled"],
                    "git_dirty": self._git_dirty(observation),
                },
            ),
            "memory": AgentRole(
                name="memory",
                purpose="retrieve and update persistent PhaseMesh shell state",
                output={
                    "memory_records": shell_result.get("data", {}).get("memory_records", 0),
                    "binding_records": shell_result.get("data", {}).get("binding_records", 0),
                },
            ),
            "reasoner": AgentRole(
                name="reasoner",
                purpose="route through verified PhaseMesh organs",
                output={
                    "route": shell_result.get("route"),
                    "reasoning": shell_result.get("data", {}).get("reasoning", {}),
                },
            ),
            "planner": AgentRole(
                name="planner",
                purpose="choose the next bounded computer action",
                output=plan,
            ),
            "critic": AgentRole(
                name="critic",
                purpose="block unsafe or ungrounded actions before execution",
                output=critic,
            ),
            "executor": AgentRole(
                name="executor",
                purpose="perform approved computer actions after policy gates",
                output=action_result,
            ),
            "recorder": AgentRole(
                name="recorder",
                purpose="persist episodes for future state-action-next-state learning",
                output={"episode_path": str(self.episode_path)},
            ),
            "trainer": AgentRole(
                name="trainer",
                purpose="convert verified episodes into future training examples",
                output={
                    "trained": False,
                    "reason": "episode dataset building is the next gated rung",
                    "episode_available": True,
                    "prediction_score": prediction_score,
                },
            ),
        }

        episode = AgentEpisode(
            status="ok",
            prompt=text,
            answer=str(shell_result.get("answer", "")),
            observation=observation,
            agents={name: asdict(role) for name, role in agents.items()},
            plan=plan,
            prediction=prediction,
            action_result=action_result,
            prediction_score=prediction_score,
            trace=list(shell_result.get("trace", [])),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        self._record_episode(episode)
        self.shell.save(self.state_dir / "shell")
        return episode.to_dict()

    def manifest(self) -> dict[str, Any]:
        return {
            "type": "phase-mesh-agent-loop",
            "roles": list(ROLE_NAMES),
            "workspace": str(self.workspace),
            "state_dir": str(self.state_dir),
            "episode_path": str(self.episode_path),
            "episode_count": self._episode_count(),
            "execute_readonly": self.execute_readonly,
            "claim_boundary": (
                "This is a bounded computer-world episode recorder and control shell, "
                "not a general autonomous LLM."
            ),
        }

    def _plan(self, prompt: str, observation: dict[str, Any], shell_result: dict[str, Any]) -> dict[str, Any]:
        route = str(shell_result.get("route", "generation"))
        lowered = prompt.lower()
        dirty = self._git_dirty(observation)
        if route == "arithmetic":
            action = "answer"
            command = None
            reason = "arithmetic route produced a deterministic answer"
        elif route == "memory":
            action = "answer"
            command = None
            reason = "memory route can answer or update persistent shell state"
        elif route == "code" and any(word in lowered for word in ("test", "failing", "failure", "bug", "fix")):
            action = "inspect"
            command = "python3 -m pytest -q"
            reason = "code/debug prompt should ground itself in tests before editing"
        elif route == "code":
            action = "inspect"
            command = "git status --short --branch"
            reason = "code prompt should inspect workspace state before proposing edits"
        elif dirty:
            action = "inspect"
            command = "git diff --stat"
            reason = "workspace has visible changes; inspect before changing behavior"
        else:
            action = "answer"
            command = None
            reason = "generation route has no safe bounded computer action yet"
        will_execute = self.execute_readonly and command is not None
        return {
            "action": action,
            "command": command,
            "reason": reason,
            "execute": will_execute,
            "safety": (
                "policy-gated read-only/test execution is enabled"
                if will_execute
                else "read-only plan; no command executed by this scaffold"
            ),
        }

    def _predict_next_observation(
        self,
        observation: dict[str, Any],
        plan: dict[str, Any],
        shell_result: dict[str, Any],
    ) -> dict[str, Any]:
        action = str(plan.get("action", "answer"))
        if action == "inspect":
            predicted = "a read-only command would add terminal output without changing files"
        elif action == "answer":
            predicted = "user receives an answer and the episode log grows by one record"
        else:
            predicted = "next observation depends on a gated future action"
        return {
            "model": "state + action -> next observation",
            "action": action,
            "prediction": predicted,
            "expected_execution": action == "inspect",
            "expected_tracked_file_change": False,
            "route": shell_result.get("route"),
            "episode_count_after": int(observation.get("episode_count", 0)) + 1,
        }

    def _critic(self, plan: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
        command = plan.get("command")
        if command is None:
            return {"approved": True, "risk": "low", "reason": "answer-only plan"}
        try:
            argv = shlex.split(str(command))
        except ValueError:
            argv = []
        approved = self._argv_allowed(argv)
        return {
            "approved": bool(approved),
            "risk": "low" if approved else "blocked",
            "reason": "read-only or test command" if approved else "command is outside scaffold policy",
        }

    def _execute_plan(self, plan: dict[str, Any], critic: dict[str, Any]) -> dict[str, Any]:
        command = plan.get("command")
        if command is None:
            return {
                "executed": False,
                "status": "skipped",
                "reason": "answer-only plan",
                "planned_command": None,
            }
        if not self.execute_readonly:
            return {
                "executed": False,
                "status": "skipped",
                "reason": "execution requires execute_readonly=True",
                "planned_command": command,
            }
        if not critic.get("approved"):
            return {
                "executed": False,
                "status": "blocked",
                "reason": critic.get("reason", "critic blocked command"),
                "planned_command": command,
            }

        before = self._run_readonly(["git", "status", "--short"])
        started = time.perf_counter()
        result = self._run_planned_command(str(command))
        after = self._run_readonly(["git", "status", "--short"])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        before_dirty = self._status_payload_dirty(before)
        after_dirty = self._status_payload_dirty(after)
        return {
            "executed": result["returncode"] != 127,
            "status": "ok" if result["returncode"] == 0 else "failed",
            "planned_command": command,
            "command": result["command"],
            "returncode": result["returncode"],
            "stdout": _truncate(result["stdout"]),
            "stderr": _truncate(result["stderr"]),
            "elapsed_ms": elapsed_ms,
            "tracked_dirty_before": before_dirty,
            "tracked_dirty_after": after_dirty,
            "tracked_status_changed": before.get("stdout", "") != after.get("stdout", ""),
        }

    def _score_prediction(self, prediction: dict[str, Any], action_result: dict[str, Any]) -> dict[str, Any]:
        if not action_result.get("executed"):
            return {
                "status": "not_scored",
                "reason": action_result.get("reason", "no executed action"),
            }
        expected_no_change = prediction.get("expected_tracked_file_change") is False
        no_change = not bool(action_result.get("tracked_status_changed"))
        command_returned = "returncode" in action_result
        checks = {
            "command_returned": command_returned,
            "tracked_files_unchanged": (no_change if expected_no_change else True),
        }
        score = sum(1 for ok in checks.values() if ok) / max(1, len(checks))
        return {
            "status": "scored",
            "score": score,
            "checks": checks,
            "prediction": prediction.get("prediction", ""),
        }

    def _workspace_files(self, *, max_files: int) -> list[str]:
        result = self._run_readonly(["git", "ls-files"])
        if result["returncode"] == 0 and result["stdout"].strip():
            return result["stdout"].splitlines()[:max_files]
        files: list[str] = []
        for path in sorted(self.workspace.rglob("*")):
            if path.is_file():
                try:
                    files.append(str(path.relative_to(self.workspace)))
                except ValueError:
                    files.append(str(path))
            if len(files) >= max_files:
                break
        return files

    def _git_dirty(self, observation: dict[str, Any]) -> bool:
        stdout = str(observation.get("git_status", {}).get("stdout", ""))
        return any(line and not line.startswith("##") for line in stdout.splitlines())

    def _status_payload_dirty(self, payload: dict[str, Any]) -> bool:
        stdout = str(payload.get("stdout", ""))
        return any(bool(line.strip()) for line in stdout.splitlines())

    def _run_readonly(self, command: list[str]) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                command,
                cwd=self.workspace,
                text=True,
                capture_output=True,
                timeout=3.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}
        return {
            "command": command,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }

    def _run_planned_command(self, command: str) -> dict[str, Any]:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}
        if not self._argv_allowed(argv):
            return {
                "command": argv,
                "returncode": 127,
                "stdout": "",
                "stderr": "command rejected by PhaseMeshAgentLoop policy",
            }
        return self._run_readonly_with_timeout(argv, timeout=self.command_timeout)

    def _run_readonly_with_timeout(self, command: list[str], *, timeout: float) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                command,
                cwd=self.workspace,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}
        return {
            "command": command,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }

    def _argv_allowed(self, argv: list[str]) -> bool:
        if argv[:2] == ["git", "status"]:
            return True
        if argv[:2] == ["git", "diff"]:
            return True
        if argv[:3] == ["python3", "-m", "pytest"]:
            return True
        if argv and Path(argv[0]).name == "pytest":
            return True
        return False

    def _episode_count(self) -> int:
        try:
            with self.episode_path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except FileNotFoundError:
            return 0

    def _record_episode(self, episode: AgentEpisode) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.episode_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode.to_dict(), sort_keys=True) + "\n")


def _truncate(value: str, *, limit: int = 8000) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"
