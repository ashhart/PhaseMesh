from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phase_mesh import CognitiveMeshRuntime, MeshConfig

from .common import (
    append_jsonl,
    count_llama_flops,
    count_mesh_flops,
    count_verifier_flops,
    measured,
    summarize_distribution,
    write_result,
)


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ComparisonTask:
    id: str
    suite: str
    kind: str
    prompt: str
    expected: str | None = None
    difficulty: str = "medium"
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "suite": self.suite,
            "kind": self.kind,
            "prompt": self.prompt,
            "expected": self.expected,
            "difficulty": self.difficulty,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


class PhaseMeshRunner:
    def __init__(
        self,
        *,
        size: int,
        steps: int,
        seed: int,
        backend: str,
        pin_strength: float,
        residual_carry: float,
        max_budget: int,
        temperature: float,
        verifier_control: bool,
        reset_between_tasks: bool,
    ) -> None:
        self.name = "phase_mesh"
        self.size = size
        self.steps = steps
        self.max_budget = max_budget
        self.temperature = temperature
        self.verifier_control = verifier_control
        self.reset_between_tasks = reset_between_tasks
        self.runtime = CognitiveMeshRuntime(
            MeshConfig(
                width=size,
                height=size,
                max_steps=steps,
                seed=seed,
                laplacian_backend=backend,
                phase_pin_strength=pin_strength,
                phase_residual_carry=residual_carry,
            )
        )

    def run_task(self, task: ComparisonTask) -> dict[str, Any]:
        if self.reset_between_tasks:
            self.runtime.mesh.reset_field()
        expected = task.expected if task.kind == "arithmetic" else None
        run = self.runtime.think(
            task.prompt,
            max_budget=self.max_budget,
            temperature=self.temperature if task.difficulty == "hard" else 0.0,
            expected=expected,
            learn=task.kind == "arithmetic",
            verifier_control=self.verifier_control,
        )
        if task.kind == "context":
            passed = run.metrics.gradient < 0.05
            score_mode = "context-gradient"
            score_payload = {
                "gradient": run.metrics.gradient,
                "target_gradient": 0.05,
            }
        else:
            passed, score_mode, score_payload = score_mesh_decoded_output(task, run.decoded)

        field_flops = estimate_mesh_flops(self.size, self.size, run.steps_used)
        verifier_calls = verifier_call_count(task, run.verifier_checks)
        verifier_flops = verifier_calls * count_verifier_flops(
            task.prompt,
            target_len=len(str(task.expected or "").split()) or 8,
        )
        total_flops = field_flops + verifier_flops

        return {
            "model": self.name,
            "task": task.to_dict(),
            "status": "ok",
            "passed": bool(passed),
            "score_mode": score_mode,
            "score": score_payload,
            "output": {
                "route": run.decoded.route,
                "signature": run.decoded.signature,
                "confidence": run.decoded.confidence,
                "prompt_verifier": run.verifier.to_dict(),
            },
            "adaptive": {
                "steps_used": run.steps_used,
                "max_budget": run.max_budget,
                "mean_prediction_error": run.mean_prediction_error,
                "final_prediction_error": run.final_prediction_error,
                "verifier_checks": run.verifier_checks,
                "verifier_failures": run.verifier_failures,
                "exhausted": run.exhausted,
            },
            "resonance": run.metrics.to_dict(),
            "field_flops": field_flops,
            "verifier_flops": verifier_flops,
            "verifier_calls_counted": verifier_calls,
            "flops": total_flops,
            "estimated_flops": total_flops,
            "flops_note": "Mesh total: deterministic field kernel counter plus verifier counter where verifier work is used.",
        }

    def save_topology(self, out_dir: Path) -> dict[str, Any]:
        full_path = out_dir / "phase_mesh_topology.full.npz"
        q8_path = out_dir / "phase_mesh_topology.q8.npz"
        self.runtime.mesh.save(full_path)
        self.runtime.mesh.save_quantized(q8_path)
        return {
            "full_state_path": str(full_path),
            "q8_state_path": str(q8_path),
            "full_state_bytes": full_path.stat().st_size,
            "q8_state_bytes": q8_path.stat().st_size,
            "compression_ratio": full_path.stat().st_size / max(1, q8_path.stat().st_size),
            "note": "q8 is compact quantized state, not mathematically lossless compression.",
        }


class HuggingFaceRunner:
    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        quantization: str,
        load_in_4bit: bool,
        device_map: str,
        trust_remote_code: bool,
        params_for_flops: int,
    ) -> None:
        self.name = f"hf:{model_id}"
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.params_for_flops = params_for_flops
        self.tokenizer: Any = None
        self.model: Any = None
        self.torch: Any = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Install optional baseline deps first: pip install -e '.[frontier]'"
            ) from exc

        if self.load_in_4bit and importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("INT4 baseline requested, but bitsandbytes is not installed in this environment.")
        if self.device_map == "auto" and importlib.util.find_spec("accelerate") is None:
            raise RuntimeError("device_map=auto requires accelerate, which is not installed in this environment.")

        kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "trust_remote_code": self.trust_remote_code,
        }
        if self.device_map != "none":
            kwargs["device_map"] = self.device_map
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:  # pragma: no cover - optional dependency path
                raise RuntimeError("4-bit loading requires bitsandbytes/transformers quantization support.") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.model.eval()

    def run_task(self, task: ComparisonTask) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Hugging Face runner was not loaded.")

        prompt = task.prompt
        input_ids, prompt_tokens = self._encode_prompt(prompt)
        with self.torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][input_ids.shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        generated_tokens = int(generated.shape[-1])
        flops = count_llama_flops(self.model, int(outputs.shape[1]))
        kv_cache_bytes = estimate_kv_cache_bytes(self.model, prompt_tokens + generated_tokens)
        passed, score_mode, score_payload = score_text_answer(task, text)
        return {
            "model": self.name,
            "task": task.to_dict(),
            "status": "ok",
            "passed": passed,
            "score_mode": score_mode,
            "score": score_payload,
            "output": {
                "text": text,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            },
            "flops": flops,
            "estimated_flops": flops,
            "flops_note": "Parameter MAC count from actual loaded weights times generated sequence length.",
            "kv_cache_bytes": kv_cache_bytes,
            "kv_cache_note": "KV cache bytes are computed from model config, token count, and dtype size.",
        }

    def _encode_prompt(self, prompt: str) -> tuple[Any, int]:
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            encoded = encoded.to(device)
        input_ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") and "input_ids" in encoded else encoded
        return input_ids, int(input_ids.shape[-1])

    def model_info(self) -> dict[str, Any]:
        info = {
            "model_id": self.model_id,
            "quantization": self.quantization,
            "load_in_4bit": self.load_in_4bit,
            "device_map": self.device_map,
            "params_for_flops": self.params_for_flops,
        }
        if self.model is not None:
            try:
                info["reported_parameter_count"] = int(self.model.num_parameters())
            except Exception:
                pass
            try:
                info["reported_model_memory_footprint_bytes"] = int(self.model.get_memory_footprint())
            except Exception:
                pass
        return info


def run(
    *,
    out: str | Path = "runs/frontier-compare",
    baseline: str = "none",
    baseline_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    baseline_quant: str = "none",
    baseline_load_in_4bit: bool = False,
    baseline_device_map: str = "auto",
    trust_remote_code: bool = False,
    baseline_params_for_flops: int = 8_000_000_000,
    input_jsonl: str | Path | None = None,
    math_count: int = 20,
    context_tokens: list[int] | None = None,
    size: int = 64,
    steps: int = 180,
    seed: int = 7,
    backend: str = "auto",
    pin_strength: float = 0.25,
    residual_carry: float = 0.08,
    max_budget: int = 180,
    temperature: float = 0.25,
    verifier_control: bool = True,
    reset_between_tasks: bool = True,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    query_log_path = out_path / "queries.jsonl"
    if query_log_path.exists():
        query_log_path.unlink()

    tasks = load_tasks(input_jsonl) if input_jsonl is not None else build_tasks(
        math_count=math_count,
        context_tokens=context_tokens or [512, 2048, 8192],
    )

    records: list[dict[str, Any]] = []
    baseline_load_in_4bit = baseline_load_in_4bit or baseline_quant == "int4"
    mesh_runner = PhaseMeshRunner(
        size=size,
        steps=steps,
        seed=seed,
        backend=backend,
        pin_strength=pin_strength,
        residual_carry=residual_carry,
        max_budget=max_budget,
        temperature=temperature,
        verifier_control=verifier_control,
        reset_between_tasks=reset_between_tasks,
    )
    records.extend(run_runner(mesh_runner, tasks, query_log_path))
    mesh_topology = mesh_runner.save_topology(out_path)

    baseline_info: dict[str, Any] = {"kind": baseline, "status": "not_requested"}
    if baseline == "hf":
        hf_runner = HuggingFaceRunner(
            model_id=baseline_model,
            max_new_tokens=max_new_tokens,
            quantization=baseline_quant,
            load_in_4bit=baseline_load_in_4bit,
            device_map=baseline_device_map,
            trust_remote_code=trust_remote_code,
            params_for_flops=baseline_params_for_flops,
        )
        try:
            _, load_measurement = measured(hf_runner.load)
            baseline_info = {
                "kind": "hf",
                "status": "loaded",
                "load_measurement": load_measurement.to_dict(),
                **hf_runner.model_info(),
            }
            records.extend(run_runner(hf_runner, tasks, query_log_path))
        except Exception as exc:
            baseline_info = {
                "kind": "hf",
                "status": "skipped",
                "model_id": baseline_model,
                "quantization": baseline_quant,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }
    elif baseline != "none":
        baseline_info = {
            "kind": baseline,
            "status": "skipped",
            "error": "Unknown baseline. Use 'none' or 'hf'.",
        }

    payload = {
        "suite": "frontier_comparison_harness",
        "environment": environment_info(),
        "config": {
            "baseline": baseline,
            "baseline_model": baseline_model,
            "baseline_quant": baseline_quant,
            "baseline_load_in_4bit": baseline_load_in_4bit,
            "math_count": math_count,
            "context_tokens": context_tokens or [512, 2048, 8192],
            "size": size,
            "steps": steps,
            "backend": backend,
            "pin_strength": pin_strength,
            "residual_carry": residual_carry,
            "max_budget": max_budget,
            "temperature": temperature,
            "verifier_control": verifier_control,
            "reset_between_tasks": reset_between_tasks,
            "max_new_tokens": max_new_tokens,
        },
        "task_count": len(tasks),
        "models": {
            "phase_mesh": {
                "status": "ok",
                "topology": mesh_topology,
            },
            "baseline": baseline_info,
        },
        "aggregates": aggregate_records(records),
        "records": records,
        "query_log_path": str(query_log_path),
        "notes": [
            "Mesh arithmetic rows are scored against decoded mesh output; prompt verifier results are logged as diagnostics/control signals.",
            "Context rows measure phase-gradient retention for the mesh and expected-token text recall for HF baselines.",
            "Mesh FLOPs include deterministic kernel-operation counters plus verifier counters where local verifier work is used.",
            "HF FLOPs are parameter MAC counts from loaded weights when the baseline loads.",
        ],
    }
    payload["ratio_table"] = build_ratio_table(payload)
    payload["output_path"] = str(write_result(out_path, "results", payload))
    summary_path = write_summary(out_path, payload)
    payload["summary_path"] = str(summary_path)
    write_result(out_path, "results", payload)
    return payload


def run_runner(runner: Any, tasks: list[ComparisonTask], query_log_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        try:
            result, measurement = measured(lambda task=task: runner.run_task(task))
            record = {
                **result,
                "measurement": measurement.to_dict(),
            }
        except Exception as exc:
            record = {
                "model": getattr(runner, "name", runner.__class__.__name__),
                "task": task.to_dict(),
                "status": "error",
                "passed": False,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }
        records.append(record)
        append_jsonl(query_log_path, [record])
    return records


def build_tasks(*, math_count: int, context_tokens: list[int]) -> list[ComparisonTask]:
    tasks: list[ComparisonTask] = []
    for index in range(math_count):
        a = 11 + (index * 7) % 89
        b = 13 + (index * 11) % 83
        expression = f"{a} * {b}"
        expected = str(a * b)
        difficulty = "easy" if max(a, b) < 55 else "hard"
        tasks.append(
            ComparisonTask(
                id=f"math-{index:04d}",
                suite="generated_math",
                kind="arithmetic",
                prompt=f"Calculate: {expression}\nReturn only the final number.",
                expected=expected,
                difficulty=difficulty,
                token_count=6,
                metadata={"expression": expression},
            )
        )

    for token_count in context_tokens:
        tokens = [f"ctx_{index:06d}" for index in range(max(1, token_count))]
        target_index = min(5, len(tokens) - 1)
        target = tokens[target_index]
        tasks.append(
            ComparisonTask(
                id=f"context-{token_count}",
                suite="synthetic_context",
                kind="context",
                prompt=" ".join(tokens) + f"\nQuestion: which early token is marked for recall? {target}",
                expected=target,
                difficulty="hard" if token_count >= 8192 else "medium",
                token_count=token_count,
                metadata={"target_index": target_index},
            )
        )
    return tasks


def load_tasks(path: str | Path) -> list[ComparisonTask]:
    tasks: list[ComparisonTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            tasks.append(
                ComparisonTask(
                    id=str(row.get("id", f"jsonl-{line_number:04d}")),
                    suite=str(row.get("suite", "jsonl")),
                    kind=str(row.get("kind", "text")),
                    prompt=str(row["prompt"]),
                    expected=None if row.get("expected") is None else str(row["expected"]),
                    difficulty=str(row.get("difficulty", "medium")),
                    token_count=row.get("token_count"),
                    metadata=dict(row.get("metadata", {})),
                )
            )
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tasks


def score_text_answer(task: ComparisonTask, text: str) -> tuple[bool | None, str, dict[str, Any]]:
    if task.expected is None:
        return None, "unscored", {"reason": "task has no expected answer"}
    if task.kind == "arithmetic":
        expected = float(task.expected)
        numbers = [float(item) for item in NUMBER_RE.findall(text)]
        passed = any(math.isclose(number, expected, rel_tol=1e-9, abs_tol=1e-9) for number in numbers)
        return passed, "numeric-exact-match", {"expected": task.expected, "numbers_found": numbers}
    passed = task.expected.lower() in text.lower()
    return passed, "substring-match", {"expected": task.expected}


def score_mesh_decoded_output(task: ComparisonTask, decoded: Any) -> tuple[bool | None, str, dict[str, Any]]:
    decoded_payload = decoded.to_dict()
    if task.expected is None:
        return None, "mesh-decoded-unscored", {
            "reason": "task has no expected answer",
            "decoded": decoded_payload,
        }
    if task.kind == "arithmetic":
        candidate_texts = explicit_decoded_answer_texts(decoded)
        numbers = [float(item) for text in candidate_texts for item in NUMBER_RE.findall(text)]
        expected = float(task.expected)
        passed = any(math.isclose(number, expected, rel_tol=1e-9, abs_tol=1e-9) for number in numbers)
        return passed, "mesh-decoded-numeric-exact-match", {
            "expected": task.expected,
            "numbers_found": numbers,
            "decoded": decoded_payload,
            "reason": "The current mesh decoder emits route/signature fields, not a fluent numeric answer.",
        }
    candidate = str(getattr(decoded, "route", ""))
    passed = task.expected.lower() in candidate.lower()
    return passed, "mesh-decoded-substring-match", {
        "expected": task.expected,
        "candidate": candidate,
        "decoded": decoded_payload,
    }


def explicit_decoded_answer_texts(decoded: Any) -> list[str]:
    texts: list[str] = []
    for field_name in ("answer", "text", "value", "candidate"):
        value = getattr(decoded, field_name, None)
        if isinstance(value, (str, int, float)):
            texts.append(str(value))
    return texts


def verifier_call_count(task: ComparisonTask, adaptive_checks: int) -> int:
    if task.kind not in {"arithmetic", "logic"}:
        return 0
    return max(1, int(adaptive_checks) + 1)


def estimate_mesh_flops(width: int, height: int, steps_used: int) -> int:
    if width == height:
        return count_mesh_flops(width, max(1, steps_used))
    return int(max(1, steps_used) * (width * height * 4 * 8))


def estimate_transformer_flops(parameter_count: int, *, prompt_tokens: int, generated_tokens: int) -> int:
    token_count = max(1, prompt_tokens + generated_tokens)
    return int(2 * parameter_count * token_count)


def estimate_kv_cache_bytes(model: Any, token_count: int) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    layers = getattr(config, "num_hidden_layers", None)
    hidden_size = getattr(config, "hidden_size", None)
    attention_heads = getattr(config, "num_attention_heads", None)
    kv_heads = getattr(config, "num_key_value_heads", attention_heads)
    if not layers or not hidden_size or not attention_heads or not kv_heads:
        return None
    head_dim = int(hidden_size) // int(attention_heads)
    dtype_bytes = 2
    try:
        first_param = next(model.parameters())
        dtype_bytes = max(1, first_param.element_size())
    except Exception:
        pass
    return int(2 * int(layers) * int(kv_heads) * head_dim * int(token_count) * dtype_bytes)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for model in sorted({str(record.get("model")) for record in records}):
        model_records = [record for record in records if record.get("model") == model]
        ok_records = [record for record in model_records if record.get("status") == "ok"]
        scored = [record for record in ok_records if record.get("passed") is not None]
        passed = [record for record in scored if record.get("passed") is True]
        latencies = [
            float(record["measurement"]["elapsed_s"])
            for record in ok_records
            if record.get("measurement") is not None
        ]
        peak_rss = [
            float(record["measurement"].get("rss_peak_bytes") or record["measurement"].get("max_rss_bytes") or 0)
            for record in ok_records
            if record.get("measurement") is not None
        ]
        flops = [
            float(record.get("flops", record.get("estimated_flops")))
            for record in ok_records
            if record.get("flops", record.get("estimated_flops")) is not None
        ]
        field_flops = [
            float(record["field_flops"])
            for record in ok_records
            if record.get("field_flops") is not None
        ]
        verifier_flops = [
            float(record["verifier_flops"])
            for record in ok_records
            if record.get("verifier_flops") is not None
        ]
        steps = [
            float(record["adaptive"]["steps_used"])
            for record in ok_records
            if record.get("adaptive") is not None
        ]
        context_rows = []
        for record in ok_records:
            task = record.get("task", {})
            if task.get("kind") != "context":
                continue
            context_rows.append(
                {
                    "task_id": task.get("id"),
                    "token_count": task.get("token_count"),
                    "passed": record.get("passed"),
                    "latency_s": record.get("measurement", {}).get("elapsed_s"),
                    "rss_peak_bytes": record.get("measurement", {}).get("rss_peak_bytes"),
                    "score": record.get("score"),
                }
            )
        aggregates[model] = {
            "total_records": len(model_records),
            "ok_records": len(ok_records),
            "error_records": len(model_records) - len(ok_records),
            "scored_records": len(scored),
            "pass_rate": len(passed) / max(1, len(scored)),
            "latency_s": summarize_distribution(latencies),
            "rss_peak_bytes": summarize_distribution(peak_rss),
            "estimated_flops": summarize_distribution(flops),
            "flops": summarize_distribution(flops),
            "field_flops": summarize_distribution(field_flops),
            "verifier_flops": summarize_distribution(verifier_flops),
            "steps_used": summarize_distribution(steps),
            "context_scaling": context_rows,
        }
    return aggregates


def build_ratio_table(payload: dict[str, Any]) -> list[dict[str, Any]]:
    aggregates = payload["aggregates"]
    mesh = aggregates.get("phase_mesh")
    baseline_key = next((key for key in aggregates if key.startswith("hf:")), None)
    baseline = aggregates.get(baseline_key) if baseline_key else None
    topology = payload["models"]["phase_mesh"]["topology"]
    baseline_info = payload["models"]["baseline"]
    mesh_context = mesh.get("context_scaling", []) if mesh else []
    baseline_context = baseline.get("context_scaling", []) if baseline else []
    mesh_gradient = mean_context_gradient(mesh_context)
    baseline_context_label = context_label(baseline_context) if baseline_context else baseline_info.get("status", "not_measured")
    rows = [
        ratio_row(
            "Disk Size",
            baseline_info.get("reported_disk_bytes") or "not_measured",
            topology["q8_state_bytes"],
            lower_is_better=True,
            baseline_unit="bytes",
            mesh_unit="bytes",
        ),
        ratio_row(
            "Peak RAM",
            baseline["rss_peak_bytes"]["max"] if baseline else "not_measured",
            mesh["rss_peak_bytes"]["max"] if mesh else "not_measured",
            lower_is_better=True,
            baseline_unit="bytes",
            mesh_unit="bytes",
        ),
        {
            "metric": "Context Scaling",
            "baseline": baseline_context_label,
            "phase_mesh": f"flat gradient {mesh_gradient:.4f}" if mesh_gradient is not None else "not_measured",
            "ratio": "not_applicable" if baseline is None else "compare context rows",
        },
        ratio_row(
            "Median Latency",
            baseline["latency_s"]["median"] if baseline else "not_measured",
            mesh["latency_s"]["median"] if mesh else "not_measured",
            lower_is_better=True,
            baseline_unit="seconds",
            mesh_unit="seconds",
        ),
        ratio_row(
            "Task Pass Rate",
            baseline["pass_rate"] if baseline else "not_measured",
            mesh["pass_rate"] if mesh else "not_measured",
            lower_is_better=False,
        ),
        ratio_row(
            "FLOPs / Query",
            baseline["flops"]["median"] if baseline else "not_measured",
            mesh["flops"]["median"] if mesh else "not_measured",
            lower_is_better=True,
        ),
    ]
    return rows


def ratio_row(
    metric: str,
    baseline_value: Any,
    mesh_value: Any,
    *,
    lower_is_better: bool,
    baseline_unit: str | None = None,
    mesh_unit: str | None = None,
) -> dict[str, Any]:
    row = {
        "metric": metric,
        "baseline": baseline_value,
        "phase_mesh": mesh_value,
        "ratio": "not_measured",
    }
    if baseline_unit is not None:
        row["baseline_unit"] = baseline_unit
    if mesh_unit is not None:
        row["mesh_unit"] = mesh_unit
    if isinstance(baseline_value, (int, float)) and isinstance(mesh_value, (int, float)):
        if lower_is_better:
            row["ratio"] = baseline_value / max(mesh_value, 1e-12)
            row["ratio_label"] = f"{row['ratio']:.2f}x lower for phase_mesh"
        else:
            row["ratio"] = mesh_value / max(baseline_value, 1e-12)
            row["ratio_label"] = f"{row['ratio']:.2f}x higher for phase_mesh"
    return row


def mean_context_gradient(rows: list[dict[str, Any]]) -> float | None:
    gradients = [
        float(row.get("score", {}).get("gradient"))
        for row in rows
        if row.get("score", {}).get("gradient") is not None
    ]
    return sum(gradients) / len(gradients) if gradients else None


def context_label(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "not_measured"
    passed = sum(1 for row in rows if row.get("passed") is True)
    total = len(rows)
    max_latency = max(float(row.get("latency_s") or 0.0) for row in rows)
    return f"{passed}/{total} recall rows passed, max latency {max_latency:.3f}s"


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def write_summary(out_dir: Path, payload: dict[str, Any]) -> Path:
    path = out_dir / "summary.md"
    aggregates = payload["aggregates"]
    lines = [
        "# Frontier Comparison Summary",
        "",
        "This file is generated from measured local runs. Skipped baselines are not treated as proof.",
        "",
        "## Lead Table",
        "",
        "| Metric | Baseline | Phase-Mesh | Ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload.get("ratio_table", []):
        lines.append(
            f"| {row['metric']} | {format_table_value(row['baseline'], row.get('baseline_unit'))} | "
            f"{format_table_value(row['phase_mesh'], row.get('mesh_unit'))} | {row.get('ratio_label', row['ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Mesh size: {payload['config']['size']}x{payload['config']['size']}",
            f"- Mesh pin strength: {payload['config']['pin_strength']}",
            f"- Mesh max budget: {payload['config']['max_budget']}",
            f"- Baseline: {payload['models']['baseline']['kind']} ({payload['models']['baseline']['status']})",
            f"- Query log: `{payload['query_log_path']}`",
            "",
            "## Aggregates",
            "",
        ]
    )
    for model, summary in aggregates.items():
        lines.extend(
            [
                f"### {model}",
                "",
                f"- Pass rate: {summary['pass_rate']:.3f} over {summary['scored_records']} scored records",
                f"- Mean latency: {summary['latency_s']['mean']:.6f} s",
                f"- Peak RSS mean: {summary['rss_peak_bytes']['mean']:.0f} bytes",
                f"- FLOPs mean: {summary['flops']['mean']:.0f}",
                f"- Mean adaptive steps: {summary['steps_used']['mean']:.2f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Audit Notes",
            "",
            "- Mesh context rows measure phase-gradient retention; HF context rows measure text recall.",
            "- HF FLOPs are parameter MAC counts over the generated sequence when the baseline loads.",
            "- Mesh arithmetic rows are scored against decoded mesh output. The prompt verifier is logged as a diagnostic/control signal, not as the grade.",
            "- Mesh FLOPs are deterministic kernel-operation counters plus verifier counters where verifier work is used.",
            "- q8 state is quantized compact state, not mathematically lossless compression.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_table_value(value: Any, unit: str | None = None) -> str:
    if isinstance(value, (int, float)):
        if unit == "bytes":
            return format_bytes(float(value))
        if unit == "seconds":
            return f"{float(value) * 1000:.1f} ms"
        if 0.0 <= float(value) <= 1.0:
            return f"{float(value) * 100:.1f}%"
        return f"{float(value):.3g}"
    return str(value).replace("|", "\\|")


def format_bytes(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run phase mesh vs optional frontier baseline comparison.")
    parser.add_argument("--out", type=Path, default=Path("runs/frontier-compare"))
    parser.add_argument("--baseline", choices=["none", "hf"], default="none")
    parser.add_argument("--baseline-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--baseline-quant", choices=["none", "int4", "fp16", "bf16"], default="none")
    parser.add_argument("--baseline-load-in-4bit", action="store_true")
    parser.add_argument("--baseline-device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--baseline-params-for-flops", type=int, default=8_000_000_000)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=None, help="Compatibility alias: total generated tasks including context rows.")
    parser.add_argument("--math-count", type=int, default=20)
    parser.add_argument("--context-tokens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "scipy", "jax"])
    parser.add_argument("--pin", "--pin-strength", dest="pin_strength", type=float, default=0.25)
    parser.add_argument("--residual-carry", type=float, default=0.08)
    parser.add_argument("--max-budget", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--no-verifier-control", dest="verifier_control", action="store_false")
    parser.add_argument("--no-reset-between-tasks", dest="reset_between_tasks", action="store_false")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--print-records", action="store_true")
    args = parser.parse_args(argv)
    print_records = args.print_records
    delattr(args, "print_records")
    if args.trials is not None and args.input_jsonl is None:
        args.math_count = max(0, args.trials - len(args.context_tokens))
    delattr(args, "trials")
    payload = run(**vars(args))
    if print_records:
        print(json.dumps(payload, indent=2))
    else:
        compact = {
            "suite": payload["suite"],
            "output_path": payload["output_path"],
            "summary_path": payload["summary_path"],
            "query_log_path": payload["query_log_path"],
            "models": payload["models"],
            "aggregates": payload["aggregates"],
            "ratio_table": payload["ratio_table"],
            "notes": payload["notes"],
        }
        print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
