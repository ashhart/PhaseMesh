from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .language_model import PhaseLanguageModel, PhaseLMConfig


DEFAULT_DISTILL_PROMPTS = [
    "PhaseMesh is a compact phase language model that",
    "A useful local language model should",
    "When the shell receives a prompt it",
    "The phase memory stores context by",
    "To improve generation quality we",
]


TRANSFORMERS_IMPORT_ERROR = (
    "Transformer distillation requires optional teacher dependencies. "
    "Install them with `pip install transformers torch` or `pip install -e '.[bench]'`."
)


def read_distill_prompts(path: str | Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_DISTILL_PROMPTS)
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            prompts.append(value)
    return prompts or list(DEFAULT_DISTILL_PROMPTS)


def train_phase_lm_from_texts(
    texts: Iterable[str],
    *,
    out_dir: str | Path,
    config: PhaseLMConfig | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    model = PhaseLanguageModel(config or PhaseLMConfig())
    summary = model.train_lines(texts, max_tokens=max_tokens)
    model.save(out_dir)
    return {"status": "ok", "model_dir": str(out_dir), "summary": summary}


def distill_transformer_to_phase_lm(
    *,
    out_dir: str | Path,
    teacher_model: str = "sshleifer/tiny-gpt2",
    prompts: list[str] | None = None,
    samples_per_prompt: int = 2,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    device: str = "auto",
    torch_dtype: str = "auto",
    chat_template: str = "auto",
    completion_only: bool = True,
    seed: int = 7,
    phase_config: PhaseLMConfig | None = None,
    max_train_tokens: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import set_seed
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(TRANSFORMERS_IMPORT_ERROR) from exc

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    prompt_list = prompts or list(DEFAULT_DISTILL_PROMPTS)
    set_seed(int(seed))

    teacher = _load_teacher(teacher_model, device=device, torch_dtype=torch_dtype)
    tokenizer = teacher["tokenizer"]
    model = teacher["model"]
    target_device = teacher["device"]

    rows: list[dict[str, Any]] = []
    generated_texts: list[str] = []
    for prompt_index, prompt in enumerate(prompt_list):
        for sample_index in range(max(1, int(samples_per_prompt))):
            encoded = _encode_teacher_prompt(tokenizer, prompt, chat_template=chat_template)
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
            input_length = int(encoded["input_ids"].shape[-1])
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    do_sample=float(temperature) > 0.0,
                    temperature=max(float(temperature), 1e-6),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if completion_only:
                text = tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()
            else:
                text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            generated_texts.append(text)
            rows.append({
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "prompt": prompt,
                "text": text,
            })

    corpus_path = path / "teacher_corpus.txt"
    jsonl_path = path / "teacher_samples.jsonl"
    corpus_path.write_text("\n".join(generated_texts).strip() + "\n", encoding="utf-8")
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    phase_model_dir = path / "phase_lm"
    train_summary = train_phase_lm_from_texts(
        generated_texts,
        out_dir=phase_model_dir,
        config=phase_config or PhaseLMConfig(seed=seed),
        max_tokens=max_train_tokens,
    )
    payload = {
        "status": "ok",
        "type": "phase-mesh-transformer-distillation",
        "teacher_model": teacher_model,
        "device": target_device,
        "chat_template": chat_template,
        "completion_only": bool(completion_only),
        "prompts": len(prompt_list),
        "samples": len(rows),
        "teacher_corpus": str(corpus_path),
        "teacher_samples": str(jsonl_path),
        "phase_lm_dir": str(phase_model_dir),
        "phase_config": asdict(phase_config or PhaseLMConfig(seed=seed)),
        "train_summary": train_summary,
    }
    (path / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def pour_transformer_into_phase_lm(
    *,
    out_dir: str | Path,
    teacher_model: str,
    prompts: list[str] | None = None,
    resume_model_dir: str | Path | None = None,
    rounds: int = 1,
    samples_per_prompt: int = 1,
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    soft_top_k: int = 8,
    soft_weight: float = 1.0,
    device: str = "auto",
    torch_dtype: str = "auto",
    chat_template: str = "auto",
    completion_only: bool = True,
    seed: int = 7,
    phase_config: PhaseLMConfig | None = None,
    max_train_tokens: int | None = None,
    snapshot_interval: int = 1,
    sleep_seconds: float = 0.0,
    forever: bool = False,
) -> dict[str, Any]:
    """Continuously pour a transformer teacher into one PhaseMesh LM artifact.

    Each round appends hard teacher completions plus soft next-token top-k
    distributions to the same PhaseLanguageModel, then saves the updated
    artifact. This is meant for long CUDA jobs where a teacher stays resident
    and PhaseMesh keeps absorbing behavior without manual batch restarts.
    """

    try:
        from transformers import set_seed
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(TRANSFORMERS_IMPORT_ERROR) from exc

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    prompt_list = prompts or list(DEFAULT_DISTILL_PROMPTS)
    set_seed(int(seed))

    teacher = _load_teacher(teacher_model, device=device, torch_dtype=torch_dtype)
    tokenizer = teacher["tokenizer"]
    model = teacher["model"]
    target_device = teacher["device"]

    phase_model_dir = path / "phase_lm"
    resume_path = Path(resume_model_dir) if resume_model_dir is not None else phase_model_dir
    if (resume_path / "model.json").exists():
        phase_model = PhaseLanguageModel.load(resume_path)
        resumed_from = str(resume_path)
    else:
        phase_model = PhaseLanguageModel(phase_config or PhaseLMConfig(seed=seed))
        resumed_from = None

    samples_path = path / "teacher_samples.jsonl"
    events_path = path / "pour_events.jsonl"
    summary_path = path / "summary.json"
    corpus_path = path / "teacher_corpus.txt"
    total_samples = 0
    total_soft_contexts = 0
    total_soft_candidates = 0
    completed_rounds = 0
    round_index = 0

    while forever or round_index < max(1, int(rounds)):
        round_samples = 0
        round_soft_contexts = 0
        round_soft_candidates = 0
        round_tokens = 0
        with samples_path.open("a", encoding="utf-8") as samples_handle, corpus_path.open("a", encoding="utf-8") as corpus_handle:
            for prompt_index, prompt in enumerate(prompt_list):
                for sample_index in range(max(1, int(samples_per_prompt))):
                    sample = _generate_teacher_sample(
                        tokenizer=tokenizer,
                        model=model,
                        prompt=prompt,
                        target_device=target_device,
                        chat_template=chat_template,
                        completion_only=completion_only,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        soft_top_k=soft_top_k,
                    )
                    text = sample["text"]
                    if text:
                        train_summary = phase_model.train_text(text, max_tokens=max_train_tokens)
                        round_tokens += int(train_summary["tokens"])
                        corpus_handle.write(text.rstrip() + "\n\n")
                    else:
                        train_summary = {"tokens": 0, "windows": 0, "vocab": len(phase_model.id_to_token)}

                    soft_added = 0
                    soft_candidates = 0
                    for soft_row in sample["soft"]:
                        soft_summary = phase_model.train_next_distribution(
                            soft_row["context"],
                            [(candidate["token"], candidate["prob"]) for candidate in soft_row["candidates"]],
                            weight_scale=soft_weight,
                        )
                        soft_added += int(soft_summary["contexts"])
                        soft_candidates += int(soft_summary["candidates_added"])
                    row = {
                        "round": round_index,
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "prompt": prompt,
                        "text": text,
                        "tokens": train_summary["tokens"],
                        "soft_contexts": soft_added,
                        "soft_candidates": soft_candidates,
                        "soft_preview": sample["soft"][:3],
                    }
                    samples_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    round_samples += 1
                    round_soft_contexts += soft_added
                    round_soft_candidates += soft_candidates

        phase_model.save(phase_model_dir)
        completed_rounds += 1
        total_samples += round_samples
        total_soft_contexts += round_soft_contexts
        total_soft_candidates += round_soft_candidates
        if snapshot_interval > 0 and completed_rounds % int(snapshot_interval) == 0:
            phase_model.save(path / "snapshots" / f"round-{completed_rounds:05d}")

        event = {
            "round": round_index,
            "samples": round_samples,
            "hard_tokens": round_tokens,
            "soft_contexts": round_soft_contexts,
            "soft_candidates": round_soft_candidates,
            "model": phase_model.summary(),
        }
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        payload = {
            "status": "ok",
            "type": "phase-mesh-continuous-transformer-pour",
            "teacher_model": teacher_model,
            "device": target_device,
            "chat_template": chat_template,
            "completion_only": bool(completion_only),
            "prompts": len(prompt_list),
            "rounds_completed": completed_rounds,
            "samples": total_samples,
            "soft_contexts": total_soft_contexts,
            "soft_candidates": total_soft_candidates,
            "resumed_from": resumed_from,
            "teacher_corpus": str(corpus_path),
            "teacher_samples": str(samples_path),
            "pour_events": str(events_path),
            "phase_lm_dir": str(phase_model_dir),
            "phase_config": asdict(phase_model.config),
            "model": phase_model.summary(),
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        round_index += 1
        if sleep_seconds > 0.0:
            time.sleep(float(sleep_seconds))
    return payload


def _resolve_device(device: str, torch_module: Any) -> str:
    requested = str(device).strip().lower()
    if requested and requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) is not None and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_teacher(teacher_model: str, *, device: str, torch_dtype: str = "auto") -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(TRANSFORMERS_IMPORT_ERROR) from exc

    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    load_kwargs = _teacher_load_kwargs(torch_dtype, torch)
    try:
        model = AutoModelForCausalLM.from_pretrained(teacher_model, **load_kwargs)
    except TypeError:
        fallback_kwargs = {}
        if "torch_dtype" in load_kwargs:
            fallback_kwargs["dtype"] = load_kwargs["torch_dtype"]
        model = AutoModelForCausalLM.from_pretrained(teacher_model, **fallback_kwargs)
    target_device = _resolve_device(device, torch)
    if target_device != "cpu":
        model = model.to(target_device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {"tokenizer": tokenizer, "model": model, "device": target_device}


def _teacher_load_kwargs(torch_dtype: str, torch_module: Any) -> dict[str, Any]:
    requested = str(torch_dtype or "").strip().lower()
    if not requested:
        return {}
    if requested == "auto":
        return {"torch_dtype": "auto"}
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    attr = aliases.get(requested, requested)
    dtype = getattr(torch_module, attr, None)
    if dtype is None:
        return {}
    return {"torch_dtype": dtype}


def _generate_teacher_sample(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    target_device: str,
    chat_template: str,
    completion_only: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    soft_top_k: int,
) -> dict[str, Any]:
    import torch

    encoded = _encode_teacher_prompt(tokenizer, prompt, chat_template=chat_template)
    encoded = {key: value.to(target_device) for key, value in encoded.items()}
    input_length = int(encoded["input_ids"].shape[-1])
    with torch.no_grad():
        output = model.generate(
            **encoded,
            do_sample=float(temperature) > 0.0,
            temperature=max(float(temperature), 1e-6),
            top_p=float(top_p),
            top_k=int(top_k),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=int(soft_top_k) > 0,
        )
    sequence = output.sequences[0]
    if completion_only:
        text = tokenizer.decode(sequence[input_length:], skip_special_tokens=True).strip()
    else:
        text = tokenizer.decode(sequence, skip_special_tokens=True).strip()
    soft_rows: list[dict[str, Any]] = []
    if int(soft_top_k) > 0:
        for step, scores in enumerate(output.scores or []):
            logits = scores[0].detach().float()
            values, indices = torch.topk(logits, k=max(1, int(soft_top_k)))
            probs = torch.softmax(values, dim=-1)
            prefix = sequence[: input_length + step]
            context = tokenizer.decode(prefix, skip_special_tokens=True).strip()
            candidates = []
            for token_id, prob in zip(indices.tolist(), probs.tolist()):
                token = tokenizer.decode([int(token_id)], skip_special_tokens=True)
                if token.strip():
                    candidates.append({"token": token, "prob": float(prob)})
            if candidates:
                soft_rows.append({"step": step, "context": context, "candidates": candidates})
    return {"text": text, "soft": soft_rows}


def _encode_teacher_prompt(tokenizer: Any, prompt: str, *, chat_template: str) -> dict[str, Any]:
    mode = str(chat_template).strip().lower()
    use_chat = mode == "always" or (mode == "auto" and bool(getattr(tokenizer, "chat_template", None)))
    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return tokenizer(rendered, return_tensors="pt")
        except Exception:
            if mode == "always":
                raise
    return tokenizer(prompt, return_tensors="pt")
