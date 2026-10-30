#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_mesh.distill import read_distill_prompts


def _render_prompt(tokenizer: object, prompt: str, chat_template: str) -> str:
    mode = str(chat_template).strip().lower()
    use_chat = mode == "always" or (mode == "auto" and bool(getattr(tokenizer, "chat_template", None)))
    if use_chat:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            if mode == "always":
                raise
    return prompt


def _load_prompt_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            teacher_prompt = str(row.get("teacher_prompt", prompt)).strip()
            if prompt and teacher_prompt:
                rows.append({"prompt": prompt, "teacher_prompt": teacher_prompt})
        return rows
    return [{"prompt": prompt, "teacher_prompt": prompt} for prompt in read_distill_prompts(path)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-pour teacher prompt/completion traces for PhaseChat retrieval.")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--chat-template", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except Exception as exc:
        raise RuntimeError("Install torch and transformers to run teacher pouring.") from exc

    set_seed(int(args.seed))
    dtype = getattr(torch, str(args.torch_dtype), None) if str(args.torch_dtype).lower() != "auto" else "auto"
    load_kwargs = {"torch_dtype": dtype} if dtype is not None else {"torch_dtype": "auto"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **load_kwargs)
    except TypeError:
        fallback_kwargs = {"dtype": dtype} if dtype is not None and dtype != "auto" else {}
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **fallback_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.device != "cpu":
        model = model.to(args.device)
    model.eval()

    prompt_rows = _load_prompt_rows(args.prompts)
    args.out.mkdir(parents=True, exist_ok=True)
    samples_path = args.out / "teacher_samples.jsonl"
    corpus_path = args.out / "teacher_corpus.txt"
    summary_path = args.out / "summary.json"
    started = time.perf_counter()
    written = 0
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    with samples_path.open("w", encoding="utf-8") as samples_handle, corpus_path.open("w", encoding="utf-8") as corpus_handle:
        for batch_start in range(0, len(prompt_rows), max(1, int(args.batch_size))):
            batch_rows = prompt_rows[batch_start : batch_start + max(1, int(args.batch_size))]
            rendered = [_render_prompt(tokenizer, row["teacher_prompt"], args.chat_template) for row in batch_rows]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    do_sample=float(args.temperature) > 0.0,
                    temperature=max(float(args.temperature), 1e-6),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    max_new_tokens=int(args.max_new_tokens),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for offset, row_data in enumerate(batch_rows):
                text = tokenizer.decode(output[offset][input_width:], skip_special_tokens=True).strip()
                row = {
                    "prompt_index": batch_start + offset,
                    "sample_index": 0,
                    "prompt": row_data["prompt"],
                    "teacher_prompt": row_data["teacher_prompt"],
                    "text": text,
                    "teacher_model": args.teacher_model,
                    "batch_size": int(args.batch_size),
                }
                samples_handle.write(json.dumps(row, sort_keys=True) + "\n")
                corpus_handle.write(text.rstrip() + "\n\n")
                written += 1
            samples_handle.flush()
            corpus_handle.flush()
            elapsed = time.perf_counter() - started
            print(json.dumps({"written": written, "total": len(prompt_rows), "elapsed_s": round(elapsed, 3)}), flush=True)

    payload = {
        "status": "ok",
        "type": "phase-chat-batched-teacher-pour",
        "teacher_model": args.teacher_model,
        "prompts": len(prompt_rows),
        "samples": written,
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "teacher_samples": str(samples_path),
        "teacher_corpus": str(corpus_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
