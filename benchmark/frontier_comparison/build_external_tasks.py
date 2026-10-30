from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build frontier comparison JSONL tasks from GSM8K and LongBench.")
    parser.add_argument("--out", type=Path, default=Path("benchmark/frontier_comparison/out/external_tasks.jsonl"))
    parser.add_argument("--gsm8k-count", type=int, default=8)
    parser.add_argument("--longbench-count", type=int, default=4)
    parser.add_argument("--longbench-subset", default="narrativeqa")
    args = parser.parse_args(argv)
    payload = build_tasks(
        gsm8k_count=args.gsm8k_count,
        longbench_count=args.longbench_count,
        longbench_subset=args.longbench_subset,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for task in payload["tasks"]:
            handle.write(json.dumps(task, sort_keys=True) + "\n")
    payload["output_path"] = str(args.out)
    sidecar = args.out.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def build_tasks(*, gsm8k_count: int, longbench_count: int, longbench_subset: str) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    if gsm8k_count > 0:
        try:
            from datasets import load_dataset

            dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{gsm8k_count}]")
            for index, row in enumerate(dataset):
                question = str(row["question"])
                answer = extract_gsm8k_answer(str(row["answer"]))
                tasks.append(
                    {
                        "id": f"gsm8k-{index:04d}",
                        "suite": "gsm8k",
                        "kind": "arithmetic",
                        "prompt": f"{question}\nReturn only the final number.",
                        "expected": answer,
                        "difficulty": "hard",
                        "token_count": len(question.split()),
                        "metadata": {"source": "openai/gsm8k", "split": "test"},
                    }
                )
            sources.append({"suite": "gsm8k", "status": "loaded", "count": len(dataset)})
        except Exception as exc:
            sources.append({"suite": "gsm8k", "status": "skipped", "error": str(exc)})

    if longbench_count > 0:
        old_error: str | None = None
        try:
            from datasets import load_dataset

            dataset = load_dataset("THUDM/LongBench", longbench_subset, split=f"test[:{longbench_count}]")
            source = "THUDM/LongBench"
            split = "test"
            suite = f"longbench:{longbench_subset}"
        except Exception as exc:
            old_error = str(exc)
            try:
                dataset = load_dataset("zai-org/LongBench-v2", split=f"train[:{longbench_count}]")
                source = "zai-org/LongBench-v2"
                split = "train"
                suite = "longbench-v2"
            except Exception as fallback_exc:
                sources.append(
                    {
                        "suite": f"longbench:{longbench_subset}",
                        "status": "skipped",
                        "error": str(fallback_exc),
                        "old_loader_error": old_error,
                    }
                )
                dataset = None

        if dataset is not None:
            for index, row in enumerate(dataset):
                prompt = longbench_prompt(row)
                expected = longbench_expected(row)
                tasks.append(
                    {
                        "id": f"{suite}-{index:04d}",
                        "suite": suite,
                        "kind": "context",
                        "prompt": prompt,
                        "expected": expected,
                        "difficulty": str(row.get("difficulty") or "hard"),
                        "token_count": len(prompt.split()),
                        "metadata": {
                            "source": source,
                            "subset": None if source.endswith("LongBench-v2") else longbench_subset,
                            "split": split,
                            "old_loader_error": old_error,
                        },
                    }
                )
            sources.append({"suite": suite, "status": "loaded", "count": len(dataset), "source": source})

    return {
        "tasks": tasks,
        "sources": sources,
        "notes": [
            "GSM8K rows are scored as numeric arithmetic tasks by bench.frontier_compare.",
            "LongBench rows are scored as context substring tasks when an expected answer is available.",
        ],
    }


def extract_gsm8k_answer(answer: str) -> str:
    match = ANSWER_RE.search(answer)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer)
    return numbers[-1].replace(",", "") if numbers else answer.strip()


def longbench_prompt(row: dict[str, Any]) -> str:
    context = str(row.get("context") or row.get("input") or "")
    question = str(row.get("question") or row.get("query") or row.get("input") or "")
    choices = []
    for label in ["A", "B", "C", "D"]:
        value = row.get(f"choice_{label}")
        if value:
            choices.append(f"{label}. {value}")
    choice_text = "\n".join(choices)
    if context and question and question not in context:
        if choice_text:
            return f"{context}\n\nQuestion: {question}\n{choice_text}\nAnswer with the single best choice letter."
        return f"{context}\n\nQuestion: {question}\nAnswer with the shortest exact answer."
    if choice_text:
        return f"{context or question}\n{choice_text}\nAnswer with the single best choice letter."
    return f"{context or question}\nAnswer with the shortest exact answer."


def longbench_expected(row: dict[str, Any]) -> str | None:
    answers = row.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0])
    answer = row.get("answer")
    if answer is None:
        return None
    return str(answer)


if __name__ == "__main__":
    raise SystemExit(main())
