#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_qa_row(row: str) -> tuple[str, str] | None:
    text = " ".join(row.strip().split())
    if not text:
        return None
    tokens = text.split()
    lowered = [token.lower().rstrip(":") for token in tokens]
    if "answer" not in lowered:
        return None
    answer_index = lowered.index("answer")
    if answer_index >= len(tokens) - 1:
        return None
    prompt = " ".join(tokens[: answer_index + 1])
    answer = " ".join(tokens[answer_index + 1 :])
    return prompt, answer


def load_qa_rows(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    rows: list[tuple[str, str]] = []
    structural_rows = load_structural_rows(text)
    if structural_rows:
        return structural_rows
    for chunk in text.splitlines():
        parsed = parse_qa_row(chunk)
        if parsed is not None:
            rows.append(parsed)
    if rows:
        return rows

    for chunk in text.split("\n\n"):
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        parsed = parse_qa_row(" ".join(lines))
        if parsed is not None:
            rows.append(parsed)
    return rows


def load_structural_rows(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if {"seq_a", "seq_b", "target"} <= set(payload):
            target = str(payload["target"]).strip()
            rows.append((f"question: {str(payload['seq_a']).strip()}\nanswer:", target))
            rows.append((f"question: {str(payload['seq_b']).strip()}\nanswer:", target))
    return rows


def negative_candidates(answer: str, *, k_neg: int = 12) -> list[str]:
    answer = answer.strip()
    if answer.lower() in {"yes", "no"}:
        return ["no" if answer.lower() == "yes" else "yes"]
    try:
        value = int(answer)
    except ValueError:
        return []

    negatives: list[str] = []
    seen = {answer}
    common = ("10", "20", "0", "1", "2", "3", "5", "7", "8", "9", "12", "15", "16", "17", "18", "19", "22", "27", "30", "42")
    for candidate in common:
        if candidate not in seen:
            negatives.append(candidate)
            seen.add(candidate)
        if len(negatives) >= k_neg:
            return negatives
    offsets = (-10, -5, -2, -1, 1, 2, 5, 10)
    for offset in offsets:
        candidate = str(value + offset)
        if candidate not in seen:
            negatives.append(candidate)
            seen.add(candidate)
        if len(negatives) >= k_neg:
            break
    return negatives


def ranking_rows(qa_path: Path, *, k_neg: int = 4, seed: int = 42) -> list[dict[str, str | int]]:
    rng = random.Random(seed)
    rows: list[dict[str, str | int]] = []
    for prompt, answer in load_qa_rows(qa_path):
        rows.append({"prompt": prompt, "candidate": answer, "label": 1})
        for candidate in negative_candidates(answer, k_neg=k_neg):
            rows.append({"prompt": prompt, "candidate": candidate, "label": 0})
    rng.shuffle(rows)
    return rows


def generate_ranking_data(qa_path: Path, out_path: Path, *, k_neg: int = 4, seed: int = 42) -> int:
    rows = ranking_rows(qa_path, k_neg=k_neg, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PhaseMesh candidate reranking pairs from QA rows.")
    parser.add_argument("qa_file", type=Path, nargs="?", default=Path("runs/qa_corpus.txt"))
    parser.add_argument("--out", type=Path, default=Path("runs/ranking_data.jsonl"))
    parser.add_argument("--k-neg", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count = generate_ranking_data(args.qa_file, args.out, k_neg=args.k_neg, seed=args.seed)
    print(f"Generated {count} ranking pairs at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
