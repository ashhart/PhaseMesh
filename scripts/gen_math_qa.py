#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def qa_lines(
    count: int,
    *,
    seed: int = 42,
    operations: tuple[str, ...] = ("plus", "minus", "times", "compare"),
) -> list[str]:
    rng = random.Random(seed)
    lines: list[str] = []
    enabled = set(operations)
    for _ in range(max(1, int(count))):
        a = rng.randint(1, 50)
        b = rng.randint(1, 50)
        if "plus" in enabled:
            lines.append(f"question {a} plus {b} answer {a + b}")
        if "minus" in enabled:
            lines.append(f"question {a} minus {b} answer {a - b}")
        if "times" in enabled:
            lines.append(f"question {a} times {b} answer {a * b}")
        if "compare" in enabled:
            lines.append(f"question is {a} greater than {b} answer {'yes' if a > b else 'no'}")
    return lines


def generate_qa_corpus(
    path: Path,
    *,
    count: int = 5_000,
    seed: int = 42,
    operations: tuple[str, ...] = ("plus", "minus", "times", "compare"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(qa_lines(count, seed=seed, operations=operations)) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate question-answer arithmetic mappings for PhaseMesh.")
    parser.add_argument("--out", type=Path, default=Path("runs/qa_corpus.txt"))
    parser.add_argument("--count", type=int, default=5_000, help="Number of random operand pairs; emits four rows per pair.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ops",
        default="plus,minus,times,compare",
        help="Comma-separated operations: plus, minus, times, compare.",
    )
    parser.add_argument("--arithmetic-only", action="store_true", help="Shortcut for --ops plus,minus,times.")
    args = parser.parse_args()

    operations = ("plus", "minus", "times") if args.arithmetic_only else tuple(
        item.strip() for item in args.ops.split(",") if item.strip()
    )
    allowed = {"plus", "minus", "times", "compare"}
    invalid = sorted(set(operations) - allowed)
    if invalid:
        raise SystemExit(f"unsupported operation(s): {', '.join(invalid)}")
    generate_qa_corpus(args.out, count=args.count, seed=args.seed, operations=operations)
    print(f"Generated {args.count * len(operations)} QA rows at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
