#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path


OPS = ("+", "-", "*", "/", "%")
CMPS = (">", "<", "==", "!=")


def word(rng: random.Random, length: int) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def arithmetic_case(rng: random.Random) -> tuple[int, str, int, int]:
    op = rng.choice(OPS)
    if op == "/":
        b = rng.randint(1, 20)
        result = rng.randint(0, 20)
        a = b * result
        return a, op, b, result
    if op == "%":
        b = rng.randint(1, 20)
        a = rng.randint(0, 200)
        return a, op, b, a % b
    a = rng.randint(0, 100)
    b = rng.randint(0, 100)
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    else:
        result = a * b
    return a, op, b, result


def comparison_result(x: int, op: str, y: int) -> bool:
    if op == ">":
        return x > y
    if op == "<":
        return x < y
    if op == "==":
        return x == y
    if op == "!=":
        return x != y
    raise ValueError(f"unsupported comparison operator: {op}")


def generate_line(rng: random.Random) -> str:
    kind = rng.randrange(6)
    if kind in (0, 1):
        a, op, b, result = arithmetic_case(rng)
        if kind == 0:
            return f"calc {a} {op} {b} = {result}"
        return f"print ( {a} {op} {b} ) = {result}"

    if kind == 2:
        name = word(rng, 3)
        value = rng.randint(0, 20)
        return f"{name} = {value}"

    if kind == 3:
        x, y = rng.randint(0, 20), rng.randint(0, 20)
        op = rng.choice(CMPS)
        result = "true" if comparison_result(x, op, y) else "false"
        return f"if {x} {op} {y} then {result}"

    if kind == 4:
        fn = word(rng, 4)
        op = rng.choice(("+", "-", "*"))
        return f"def {fn} ( x , y ) return x {op} y"

    a, op, b, result = arithmetic_case(rng)
    return f"return {a} {op} {b} = {result}"


def generate_logic_corpus(path: Path, *, lines: int = 10_000, seed: int = 42) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(max(1, int(lines))):
            handle.write(generate_line(rng) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a compact logic-first corpus for PhaseMesh.")
    parser.add_argument("--out", type=Path, default=Path("runs/logic_corpus.txt"))
    parser.add_argument("--lines", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_logic_corpus(args.out, lines=args.lines, seed=args.seed)
    print(f"Generated {args.lines} logic lines at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
