#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def equivalence_rows(count: int, *, seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for _ in range(max(1, int(count))):
        a = rng.randint(1, 99)
        b = rng.randint(1, 99)
        rows.append({"seq_a": f"{a} plus {b}", "seq_b": f"{b} plus {a}", "target": str(a + b)})
        rows.append({"seq_a": f"{a} times {b}", "seq_b": f"{b} times {a}", "target": str(a * b)})

        low, high = sorted((a, b))
        if low != high:
            rows.append({"seq_a": f"is {high} greater than {low}", "seq_b": f"is {low} less than {high}", "target": "yes"})
            rows.append({"seq_a": f"is {low} greater than {high}", "seq_b": f"is {high} less than {low}", "target": "no"})
        else:
            rows.append({"seq_a": f"is {a} greater than {b}", "seq_b": f"is {b} less than {a}", "target": "no"})
    return rows


def generate_equivalence_data(path: Path, *, count: int = 20_000, seed: int = 42) -> int:
    rows = equivalence_rows(count, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PhaseMesh equivalence triples for structural basin collapse.")
    parser.add_argument("--out", type=Path, default=Path("runs/equiv_data.jsonl"))
    parser.add_argument("--count", type=int, default=20_000, help="Number of random operand pairs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_equivalence_data(args.out, count=args.count, seed=args.seed)
    print(f"Generated {rows} equivalence rows at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
