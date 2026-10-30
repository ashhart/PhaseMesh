#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase_mesh.trainer import repulsion_rows_from_any_row  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def write_repulsion_data(source: Path, out: Path, *, max_rows: int | None = None) -> int:
    source_rows = load_jsonl(source)
    if not source_rows:
        raise FileNotFoundError(f"No source rows found at {source}")
    output_rows: list[dict[str, object]] = []
    for row in source_rows:
        for expanded in repulsion_rows_from_any_row(row):
            output_rows.append(expanded)
            if max_rows is not None and len(output_rows) >= max_rows:
                break
        if max_rows is not None and len(output_rows) >= max_rows:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    return len(output_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PhaseMesh result-localization repulsion rows.")
    parser.add_argument("--source", type=Path, default=Path("runs/equiv_data.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/repulsion_data.jsonl"))
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    count = write_repulsion_data(args.source, args.out, max_rows=args.max_rows)
    print(f"Generated {count} repulsion rows at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
