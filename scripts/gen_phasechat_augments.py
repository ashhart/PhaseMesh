#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen_phasechat_coverage import BUG_TASKS, REASONING_TASKS


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final-first augmentation prompts for PhaseChat.")
    parser.add_argument("--out", type=Path, default=Path("examples/qwen_coding_reasoning_augments_beast.jsonl"))
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for prompt, _, _ in BUG_TASKS:
        rows.append({
            "category": "bug",
            "prompt": prompt,
            "teacher_prompt": (
                "Answer in at most 5 lines. First show the corrected Python line or function. "
                "Then give one concise sentence explaining the bug. "
                f"Task: {prompt}"
            ),
        })
    for prompt, _ in REASONING_TASKS:
        rows.append({
            "category": "reasoning",
            "prompt": prompt,
            "teacher_prompt": (
                "Answer in at most 4 lines. Put the final numeric answer on the first line. "
                "Then show the calculation in one concise line. "
                f"Task: {prompt}"
            ),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(rows), "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
