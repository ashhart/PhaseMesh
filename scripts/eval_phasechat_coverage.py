#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_mesh.chat_lm import PhaseChatModel


def _contains_any(text: str, needles: list[str]) -> bool:
    if not needles:
        return True
    lowered = text.lower()
    return any(str(needle).lower() in lowered for needle in needles)


def _contains_none(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return not any(str(needle).lower() in lowered for needle in needles if str(needle).strip())


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a PhaseChat model against a coverage JSONL file.")
    parser.add_argument("eval_jsonl", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--allow-fallback", action="store_true", help="Generate from the compact fallback LM instead of fast abstaining.")
    args = parser.parse_args()

    rows = _load_rows(args.eval_jsonl)
    model = PhaseChatModel.load(args.model_dir)
    results = []
    latencies = []
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "retrieved": 0, "passed": 0, "fallback": 0})

    for row in rows:
        prompt = str(row["prompt"])
        started = time.perf_counter()
        answer = model.answer(prompt, top_k=args.top_k, max_tokens=args.max_tokens, allow_fallback=args.allow_fallback)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)
        text = str(answer.get("completion") or answer.get("text") or "")
        mode = str(answer.get("mode", ""))
        category = str(row.get("category", "uncategorized"))
        must_not = [] if category == "bug" else list(row.get("must_not_contain_any", []))
        passed = _contains_any(text, list(row.get("must_contain_any", []))) and _contains_none(text, must_not)
        by_category[category]["rows"] += 1
        by_category[category]["passed"] += int(passed)
        by_category[category]["retrieved"] += int(mode == "phase-chat-retrieval")
        by_category[category]["fallback"] += int(mode == "phase-chat-fallback")
        by_category[category].setdefault("abstained", 0)
        by_category[category]["abstained"] += int(mode == "phase-chat-abstain")
        results.append({
            "category": category,
            "prompt": prompt,
            "mode": mode,
            "passed": bool(passed),
            "latency_ms": round(elapsed_ms, 4),
            "confidence": answer.get("confidence", {}),
            "answer_preview": text[:280],
        })

    summary = {
        "status": "ok",
        "model_dir": str(args.model_dir),
        "eval_jsonl": str(args.eval_jsonl),
        "rows": len(rows),
        "retrieval_rate": sum(item["mode"] == "phase-chat-retrieval" for item in results) / max(1, len(results)),
        "abstain_rate": sum(item["mode"] == "phase-chat-abstain" for item in results) / max(1, len(results)),
        "pass_rate": sum(item["passed"] for item in results) / max(1, len(results)),
        "median_latency_ms": median(latencies) if latencies else 0.0,
        "by_category": {
            key: {
                **value,
                "retrieval_rate": value["retrieved"] / max(1, value["rows"]),
                "pass_rate": value["passed"] / max(1, value["rows"]),
            }
            for key, value in sorted(by_category.items())
        },
    }
    payload = {"summary": summary, "rows": results}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if summary["pass_rate"] >= 0.8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
