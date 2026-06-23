#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FUNCTION_TASKS = [
    ("add two numbers", ["add", "sum", "return"]),
    ("subtract two numbers", ["subtract", "return"]),
    ("multiply two numbers", ["multiply", "return"]),
    ("divide two numbers safely and return None for division by zero", ["divide", "zero", "none"]),
    ("reverse a string", ["reverse", "string"]),
    ("check whether a string is a palindrome", ["palindrome", "reverse"]),
    ("count words in a string", ["count", "word"]),
    ("count vowels in a string", ["vowel", "count"]),
    ("filter even numbers from a list", ["even", "filter"]),
    ("filter odd numbers from a list", ["odd", "filter"]),
    ("return the maximum value in a list without using max()", ["maximum", "list"]),
    ("return the minimum value in a list without using min()", ["minimum", "list"]),
    ("remove duplicates from a list while preserving order", ["duplicate", "order"]),
    ("flatten a list of lists by one level", ["flatten", "list"]),
    ("merge two dictionaries", ["dictionary", "merge"]),
    ("sort a list of dictionaries by the name field", ["sort", "dictionary"]),
    ("group records by a category key", ["group", "category"]),
    ("parse JSON text safely and return None on invalid JSON", ["json", "invalid", "none"]),
    ("read all lines from a text file", ["read", "line", "file"]),
    ("write text to a file using a context manager", ["write", "file", "context"]),
    ("retry a flaky function three times before raising", ["retry", "raise"]),
    ("chunk a list into batches of size n", ["chunk", "batch"]),
    ("compute factorial iteratively", ["factorial", "iterative"]),
    ("compute fibonacci numbers iteratively", ["fibonacci", "iterative"]),
    ("perform binary search on a sorted list", ["binary", "search"]),
    ("validate an email address with simple checks", ["email", "validate"]),
    ("turn a sentence into a URL slug", ["slug", "sentence"]),
    ("transpose a matrix represented as nested lists", ["transpose", "matrix"]),
    ("calculate a moving average over a list of numbers", ["moving", "average"]),
    ("find the first duplicate item in a list", ["duplicate", "first"]),
]


BUG_TASKS = [
    ("Find the bug in this Python function: def add(a, b): return a - b", ["return a + b"], ["return a - b"]),
    ("Find the bug in this Python function: def is_even(n): return n % 2 == 1", ["== 0", "even"], ["== 1"]),
    ("Find the bug in this Python function: def first(items): return items[1]", ["items[0]", "first"], ["items[1]"]),
    ("Find the bug in this Python function: def append_item(x, items=[]): items.append(x); return items", ["none", "mutable"], ["items=[]"]),
    ("Find the bug in this Python function: def percent(x): return x / 1000", ["100", "percent"], ["1000"]),
    ("Find the bug in this Python function: def area(width, height): return width + height", ["*", "multiply"], ["+ height"]),
    ("Find the bug in this Python function: def last(items): return items[len(items)]", ["len(items) - 1", "index"], ["items[len(items)]"]),
    ("Find the bug in this Python function: def average(nums): return sum(nums) / len(num)", ["nums", "NameError"], ["len(num)"]),
    ("Find the bug in this Python function: def greet(name): return 'Hello' + name when name is an int", ["str", "TypeError"], []),
    ("Find the bug in this Python function: async def fetch(client): return client.get('/x')", ["await", "async"], []),
]


EXPLAIN_TASKS = [
    ("Explain how to debug a Python TypeError.", ["typeerror", "type"]),
    ("Explain how to debug a Python NameError.", ["nameerror", "defined"]),
    ("Explain how to profile slow Python code.", ["profile", "cprofile"]),
    ("Explain how to use cProfile to find a bottleneck.", ["cprofile", "bottleneck"]),
    ("Explain how to inspect a failing pytest test.", ["pytest", "failing"]),
    ("Explain how to reduce memory use in a Python script.", ["memory", "profile"]),
    ("Explain how to parse JSON safely in Python.", ["json", "exception"]),
    ("Explain when to use a dataclass in Python.", ["dataclass", "class"]),
    ("Explain how a tokenizer turns text into model inputs.", ["tokenizer", "token"]),
    ("Explain what a model checkpoint contains.", ["checkpoint", "weight"]),
    ("Explain the difference between retrieval and generation.", ["retrieval", "generation"]),
    ("Explain what a vector database is useful for.", ["vector", "database"]),
    ("Explain what a cache hit rate means.", ["cache", "hit"]),
    ("Explain how to read a Python stack trace.", ["stack", "trace"]),
    ("Explain why a failing unit test is useful evidence.", ["test", "evidence"]),
]


REASONING_TASKS = [
    ("Solve step by step: If a server handles 12 requests per second for 5 minutes, how many requests is that?", ["3600"]),
    ("Solve step by step: A script processes 45 files per minute for 8 minutes. How many files is that?", ["360"]),
    ("Solve step by step: If a cache hit rate is 80 percent on 250 requests, how many misses are there?", ["50"]),
    ("Solve step by step: A build takes 14 seconds and runs 30 times. How many seconds total?", ["420"]),
    ("Solve step by step: A job uses 3 workers for 40 minutes each. How many worker-minutes is that?", ["120"]),
    ("Solve step by step: A queue has 180 tasks and finishes 30 tasks per minute. How many minutes?", ["6"]),
    ("Solve step by step: A service has 2 percent errors over 500 requests. How many errors?", ["10"]),
    ("Solve step by step: A model emits 24 tokens per second for 15 seconds. How many tokens?", ["360"]),
    ("Solve step by step: If 7 tests fail out of 140, what percent failed?", ["5"]),
    ("Solve step by step: If latency drops from 250 ms to 100 ms, what is the speedup ratio?", ["2.5"]),
]


PARAPHRASE_PREFIXES = [
    "Write a concise answer for this task:",
    "Give a practical Python answer:",
    "Answer like a coding assistant:",
    "Provide the fix or implementation:",
]


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = " ".join(item.strip().split()).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def build_prompts() -> list[str]:
    prompts: list[str] = []
    for task, _ in FUNCTION_TASKS:
        prompts.extend([
            f"Write a Python function to {task}.",
            f"Create a small Python helper that can {task}.",
            f"Implement and briefly explain Python code to {task}.",
        ])
    for task, _, _ in BUG_TASKS:
        prompts.extend([task, f"Explain the bug and corrected code: {task}"])
    for task, _ in EXPLAIN_TASKS:
        prompts.extend([task, f"{PARAPHRASE_PREFIXES[len(prompts) % len(PARAPHRASE_PREFIXES)]} {task}"])
    for task, _ in REASONING_TASKS:
        prompts.extend([task, task.replace("Solve step by step:", "Compute carefully:")])
    return _dedupe(prompts)


def build_eval_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task, must in FUNCTION_TASKS:
        rows.append({
            "category": "function",
            "prompt": f"Can you write Python code to {task}?",
            "must_contain_any": must,
            "must_not_contain_any": [],
        })
    for task, must, must_not in BUG_TASKS:
        rows.append({
            "category": "bug",
            "prompt": task,
            "must_contain_any": must,
            "must_not_contain_any": must_not,
        })
    for task, must in EXPLAIN_TASKS:
        rows.append({
            "category": "explain",
            "prompt": task.replace("Explain", "Briefly explain"),
            "must_contain_any": must,
            "must_not_contain_any": [],
        })
    for task, must in REASONING_TASKS:
        rows.append({
            "category": "reasoning",
            "prompt": task,
            "must_contain_any": must,
            "must_not_contain_any": [],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PhaseChat coding/reasoning coverage prompts and eval rows.")
    parser.add_argument("--prompts-out", type=Path, default=Path("examples/qwen_coding_reasoning_prompts_beast.txt"))
    parser.add_argument("--eval-out", type=Path, default=Path("examples/qwen_coding_reasoning_eval_beast.jsonl"))
    args = parser.parse_args()

    prompts = build_prompts()
    rows = build_eval_rows()
    args.prompts_out.parent.mkdir(parents=True, exist_ok=True)
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    args.prompts_out.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    args.eval_out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "prompts": len(prompts),
        "eval_rows": len(rows),
        "prompts_out": str(args.prompts_out),
        "eval_out": str(args.eval_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
