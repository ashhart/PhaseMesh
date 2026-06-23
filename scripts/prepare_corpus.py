#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a line-oriented corpus for phase-mesh model training.")
    parser.add_argument("source", help="Local file path or http(s) URL. Use '-' for stdin.")
    parser.add_argument("--out", type=Path, required=True, help="Output text file.")
    parser.add_argument("--max-lines", type=int, default=100_000, help="Maximum input lines to read.")
    parser.add_argument("--max-output-lines", type=int, default=None, help="Maximum non-empty output chunks to write.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum whitespace tokens per output chunk.")
    parser.add_argument(
        "--jsonl-fields",
        default="text,content,code,completion,prompt",
        help="Comma-separated JSONL fields to try before falling back to raw line text.",
    )
    args = parser.parse_args(argv)
    fields = [item.strip() for item in args.jsonl_fields.split(",") if item.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(iter_lines(args.source), start=1):
            if args.max_lines is not None and line_number > args.max_lines:
                break
            text = extract_text(line, fields)
            for chunk in chunk_text(text, max_tokens=args.max_tokens):
                output.write(chunk + "\n")
                written += 1
                if args.max_output_lines is not None and written >= args.max_output_lines:
                    print(json.dumps({"source_lines": line_number, "output_lines": written, "out": str(args.out)}, indent=2))
                    return 0

    print(json.dumps({"source_lines": min(line_number if "line_number" in locals() else 0, args.max_lines), "output_lines": written, "out": str(args.out)}, indent=2))
    return 0


def iter_lines(source: str) -> Iterator[str]:
    if source == "-":
        for line in sys.stdin:
            yield line
        return
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "phase-mesh-corpus-prep/0.1"})
        with urllib.request.urlopen(request) as response:  # noqa: S310 - user-supplied corpus URL
            for raw in response:
                yield raw.decode("utf-8", errors="replace")
        return
    with Path(source).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line


def extract_text(line: str, fields: Iterable[str]) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("{"):
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        for field in fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                return value
        parts = [value for value in row.values() if isinstance(value, str) and value.strip()]
        if parts:
            return "\n".join(parts)
    return stripped


def chunk_text(text: str, *, max_tokens: int) -> Iterator[str]:
    tokens = text.split()
    if not tokens:
        return
    width = max(8, int(max_tokens))
    for index in range(0, len(tokens), width):
        chunk = " ".join(tokens[index : index + width]).strip()
        if chunk:
            yield chunk


if __name__ == "__main__":
    raise SystemExit(main())
