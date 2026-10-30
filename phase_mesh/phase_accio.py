from __future__ import annotations

import hashlib
import html
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_:-]+")
KEY_RE = re.compile(r"^k_[a-f0-9]{10}$", re.IGNORECASE)
VALUE_RE = re.compile(r"^v_[a-f0-9]{16}$", re.IGNORECASE)
TWO_PI = 2.0 * np.pi

NOISE_WORDS = (
    "audit", "memo", "ledger", "operator", "handoff", "north", "amber", "buffer",
    "signal", "transit", "packet", "review", "archive", "control", "routing",
    "sample", "bay", "notice", "trace", "field", "dispatch", "window", "status",
    "silent", "later", "because", "while", "before", "after", "under", "between",
    "ordinary", "plain", "working", "scratch", "daily", "local", "manual",
)

NATURAL_TEMPLATES = (
    "During the audit handoff the identifier {key} carried marker {value} after the operator review closed.",
    "The routing memo says that case {key} should resolve through the private marker {value} before dispatch.",
    "A ledger note recorded identifier {key} beside recovery marker {value} while the surrounding entries stayed noisy.",
    "In the archive paragraph the compact handle {key} was paired with marker {value} for later retrieval.",
    "The control-room note linked handle {key} to marker {value} after duplicate traces were rejected.",
    "For the overnight packet the lookup handle {key} carried answer marker {value} in the middle of unrelated prose.",
)

RECORD_TEMPLATE = "record {key} value {value}"


@dataclass(frozen=True)
class PhaseAccioRecord:
    key: str
    value: str


@dataclass(frozen=True)
class PhaseAccioQuery:
    key: str
    expected: str
    candidates: list[str]


class CandidateRanker(Protocol):
    @property
    def memory_bytes(self) -> int:
        ...

    def rank(self, key: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        ...


class PhaseAccioSketch:
    """Fixed-size complex phase-binding sketch for key/candidate retrieval."""

    def __init__(
        self,
        *,
        grid_size: int = 128,
        slots_per_symbol: int = 24,
        seed: int = 7,
        pin_strength: float = 0.25,
        filler_noise: float = 0.001,
        filler_stride: int = 32,
        proximity_window: int = 12,
        salt: str = "phase-accio",
        score_salt: str | None = None,
    ) -> None:
        self.grid_size = int(grid_size)
        self.slots_per_symbol = int(slots_per_symbol)
        self.seed = int(seed)
        self.pin_strength = float(pin_strength)
        self.filler_noise = float(filler_noise)
        self.filler_stride = max(1, int(filler_stride))
        self.proximity_window = max(1, int(proximity_window))
        self.salt = str(salt)
        self.score_salt = str(score_salt) if score_salt is not None else self.salt
        self.size = self.grid_size * self.grid_size
        self.memory = np.zeros(self.size, dtype=np.complex64)
        self.records_seen = 0
        self.filler_seen = 0

    @property
    def memory_bytes(self) -> int:
        return int(self.memory.nbytes)

    def ingest(self, text: str) -> None:
        tokens = tokenize(text)
        self.ingest_tokens(tokens)

    def ingest_tokens(self, tokens: Sequence[str], bindings: Sequence[tuple[str, str]] | None = None) -> None:
        if bindings is None:
            bindings = extract_proximity_bindings(tokens, self.proximity_window)
        for key, value in bindings:
            self.bind(key, value)
        if self.filler_noise > 0.0:
            for index, token in enumerate(tokens):
                if index % self.filler_stride != 0 or is_key_token(token) or is_value_token(token):
                    continue
                self.add_filler(token)
        self.normalize()

    def bind(self, key: str, value: str) -> None:
        if self.pin_strength <= 0.0:
            return
        binding = self.binding_vector(key, value, salt=self.salt)
        self.memory += np.asarray(self.pin_strength * binding, dtype=np.complex64)
        self.records_seen += 1

    def add_filler(self, token: str) -> None:
        if self.filler_noise <= 0.0:
            return
        self.add_symbol_in_place(f"filler:{token}", self.filler_noise, salt=self.salt)
        self.filler_seen += 1

    def normalize(self) -> None:
        norm = float(np.linalg.norm(self.memory))
        if norm > 0.0:
            self.memory = np.asarray(self.memory / norm, dtype=np.complex64)

    def score(self, key: str, candidate: str) -> float:
        probe = self.binding_vector(key.lower(), candidate.lower(), salt=self.score_salt)
        denominator = float(np.linalg.norm(self.memory) * np.linalg.norm(probe))
        if denominator <= 0.0:
            return 0.0
        return float(np.real(np.vdot(probe, self.memory)) / denominator)

    def rank(self, key: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        rows = [
            {"candidate": str(candidate), "score": self.score(str(key), str(candidate))}
            for candidate in candidates
        ]
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows

    def binding_vector(self, key: str, value: str, *, salt: str | None = None) -> np.ndarray:
        return self.symbol_vector(f"bind:{key}:{value}", salt=salt or self.salt)

    def add_symbol_in_place(self, symbol: str, scale: float, *, salt: str | None = None) -> None:
        entries = self.symbol_entries(symbol, salt=salt or self.salt)
        norm = np.sqrt(sum(float(np.abs(value) ** 2) for value in entries.values()))
        if norm <= 0.0:
            return
        for index, value in entries.items():
            self.memory[index] += np.complex64((float(scale) * value) / norm)

    def symbol_vector(self, symbol: str, *, salt: str | None = None) -> np.ndarray:
        vector = np.zeros(self.size, dtype=np.complex64)
        entries = self.symbol_entries(symbol, salt=salt or self.salt)
        for index, value in entries.items():
            vector[index] += np.complex64(value)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = np.asarray(vector / norm, dtype=np.complex64)
        return vector

    def symbol_entries(self, symbol: str, *, salt: str | None = None) -> dict[int, complex]:
        entries: dict[int, complex] = {}
        digest = _digest(f"{salt or self.salt}:{self.seed}:{symbol}")
        cursor = 0
        for _slot in range(self.slots_per_symbol):
            if cursor + 8 > len(digest):
                digest = _digest(digest.hex())
                cursor = 0
            index = int.from_bytes(digest[cursor:cursor + 4], "big") % self.size
            phase_byte = digest[cursor + 4]
            phase = (phase_byte / 255.0) * TWO_PI
            entries[index] = entries.get(index, 0.0 + 0.0j) + complex(np.cos(phase), np.sin(phase))
            cursor += 5
        return entries


class HashMapBaseline:
    """Exact proximity-extracted map. Useful as an upper-bound baseline."""

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}

    @property
    def memory_bytes(self) -> int:
        return sum(len(key) + len(value) + 96 for key, value in self.mapping.items())

    def ingest_bindings(self, bindings: Sequence[tuple[str, str]]) -> None:
        for key, value in bindings:
            self.mapping[str(key)] = str(value)

    def score(self, key: str, candidate: str) -> float:
        return 1.0 if self.mapping.get(str(key)) == str(candidate) else 0.0

    def rank(self, key: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        rows = [{"candidate": str(candidate), "score": self.score(str(key), str(candidate))} for candidate in candidates]
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows


class BloomPairBaseline:
    """Classical bit-sketch baseline over key/value pair membership."""

    def __init__(self, *, bits: int = 131072, slots_per_pair: int = 12, seed: int = 7) -> None:
        self.bits = max(8, int(bits))
        self.slots_per_pair = max(1, int(slots_per_pair))
        self.seed = int(seed)
        self.array = np.zeros(self.bits, dtype=np.bool_)

    @property
    def memory_bytes(self) -> int:
        return int(np.ceil(self.bits / 8))

    def ingest_bindings(self, bindings: Sequence[tuple[str, str]]) -> None:
        for key, value in bindings:
            for index in self.indices(key, value):
                self.array[index] = True

    def indices(self, key: str, value: str) -> list[int]:
        digest = _digest(f"bloom:{self.seed}:{key}:{value}")
        indices = []
        cursor = 0
        for _slot in range(self.slots_per_pair):
            if cursor + 4 > len(digest):
                digest = _digest(digest.hex())
                cursor = 0
            indices.append(int.from_bytes(digest[cursor:cursor + 4], "big") % self.bits)
            cursor += 4
        return indices

    def score(self, key: str, candidate: str) -> float:
        indices = self.indices(str(key), str(candidate))
        return sum(1 for index in indices if self.array[index]) / len(indices)

    def rank(self, key: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        rows = [{"candidate": str(candidate), "score": self.score(str(key), str(candidate))} for candidate in candidates]
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows


class RandomCandidateBaseline:
    """Deterministic chance baseline."""

    def __init__(self, *, seed: int = 7) -> None:
        self.seed = int(seed)

    @property
    def memory_bytes(self) -> int:
        return 0

    def rank(self, key: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        rows = []
        for candidate in candidates:
            score = int.from_bytes(_digest(f"random:{self.seed}:{key}:{candidate}")[:8], "big") / float(2**64)
            rows.append({"candidate": str(candidate), "score": score})
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows


def run_phase_accio(
    *,
    out_dir: str | Path = "runs/phase-accio",
    context_tokens: int = 1_048_576,
    needles: int = 100,
    candidates: int = 8,
    seeds: int = 3,
    grid_size: int = 128,
    slots_per_symbol: int = 24,
    pin_strength: float = 0.25,
    filler_noise: float = 0.001,
    filler_stride: int = 32,
    proximity_window: int = 12,
    context_style: str = "natural",
    include_baselines: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows_path = out_path / "rows.jsonl"

    rows: list[dict[str, Any]] = []
    with rows_path.open("w", encoding="utf-8") as handle:
        for seed in range(int(seeds)):
            task = make_phase_accio_task(
                context_tokens=int(context_tokens),
                needles=int(needles),
                candidates=int(candidates),
                seed=seed,
                context_style=context_style,
            )
            seed_rows = evaluate_task(
                task,
                seed=seed,
                grid_size=grid_size,
                slots_per_symbol=slots_per_symbol,
                pin_strength=pin_strength,
                filler_noise=filler_noise,
                filler_stride=filler_stride,
                proximity_window=proximity_window,
                include_baselines=include_baselines,
            )
            for row in seed_rows:
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize_rows(rows)
    skipped_baselines = {
        "faiss_flat": "not run in the default local artifact; use the emitted rows.jsonl to run an external FAISS baseline on identical queries",
        "transformer": "not run in the default local artifact; this benchmark intentionally keeps the local run dependency-light",
    }
    payload = {
        "type": "phase-accio",
        "version": 2,
        "status": "pass" if (
            summary["pin_on"]["accuracy"] >= 0.9
            and summary["pin_off"]["accuracy"] <= 0.35
            and summary["scrambled"]["accuracy"] <= 0.35
        ) else "red",
        "elapsed_s": time.perf_counter() - started,
        "config": {
            "context_tokens": int(context_tokens),
            "needles": int(needles),
            "candidates": int(candidates),
            "seeds": int(seeds),
            "grid_size": int(grid_size),
            "slots_per_symbol": int(slots_per_symbol),
            "pin_strength": float(pin_strength),
            "filler_noise": float(filler_noise),
            "filler_stride": int(filler_stride),
            "proximity_window": int(proximity_window),
            "context_style": str(context_style),
            "include_baselines": bool(include_baselines),
        },
        "summary": summary,
        "collapse_series": build_collapse_series(rows),
        "sample_rows": rows[:18],
        "skipped_baselines": skipped_baselines,
        "paths": {
            "rows_jsonl": str(rows_path),
            "summary_json": str(out_path / "summary.json"),
            "index_html": str(out_path / "index.html"),
            "summary_md": str(out_path / "summary.md"),
        },
        "claims": [
            "Candidate scores are computed by fixed-size phase-binding resonance over extracted key/value proximity pairs.",
            "Pin-off is a causal ablation: the same generated natural notes and candidates are evaluated without key/value phase binding.",
            "Scrambled scoring uses the same memory but mismatched phase probes, so it tests whether the binding geometry matters.",
            "The benchmark is structured long-context retrieval, not open-ended language understanding.",
        ],
    }
    (out_path / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_path / "summary.md").write_text(render_phase_accio_markdown(payload), encoding="utf-8")
    (out_path / "index.html").write_text(render_phase_accio_html(payload), encoding="utf-8")
    return payload


def make_phase_accio_task(
    *,
    context_tokens: int,
    needles: int,
    candidates: int,
    seed: int,
    context_style: str = "natural",
) -> dict[str, Any]:
    rng = random.Random(seed)
    records = [
        PhaseAccioRecord(key=f"k_{_token(rng, 10)}", value=f"v_{_token(rng, 16)}")
        for _ in range(int(needles))
    ]
    record_chunks = [render_record_chunk(record, rng, context_style=context_style) for record in records]
    record_token_count = sum(len(chunk) for chunk in record_chunks)
    filler_count = max(0, int(context_tokens) - record_token_count)
    filler = make_noise_tokens(rng, filler_count)
    positions = sorted(rng.sample(range(filler_count + len(records)), len(records))) if filler_count else list(range(len(records)))
    record_by_pos = dict(zip(positions, record_chunks))
    parts: list[str] = []
    filler_index = 0
    for index in range(filler_count + len(records)):
        if index in record_by_pos:
            parts.extend(record_by_pos[index])
        else:
            parts.append(filler[filler_index])
            filler_index += 1
    values = [record.value for record in records]
    queries = []
    for record in records:
        wrong_pool = [value for value in values if value != record.value]
        wrong = rng.sample(wrong_pool, min(max(0, candidates - 1), len(wrong_pool)))
        while len(wrong) < max(0, candidates - 1):
            wrong.append(f"v_{_token(rng, 16)}")
        candidate_values = [record.value, *wrong]
        rng.shuffle(candidate_values)
        queries.append(PhaseAccioQuery(key=record.key, expected=record.value, candidates=candidate_values))
    return {
        "seed": int(seed),
        "context": " ".join(parts),
        "records": records,
        "queries": queries,
        "context_tokens": int(context_tokens),
        "context_style": str(context_style),
    }


def evaluate_task(
    task: dict[str, Any],
    *,
    seed: int,
    grid_size: int,
    slots_per_symbol: int,
    pin_strength: float,
    filler_noise: float,
    filler_stride: int,
    proximity_window: int,
    include_baselines: bool,
) -> list[dict[str, Any]]:
    tokens = tokenize(str(task["context"]))
    bindings = extract_proximity_bindings(tokens, window=proximity_window)
    rankers: dict[str, CandidateRanker] = {
        "pin_on": PhaseAccioSketch(
            grid_size=grid_size,
            slots_per_symbol=slots_per_symbol,
            seed=seed,
            pin_strength=pin_strength,
            filler_noise=filler_noise,
            filler_stride=filler_stride,
            proximity_window=proximity_window,
        ),
        "pin_off": PhaseAccioSketch(
            grid_size=grid_size,
            slots_per_symbol=slots_per_symbol,
            seed=seed,
            pin_strength=0.0,
            filler_noise=filler_noise,
            filler_stride=filler_stride,
            proximity_window=proximity_window,
        ),
        "scrambled": PhaseAccioSketch(
            grid_size=grid_size,
            slots_per_symbol=slots_per_symbol,
            seed=seed,
            pin_strength=pin_strength,
            filler_noise=filler_noise,
            filler_stride=filler_stride,
            proximity_window=proximity_window,
            salt="phase-accio",
            score_salt="phase-accio-scrambled",
        ),
    }
    for ranker in rankers.values():
        if isinstance(ranker, PhaseAccioSketch):
            ranker.ingest_tokens(tokens, bindings=bindings)

    if include_baselines:
        hash_map = HashMapBaseline()
        hash_map.ingest_bindings(bindings)
        bloom = BloomPairBaseline(bits=grid_size * grid_size, slots_per_pair=slots_per_symbol, seed=seed)
        bloom.ingest_bindings(bindings)
        rankers["hash_map"] = hash_map
        rankers["bloom_filter"] = bloom
        rankers["random_candidate"] = RandomCandidateBaseline(seed=seed)

    rows = []
    for query_index, query in enumerate(task["queries"]):
        for variant, ranker in rankers.items():
            ranked = ranker.rank(query.key, query.candidates)
            predicted = ranked[0]["candidate"] if ranked else ""
            expected_rank = next(
                (index + 1 for index, item in enumerate(ranked) if item["candidate"] == query.expected),
                None,
            )
            rows.append({
                "seed": int(seed),
                "query_index": int(query_index),
                "variant": variant,
                "kind": "phase" if variant in {"pin_on", "pin_off", "scrambled"} else "baseline",
                "context_tokens": int(task["context_tokens"]),
                "context_style": str(task.get("context_style", "natural")),
                "needles": len(task["records"]),
                "bindings_extracted": len(bindings),
                "candidates": len(query.candidates),
                "key": query.key,
                "expected": query.expected,
                "predicted": predicted,
                "passed": predicted == query.expected,
                "expected_rank": expected_rank,
                "top_score": float(ranked[0]["score"]) if ranked else 0.0,
                "expected_score": float(next((item["score"] for item in ranked if item["candidate"] == query.expected), 0.0)),
                "margin": float(ranked[0]["score"] - ranked[1]["score"]) if len(ranked) > 1 else 0.0,
                "memory_bytes": int(ranker.memory_bytes),
                "score_mode": score_mode_for(variant),
            })
    return rows


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def extract_proximity_bindings(tokens: Sequence[str], window: int = 12) -> list[tuple[str, str]]:
    bindings: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    max_distance = max(1, int(window))
    for index, token in enumerate(tokens):
        if not is_key_token(token):
            continue
        value = nearest_value_token(tokens, index, max_distance)
        if value is None:
            continue
        pair = (str(token).lower(), value.lower())
        if pair not in seen:
            seen.add(pair)
            bindings.append(pair)
    return bindings


def nearest_value_token(tokens: Sequence[str], key_index: int, window: int) -> str | None:
    best: tuple[int, int, str] | None = None
    start = max(0, key_index - window)
    stop = min(len(tokens), key_index + window + 1)
    for index in range(start, stop):
        token = str(tokens[index])
        if not is_value_token(token):
            continue
        distance = abs(index - key_index)
        direction_bias = 0 if index > key_index else 1
        candidate = (distance, direction_bias, token)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best is not None else None


def is_key_token(token: str) -> bool:
    return KEY_RE.match(str(token)) is not None


def is_value_token(token: str) -> bool:
    return VALUE_RE.match(str(token)) is not None


def render_record_chunk(record: PhaseAccioRecord, rng: random.Random, *, context_style: str) -> list[str]:
    if context_style == "record":
        return RECORD_TEMPLATE.format(key=record.key, value=record.value).split()
    if context_style != "natural":
        raise ValueError(f"unknown PhaseAccio context style: {context_style}")
    prefix = rng.sample(NOISE_WORDS, 3)
    suffix = rng.sample(NOISE_WORDS, 4)
    template = rng.choice(NATURAL_TEMPLATES).format(key=record.key, value=record.value)
    return [*prefix, *tokenize(template), *suffix]


def make_noise_tokens(rng: random.Random, count: int) -> list[str]:
    tokens: list[str] = []
    while len(tokens) < count:
        word = rng.choice(NOISE_WORDS)
        if rng.random() < 0.08:
            word = f"{word}_{rng.randint(10, 999)}"
        tokens.append(word)
    return tokens


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        passed = sum(1 for row in subset if row["passed"])
        ranks = [int(row["expected_rank"]) for row in subset if row.get("expected_rank") is not None]
        summary[variant] = {
            "kind": subset[0].get("kind", "phase") if subset else "unknown",
            "rows": len(subset),
            "passed": passed,
            "accuracy": passed / len(subset) if subset else 0.0,
            "mean_expected_rank": sum(ranks) / len(ranks) if ranks else 0.0,
            "mean_margin": sum(float(row["margin"]) for row in subset) / len(subset) if subset else 0.0,
            "memory_bytes": max([int(row["memory_bytes"]) for row in subset] or [0]),
            "bindings_extracted": max([int(row.get("bindings_extracted", 0)) for row in subset] or [0]),
        }
    pin_on = summary.get("pin_on", {}).get("accuracy", 0.0)
    pin_off = summary.get("pin_off", {}).get("accuracy", 0.0)
    summary["accuracy_lift_vs_pin_off"] = pin_on - pin_off
    return summary


def build_collapse_series(rows: list[dict[str, Any]], *, points: int = 40) -> dict[str, list[dict[str, float]]]:
    series: dict[str, list[dict[str, float]]] = {}
    for variant in ("pin_on", "pin_off", "scrambled", "bloom_filter", "random_candidate"):
        subset = [row for row in rows if row["variant"] == variant]
        if not subset:
            continue
        step = max(1, len(subset) // max(1, points))
        passed = 0
        values = []
        for index, row in enumerate(subset, start=1):
            if row["passed"]:
                passed += 1
            if index == len(subset) or index % step == 0:
                values.append({"rows": float(index), "accuracy": passed / index})
        series[variant] = values
    return series


def render_phase_accio_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# PhaseAccio",
        "",
        "Candidate-conditioned long-context retrieval with proximity extraction, phase-pinning ablations, and local baselines.",
        "",
        "## Phase Controls",
        "",
        render_markdown_table(summary, ("pin_on", "pin_off", "scrambled")),
        "",
        "## Baselines",
        "",
        render_markdown_table(summary, ("hash_map", "bloom_filter", "random_candidate")),
        "",
        "## Skipped External Baselines",
        "",
    ]
    for name, reason in payload.get("skipped_baselines", {}).items():
        lines.append(f"- `{name}`: {reason}")
    lines.extend([
        "",
        "## Limits",
        "",
        "- Structured retrieval benchmark, not open-ended language understanding.",
        "- Context is synthetically generated natural-note noise with explicit synthetic key/value identifiers.",
        "- Ingest binds nearby key/value identifiers by token-window proximity; it does not parse `record KEY value VALUE` rows by default.",
        "- Scoring uses fixed-size phase-binding resonance and reports pin-off and scrambled controls.",
        "",
    ])
    return "\n".join(lines)


def render_markdown_table(summary: dict[str, Any], variants: Sequence[str]) -> str:
    lines = [
        "| Variant | Accuracy | Rows | Mean Rank | Mean Margin | Memory |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in variants:
        row = summary.get(variant)
        if not row:
            continue
        lines.append(
            f"| {variant} | {float(row.get('accuracy', 0.0)):.3f} | {int(row.get('rows', 0))} | "
            f"{float(row.get('mean_expected_rank', 0.0)):.2f} | {float(row.get('mean_margin', 0.0)):.4f} | "
            f"{format_bytes(row.get('memory_bytes', 0))} |"
        )
    return "\n".join(lines)


def render_phase_accio_html(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    config = payload["config"]
    data_json = html.escape(json.dumps(payload, sort_keys=True), quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhaseAccio</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07090c; --panel:#101720; --line:#293642; --text:#eef4f8; --muted:#9aacb8; --green:#68eba9; --red:#ff6b80; --blue:#7cc7ff; --amber:#ffd166; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0; }}
    main {{ max-width:1180px; margin:0 auto; padding:28px; }}
    header {{ display:flex; justify-content:space-between; align-items:end; gap:16px; border-bottom:1px solid var(--line); padding-bottom:18px; }}
    h1 {{ font-size:38px; margin:0; }}
    h2 {{ font-size:18px; margin:0 0 12px; }}
    p, li {{ color:var(--muted); line-height:1.45; }}
    .badge {{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; color:var(--green); background:#101720; font-weight:750; }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }}
    .card, section {{ border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:14px; }}
    .card span {{ display:block; color:var(--muted); font-size:12px; margin-bottom:8px; }}
    .card strong {{ display:block; font-size:25px; overflow-wrap:anywhere; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:10px 8px; text-align:left; }}
    th {{ color:var(--muted); }}
    .pass {{ color:var(--green); font-weight:750; }}
    .fail {{ color:var(--red); font-weight:750; }}
    .control {{ color:var(--blue); font-weight:750; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; }}
    .bar-row {{ display:grid; grid-template-columns:150px 1fr 70px; gap:10px; align-items:center; margin:10px 0; }}
    .bar-track {{ height:18px; border:1px solid var(--line); border-radius:6px; overflow:hidden; background:#080d12; }}
    .bar {{ height:100%; width:0%; transition:width 220ms linear; background:var(--blue); }}
    .bar.pin {{ background:var(--green); }}
    .bar.random {{ background:var(--red); }}
    .mono, code {{ color:#d9e8f3; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    @media (max-width: 850px) {{ header,.cards,.grid {{ display:grid; grid-template-columns:1fr; }} .bar-row {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>PhaseAccio</h1>
        <p>Natural-note candidate retrieval with proximity binding, phase controls, and local baselines.</p>
      </div>
      <div class="badge">{html.escape(str(payload["status"]).upper())}</div>
    </header>
    <div class="cards">
      <div class="card"><span>Context Tokens</span><strong>{int(config.get("context_tokens", 0)):,}</strong></div>
      <div class="card"><span>Pin-On Accuracy</span><strong>{float(summary.get("pin_on", {}).get("accuracy", 0.0)):.1%}</strong></div>
      <div class="card"><span>Control Accuracy</span><strong>{float(summary.get("scrambled", {}).get("accuracy", 0.0)):.1%}</strong></div>
      <div class="card"><span>Phase Memory</span><strong>{format_bytes(summary.get("pin_on", {}).get("memory_bytes", 0))}</strong></div>
    </div>
    <section>
      <h2>Live Control Collapse Replay</h2>
      <p>The replay walks cumulative accuracy over the emitted audit rows. Pin-on should stay high; pin-off and scrambled controls should drift toward chance.</p>
      <div id="collapse"></div>
      <p class="mono" id="collapse-step"></p>
    </section>
    <div class="grid">
      <section>
        <h2>Phase Controls</h2>
        {render_summary_table(summary, ("pin_on", "pin_off", "scrambled"))}
      </section>
      <section>
        <h2>Local Baselines</h2>
        {render_summary_table(summary, ("hash_map", "bloom_filter", "random_candidate"))}
      </section>
    </div>
    <div class="grid">
      <section>
        <h2>Config</h2>
        {render_config_table(config)}
      </section>
      <section>
        <h2>Claim Boundary</h2>
        <ul>
          <li>Structured long-context retrieval, not a general LLM.</li>
          <li>Natural-note synthetic context, not exact <code>record KEY value VALUE</code> parsing.</li>
          <li>Hash-map and Bloom baselines are shown because this is associative retrieval.</li>
          <li>FAISS and transformer baselines are left as external audit hooks in this local artifact.</li>
        </ul>
      </section>
    </div>
    <script type="application/json" id="phase-accio-data">{data_json}</script>
    <script>
      const payload = JSON.parse(document.getElementById('phase-accio-data').textContent);
      const variants = ['pin_on', 'pin_off', 'scrambled', 'bloom_filter', 'random_candidate'];
      const labels = {{pin_on:'pin_on', pin_off:'pin_off', scrambled:'scrambled', bloom_filter:'bloom_filter', random_candidate:'random'}};
      const root = document.getElementById('collapse');
      const stepLabel = document.getElementById('collapse-step');
      root.innerHTML = variants.filter(v => payload.collapse_series[v]).map(v => `
        <div class="bar-row">
          <div><code>${{labels[v]}}</code></div>
          <div class="bar-track"><div class="bar ${{v === 'pin_on' ? 'pin' : v === 'random_candidate' ? 'random' : ''}}" id="bar-${{v}}"></div></div>
          <div class="mono" id="val-${{v}}">0.0%</div>
        </div>`).join('');
      let frame = 0;
      const maxFrames = Math.max(...Object.values(payload.collapse_series).map(items => items.length));
      function paint() {{
        for (const variant of variants) {{
          const series = payload.collapse_series[variant];
          if (!series) continue;
          const point = series[Math.min(frame, series.length - 1)];
          const pct = Math.max(0, Math.min(1, point.accuracy));
          document.getElementById(`bar-${{variant}}`).style.width = `${{pct * 100}}%`;
          document.getElementById(`val-${{variant}}`).textContent = `${{(pct * 100).toFixed(1)}}%`;
          stepLabel.textContent = `audit rows replayed: ${{Math.trunc(point.rows)}}`;
        }}
        frame = (frame + 1) % maxFrames;
      }}
      paint();
      setInterval(paint, 260);
    </script>
  </main>
</body>
</html>
"""


def render_summary_table(summary: dict[str, Any], variants: Sequence[str]) -> str:
    rows = []
    for variant in variants:
        row = summary.get(variant)
        if not row:
            continue
        is_primary_pass = variant == "pin_on" and float(row.get("accuracy", 0.0)) >= 0.9
        label = "pass" if is_primary_pass else "baseline" if row.get("kind") == "baseline" else "control"
        class_name = "pass" if is_primary_pass else "control" if row.get("kind") == "baseline" else "fail"
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(variant)}</code></td>"
            f"<td>{float(row.get('accuracy', 0.0)):.3f}</td>"
            f"<td>{int(row.get('rows', 0))}</td>"
            f"<td>{float(row.get('mean_expected_rank', 0.0)):.2f}</td>"
            f"<td>{float(row.get('mean_margin', 0.0)):.4f}</td>"
            f"<td>{format_bytes(row.get('memory_bytes', 0))}</td>"
            f"<td><span class=\"{class_name}\">{label}</span></td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Variant</th><th>Accuracy</th><th>Rows</th><th>Mean Rank</th><th>Margin</th><th>Memory</th><th>Type</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def render_config_table(config: dict[str, Any]) -> str:
    rows = [
        f"<tr><td><code>{html.escape(str(key))}</code></td><td>{html.escape(json.dumps(value))}</td></tr>"
        for key, value in sorted(config.items())
    ]
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def score_mode_for(variant: str) -> str:
    modes = {
        "pin_on": "fixed_phase_binding_cosine",
        "pin_off": "pin_off_phase_control",
        "scrambled": "scrambled_probe_phase_control",
        "hash_map": "exact_proximity_hash_map",
        "bloom_filter": "bloom_pair_membership",
        "random_candidate": "deterministic_random_candidate",
    }
    return modes.get(variant, "unknown")


def _token(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("abcdef0123456789") for _ in range(int(length)))


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()


def format_bytes(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        amount = 0.0
    units = ("B", "KB", "MB", "GB")
    index = 0
    while amount >= 1024.0 and index < len(units) - 1:
        amount /= 1024.0
        index += 1
    return f"{amount:.1f} {units[index]}" if index else f"{int(amount)} {units[index]}"
