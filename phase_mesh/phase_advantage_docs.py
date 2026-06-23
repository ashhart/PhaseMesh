from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)
TWO_PI = 2.0 * math.pi

PEOPLE = (
    "mira", "jonas", "leena", "tomas", "ravi", "nora", "elias", "sana", "owen", "ivy",
    "marin", "selene", "dax", "kyra", "noel", "arden", "mina", "felix", "sol", "vera",
)
OBJECTS = (
    "ceramic invoice", "blue audit folder", "copper relay", "glass ledger", "silver map",
    "amber receipt", "violet notebook", "carbon keycard", "linen manifest", "magnet spool",
    "brass adapter", "onyx report", "paper token", "green dispatch card", "signal tile",
)
PLACES = (
    "north archive", "quiet loading bay", "river office", "west stairwell", "cedar lab",
    "lower records room", "harbor desk", "red corridor", "signal alcove", "garden annex",
)
ACTIONS = (
    "moved", "sealed", "tagged", "checked", "routed", "paired", "indexed", "returned",
    "flagged", "confirmed", "placed", "reviewed", "balanced", "attached", "reconciled",
)
QUALITIES = (
    "after the late audit", "before the courier arrived", "during the quiet handoff",
    "while the backup log was open", "after duplicate notes were rejected",
    "before the routing window closed", "during the supervisor review",
)
OUTCOME_WORDS = (
    "quartz", "lantern", "harbor", "ember", "ribbon", "orbit", "cobalt", "mirror",
    "willow", "canyon", "silver", "drift", "atlas", "comet", "frost", "signal",
    "cedar", "violet", "linen", "brass", "amber", "field", "delta", "north",
)
FILLER_WORDS = (
    "ordinary", "meeting", "draft", "window", "manual", "operator", "review", "paper",
    "schedule", "quiet", "signal", "local", "buffer", "trace", "plain", "weekly",
    "ledger", "archive", "folder", "note", "later", "before", "after", "under",
    "between", "routing", "sample", "status", "control", "working", "daily", "handoff",
)
SYNONYMS = {
    "moved": ("shifted", "carried", "transferred"),
    "sealed": ("closed", "locked", "wrapped"),
    "tagged": ("marked", "labelled", "flagged"),
    "checked": ("reviewed", "verified", "inspected"),
    "routed": ("sent", "directed", "forwarded"),
    "paired": ("matched", "linked", "joined"),
    "archive": ("records", "store", "file"),
    "office": ("desk", "room", "workspace"),
    "folder": ("binder", "file", "packet"),
    "invoice": ("bill", "receipt", "ledger"),
    "audit": ("review", "check", "inspection"),
}


@dataclass(frozen=True)
class NaturalRecord:
    id: int
    document: str
    query: str
    key_tokens: tuple[str, ...]
    value: str


class NaturalPhaseMemory:
    """Fixed-size phase memory over semantic token/value bindings."""

    def __init__(self, *, cells: int = 8192, slots: int = 5, seed: int = 11, mode: str = "distributed") -> None:
        self.cells = int(cells)
        self.slots = int(slots)
        self.seed = int(seed)
        self.mode = str(mode)
        self.memory = np.zeros(self.cells, dtype=np.complex64)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def memory_bytes(self) -> int:
        return int(self.memory.nbytes)

    def add(self, tokens: Sequence[str], value: str) -> None:
        features = self.features(tokens)
        scale = 1.0 / max(1, len(features))
        for feature in features:
            self.memory += np.asarray(scale * self.vector(feature, value), dtype=np.complex64)

    def normalize(self) -> None:
        norm = float(np.linalg.norm(self.memory))
        if norm > 0.0:
            self.memory = np.asarray(self.memory / norm, dtype=np.complex64)

    def features(self, tokens: Sequence[str]) -> list[str]:
        clean = normalize_tokens(tokens)
        if self.mode == "whole":
            return ["whole:" + " ".join(clean)]
        features = [f"tok:{token}" for token in clean]
        features.extend(f"pair:{a}:{b}" for a, b in zip(clean, clean[1:]))
        features.extend(f"skip:{clean[index]}:{clean[index + 2]}" for index in range(max(0, len(clean) - 2)))
        return features

    def score(self, tokens: Sequence[str], value: str) -> float:
        features = self.features(tokens)
        if not features:
            return 0.0
        probe = np.zeros(self.cells, dtype=np.complex64)
        for feature in features:
            probe += self.vector(feature, value)
        probe = np.asarray(probe / len(features), dtype=np.complex64)
        denom = float(np.linalg.norm(probe) * np.linalg.norm(self.memory))
        if denom <= 0.0:
            return 0.0
        return float(np.real(np.vdot(probe, self.memory)) / denom)

    def rank(self, tokens: Sequence[str], candidates: Sequence[str]) -> list[dict[str, Any]]:
        rows = [{"candidate": str(candidate), "score": self.score(tokens, str(candidate))} for candidate in candidates]
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows

    def vector(self, feature: str, value: str) -> np.ndarray:
        cache_key = f"{feature}->{value}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        vec = np.zeros(self.cells, dtype=np.complex64)
        digest = digest_bytes(f"natural-phase:{self.seed}:{feature}:{value}")
        cursor = 0
        for _slot in range(self.slots):
            if cursor + 5 > len(digest):
                digest = digest_bytes(digest.hex())
                cursor = 0
            index = int.from_bytes(digest[cursor:cursor + 4], "big") % self.cells
            phase = (digest[cursor + 4] / 255.0) * TWO_PI
            vec[index] += np.complex64(math.cos(phase) + 1j * math.sin(phase))
            cursor += 5
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = np.asarray(vec / norm, dtype=np.complex64)
        self._cache[cache_key] = vec
        return vec


class ExactNaturalHash:
    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}

    @property
    def memory_bytes(self) -> int:
        return sum(len(key) + len(value) + 96 for key, value in self.mapping.items())

    def add(self, record: NaturalRecord) -> None:
        self.mapping[canonical_query(record.query)] = record.value

    def rank(self, query: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        value = self.mapping.get(canonical_query(query))
        rows = [{"candidate": str(candidate), "score": 1.0 if candidate == value else 0.0} for candidate in candidates]
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows


class ExactNaturalBloom:
    def __init__(self, *, bits: int = 262144, slots: int = 7, seed: int = 11) -> None:
        self.bits = max(8, int(bits))
        self.slots = max(1, int(slots))
        self.seed = int(seed)
        self.array = np.zeros(self.bits, dtype=np.bool_)

    @property
    def memory_bytes(self) -> int:
        return int(math.ceil(self.bits / 8))

    def add(self, record: NaturalRecord) -> None:
        for index in self.indices(record.query, record.value):
            self.array[index] = True

    def indices(self, query: str, value: str) -> list[int]:
        digest = digest_bytes(f"natural-bloom:{self.seed}:{canonical_query(query)}:{value}")
        cursor = 0
        indices = []
        for _ in range(self.slots):
            if cursor + 4 > len(digest):
                digest = digest_bytes(digest.hex())
                cursor = 0
            indices.append(int.from_bytes(digest[cursor:cursor + 4], "big") % self.bits)
            cursor += 4
        return indices

    def rank(self, query: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        rows = []
        for candidate in candidates:
            indices = self.indices(query, str(candidate))
            score = sum(1 for index in indices if self.array[index]) / len(indices)
            rows.append({"candidate": str(candidate), "score": 1.0 if score >= 1.0 else 0.0})
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows


class BM25Baseline:
    def __init__(self, records: Sequence[NaturalRecord]) -> None:
        self.docs = [normalize_tokens(tokenize(record.document + " " + record.query)) for record in records]
        self.values = [record.value for record in records]
        self.avg_len = sum(len(doc) for doc in self.docs) / max(1, len(self.docs))
        self.df: dict[str, int] = {}
        for doc in self.docs:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

    @property
    def memory_bytes(self) -> int:
        return sum(sum(len(token) + 8 for token in doc) + len(value) + 64 for doc, value in zip(self.docs, self.values))

    def rank(self, query: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        q_tokens = normalize_tokens(tokenize(query))
        candidate_set = set(str(candidate) for candidate in candidates)
        best: dict[str, float] = {str(candidate): -1e9 for candidate in candidates}
        for doc, value in zip(self.docs, self.values):
            if value not in candidate_set:
                continue
            score = self.score_doc(q_tokens, doc)
            if score > best[value]:
                best[value] = score
        rows = [{"candidate": candidate, "score": float(best[str(candidate)])} for candidate in candidates]
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows

    def score_doc(self, query_tokens: Sequence[str], doc: Sequence[str]) -> float:
        if not doc:
            return 0.0
        tf: dict[str, int] = {}
        for token in doc:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        k1 = 1.2
        b = 0.75
        n_docs = max(1, len(self.docs))
        for token in query_tokens:
            if token not in tf:
                continue
            df = self.df.get(token, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            freq = tf[token]
            denom = freq + k1 * (1.0 - b + b * len(doc) / max(1e-6, self.avg_len))
            score += idf * (freq * (k1 + 1.0)) / denom
        return score


class NGramHashBaseline:
    def __init__(self, records: Sequence[NaturalRecord], *, dims: int = 8192, seed: int = 11) -> None:
        self.dims = int(dims)
        self.seed = int(seed)
        self.value_vectors: dict[str, np.ndarray] = {}
        for record in records:
            vec = self.query_vector(record.key_tokens)
            self.value_vectors[record.value] = self.value_vectors.get(record.value, np.zeros(self.dims, dtype=np.float32)) + vec
        for value, vec in list(self.value_vectors.items()):
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                self.value_vectors[value] = np.asarray(vec / norm, dtype=np.float32)

    @property
    def memory_bytes(self) -> int:
        return len(self.value_vectors) * self.dims * 4

    def query_vector(self, tokens: Sequence[str]) -> np.ndarray:
        clean = normalize_tokens(tokens)
        vec = np.zeros(self.dims, dtype=np.float32)
        grams = [f"tok:{token}" for token in clean]
        grams.extend(f"bi:{a}:{b}" for a, b in zip(clean, clean[1:]))
        for gram in grams:
            digest = digest_bytes(f"ngram:{self.seed}:{gram}")
            idx = int.from_bytes(digest[:4], "big") % self.dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return np.asarray(vec / norm, dtype=np.float32) if norm > 0.0 else vec

    def rank(self, query: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        q_vec = self.query_vector(tokenize(query))
        rows = []
        for candidate in candidates:
            vec = self.value_vectors.get(str(candidate))
            score = float(np.dot(q_vec, vec)) if vec is not None else 0.0
            rows.append({"candidate": str(candidate), "score": score})
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows


class VectorBaseline:
    def __init__(self, records: Sequence[NaturalRecord], *, dims: int = 128, seed: int = 11) -> None:
        self.dims = int(dims)
        self.seed = int(seed)
        self.values = [record.value for record in records]
        self.matrix = np.stack([self.embed(tokenize(record.document + " " + record.query)) for record in records])
        self.backend = "numpy-flat"
        self.faiss_index = None
        try:
            import faiss  # type: ignore

            self.faiss_index = faiss.IndexFlatIP(self.dims)
            self.faiss_index.add(self.matrix.astype(np.float32))
            self.backend = "faiss-flat"
        except Exception:
            self.faiss_index = None

    @property
    def memory_bytes(self) -> int:
        return int(self.matrix.nbytes + sum(len(value) + 48 for value in self.values))

    def token_vector(self, token: str) -> np.ndarray:
        digest = digest_bytes(f"vec:{self.seed}:{token}")
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return rng.normal(0.0, 1.0, self.dims).astype(np.float32)

    def embed(self, tokens: Sequence[str]) -> np.ndarray:
        clean = normalize_tokens(tokens)
        if not clean:
            return np.zeros(self.dims, dtype=np.float32)
        vec = np.zeros(self.dims, dtype=np.float32)
        for token in clean:
            vec += self.token_vector(token)
        norm = float(np.linalg.norm(vec))
        return np.asarray(vec / norm, dtype=np.float32) if norm > 0.0 else vec

    def rank(self, query: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        q_vec = self.embed(tokenize(query))
        candidate_set = set(str(candidate) for candidate in candidates)
        scores = {str(candidate): -1e9 for candidate in candidates}
        if self.faiss_index is not None:
            k = min(len(self.values), max(32, len(candidates) * 4))
            sims, indices = self.faiss_index.search(q_vec.reshape(1, -1).astype(np.float32), k)
            for score, index in zip(sims[0], indices[0]):
                if index < 0:
                    continue
                value = self.values[int(index)]
                if value in candidate_set and float(score) > scores[value]:
                    scores[value] = float(score)
        else:
            sims = self.matrix @ q_vec
            for index, score in enumerate(sims):
                value = self.values[index]
                if value in candidate_set and float(score) > scores[value]:
                    scores[value] = float(score)
        rows = [{"candidate": candidate, "score": float(scores[str(candidate)])} for candidate in candidates]
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows


class RandomReservoirBaseline:
    def __init__(self, records: Sequence[NaturalRecord], *, dims: int = 128, seed: int = 11) -> None:
        self.dims = int(dims)
        self.seed = int(seed)
        self.states = {record.value: self.state(record.key_tokens) for record in records}

    @property
    def memory_bytes(self) -> int:
        return len(self.states) * self.dims * 4

    def state(self, tokens: Sequence[str]) -> np.ndarray:
        state = np.zeros(self.dims, dtype=np.float32)
        recurrent = self.recurrent_vector()
        for token in normalize_tokens(tokens):
            drive = self.drive_vector(token)
            state = np.tanh(0.72 * state + 0.18 * np.roll(state, 1) + 0.10 * recurrent + drive)
        norm = float(np.linalg.norm(state))
        return np.asarray(state / norm, dtype=np.float32) if norm > 0.0 else state

    def recurrent_vector(self) -> np.ndarray:
        rng = np.random.default_rng(int.from_bytes(digest_bytes(f"reservoir:{self.seed}:recurrent")[:8], "big"))
        return rng.normal(0.0, 0.1, self.dims).astype(np.float32)

    def drive_vector(self, token: str) -> np.ndarray:
        rng = np.random.default_rng(int.from_bytes(digest_bytes(f"reservoir:{self.seed}:{token}")[:8], "big"))
        return rng.normal(0.0, 0.35, self.dims).astype(np.float32)

    def rank(self, query: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        q_state = self.state(tokenize(query))
        rows = []
        for candidate in candidates:
            state = self.states.get(str(candidate))
            score = float(np.dot(q_state, state)) if state is not None else 0.0
            rows.append({"candidate": str(candidate), "score": score})
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows


def run_phase_advantage_docs(
    *,
    out_dir: str | Path = "runs/phase-advantage-docs",
    context_tokens: int = 1_048_576,
    records: int = 500,
    candidates: int = 16,
    trials: int = 240,
    corruption_rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    phase_cells: int = 8192,
    slots: int = 5,
    seed: int = 11,
    architecture_epochs: int = 120,
    skip_architecture: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    natural_records = make_natural_records(records, seed=seed)
    context_info = write_context_artifacts(natural_records, context_tokens, out_path, rng)
    baselines = build_doc_baselines(natural_records, phase_cells=phase_cells, slots=slots, seed=seed)
    corruption = run_doc_corruption_curve(
        natural_records,
        baselines,
        candidates=candidates,
        trials=trials,
        rates=corruption_rates,
        seed=seed,
    )
    architecture = None if skip_architecture else run_architecture_rung(out_path, architecture_epochs, seed)
    status = "pass" if (
        corruption["by_rate"]["0.30"]["phase_completion"]["accuracy"] >= 0.45
        and corruption["by_rate"]["0.30"]["exact_hash"]["accuracy"] == 0.0
        and corruption["by_rate"]["0.30"]["exact_bloom"]["accuracy"] == 0.0
        and corruption["by_rate"]["0.30"]["whole_key_phase"]["accuracy"] <= 0.15
        and corruption["control_collapse"]["phase_vs_whole_at_30"] >= 3.0
        and (architecture is None or architecture["status"] in {"pass", "red"})
    ) else "red"
    payload = {
        "type": "phase-mesh-natural-document-advantage",
        "version": 1,
        "status": status,
        "elapsed_s": time.perf_counter() - started,
        "config": {
            "seed": seed,
            "context_tokens": context_tokens,
            "records": records,
            "candidates": candidates,
            "trials": trials,
            "corruption_rates": list(corruption_rates),
            "phase_cells": phase_cells,
            "slots": slots,
            "phase_memory_bytes": baselines["phase_completion"].memory_bytes,
        },
        "context": context_info,
        "baselines": baseline_memory_table(baselines),
        "corruption_curve": corruption,
        "architecture_rung": architecture,
        "claim_boundary": [
            "This is natural-document corrupted retrieval, not free-form question answering.",
            "Hash and Bloom are exact controls; they intentionally do not get fuzzy lookup.",
            "BM25, vector, n-gram hash, and random reservoir are included as stronger retrieval baselines.",
            "The architecture rung is a toy differentiable-core probe, not an LLM result.",
        ],
    }
    (out_path / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_path / "summary.md").write_text(render_docs_markdown(payload), encoding="utf-8")
    (out_path / "index.html").write_text(render_docs_html(payload), encoding="utf-8")
    return payload


def build_doc_baselines(records: Sequence[NaturalRecord], *, phase_cells: int, slots: int, seed: int) -> dict[str, Any]:
    phase = NaturalPhaseMemory(cells=phase_cells, slots=slots, seed=seed, mode="distributed")
    whole = NaturalPhaseMemory(cells=phase_cells, slots=slots, seed=seed, mode="whole")
    exact = ExactNaturalHash()
    bloom = ExactNaturalBloom(bits=max(phase_cells * 64, len(records) * 7 * 512), slots=7, seed=seed)
    for record in records:
        phase.add(record.key_tokens, record.value)
        whole.add(record.key_tokens, record.value)
        exact.add(record)
        bloom.add(record)
    phase.normalize()
    whole.normalize()
    return {
        "phase_completion": phase,
        "whole_key_phase": whole,
        "exact_hash": exact,
        "exact_bloom": bloom,
        "bm25": BM25Baseline(records),
        "vector_faiss": VectorBaseline(records, dims=128, seed=seed),
        "ngram_hash": NGramHashBaseline(records, dims=max(2048, phase_cells), seed=seed),
        "random_reservoir": RandomReservoirBaseline(records, dims=128, seed=seed),
    }


def run_doc_corruption_curve(
    records: Sequence[NaturalRecord],
    baselines: dict[str, Any],
    *,
    candidates: int,
    trials: int,
    rates: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed + 313)
    values = [record.value for record in records]
    subset = list(records[: min(trials, len(records))])
    by_rate: dict[str, Any] = {}
    examples: list[dict[str, Any]] = []
    for rate in rates:
        model_hits = {name: 0 for name in baselines}
        model_rank_sum = {name: 0.0 for name in baselines}
        for row_index, record in enumerate(subset):
            corrupted = corrupt_natural_query(record.query, rate, rng)
            candidate_values = make_candidates(record.value, values, candidates, rng)
            for name, model in baselines.items():
                ranking = rank_model(model, corrupted, candidate_values)
                prediction = predicted_candidate(name, ranking)
                model_hits[name] += prediction == record.value
                model_rank_sum[name] += candidate_rank(ranking, record.value)
            if len(examples) < 8 and rate in {0.3, 0.4} and row_index % 17 == 0:
                examples.append({
                    "rate": rate,
                    "query": corrupted,
                    "expected": record.value,
                    "phase_top": rank_model(baselines["phase_completion"], corrupted, candidate_values)[:3],
                    "bm25_top": rank_model(baselines["bm25"], corrupted, candidate_values)[:3],
                })
        by_rate[f"{rate:.2f}"] = {
            name: {
                "passed": model_hits[name],
                "rows": len(subset),
                "accuracy": model_hits[name] / len(subset) if subset else 0.0,
                "mean_rank": model_rank_sum[name] / len(subset) if subset else 0.0,
            }
            for name in baselines
        }
    phase_30 = by_rate["0.30"]["phase_completion"]["accuracy"]
    whole_30 = by_rate["0.30"]["whole_key_phase"]["accuracy"]
    return {
        "task": "natural_document_corrupted_retrieval",
        "rates": list(rates),
        "by_rate": by_rate,
        "examples": examples,
        "control_collapse": {
            "phase_vs_whole_at_30": phase_30 / max(1e-9, whole_30),
            "exact_controls_at_30": {
                "hash": by_rate["0.30"]["exact_hash"]["accuracy"],
                "bloom": by_rate["0.30"]["exact_bloom"]["accuracy"],
            },
        },
    }


def rank_model(model: Any, query: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
    if isinstance(model, NaturalPhaseMemory):
        return model.rank(tokenize(query), candidates)
    return model.rank(query, candidates)


def candidate_rank(ranking: Sequence[dict[str, Any]], expected: str) -> int:
    for index, row in enumerate(ranking, start=1):
        if row["candidate"] == expected:
            return index
    return len(ranking) + 1


def predicted_candidate(model_name: str, ranking: Sequence[dict[str, Any]]) -> str | None:
    if not ranking:
        return None
    if model_name in {"exact_hash", "exact_bloom"} and float(ranking[0]["score"]) <= 0.0:
        return None
    return str(ranking[0]["candidate"])


def make_natural_records(count: int, *, seed: int) -> list[NaturalRecord]:
    rng = random.Random(seed)
    records = []
    used_values: set[str] = set()
    for index in range(count):
        person = rng.choice(PEOPLE)
        obj = rng.choice(OBJECTS)
        place = rng.choice(PLACES)
        action = rng.choice(ACTIONS)
        quality = rng.choice(QUALITIES)
        value = make_value_phrase(index, rng, used_values)
        object_bits = obj.split()
        place_bits = place.split()
        key_tokens = tuple(normalize_tokens([person, action, *object_bits, *place_bits, *quality.split()[:3]]))
        document = (
            f"{person.title()} {action} the {obj} near the {place} {quality}. "
            f"The remembered outcome for that event was {value}. "
            f"A nearby note mentioned {rng.choice(FILLER_WORDS)} {rng.choice(FILLER_WORDS)} {rng.choice(FILLER_WORDS)}."
        )
        query = f"{person} {action} {obj} {place} {quality}"
        records.append(NaturalRecord(index, document, query, key_tokens, value))
    return records


def make_value_phrase(index: int, rng: random.Random, used: set[str]) -> str:
    while True:
        value = f"{rng.choice(OUTCOME_WORDS)}-{rng.choice(OUTCOME_WORDS)}-{(index * 37 + rng.randrange(997)) % 10000:04d}"
        if value not in used:
            used.add(value)
            return value


def corrupt_natural_query(query: str, rate: float, rng: random.Random) -> str:
    tokens = tokenize(query)
    if not tokens:
        return query
    changes = max(1, int(round(len(tokens) * float(rate)))) if rate > 0 else 0
    mutable = list(tokens)
    for index in rng.sample(range(len(tokens)), min(changes, len(tokens))):
        token = mutable[index].lower()
        mode = rng.choice(["drop", "replace", "synonym", "typo"])
        if mode == "drop":
            mutable[index] = ""
        elif mode == "synonym" and token in SYNONYMS:
            mutable[index] = rng.choice(SYNONYMS[token])
        elif mode == "typo" and len(token) > 4:
            pos = rng.randrange(1, len(token) - 1)
            mutable[index] = token[:pos] + token[pos + 1] + token[pos] + token[pos + 2:]
        else:
            mutable[index] = rng.choice(FILLER_WORDS)
    return " ".join(token for token in mutable if token)


def write_context_artifacts(records: Sequence[NaturalRecord], token_target: int, out_path: Path, rng: random.Random) -> dict[str, Any]:
    tokens: list[str] = []
    record_iter = iter(records)
    next_record = next(record_iter, None)
    inserted = 0
    while len(tokens) < token_target:
        if next_record is not None and (len(tokens) // 180) >= inserted:
            doc_tokens = tokenize(next_record.document)
            tokens.extend(doc_tokens)
            inserted += 1
            next_record = next(record_iter, None)
        else:
            tokens.extend(rng.choices(FILLER_WORDS, k=min(64, token_target - len(tokens))))
    tokens = tokens[:token_target]
    sample_path = out_path / "context_sample.txt"
    sample_path.write_text(" ".join(tokens[: min(len(tokens), 4096)]) + "\n", encoding="utf-8")
    return {
        "target_tokens": int(token_target),
        "actual_tokens": len(tokens),
        "records_inserted": inserted,
        "sample_path": str(sample_path),
        "sample_tokens": min(len(tokens), 4096),
    }


def run_architecture_rung(out_path: Path, epochs: int, seed: int) -> dict[str, Any]:
    try:
        from .learnable_core import run_learnable_core_probe

        return run_learnable_core_probe(
            out_dir=out_path / "learnable-core",
            sequence_length=24,
            train_size=512,
            test_size=512,
            epochs=epochs,
            batch_size=128,
            oscillators=24,
            hidden=48,
            seed=seed,
        )
    except Exception as exc:
        return {
            "type": "phase-mesh-learnable-core-probe",
            "status": "error",
            "error": str(exc),
        }


def baseline_memory_table(baselines: dict[str, Any]) -> dict[str, Any]:
    table = {}
    for name, model in baselines.items():
        table[name] = {
            "memory_bytes": int(getattr(model, "memory_bytes", 0)),
        }
        if name == "vector_faiss":
            table[name]["backend"] = getattr(model, "backend", "unknown")
    return table


def tokenize(text: str | Sequence[str]) -> list[str]:
    if isinstance(text, str):
        return TOKEN_RE.findall(text.lower())
    return [str(token).lower() for token in text]


def normalize_tokens(tokens: Sequence[str]) -> list[str]:
    return [token.lower() for token in tokens if token and len(token) > 1]


def canonical_query(query: str) -> str:
    return " ".join(normalize_tokens(tokenize(query)))


def make_candidates(correct: str, values: Sequence[str], count: int, rng: random.Random) -> list[str]:
    pool = [value for value in values if value != correct]
    wrong = rng.sample(pool, min(max(0, count - 1), len(pool)))
    candidates = [correct, *wrong]
    rng.shuffle(candidates)
    return candidates


def render_docs_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PhaseMesh Natural-Document Advantage",
        "",
        f"Status: **{payload['status'].upper()}**",
        "",
        "## Corrupted Natural Queries",
        "",
        "| Corruption | Phase | Whole-Key | Hash | Bloom | BM25 | Vector/FAISS | N-gram Hash | Reservoir |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_rate = payload["corruption_curve"]["by_rate"]
    for rate in sorted(by_rate):
        row = by_rate[rate]
        lines.append(
            f"| {float(rate):.0%} | "
            f"{row['phase_completion']['accuracy']:.3f} | "
            f"{row['whole_key_phase']['accuracy']:.3f} | "
            f"{row['exact_hash']['accuracy']:.3f} | "
            f"{row['exact_bloom']['accuracy']:.3f} | "
            f"{row['bm25']['accuracy']:.3f} | "
            f"{row['vector_faiss']['accuracy']:.3f} | "
            f"{row['ngram_hash']['accuracy']:.3f} | "
            f"{row['random_reservoir']['accuracy']:.3f} |"
        )
    lines.extend([
        "",
        "## Baseline Memory",
        "",
        "| Model | Bytes | Notes |",
        "| --- | ---: | --- |",
    ])
    for name, row in payload["baselines"].items():
        notes = row.get("backend", "")
        lines.append(f"| {name} | {row['memory_bytes']} | {notes} |")
    architecture = payload.get("architecture_rung")
    if architecture is not None:
        lines.extend(["", "## Architecture Rung", ""])
        if "results" in architecture:
            lines.append("| Model | Test Accuracy | Trainable Params |")
            lines.append("| --- | ---: | ---: |")
            for name in ("learned_phase", "frozen_phase", "bag_mlp"):
                result = architecture["results"][name]
                lines.append(f"| {name} | {result['test_accuracy']:.3f} | {result['trainable_parameter_count']} |")
        else:
            lines.append(f"Architecture rung status: `{architecture.get('status')}`")
    lines.extend(["", "## Claim Boundary", ""])
    lines.extend(f"- {item}" for item in payload["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


def render_docs_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload["corruption_curve"]["by_rate"])
    status_class = "pass" if payload["status"] == "pass" else "red"
    summary_rows = markdown_table_rows(payload)
    examples = payload["corruption_curve"].get("examples", [])
    architecture = payload.get("architecture_rung") or {}
    arch_html = render_architecture_html(architecture)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhaseMesh Natural-Document Advantage</title>
  <style>
    :root {{
      --ink:#15201a; --muted:#5c6b61; --line:#d5ded8; --bg:#f6f7f2; --panel:#fff;
      --phase:#0f7b63; --whole:#b44848; --bm25:#3266a8; --vector:#7759a6;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:28px 20px 44px; }}
    header {{ display:flex; justify-content:space-between; gap:18px; align-items:end; border-bottom:1px solid var(--line); padding-bottom:18px; }}
    h1 {{ margin:0; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:0 0 10px; font-size:17px; }}
    p {{ color:var(--muted); margin:8px 0; max-width:860px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin-top:16px; }}
    .grid {{ display:grid; grid-template-columns:1.2fr .8fr; gap:16px; }}
    .badge {{ padding:6px 10px; border-radius:6px; font-weight:800; }}
    .badge.pass {{ background:#dcefe8; color:#0b5c47; }}
    .badge.red {{ background:#f4dada; color:#822628; }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ border-bottom:1px solid var(--line); padding:7px 6px; text-align:right; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ color:var(--muted); font-size:12px; }}
    svg {{ width:100%; height:auto; display:block; }}
    input[type=range] {{ width:100%; }}
    .metric {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }}
    .metric div {{ border:1px solid var(--line); border-radius:6px; padding:10px; }}
    .metric strong {{ display:block; font-size:20px; }}
    pre {{ white-space:pre-wrap; background:#f1f4ef; border:1px solid var(--line); border-radius:6px; padding:10px; color:#26332b; }}
    @media (max-width:900px) {{ .grid,.metric {{ grid-template-columns:1fr; }} header {{ align-items:start; flex-direction:column; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>PhaseMesh Natural-Document Advantage</h1>
      <p>Natural noisy paragraphs, corrupted/paraphrased queries, exact controls, retrieval baselines, 1M-token context artifact, and a learnable-core rung.</p>
    </div>
    <div class="badge {status_class}">{html.escape(payload["status"].upper())}</div>
  </header>
  <div class="grid">
    <section>
      <h2>Control Collapse Live</h2>
      <p>Move the corruption slider. Exact controls should fall immediately; whole-key phase should collapse; distributed PhaseMesh should degrade smoothly.</p>
      <input id="rate" type="range" min="0" max="5" value="3" step="1">
      <p id="rateLabel"></p>
      <svg id="chart" viewBox="0 0 720 300" role="img" aria-label="corruption curve"></svg>
    </section>
    <section>
      <h2>Selected Rate</h2>
      <div id="metrics" class="metric"></div>
      <h2 style="margin-top:18px">1M Context</h2>
      <table>
        <tbody>
          <tr><td>Target tokens</td><td>{payload["context"]["target_tokens"]}</td></tr>
          <tr><td>Actual tokens</td><td>{payload["context"]["actual_tokens"]}</td></tr>
          <tr><td>Records inserted</td><td>{payload["context"]["records_inserted"]}</td></tr>
          <tr><td>Sample saved</td><td>{html.escape(payload["context"]["sample_path"])}</td></tr>
        </tbody>
      </table>
    </section>
  </div>
  <section>
    <h2>Accuracy Table</h2>
    {summary_rows}
  </section>
  <section>
    <h2>Example Corrupted Queries</h2>
    {''.join(f"<pre>{html.escape(example['query'])}\\nexpected: {html.escape(example['expected'])}</pre>" for example in examples[:4])}
  </section>
  <section>
    <h2>Architecture Rung</h2>
    {arch_html}
  </section>
  <section>
    <h2>Claim Boundary</h2>
    <ul>{''.join(f"<li>{html.escape(item)}</li>" for item in payload["claim_boundary"])}</ul>
  </section>
</main>
<script>
const rows = {data};
const rates = Object.keys(rows).sort();
const names = ["phase_completion","whole_key_phase","exact_hash","exact_bloom","bm25","vector_faiss","ngram_hash","random_reservoir"];
const labels = {{
  phase_completion:"PhaseMesh", whole_key_phase:"Whole-key", exact_hash:"Hash", exact_bloom:"Bloom",
  bm25:"BM25", vector_faiss:"Vector/FAISS", ngram_hash:"N-gram", random_reservoir:"Reservoir"
}};
const colors = {{
  phase_completion:"#0f7b63", whole_key_phase:"#b44848", exact_hash:"#4d596d", exact_bloom:"#8a7023",
  bm25:"#3266a8", vector_faiss:"#7759a6", ngram_hash:"#b4662b", random_reservoir:"#777"
}};
const slider = document.getElementById("rate");
function draw() {{
  const svg = document.getElementById("chart");
  const w=720,h=300,l=42,r=18,t=18,b=34;
  const x = i => l + i/(rates.length-1)*(w-l-r);
  const y = v => t + (1-v)*(h-t-b);
  let out = `<rect width="${{w}}" height="${{h}}" fill="#fff"/>`;
  [0,.5,1].forEach(v => out += `<line x1="${{l}}" x2="${{w-r}}" y1="${{y(v)}}" y2="${{y(v)}}" stroke="#d5ded8"/><text x="30" y="${{y(v)+4}}" text-anchor="end">${{v.toFixed(1)}}</text>`);
  rates.forEach((rate,i) => out += `<text x="${{x(i)}}" y="${{h-8}}" text-anchor="middle">${{Math.round(parseFloat(rate)*100)}}%</text>`);
  names.forEach(name => {{
    const pts = rates.map((rate,i) => `${{x(i)}},${{y(rows[rate][name].accuracy)}}`).join(" ");
    out += `<polyline points="${{pts}}" fill="none" stroke="${{colors[name]}}" stroke-width="${{name === "phase_completion" ? 4 : 2}}" opacity="${{name === "phase_completion" ? 1 : .72}}"/>`;
  }});
  svg.innerHTML = out;
}}
function update() {{
  const rate = rates[Number(slider.value)];
  const row = rows[rate];
  document.getElementById("rateLabel").textContent = `Selected corruption: ${{Math.round(parseFloat(rate)*100)}}%`;
  document.getElementById("metrics").innerHTML = names.slice(0,4).map(name => `<div><span>${{labels[name]}}</span><strong>${{(row[name].accuracy*100).toFixed(1)}}%</strong></div>`).join("");
}}
slider.max = String(rates.length - 1);
slider.value = String(Math.min(3, rates.length - 1));
slider.addEventListener("input", update);
draw(); update();
</script>
</body>
</html>"""


def markdown_table_rows(payload: dict[str, Any]) -> str:
    lines = [
        "<table><thead><tr><th>Corruption</th><th>PhaseMesh</th><th>Whole-key</th><th>Hash</th><th>Bloom</th><th>BM25</th><th>Vector/FAISS</th><th>N-gram</th><th>Reservoir</th></tr></thead><tbody>"
    ]
    for rate in sorted(payload["corruption_curve"]["by_rate"]):
        row = payload["corruption_curve"]["by_rate"][rate]
        lines.append(
            "<tr>"
            f"<td>{float(rate):.0%}</td>"
            f"<td>{row['phase_completion']['accuracy']:.3f}</td>"
            f"<td>{row['whole_key_phase']['accuracy']:.3f}</td>"
            f"<td>{row['exact_hash']['accuracy']:.3f}</td>"
            f"<td>{row['exact_bloom']['accuracy']:.3f}</td>"
            f"<td>{row['bm25']['accuracy']:.3f}</td>"
            f"<td>{row['vector_faiss']['accuracy']:.3f}</td>"
            f"<td>{row['ngram_hash']['accuracy']:.3f}</td>"
            f"<td>{row['random_reservoir']['accuracy']:.3f}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "".join(lines)


def render_architecture_html(architecture: dict[str, Any]) -> str:
    if not architecture:
        return "<p>Skipped.</p>"
    if "results" not in architecture:
        return f"<p>Status: {html.escape(str(architecture.get('status')))}. {html.escape(str(architecture.get('error', '')))}</p>"
    rows = []
    for name in ("learned_phase", "frozen_phase", "bag_mlp"):
        result = architecture["results"][name]
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{result['test_accuracy']:.3f}</td><td>{result['trainable_parameter_count']}</td></tr>"
        )
    return "<table><thead><tr><th>Model</th><th>Test Acc</th><th>Trainable Params</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def digest_bytes(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
