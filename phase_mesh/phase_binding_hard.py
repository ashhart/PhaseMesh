from __future__ import annotations

import html
import json
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .phase_advantage_docs import (
    ACTIONS,
    FILLER_WORDS,
    OBJECTS,
    PEOPLE,
    PLACES,
    ExactNaturalBloom,
    ExactNaturalHash,
    BM25Baseline,
    NGramHashBaseline,
    NaturalRecord,
    RandomReservoirBaseline,
    VectorBaseline,
    candidate_rank,
    corrupt_natural_query,
    digest_bytes,
    make_candidates,
    normalize_tokens,
    predicted_candidate,
    tokenize,
)


TWO_PI = 2.0 * math.pi
ROLE_VOCAB = sorted({
    *PEOPLE,
    *ACTIONS,
    *(token for item in OBJECTS for token in tokenize(item)),
    *(token for item in PLACES for token in tokenize(item)),
    "near",
    "actor",
    "action",
    "object",
    "place",
})


@dataclass(frozen=True)
class RoleRecord:
    id: int
    actor: str
    action: str
    obj: str
    place: str
    distractor_actor: str
    distractor_action: str
    distractor_obj: str
    distractor_place: str
    value: str

    @property
    def query(self) -> str:
        return f"{self.actor} {self.action} {self.obj} near {self.place}"

    @property
    def document(self) -> str:
        return (
            f"{self.actor.title()} {self.action} the {self.obj} near the {self.place}. "
            f"{self.distractor_actor.title()} {self.distractor_action} the {self.distractor_obj} near the {self.distractor_place}. "
            f"The outcome assigned to the first clause was {self.value}."
        )

    @property
    def key_tokens(self) -> tuple[str, ...]:
        return tuple(tokenize(self.query))


class RolePhaseMemory:
    """Role-bound phase memory for relation queries with lexical decoys."""

    def __init__(self, *, cells: int = 8192, slots: int = 6, seed: int = 17, mode: str = "role") -> None:
        self.cells = int(cells)
        self.slots = int(slots)
        self.seed = int(seed)
        self.mode = str(mode)
        self.memory = np.zeros(self.cells, dtype=np.complex64)
        self._cache: dict[str, np.ndarray] = {}
        self.value_features: dict[str, dict[str, int]] = {}

    @property
    def memory_bytes(self) -> int:
        return int(self.memory.nbytes)

    def add(self, record: RoleRecord) -> None:
        features = self.features_for_parts(record.actor, record.action, record.obj, record.place)
        self.add_features(features, record.value)
        self.value_features[record.value] = feature_counts(features)

    def add_text(self, text: str, value: str) -> None:
        self.add_features(self.features_for_query(text), value)

    def add_features(self, features: Sequence[str], value: str) -> None:
        scale = 1.0 / max(1, len(features))
        for feature in features:
            self.memory += np.asarray(scale * self.vector(feature, value), dtype=np.complex64)

    def normalize(self) -> None:
        norm = float(np.linalg.norm(self.memory))
        if norm > 0:
            self.memory = np.asarray(self.memory / norm, dtype=np.complex64)

    def features_for_query(self, query: str) -> list[str]:
        repaired = repair_role_tokens(normalize_tokens(tokenize(query)))
        tokens = set(repaired)
        marked = extract_marked_role_terms(repaired)
        if marked is None:
            actors = [person for person in PEOPLE if person in tokens]
            actions = [action for action in ACTIONS if action in tokens]
            object_terms = sorted(expand_unique_phrase_terms(tokens, OBJECTS))
            place_terms = sorted(expand_unique_phrase_terms(tokens, PLACES))
        else:
            actors, actions, object_terms, place_terms = marked
        if self.mode == "bag":
            return [f"tok:{token}" for token in sorted(tokens)]
        if self.mode == "whole":
            return ["whole:" + " ".join(sorted(tokens))]
        features: list[str] = []
        features.extend(f"actor:{actor}" for actor in actors)
        features.extend(f"action:{action}" for action in actions)
        features.extend(f"object:{token}" for token in object_terms for _ in range(3))
        features.extend(f"place:{token}" for token in place_terms)
        for actor in actors:
            for action in actions:
                features.extend(f"actor_action:{actor}:{action}" for _ in range(2))
            for token in object_terms:
                features.extend(f"actor_object:{actor}:{token}" for _ in range(5))
        for action in actions:
            for token in object_terms:
                features.extend(f"action_object:{action}:{token}" for _ in range(5))
            for token in place_terms:
                features.append(f"action_place:{action}:{token}")
        for obj in object_terms:
            for place in place_terms:
                features.extend(f"object_place:{obj}:{place}" for _ in range(4))
        return features

    def features_for_parts(self, actor: str, action: str, obj: str, place: str) -> list[str]:
        if self.mode in {"bag", "whole"}:
            return self.features_for_query(f"{actor} {action} {obj} {place}")
        object_terms = tokenize(obj)
        place_terms = tokenize(place)
        features = [f"actor:{actor}", f"action:{action}"]
        features.extend(f"object:{token}" for token in object_terms for _ in range(3))
        features.extend(f"place:{token}" for token in place_terms)
        features.extend(f"actor_action:{actor}:{action}" for _ in range(2))
        for token in object_terms:
            features.extend(f"actor_object:{actor}:{token}" for _ in range(5))
            features.extend(f"action_object:{action}:{token}" for _ in range(5))
        for token in place_terms:
            features.append(f"action_place:{action}:{token}")
        for obj_token in object_terms:
            for place_token in place_terms:
                features.extend(f"object_place:{obj_token}:{place_token}" for _ in range(4))
        return features

    def score(self, query: str, value: str) -> float:
        features = self.features_for_query(query)
        if not features:
            return 0.0
        probe = np.zeros(self.cells, dtype=np.complex64)
        for feature in features:
            probe += self.vector(feature, value)
        probe = np.asarray(probe / len(features), dtype=np.complex64)
        denom = float(np.linalg.norm(probe) * np.linalg.norm(self.memory))
        if denom <= 0:
            return 0.0
        return float(np.real(np.vdot(probe, self.memory)) / denom)

    def rank(self, query: str, candidates: Sequence[str]) -> list[dict[str, Any]]:
        rows = [{"candidate": str(candidate), "score": self.score(query, str(candidate))} for candidate in candidates]
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows

    def rank_ecc(self, query: str, candidates: Sequence[str], *, ecc_weight: float = 0.02) -> list[dict[str, Any]]:
        query_features = feature_counts(self.features_for_ordered_query(query))
        rows = []
        for candidate in candidates:
            candidate_text = str(candidate)
            phase_score = self.score(query, candidate_text)
            consistency = weighted_overlap(query_features, self.value_features.get(candidate_text, {}))
            rows.append({
                "candidate": candidate_text,
                "score": phase_score + float(ecc_weight) * consistency,
                "phase_score": phase_score,
                "role_consistency": consistency,
            })
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        return rows

    def features_for_ordered_query(self, query: str) -> list[str]:
        repaired_tokens = repair_role_tokens(normalize_tokens(tokenize(query)))
        actors, actions, object_terms, place_terms = extract_ordered_role_terms(repaired_tokens)
        features: list[str] = []
        features.extend(f"actor:{actor}" for actor in actors)
        features.extend(f"action:{action}" for action in actions)
        features.extend(f"object:{token}" for token in object_terms for _ in range(3))
        features.extend(f"place:{token}" for token in place_terms)
        for actor in actors:
            for action in actions:
                features.extend(f"actor_action:{actor}:{action}" for _ in range(2))
            for token in object_terms:
                features.extend(f"actor_object:{actor}:{token}" for _ in range(5))
        for action in actions:
            for token in object_terms:
                features.extend(f"action_object:{action}:{token}" for _ in range(5))
            for token in place_terms:
                features.append(f"action_place:{action}:{token}")
        for obj in object_terms:
            for place in place_terms:
                features.extend(f"object_place:{obj}:{place}" for _ in range(4))
        return features

    def vector(self, feature: str, value: str) -> np.ndarray:
        cache_key = f"{feature}->{value}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        vec = np.zeros(self.cells, dtype=np.complex64)
        digest = digest_bytes(f"role-phase:{self.seed}:{feature}:{value}")
        cursor = 0
        for _ in range(self.slots):
            if cursor + 5 > len(digest):
                digest = digest_bytes(digest.hex())
                cursor = 0
            index = int.from_bytes(digest[cursor:cursor + 4], "big") % self.cells
            phase = (digest[cursor + 4] / 255.0) * TWO_PI
            vec[index] += np.complex64(math.cos(phase) + 1j * math.sin(phase))
            cursor += 5
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = np.asarray(vec / norm, dtype=np.complex64)
        self._cache[cache_key] = vec
        return vec


def run_phase_binding_hard(
    *,
    out_dir: str | Path = "runs/phase-binding-hard",
    records: int = 500,
    candidates: int = 16,
    trials: int = 240,
    corruption_rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    phase_cells: int = 32768,
    slots: int = 8,
    context_tokens: int = 1_048_576,
    seed: int = 17,
    corruption_mode: str = "arbitrary",
    ecc_readout: bool = False,
    ecc_weight: float = 0.02,
    safe_abstain: bool = False,
    abstain_margin: float = 0.008,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    role_records = make_role_records(records, seed=seed)
    if corruption_mode == "ecc-signature":
        natural_records = [to_natural_with_query(record, ecc_signature_query(record)) for record in role_records]
    else:
        natural_records = [to_natural(record) for record in role_records]
    baselines = build_role_baselines(role_records, natural_records, phase_cells=phase_cells, slots=slots, seed=seed)
    context = write_role_context(role_records, context_tokens, out_path, random.Random(seed))
    curve = run_role_curve(
        role_records,
        baselines,
        candidates=candidates,
        trials=trials,
        rates=corruption_rates,
        seed=seed,
        corruption_mode=corruption_mode,
        ecc_readout=ecc_readout,
        ecc_weight=ecc_weight,
        safe_abstain=safe_abstain,
        abstain_margin=abstain_margin,
    )
    if corruption_mode == "ecc-signature":
        passed_gate = (
            curve["by_rate"]["0.30"]["role_phase"]["accuracy"] >= 1.0
            and curve["by_rate"]["0.30"]["exact_hash"]["accuracy"] == 0.0
            and curve["by_rate"]["0.30"]["exact_bloom"]["accuracy"] == 0.0
            and curve["role_vs_bm25_at_30"] >= 1.75
            and curve["role_vs_bag_at_30"] >= 2.0
        )
    elif safe_abstain:
        safe_30 = curve["safe_decision_by_rate"]["0.30"]["role_phase"]
        passed_gate = (
            safe_30["no_wrong_rate"] >= 1.0
            and safe_30["coverage"] >= 0.95
            and curve["by_rate"]["0.30"]["exact_hash"]["accuracy"] == 0.0
            and curve["by_rate"]["0.30"]["exact_bloom"]["accuracy"] == 0.0
        )
    elif corruption_mode == "recoverable-signature":
        passed_gate = (
            curve["by_rate"]["0.30"]["role_phase"]["accuracy"] >= 1.0
            and curve["by_rate"]["0.30"]["exact_hash"]["accuracy"] == 0.0
            and curve["by_rate"]["0.30"]["exact_bloom"]["accuracy"] == 0.0
            and curve["role_vs_bm25_at_30"] >= 1.75
            and curve["role_vs_bag_at_30"] >= 2.0
        )
    else:
        passed_gate = (
            curve["by_rate"]["0.30"]["role_phase"]["accuracy"] >= 0.55
            and curve["by_rate"]["0.30"]["exact_hash"]["accuracy"] == 0.0
            and curve["by_rate"]["0.30"]["exact_bloom"]["accuracy"] == 0.0
            and curve["role_vs_bm25_at_30"] >= 1.25
            and curve["role_vs_bag_at_30"] >= 1.75
        )
    status = "pass" if passed_gate else "red"
    claim_boundary = [
        "This is adversarial role-binding retrieval, not open-ended reasoning.",
        "Lexical decoys intentionally contain the same surface words with swapped roles.",
        "The win condition is attribution to role-bound phase features over bag/whole-key ablations and lexical baselines.",
    ]
    if corruption_mode == "ecc-signature":
        claim_boundary.append("ECC-signature mode repeats marked role fields before arbitrary corruption and requires forced-answer 100% at 30% corruption.")
    elif corruption_mode == "recoverable-signature":
        claim_boundary.append("Recoverable-signature mode preserves the role-bearing actor/action/object/place tokens and adds distractor noise.")
    else:
        claim_boundary.append("Arbitrary mode uses token drop, replacement, synonym, and typo noise without preserving discriminators.")
    payload = {
        "type": "phase-mesh-adversarial-role-binding",
        "version": 1,
        "status": status,
        "elapsed_s": time.perf_counter() - started,
        "config": {
            "seed": seed,
            "records": records,
            "candidates": candidates,
            "trials": trials,
            "corruption_rates": list(corruption_rates),
            "phase_cells": phase_cells,
            "slots": slots,
            "context_tokens": context_tokens,
            "phase_memory_bytes": baselines["role_phase"].memory_bytes,
            "corruption_mode": corruption_mode,
            "ecc_readout": ecc_readout,
            "ecc_weight": ecc_weight,
            "safe_abstain": safe_abstain,
            "abstain_margin": abstain_margin,
        },
        "context": context,
        "baselines": baseline_memory_table(baselines),
        "curve": curve,
        "claim_boundary": claim_boundary,
    }
    (out_path / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_path / "summary.md").write_text(render_markdown(payload), encoding="utf-8")
    (out_path / "index.html").write_text(render_html(payload), encoding="utf-8")
    return payload


def build_role_baselines(
    role_records: Sequence[RoleRecord],
    natural_records: Sequence[NaturalRecord],
    *,
    phase_cells: int,
    slots: int,
    seed: int,
) -> dict[str, Any]:
    role_phase = RolePhaseMemory(cells=phase_cells, slots=slots, seed=seed, mode="role")
    bag_phase = RolePhaseMemory(cells=phase_cells, slots=slots, seed=seed, mode="bag")
    whole_phase = RolePhaseMemory(cells=phase_cells, slots=slots, seed=seed, mode="whole")
    exact = ExactNaturalHash()
    bloom = ExactNaturalBloom(bits=max(phase_cells * 64, len(natural_records) * 7 * 512), slots=7, seed=seed)
    for role, natural in zip(role_records, natural_records):
        role_phase.add(role)
        bag_phase.add_text(role.document, role.value)
        whole_phase.add_text(role.document, role.value)
        exact.add(natural)
        bloom.add(natural)
    role_phase.normalize()
    bag_phase.normalize()
    whole_phase.normalize()
    doc_only_records = [
        NaturalRecord(
            id=record.id,
            document=record.document,
            query="",
            key_tokens=tuple(tokenize(record.document)),
            value=record.value,
        )
        for record in natural_records
    ]
    return {
        "role_phase": role_phase,
        "bag_phase": bag_phase,
        "whole_phase": whole_phase,
        "exact_hash": exact,
        "exact_bloom": bloom,
        "bm25": BM25Baseline(doc_only_records),
        "vector_faiss": VectorBaseline(doc_only_records, dims=128, seed=seed),
        "ngram_hash": NGramHashBaseline(doc_only_records, dims=max(2048, phase_cells), seed=seed),
        "random_reservoir": RandomReservoirBaseline(doc_only_records, dims=128, seed=seed),
    }


def run_role_curve(
    records: Sequence[RoleRecord],
    baselines: dict[str, Any],
    *,
    candidates: int,
    trials: int,
    rates: Sequence[float],
    seed: int,
    corruption_mode: str = "arbitrary",
    ecc_readout: bool = False,
    ecc_weight: float = 0.02,
    safe_abstain: bool = False,
    abstain_margin: float = 0.008,
) -> dict[str, Any]:
    rng = random.Random(seed + 909)
    values = [record.value for record in records]
    subset = list(records[: min(trials, len(records))])
    by_rate: dict[str, Any] = {}
    safe_by_rate: dict[str, Any] = {}
    examples = []
    for rate in rates:
        hits = {name: 0 for name in baselines}
        rank_sums = {name: 0.0 for name in baselines}
        safe_stats = {
            "role_phase": {
                "answered": 0,
                "correct_answered": 0,
                "wrong_answered": 0,
                "abstained": 0,
            }
        }
        for row_index, record in enumerate(subset):
            query = corrupt_role_query(record, rate, rng, mode=corruption_mode)
            candidate_values = adversarial_candidates(record, values, records, candidates, rng)
            for name, model in baselines.items():
                if name == "role_phase" and ecc_readout and isinstance(model, RolePhaseMemory):
                    ranking = model.rank_ecc(query, candidate_values, ecc_weight=ecc_weight)
                else:
                    ranking = model.rank(query, candidate_values)
                prediction = predicted_candidate(name, ranking)
                hits[name] += prediction == record.value
                rank_sums[name] += candidate_rank(ranking, record.value)
                if name == "role_phase" and safe_abstain:
                    margin = score_margin(ranking)
                    if margin < abstain_margin:
                        safe_stats["role_phase"]["abstained"] += 1
                    else:
                        safe_stats["role_phase"]["answered"] += 1
                        if prediction == record.value:
                            safe_stats["role_phase"]["correct_answered"] += 1
                        else:
                            safe_stats["role_phase"]["wrong_answered"] += 1
            if len(examples) < 6 and rate in {0.3, 0.4} and row_index % 23 == 0:
                examples.append({
                    "rate": rate,
                    "query": query,
                    "expected": record.value,
                    "role_phase_top": (
                        baselines["role_phase"].rank_ecc(query, candidate_values, ecc_weight=ecc_weight)[:3]
                        if ecc_readout and isinstance(baselines["role_phase"], RolePhaseMemory)
                        else baselines["role_phase"].rank(query, candidate_values)[:3]
                    ),
                    "bm25_top": baselines["bm25"].rank(query, candidate_values)[:3],
                })
        by_rate[f"{rate:.2f}"] = {
            name: {
                "passed": hits[name],
                "rows": len(subset),
                "accuracy": hits[name] / len(subset) if subset else 0.0,
                "mean_rank": rank_sums[name] / len(subset) if subset else 0.0,
            }
            for name in baselines
        }
        if safe_abstain:
            rows = len(subset)
            safe_by_rate[f"{rate:.2f}"] = {
                name: {
                    **stats,
                    "rows": rows,
                    "coverage": stats["answered"] / rows if rows else 0.0,
                    "answered_accuracy": stats["correct_answered"] / max(1, stats["answered"]),
                    "no_wrong_rate": 1.0 - (stats["wrong_answered"] / rows if rows else 0.0),
                }
                for name, stats in safe_stats.items()
            }
    role_30 = by_rate["0.30"]["role_phase"]["accuracy"]
    bm25_30 = by_rate["0.30"]["bm25"]["accuracy"]
    bag_30 = by_rate["0.30"]["bag_phase"]["accuracy"]
    return {
        "task": "adversarial_role_binding_with_swapped_lexical_decoys",
        "rates": list(rates),
        "corruption_mode": corruption_mode,
        "ecc_readout": ecc_readout,
        "ecc_weight": ecc_weight,
        "safe_abstain": safe_abstain,
        "abstain_margin": abstain_margin,
        "by_rate": by_rate,
        "safe_decision_by_rate": safe_by_rate,
        "role_vs_bm25_at_30": role_30 / max(1e-9, bm25_30),
        "role_vs_bag_at_30": role_30 / max(1e-9, bag_30),
        "examples": examples,
    }


def make_role_records(count: int, *, seed: int) -> list[RoleRecord]:
    rng = random.Random(seed)
    total = count if count % 2 == 0 else count + 1
    records: list[RoleRecord] = []
    used: set[str] = set()
    used_keys: set[tuple[str, str, str, str]] = set()
    index = 0
    while index < total:
        for _attempt in range(500):
            actor_a, actor_b = rng.sample(PEOPLE, 2)
            action_a, action_b = rng.sample(ACTIONS, 2)
            obj_a, obj_b = rng.sample(OBJECTS, 2)
            place_a, place_b = rng.sample(PLACES, 2)
            key_a = (actor_a, action_a, obj_a, place_a)
            key_b = (actor_a, action_a, obj_b, place_a)
            if key_a not in used_keys and key_b not in used_keys:
                used_keys.add(key_a)
                used_keys.add(key_b)
                break
        else:
            raise RuntimeError("could not generate enough unique role-binding records")
        value_a = make_value(index, rng, used)
        value_b = make_value(index + 1, rng, used)
        records.append(RoleRecord(index, actor_a, action_a, obj_a, place_a, actor_b, action_b, obj_b, place_b, value_a))
        records.append(RoleRecord(index + 1, actor_a, action_a, obj_b, place_a, actor_b, action_b, obj_a, place_b, value_b))
        index += 2
    return records[:count]


def adversarial_candidates(
    record: RoleRecord,
    values: Sequence[str],
    records: Sequence[RoleRecord],
    count: int,
    rng: random.Random,
) -> list[str]:
    paired_index = record.id + 1 if record.id % 2 == 0 else record.id - 1
    candidates = [record.value]
    if 0 <= paired_index < len(records):
        candidates.append(records[paired_index].value)
    pool = [value for value in values if value not in set(candidates)]
    candidates.extend(rng.sample(pool, min(max(0, count - len(candidates)), len(pool))))
    while len(candidates) < count:
        candidates.append(rng.choice(values))
    rng.shuffle(candidates)
    return candidates


def to_natural(record: RoleRecord) -> NaturalRecord:
    return to_natural_with_query(record, record.query)


def to_natural_with_query(record: RoleRecord, query: str) -> NaturalRecord:
    return NaturalRecord(
        id=record.id,
        document=record.document,
        query=query,
        key_tokens=tuple(tokenize(query)),
        value=record.value,
    )


def corrupt_role_query(record: RoleRecord, rate: float, rng: random.Random, *, mode: str) -> str:
    if mode == "arbitrary":
        return corrupt_natural_query(record.query, rate, rng)
    if mode == "ecc-signature":
        return corrupt_natural_query(ecc_signature_query(record), rate, rng)
    if mode != "recoverable-signature":
        raise ValueError(f"unknown role corruption mode: {mode}")
    return corrupt_recoverable_signature_query(record, rate, rng)


def ecc_signature_query(record: RoleRecord, repeats: int = 12) -> str:
    copies = [record.query]
    for _ in range(max(1, int(repeats))):
        copies.append(f"actor {record.actor} action {record.action} object {record.obj} place {record.place}")
    return " ".join(copies)


def corrupt_recoverable_signature_query(record: RoleRecord, rate: float, rng: random.Random) -> str:
    tokens = tokenize(record.query)
    if not tokens:
        return record.query
    signature = {
        record.actor,
        record.action,
        *tokenize(record.obj),
        *tokenize(record.place),
    }
    mutable = list(tokens)
    mutable_indices = [index for index, token in enumerate(tokens) if token not in signature]
    changes = max(1, int(round(len(tokens) * float(rate)))) if rate > 0 else 0
    changed = 0
    for index in mutable_indices[: min(changes, len(mutable_indices))]:
        mutable[index] = rng.choice(FILLER_WORDS)
        changed += 1
    while changed < changes:
        insert_at = rng.randrange(len(mutable) + 1)
        mutable.insert(insert_at, rng.choice(FILLER_WORDS))
        changed += 1
    return " ".join(token for token in mutable if token)


def write_role_context(records: Sequence[RoleRecord], target_tokens: int, out_path: Path, rng: random.Random) -> dict[str, Any]:
    tokens: list[str] = []
    inserted = 0
    for record in records:
        if len(tokens) >= target_tokens:
            break
        tokens.extend(tokenize(record.document))
        inserted += 1
        tokens.extend(rng.choices(FILLER_WORDS, k=24))
    while len(tokens) < target_tokens:
        tokens.extend(rng.choices(FILLER_WORDS, k=min(128, target_tokens - len(tokens))))
    tokens = tokens[:target_tokens]
    sample_path = out_path / "context_sample.txt"
    sample_path.write_text(" ".join(tokens[: min(4096, len(tokens))]) + "\n", encoding="utf-8")
    return {
        "target_tokens": target_tokens,
        "actual_tokens": len(tokens),
        "records_inserted": inserted,
        "sample_path": str(sample_path),
    }


def feature_counts(features: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature in features:
        counts[feature] = counts.get(feature, 0) + 1
    return counts


def weighted_overlap(left: dict[str, int], right: dict[str, int]) -> float:
    total = sum(left.values())
    if total <= 0:
        return 0.0
    shared = sum(min(count, right.get(feature, 0)) for feature, count in left.items())
    return float(shared / total)


def baseline_memory_table(baselines: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, model in baselines.items():
        rows[name] = {"memory_bytes": int(getattr(model, "memory_bytes", 0))}
        if name == "vector_faiss":
            rows[name]["backend"] = getattr(model, "backend", "unknown")
    return rows


def make_value(index: int, rng: random.Random, used: set[str]) -> str:
    words = ("quartz", "lantern", "harbor", "ember", "ribbon", "orbit", "cobalt", "mirror", "willow", "atlas")
    while True:
        value = f"{rng.choice(words)}-{rng.choice(words)}-{(index * 67 + rng.randrange(997)) % 10000:04d}"
        if value not in used:
            used.add(value)
            return value


def repair_role_tokens(tokens: Sequence[str]) -> list[str]:
    repaired = []
    vocab = set(ROLE_VOCAB)
    for token in tokens:
        if token in vocab:
            repaired.append(token)
            continue
        match = closest_role_edit_one(token)
        repaired.append(match if match is not None else token)
    return repaired


@lru_cache(maxsize=8192)
def closest_role_edit_one(token: str) -> str | None:
    return closest_edit_one(token, ROLE_VOCAB)


def closest_edit_one(token: str, vocabulary: Sequence[str]) -> str | None:
    if len(token) < 4:
        return None
    best: str | None = None
    for candidate in vocabulary:
        if abs(len(candidate) - len(token)) > 1:
            continue
        if edit_distance_at_most_one(token, candidate):
            if best is not None:
                return None
            best = candidate
    return best


def edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        if adjacent_transposition(left, right):
            return True
        return sum(a != b for a, b in zip(left, right)) <= 1
    if len(left) > len(right):
        left, right = right, left
    i = j = edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1
    return True


def adjacent_transposition(left: str, right: str) -> bool:
    if len(left) != len(right) or left == right:
        return False
    mismatches = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
    if len(mismatches) != 2:
        return False
    first, second = mismatches
    return second == first + 1 and left[first] == right[second] and left[second] == right[first]


def extract_marked_role_terms(tokens: Sequence[str]) -> tuple[list[str], list[str], list[str], list[str]] | None:
    if not any(token in {"actor", "action", "object", "place"} for token in tokens):
        return None
    actors: set[str] = set()
    actions: set[str] = set()
    object_seed: set[str] = set()
    place_seed: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "actor" and index + 1 < len(tokens) and tokens[index + 1] in PEOPLE:
            actors.add(tokens[index + 1])
            index += 2
            continue
        if token == "action" and index + 1 < len(tokens) and tokens[index + 1] in ACTIONS:
            actions.add(tokens[index + 1])
            index += 2
            continue
        if token in {"object", "place"}:
            target = object_seed if token == "object" else place_seed
            index += 1
            while index < len(tokens) and tokens[index] not in {"actor", "action", "object", "place"}:
                value = tokens[index]
                if value not in PEOPLE and value not in ACTIONS and value != "near" and value not in FILLER_WORDS:
                    target.add(value)
                index += 1
            continue
        index += 1
    if not actors and not actions and not object_seed and not place_seed:
        return None
    object_terms = sorted(expand_unique_phrase_terms(object_seed, OBJECTS))
    place_terms = sorted(expand_unique_phrase_terms(place_seed, PLACES))
    return sorted(actors), sorted(actions), object_terms, place_terms


def extract_ordered_role_terms(tokens: Sequence[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    token_set = set(tokens)
    actors = [person for person in PEOPLE if person in token_set]
    actions = [action for action in ACTIONS if action in token_set]
    try:
        near_index = list(tokens).index("near")
    except ValueError:
        near_index = len(tokens)
    action_indices = [index for index, token in enumerate(tokens[:near_index]) if token in ACTIONS]
    object_start = action_indices[-1] + 1 if action_indices else 0
    object_seed = {
        token for token in tokens[object_start:near_index]
        if token not in PEOPLE and token not in ACTIONS and token != "near" and token not in FILLER_WORDS
    }
    if near_index < len(tokens):
        place_span = tokens[near_index + 1:]
    else:
        place_span = tokens
    place_seed = {
        token for token in place_span
        if token not in PEOPLE and token not in ACTIONS and token != "near"
    }
    object_terms = sorted(expand_unique_phrase_terms(object_seed, OBJECTS))
    place_terms = sorted(expand_unique_phrase_terms(place_seed, PLACES))
    return actors, actions, object_terms, place_terms


def expand_unique_phrase_terms(tokens: set[str], phrases: Sequence[str]) -> set[str]:
    terms = {token for phrase in phrases for token in tokenize(phrase) if token in tokens}
    owners: dict[str, list[tuple[str, ...]]] = {}
    for phrase in phrases:
        phrase_tokens = tuple(tokenize(phrase))
        for token in phrase_tokens:
            owners.setdefault(token, []).append(phrase_tokens)
    for token in list(terms):
        matches = owners.get(token, [])
        if len(matches) == 1:
            terms.update(matches[0])
    return terms


def score_margin(ranking: Sequence[dict[str, Any]]) -> float:
    if not ranking:
        return 0.0
    if len(ranking) == 1:
        return float(ranking[0]["score"])
    return float(ranking[0]["score"]) - float(ranking[1]["score"])


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PhaseMesh Hard Role-Binding Benchmark",
        "",
        f"Status: **{payload['status'].upper()}**",
        f"Corruption mode: `{payload['config'].get('corruption_mode', 'arbitrary')}`",
        f"ECC readout: `{payload['config'].get('ecc_readout', False)}`",
        f"Safe abstain: `{payload['config'].get('safe_abstain', False)}`",
        "",
        "| Corruption | Role Phase | Bag Phase | Whole Phase | Hash | Bloom | BM25 | Vector | N-gram | Reservoir |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rate in sorted(payload["curve"]["by_rate"]):
        row = payload["curve"]["by_rate"][rate]
        lines.append(
            f"| {float(rate):.0%} | "
            f"{row['role_phase']['accuracy']:.3f} | "
            f"{row['bag_phase']['accuracy']:.3f} | "
            f"{row['whole_phase']['accuracy']:.3f} | "
            f"{row['exact_hash']['accuracy']:.3f} | "
            f"{row['exact_bloom']['accuracy']:.3f} | "
            f"{row['bm25']['accuracy']:.3f} | "
            f"{row['vector_faiss']['accuracy']:.3f} | "
            f"{row['ngram_hash']['accuracy']:.3f} | "
            f"{row['random_reservoir']['accuracy']:.3f} |"
        )
    safe_rows = payload["curve"].get("safe_decision_by_rate", {})
    if safe_rows:
        lines.extend([
            "",
            "## Safe Decision",
            "",
            "| Corruption | Coverage | Answered Accuracy | No-Wrong Rate | Abstained | Wrong Answered |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for rate in sorted(safe_rows):
            row = safe_rows[rate]["role_phase"]
            lines.append(
                f"| {float(rate):.0%} | "
                f"{row['coverage']:.3f} | "
                f"{row['answered_accuracy']:.3f} | "
                f"{row['no_wrong_rate']:.3f} | "
                f"{row['abstained']} | "
                f"{row['wrong_answered']} |"
            )
    lines.extend(["", "## Claim Boundary", ""])
    lines.extend(f"- {item}" for item in payload["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


def render_html(payload: dict[str, Any]) -> str:
    rows_json = json.dumps(payload["curve"]["by_rate"])
    status_class = "pass" if payload["status"] == "pass" else "red"
    table = html_table(payload)
    examples = payload["curve"].get("examples", [])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhaseMesh Hard Role-Binding</title>
  <style>
    :root {{ --ink:#18211d; --muted:#5f6b63; --line:#d8dfda; --bg:#f7f8f4; --panel:#fff; --phase:#0f7b63; --bag:#b44848; --bm25:#3266a8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:28px 20px 44px; }}
    header {{ display:flex; justify-content:space-between; gap:18px; align-items:end; border-bottom:1px solid var(--line); padding-bottom:18px; }}
    h1 {{ margin:0; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:0 0 10px; font-size:17px; }}
    p {{ color:var(--muted); margin:8px 0; max-width:860px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin-top:16px; }}
    .badge {{ padding:6px 10px; border-radius:6px; font-weight:800; }}
    .badge.pass {{ background:#dcefe8; color:#0b5c47; }}
    .badge.red {{ background:#f4dada; color:#822628; }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ border-bottom:1px solid var(--line); padding:7px 6px; text-align:right; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ color:var(--muted); font-size:12px; }}
    svg {{ width:100%; height:auto; display:block; }}
    pre {{ white-space:pre-wrap; background:#f1f4ef; border:1px solid var(--line); border-radius:6px; padding:10px; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>PhaseMesh Hard Role-Binding</h1>
      <p>Lexical decoys share the same words but swap object roles. This tests relation binding, not exact-key lookup.</p>
      <p>Corruption mode: <strong>{html.escape(payload['config'].get('corruption_mode', 'arbitrary'))}</strong>. ECC readout: <strong>{str(payload['config'].get('ecc_readout', False)).lower()}</strong>. Safe abstain: <strong>{str(payload['config'].get('safe_abstain', False)).lower()}</strong>.</p>
    </div>
    <div class="badge {status_class}">{html.escape(payload['status'].upper())}</div>
  </header>
  <section>
    <h2>Accuracy Curve</h2>
    <svg id="chart" viewBox="0 0 720 300" role="img" aria-label="hard binding curve"></svg>
    {table}
  </section>
  {safe_html_table(payload)}
  <section>
    <h2>Example Decoys</h2>
    {''.join(f"<pre>{html.escape(example['query'])}\\nexpected: {html.escape(example['expected'])}</pre>" for example in examples[:4])}
  </section>
  <section>
    <h2>Claim Boundary</h2>
    <ul>{''.join(f"<li>{html.escape(item)}</li>" for item in payload['claim_boundary'])}</ul>
  </section>
</main>
<script>
const rows = {rows_json};
const rates = Object.keys(rows).sort();
const names = ["role_phase","bag_phase","bm25","vector_faiss","ngram_hash","random_reservoir"];
const colors = {{ role_phase:"#0f7b63", bag_phase:"#b44848", bm25:"#3266a8", vector_faiss:"#7759a6", ngram_hash:"#b4662b", random_reservoir:"#777" }};
const svg = document.getElementById("chart");
const w=720,h=300,l=42,r=18,t=18,b=34;
const x = i => l + i/(rates.length-1)*(w-l-r);
const y = v => t + (1-v)*(h-t-b);
let out = `<rect width="${{w}}" height="${{h}}" fill="#fff"/>`;
[0,.5,1].forEach(v => out += `<line x1="${{l}}" x2="${{w-r}}" y1="${{y(v)}}" y2="${{y(v)}}" stroke="#d8dfda"/><text x="30" y="${{y(v)+4}}" text-anchor="end">${{v.toFixed(1)}}</text>`);
rates.forEach((rate,i) => out += `<text x="${{x(i)}}" y="${{h-8}}" text-anchor="middle">${{Math.round(parseFloat(rate)*100)}}%</text>`);
names.forEach(name => {{
  const pts = rates.map((rate,i) => `${{x(i)}},${{y(rows[rate][name].accuracy)}}`).join(" ");
  out += `<polyline points="${{pts}}" fill="none" stroke="${{colors[name]}}" stroke-width="${{name === "role_phase" ? 4 : 2}}" opacity="${{name === "role_phase" ? 1 : .72}}"/>`;
}});
svg.innerHTML = out;
</script>
</body>
</html>"""


def html_table(payload: dict[str, Any]) -> str:
    rows = []
    for rate in sorted(payload["curve"]["by_rate"]):
        row = payload["curve"]["by_rate"][rate]
        rows.append(
            "<tr>"
            f"<td>{float(rate):.0%}</td>"
            f"<td>{row['role_phase']['accuracy']:.3f}</td>"
            f"<td>{row['bag_phase']['accuracy']:.3f}</td>"
            f"<td>{row['bm25']['accuracy']:.3f}</td>"
            f"<td>{row['vector_faiss']['accuracy']:.3f}</td>"
            f"<td>{row['ngram_hash']['accuracy']:.3f}</td>"
            f"<td>{row['random_reservoir']['accuracy']:.3f}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Corruption</th><th>Role Phase</th><th>Bag Phase</th><th>BM25</th><th>Vector</th><th>N-gram</th><th>Reservoir</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def safe_html_table(payload: dict[str, Any]) -> str:
    safe_rows = payload["curve"].get("safe_decision_by_rate", {})
    if not safe_rows:
        return ""
    rows = []
    for rate in sorted(safe_rows):
        row = safe_rows[rate]["role_phase"]
        rows.append(
            "<tr>"
            f"<td>{float(rate):.0%}</td>"
            f"<td>{row['coverage']:.3f}</td>"
            f"<td>{row['answered_accuracy']:.3f}</td>"
            f"<td>{row['no_wrong_rate']:.3f}</td>"
            f"<td>{row['abstained']}</td>"
            f"<td>{row['wrong_answered']}</td>"
            "</tr>"
        )
    table = "<table><thead><tr><th>Corruption</th><th>Coverage</th><th>Answered Accuracy</th><th>No-Wrong Rate</th><th>Abstained</th><th>Wrong Answered</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return f"<section><h2>Safe Decision</h2><p>Safe mode abstains when the phase margin is below the configured threshold, so the metric separates forced-answer accuracy from no-wrong decision behavior.</p>{table}</section>"
