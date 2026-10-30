from __future__ import annotations

import hashlib
import html
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class PatternRecord:
    key_tokens: tuple[str, ...]
    value: str

    @property
    def key(self) -> str:
        return "|".join(self.key_tokens)


class DistributedPhaseAssociativeMemory:
    """Superposed feature-to-value phase memory with partial-key completion."""

    def __init__(
        self,
        *,
        size: int = 4096,
        slots: int = 4,
        seed: int = 7,
        feature_mode: str = "distributed",
    ) -> None:
        self.size = int(size)
        self.slots = int(slots)
        self.seed = int(seed)
        self.feature_mode = str(feature_mode)
        self.memory = np.zeros(self.size, dtype=np.complex64)
        self._vector_cache: dict[str, np.ndarray] = {}
        self.records = 0

    @property
    def memory_bytes(self) -> int:
        return int(self.memory.nbytes)

    def add(self, key_tokens: Sequence[str], value: str) -> None:
        features = self.key_features(key_tokens)
        scale = 1.0 / max(1, len(features))
        for feature in features:
            self.memory += np.asarray(scale * self.binding_vector(feature, value), dtype=np.complex64)
        self.records += 1

    def normalize(self) -> None:
        norm = float(np.linalg.norm(self.memory))
        if norm > 0:
            self.memory = np.asarray(self.memory / norm, dtype=np.complex64)

    def key_features(self, key_tokens: Sequence[str]) -> list[str]:
        tokens = [str(token) for token in key_tokens]
        if self.feature_mode == "whole":
            return ["whole:" + "|".join(tokens)]
        features = [f"pos:{index}:{token}" for index, token in enumerate(tokens)]
        features.extend(f"tok:{token}" for token in tokens)
        return features

    def score(self, key_tokens: Sequence[str], value: str) -> float:
        probe = np.zeros(self.size, dtype=np.complex64)
        features = self.key_features(key_tokens)
        for feature in features:
            probe += self.binding_vector(feature, value)
        if features:
            probe = np.asarray(probe / len(features), dtype=np.complex64)
        denominator = float(np.linalg.norm(probe) * np.linalg.norm(self.memory))
        if denominator <= 0:
            return 0.0
        return float(np.real(np.vdot(probe, self.memory)) / denominator)

    def predict(self, key_tokens: Sequence[str], candidates: Sequence[str]) -> str | None:
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: self.score(key_tokens, candidate))

    def binding_vector(self, feature: str, value: str) -> np.ndarray:
        cache_key = f"{feature}->{value}"
        cached = self._vector_cache.get(cache_key)
        if cached is not None:
            return cached
        vector = np.zeros(self.size, dtype=np.complex64)
        digest = _digest(f"phase:{self.seed}:{feature}:{value}")
        cursor = 0
        for _slot in range(self.slots):
            if cursor + 5 > len(digest):
                digest = _digest(digest.hex())
                cursor = 0
            index = int.from_bytes(digest[cursor:cursor + 4], "big") % self.size
            phase = (digest[cursor + 4] / 255.0) * TWO_PI
            vector[index] += np.complex64(math.cos(phase) + 1j * math.sin(phase))
            cursor += 5
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = np.asarray(vector / norm, dtype=np.complex64)
        self._vector_cache[cache_key] = vector
        return vector


class ExactHashBaseline:
    def __init__(self, *, max_entries: int | None = None) -> None:
        self.max_entries = max_entries
        self.mapping: dict[str, str] = {}

    @property
    def memory_bytes(self) -> int:
        return sum(len(key) + len(value) + 96 for key, value in self.mapping.items())

    def add(self, record: PatternRecord) -> None:
        if self.max_entries is not None and len(self.mapping) >= self.max_entries:
            return
        self.mapping[record.key] = record.value

    def predict(self, key_tokens: Sequence[str], candidates: Sequence[str]) -> str | None:
        value = self.mapping.get("|".join(str(token) for token in key_tokens))
        return value if value in set(candidates) else None


class ExactBloomBaseline:
    def __init__(self, *, bits: int = 32768, slots: int = 4, seed: int = 7) -> None:
        self.bits = max(8, int(bits))
        self.slots = max(1, int(slots))
        self.seed = int(seed)
        self.array = np.zeros(self.bits, dtype=np.bool_)

    @property
    def memory_bytes(self) -> int:
        return int(math.ceil(self.bits / 8))

    def add(self, record: PatternRecord) -> None:
        for index in self.indices(record.key_tokens, record.value):
            self.array[index] = True

    def indices(self, key_tokens: Sequence[str], value: str) -> list[int]:
        digest = _digest(f"bloom:{self.seed}:{'|'.join(key_tokens)}:{value}")
        cursor = 0
        indices = []
        for _slot in range(self.slots):
            if cursor + 4 > len(digest):
                digest = _digest(digest.hex())
                cursor = 0
            indices.append(int.from_bytes(digest[cursor:cursor + 4], "big") % self.bits)
            cursor += 4
        return indices

    def score(self, key_tokens: Sequence[str], value: str) -> float:
        indices = self.indices(key_tokens, value)
        return sum(1 for index in indices if self.array[index]) / len(indices)

    def predict(self, key_tokens: Sequence[str], candidates: Sequence[str]) -> str | None:
        scored = [(candidate, self.score(key_tokens, candidate)) for candidate in candidates]
        best_value, best_score = max(scored, key=lambda item: item[1])
        return best_value if best_score >= 1.0 else None


def run_phase_advantage(
    *,
    out_dir: str | Path = "runs/phase-advantage",
    seed: int = 7,
    items: int = 800,
    key_length: int = 12,
    vocab_size: int = 2000,
    candidates: int = 32,
    trials: int = 300,
    memory_size: int = 4096,
    slots: int = 4,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    records, vocabulary = make_pattern_records(items, key_length, vocab_size, seed=seed)
    corruption = run_corruption_curve(
        records,
        vocabulary,
        candidates=candidates,
        trials=trials,
        memory_size=memory_size,
        slots=slots,
        seed=seed,
    )
    capacity = run_capacity_curve(
        key_length=key_length,
        vocab_size=vocab_size,
        candidates=candidates,
        trials=max(120, min(trials, 240)),
        memory_size=memory_size,
        slots=slots,
        seed=seed,
    )
    segmentation = run_phase_segmentation(seed=seed)
    payload = {
        "type": "phase-mesh-advantage-probes",
        "version": 1,
        "status": "pass" if (
            corruption["by_rate"]["0.30"]["phase_completion"]["accuracy"] >= 0.50
            and corruption["by_rate"]["0.30"]["exact_hash"]["accuracy"] == 0.0
            and corruption["by_rate"]["0.30"]["exact_bloom"]["accuracy"] == 0.0
            and corruption["by_rate"]["0.30"]["whole_key_phase"]["accuracy"] <= 0.10
            and segmentation["coupled"]["pair_accuracy"] >= 0.90
            and segmentation["no_coupling"]["pair_accuracy"] <= 0.65
        ) else "red",
        "elapsed_s": time.perf_counter() - started,
        "config": {
            "seed": seed,
            "items": items,
            "key_length": key_length,
            "vocab_size": vocab_size,
            "candidates": candidates,
            "trials": trials,
            "memory_size": memory_size,
            "slots": slots,
            "phase_memory_bytes": memory_size * 8,
        },
        "corruption_curve": corruption,
        "capacity_curve": capacity,
        "segmentation": segmentation,
        "claim_boundary": [
            "Corruption completion tests partial-key basin overlap; exact maps and exact Bloom filters intentionally get no fuzzy lookup path.",
            "The capacity curve is a fixed-byte synthetic stress test, not a production database benchmark.",
            "The segmentation probe is hand-built phase synchrony dynamics; it is a mechanistic sanity check, not learned object binding.",
        ],
    }
    (out_path / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_path / "summary.md").write_text(render_phase_advantage_markdown(payload), encoding="utf-8")
    (out_path / "index.html").write_text(render_phase_advantage_html(payload), encoding="utf-8")
    return payload


def run_corruption_curve(
    records: Sequence[PatternRecord],
    vocabulary: Sequence[str],
    *,
    candidates: int,
    trials: int,
    memory_size: int,
    slots: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed + 101)
    phase = DistributedPhaseAssociativeMemory(size=memory_size, slots=slots, seed=seed, feature_mode="distributed")
    whole = DistributedPhaseAssociativeMemory(size=memory_size, slots=slots, seed=seed, feature_mode="whole")
    exact = ExactHashBaseline()
    bloom_bits = max(memory_size * 64, len(records) * slots * 512)
    bloom = ExactBloomBaseline(bits=bloom_bits, slots=slots, seed=seed)
    for record in records:
        phase.add(record.key_tokens, record.value)
        whole.add(record.key_tokens, record.value)
        exact.add(record)
        bloom.add(record)
    phase.normalize()
    whole.normalize()
    values = [record.value for record in records]
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    by_rate: dict[str, Any] = {}
    for rate in rates:
        rows = []
        subset = list(records[:min(trials, len(records))])
        for record in subset:
            query = corrupt_key(record.key_tokens, vocabulary, rate, rng)
            candidate_values = make_candidates(record.value, values, candidates, rng)
            rows.append(evaluate_corruption_row(record, query, candidate_values, phase, whole, exact, bloom))
        by_rate[f"{rate:.2f}"] = summarize_model_rows(rows)
    return {
        "task": "corrupted_key_pattern_completion",
        "rates": rates,
        "by_rate": by_rate,
        "bloom_control_bits": bloom_bits,
        "models": ["phase_completion", "whole_key_phase", "exact_hash", "exact_bloom"],
    }


def evaluate_corruption_row(
    record: PatternRecord,
    query: tuple[str, ...],
    candidates: Sequence[str],
    phase: DistributedPhaseAssociativeMemory,
    whole: DistributedPhaseAssociativeMemory,
    exact: ExactHashBaseline,
    bloom: ExactBloomBaseline,
) -> dict[str, bool]:
    predictions = {
        "phase_completion": phase.predict(query, candidates),
        "whole_key_phase": whole.predict(query, candidates),
        "exact_hash": exact.predict(query, candidates),
        "exact_bloom": bloom.predict(query, candidates),
    }
    return {name: prediction == record.value for name, prediction in predictions.items()}


def run_capacity_curve(
    *,
    key_length: int,
    vocab_size: int,
    candidates: int,
    trials: int,
    memory_size: int,
    slots: int,
    seed: int,
) -> dict[str, Any]:
    loads = [100, 200, 400, 800, 1200, 1600]
    rows = []
    for load in loads:
        records, vocabulary = make_pattern_records(load, key_length, vocab_size, seed=seed + load)
        rng = random.Random(seed + 503 + load)
        phase = DistributedPhaseAssociativeMemory(size=memory_size, slots=slots, seed=seed, feature_mode="distributed")
        bloom = ExactBloomBaseline(bits=memory_size * 64, slots=slots, seed=seed)
        max_hash_entries = max(1, (memory_size * 8) // (key_length * 6 + 128))
        exact = ExactHashBaseline(max_entries=max_hash_entries)
        for record in records:
            phase.add(record.key_tokens, record.value)
            bloom.add(record)
            exact.add(record)
        phase.normalize()
        values = [record.value for record in records]
        model_hits = {"phase_completion": 0, "exact_hash_budget": 0, "exact_bloom": 0}
        subset = list(records[:min(trials, len(records))])
        for record in subset:
            candidate_values = make_candidates(record.value, values, candidates, rng)
            model_hits["phase_completion"] += phase.predict(record.key_tokens, candidate_values) == record.value
            model_hits["exact_hash_budget"] += exact.predict(record.key_tokens, candidate_values) == record.value
            model_hits["exact_bloom"] += bloom.predict(record.key_tokens, candidate_values) == record.value
        rows.append({
            "items": load,
            "phase_completion": model_hits["phase_completion"] / len(subset),
            "exact_hash_budget": model_hits["exact_hash_budget"] / len(subset),
            "exact_bloom": model_hits["exact_bloom"] / len(subset),
            "phase_memory_bytes": phase.memory_bytes,
            "hash_memory_bytes": exact.memory_bytes,
            "bloom_memory_bytes": bloom.memory_bytes,
            "hash_entries_stored": len(exact.mapping),
        })
    return {
        "task": "fixed_byte_capacity_rolloff",
        "rows": rows,
    }


def run_phase_segmentation(*, seed: int = 7, objects: int = 2, features_per_object: int = 6, steps: int = 120) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    count = objects * features_per_object
    labels = np.repeat(np.arange(objects), features_per_object)
    initial = rng.uniform(-math.pi, math.pi, size=count)

    def simulate(coupled: bool) -> dict[str, Any]:
        theta = initial.copy()
        coupling = np.zeros((count, count), dtype=np.float32)
        if coupled:
            for i in range(count):
                for j in range(count):
                    if i != j and labels[i] == labels[j]:
                        coupling[i, j] = 0.16
        for _ in range(steps):
            delta = theta[None, :] - theta[:, None]
            theta = theta + np.sum(coupling * np.sin(delta), axis=1)
            theta = np.arctan2(np.sin(theta), np.cos(theta))
        similarity = np.cos(theta[:, None] - theta[None, :])
        predicted_same = similarity > 0.88
        true_same = labels[:, None] == labels[None, :]
        pair_mask = ~np.eye(count, dtype=bool)
        pair_accuracy = float(np.mean(predicted_same[pair_mask] == true_same[pair_mask]))
        within = float(np.mean(similarity[(true_same) & pair_mask]))
        across = float(np.mean(similarity[(~true_same) & pair_mask]))
        return {
            "pair_accuracy": pair_accuracy,
            "within_similarity": within,
            "across_similarity": across,
            "phase_snapshot": [float(value) for value in theta],
        }

    return {
        "task": "phase_synchrony_segmentation",
        "objects": objects,
        "features_per_object": features_per_object,
        "coupled": simulate(True),
        "no_coupling": simulate(False),
    }


def summarize_model_rows(rows: Sequence[dict[str, bool]]) -> dict[str, Any]:
    models = rows[0].keys() if rows else []
    return {
        model: {
            "passed": sum(1 for row in rows if row[model]),
            "rows": len(rows),
            "accuracy": sum(1 for row in rows if row[model]) / len(rows) if rows else 0.0,
        }
        for model in models
    }


def make_pattern_records(count: int, key_length: int, vocab_size: int, *, seed: int) -> tuple[list[PatternRecord], list[str]]:
    rng = random.Random(seed)
    vocabulary = [f"k{i:04d}" for i in range(vocab_size)]
    records = [
        PatternRecord(tuple(rng.sample(vocabulary, key_length)), f"v{i:04d}")
        for i in range(count)
    ]
    return records, vocabulary


def corrupt_key(tokens: Sequence[str], vocabulary: Sequence[str], rate: float, rng: random.Random) -> tuple[str, ...]:
    corrupted = list(tokens)
    count = int(round(len(corrupted) * float(rate)))
    if count <= 0:
        return tuple(corrupted)
    for index in rng.sample(range(len(corrupted)), count):
        replacement = rng.choice(vocabulary)
        while replacement == corrupted[index]:
            replacement = rng.choice(vocabulary)
        corrupted[index] = replacement
    return tuple(corrupted)


def make_candidates(correct: str, values: Sequence[str], count: int, rng: random.Random) -> list[str]:
    wrong_pool = [value for value in values if value != correct]
    wrong = rng.sample(wrong_pool, min(max(0, count - 1), len(wrong_pool)))
    while len(wrong) < max(0, count - 1):
        wrong.append(f"v_extra_{rng.randrange(10_000_000)}")
    candidates = [correct, *wrong]
    rng.shuffle(candidates)
    return candidates


def render_phase_advantage_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# PhaseMesh Advantage Probes",
        "",
        "## Corrupted-Key Pattern Completion",
        "",
        "| Corruption | Phase Completion | Whole-Key Phase | Exact Hash | Exact Bloom |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    by_rate = payload["corruption_curve"]["by_rate"]
    for rate in sorted(by_rate.keys()):
        row = by_rate[rate]
        lines.append(
            f"| {float(rate):.0%} | "
            f"{row['phase_completion']['accuracy']:.3f} | "
            f"{row['whole_key_phase']['accuracy']:.3f} | "
            f"{row['exact_hash']['accuracy']:.3f} | "
            f"{row['exact_bloom']['accuracy']:.3f} |"
        )
    lines.extend([
        "",
        "## Fixed-Byte Capacity Rolloff",
        "",
        "| Items | Phase Completion | Hash Budget | Exact Bloom | Hash Entries Stored |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["capacity_curve"]["rows"]:
        lines.append(
            f"| {row['items']} | {row['phase_completion']:.3f} | "
            f"{row['exact_hash_budget']:.3f} | {row['exact_bloom']:.3f} | {row['hash_entries_stored']} |"
        )
    seg = payload["segmentation"]
    lines.extend([
        "",
        "## Phase-Synchrony Segmentation",
        "",
        "| Variant | Pair Accuracy | Within Similarity | Across Similarity |",
        "| --- | ---: | ---: | ---: |",
        f"| coupled | {seg['coupled']['pair_accuracy']:.3f} | {seg['coupled']['within_similarity']:.3f} | {seg['coupled']['across_similarity']:.3f} |",
        f"| no_coupling | {seg['no_coupling']['pair_accuracy']:.3f} | {seg['no_coupling']['within_similarity']:.3f} | {seg['no_coupling']['across_similarity']:.3f} |",
        "",
        "## Claim Boundary",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


def render_phase_advantage_html(payload: dict[str, Any]) -> str:
    by_rate = payload["corruption_curve"]["by_rate"]
    corruption_rows = [
        {
            "rate": float(rate),
            "phase": row["phase_completion"]["accuracy"],
            "whole": row["whole_key_phase"]["accuracy"],
            "hash": row["exact_hash"]["accuracy"],
            "bloom": row["exact_bloom"]["accuracy"],
        }
        for rate, row in sorted(by_rate.items())
    ]
    capacity_rows = payload["capacity_curve"]["rows"]
    seg = payload["segmentation"]
    status = payload["status"].upper()
    status_class = "pass" if payload["status"] == "pass" else "red"
    config = payload["config"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhaseMesh Advantage Probes</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201b;
      --muted: #5d6a62;
      --line: #d6ded8;
      --field: #0f7b63;
      --whole: #a63d40;
      --hash: #49536a;
      --bloom: #8b6f20;
      --bg: #f7f8f4;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 20px 40px; }}
    header {{ display: flex; align-items: end; justify-content: space-between; gap: 18px; border-bottom: 1px solid var(--line); padding-bottom: 18px; }}
    h1 {{ margin: 0; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 26px 0 10px; font-size: 17px; letter-spacing: 0; }}
    p {{ margin: 8px 0; color: var(--muted); max-width: 820px; }}
    .badge {{ display: inline-flex; align-items: center; height: 30px; padding: 0 10px; border-radius: 6px; font-weight: 700; }}
    .badge.pass {{ background: #dcefe8; color: #0b5c47; }}
    .badge.red {{ background: #f4dada; color: #822628; }}
    .grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; margin-top: 18px; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    th, td {{ padding: 8px 6px; border-bottom: 1px solid var(--line); text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px; color: var(--muted); font-size: 12px; margin: 8px 0 4px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
    .note {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line); font-size: 12px; color: var(--muted); }}
    svg {{ width: 100%; height: auto; display: block; }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} header {{ align-items: start; flex-direction: column; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>PhaseMesh Advantage Probes</h1>
      <p>Corrupted-key pattern completion, fixed-byte capacity rolloff, and phase-synchrony binding controls.</p>
    </div>
    <div class="badge {status_class}">{html.escape(status)}</div>
  </header>

  <div class="grid">
    <section>
      <h2>Corrupted-Key Pattern Completion</h2>
      <div class="legend">
        <span><span class="swatch" style="background:var(--field)"></span>Distributed phase</span>
        <span><span class="swatch" style="background:var(--whole)"></span>Whole-key phase ablation</span>
        <span><span class="swatch" style="background:var(--hash)"></span>Exact hash</span>
        <span><span class="swatch" style="background:var(--bloom)"></span>Exact Bloom</span>
      </div>
      {_line_chart(corruption_rows)}
      {_corruption_table(corruption_rows)}
      <p class="note">The exact controls have no fuzzy retrieval path. The distributed phase memory completes from surviving feature overlap.</p>
    </section>

    <section>
      <h2>Run Config</h2>
      <table>
        <tbody>
          <tr><td>Associations</td><td>{config["items"]}</td></tr>
          <tr><td>Tokens/key</td><td>{config["key_length"]}</td></tr>
          <tr><td>Candidates/query</td><td>{config["candidates"]}</td></tr>
          <tr><td>Trials/rate</td><td>{config["trials"]}</td></tr>
          <tr><td>Phase memory</td><td>{config["phase_memory_bytes"] / 1024:.1f} KB</td></tr>
        </tbody>
      </table>
      <h2>Segmentation Control</h2>
      <table>
        <thead><tr><th>Variant</th><th>Pair Acc</th><th>Within</th><th>Across</th></tr></thead>
        <tbody>
          <tr><td>coupled</td><td>{seg["coupled"]["pair_accuracy"]:.3f}</td><td>{seg["coupled"]["within_similarity"]:.3f}</td><td>{seg["coupled"]["across_similarity"]:.3f}</td></tr>
          <tr><td>no coupling</td><td>{seg["no_coupling"]["pair_accuracy"]:.3f}</td><td>{seg["no_coupling"]["within_similarity"]:.3f}</td><td>{seg["no_coupling"]["across_similarity"]:.3f}</td></tr>
        </tbody>
      </table>
    </section>
  </div>

  <section>
    <h2>Fixed-Byte Capacity Rolloff</h2>
    {_capacity_table(capacity_rows)}
    <p class="note">Clean exact-key Bloom remains exact at this byte budget. This row is a load diagnostic, not a Bloom-beating claim.</p>
  </section>

  <section>
    <h2>Claim Boundary</h2>
    <ul>
      {"".join(f"<li>{html.escape(item)}</li>" for item in payload["claim_boundary"])}
    </ul>
  </section>
</main>
</body>
</html>
"""


def _line_chart(rows: Sequence[dict[str, float]]) -> str:
    width = 720
    height = 280
    pad_l = 42
    pad_r = 18
    pad_t = 18
    pad_b = 34

    def point(row: dict[str, float], key: str) -> tuple[float, float]:
        x = pad_l + row["rate"] / 0.4 * (width - pad_l - pad_r)
        y = pad_t + (1.0 - row[key]) * (height - pad_t - pad_b)
        return x, y

    def polyline(key: str, color: str) -> str:
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(row, key) for row in rows))
        circles = "\n".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />'
            for x, y in (point(row, key) for row in rows)
        )
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" />\n{circles}'

    x_ticks = "\n".join(
        f'<text x="{pad_l + rate / 0.4 * (width - pad_l - pad_r):.1f}" y="{height - 8}" text-anchor="middle">{int(rate * 100)}%</text>'
        for rate in [0.0, 0.1, 0.2, 0.3, 0.4]
    )
    y_ticks = "\n".join(
        f'<text x="30" y="{pad_t + (1.0 - value) * (height - pad_t - pad_b) + 4:.1f}" text-anchor="end">{value:.1f}</text>'
        for value in [0.0, 0.5, 1.0]
    )
    grid = "\n".join(
        f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{pad_t + (1.0 - value) * (height - pad_t - pad_b):.1f}" y2="{pad_t + (1.0 - value) * (height - pad_t - pad_b):.1f}" stroke="#d6ded8" />'
        for value in [0.0, 0.5, 1.0]
    )
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="Accuracy by key corruption">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
  {grid}
  <line x1="{pad_l}" x2="{width - pad_r}" y1="{height - pad_b}" y2="{height - pad_b}" stroke="#9ba8a0" />
  <line x1="{pad_l}" x2="{pad_l}" y1="{pad_t}" y2="{height - pad_b}" stroke="#9ba8a0" />
  {x_ticks}
  {y_ticks}
  {polyline("phase", "#0f7b63")}
  {polyline("whole", "#a63d40")}
  {polyline("hash", "#49536a")}
  {polyline("bloom", "#8b6f20")}
</svg>"""


def _corruption_table(rows: Sequence[dict[str, float]]) -> str:
    body = "\n".join(
        "<tr>"
        f"<td>{row['rate']:.0%}</td>"
        f"<td>{row['phase']:.3f}</td>"
        f"<td>{row['whole']:.3f}</td>"
        f"<td>{row['hash']:.3f}</td>"
        f"<td>{row['bloom']:.3f}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<table>
  <thead><tr><th>Corruption</th><th>Phase</th><th>Whole-key</th><th>Hash</th><th>Bloom</th></tr></thead>
  <tbody>{body}</tbody>
</table>"""


def _capacity_table(rows: Sequence[dict[str, Any]]) -> str:
    body = "\n".join(
        "<tr>"
        f"<td>{row['items']}</td>"
        f"<td>{row['phase_completion']:.3f}</td>"
        f"<td>{row['exact_hash_budget']:.3f}</td>"
        f"<td>{row['exact_bloom']:.3f}</td>"
        f"<td>{row['hash_entries_stored']}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<table>
  <thead><tr><th>Items</th><th>Phase</th><th>Hash budget</th><th>Bloom</th><th>Hash entries</th></tr></thead>
  <tbody>{body}</tbody>
</table>"""


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
