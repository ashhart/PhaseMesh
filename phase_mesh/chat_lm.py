from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .language_model import PhaseLMConfig, PhaseLanguageModel, tokenize


@dataclass
class PhaseChatConfig:
    signature_cells: int = 4096
    seed: int = 7
    retrieval_threshold: float = 0.18
    topic_coverage_threshold: float = 0.66
    fallback_max_tokens: int = 64


@dataclass
class PhaseChatRecord:
    prompt: str
    response: str
    source: str = "teacher"


class PhaseChatModel:
    """Prompt-conditioned PhaseMesh LM surface.

    The next-token PhaseLanguageModel is good at local continuation but weak at
    selecting which teacher answer belongs to which prompt. This layer stores
    prompt signatures and returns the response whose phase signature resonates
    most with the active prompt, with a PhaseLanguageModel fallback.
    """

    manifest_name = "chat_model.json"

    def __init__(
        self,
        config: PhaseChatConfig | None = None,
        *,
        records: list[PhaseChatRecord] | None = None,
        language_model: PhaseLanguageModel | None = None,
    ) -> None:
        self.config = config or PhaseChatConfig()
        self.records = records or []
        self.language_model = language_model or PhaseLanguageModel(
            PhaseLMConfig(
                order=4,
                phase_cells=max(2048, int(self.config.signature_cells) * 4),
                vocab_size=16384,
                seed=int(self.config.seed),
                phase_weight=0.05,
                ngram_weight=2.5,
                unigram_weight=0.1,
            )
        )
        self.signatures = np.zeros((0, int(self.config.signature_cells)), dtype=np.complex64)
        if self.records:
            self._rebuild_signatures()

    @classmethod
    def from_teacher_samples(
        cls,
        samples_path: str | Path,
        *,
        config: PhaseChatConfig | None = None,
        max_records: int | None = None,
    ) -> "PhaseChatModel":
        records: list[PhaseChatRecord] = []
        for line in Path(samples_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            response = _extract_assistant_response(str(row.get("text", "")).strip())
            if prompt and response:
                records.append(PhaseChatRecord(prompt=prompt, response=response, source=str(row.get("teacher_model", "teacher"))))
            if max_records is not None and len(records) >= int(max_records):
                break
        model = cls(config=config, records=records)
        model.train_fallback()
        return model

    def add(self, prompt: str, response: str, *, source: str = "user") -> None:
        self.records.append(PhaseChatRecord(prompt=str(prompt), response=str(response), source=str(source)))
        self._rebuild_signatures()
        self.train_fallback()

    def train_fallback(self) -> dict[str, Any]:
        texts = [f"user\n{record.prompt}\nassistant\n{record.response}" for record in self.records]
        self.language_model = PhaseLanguageModel(self.language_model.config)
        summary = self.language_model.train_lines(texts)
        return summary

    def answer(
        self,
        prompt: str,
        *,
        top_k: int = 3,
        threshold: float | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        allow_fallback: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        query = self._signature(prompt)
        ranks = self.rank(prompt, top_k=max(1, int(top_k)), query_signature=query)
        cutoff = self.config.retrieval_threshold if threshold is None else float(threshold)
        margin = float(ranks[0]["score"] - ranks[1]["score"]) if len(ranks) > 1 else float(ranks[0]["score"]) if ranks else 0.0
        if ranks:
            ranks[0]["margin"] = margin
        if (
            ranks
            and ranks[0]["score"] >= cutoff
            and ranks[0].get("topic_coverage", 0.0) >= float(self.config.topic_coverage_threshold)
        ):
            record = self.records[int(ranks[0]["index"])]
            return {
                "prompt": prompt,
                "completion": record.response,
                "text": record.response,
                "tokens": tokenize(record.response),
                "mode": "phase-chat-retrieval",
                "score": float(ranks[0]["score"]),
                "confidence": {
                    "score": float(ranks[0]["score"]),
                    "margin": margin,
                    "topic_coverage": float(ranks[0].get("topic_coverage", 0.0)),
                    "threshold": cutoff,
                    "topic_coverage_threshold": float(self.config.topic_coverage_threshold),
                },
                "retrieval": ranks,
                "model": self.summary(),
            }
        confidence = {
            "score": float(ranks[0]["score"]) if ranks else 0.0,
            "margin": margin,
            "topic_coverage": float(ranks[0].get("topic_coverage", 0.0)) if ranks else 0.0,
            "threshold": cutoff,
            "topic_coverage_threshold": float(self.config.topic_coverage_threshold),
        }
        if not allow_fallback:
            return {
                "prompt": prompt,
                "completion": "",
                "text": "",
                "tokens": [],
                "mode": "phase-chat-abstain",
                "score": confidence["score"],
                "confidence": confidence,
                "retrieval": ranks,
                "model": self.summary(),
            }
        generated = self.language_model.generate(
            prompt,
            max_tokens=max_tokens or int(self.config.fallback_max_tokens),
            temperature=temperature,
            top_k=24,
        )
        generated["mode"] = "phase-chat-fallback"
        generated["score"] = confidence["score"]
        generated["confidence"] = confidence
        generated["retrieval"] = ranks
        generated["model"] = self.summary()
        return generated

    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return self.answer(prompt, **kwargs)

    def rank(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        query_signature: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        if self.signatures.size == 0:
            return []
        query = query_signature if query_signature is not None else self._signature(prompt)
        phase_scores = np.real(self.signatures @ np.conj(query)).astype(np.float32)
        scores = np.asarray([
            0.45 * float(phase_scores[index]) + 0.55 * _lexical_score(prompt, record.prompt)
            for index, record in enumerate(self.records)
        ], dtype=np.float32)
        if len(scores) > 1:
            scores = scores + (np.arange(len(scores), dtype=np.float32) * np.float32(1e-7))
        rows: list[dict[str, Any]] = []
        for index in np.argsort(scores)[::-1][: max(1, int(top_k))]:
            record = self.records[int(index)]
            rows.append({
                "index": int(index),
                "score": float(scores[int(index)]),
                "phase_score": float(phase_scores[int(index)]),
                "lexical_score": float(_lexical_score(prompt, record.prompt)),
                "topic_coverage": float(_topic_coverage(prompt, record.prompt)),
                "prompt": record.prompt,
                "preview": record.response[:240],
            })
        return rows

    def evaluate_pairs(self, pairs: Iterable[tuple[str, str]], *, top_k: int = 1) -> dict[str, Any]:
        rows = list(pairs)
        hits = 0
        reciprocal = []
        for prompt, expected in rows:
            ranks = self.rank(prompt, top_k=max(int(top_k), len(self.records) or 1))
            rank = 0
            for idx, row in enumerate(ranks, start=1):
                if expected.strip()[:80] in row["preview"] or row["preview"][:80] in expected:
                    rank = idx
                    break
            if rank == 1:
                hits += 1
            reciprocal.append(1.0 / rank if rank else 0.0)
        return {
            "pairs": len(rows),
            "top1": hits / max(1, len(rows)),
            "mrr": float(np.mean(reciprocal)) if reciprocal else 0.0,
        }

    def save(self, model_dir: str | Path) -> Path:
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "phase-chat-model",
            "config": asdict(self.config),
            "records": [asdict(record) for record in self.records],
        }
        (path / self.manifest_name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        np.savez_compressed(path / "prompt_signatures.npz", real=self.signatures.real, imag=self.signatures.imag)
        self.language_model.save(path / "phase_lm")
        return path

    @classmethod
    def load(cls, model_dir: str | Path) -> "PhaseChatModel":
        path = Path(model_dir)
        payload = json.loads((path / cls.manifest_name).read_text(encoding="utf-8"))
        records = [PhaseChatRecord(**item) for item in payload.get("records", [])]
        model = cls(PhaseChatConfig(**payload["config"]), records=records, language_model=PhaseLanguageModel.load(path / "phase_lm"))
        signatures_path = path / "prompt_signatures.npz"
        if signatures_path.exists():
            sig = np.load(signatures_path)
            model.signatures = (sig["real"] + 1j * sig["imag"]).astype(np.complex64)
        return model

    def summary(self) -> dict[str, Any]:
        return {
            "type": "phase-chat-model",
            "config": asdict(self.config),
            "records": len(self.records),
            "fallback": self.language_model.summary(),
        }

    def _rebuild_signatures(self) -> None:
        if not self.records:
            self.signatures = np.zeros((0, int(self.config.signature_cells)), dtype=np.complex64)
            return
        self.signatures = np.vstack([self._signature(record.prompt) for record in self.records]).astype(np.complex64)

    def _signature(self, prompt: str) -> np.ndarray:
        cells = int(self.config.signature_cells)
        vector = np.zeros(cells, dtype=np.complex64)
        tokens = _chat_tokens(prompt)
        for index, token in enumerate(tokens):
            vector += _phasor(f"tok:{token}", cells, int(self.config.seed)) * _phasor(f"pos:{index % 31}", cells, int(self.config.seed))
            if len(token) > 3:
                for start in range(0, len(token) - 2):
                    vector += 0.35 * _phasor(f"tri:{token[start:start+3]}", cells, int(self.config.seed))
        for left, right in zip(tokens, tokens[1:]):
            vector += 0.5 * _phasor(f"pair:{left}:{right}", cells, int(self.config.seed))
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        return vector.astype(np.complex64)


_PHASOR_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


def _phasor(label: str, cells: int, seed: int) -> np.ndarray:
    key = (label, int(cells), int(seed))
    cached = _PHASOR_CACHE.get(key)
    if cached is not None:
        return cached
    import hashlib

    digest = hashlib.blake2b(f"{seed}:{label}".encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:4], "big"))
    angles = rng.uniform(0.0, 2.0 * math.pi, int(cells))
    value = np.exp(1j * angles).astype(np.complex64)
    _PHASOR_CACHE[key] = value
    return value


def _extract_assistant_response(text: str) -> str:
    value = str(text).strip()
    marker = "assistant\n"
    lowered = value.lower()
    index = lowered.rfind(marker)
    if index >= 0:
        value = value[index + len(marker) :]
    return value.strip()


def _lexical_score(query: str, candidate: str) -> float:
    q = _chat_tokens(query)
    c = _chat_tokens(candidate)
    if not q or not c:
        return 0.0
    q_set = set(q)
    c_set = set(c)
    overlap = q_set & c_set
    cosine = len(overlap) / math.sqrt(max(1, len(q_set) * len(c_set)))
    recall = len(overlap) / max(1, len(q_set))
    ordered = _ordered_overlap(q, c)
    return min(1.0, 0.45 * cosine + 0.35 * recall + 0.20 * ordered)


def _topic_coverage(query: str, candidate: str) -> float:
    q = _topic_tokens(query)
    if not q:
        return 1.0
    c = set(_topic_tokens(candidate))
    if not c:
        return 0.0
    return len(set(q) & c) / max(1, len(set(q)))


def _ordered_overlap(query: list[str], candidate: list[str]) -> float:
    if not query or not candidate:
        return 0.0
    c_positions: dict[str, list[int]] = {}
    for index, token in enumerate(candidate):
        c_positions.setdefault(token, []).append(index)
    last = -1
    hits = 0
    for token in query:
        positions = c_positions.get(token, [])
        next_positions = [pos for pos in positions if pos > last]
        if next_positions:
            last = next_positions[0]
            hits += 1
    return hits / max(1, len(query))


def _chat_tokens(text: str) -> list[str]:
    useful = []
    for token in tokenize(text):
        if token in {"a", "an", "the", "and", "or", "in", "of", "to", "that", "is", "it"}:
            continue
        useful.append(_stem(token))
    return useful


_TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "briefly",
    "can",
    "code",
    "concise",
    "explain",
    "function",
    "how",
    "in",
    "is",
    "it",
    "me",
    "please",
    "python",
    "step",
    "sure",
    "that",
    "the",
    "there",
    "this",
    "to",
    "using",
    "what",
    "why",
    "with",
    "without",
    "write",
    "you",
}


def _topic_tokens(text: str) -> list[str]:
    return [token for token in _chat_tokens(text) if token not in _TOPIC_STOPWORDS and len(token) > 1]


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
