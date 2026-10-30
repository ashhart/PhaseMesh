from __future__ import annotations

import json
import math
import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)
PUNCT_NO_SPACE_BEFORE = {".", ",", ":", ";", "?", "!", ")", "]", "}", "%"}
PUNCT_NO_SPACE_AFTER = {"(", "[", "{", "$", "#"}


@dataclass
class PhaseLMConfig:
    order: int = 4
    phase_cells: int = 2048
    vocab_size: int = 4096
    seed: int = 7
    phase_weight: float = 1.0
    ngram_weight: float = 0.85
    unigram_weight: float = 0.15


class PhaseLanguageModel:
    """A compact phase-associative next-token language model.

    It is intentionally simple: ordered context tokens are bound into a complex
    phase key, next tokens are stored as value phasors, and generation scores
    candidates by unbinding the active context from the learned phase memory.
    N-gram counts are kept as an explicit fluency backoff, not as a transformer.
    """

    bos = "<bos>"
    eos = "<eos>"
    unk = "<unk>"

    def __init__(self, config: PhaseLMConfig | None = None) -> None:
        self.config = config or PhaseLMConfig()
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        for token in (self.unk, self.bos, self.eos):
            self._add_token(token)
        self.phase_memory = np.zeros(int(self.config.phase_cells), dtype=np.complex64)
        self.context_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.unigram_counts: Counter[int] = Counter()
        self.training_tokens = 0
        self.training_windows = 0

    def train_text(self, text: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        tokens = tokenize(text)
        if max_tokens is not None:
            tokens = tokens[: int(max_tokens)]
        ids = [self._add_token(token) for token in tokens]
        return self.train_token_ids(ids)

    def train_lines(self, lines: Iterable[str], *, max_tokens: int | None = None) -> dict[str, Any]:
        used = 0
        windows = 0
        for line in lines:
            tokens = tokenize(line)
            if not tokens:
                continue
            remaining = None if max_tokens is None else max(0, int(max_tokens) - used)
            if remaining == 0:
                break
            if remaining is not None:
                tokens = tokens[:remaining]
            ids = [self._add_token(token) for token in tokens]
            summary = self.train_token_ids(ids)
            used += int(summary["tokens"])
            windows += int(summary["windows"])
            if max_tokens is not None and used >= int(max_tokens):
                break
        return self.summary(extra={"tokens_added": used, "windows_added": windows})

    def train_token_ids(self, token_ids: list[int]) -> dict[str, Any]:
        order = int(self.config.order)
        sequence = [self.token_to_id[self.bos]] * order + list(token_ids) + [self.token_to_id[self.eos]]
        windows = 0
        for index in range(order, len(sequence)):
            context = tuple(sequence[index - order : index])
            target = int(sequence[index])
            for width in range(0, order + 1):
                suffix = context[-width:] if width else ()
                self.context_counts[suffix][target] += 1
            self.unigram_counts[target] += 1
            self._bind_context_to_token(context, target)
            windows += 1
        self.training_tokens += len(token_ids)
        self.training_windows += windows
        return {"tokens": len(token_ids), "windows": windows, "vocab": len(self.id_to_token)}

    def train_next_distribution(
        self,
        context: str | Iterable[str] | Iterable[int],
        candidates: Iterable[tuple[str, float]],
        *,
        weight_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Inject a soft teacher next-token distribution into phase memory.

        This is the distillation path used by `lm-pour`: instead of training only
        on sampled teacher text, PhaseMesh receives the teacher's top-k next-token
        probabilities as weighted context->token bindings.
        """

        if isinstance(context, str):
            context_ids = [self._add_token(token) for token in tokenize(context)]
        else:
            values = list(context)
            if values and isinstance(values[0], int):
                context_ids = [int(item) for item in values]
            else:
                context_ids = [self._add_token(str(item)) for item in values]
        phase_context = self._context_from_ids(context_ids)
        added = 0
        skipped = 0
        total_weight = 0.0
        for candidate_text, raw_weight in candidates:
            weight = max(0.0, float(raw_weight)) * max(0.0, float(weight_scale))
            if weight <= 0.0:
                skipped += 1
                continue
            tokens = tokenize(candidate_text)
            if len(tokens) != 1:
                skipped += 1
                continue
            target = self._add_token(tokens[0])
            for width in range(0, int(self.config.order) + 1):
                suffix = phase_context[-width:] if width else ()
                self.context_counts[suffix][target] += weight
            self.unigram_counts[target] += weight
            self._bind_context_to_token(phase_context, target, weight=weight)
            added += 1
            total_weight += weight
        if added:
            self.training_windows += 1
            self.training_tokens += len(context_ids)
        return {
            "contexts": 1 if added else 0,
            "candidates_added": added,
            "candidates_skipped": skipped,
            "weight": float(total_weight),
            "vocab": len(self.id_to_token),
        }

    def next_scores(self, context_tokens: Iterable[str] | Iterable[int], *, top_k: int | None = None) -> list[tuple[str, float]]:
        context = self._context_from_tokens(context_tokens)
        candidate_ids = range(3, len(self.id_to_token))
        scored = [(token_id, self._score_token(context, token_id)) for token_id in candidate_ids]
        scored.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            scored = scored[: max(1, int(top_k))]
        return [(self.id_to_token[token_id], float(score)) for token_id, score in scored]

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 40,
        temperature: float = 0.7,
        top_k: int = 24,
        repeat_penalty: float = 1.08,
        no_repeat_ngram: int = 3,
        max_token_repeats: int = 4,
        seed: int | None = None,
    ) -> dict[str, Any]:
        prompt_tokens = tokenize(prompt)
        generated: list[str] = []
        rng = np.random.default_rng(self.config.seed if seed is None else int(seed))
        context_ids = [self._id_for_token(token) for token in prompt_tokens]
        for _ in range(max(0, int(max_tokens))):
            context = self._context_from_ids(context_ids)
            scored = []
            recent = Counter(context_ids[-16:])
            for token_id in range(3, len(self.id_to_token)):
                score = self._score_token(context, token_id)
                if max_token_repeats > 0 and recent[token_id] >= int(max_token_repeats):
                    score = -1.0e12
                if token_id in recent and repeat_penalty > 1.0:
                    score -= math.log(float(repeat_penalty)) * (1.0 + recent[token_id])
                if no_repeat_ngram > 1 and _would_repeat_ngram(context_ids, token_id, int(no_repeat_ngram)):
                    score = -1.0e12
                if context_ids and _is_punctuation(self.id_to_token[context_ids[-1]]) and _is_punctuation(self.id_to_token[token_id]):
                    score -= 6.0
                scored.append((token_id, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            shortlist = scored[: max(1, int(top_k))]
            token_id = self._sample(shortlist, temperature=float(temperature), rng=rng)
            if token_id == self.token_to_id[self.eos]:
                break
            token = self.id_to_token[token_id]
            generated.append(token)
            context_ids.append(token_id)
        return {
            "prompt": prompt,
            "tokens": generated,
            "text": detokenize(prompt_tokens + generated),
            "completion": detokenize(generated),
            "model": self.summary(),
        }

    def evaluate_text(self, text: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        tokens = tokenize(text)
        if max_tokens is not None:
            tokens = tokens[: int(max_tokens)]
        ids = [self._id_for_token(token) for token in tokens]
        order = int(self.config.order)
        sequence = [self.token_to_id[self.bos]] * order + ids + [self.token_to_id[self.eos]]
        ranks: list[int] = []
        losses: list[float] = []
        for index in range(order, len(sequence)):
            context = tuple(sequence[index - order : index])
            target = int(sequence[index])
            scores = np.asarray([self._score_token(context, token_id) for token_id in range(len(self.id_to_token))], dtype=np.float64)
            scores[:3] = -1e9
            shifted = scores - float(np.max(scores))
            probs = np.exp(shifted)
            probs /= float(np.sum(probs)) if float(np.sum(probs)) > 0 else 1.0
            target_prob = float(probs[target]) if target < probs.size else 1e-12
            losses.append(-math.log(max(target_prob, 1e-12)))
            rank = 1 + int(np.sum(scores > scores[target]))
            ranks.append(rank)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        return {
            "tokens": len(tokens),
            "windows": len(ranks),
            "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
            "top1": float(np.mean([rank == 1 for rank in ranks])) if ranks else 0.0,
            "top5": float(np.mean([rank <= 5 for rank in ranks])) if ranks else 0.0,
            "cross_entropy": mean_loss,
            "perplexity": float(math.exp(min(20.0, mean_loss))) if losses else 0.0,
        }

    def summary(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "type": "phase-language-model",
            "config": asdict(self.config),
            "vocab": len(self.id_to_token),
            "training_tokens": int(self.training_tokens),
            "training_windows": int(self.training_windows),
            "phase_memory_norm": float(np.linalg.norm(self.phase_memory)),
            "contexts": len(self.context_counts),
        }
        if extra:
            payload.update(extra)
        return payload

    def save(self, model_dir: str | Path) -> Path:
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "type": "phase-language-model",
            "config": asdict(self.config),
            "id_to_token": self.id_to_token,
            "training_tokens": self.training_tokens,
            "training_windows": self.training_windows,
            "context_counts": {
                " ".join(str(item) for item in key): dict(counter)
                for key, counter in self.context_counts.items()
            },
            "unigram_counts": dict(self.unigram_counts),
        }
        (path / "model.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        np.savez_compressed(path / "phase_memory.npz", real=self.phase_memory.real, imag=self.phase_memory.imag)
        return path

    @classmethod
    def load(cls, model_dir: str | Path) -> "PhaseLanguageModel":
        path = Path(model_dir)
        metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        model = cls(PhaseLMConfig(**metadata["config"]))
        model.id_to_token = [str(token) for token in metadata["id_to_token"]]
        model.token_to_id = {token: index for index, token in enumerate(model.id_to_token)}
        phase = np.load(path / "phase_memory.npz")
        model.phase_memory = (phase["real"] + 1j * phase["imag"]).astype(np.complex64)
        model.training_tokens = int(metadata.get("training_tokens", 0))
        model.training_windows = int(metadata.get("training_windows", 0))
        model.context_counts = defaultdict(Counter)
        for key, values in metadata.get("context_counts", {}).items():
            context = tuple(int(item) for item in key.split()) if key else ()
            model.context_counts[context] = Counter({int(k): float(v) for k, v in values.items()})
        model.unigram_counts = Counter({int(k): float(v) for k, v in metadata.get("unigram_counts", {}).items()})
        return model

    def _add_token(self, token: str) -> int:
        token = normalize_token(token)
        if token in self.token_to_id:
            return self.token_to_id[token]
        if len(self.id_to_token) >= int(self.config.vocab_size):
            return self.token_to_id[self.unk]
        token_id = len(self.id_to_token)
        self.token_to_id[token] = token_id
        self.id_to_token.append(token)
        return token_id

    def _id_for_token(self, token: str) -> int:
        return self.token_to_id.get(normalize_token(token), self.token_to_id[self.unk])

    def _context_from_tokens(self, tokens: Iterable[str] | Iterable[int]) -> tuple[int, ...]:
        values = list(tokens)
        if values and isinstance(values[0], int):
            return self._context_from_ids([int(item) for item in values])
        return self._context_from_ids([self._id_for_token(str(item)) for item in values])

    def _context_from_ids(self, token_ids: list[int]) -> tuple[int, ...]:
        order = int(self.config.order)
        ids = [self.token_to_id[self.bos]] * order + list(token_ids)
        return tuple(ids[-order:])

    def _bind_context_to_token(self, context: tuple[int, ...], token_id: int, *, weight: float = 1.0) -> None:
        key = self._context_phasor(context)
        value = self._token_phasor(token_id)
        self.phase_memory += (float(weight) * np.conj(key) * value).astype(np.complex64)

    def _score_token(self, context: tuple[int, ...], token_id: int) -> float:
        key = self._context_phasor(context)
        predicted = key * self.phase_memory
        value = self._token_phasor(token_id)
        phase_score = float(np.real(np.vdot(value, predicted)) / max(1, int(self.config.phase_cells)))
        ngram_score = self._ngram_log_score(context, token_id)
        unigram_total = sum(self.unigram_counts.values()) or 1
        unigram_score = math.log((self.unigram_counts.get(token_id, 0) + 1.0) / (unigram_total + len(self.id_to_token)))
        return (
            float(self.config.phase_weight) * phase_score
            + float(self.config.ngram_weight) * ngram_score
            + float(self.config.unigram_weight) * unigram_score
        )

    def _ngram_log_score(self, context: tuple[int, ...], token_id: int) -> float:
        vocab = max(1, len(self.id_to_token) - 3)
        weighted = 0.0
        total_weight = 0.0
        max_width = min(int(self.config.order), len(context))
        for width in range(max_width, -1, -1):
            suffix = context[-width:] if width else ()
            counter = self.context_counts.get(suffix)
            if not counter:
                continue
            count = float(counter.get(token_id, 0))
            total = float(sum(counter.values()))
            alpha = 0.10
            prob = (count + alpha) / max(alpha * vocab, total + alpha * vocab)
            weight = float((width + 1) ** 2)
            weighted += weight * math.log(max(prob, 1e-12))
            total_weight += weight
        if total_weight <= 0.0:
            return -math.log(max(1, vocab))
        return weighted / total_weight

    def _context_phasor(self, context: tuple[int, ...]) -> np.ndarray:
        cells = int(self.config.phase_cells)
        acc = np.zeros(cells, dtype=np.complex64)
        for position, token_id in enumerate(context):
            acc += self._token_phasor(token_id) * self._position_phasor(position)
        magnitude = np.abs(acc)
        magnitude[magnitude == 0.0] = 1.0
        return (acc / magnitude).astype(np.complex64)

    def _token_phasor(self, token_id: int) -> np.ndarray:
        return _dense_phasor(f"tok:{int(token_id)}", int(self.config.phase_cells), int(self.config.seed))

    def _position_phasor(self, position: int) -> np.ndarray:
        return _dense_phasor(f"pos:{int(position)}", int(self.config.phase_cells), int(self.config.seed))

    def _sample(self, scored: list[tuple[int, float]], *, temperature: float, rng: np.random.Generator) -> int:
        if temperature <= 0.0:
            return int(scored[0][0])
        ids = np.asarray([item[0] for item in scored], dtype=np.int64)
        scores = np.asarray([item[1] for item in scored], dtype=np.float64)
        scores = scores / max(1e-6, temperature)
        scores -= float(np.max(scores))
        probs = np.exp(scores)
        probs_sum = float(np.sum(probs))
        if probs_sum <= 0.0 or not np.isfinite(probs_sum):
            return int(ids[0])
        probs /= probs_sum
        return int(rng.choice(ids, p=probs))


_PHASOR_CACHE: dict[tuple[str, int, int], np.ndarray] = {}


def _dense_phasor(label: str, cells: int, seed: int) -> np.ndarray:
    key = (label, int(cells), int(seed))
    cached = _PHASOR_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.blake2b(f"{int(seed)}:{label}".encode("utf-8"), digest_size=8).digest()
    label_seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(label_seed)
    angles = rng.uniform(0.0, 2.0 * math.pi, int(cells))
    phasor = np.exp(1j * angles).astype(np.complex64)
    _PHASOR_CACHE[key] = phasor
    return phasor


def tokenize(text: str) -> list[str]:
    return [normalize_token(token) for token in TOKEN_RE.findall(str(text)) if token.strip()]


def normalize_token(token: str) -> str:
    return str(token).strip().lower()


def detokenize(tokens: Iterable[str]) -> str:
    out = ""
    previous = ""
    for token in tokens:
        if not token or token in {PhaseLanguageModel.bos, PhaseLanguageModel.eos, PhaseLanguageModel.unk}:
            continue
        if not out:
            out = token
        elif token in PUNCT_NO_SPACE_BEFORE or previous in PUNCT_NO_SPACE_AFTER:
            out += token
        else:
            out += " " + token
        previous = token
    return out


def _is_punctuation(token: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]", str(token)))


def _would_repeat_ngram(context_ids: list[int], token_id: int, ngram: int) -> bool:
    if ngram <= 1 or len(context_ids) < ngram - 1:
        return False
    candidate = tuple(context_ids[-(ngram - 1) :] + [int(token_id)])
    history = context_ids + [int(token_id)]
    for index in range(0, len(history) - ngram):
        if tuple(history[index : index + ngram]) == candidate:
            return True
    return False
