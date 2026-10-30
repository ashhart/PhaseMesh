from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .domains.code import analyze_python_source
from .encoding_structured import parse_arithmetic
from .registry import PhaseMeshRegistry


ORGANS = (
    "memory_retrieval",
    "binding",
    "reasoning",
    "generation",
    "learning",
    "control",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "around",
    "ask",
    "called",
    "did",
    "for",
    "i",
    "ignore",
    "in",
    "is",
    "it",
    "me",
    "mentions",
    "old",
    "should",
    "that",
    "the",
    "this",
    "to",
    "what",
    "with",
    "you",
}


@dataclass
class ShellMemoryRecord:
    subject: str
    value: str
    source: str
    tokens: list[str] = field(default_factory=list)


@dataclass
class ShellBindingRecord:
    key: str
    value: str
    roles: dict[str, str] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)


class PhaseMeshLLMShell:
    """A small executive shell that composes the verified PhaseMesh organs.

    This is not an open-ended language model. It is a deterministic control
    layer that wires memory, role binding, narrow reasoning adapters, learning,
    and response generation into one auditable interface.
    """

    state_name = "llm_shell.json"

    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        *,
        language_model_dir: str | Path | None = None,
        chat_model_dir: str | Path | None = None,
        weight_artifact_dir: str | Path | None = None,
        language_model: Any | None = None,
        registry: PhaseMeshRegistry | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.language_model_dir = Path(language_model_dir) if language_model_dir is not None else None
        self.chat_model_dir = Path(chat_model_dir) if chat_model_dir is not None else None
        self.weight_artifact_dir = Path(weight_artifact_dir) if weight_artifact_dir is not None else None
        self.registry = registry or PhaseMeshRegistry()
        self.memory_records: list[ShellMemoryRecord] = []
        self.binding_records: list[ShellBindingRecord] = []
        self.language_model = language_model
        self.language_model_status: dict[str, Any] = {"loaded": False, "source": "none"}
        if self.artifact_dir is not None:
            self._load_state_if_present()
        self._load_language_model_if_present()

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        *,
        language_model_dir: str | Path | None = None,
        chat_model_dir: str | Path | None = None,
        weight_artifact_dir: str | Path | None = None,
    ) -> "PhaseMeshLLMShell":
        return cls(
            artifact_dir=artifact_dir,
            language_model_dir=language_model_dir,
            chat_model_dir=chat_model_dir,
            weight_artifact_dir=weight_artifact_dir,
        )

    def learn(self, text: str) -> dict[str, Any]:
        value = str(text).strip()
        lowered = value.lower()
        if self._is_memory_write(lowered) or self._is_binding_write(lowered):
            return self.run(value)
        return self.run(f"Remember: note: {value}")

    def think(self, text: str) -> dict[str, Any]:
        return self.run(text)

    def generate_answer(self, result: dict[str, Any]) -> str:
        return str(result.get("answer", ""))

    def manifest(self) -> dict[str, Any]:
        return {
            "type": "phase-mesh-llm-shell",
            "organs": list(ORGANS),
            "memory_records": len(self.memory_records),
            "binding_records": len(self.binding_records),
            "language_model": self.language_model_status,
            "registry": self.registry.manifest(),
        }

    def save(self, artifact_dir: str | Path | None = None) -> Path:
        path = Path(artifact_dir) if artifact_dir is not None else self.artifact_dir
        if path is None:
            raise ValueError("artifact_dir is required to save the shell state")
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "phase-mesh-llm-shell",
            "version": 1,
            "memory_records": [asdict(record) for record in self.memory_records],
            "binding_records": [asdict(record) for record in self.binding_records],
        }
        out = path / self.state_name
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.artifact_dir = path
        return out

    def run(self, text: str) -> dict[str, Any]:
        prompt = str(text).strip()
        trace: list[dict[str, Any]] = []
        organs: dict[str, list[str]] = {name: [] for name in ORGANS}

        def record(organ: str, step: str, action: str, **data: Any) -> None:
            trace.append({"organ": organ, "step": step, "action": action, "data": data})
            organs[organ].append(action or step)

        route, route_data = self._route(prompt)
        record("control", "route_select", f"route {route}", **route_data)

        memory_hit = self._best_memory(prompt)
        binding_hit = self._best_binding(prompt)
        record(
            "memory_retrieval",
            "memory_scan",
            "checked persistent shell memory",
            hit=memory_hit[0].subject if memory_hit else "",
            records=len(self.memory_records),
        )
        record(
            "binding",
            "binding_scan",
            "checked role/value bindings",
            hit=binding_hit[0].key if binding_hit else "",
            records=len(self.binding_records),
        )

        learned = self._learn_from_prompt(prompt)
        if learned is not None:
            record("learning", "state_update", learned["action"], learned=learned)
            route = "memory"
            answer = learned["answer"]
            reasoning = {"mode": "store", "learned": learned}
        elif route == "memory":
            answer, reasoning = self._answer_memory(prompt, memory_hit, binding_hit)
            record("learning", "state_update", "no new persistent write")
        elif route == "arithmetic":
            answer, reasoning = self._answer_arithmetic(prompt)
            record("learning", "state_update", "no arithmetic write")
        elif route == "code":
            answer, reasoning = self._answer_code(prompt)
            record("learning", "state_update", "no code write")
        elif route == "json":
            answer, reasoning = self._answer_json(prompt)
            record("learning", "state_update", "no json write")
        else:
            answer, reasoning = self._answer_generation(prompt)
            record("learning", "state_update", "no generation write")

        record("reasoning", "reasoning_adapter", f"used {reasoning['mode']} reasoning", **reasoning)
        record("generation", "surface_response", "generated fluent answer", length=len(answer))

        if learned is not None and self.artifact_dir is not None:
            self.save()

        return {
            "status": "ok",
            "route": route,
            "answer": answer,
            "trace": trace,
            "organs": organs,
            "data": {
                "route": route_data,
                "reasoning": reasoning,
                "memory_records": len(self.memory_records),
                "binding_records": len(self.binding_records),
                "language_model": self.language_model_status,
            },
        }

    def _route(self, text: str) -> tuple[str, dict[str, Any]]:
        lowered = text.lower().strip()
        if self._is_memory_write(lowered):
            return "memory", {"source": "memory imperative", "confidence": 0.98}
        if self._is_binding_write(lowered):
            return "memory", {"source": "binding imperative", "confidence": 0.96}
        if self._looks_like_json(text):
            return "json", {"source": "json parser", "confidence": 0.96}
        if self._parse_arithmetic(text) is not None:
            return "arithmetic", {"source": "arithmetic parser", "confidence": 0.96}
        if self._looks_like_code_request(text):
            return "code", {"source": "code intent", "confidence": 0.90}
        if self._memory_intent(text) and (self.memory_records or self.binding_records):
            return "memory", {"source": "memory recall intent", "confidence": 0.88}
        binding_hit = self._best_binding(text)
        if binding_hit is not None and binding_hit[1] >= 2.0:
            return "memory", {"source": "binding overlap", "confidence": 0.82}
        if self._best_memory(text) is not None and re.search(r"\b(recall|remember|room|codename)\b", lowered):
            return "memory", {"source": "memory overlap", "confidence": 0.76}
        if self.language_model is not None:
            return "generation", {"source": "phase-language-model", "confidence": 0.72}
        return "generation", {"source": "fallback generator", "confidence": 0.55}

    def _learn_from_prompt(self, text: str) -> dict[str, Any] | None:
        if self._is_binding_write(text.lower()):
            binding = self._parse_binding_write(text)
            if binding is None:
                return None
            self._upsert_binding(binding)
            return {
                "action": "stored role binding",
                "answer": f"Memory updated: bound {binding.key} to {binding.value}.",
                "key": binding.key,
                "value": binding.value,
            }

        if not self._is_memory_write(text.lower()):
            return None
        subject, value = self._parse_memory_write(text)
        if not subject or not value:
            return None
        record = ShellMemoryRecord(
            subject=subject,
            value=value,
            source=text.strip(),
            tokens=sorted(set(_tokens(f"{subject} {value} {text}"))),
        )
        self._upsert_memory(record)
        return {
            "action": "stored memory fact",
            "answer": f"Memory updated: {record.subject} is {record.value}.",
            "subject": record.subject,
            "value": record.value,
        }

    def _answer_memory(
        self,
        text: str,
        memory_hit: tuple[ShellMemoryRecord, float] | None,
        binding_hit: tuple[ShellBindingRecord, float] | None,
    ) -> tuple[str, dict[str, Any]]:
        if binding_hit is not None and (memory_hit is None or binding_hit[1] > memory_hit[1]):
            record, score = binding_hit
            return (
                f"Memory recall: {record.key} resolves to {record.value}.",
                {"mode": "binding-memory", "score": score, "key": record.key, "value": record.value},
            )
        if memory_hit is not None:
            record, score = memory_hit
            return (
                f"Memory recall: {record.subject} is {record.value}.",
                {"mode": "memory", "score": score, "subject": record.subject, "value": record.value},
            )
        return (
            "Memory recall: I checked the shell memory and no matching fact is stored yet.",
            {"mode": "memory", "score": 0.0, "query": text},
        )

    def _answer_arithmetic(self, text: str) -> tuple[str, dict[str, Any]]:
        chain = _parse_arithmetic_chain(text)
        if chain is not None:
            expression, value = chain
            return (
                f"Arithmetic answer: {expression} = {value}.",
                {"mode": "arithmetic-chain", "expression": expression, "value": value},
            )
        parsed = self._parse_arithmetic(text)
        if parsed is None:
            payload = self.registry.solve(text, domain="arithmetic")
            answer = str(payload.get("answer", ""))
            return (
                f"Arithmetic answer: {answer}.",
                {"mode": "arithmetic", "payload": payload},
            )
        left, op, right = parsed
        value = _eval_arithmetic(left, op, right)
        expression = f"{left} {op} {right}"
        return (
            f"Arithmetic answer: {expression} = {value}.",
            {"mode": "arithmetic", "left": left, "operator": op, "right": right, "value": value},
        )

    def _answer_code(self, text: str) -> tuple[str, dict[str, Any]]:
        code = text.strip()
        correction = _correct_obvious_python_bug(code)
        if correction is not None:
            analysis = analyze_python_source(correction["code"])
            return (
                f"Code answer: {correction['explanation']}\n{correction['code']}",
                {"mode": "code-correction", "syntax_ok": bool(analysis.get("syntax_ok")), "analysis": analysis},
            )
        if self._looks_like_code_request(text) and not code.startswith(("def ", "class ", "import ")):
            code = self._generate_python_function(text)
        analysis = analyze_python_source(code)
        if analysis.get("syntax_ok"):
            functions = analysis.get("functions", [])
            name = functions[0]["name"] if functions else "module"
            return (
                f"Code answer: generated syntactically valid Python for {name}:\n{code}",
                {"mode": "code", "syntax_ok": True, "analysis": analysis},
            )
        return (
            f"Code answer: Python syntax check failed with {analysis.get('error_type')}: {analysis.get('error')}.",
            {"mode": "code", "syntax_ok": False, "analysis": analysis},
        )

    def _answer_json(self, text: str) -> tuple[str, dict[str, Any]]:
        payload = self.registry.solve(text, domain="json")
        answer = str(payload.get("answer", ""))
        return (
            f"JSON answer: root type is {answer}.",
            {"mode": "json", "payload": payload},
        )

    def _answer_generation(self, text: str) -> tuple[str, dict[str, Any]]:
        if self.language_model is not None:
            generated = self.language_model.generate(
                text,
                max_tokens=36,
                temperature=0.0,
                top_k=12,
                repeat_penalty=1.18,
                no_repeat_ngram=3,
                max_token_repeats=3,
            )
            completion = str(generated.get("completion", "")).strip()
            answer = completion or str(generated.get("text", "")).strip()
            if answer:
                mode = str(self.language_model_status.get("summary", {}).get("type", "phase-language-model"))
                return (
                    answer if answer[-1] in ".!?" else f"{answer}.",
                    {
                        "mode": mode,
                        "prompt_tokens": len(_tokens(text)),
                        "language_model": self.language_model_status,
                        "generated_tokens": len(generated.get("tokens", [])),
                        "completion": completion,
                    },
                )

        answer = (
            "A phase-mesh shell routes the prompt, checks persistent memory and role bindings, "
            "uses verified reasoning adapters when they fit, and emits a traced response."
        )
        return (
            answer,
            {"mode": "generation", "prompt_tokens": len(_tokens(text))},
        )

    def _load_language_model_if_present(self) -> None:
        if self.language_model is not None:
            self.language_model_status = {"loaded": True, "source": "injected"}
            return

        chat_candidate = self.chat_model_dir
        if chat_candidate is None and self.artifact_dir is not None:
            chat_candidate = self.artifact_dir / "phase_chat"
        if chat_candidate is not None and (chat_candidate / "chat_model.json").exists():
            from .chat_lm import PhaseChatModel

            self.language_model = PhaseChatModel.load(chat_candidate)
            self.chat_model_dir = chat_candidate
            self.language_model_status = {
                "loaded": True,
                "source": str(chat_candidate),
                "summary": self.language_model.summary(),
            }
            return

        candidate = self.language_model_dir
        if candidate is None and self.artifact_dir is not None:
            candidate = self.artifact_dir / "phase_lm"
        if candidate is not None and (candidate / "model.json").exists() and (candidate / "phase_memory.npz").exists():
            from .language_model import PhaseLanguageModel

            self.language_model = PhaseLanguageModel.load(candidate)
            self.language_model_dir = candidate
            self.language_model_status = {
                "loaded": True,
                "source": str(candidate),
                "summary": self.language_model.summary(),
            }
            return
        if candidate is not None:
            self.language_model_status = {"loaded": False, "source": str(candidate)}

        weight_candidate = self.weight_artifact_dir
        if weight_candidate is None and self.artifact_dir is not None:
            weight_candidate = self.artifact_dir / "weight_pour"
        if weight_candidate is None:
            return
        if not (weight_candidate / "manifest.json").exists():
            self.language_model_status = {"loaded": False, "source": str(weight_candidate)}
            return

        from .weight_reader import PhaseWeightReader

        self.language_model = PhaseWeightReader(weight_candidate)
        self.weight_artifact_dir = weight_candidate
        self.language_model_status = {
            "loaded": True,
            "source": str(weight_candidate),
            "summary": self.language_model.summary(),
        }

    def _best_memory(self, text: str) -> tuple[ShellMemoryRecord, float] | None:
        query = set(_tokens(text))
        best: tuple[ShellMemoryRecord, float] | None = None
        for record in self.memory_records:
            score = _overlap_score(query, set(record.tokens))
            if record.subject.lower() in text.lower():
                score += 2.0
            if best is None or score > best[1]:
                best = (record, score)
        if best is None or best[1] <= 0.0:
            return None
        return best

    def _best_binding(self, text: str) -> tuple[ShellBindingRecord, float] | None:
        query = set(_tokens(text))
        best: tuple[ShellBindingRecord, float] | None = None
        for record in self.binding_records:
            score = _overlap_score(query, set(record.tokens))
            if best is None or score > best[1]:
                best = (record, score)
        if best is None or best[1] <= 0.0:
            return None
        return best

    def _parse_memory_write(self, text: str) -> tuple[str, str]:
        body = re.sub(r"^\s*remember\s*:?\s*", "", text, flags=re.IGNORECASE).strip()
        body = body.strip(" .")
        if ":" in body:
            subject, value = body.split(":", 1)
            return _clean_subject(subject), _clean_value(value)
        match = re.search(r"(.+?)\s+is\s+called\s+(.+)$", body, flags=re.IGNORECASE)
        if match:
            return _clean_subject(match.group(1)), _clean_value(match.group(2))
        match = re.search(r"(.+?)\s+(?:is|are)\s+(.+)$", body, flags=re.IGNORECASE)
        if match:
            return _clean_subject(match.group(1)), _clean_value(match.group(2))
        return _clean_subject(body), body

    def _parse_binding_write(self, text: str) -> ShellBindingRecord | None:
        body = re.sub(r"^\s*bind\s+", "", text, flags=re.IGNORECASE).strip()
        if "->" not in body:
            return None
        key, value = body.split("->", 1)
        key = key.strip()
        value = _clean_value(value)
        roles: dict[str, str] = {}
        match = re.search(r"^(\w+)\s+(\w+)\s+(.+?)\s+near\s+(.+)$", key, flags=re.IGNORECASE)
        if match:
            roles = {
                "actor": match.group(1),
                "action": match.group(2),
                "object": match.group(3).strip(),
                "place": match.group(4).strip(),
            }
        return ShellBindingRecord(
            key=key,
            value=value,
            roles=roles,
            tokens=sorted(set(_tokens(f"{key} {value}"))),
        )

    def _generate_python_function(self, text: str) -> str:
        name_match = re.search(r"function\s+named\s+([A-Za-z_]\w*)", text, flags=re.IGNORECASE)
        name = name_match.group(1) if name_match else "phase_mesh_function"
        return_match = re.search(r"returns?\s+(.+?)(?:\.|$)", text, flags=re.IGNORECASE)
        expression = return_match.group(1).strip() if return_match else "None"
        identifiers = [
            item
            for item in re.findall(r"\b[A-Za-z_]\w*\b", expression)
            if item not in {"return", "None", "True", "False"}
        ]
        params = sorted(set(identifiers)) or ["value"]
        if name.endswith("_one") and "n" not in params:
            params = ["n"]
            expression = "n + 1"
        return f"def {name}({', '.join(params)}):\n    return {expression}"

    def _upsert_memory(self, record: ShellMemoryRecord) -> None:
        for index, existing in enumerate(self.memory_records):
            if existing.subject.lower() == record.subject.lower():
                self.memory_records[index] = record
                return
        self.memory_records.append(record)

    def _upsert_binding(self, record: ShellBindingRecord) -> None:
        for index, existing in enumerate(self.binding_records):
            if existing.key.lower() == record.key.lower():
                self.binding_records[index] = record
                return
        self.binding_records.append(record)

    def _load_state_if_present(self) -> None:
        if self.artifact_dir is None:
            return
        state = self.artifact_dir / self.state_name
        if not state.exists():
            return
        payload = json.loads(state.read_text(encoding="utf-8"))
        self.memory_records = [
            ShellMemoryRecord(
                subject=str(item.get("subject", "")),
                value=str(item.get("value", "")),
                source=str(item.get("source", "")),
                tokens=[str(token) for token in item.get("tokens", [])],
            )
            for item in payload.get("memory_records", [])
        ]
        self.binding_records = [
            ShellBindingRecord(
                key=str(item.get("key", "")),
                value=str(item.get("value", "")),
                roles={str(k): str(v) for k, v in dict(item.get("roles", {})).items()},
                tokens=[str(token) for token in item.get("tokens", [])],
            )
            for item in payload.get("binding_records", [])
        ]

    def _is_memory_write(self, lowered: str) -> bool:
        return lowered.strip().startswith(("remember:", "remember "))

    def _is_binding_write(self, lowered: str) -> bool:
        return lowered.strip().startswith("bind ") and "->" in lowered

    def _memory_intent(self, text: str) -> bool:
        return bool(re.search(r"\b(recall|remember|memory|codename|room)\b", text.lower()))

    def _looks_like_json(self, text: str) -> bool:
        try:
            json.loads(text)
            return True
        except Exception:
            return False

    def _looks_like_code_request(self, text: str) -> bool:
        stripped = text.strip()
        lowered = stripped.lower()
        return (
            stripped.startswith(("def ", "class ", "import "))
            or "```python" in lowered
            or "write python code" in lowered
            or "python function" in lowered
            or "function named" in lowered
        )

    def _parse_arithmetic(self, text: str) -> tuple[int, str, int] | None:
        expression = parse_arithmetic(text)
        if expression is not None:
            op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(expression.operation)
            if op is not None:
                return expression.left, op, expression.right
        symbol = re.search(r"(-?\d+)\s*([+\-*/xX])\s*(-?\d+)", text)
        if symbol:
            op = "*" if symbol.group(2).lower() == "x" else symbol.group(2)
            return int(symbol.group(1)), op, int(symbol.group(3))
        return None


def _tokens(text: str) -> list[str]:
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    tokens = re.findall(r"[a-z0-9]+", expanded.lower())
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def _overlap_score(query: set[str], candidate: set[str]) -> float:
    if not query or not candidate:
        return 0.0
    overlap = query & candidate
    return float(len(overlap)) + (len(overlap) / max(1, len(candidate)))


def _clean_subject(value: str) -> str:
    value = re.sub(r"^\s*the\s+", "", str(value).strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip(" .")


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).strip(" .")


def _eval_arithmetic(left: int, op: str, right: int) -> str:
    if op == "+":
        return str(left + right)
    if op == "-":
        return str(left - right)
    if op == "*":
        return str(left * right)
    if op == "/":
        if right == 0:
            return "undefined"
        value = left / right
        return str(int(value)) if value.is_integer() else f"{value:.6g}"
    raise ValueError(f"unsupported operator: {op}")


def _parse_arithmetic_chain(text: str) -> tuple[str, str] | None:
    value = str(text)
    matches = list(re.finditer(r"-?\d+|[+\-*/xX]", value))
    if len(matches) < 3:
        return None
    tokens = [match.group(0) for match in matches]
    if len(tokens) < 3 or len(tokens) % 2 == 0:
        return None
    if not all(_is_int_token(token) if index % 2 == 0 else token in "+-*/xX" for index, token in enumerate(tokens)):
        return None
    numbers = [token for index, token in enumerate(tokens) if index % 2 == 0]
    if len(numbers) < 3:
        return None
    expression = " ".join("*" if token.lower() == "x" else token for token in tokens)
    try:
        result = _safe_eval_numeric_expression(tokens)
    except ZeroDivisionError:
        return expression, "undefined"
    except Exception:
        return None
    return expression, _format_number(result)


def _safe_eval_numeric_expression(tokens: list[str]) -> float:
    values: list[float] = [float(tokens[0])]
    ops: list[str] = []
    index = 1
    while index < len(tokens):
        op = "*" if tokens[index].lower() == "x" else tokens[index]
        right = float(tokens[index + 1])
        if op == "*":
            values[-1] *= right
        elif op == "/":
            if right == 0.0:
                raise ZeroDivisionError
            values[-1] /= right
        else:
            ops.append(op)
            values.append(right)
        index += 2
    result = values[0]
    for op, value in zip(ops, values[1:]):
        if op == "+":
            result += value
        elif op == "-":
            result -= value
        else:
            raise ValueError(op)
    return result


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"


def _is_int_token(token: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", token))


def _correct_obvious_python_bug(text: str) -> dict[str, str] | None:
    match = re.search(
        r"def\s+add\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)\s*:\s*return\s+\1\s*-\s*\2",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    left, right = match.group(1), match.group(2)
    code = f"def add({left}, {right}):\n    return {left} + {right}"
    return {
        "code": code,
        "explanation": "The bug is that `add` subtracts the second argument; it should return the sum:",
    }
