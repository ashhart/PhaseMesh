from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any


ARITHMETIC_RE = re.compile(
    r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:\s*(?:\*\*|[+\-*/%])\s*[-+]?\d+(?:\.\d+)?)+(?![\w.])"
)
EQUATION_RE = re.compile(r"(.+?)=\s*([-+]?\d+(?:\.\d+)?)\b")


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    checker: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checker": self.checker,
            "message": self.message,
            "payload": self.payload,
        }


class VerifierRouter:
    """Routes decoded output through cheap local verifiers."""

    def verify(
        self,
        prompt: str,
        *,
        candidate: str | None = None,
        expected: str | None = None,
        coherence: float | None = None,
    ) -> VerifierResult:
        arithmetic = self._verify_arithmetic(prompt, expected=expected)
        if arithmetic is not None:
            return arithmetic

        for text in self._candidate_texts(prompt, candidate):
            json_result = self._verify_json(text)
            if json_result is not None:
                return json_result

        for text in self._candidate_texts(prompt, candidate):
            code_result = self._verify_python(text)
            if code_result is not None:
                return code_result

        if coherence is not None:
            passed = coherence >= 0.96
            return VerifierResult(
                passed=passed,
                checker="coherence",
                message="coherence threshold passed" if passed else "coherence threshold failed",
                payload={"coherence": coherence, "threshold": 0.96},
            )

        return VerifierResult(
            passed=True,
            checker="none",
            message="no verifier matched; accepted as exploratory resonance",
        )

    @staticmethod
    def _candidate_texts(prompt: str, candidate: str | None) -> tuple[str, ...]:
        if candidate is None or candidate == prompt:
            return (prompt,)
        return (candidate, prompt)

    def _verify_arithmetic(self, prompt: str, *, expected: str | None) -> VerifierResult | None:
        equation = EQUATION_RE.search(prompt)
        if equation is not None:
            lhs_text = equation.group(1)
            lhs_match = ARITHMETIC_RE.search(lhs_text)
            if lhs_match is not None:
                lhs_value = safe_eval_arithmetic(lhs_match.group(0))
                rhs_value = float(equation.group(2))
                passed = math.isclose(lhs_value, rhs_value, rel_tol=1e-9, abs_tol=1e-9)
                return VerifierResult(
                    passed=passed,
                    checker="arithmetic-equation",
                    message=f"{lhs_match.group(0)} evaluated to {lhs_value:g}",
                    payload={"actual": lhs_value, "claimed": rhs_value},
                )

        expression_match = ARITHMETIC_RE.search(prompt)
        if expression_match is None:
            return None

        value = safe_eval_arithmetic(expression_match.group(0))
        if expected is None:
            return VerifierResult(
                passed=True,
                checker="arithmetic-eval",
                message=f"{expression_match.group(0)} evaluated to {value:g}",
                payload={"actual": value},
            )

        expected_value = safe_eval_arithmetic(expected)
        passed = math.isclose(value, expected_value, rel_tol=1e-9, abs_tol=1e-9)
        return VerifierResult(
            passed=passed,
            checker="arithmetic-expected",
            message=f"{expression_match.group(0)} evaluated to {value:g}",
            payload={"actual": value, "expected": expected_value},
        )

    @staticmethod
    def _verify_json(text: str) -> VerifierResult | None:
        stripped = text.strip()
        if not stripped.startswith(("{", "[")):
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return VerifierResult(
                passed=False,
                checker="json",
                message=str(exc),
                payload={"line": exc.lineno, "column": exc.colno},
            )
        return VerifierResult(
            passed=True,
            checker="json",
            message="valid JSON",
            payload={"type": type(parsed).__name__},
        )

    @staticmethod
    def _verify_python(text: str) -> VerifierResult | None:
        stripped = extract_code_candidate(text)
        if stripped is None:
            return None
        try:
            compile(stripped, "<phase-mesh-candidate>", "exec")
        except SyntaxError as exc:
            return VerifierResult(
                passed=False,
                checker="python-compile",
                message=exc.msg,
                payload={"line": exc.lineno, "offset": exc.offset},
            )
        return VerifierResult(
            passed=True,
            checker="python-compile",
            message="python compiled",
        )


def extract_code_candidate(text: str) -> str | None:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence is not None:
        return fence.group(1).strip()
    stripped = text.strip()
    code_markers = ("def ", "class ", "import ", "from ", "print(", "for ", "while ", "if ")
    if any(stripped.startswith(marker) for marker in code_markers):
        return stripped
    return None


def safe_eval_arithmetic(expression: str) -> float:
    node = ast.parse(expression, mode="eval")
    return float(_eval_node(node.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(right) > 12:
                raise ValueError("Exponent too large for safe arithmetic verifier.")
            return left**right
    raise ValueError(f"Unsupported arithmetic expression: {ast.dump(node)}")
