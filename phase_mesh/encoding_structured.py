#!/usr/bin/env python3
"""Structured text-to-phase encoder for arithmetic tasks.

Raw token hashing was collapsing arithmetic prompts into a broad "math fog".
This encoder preserves the factors of the problem as typed wave packets:

- First operand -> left region
- Operator -> center region
- Second operand -> right region
- Optional digit/place packets -> local substructure for multi-digit values
- Optional unordered pair packet -> commutative hint without revealing result

By default this does **not** inject a result hint. The representation probe should
prove that operands/operators survive before any answer layer is trained.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


TWO_PI = 2.0 * math.pi

# Simple regex for extracting arithmetic expressions
ARITH_RE = re.compile(
    r"(\d+)\s*(plus|minus|times|divided\s+by|added\s+to|subtracted\s+from|multiplied\s+by)\s+(\d+)"
)
OPERATOR_MAP = {
    "plus": "add",
    "minus": "sub",
    "times": "mul",
    "divided by": "div",
    "added to": "add",
    "subtracted from": "sub",
    "multiplied by": "mul",
}
OPERATION_IDS = {
    "add": 0,
    "sub": 1,
    "mul": 2,
    "div": 3,
}


@dataclass(frozen=True)
class WavePacket:
    """A compact phase-modulated injection packet."""
    x: int
    y: int
    amplitude: float
    frequency: float
    phase: float
    radius: int
    label: str = ""


@dataclass(frozen=True)
class ArithmeticExpression:
    left: int
    operation: str
    right: int
    operator_text: str


def parse_arithmetic(text: str) -> ArithmeticExpression | None:
    """Parse a simple arithmetic expression without evaluating its result."""

    match = ARITH_RE.search(str(text).strip().lower())
    if not match:
        return None
    operator_text = " ".join(match.group(2).strip().split())
    return ArithmeticExpression(
        left=int(match.group(1)),
        operation=OPERATOR_MAP.get(operator_text, "add"),
        right=int(match.group(3)),
        operator_text=operator_text,
    )


def structured_arithmetic_feature_vector(text: str, feature_dim: int) -> np.ndarray:
    """Return a fixed-width typed feature vector for op/left/right factors.

    This deliberately avoids result features. It is the readout-side companion to
    the structured wave packets: if the 2D basin localizer picks one attractor,
    this side-band keeps the other typed factors available to downstream probes.
    """

    feature_dim = max(1, int(feature_dim))
    vector = np.zeros(feature_dim, dtype=np.float32)
    expression = parse_arithmetic(text)
    if expression is None:
        return vector

    op_width = max(4, feature_dim // 8)
    remaining = max(1, feature_dim - op_width)
    left_width = max(2, remaining // 3)
    right_width = max(2, remaining // 3)
    scalar_width = max(1, feature_dim - op_width - left_width - right_width)
    left_offset = op_width
    right_offset = left_offset + left_width
    scalar_offset = right_offset + right_width

    op_index = OPERATION_IDS.get(expression.operation, 0) % op_width
    vector[op_index] = 2.0

    def stamp_value(value: int, offset: int, width: int, prefix: str) -> None:
        if width <= 0:
            return
        abs_value = abs(int(value))
        sign = -1.0 if int(value) < 0 else 1.0
        vector[offset + (abs_value % width)] += 1.4 * sign
        vector[offset + ((abs_value * 7 + 3) % width)] += 0.7 * sign
        for place, char in enumerate(reversed(str(abs_value)[-4:])):
            digit = int(char)
            vector[offset + ((digit + place * 11) % width)] += 0.35 / np.sqrt(1.0 + place)
        hashed = int.from_bytes(
            hashlib.blake2b(f"{prefix}:{value}".encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        vector[offset + (hashed % width)] += 0.45

    stamp_value(expression.left, left_offset, left_width, "left")
    stamp_value(expression.right, right_offset, right_width, "right")

    scalars = np.asarray(
        [
            np.sin(expression.left * 0.13),
            np.cos(expression.left * 0.13),
            np.sin(expression.right * 0.17),
            np.cos(expression.right * 0.17),
            np.sin((expression.left - expression.right) * 0.07),
            np.cos((expression.left + expression.right) * 0.05),
        ],
        dtype=np.float32,
    )
    for index in range(scalar_width):
        vector[scalar_offset + index] += scalars[index % scalars.size] * 0.5

    norm = float(np.linalg.norm(vector, ord=2))
    if norm > 1e-9:
        vector = vector / norm
    return vector.astype(np.float32)


class StructuredPhaseEncoder:
    """Maps arithmetic expressions to structured phase packets.

    Spatial layout (128x128 grid):
    - Left region (x: 0-42): first operand, y determined by value
    - Center (x: 88-112, y: 54-78): operator type
    - Right region (x: 85-127): second operand, y determined by value
    - Top strip (y: 0-12): optional unordered pair packet

    This means:
    - "8 plus 9" and "9 plus 8" produce mirror-symmetric packets
    - "8 plus 9" and "15 plus 82" share operator but differ in operand region
    - "8 plus 9" and "8 times 9" differ only in operator region
    """

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        *,
        include_result_hint: bool = False,
        include_place_value: bool = True,
        include_commutative_pair: bool = True,
    ):
        self.width = width
        self.height = height
        self.include_result_hint = bool(include_result_hint)
        self.include_place_value = bool(include_place_value)
        self.include_commutative_pair = bool(include_commutative_pair)

    def encode(self, text: str, *, max_packets: int = 16) -> list[WavePacket]:
        text = text.strip().lower()
        packets = []

        # Try structured arithmetic encoding first
        expression = parse_arithmetic(text)
        if expression is not None:
            packets = self._encode_arithmetic(expression)
            return packets[:max_packets]

        # Fall back to global packet for non-arithmetic text
        packets.append(self._global_packet(text))
        return packets[:max_packets]

    def _encode_arithmetic(self, expression: ArithmeticExpression) -> list[WavePacket]:
        """Encode an arithmetic expression with structured spatial layout."""
        a = int(expression.left)
        b = int(expression.right)
        op = expression.operation
        op_text = expression.operator_text

        packets = []

        # Global packet: task identity only, not result.
        packets.append(self._global_packet(f"{a} {op_text} {b}"))

        # Left operand: position determined by value
        left_x = max(5, min(42, (a % 38) + 2))
        left_y = max(20, min(108, (a * 7 + 13) % 89 + 20))
        packets.append(WavePacket(
            x=left_x, y=left_y,
            amplitude=0.8,
            frequency=self._value_to_freq(a),
            phase=self._value_to_phase(a),
            radius=max(3, min(8, 3 + a % 6)),
            label=f"a:{a}",
        ))
        if self.include_place_value:
            packets.extend(self._place_value_packets(a, side="left"))

        # Operator: fixed center position, frequency by operator type
        op_freq = {"add": 1.5, "sub": 2.5, "mul": 3.5, "div": 4.5}
        op_phase = {"add": 0.0, "sub": math.pi / 2, "mul": math.pi, "div": 3 * math.pi / 2}
        packets.append(WavePacket(
            x=100, y=64,
            amplitude=0.9,
            frequency=op_freq.get(op, 1.5),
            phase=op_phase.get(op, 0.0),
            radius=6,
            label=f"op:{op}",
        ))

        # Right operand: position determined by value
        right_x = max(85, min(127, (b % 43) + 85))
        right_y = max(20, min(108, (b * 11 + 7) % 89 + 20))
        packets.append(WavePacket(
            x=right_x, y=right_y,
            amplitude=0.8,
            frequency=self._value_to_freq(b),
            phase=self._value_to_phase(b),
            radius=max(3, min(8, 3 + b % 6)),
            label=f"b:{b}",
        ))
        if self.include_place_value:
            packets.extend(self._place_value_packets(b, side="right"))

        if self.include_commutative_pair:
            low, high = sorted((a, b))
            pair_hash = self._stable_int(f"pair:{op}:{low}:{high}")
            packets.append(WavePacket(
                x=int(pair_hash % max(1, self.width)),
                y=max(2, min(max(2, self.height // 8), 2 + (pair_hash // 97) % max(1, self.height // 8))),
                amplitude=0.38,
                frequency=0.9 + ((low + high) % 37) * 0.035,
                phase=self._value_to_phase((low * 131 + high * 17) % 360),
                radius=max(2, min(5, min(self.width, self.height) // 24)),
                label=f"pair:{op}:{low}:{high}",
            ))

        if self.include_result_hint:
            # Off by default. Useful only for explicit leakage/upper-bound ablations.
            result_approx = self._approx_result(a, op, b)
            result_x = max(50, min(max(50, self.width - 50), (result_approx % 29) + 50))
            packets.append(WavePacket(
                x=result_x % self.width,
                y=max(1, min(self.height - 1, 6)),
                amplitude=0.4,
                frequency=self._value_to_freq(result_approx),
                phase=self._value_to_phase(result_approx),
                radius=3,
                label=f"r:{result_approx}",
            ))

        return packets

    def _place_value_packets(self, value: int, *, side: str) -> list[WavePacket]:
        """Emit typed digit/place packets without evaluating an answer."""

        sign = -1 if int(value) < 0 else 1
        digits = list(str(abs(int(value))))[-4:]
        x_base = self.width // 5 if side == "left" else (self.width * 4) // 5
        y_base = (self.height * 3) // 4 if side == "left" else self.height // 4
        packets: list[WavePacket] = []
        for place, char in enumerate(reversed(digits)):
            digit = int(char)
            signed_digit = digit * sign
            x_offset = (-place * 3) if side == "left" else (place * 3)
            packets.append(WavePacket(
                x=int((x_base + x_offset) % self.width),
                y=int((y_base + digit * 5 + place * 7) % self.height),
                amplitude=0.32 / math.sqrt(1.0 + place),
                frequency=0.8 + digit * 0.09 + place * 0.17,
                phase=self._value_to_phase(signed_digit + place * 41),
                radius=max(2, min(4, min(self.width, self.height) // 28)),
                label=f"{side[0]}{10 ** place}:{signed_digit}",
            ))
        return packets

    def _approx_result(self, a: int, op: str, b: int) -> int:
        """Approximate result for position hint (integer only)."""
        if op == "add":
            return a + b
        if op == "sub":
            return abs(a - b)
        if op == "mul":
            return a * b if a * b < 10000 else a * b % 9973
        if op == "div":
            return a // b if b != 0 else 0
        return a + b

    def _value_to_freq(self, v: int) -> float:
        """Map value to frequency (modulated to avoid collisions)."""
        return 1.0 + (v % 100) * 0.05

    def _value_to_phase(self, v: int) -> float:
        """Map value to phase angle."""
        return (v % 360) * TWO_PI / 360

    @staticmethod
    def _stable_int(text: str) -> int:
        return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")

    def _global_packet(self, text: str) -> WavePacket:
        digest = hashlib.blake2b(f"global:{text}".encode(), digest_size=16).digest()
        x = (self.width // 2 + digest[0] - 127) % self.width
        y = (self.height // 2 + digest[1] - 127) % self.height
        return WavePacket(
            x=int(x), y=int(y),
            amplitude=0.7,
            frequency=0.35 + (digest[2] / 255.0) * 1.1,
            phase=(digest[3] / 255.0) * TWO_PI,
            radius=max(5, min(self.width, self.height) // 18),
            label="<global>",
        )


def compare_structured_vs_random():
    """Compare structured encoding vs random hash encoding."""
    from phase_mesh.encoding import TextPhaseEncoder as RandomEncoder
    from phase_mesh.field import PhaseFieldMesh
    from phase_mesh.config import MeshConfig

    config = MeshConfig(width=128, height=128, max_steps=40, seed=42)
    struct_enc = StructuredPhaseEncoder(128, 128)
    rand_enc = RandomEncoder(128, 128)

    prompts = [
        ("8 plus 9", "9 plus 8"),      # commutative, should be similar
        ("8 plus 9", "15 plus 82"),     # same op, different operands
        ("8 plus 9", "8 times 9"),      # different op
        ("17 plus 23", "23 plus 17"),   # commutative
        ("7 times 3", "3 times 7"),     # commutative mul
    ]

    print("=== Structured Encoding ===")
    for p1, p2 in prompts:
        mesh = PhaseFieldMesh(config)
        for pkt in struct_enc.encode(p1):
            mesh.inject_packet(pkt)
        mesh.run_until_resonance(max_steps=20)
        theta1 = mesh.theta.copy()

        mesh = PhaseFieldMesh(config)
        for pkt in struct_enc.encode(p2):
            mesh.inject_packet(pkt)
        mesh.run_until_resonance(max_steps=20)
        theta2 = mesh.theta.copy()

        dist = float(np.linalg.norm(theta1 - theta2))
        coh1 = float(abs(np.mean(np.exp(1j * theta1))))
        coh2 = float(abs(np.mean(np.exp(1j * theta2))))
        print(f"  {p1} vs {p2}: dist={dist:.3f} coh=({coh1:.3f}, {coh2:.3f})")

    print("\n=== Random Hash Encoding ===")
    for p1, p2 in prompts:
        mesh = PhaseFieldMesh(config)
        mesh.inject_text(p1, rand_enc)
        mesh.run_until_resonance(max_steps=20)
        theta1 = mesh.theta.copy()

        mesh = PhaseFieldMesh(config)
        mesh.inject_text(p2, rand_enc)
        mesh.run_until_resonance(max_steps=20)
        theta2 = mesh.theta.copy()

        dist = float(np.linalg.norm(theta1 - theta2))
        coh1 = float(abs(np.mean(np.exp(1j * theta1))))
        coh2 = float(abs(np.mean(np.exp(1j * theta2))))
        print(f"  {p1} vs {p2}: dist={dist:.3f} coh=({coh1:.3f}, {coh2:.3f})")


if __name__ == "__main__":
    compare_structured_vs_random()
