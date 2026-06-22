from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", re.UNICODE)
TWO_PI = 2.0 * math.pi


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


class TextPhaseEncoder:
    """Deterministically maps text into local wave packets.

    This is deliberately not a language model. It is a cheap bridge from symbols to
    boundary conditions that lets the field accumulate repeatable topological
    structure around recurring text patterns.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        max_packets: int = 64,
        base_amplitude: float = 0.65,
    ) -> None:
        self.width = width
        self.height = height
        self.max_packets = max_packets
        self.base_amplitude = base_amplitude

    def tokenize(self, text: str) -> list[str]:
        tokens = TOKEN_RE.findall(text.lower())
        return tokens if tokens else ["<empty>"]

    def encode(self, text: str, *, max_packets: int | None = None) -> list[WavePacket]:
        tokens = self.tokenize(text)
        packet_limit = max_packets or self.max_packets
        packets: list[WavePacket] = [self._global_packet(text)]

        for index, token in enumerate(tokens[: max(0, packet_limit - 1)]):
            digest = self._digest(f"{index}:{token}")
            x = (digest[0] + 31 * index + digest[8]) % self.width
            y = (digest[1] + 17 * index + digest[9]) % self.height
            frequency = 0.65 + (digest[2] / 255.0) * 3.35
            phase = ((digest[3] / 255.0) * TWO_PI + index * 0.377) % TWO_PI
            radius = 3 + digest[4] % 5
            token_weight = 1.0 if any(ch.isalnum() for ch in token) else 0.55
            amplitude = self.base_amplitude * token_weight / math.sqrt(1.0 + index * 0.045)
            packets.append(
                WavePacket(
                    x=int(x),
                    y=int(y),
                    amplitude=float(amplitude),
                    frequency=float(frequency),
                    phase=float(phase),
                    radius=int(radius),
                    label=token,
                )
            )

        return packets[:packet_limit]

    def encode_feedback(self, message: str, *, success: bool) -> list[WavePacket]:
        prefix = "success" if success else "counter"
        packets = self.encode(f"{prefix}:{message}", max_packets=24)
        if success:
            return packets
        return [
            WavePacket(
                x=packet.x,
                y=packet.y,
                amplitude=-abs(packet.amplitude),
                frequency=packet.frequency,
                phase=(packet.phase + math.pi) % TWO_PI,
                radius=packet.radius,
                label=packet.label,
            )
            for packet in packets
        ]

    def _global_packet(self, text: str) -> WavePacket:
        digest = self._digest(f"global:{text}")
        x = (self.width // 2 + digest[0] - 127) % self.width
        y = (self.height // 2 + digest[1] - 127) % self.height
        frequency = 0.35 + (digest[2] / 255.0) * 1.1
        phase = (digest[3] / 255.0) * TWO_PI
        radius = max(5, min(self.width, self.height) // 18)
        return WavePacket(
            x=int(x),
            y=int(y),
            amplitude=self.base_amplitude * 0.85,
            frequency=float(frequency),
            phase=float(phase),
            radius=int(radius),
            label="<global>",
        )

    @staticmethod
    def _digest(text: str) -> bytes:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


@dataclass(frozen=True)
class DecodedResonance:
    route: str
    signature: str
    dominant_sector: int
    confidence: float
    sector_histogram: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "signature": self.signature,
            "dominant_sector": self.dominant_sector,
            "confidence": self.confidence,
            "sector_histogram": self.sector_histogram,
        }


class PhaseDecoder:
    """Reads a settled phase landscape into a small symbolic decision."""

    ROUTES = (
        "recall",
        "calculate",
        "code",
        "verify",
        "route",
        "plan",
        "compress",
        "explore",
        "search",
        "write",
        "observe",
        "act",
    )

    def __init__(self, sectors: int = 12) -> None:
        if sectors < 2:
            raise ValueError("sectors must be at least 2.")
        self.sectors = sectors

    def decode(self, theta: np.ndarray, *, coherence: float) -> DecodedResonance:
        phase = (theta + math.pi) % TWO_PI
        sector_width = TWO_PI / self.sectors
        sector_ids = np.floor(phase / sector_width).astype(np.int64)
        histogram = np.bincount(sector_ids.ravel(), minlength=self.sectors)
        dominant = int(np.argmax(histogram))
        base_confidence = float(histogram[dominant] / theta.size)
        confidence = max(0.0, min(1.0, base_confidence * (0.45 + 0.55 * coherence)))
        route = self.ROUTES[dominant % len(self.ROUTES)]
        signature = self._signature(theta)
        return DecodedResonance(
            route=route,
            signature=signature,
            dominant_sector=dominant,
            confidence=confidence,
            sector_histogram=[int(value) for value in histogram.tolist()],
        )

    @staticmethod
    def _signature(theta: np.ndarray) -> str:
        y_bins = np.array_split(theta, 8, axis=0)
        pooled_rows: list[np.ndarray] = []
        for band in y_bins:
            x_bins = np.array_split(band, 8, axis=1)
            pooled_rows.append(np.array([np.mean(cell) for cell in x_bins]))
        coarse = np.vstack(pooled_rows)
        quantized = np.round(coarse, 3).astype(np.float32)
        return hashlib.blake2b(quantized.tobytes(), digest_size=8).hexdigest()


def packets_to_labels(packets: Iterable[WavePacket]) -> list[str]:
    return [packet.label for packet in packets]

