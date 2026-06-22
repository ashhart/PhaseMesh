from __future__ import annotations

from typing import Protocol

from .encoding import DecodedResonance


class ResonanceLike(Protocol):
    decoded: DecodedResonance

