"""Phase-field cognitive mesh prototype."""

from .config import MeshConfig
from .consolidation import BasinSnapshot, BasinTracker
from .encoding import TextPhaseEncoder, WavePacket
from .field import PhaseFieldMesh, ResonanceMetrics
from .memory import MemoryEntry, RecallResult, TopologicalMemory
from .runtime import CognitiveMeshRuntime
from .verifier import VerifierResult, VerifierRouter

__all__ = [
    "CognitiveMeshRuntime",
    "BasinSnapshot",
    "BasinTracker",
    "MeshConfig",
    "MemoryEntry",
    "PhaseFieldMesh",
    "RecallResult",
    "ResonanceMetrics",
    "TextPhaseEncoder",
    "TopologicalMemory",
    "VerifierResult",
    "VerifierRouter",
    "WavePacket",
]
