"""Phase-field cognitive mesh prototype."""

from .config import MeshConfig
from .consolidation import BasinSnapshot, BasinTracker
from .encoding import TextPhaseEncoder, WavePacket
from .encoding_structured import (
    ArithmeticExpression,
    StructuredPhaseEncoder,
    parse_arithmetic,
    structured_arithmetic_feature_vector,
)
from .field import BasinFeature, PhaseFieldMesh, ResonanceMetrics
from .memory import MemoryEntry, RecallResult, TopologicalMemory
from .model import PhaseModel, PhaseObservation, PhaseVocabulary
from .probes import ArithmeticFactorReadout, fit_arithmetic_factor_readout
from .runtime import CognitiveMeshRuntime
from .verifier import VerifierResult, VerifierRouter

__all__ = [
    "CognitiveMeshRuntime",
    "BasinFeature",
    "BasinSnapshot",
    "BasinTracker",
    "MeshConfig",
    "MemoryEntry",
    "PhaseFieldMesh",
    "PhaseModel",
    "PhaseObservation",
    "PhaseVocabulary",
    "RecallResult",
    "ResonanceMetrics",
    "ArithmeticExpression",
    "ArithmeticFactorReadout",
    "StructuredPhaseEncoder",
    "TextPhaseEncoder",
    "TopologicalMemory",
    "VerifierResult",
    "VerifierRouter",
    "WavePacket",
    "parse_arithmetic",
    "fit_arithmetic_factor_readout",
    "structured_arithmetic_feature_vector",
]
