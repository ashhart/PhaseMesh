"""Phase-field cognitive mesh prototype."""

from .config import MeshConfig
from .chat_lm import PhaseChatConfig, PhaseChatModel, PhaseChatRecord
from .consolidation import BasinSnapshot, BasinTracker
from .encoding import TextPhaseEncoder, WavePacket
from .encoding_structured import (
    ArithmeticExpression,
    StructuredPhaseEncoder,
    parse_arithmetic,
    structured_arithmetic_feature_vector,
)
from .field import BasinFeature, PhaseFieldMesh, ResonanceMetrics
from .learnable_core import run_learnable_core_probe
from .language_model import PhaseLanguageModel, PhaseLMConfig
from .memory import MemoryEntry, RecallResult, TopologicalMemory
from .model import PhaseModel, PhaseObservation, PhaseVocabulary
from .phase_accio import PhaseAccioSketch, run_phase_accio
from .phase_advantage import DistributedPhaseAssociativeMemory, run_phase_advantage
from .phase_advantage_docs import NaturalPhaseMemory, run_phase_advantage_docs
from .phase_binding_hard import RolePhaseMemory, run_phase_binding_hard
from .probes import ArithmeticFactorReadout, fit_arithmetic_factor_readout
from .runtime import CognitiveMeshRuntime
from .verifier import VerifierResult, VerifierRouter
from .registry import PhaseMeshRegistry
from .llm_shell import PhaseMeshLLMShell
from .weight_reader import PhaseWeightReader, PhaseWeightReadoutConfig
from .weight_pour import PhaseWeightPourConfig, load_weight_manifest, pour_arrays_to_phase, pour_hf_checkpoint_to_phase

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
    "PhaseLanguageModel",
    "PhaseLMConfig",
    "PhaseChatConfig",
    "PhaseChatModel",
    "PhaseChatRecord",
    "PhaseWeightPourConfig",
    "PhaseWeightReader",
    "PhaseWeightReadoutConfig",
    "PhaseAccioSketch",
    "DistributedPhaseAssociativeMemory",
    "NaturalPhaseMemory",
    "RolePhaseMemory",
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
    "PhaseMeshRegistry",
    "PhaseMeshLLMShell",
    "parse_arithmetic",
    "fit_arithmetic_factor_readout",
    "load_weight_manifest",
    "pour_arrays_to_phase",
    "pour_hf_checkpoint_to_phase",
    "run_learnable_core_probe",
    "run_phase_accio",
    "run_phase_advantage",
    "run_phase_advantage_docs",
    "run_phase_binding_hard",
    "structured_arithmetic_feature_vector",
]
