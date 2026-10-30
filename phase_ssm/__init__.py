"""PhaseSSM — a trainable damped-oscillator state-space language model.

The PhaseMesh phase-field intuition (channels as damped oscillators with their own
frequency and damping), but with every parameter learned by gradient descent.
"""
from .model import PhaseSSMConfig, PhaseSSMLM, OscillatorySSM
from .recurrent import ssm_chunked, ssm_chunked_real, ssm_recurrent, ssm_recurrent_real

__all__ = [
    "PhaseSSMConfig",
    "PhaseSSMLM",
    "OscillatorySSM",
    "ssm_chunked",
    "ssm_chunked_real",
    "ssm_recurrent",
    "ssm_recurrent_real",
]
