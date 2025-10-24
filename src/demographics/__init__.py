# ABOUTME: Demographics module for synthetic consumer attribute management
# ABOUTME: Handles demographic profiles, sampling strategies, persona conditioning, and bias detection

from .profiles import DemographicProfile, Location
from .sampling import DemographicSampler, SamplingStrategy, SamplingConfig
from .persona_conditioning import (
    PersonaConditioner,
    ConditioningMode,
    ConditioningConfig,
    ConditioningResult,
)
from .bias_detection import BiasDetector, BiasReport

__all__ = [
    "DemographicProfile",
    "Location",
    "DemographicSampler",
    "SamplingStrategy",
    "SamplingConfig",
    "PersonaConditioner",
    "ConditioningMode",
    "ConditioningConfig",
    "ConditioningResult",
    "BiasDetector",
    "BiasReport",
]
