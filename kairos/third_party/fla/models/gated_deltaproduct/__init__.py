from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.third_party.fla.models.gated_deltaproduct.configuration_gated_deltaproduct import GatedDeltaProductConfig
from kairos.third_party.fla.models.gated_deltaproduct.modeling_gated_deltaproduct import GatedDeltaProductForCausalLM, GatedDeltaProductModel

AutoConfig.register(GatedDeltaProductConfig.model_type, GatedDeltaProductConfig, exist_ok=True)
AutoModel.register(GatedDeltaProductConfig, GatedDeltaProductModel, exist_ok=True)
AutoModelForCausalLM.register(GatedDeltaProductConfig, GatedDeltaProductForCausalLM, exist_ok=True)

__all__ = [
    "GatedDeltaProductConfig",
    "GatedDeltaProductForCausalLM",
    "GatedDeltaProductModel",
]
