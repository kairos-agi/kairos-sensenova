
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.modules.dits.fla_local.models.nsa.configuration_nsa import NSAConfig
from kairos.modules.dits.fla_local.models.nsa.modeling_nsa import NSAForCausalLM, NSAModel

AutoConfig.register(NSAConfig.model_type, NSAConfig, exist_ok=True)
AutoModel.register(NSAConfig, NSAModel, exist_ok=True)
AutoModelForCausalLM.register(NSAConfig, NSAForCausalLM, exist_ok=True)


__all__ = [
    'NSAConfig', 'NSAModel', 'NSAForCausalLM',
]
