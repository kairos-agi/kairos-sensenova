
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.modules.dits.fla_local.models.mla.configuration_mla import MLAConfig
from kairos.modules.dits.fla_local.models.mla.modeling_mla import MLAForCausalLM, MLAModel

AutoConfig.register(MLAConfig.model_type, MLAConfig, exist_ok=True)
AutoModel.register(MLAConfig, MLAModel, exist_ok=True)
AutoModelForCausalLM.register(MLAConfig, MLAForCausalLM, exist_ok=True)


__all__ = [
    'MLAConfig', 'MLAModel', 'MLAForCausalLM',
]
