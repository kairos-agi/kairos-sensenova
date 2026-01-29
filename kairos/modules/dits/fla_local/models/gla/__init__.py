
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.modules.dits.fla_local.models.gla.configuration_gla import GLAConfig
from kairos.modules.dits.fla_local.models.gla.modeling_gla import GLAForCausalLM, GLAModel

AutoConfig.register(GLAConfig.model_type, GLAConfig, exist_ok=True)
AutoModel.register(GLAConfig, GLAModel, exist_ok=True)
AutoModelForCausalLM.register(GLAConfig, GLAForCausalLM, exist_ok=True)


__all__ = ['GLAConfig', 'GLAForCausalLM', 'GLAModel']
