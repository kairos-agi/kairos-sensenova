
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.modules.dits.fla_local.models.kda.configuration_kda import KDAConfig
from kairos.modules.dits.fla_local.models.kda.modeling_kda import KDAForCausalLM, KDAModel

AutoConfig.register(KDAConfig.model_type, KDAConfig, exist_ok=True)
AutoModel.register(KDAConfig, KDAModel, exist_ok=True)
AutoModelForCausalLM.register(KDAConfig, KDAForCausalLM, exist_ok=True)

__all__ = ['KDAConfig', 'KDAForCausalLM', 'KDAModel']
