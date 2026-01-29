
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.modules.dits.fla_local.models.hgrn.configuration_hgrn import HGRNConfig
from kairos.modules.dits.fla_local.models.hgrn.modeling_hgrn import HGRNForCausalLM, HGRNModel

AutoConfig.register(HGRNConfig.model_type, HGRNConfig, exist_ok=True)
AutoModel.register(HGRNConfig, HGRNModel, exist_ok=True)
AutoModelForCausalLM.register(HGRNConfig, HGRNForCausalLM, exist_ok=True)


__all__ = ['HGRNConfig', 'HGRNForCausalLM', 'HGRNModel']
