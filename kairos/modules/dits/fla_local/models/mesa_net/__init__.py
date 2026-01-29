
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.modules.dits.fla_local.models.mesa_net.configuration_mesa_net import MesaNetConfig
from kairos.modules.dits.fla_local.models.mesa_net.modeling_mesa_net import MesaNetForCausalLM, MesaNetModel

AutoConfig.register(MesaNetConfig.model_type, MesaNetConfig, exist_ok=True)
AutoModel.register(MesaNetConfig, MesaNetModel, exist_ok=True)
AutoModelForCausalLM.register(MesaNetConfig, MesaNetForCausalLM, exist_ok=True)

__all__ = ['MesaNetConfig', 'MesaNetForCausalLM', 'MesaNetModel']
