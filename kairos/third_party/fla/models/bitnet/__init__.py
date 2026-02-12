
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.third_party.fla.models.bitnet.configuration_bitnet import BitNetConfig
from kairos.third_party.fla.models.bitnet.modeling_bitnet import BitNetForCausalLM, BitNetModel

AutoConfig.register(BitNetConfig.model_type, BitNetConfig, exist_ok=True)
AutoModel.register(BitNetConfig, BitNetModel, exist_ok=True)
AutoModelForCausalLM.register(BitNetConfig, BitNetForCausalLM, exist_ok=True)


__all__ = ['BitNetConfig', 'BitNetForCausalLM', 'BitNetModel']
