
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.modules.dits.fla_local.models.transformer.configuration_transformer import TransformerConfig
from kairos.modules.dits.fla_local.models.transformer.modeling_transformer import TransformerForCausalLM, TransformerModel

AutoConfig.register(TransformerConfig.model_type, TransformerConfig, exist_ok=True)
AutoModel.register(TransformerConfig, TransformerModel, exist_ok=True)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)


__all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel']
