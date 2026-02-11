
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from kairos.third_party.fla.models.path_attn.configuration_path_attention import PaTHAttentionConfig
from kairos.third_party.fla.models.path_attn.modeling_path_attention import PaTHAttentionForCausalLM, PaTHAttentionModel

AutoConfig.register(PaTHAttentionConfig.model_type, PaTHAttentionConfig, exist_ok=True)
AutoModel.register(PaTHAttentionConfig, PaTHAttentionModel, exist_ok=True)
AutoModelForCausalLM.register(PaTHAttentionConfig, PaTHAttentionForCausalLM, exist_ok=True)


__all__ = ['PaTHAttentionConfig', 'PaTHAttentionForCausalLM', 'PaTHAttentionModel']
