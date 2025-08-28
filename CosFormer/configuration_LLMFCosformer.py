from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from typing import Optional, Dict, Any
from omegaconf import OmegaConf
import os

class LLMFCosformerConfig(PretrainedConfig):
    model_type = "llmfcosformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.1,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=2000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.1,
        pos_embedding_dim=64,
        kv_rope_head_dim=32,
        max_position_seq=2048,
        base=10000.0,
        time_embedding_dim=512,
        mlp_bias=False,
        flow: Dict={
            'scheduler_type': 'polynomial',
            'exponent': 2.0,
            'loss_function': 'generalized_kl',
            'source_distribution': 'uniform',
            'time_epsilon': 0.001,
            'sde_t': 1.0
        },
        scale=1.0,
        **kwargs,
    ):
        # Try to load configuration from train.yaml file if exists
        yaml_config = self._load_yaml_config()
        
        # Override with train.yaml values if available
        if yaml_config:
            # Model config
            model_config = yaml_config.get('model', {})
            if model_config:
                vocab_size = model_config.get('vocab_size', vocab_size)
                hidden_size = model_config.get('hidden_size', hidden_size)
                intermediate_size = model_config.get('intermediate_size', intermediate_size)
                num_hidden_layers = model_config.get('num_hidden_layers', num_hidden_layers)
                num_attention_heads = model_config.get('num_attention_heads', num_attention_heads)
                max_position_embeddings = model_config.get('max_position_embeddings', max_position_embeddings)
                rms_norm_eps = model_config.get('rms_norm_eps', rms_norm_eps)
                kv_rope_head_dim = model_config.get('kv_rope_head_dim', kv_rope_head_dim)
                hidden_act = model_config.get('hidden_act', hidden_act)
                attention_bias = model_config.get('attention_bias', attention_bias)
                attention_dropout = model_config.get('attention_dropout', attention_dropout)
                base = model_config.get('base', base)
                max_position_seq = model_config.get('max_position_seq', max_position_seq)
                time_embedding_dim = model_config.get('time_embedding_dim', time_embedding_dim)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pos_embedding_dim = pos_embedding_dim
        self.max_position_seq = max_position_seq
        self.base = base
        self.kv_rope_head_dim = kv_rope_head_dim
        self.mlp_bias = mlp_bias
        self.time_embedding_dim = time_embedding_dim
        
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.scale = scale
        self.dropout = attention_dropout
        self.source_distribution = flow["source_distribution"]
        self.scheduler_type = flow["scheduler_type"]
        self.exponent = flow["exponent"]
        self.loss_function = flow["loss_function"]
        self.time_epsilon = flow["time_epsilon"]
        self.sde_t = flow["sde_t"]
        
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
    
    def _load_yaml_config(self):
        """Try to load configuration from train.yaml"""
        config_path = "config/train.yaml"
        if os.path.exists(config_path):
            try:
                return OmegaConf.load(config_path)
            except Exception:
                return None
        return None


__all__ = ["LLMFCosformerConfig"]