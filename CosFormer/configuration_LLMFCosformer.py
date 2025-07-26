from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from typing import Optional, Dict, Any

class LLMFCosformerConfig(PretrainedConfig):
    model_type = "llmfcosformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=36,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=2000000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=None,
        no_rope_layers=None,
        no_rope_layer_interval=4,
        layer_types=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        flow_matching: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_position_embeddings = max_position_embeddings
        self.mlp_bias = mlp_bias
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.flow_matching = flow_matching if flow_matching is not None else {
            "timestep_emb_dim": 256,
            "cond_dim": self.hidden_size,
            "n_blocks": 4,
            "n_heads": self.num_attention_heads,
            "mlp_ratio": 4,
            "dropout": self.attention_dropout 
        }
        if no_rope_layers is None:
            self.no_rope_layers = [
                int((layer_idx + 1) % no_rope_layer_interval != 0) for layer_idx in range(num_hidden_layers)
            ]
        else:
            self.no_rope_layers = no_rope_layers

        self.no_rope_layer_interval = no_rope_layer_interval

        # Update layer_types based on sliding window and NoPE pattern
        if layer_types is None:
            layer_types = []
            for layer_idx in range(num_hidden_layers):
                has_rope = self.no_rope_layers[layer_idx]
                if use_sliding_window and sliding_window is not None and not has_rope:
                    layer_types.append("sliding_attention")
                else:
                    layer_types.append("full_attention")

        self.layer_types = layer_types
        layer_type_validation(self.layer_types)

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)


__all__ = ["LLMFCosformerConfig"]
