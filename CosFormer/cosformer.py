# Large Language Flow Matching Model based on CosFormer (LLFM-CosFormer)
# author: Liuxiaoyan
# email: liuxiaoyan97@seu.edu.cn

from typing import Callable, Optional, Union
import torch
import math
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel
from torch import Tensor
from typing import Optional, Tuple
from torch import nn
from transformers import GradientCheckpointingLayer
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN
from transformers.utils import auto_docstring, can_return_tuple
from transformers.masking_utils import create_causal_mask
from transformers.modeling_rope_utils import dynamic_rope_update, ROPE_INIT_FUNCTIONS
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from CosFormer.configuration_LLMFCosformer import LLMFCosformerConfig


def bias_dropout_add_scale(
    x: Tensor, scale: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])

        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CosformerAttention(nn.Module):
    """
    cosformer attention in "cosFormer: Rethinking Softmax In Attention"
    https://arxiv.org/abs/2202.08791
    """
    def __init__(
        self,
        config: LLMFCosformerConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.kdim = config.hidden_size if config.hidden_size is not None else config.hidden_size
        self.vdim = config.hidden_size if config.hidden_size is not None else config.hidden_size
        self.num_heads = config.num_attention_heads
        self.act_fun = ACT2FN[config.hidden_act]
        # q, k, v projection
        self.k_proj = nn.Linear(self.kdim, config.hidden_size)
        self.v_proj = nn.Linear(self.vdim, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # outprojection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # dropout rate
        self.dropout_rate = config.attention_dropout
        # causal
        self.causal = False
        self.has_outproj = True

        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)


    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            attn_mask (Optional[Tensor], optional): typically used to implement causal attention, 
            where the mask prevents the attention from looking forward in time (default: None).
        """
        if key == None:
            key = query
        if value == None:
            value = query
        
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.q_proj(query)
        # (S, N, E)
        k = self.k_proj(key)
        # (S, N, E)
        v = self.v_proj(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
            kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
            qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
            k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
            denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
            attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)

        return attn_output

    def left_product(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            attn_mask (Optional[Tensor], optional): typically used to implement causal attention, 
            where the mask prevents the attention from looking forward in time (default: None).
        """
        # test for the correctness of the program
        if key == None:
            key = query
        if value == None:
            value = query
        
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.q_proj(query)
        # (S, N, E)
        k = self.k_proj(key)
        # (S, N, E)
        v = self.v_proj(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            weights = weights.masked_fill(attn_mask==float("-inf"), 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)

        return attn_output

class SmolLM3RMSNorm(nn.Module):
    """
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/smollm3/modeling_smollm3.py#L203
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        SmolLM3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SmolLM3MLP(nn.Module):
    """
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/smollm3/modeling_smollm3.py#L223
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class CosformerDDiTBlock(GradientCheckpointingLayer):
    def __init__(self, config: "LLMFCosformerConfig", layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.cond_dim = config.hidden_size
        self.dropout = config.attention_dropout
        self.mlp_ratio = 4 
        self.layer_idx = layer_idx
        self.norm1 = SmolLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = SmolLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CosformerAttention(config=config, layer_idx=layer_idx)
        self.mlp = SmolLM3MLP(config)
        self.adaLN_modulation = nn.Linear(self.cond_dim, 6 * self.hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
        self,
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        c: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor]:
        """
            Forward pass for the Cosformer-based DDiT Block.

            Args:
                hidden_states (torch.Tensor): Input tensor of shape [B, S, D].
                attention_mask (Optional[torch.Tensor]): Attention mask.
                position_ids (Optional[torch.LongTensor]): Position IDs.
                past_key_value (Optional[Cache]): Past key/value cache.
                use_cache (Optional[bool]): Whether to use cache.
                cache_position (Optional[torch.LongTensor]): Cache position.
                position_embeddings (Optional[Tuple[torch.Tensor, torch.Tensor]]): Rotary position embeddings (not used here).
                c (Optional[torch.Tensor]): Conditioning tensor of shape [B, Cond_Dim].

            Returns:
                Tuple[torch.Tensor]: Output tensor of shape [B, S, D].
            """
        batch_size, seq_len, _ = hidden_states.shape
        if c is None:
            raise ValueError("Conditioning tensor 'c' must be provided for CosformerDDiTBlock.")

        modulation_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation_params.chunk(6, dim=1)
        shift_msa = shift_msa.unsqueeze(1) # [B, 1, D]
        scale_msa = scale_msa.unsqueeze(1) # [B, 1, D]
        gate_msa = gate_msa.unsqueeze(1)   # [B, 1, D]
        shift_mlp = shift_mlp.unsqueeze(1) # [B, 1, D]
        scale_mlp = scale_mlp.unsqueeze(1) # [B, 1, D]
        gate_mlp = gate_mlp.unsqueeze(1)   # [B, 1, D]

        residual = hidden_states
        hidden_states_normed = modulate(self.norm1(hidden_states), shift_msa, scale_msa) 
        hidden_states_normed_transposed = hidden_states_normed.transpose(0, 1)
        attn_output_transposed = self.self_attn(
            query=hidden_states_normed_transposed,
            key=hidden_states_normed_transposed,
            value=hidden_states_normed_transposed,
            attn_mask=attention_mask, 
        )
        attn_output = attn_output_transposed.transpose(0, 1)
        hidden_states = bias_dropout_add_scale(
            x=attn_output,
            scale=gate_msa,
            residual=residual,
            prob=self.dropout,
            training=self.training,
        )
        residual = hidden_states
        hidden_states_normed = modulate(self.norm2(hidden_states), shift_mlp, scale_mlp) 
        mlp_output = self.mlp(hidden_states_normed)
        hidden_states = bias_dropout_add_scale(
            x=mlp_output,
            scale=gate_mlp,
            residual=residual,
            prob=self.dropout,
            training=self.training,
        ) 
        return (hidden_states,)


class SmolLM3RotaryEmbedding(nn.Module):
    """
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/smollm3/modeling_smollm3.py#L319
    """
    def __init__(self, config: LLMFCosformerConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int, rms_norm_eps: float = 1e-6):
        """
        Final layer for DDiT, adapted for CosformerDDiTBlock context.
        Uses SmolLM3RMSNorm for consistency with the main model.

        Args:
            hidden_size (int): The dimension of the input hidden states (D).
            out_channels (int): The number of output channels (e.g., vocab size).
            cond_dim (int): The dimension of the conditioning signal (c). Should match
                            the output dimension of TimestepEmbedder (often hidden_size).
            rms_norm_eps (float): Epsilon for RMSNorm.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.cond_dim = cond_dim # Store for clarity

        # 使用与主干相同的 RMSNorm
        self.norm_final = SmolLM3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.linear = nn.Linear(hidden_size, out_channels)

        # Adaptive Layer Normalization modulation
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Forward pass for the final DDiT layer.

        Args:
            x (Tensor): Input hidden states of shape [B, S, D].
            c (Tensor): Conditioning signal of shape [B, Cond_Dim].

        Returns:
            Tensor: Output logits of shape [B, S, Out_Channels].
        """
        batch_size = x.shape[0]
        modulation_params = self.adaLN_modulation(c)
        shift, scale = modulation_params.chunk(2, dim=1) 
        shift = shift.unsqueeze(1) # [B, 1, D]
        scale = scale.unsqueeze(1) # [B, 1, D]

        x_modulated = modulate(self.norm_final(x), shift, scale) # [B, S, D]
        logits = self.linear(x_modulated) # [B, S, Out_Channels]

        return logits

class LLFMCosformerModelBase(PreTrainedModel):
    config_class = LLMFCosformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CosformerDDiTBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": CosformerDDiTBlock,
        "attentions": CosformerAttention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SmolLM3RMSNorm):
            module.weight.data.fill_(1.0)


class LLFMCosformerForFlowMatching(LLFMCosformerModelBase, GenerationMixin):
    def __init__(self, config: LLMFCosformerConfig, masked: bool = False):
        super().__init__(config)
        self.vocab_size = config.vocab_size

        flow_config = getattr(config, 'flow_matching', {})
        self.timestep_emb_dim = flow_config.get('timestep_emb_dim', 256)
        self.cond_dim = flow_config.get('cond_dim', config.hidden_size)
        self.flow_n_blocks = flow_config.get('n_blocks', 4)
        self.flow_n_heads = flow_config.get('n_heads', config.num_attention_heads)
        self.flow_mlp_ratio = flow_config.get('mlp_ratio', 4)
        self.flow_dropout = flow_config.get('dropout', config.attention_dropout) # 使用 attention_dropout 作为默认

        self.time_embedding = TimestepEmbedder(hidden_size=self.cond_dim, frequency_embedding_size=self.timestep_emb_dim)
        add_token = 1 if masked else 0

        self.vocab_embed = nn.Embedding(self.vocab_size + add_token, config.hidden_size)
        self.rotary_emb = SmolLM3RotaryEmbedding(config=config)
        self.flow_blocks = nn.ModuleList([
            CosformerDDiTBlock(
                config=config,
                layer_idx=i, 
            ) for i in range(self.flow_n_blocks)
        ])

        self.flow_output_layer = DDitFinalLayer(
            hidden_size=config.hidden_size,
            out_channels=self.vocab_size, 
            cond_dim=self.cond_dim,
            rms_norm_eps=config.rms_norm_eps, 
        )
        # ------------------

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        timesteps: Optional[torch.FloatTensor] = None, 
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args, **kwargs
    ) -> Union[CausalLMOutputWithPast, Tuple]:

        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.vocab_embed(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        causal_mask_mapping = {"full_attention": attention_mask} 
        all_hidden_states = () if output_hidden_states else None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if timesteps is None:
            raise ValueError("timesteps must be provided for flow matching forward pass.")
        # print(f"------->{timesteps.dtype}")
        c = F.silu(self.time_embedding(timesteps.float())) 

        for block in self.flow_blocks:
            hidden_states, = block(
                hidden_states,
                attention_mask=attention_mask,
                c=c,
                position_embeddings=position_embeddings,
                position_ids=position_ids, 
                past_key_value=None, use_cache=False, cache_position=None # Flow matching 通常不使用 cache
            )


        logits = self.flow_output_layer(x=hidden_states, c=c) # Shape: (B, S, Vocab_Size)
        loss = None
        if not return_dict:
            outputs = (logits,) + outputs[1:] 
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
