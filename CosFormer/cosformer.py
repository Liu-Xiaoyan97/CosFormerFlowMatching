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
from transformers.cache_utils import Cache, DynamicCache
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
    if training and prob > 0:
        x = F.dropout(x, p=prob, training=training)
    return residual + scale * x


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # cos = cos.unsqueeze(unsqueeze_dim)
    # sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimestepEmbedder(nn.Module):
    """
    改进版：更好的初始化和数值稳定性
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        # 使用更保守的初始化
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        # 改进的初始化策略
        self._init_weights()

    def _init_weights(self):
        """使用更保守的初始化避免梯度爆炸"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                # 使用较小的初始化范围
                std = 0.02  # 更小的标准差
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """创建正弦时间嵌入，添加数值稳定性"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time.device)

        # 添加小的 epsilon 防止数值问题
        args = time[:, None].float() * freqs[None]

        # 限制范围避免数值溢出
        args = torch.clamp(args, -100, 100)

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        # 归一化以保持稳定的范围
        embedding = embedding / math.sqrt(dim)

        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        # 添加 tanh 限制输出范围
        t_emb = torch.tanh(t_emb)
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
        self.k_proj = nn.Linear(self.kdim, config.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.vdim, config.hidden_size, bias=config.attention_bias)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        # outprojection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        # dropout rate
        self.dropout_rate = config.attention_dropout
        # causal
        self.causal = False
        self.has_outproj = True
        
        # Pre-compute max index size to avoid dynamic parameter creation
        self.max_seq_len = config.max_position_embeddings
        self._precompute_indices()

        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"

    def _precompute_indices(self):
        """Pre-compute indices for all possible sequence lengths"""
        max_index = np.pi / 2 * torch.arange(1, self.max_seq_len + 1).reshape(1, -1, 1)
        self.register_buffer('precomputed_index', max_index, persistent=False)

    def get_index(self, seq_len):
        """Get pre-computed index for given sequence length"""
        if seq_len > self.max_seq_len:
            # Dynamically compute for longer sequences (rare case)
            index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
            return index.to(self.precomputed_index.device)
        return self.precomputed_index[:, :seq_len, :]

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        position_embeddings: Optional[Tensor] = None,
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
        if key is None:
            key = query
        if value is None:
            value = query
        cos, sin = position_embeddings
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

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # cos transform
        m = max(src_len, tgt_len)
        # get index - now using pre-computed buffer
        weight_index = self.get_index(m)
        # (N * h, L, 2 * d)
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), 
                       q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), 
                       k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

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


class CosformerDDiTBlock(nn.Module):
    def __init__(self, config: "LLMFCosformerConfig", layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.cond_dim = config.flow_matching.get('cond_dim', config.hidden_size)
        self.dropout = config.attention_dropout
        self.mlp_ratio = 4
        self.layer_idx = layer_idx

        self.norm1 = SmolLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = SmolLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CosformerAttention(config=config, layer_idx=layer_idx)
        self.mlp = SmolLM3MLP(config)

        # AdaLN modulation with improved initialization
        self.adaLN_modulation = nn.Linear(self.cond_dim, 6 * self.hidden_size, bias=True)

        # 关键：使用接近零的初始化，让模型开始时接近恒等变换
        self._init_adaln()

    def _init_adaln(self):
        """初始化 AdaLN 使其开始时接近恒等变换"""
        # 非常小的权重初始化
        nn.init.normal_(self.adaLN_modulation.weight, mean=0.0, std=0.01)

        # 偏置初始化很关键
        with torch.no_grad():
            # 将偏置分成6部分
            bias = self.adaLN_modulation.bias.data
            bias_chunks = bias.chunk(6)

            # shift 部分初始化为0
            bias_chunks[0].zero_()  # shift_msa
            bias_chunks[3].zero_()  # shift_mlp

            # scale 部分初始化为0（因为会加1）
            bias_chunks[1].zero_()  # scale_msa
            bias_chunks[4].zero_()  # scale_mlp

            # gate 部分初始化为小值，逐层递增
            gate_init = 0.1 + 0.05 * self.layer_idx  # 逐层递增的门控值
            bias_chunks[2].fill_(gate_init)  # gate_msa
            bias_chunks[5].fill_(gate_init)  # gate_mlp

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
        if c is None:
            raise ValueError("Conditioning tensor 'c' must be provided for CosformerDDiTBlock.")

        # 应用 modulation
        modulation_params = self.adaLN_modulation(c)

        # 添加梯度裁剪以防止爆炸
        modulation_params = torch.clamp(modulation_params, -10, 10)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation_params.chunk(6, dim=1)

        # 限制 scale 的范围
        scale_msa = torch.tanh(scale_msa.unsqueeze(1) / 3) * 3  # 范围 [-3, 3]
        scale_mlp = torch.tanh(scale_mlp.unsqueeze(1) / 3) * 3

        # 限制 gate 的范围
        gate_msa = torch.sigmoid(gate_msa.unsqueeze(1))  # 范围 [0, 1]
        gate_mlp = torch.sigmoid(gate_mlp.unsqueeze(1))

        shift_msa = shift_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)

        # Self-attention block
        residual = hidden_states
        hidden_states_normed = modulate(self.norm1(hidden_states), shift_msa, scale_msa)
        hidden_states_normed_transposed = hidden_states_normed.transpose(0, 1)
        attn_output_transposed = self.self_attn(
            query=hidden_states_normed_transposed,
            key=hidden_states_normed_transposed,
            value=hidden_states_normed_transposed,
            attn_mask=attention_mask,
            position_embeddings=position_embeddings
        )
        attn_output = attn_output_transposed.transpose(0, 1)

        hidden_states = bias_dropout_add_scale(
            x=attn_output,
            scale=gate_msa,
            residual=residual,
            prob=self.dropout,
            training=self.training,
        )

        # MLP block
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
    def forward(self, x, position_ids):
        # Use no_grad to ensure no gradient computation
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
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.cond_dim = cond_dim

        self.norm_final = SmolLM3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)

        # 改进的初始化
        self._init_weights()

    def _init_weights(self):
        """初始化为接近恒等变换"""
        nn.init.normal_(self.adaLN_modulation.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            bias = self.adaLN_modulation.bias.data
            shift_bias, scale_bias = bias.chunk(2)
            shift_bias.zero_()  # shift 初始化为0
            scale_bias.zero_()  # scale 初始化为0（因为会加1）

    def forward(self, x: Tensor, c: Tensor, embedding_weight: Tensor, vocab_size: int) -> Tensor:
        modulation_params = self.adaLN_modulation(c)

        # 限制范围
        modulation_params = torch.clamp(modulation_params, -10, 10)

        shift, scale = modulation_params.chunk(2, dim=1)
        shift = shift.unsqueeze(1)
        scale = torch.tanh(scale.unsqueeze(1) / 3) * 3  # 限制 scale 范围

        x_modulated = modulate(self.norm_final(x), shift, scale)

        # 使用权重共享
        output_weight = embedding_weight[:vocab_size, :]

        # 添加温度缩放以控制初始预测的置信度
        temperature = 1.0  # 可以作为超参数调整
        logits = F.linear(x_modulated, output_weight) / temperature

        return logits

class LLFMCosformerModelBase(PreTrainedModel):
    config_class = LLMFCosformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False  # Disable gradient checkpointing
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
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Override to prevent gradient checkpointing from being enabled"""
        pass
    
    def gradient_checkpointing_disable(self):
        """Override to ensure gradient checkpointing is disabled"""
        self.gradient_checkpointing = False


class LLFMCosformerForFlowMatching(LLFMCosformerModelBase, GenerationMixin):
    def __init__(self, config: LLMFCosformerConfig, masked: bool = False):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.masked = masked

        flow_config = getattr(config, 'flow_matching', {})
        self.timestep_emb_dim = flow_config.get('timestep_emb_dim', 256)
        self.cond_dim = flow_config.get('cond_dim', config.hidden_size)
        self.flow_n_blocks = flow_config.get('n_blocks', 4)

        # 使用更保守的层归一化
        self.preNorm = SmolLM3RMSNorm(config.hidden_size, eps=1e-5)  # 增大 eps

        self.time_embedding = TimestepEmbedder(
            hidden_size=self.cond_dim,
            frequency_embedding_size=self.timestep_emb_dim
        )

        add_token = 1 if masked else 0
        self.vocab_embed = nn.Embedding(self.vocab_size + add_token, config.hidden_size)

        # 改进 embedding 初始化
        self._init_embeddings()

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
            rms_norm_eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False
        self.post_init()

    def _init_embeddings(self):
        """使用更小的初始化范围"""
        # 使用较小的标准差
        std = 0.02  # 而不是默认的 0.02 或更大
        nn.init.normal_(self.vocab_embed.weight, mean=0.0, std=std)

        # 如果有 padding_idx，将其设为零
        if self.vocab_embed.padding_idx is not None:
            with torch.no_grad():
                self.vocab_embed.weight[self.vocab_embed.padding_idx].fill_(0)

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
        **kwargs
    ) -> Union[CausalLMOutputWithPast, Tuple]:

        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Input validation
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if timesteps is None:
            raise ValueError("timesteps must be provided for flow matching forward pass.")

        # 归一化 timesteps 到 [0, 1] 范围
        timesteps = torch.clamp(timesteps, 0, 1)

        if input_ids is not None:
            inputs_embeds = self.vocab_embed(input_ids)
        else:
            inputs_embeds = kwargs.get('inputs_embeds')

        # Cache setup
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Main forward pass
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        # Position embeddings with no_grad to avoid gradient issues
        with torch.no_grad():
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = self.preNorm(hidden_states)
        # 计算时间嵌入
        c = self.time_embedding(timesteps.float())
        # 通过所有块
        for block in self.flow_blocks:
            hidden_states, = block(
                hidden_states,
                c=c,
                position_embeddings=position_embeddings,
            )

        # 最终输出
        logits = self.flow_output_layer(
            x=hidden_states,
            c=c,
            embedding_weight=self.vocab_embed.weight,
            vocab_size=self.vocab_size
        )

        return CausalLMOutputWithPast(logits=logits)