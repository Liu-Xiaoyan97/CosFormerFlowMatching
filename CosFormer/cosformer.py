# Large Language Flow Matching Model based on CosFormer (LLFM-CosFormer)
# author: Liuxiaoyan
# email: liuxiaoyan97@seu.edu.cn

from typing import Callable, List, Optional, Union
import torch
import math
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel
from typing import Optional, Tuple
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from utils.flow import get_source_distribution, get_path, get_loss_function
from CosFormer.modeling_outputs import BaseCosformerOutput, CosformerOutputForCausalLM
from CosFormer.configuration_LLMFCosformer import LLMFCosformerConfig
from CosFormer.utils_nn import CosformerDecoderLayer, CosformerRMSNorm, TimestepEmbedder


class LLFMCosformerPreTrainedModel(PreTrainedModel):
    config_class = LLMFCosformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CosformerDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_cache_class = False

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


class SafeEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, scale=1.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.scale = scale
    
    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / (norm + 1e-8) * self.scale


class LLFMCosformerBaseModel(LLFMCosformerPreTrainedModel):
    def __init__(self, config: LLMFCosformerConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed_tokens = SafeEmbedding(config.vocab_size, config.hidden_size, config.scale)
        self.layers = nn.ModuleList([
            CosformerDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = CosformerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.time_embedding = TimestepEmbedder(hidden_size=config.time_embedding_dim)

        self.source_distribution = get_source_distribution(
            source_distribution=config.source_distribution,
            vocab_size=config.vocab_size
        )
        self.path = get_path(
            scheduler_type=config.scheduler_type,
            exponent=config.exponent
        )

        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        timesteps: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseCosformerOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        # https://github.com/facebookresearch/flow_matching/blob/main/examples/text/logic/training.py#L100
        # x_0 是噪声分布， t现在为timesteps, x_t是从x_0 \sim p_0, x_1 \sim p_1分布中采样的中间态
        with torch.no_grad():
            x_1 = input_ids.clone().detach()
            x_0 = self.source_distribution.sample_like(x_1)

            path_sample = self.path.sample(t=timesteps, x_0=x_0, x_1=x_1)

        time_embedding = self.time_embedding(timesteps)

        x_t_embed = self.embed_tokens(path_sample.x_t)
        model_inputs = x_t_embed + time_embedding.unsqueeze(1).expand(-1, seq_length, -1)

        # embed positions
        hidden_states = self.norm(model_inputs)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseCosformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            x_0=x_0,
            x_t=path_sample.x_t,
            x_1=x_1,
            t=timesteps,
        )


class LLMFCosformerForCausalLM(LLFMCosformerPreTrainedModel):
    _tied_weights_keys = ["embed_tokens.embedding", "lm_head"]
    def __init__(self, config: LLMFCosformerConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.model = LLFMCosformerBaseModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fn = get_loss_function(
            loss_function=config.loss_function,
            path=get_path(
                scheduler_type=config.scheduler_type,
                exponent=config.exponent
            )
        )

        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        timesteps: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CosformerOutputForCausalLM]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            timesteps=timesteps,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        x_0 = outputs.x_0
        x_1 = outputs.x_1
        x_t = outputs.x_t
        t = outputs.t
        
        logits = self.lm_head(hidden_states)

        loss_2 = F.cross_entropy(logits.view(-1, logits.size(-1)), x_1.view(-1), reduction="mean", ignore_index=-100)

        if self.config.loss_function == "cross_entropy":
            loss_1 = F.cross_entropy(logits.view(-1, logits.size(-1)), x_1.view(-1), reduction="mean", ignore_index=-100)
        elif self.config.loss_function == "generalized_kl":
            loss_1 = self.loss_fn(logits=logits, x_1=x_1, x_t=x_t, t=t).mean()
            loss_1 = torch.clamp(loss_1, max=10.0)
        else:
            raise ValueError(f"{self.config.loss_function} is not supported")

        loss = 0.6 * loss_1 + 0.4 * loss_2

        # print(f"loss_1: {loss_1.item():.4f}, loss_2: {loss_2.item():.4f}, total_loss: {loss.item():.4f}")

        return CosformerOutputForCausalLM(
            loss_1=loss_1,
            loss_2=loss_2,
            total_loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

