# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import send2trash
import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import is_hip, print_warning_once

from .interfaces import SupportsLoRA
from vllm.distributed import (broadcast_tensor_dict, get_pp_group,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config=config,
                              cache_config=cache_config,
                              quant_config=quant_config)
            for idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

def pad_path(path, length, pad_value=-2):
    return path + [pad_value] * (length - len(path))

class node:

    def __init__(self, parent=None, value=None, dict_key=None):
        self.parent = parent
        self.value = value
        if parent:
            self.depth = parent.depth + 1
            parent.children.append(self)
        else:
            self.depth = 0
        self.children = []
        self.dict_key = dict_key

    def is_leaf(self):
        return len(self.children) == 0

    def all_index(self):
        if not self.parent.parent:
            return [self.index]
        else:
            return self.parent.all_index() + [self.index]

class Tree:

    def __init__(self, tree_list):
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root = node()
        self.node_dic = {}
        for tree_node in sorted_tree_list:
            cur_value = tree_node[-1]
            if len(tree_node) == 1:
                cur_node = node(parent=self.root,
                                value=cur_value,
                                dict_key=tuple(tree_node))
            else:
                cur_parent = self.node_dic[tuple(tree_node[:-1])]
                cur_node = node(parent=cur_parent,
                                value=cur_value,
                                dict_key=tuple(tree_node))
            self.node_dic[tuple(tree_node)] = cur_node
        self.indexnode()

    def max_depth(self):
        return max([item.depth for item in self.node_dic.values()])

    def num_node_wchild(self):
        num_c = 0
        for item in self.node_dic.values():
            if not item.is_leaf():
                num_c += 1
        return num_c

    def get_node_wchild(self):
        ns = []
        for item in self.node_dic.values():
            if not item.is_leaf():
                ns.append(item)
        return ns

    def indexnode(self):
        cur_index = 0
        for key in self.node_dic:
            cur_node = self.node_dic[key]
            if not cur_node.is_leaf():
                cur_node.index = cur_index
                cur_index += 1

class LlamaForCausalLM(nn.Module, SupportsLoRA):
    supports_lora = True

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

        self.tree_choices = [[0], [1], [2], [3], [0, 0], [0, 1], [0, 2],
                             [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [0, 0, 0],
                             [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1],
                             [0, 2, 0], [0, 2, 1], [1, 0, 0], [0, 0, 0, 0],
                             [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1]]
        # self.tree_choices = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0],
        #                      [0, 0, 0, 0, 0]]  # # 5st depth we choose top 1
        self.tree_topk = 10
        self.target_tree_buffer = self.generate_target_tree_buffers(
            self.tree_choices)
        self.tree_mask = self.target_tree_buffer["tree_attn_mask"]

    def generate_target_tree_buffers(self, tree_choices):
        import copy
        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(
                        sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[
                    start + j +
                    1] = cur_tree_choice[-1] + self.tree_topk * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + self.tree_topk * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1:start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(
                        sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [
            pad_path(path, max_length) for path in retrieve_indices_nest
        ]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([
            torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long),
            retrieve_indices
        ],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5

        def custom_sort(lst):
            # sort_keys=[len(list)]
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

        p_indices = torch.tensor(p_indices)
        p_indices_new = p_indices[retrieve_indices]
        p_indices_new = p_indices_new.tolist()

        b_indices = [[]] + b_indices
        b_indices_new = []
        for ib in range(retrieve_indices.shape[0]):
            iblist = []
            for jb in range(retrieve_indices.shape[1]):
                index = retrieve_indices[ib, jb]
                if index == -1:
                    iblist.append([])
                else:
                    b = b_indices[index]
                    if len(b) > 0:
                        bt = []
                        for bi in b:
                            bt.append(
                                torch.where(tree_indices == bi)[0].item())
                        iblist.append(torch.tensor(bt))
                    else:
                        iblist.append(b)
            b_indices_new.append(iblist)

        # Aggregate the generated buffers into a dictionary
        tree_buffers = {
            "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
            "tree_indices": tree_indices,
            "tree_position_ids": tree_position_ids,
            "retrieve_indices": retrieve_indices,
        }

        # Move the tensors in the dictionary to the specified device
        tree_buffers = {
            k: v.clone() if isinstance(v, torch.Tensor) else torch.tensor(v)
            for k, v in tree_buffers.items()
        }
        tree_buffers["p_indices"] = p_indices_new
        tree_buffers["b_indices"] = b_indices_new
        return tree_buffers

    def _make_causal_mask(
        self,
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len),
                          torch.finfo(dtype).min,
                          device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1),
                          0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [
                    torch.zeros(tgt_len,
                                past_key_values_length,
                                dtype=dtype,
                                device=device),
                    mask,
                ],
                dim=-1,
            )
        return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                             tgt_len + past_key_values_length)

    def _expand_mask(self,
                     mask: torch.Tensor,
                     dtype: torch.dtype,
                     tgt_len: Optional[int] = None):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len,
                                                      src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool),
                                         torch.finfo(dtype).min)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask,
                                                   inputs_embeds.dtype,
                                                   tgt_len=input_shape[-1]).to(
                                                       inputs_embeds.device)
            combined_attention_mask = (expanded_attn_mask
                                       if combined_attention_mask is None else
                                       expanded_attn_mask +
                                       combined_attention_mask)

        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            bs = combined_attention_mask.size(0)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask.repeat(bs, 1, 1,
                                 1) == 0] = combined_attention_mask.min()

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.tree_choices != [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]] and attn_metadata.num_decode_tokens > 0 and attn_metadata.max_query_len > 1:
            position_ids = self.target_tree_buffer["tree_position_ids"][
                None, :].to(attn_metadata.context_lens_tensor.device
                            ) + attn_metadata.context_lens_tensor

            # Initialize a mask of zeros with the desired shape (4x9)
            mask = torch.zeros(
                (len(attn_metadata.decode_metadata.seq_lens_tensor),
                 max(attn_metadata.decode_metadata.seq_lens_tensor)),
                device=input_ids.device,
                dtype=torch.int32)

            # Fill the mask based on the lengths in context_lens_tensor
            for i, length in enumerate(
                    attn_metadata.decode_metadata.seq_lens_tensor):
                mask[i, -length:] = 1

            attention_mask = self._prepare_decoder_attention_mask(
                mask, (len(attn_metadata.decode_metadata.seq_lens_tensor),
                       len(self.target_tree_buffer["tree_attn_mask"][0][0])),
                kv_caches[0],
                max(attn_metadata.decode_metadata.context_lens_tensor))

            attn_metadata.decode_metadata.attn_masks = attention_mask
            hidden_states = self.model(input_ids, position_ids, kv_caches,
                                       attn_metadata)

            logits = torch.matmul(hidden_states, self.lm_head.weight.t())
            logits = tensor_model_parallel_all_gather(logits)
            probs = torch.softmax(logits, dim=-1)

            logits = logits.view(
                len(attn_metadata.block_tables), -1,
                logits.shape[-1])[:,
                                  self.target_tree_buffer["retrieve_indices"]]
            probs = probs.view(
                len(attn_metadata.block_tables), -1,
                probs.shape[-1])[:,
                                 self.target_tree_buffer["retrieve_indices"]]

            scorer_hidden_states = hidden_states.view(
                len(attn_metadata.block_tables), -1, hidden_states.shape[-1]
            )[:, self.target_tree_buffer["retrieve_indices"]]
            token_ids = torch.argmax(logits, dim=-1)

            return token_ids, probs, scorer_hidden_states
        else:
            hidden_states = self.model(input_ids, positions, kv_caches,
                                    attn_metadata)
            return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(
                        ".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        print_warning_once(
                            f"Found kv scale in the checkpoint (e.g. {name}), "
                            "but not found the expected name in the model "
                            f"(e.g. {remapped_kv_scale_name}). kv-scale is "
                            "not loaded.")
                        continue
                    else:
                        name = remapped_kv_scale_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")

    def split_kv_cache(
        self,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    def defragment_accepted_kv_blocks(
            self,
            attn_metadata: AttentionMetadata,
            best_candidate_index: torch.Tensor,
            accepted_token_ids: torch.Tensor,
            kv_caches: List[torch.Tensor]):
        pass

        slot_mapping: List[int] = []
        src_slot_mapping: List[int] = []
        dst_slot_mapping: List[int] = []    
        self.block_size = 16
        accepted_lengths = (accepted_token_ids != -1).sum(dim=1, keepdim=True)

        for batch_size, block_table in enumerate(attn_metadata.block_tables):
            context_len = attn_metadata.seq_lens[batch_size] - 1
            for i in range(
                    context_len, context_len +
                    len(self.target_tree_buffer["tree_indices"])):
                block_number = block_table[i // self.block_size].item()
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)
            for dst_index, src_index in enumerate(
                        self.target_tree_buffer["retrieve_indices"][
                            best_candidate_index[batch_size]]):
                if dst_index <= accepted_lengths[batch_size] and dst_index != src_index:
                    src_slot_mapping.append(slot_mapping[src_index])
                    dst_slot_mapping.append(slot_mapping[dst_index])

        for layer_id in range(len(self.model.layers)):
            for i, src_slot in enumerate(src_slot_mapping):
                key_cache, value_cache = self.split_kv_cache(
                    kv_caches[layer_id],
                    self.model.layers[0].self_attn.num_kv_heads,
                    self.model.layers[0].self_attn.head_dim)
                src_block_number = int(
                    src_slot_mapping[i] //
                                self.block_size)
                src_block_offset = src_slot_mapping[
                    i] % self.block_size
                dst_block_number = int(
                    dst_slot_mapping[i] //
                                self.block_size)
                dst_block_offset = dst_slot_mapping[
                    i] % self.block_size

                key_cache[dst_block_number, :, :,
                            dst_block_offset, :] = key_cache[
                                src_block_number, :, :,
                                src_block_offset, :]
                value_cache[dst_block_number, :, :,
                            dst_block_offset] = value_cache[
                                src_block_number, :, :,
                                src_block_offset]
