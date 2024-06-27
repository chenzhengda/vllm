import random
from typing import List, Optional, Tuple
import pytest
import torch
import argparse
import triton
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vllm.utils import create_kv_caches_with_random
from vllm.attention.ops.tree_attention import tree_attention_fwd
from vllm import _custom_ops as ops
import time

import pdb
FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
# MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 10000  # Arbitrary values for testing
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = [torch.half, torch.bfloat16]

NUM_GEN_SEQS = [1, 9, 35]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
# NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing
NUM_HEADS = [(40, 40)]
# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
# HEAD_SIZES = [64, 80, 96, 112, 128, 192, 256
#               ] if not is_hip() else [64, 80, 96, 112, 128]
HEAD_SIZES = [64, 128]

BLOCK_SIZES = [16, 32]
# USE_ALIBI = [False, True]
USE_ALIBI = [False]
# KV_CACHE_DTYPE = ["auto", "fp8"]
KV_CACHE_DTYPE = ["auto"]
SEEDS = [0]
# CUDA_DEVICES = [
#     f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
# ]

CUDA_DEVICES = [
    "cuda:0"
]

QUERY_LEN =  [1, 7, 26] # less than 32

def create_tree_attention_mask(context_len, q_len, num_kv_head, dtype):
    mask = torch.zeros((num_kv_head, q_len, context_len), dtype=dtype)
    
    min_value = torch.finfo(dtype).min
    mask_idx = torch.randperm(q_len * q_len)

    attn_mask = torch.triu(torch.ones(q_len, q_len, dtype=dtype), diagonal=1)
    attn_mask = attn_mask * torch.finfo(dtype).min
    attn_mask = attn_mask.to(dtype=dtype).view(-1)
    attn_mask[mask_idx[:q_len * 2]] = min_value
    for b in range(num_kv_head):
        mask[b, -q_len:, -q_len:] = attn_mask.view(q_len, q_len)

    # for s in range(q_len):
    #     num_masked = torch.randint(1, context_len, (1,)).item()
    #     mask_indices = torch.randperm(context_len)[:num_masked]
    #     for b in range(num_kv_head):
    #         mask[b, s, mask_indices] = min_value

    return mask

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()

    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    masks: torch.Tensor
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    block_tables = block_tables.cpu().tolist()
    seq_lens = seq_lens.cpu().tolist()
    num_seqs = len(seq_lens)

    query = query.reshape(num_seqs, -1, num_query_heads, head_size)
    output = output.reshape(query.shape)

    for i in range(num_seqs):
        q = query[i]
        block_table = block_tables[i]
        seq_len = int(seq_lens[i])

        keys = []
        values = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        mask = masks[i]
        # print(f"{mask.shape=}")
        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)
            mask += alibi_bias
        out = ref_masked_attention(q, keys, values, scale, mask)
        out = out.view(-1, num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    output.reshape(-1, num_kv_heads, head_size)


def get_input(
    kv_cache_factory,
    num_seqs: int,
    q_len: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
    MAX_SEQ_LEN: int,
):
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs * q_len, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)
    
    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None

    seq_lens = [random.randint(q_len, MAX_SEQ_LEN) for _ in range(num_seqs)]
    for i in range(len(seq_lens)):
        seq_lens[i] = seq_lens[i] if seq_lens[i] > 10 else (seq_lens[i] + 10)
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    prompt_lens = [x - q_len for x in seq_lens]
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)
    prompt_lens = torch.tensor(prompt_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    kv_scale = 1.0

    # Test for common tree_attention_fwd
    custom_masks = []
    flattened_mask_tensor = torch.zeros(num_seqs, q_len, max_seq_len, dtype=torch.float)
    for _ in range(num_seqs):
        # [num_query_heads, q_len, seq_len]
        custom_mask = create_tree_attention_mask(
            seq_lens[_], 
            q_len, 
            num_query_heads, 
            dtype=torch.float
        )
        custom_masks.append(custom_mask)
        flattened_mask_tensor[_, :, :custom_mask.shape[-1]] = custom_mask[0]
    
    return query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len, flattened_mask_tensor, num_queries_per_kv, custom_masks


@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("q_len", QUERY_LEN)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_tree_attention_v2(
    kv_cache_factory,
    num_seqs: int,
    q_len: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    alibi_slopes = None

    MAX_SEQ_LEN = 1024
    query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len, flattened_mask_tensor, num_queries_per_kv, custom_masks = get_input(
        kv_cache_factory, num_seqs, q_len, num_heads, head_size, use_alibi, block_size, dtype, kv_cache_dtype, seed, device, MAX_SEQ_LEN=MAX_SEQ_LEN,
    )

    # flattened_mask_tensor   [4, 26, 1024]
    output = torch.empty_like(query)
    tree_attention_fwd(output, query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, \
                       max_seq_len, flattened_mask_tensor)

    ref_output = torch.empty_like(query)
    ref_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        alibi_slopes,
        custom_masks
    )
    diff = torch.abs(output - ref_output)
    print(f"diff: {diff.max()}")
    (diff > 0.01).nonzero()
    atol, rtol = 1e-3, 1e-5
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol)
