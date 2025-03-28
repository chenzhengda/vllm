"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The LMCacheConnector can (1) transfer KV caches between prefill vLLM worker
(KV cache producer) and decode vLLM worker (KV cache consumer) using LMCache;
(2) offload and share KV caches. Only (2) is supported for now.
"""

from typing import TYPE_CHECKING, List, Tuple, Union

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
import time

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class LMCacheConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):

        self.transfer_config = config.kv_transfer_config
        self.vllm_config = config

        from lmcache.integration.vllm.vllm_adapter import (
            RetrieveStatus, StoreStatus, init_lmcache_engine,
            lmcache_retrieve_kv, lmcache_should_store, lmcache_store_kv, lmcache_store_kv_layerwise, lmcache_store_hidden_states)

        logger.info("Initializing LMCacheConfig under kv_transfer_config %s",
                    self.transfer_config)

        # TODO (Jiayi): Find model_config, parallel_config, and cache_config
        self.engine = init_lmcache_engine(config.model_config,
                                          config.parallel_config,
                                          config.cache_config,
                                          config.kv_transfer_config)

        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config
        self.lmcache_retrieve_kv = lmcache_retrieve_kv
        self.lmcache_store_kv = lmcache_store_kv
        self.lmcache_store_kv_layerwise = lmcache_store_kv_layerwise
        self.lmcache_store_hidden_states = lmcache_store_hidden_states
        self.lmcache_should_store = lmcache_should_store
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        start_time = time.time()
        model_input, bypass_model_exec, hidden_or_intermediate_states = \
            self.lmcache_retrieve_kv(
                self.model_config,
                self.parallel_config,
                self.cache_config,
                model_executable,
                model_input,
                kv_caches
            )
        retrieve_time = (time.time() - start_time) * 1000  # 转换为毫秒
        logger.info(f"[{int(time.time() * 1000)}ms] KV缓存和隐藏状态获取时间: {retrieve_time:.2f}ms")

        if hidden_or_intermediate_states is None:
            bypass_model_exec = False

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        # 分层存储KV缓存
        if self.engine.config.enable_layerwise_kv:
            num_layers = self.model_config.get_num_layers(self.parallel_config)
            total_kv_time = 0
            for layer_id in range(num_layers):
                start_time = time.time()
                self.lmcache_store_kv_layerwise(
                    self.model_config,
                    self.parallel_config,
                    self.cache_config,
                    model_executable,
                    model_input,
                    kv_caches,
                    hidden_or_intermediate_states,
                    layer_id,
                )
                layer_time = (time.time() - start_time) * 1000  # 转换为毫秒
                total_kv_time += layer_time
                logger.info(f"第 {layer_id} 层 KV 缓存存储时间: {layer_time:.2f}ms")
            
            logger.info(f"[{int(time.time() * 1000)}ms] KV 缓存总存储时间: {total_kv_time:.2f}ms")

            start_time = time.time()
            self.lmcache_store_hidden_states(
                self.model_config,
                self.parallel_config,
                self.cache_config,
                model_executable,
                model_input,
                hidden_or_intermediate_states,
            )
            hidden_time = (time.time() - start_time) * 1000  # 转换为毫秒
            logger.info(f"[{int(time.time() * 1000)}ms] 隐藏状态存储时间: {hidden_time:.2f}ms")
        else:
            self.lmcache_store_kv(
                self.model_config,
                self.parallel_config,
                self.cache_config,
                model_executable,
                model_input,
                kv_caches,
                # store_status,
                hidden_or_intermediate_states,
                )

    def send_one_layer_kv_cache(self,
                                input_token_hash: List[str],    
                                model_executable: torch.nn.Module,
                                model_input: "ModelInputForGPUWithSamplingMetadata",
                                kv_caches: List[torch.Tensor],
                                layer_id: int) -> None:
        print("TODO: send_one_layer_kv_cache for layer_id: ", layer_id, "input_token_hash: ", input_token_hash, "seq_lens: ", model_input.seq_lens)
        # self.lmcache_store_kv_layerwise(
        #     self.model_config,
        #     self.parallel_config,
        #     self.cache_config,
        #     model_executable,
        #     model_input,
        #     kv_caches,
        #     None,
        #     layer_id,
        # )
        # seq_lens = attn_metadata.seq_lens
        # slot_mapping_flat = attn_metadata.slot_mapping.flatten()
        # assert len(input_token_hash) == len(seq_lens)

    def send_hidden_states(self, input_token_hash: List[str],
                           hidden_states: torch.Tensor,
                           model_input: "ModelInputForGPUWithSamplingMetadata") -> None:
        print("TODO: send_hidden_states for input_token_hash: ", input_token_hash, "seq_lens: ", model_input.seq_lens)
        # self.lmcache_store_hidden_states(
        #     self.model_config,
        #     self.parallel_config,
        #     self.cache_config,
        #     model_executable=None,  # 无需传递model_executable
        #     model_input=model_input,
        #     hidden_states=hidden_states,
        # )
        # seq_lens = attn_metadata.seq_lens
        # assert len(input_token_hash) == len(seq_lens)

    def close(self):
        self.engine.close()
