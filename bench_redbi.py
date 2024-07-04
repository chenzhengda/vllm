from vllm import LLM
from vllm.sampling_params import SamplingParams
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

# tp8  65toks/s vs
# os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
# llm = LLM(
#     model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf",
#     # model="/data/jieni/workspace/models/meta-llama/Llama-2-70b-chat-hf",
#     speculative_model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/EAGLE-llama2-chat-7B",
#     # speculative_model="/data/jieni/workspace/models/yuhuili/EAGLE-llama2-chat-70B",
#     # num_speculative_candidates=1,
#     # num_speculative_tokens=4,
#     # num_lookahead_slots=4,
#     num_speculative_candidates=15,
#     num_speculative_tokens=5,
#     num_lookahead_slots=75,
#     use_v2_block_manager=True,
#     tensor_parallel_size=2,
#     speculative_draft_tensor_parallel_size=1,
#     gpu_memory_utilization=0.5,
#     enforce_eager=True,
# )

# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
llm = LLM(
    # model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf",
    # model="/data/jieni/workspace/models/meta-llama/Llama-2-70b-chat-hf",
    model="/data/jieni/workspace/models/Qwen/Qwen2-72B-Instruct-AWQ/",
    # speculative_model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/EAGLE-llama2-chat-7B",
    # speculative_model="/data/jieni/workspace/models/yuhuili/EAGLE-llama2-chat-70B",
    # speculative_model="/data/jieni/workspace/models/yuhuili/EAGLE-Qwen2-72B-Instruct/",
    # num_speculative_candidates=1,
    # num_speculative_tokens=1,
    # num_lookahead_slots=1,
    use_v2_block_manager=True,
    tensor_parallel_size=8,
    # speculative_draft_tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    enforce_eager=True,
    # quantization="fp8",
    # kv_cache_dtype="fp8",
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

import pandas as pd

df = pd.read_csv('../../llm3_test.csv')
system_prompt = df['to_sql.system_prompt'].tolist()
user_prompt = df['to_sql.user_prompt'].tolist()

# Warmup
prompts = "<|im_start|>system\n" +  system_prompt[0] + "<|im_end|>\n" + "<|im_start|>user\n" + user_prompt[0] + "<|im_end|>\n" + "<|im_start|>assistant\n"
request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
print(f"{request_outputs[0].outputs[0].text=}")

import time
import numpy as np    
timing_results = []

# for request_id in range(len(user_prompt)):
for request_id in range(10):
    start_time = time.time()
    prompts = "system\n" + system_prompt[request_id] + "\n" + "user\n" + user_prompt[request_id] + "\n" + "assistant\n"
    request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    end_time = time.time()
    elapsed_time = end_time - start_time
    timing_results.append(elapsed_time)
    print(f"Request {request_id} took {elapsed_time:.2f} seconds")

print("Timing results:", timing_results)
p90 = np.percentile(timing_results, 90)
print(f"The p90 response time is {p90:.2f} seconds")