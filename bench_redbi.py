from vllm import LLM
from vllm.sampling_params import SamplingParams
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

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
    model="/data/jieni/workspace/models/Qwen/Qwen2-57B-A14B-Instruct/",
    speculative_model="/mnt/nj-cos/usr/xiaodan2/code/inference/sd/eagle2/eagle/outputs/qwen2_54b_instruct/RedBI_v1/checkpoints/state_70/jieni_merged_ckpt",
    num_speculative_candidates=1,
    num_speculative_tokens=4,
    num_lookahead_slots=4,
    distributed_executor_backend="ray",
    ray_workers_use_nsight=True,
    use_v2_block_manager=True,
    tensor_parallel_size=4,
    speculative_draft_tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

import pandas as pd

df = pd.read_csv('../../llm3_test.csv')
system_prompt = df['to_sql.system_prompt'].tolist()
user_prompt = df['to_sql.user_prompt'].tolist()

# Warmup
prompts = "<|im_start|>system\n" +  system_prompt[2] + "<|im_end|>\n" + "<|im_start|>user\n" + user_prompt[2] + "<|im_end|>\n" + "<|im_start|>assistant\n"
request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
print(f"{request_outputs[0].outputs[0].text=}")

import time
import numpy as np    
timing_results = []

# for request_id in range(len(user_prompt)):
# # for request_id in range(3):
#     start_time = time.time()
#     prompts = "<|im_start|>system\n" +  system_prompt[request_id] + "<|im_end|>\n" + "<|im_start|>user\n" + user_prompt[request_id] + "<|im_end|>\n" + "<|im_start|>assistant\n"
#     request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
#     # time.sleep(0.3)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     timing_results.append(elapsed_time)
#     print(f"Request {request_id} took {elapsed_time:.2f} seconds")
#     print(f"{request_outputs[0].outputs[0].text=}")
# print("Timing results:", timing_results)
# p100 = np.percentile(timing_results, 100)
# p99 = np.percentile(timing_results, 99)
# p95 = np.percentile(timing_results, 95)
# p90 = np.percentile(timing_results, 90)
# median = np.median(timing_results)
# mean = np.mean(timing_results)
# print(f"The p100 response time is {p100:.2f} seconds")
# print(f"The p99 response time is {p99:.2f} seconds")
# print(f"The p95 response time is {p95:.2f} seconds")
# print(f"The p90 response time is {p90:.2f} seconds")
# print(f"The p90 response time is {p90:.2f} seconds")
# print(f"The median response time is {median:.2f} seconds")
# print(f"The mean response time is {mean:.2f} seconds")