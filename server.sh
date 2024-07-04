python -m vllm.entrypoints.openai.api_server \
--model /data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf \
--speculative-model /data/jieni/workspace/code/inference-toolboxes/hf_experimanets/EAGLE-llama2-chat-7B \
--num-speculative-candidates 1 \
--num-speculative-tokens 4 \
--num-lookahead-slots 4 \
--use-v2-block-manager \
--tensor-parallel-size 1 \
--speculative-draft-tensor-parallel-size 1


curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'


python3 benchmarks/benchmark_serving.py \
    --backend vllm \
    --dataset-name sharegpt \
    --dataset-path /data/jieni/workspace/datastes/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf \
    --num-prompts 2000 \
    --endpoint /v1/completions \
    --tokenizer /data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf \
    --save-result \
    2>&1 | tee benchmark_serving.txt