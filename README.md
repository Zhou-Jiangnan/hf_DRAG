# Distributed Retrieval-Augmented Generation

https://platform.openai.com/docs/api-reference/completions/create

vllm serve meta-llama/Llama-3.2-1B

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "meta-llama/Llama-3.2-1B",
     "prompt": "San Francisco is a",
     "temperature": 0
   }'


