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

## Prerequisites

- Ubuntu 22.04
- Python 3.10

Install Python requirements:

```bash
$ python install -r requirements.txt 
```

## Models

- Llama 3.2: (1B, 3B, 11B)
- Gemma 2: (2B, 9B)
- GPT (GPT-4o, GPT-4o mini, GPT-3.5)
- Claude 3.5 (Haiku, Sonnet)

## Datasets

Use `HF_HUB_CACHE` to configure where repositories from the Hub will be cached locally (models, datasets and spaces).

### Dataset Settings

| Dataset   | Repository            | Split        | Remarks |
|-----------|-----------------------|--------------|---------|
| TriviaqQA | mandarjoshi/trivia_qa | rc.nocontext |         |
|           |                       |              |         |
|           |                       |              |         |
|           |                       |              |         |
|           |                       |              |         |
|           |                       |              |         |

## Startup

```bash
$ python drag.py --config="./config/default.yaml"
```
