# Distributed Retrieval-Augmented Generation

## Prerequisites

- Ubuntu 22.04
- Python 3.10
- Ollama 0.4.0

Install Python requirements:

```bash
$ python install -r requirements.txt 
```

Start [Ollama](https://github.com/ollama/ollama/blob/main/docs/faq.md) local LLM service:

```bash
# start ollama service:
$ sudo systemctl start ollama
# start serving model "Llama 3.2 - 3B"
# for other Ollama models: https://ollama.com/library
$ ollama run llama3.2
# preload a model into Ollama to get faster response times:
$ curl http://localhost:11434/api/generate -d '{"model": "llama3.2"}'
```

Install [NLTK Data](https://www.nltk.org/data.html)

```bash
$ python -m nltk.downloader all
```

## Models

- Llama 3.2: (1B, 3B, 11B)
- Gemma 2: (2B, 9B)
- GPT (GPT-4o, GPT-4o mini, GPT-3.5)
- Claude 3.5 (Haiku, Sonnet)

## Datasets

Use `HF_HUB_CACHE` to configure where repositories from the Hub will be cached locally (models, datasets and spaces).

### Dataset Settings

| Dataset          | Repository                        | Split                     | Remarks |
|------------------|-----------------------------------|---------------------------|---------|
| TriviaqQA        | mandarjoshi/trivia_qa             | rc.nocontext - validation |         |
| NaturalQuestions | lighteval/natural_questions_clean | validation                |         |
|                  |                                   |                           |         |
|                  |                                   |                           |         |
|                  |                                   |                           |         |
|                  |                                   |                           |         |

## Startup

```bash
$ python drag.py --config="./config/default.yaml"
```

## DEBUG


