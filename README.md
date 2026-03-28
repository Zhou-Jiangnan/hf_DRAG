# Distributed Retrieval-Augmented Generation

## Prerequisites

- Ubuntu 22.04
- Python 3.10
- Ollama 0.5.7

Install Python requirements:

```bash
$ pip install -r requirements.txt 
```

Install and start [Ollama](https://ollama.com/download/linux) local LLM service:

```bash
# start ollama service:
$ sudo systemctl start ollama
# start serving model "Llama 3.2 - 3B"
# for other Ollama models: https://ollama.com/library
$ ollama run llama3.2:3b
# preload a model into Ollama to get faster response times:
$ curl http://localhost:11434/api/generate -d '{"model": "llama3.2:3b"}'
# check ollama log
$ journalctl -u ollama
```

If you do not have root permission, install and start Ollama [manually](https://github.com/ollama/ollama/blob/main/docs/linux.md):

```bash
# Download and extract the package:
$ curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
$ sudo tar -C <your-path>/ -xzf ollama-linux-amd64.tgz
# Create user-mode systemd.service file `~/.config/systemd/user/ollama.service` with the following content:
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=<your-path>/bin/ollama serve
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
# Start systemd ollama service in user mode:
$ systemctl --user enable ollama
$ systemctl --user start ollama
# Check ollama service log:
$ journalctl --user -f -u ollama
```

Install [NLTK Data](https://www.nltk.org/data.html)

```bash
$ python -m nltk.downloader all
```

## Models

The following Ollama models are utilized in our experiments.

- [Llama 3.2 3B](https://ollama.com/library/llama3.2:3b)
- [Gemma 2 2B](https://ollama.com/library/gemma2:2b)
- [Qwen 2.5 3B](https://ollama.com/library/qwen2.5:3b)

## Datasets

The following Hugging Face datasets are utilized in our experiments.

- [MMLU](https://huggingface.co/datasets/cais/mmlu)
- [medical_extended](https://huggingface.co/datasets/sarus-tech/medical_extended)
- [news-category-dataset](https://huggingface.co/datasets/heegyu/news-category-dataset)
- [NQ-Open](https://huggingface.co/datasets/google-research-datasets/nq_open)
- [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)

> Use `HF_HUB_CACHE` to configure where repositories from the Hub will be cached locally (models, datasets and spaces).

## Run

```bash
$ python simulator.py
```

## PPO Routing (DRAG)

Set `rag.search_algorithm: 'PPO'` in `config/rag.yaml` to train a PPO router before inference.
The PPO router is trained on retrieval-only signals (relevance/hit/hop/message rewards) and then
used as new routing policies alongside TARW/RW/FL (PPO and GRPO).

By default, PPO uses `rag.ppo_device: auto` and will run on GPU (`cuda`) when available,
otherwise fallback to CPU.

When `rag.test_mode` is enabled, the simulator now uses `rag.test_num_samples`
instead of a hard-coded 20 samples, so you can quickly scale debug/eval size.

### How to train and test PPO in this project

The simulator performs **training first** and then **evaluation** in one run when
`rag.search_algorithm=PPO` and `rag.network_type=DRAG`.

1. **Select dataset config**
   - MMLU (default): `./config/data/mmlu.yaml`
   - medical: `./config/data/medical.yaml`
   - news: `./config/data/news.yaml`
   - NQ-Open: `./config/data/nq_open.yaml`
   - HotpotQA: `./config/data/hotpotqa.yaml`

2. **Run a quick PPO smoke test (20 samples)**

```bash
python simulator.py \
  --config ./config/rag.yaml \
  --config ./config/llm/llama32_3b.yaml \
  --config ./config/data/mmlu.yaml \
  --rag.network_type DRAG \
  --rag.search_algorithm PPO \
  --rag.test_mode true \
  --rag.test_num_samples 20 \
  --rag.ppo_train_episodes 20
```

3. **Run a fuller PPO experiment**

```bash
python simulator.py \
  --config ./config/rag.yaml \
  --config ./config/llm/llama32_3b.yaml \
  --config ./config/data/mmlu.yaml \
  --rag.network_type DRAG \
  --rag.search_algorithm PPO \
  --rag.test_mode false \
  --rag.ppo_train_episodes 200
```

4. **Compare PPO vs baselines (same setup)**

Run the same command while switching:
- `--rag.search_algorithm TARW`
- `--rag.search_algorithm RW`
- `--rag.search_algorithm FL`
- `--rag.search_algorithm PPO`
- `--rag.search_algorithm GRPO`

Keep all other arguments (dataset, seed, TTL, number of peers, confidence threshold)
the same for fair comparison.

5. **Where training/testing outputs are saved**

Each run writes an experiment folder with:
- `config.yaml`: effective merged config for reproducibility
- `test_cases.csv`: per-query outputs and route stats
- `metrics.csv`: aggregated metrics during/after evaluation

Tip: use `--rag.random_seed` to keep the graph/data order fixed while comparing PPO
to TARW/RW/FL.

For datasets without explicit topic labels (e.g., NQ-Open / HotpotQA), this project supports
`topic_path: __const__:<topic_name>` to assign a constant topic for DRAG partitioning.
If `answer_path` points to a list field, the simulator will automatically take the first answer.

GRPO uses group-relative advantages and can be enabled with `--rag.search_algorithm GRPO` plus `--rag.grpo_*` hyperparameters.

If PPO underperforms TARW, start tuning these reward-shaping knobs:
- `--rag.ppo_progress_weight` (encourage moving toward higher-relevance neighbors)
- `--rag.ppo_topic_match_bonus` (bias toward topic-consistent hops)
- `--rag.ppo_revisit_penalty` (discourage cycling back to visited peers)
- `--rag.ppo_hop_progressive_penalty` (increase hop penalty at later steps)
- `--rag.ppo_early_hit_bonus` (reward early successful hits to reduce avg_num_hops)

## DEBUG
