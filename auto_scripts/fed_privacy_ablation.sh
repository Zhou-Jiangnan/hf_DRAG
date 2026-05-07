#!/usr/bin/env bash
set -euo pipefail

# Ablation runner for federated privacy PPO.
# Focuses on DP noise / client fraction / local epochs / malicious ratio placeholders.

BASE=(
  python simulator.py
  --config ./config/rag.yaml
  --config ./config/llm/llama32_3b.yaml
  --config ./config/data/mmlu.yaml
  --rag.network_type DRAG
  --rag.search_algorithm PPO
  --rag.test_mode true
  --rag.test_num_samples 80
  --rag.enable_federated_privacy true
  --rag.fed_rounds 20
  --rag.fed_privacy_attack_eval true
  --rag.fed_mia_holdout_ratio 0.2
  --rag.fed_mia_max_samples 80
  --rag.fed_dp_mechanism dp_fedavg
  --rag.fed_dp_clip_norm 1.0
)

# A1: noise ablation
for SIGMA in 0.0 0.3 0.6 1.0 1.4; do
  "${BASE[@]}" \
    --rag.fed_local_epochs 1 \
    --rag.fed_client_fraction 0.5 \
    --rag.fed_dp_noise_multiplier "${SIGMA}"
done

# A2: client fraction ablation
for FRAC in 0.2 0.5 1.0; do
  "${BASE[@]}" \
    --rag.fed_local_epochs 1 \
    --rag.fed_client_fraction "${FRAC}" \
    --rag.fed_dp_noise_multiplier 0.6
done

# A3: local epochs ablation
for EPOCH in 1 2 4; do
  "${BASE[@]}" \
    --rag.fed_local_epochs "${EPOCH}" \
    --rag.fed_client_fraction 0.5 \
    --rag.fed_dp_noise_multiplier 0.6
done
