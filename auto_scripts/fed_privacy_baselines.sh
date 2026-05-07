#!/usr/bin/env bash
set -euo pipefail

# Quick matrix runner for federated-privacy PPO baselines.
# Usage:
#   bash auto_scripts/fed_privacy_baselines.sh

COMMON=(
  python simulator.py
  --config ./config/rag.yaml
  --config ./config/llm/llama32_3b.yaml
  --config ./config/data/mmlu.yaml
  --rag.network_type DRAG
  --rag.search_algorithm PPO
  --rag.test_mode true
  --rag.test_num_samples 50
  --rag.enable_federated_privacy true
  --rag.fed_rounds 20
  --rag.fed_local_epochs 1
  --rag.fed_client_fraction 0.5
  --rag.fed_privacy_attack_eval true
  --rag.fed_mia_holdout_ratio 0.2
  --rag.fed_mia_max_samples 50
)

# F1: FedAvg-PPO (no DP)
"${COMMON[@]}" \
  --rag.fed_dp_mechanism none \
  --rag.fed_dp_clip_norm 1.0 \
  --rag.fed_dp_noise_multiplier 0.0

# F2: Clip only
"${COMMON[@]}" \
  --rag.fed_dp_mechanism dp_fedavg \
  --rag.fed_dp_clip_norm 1.0 \
  --rag.fed_dp_noise_multiplier 0.0

# F3/F4/F5: DP noise sweep
for SIGMA in 0.3 0.6 1.0; do
  "${COMMON[@]}" \
    --rag.fed_dp_mechanism dp_fedavg \
    --rag.fed_dp_clip_norm 1.0 \
    --rag.fed_dp_noise_multiplier "${SIGMA}"
done
