from dataclasses import dataclass
import math
import random
from typing import Dict, List

from loguru import logger
import torch

from modules.data_types import Datapoint
from modules.ppo_router import PPOConfig, PPORouter
from modules.rag_network import DRAGNetwork


@dataclass
class FederatedPrivacyConfig:
    rounds: int = 20
    local_epochs: int = 1
    client_fraction: float = 1.0
    dp_mechanism: str = "none"  # none, dp_sgd, dp_fedavg, adaptive_dp
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.8
    secure_agg: bool = False


class FederatedPPORunner:
    """Federated training runner for DRAG PPO router.

    Simulation design:
    - client == peer node
    - each client trains only from its local-topic datapoints
    - local updates are aggregated by FedAvg
    - optional privacy: update clipping + Gaussian noise
    """

    def __init__(self, cfg: FederatedPrivacyConfig):
        self.cfg = cfg

    @staticmethod
    def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in state_dict.items()}

    @staticmethod
    def _subtract_state_dict(
        state_a: Dict[str, torch.Tensor],
        state_b: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {k: state_a[k] - state_b[k] for k in state_a.keys()}

    @staticmethod
    def _add_state_dict(
        state_a: Dict[str, torch.Tensor],
        state_b: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {k: state_a[k] + state_b[k] for k in state_a.keys()}

    @staticmethod
    def _scale_state_dict(
        state: Dict[str, torch.Tensor],
        scale: float,
    ) -> Dict[str, torch.Tensor]:
        return {k: state[k] * scale for k in state.keys()}

    @staticmethod
    def _zeros_like_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: torch.zeros_like(v) for k, v in state.items()}

    @staticmethod
    def _global_l2_norm(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
        sq_sum = torch.tensor(0.0)
        for v in delta.values():
            sq_sum = sq_sum + torch.sum(v.float() ** 2)
        return torch.sqrt(sq_sum)

    def _clip_update(self, delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.cfg.dp_clip_norm <= 0:
            return delta
        norm = self._global_l2_norm(delta)
        clip_coef = min(1.0, self.cfg.dp_clip_norm / (norm.item() + 1e-12))
        if clip_coef >= 1.0:
            return delta
        return self._scale_state_dict(delta, clip_coef)

    def _noise_scale(self) -> float:
        if self.cfg.dp_mechanism in ("dp_sgd", "dp_fedavg", "adaptive_dp"):
            return max(0.0, self.cfg.dp_noise_multiplier) * max(0.0, self.cfg.dp_clip_norm)
        return 0.0

    def _add_dp_noise(self, delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        noise_std = self._noise_scale()
        if noise_std <= 0:
            return delta
        noised = {}
        for k, v in delta.items():
            noised[k] = v + torch.randn_like(v) * noise_std
        return noised

    def _sample_clients(self, all_client_ids: List[int]) -> List[int]:
        if len(all_client_ids) == 0:
            return []
        fraction = min(max(self.cfg.client_fraction, 0.0), 1.0)
        sample_size = max(1, math.ceil(len(all_client_ids) * fraction))
        sample_size = min(sample_size, len(all_client_ids))
        return random.sample(all_client_ids, k=sample_size)

    def _build_client_local_data(
        self,
        rag_net: DRAGNetwork,
        data_points: List[Datapoint],
    ) -> Dict[int, List[Datapoint]]:
        by_topic: Dict[str, List[Datapoint]] = {}
        for dp in data_points:
            by_topic.setdefault(dp.topic, []).append(dp)

        client_data: Dict[int, List[Datapoint]] = {peer_id: [] for peer_id in range(rag_net.num_peers)}
        for peer_id, topics in rag_net.peer_topics.items():
            local = []
            for topic in topics:
                local.extend(by_topic.get(topic, []))
            client_data[peer_id] = local
        return client_data

    def train(
        self,
        rag_net: DRAGNetwork,
        global_router: PPORouter,
        ppo_cfg: PPOConfig,
        data_points: List[Datapoint],
        max_ttl: int,
        query_confidence_threshold: float,
        reward_hit: float,
        reward_miss: float,
        message_penalty: float,
        hop_penalty: float,
        relevance_weight: float,
        progress_weight: float,
        topic_match_bonus: float,
        revisit_penalty: float,
        hop_progressive_penalty: float,
        early_hit_bonus: float,
    ):
        client_local_data = self._build_client_local_data(rag_net, data_points)
        available_clients = [cid for cid, dps in client_local_data.items() if len(dps) > 0]

        if len(available_clients) == 0:
            logger.warning("Skip federated PPO training since no client has local datapoints")
            return

        for rnd in range(self.cfg.rounds):
            sampled_clients = self._sample_clients(available_clients)
            global_state = self._clone_state_dict(global_router.model.state_dict())
            weighted_delta_sum = self._zeros_like_state_dict(global_state)
            total_weight = 0.0

            for client_id in sampled_clients:
                local_data = client_local_data.get(client_id, [])
                if len(local_data) == 0:
                    continue

                local_router = PPORouter(ppo_cfg, device=global_router.device)
                local_router.model.load_state_dict(global_state)

                rag_net.ppo_train(
                    router=local_router,
                    data_points=local_data,
                    num_episodes=max(1, self.cfg.local_epochs),
                    max_ttl=max_ttl,
                    query_confidence_threshold=query_confidence_threshold,
                    reward_hit=reward_hit,
                    reward_miss=reward_miss,
                    message_penalty=message_penalty,
                    hop_penalty=hop_penalty,
                    relevance_weight=relevance_weight,
                    progress_weight=progress_weight,
                    topic_match_bonus=topic_match_bonus,
                    revisit_penalty=revisit_penalty,
                    hop_progressive_penalty=hop_progressive_penalty,
                    early_hit_bonus=early_hit_bonus,
                    query_peer_ids=[client_id],
                )

                local_state = self._clone_state_dict(local_router.model.state_dict())
                delta = self._subtract_state_dict(local_state, global_state)
                if self.cfg.dp_mechanism != "none":
                    delta = self._clip_update(delta)

                weight = float(len(local_data))
                total_weight += weight
                weighted_delta_sum = self._add_state_dict(weighted_delta_sum, self._scale_state_dict(delta, weight))

            if total_weight <= 0:
                logger.warning(f"Federated round {rnd + 1}/{self.cfg.rounds}: no valid client updates")
                continue

            avg_delta = self._scale_state_dict(weighted_delta_sum, 1.0 / total_weight)
            if self.cfg.dp_mechanism != "none":
                avg_delta = self._add_dp_noise(avg_delta)

            updated_global_state = self._add_state_dict(global_state, avg_delta)
            global_router.model.load_state_dict(updated_global_state)

            logger.info(
                f"Federated PPO round {rnd + 1}/{self.cfg.rounds} done, "
                f"clients={len(sampled_clients)}, secure_agg={self.cfg.secure_agg}, dp={self.cfg.dp_mechanism}"
            )
