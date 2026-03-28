from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class GRPOConfig:
    state_dim: int = 7
    candidate_dim: int = 4
    hidden_dim: int = 64
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    gamma: float = 0.99
    update_epochs: int = 4
    max_grad_norm: float = 0.5
    max_candidates: int = 8
    group_size: int = 4


class _PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, candidate_dim: int, hidden_dim: int):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + candidate_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_tensor: torch.Tensor, candidate_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_state = self.state_encoder(state_tensor)
        _, action_size, _ = candidate_tensor.shape
        repeated_state = encoded_state.unsqueeze(1).repeat(1, action_size, 1)
        action_input = torch.cat([repeated_state, candidate_tensor], dim=-1)
        logits = self.action_head(action_input).squeeze(-1)
        values = self.value_head(encoded_state).squeeze(-1)
        return logits, values


class GRPORouter:
    def __init__(self, cfg: GRPOConfig, device: Optional[str] = "auto"):
        self.cfg = cfg
        if device in (None, "auto"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.model = _PolicyValueNet(cfg.state_dim, cfg.candidate_dim, cfg.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

    def _masked_categorical(self, logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        return Categorical(logits=masked_logits)

    def act(
        self,
        state: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, float]:
        with torch.no_grad():
            logits, value = self.model(state.unsqueeze(0), candidates.unsqueeze(0))
            dist = self._masked_categorical(logits.squeeze(0), mask)
            if deterministic:
                action = torch.argmax(dist.probs).item()
            else:
                action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
            entropy = dist.entropy().item()
            return action, log_prob, value.item(), entropy

    def _discounted_returns(self, rewards: List[float]) -> List[float]:
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.cfg.gamma * running
            returns.append(running)
        return list(reversed(returns))

    def update(self, grouped_episodes: List[List[dict]]):
        if not grouped_episodes:
            return

        episodes = [ep for ep in grouped_episodes if len(ep) > 0]
        if not episodes:
            return

        episode_scores = torch.tensor([sum(step["reward"] for step in ep) for ep in episodes], dtype=torch.float)
        score_std = episode_scores.std(unbiased=False)
        if score_std > 1e-8:
            group_adv = (episode_scores - episode_scores.mean()) / (score_std + 1e-8)
        else:
            group_adv = episode_scores - episode_scores.mean()

        states = []
        candidates = []
        masks = []
        actions = []
        old_log_probs = []
        advantages = []
        returns = []

        for ep_idx, episode in enumerate(episodes):
            ep_returns = self._discounted_returns([step["reward"] for step in episode])
            for step_idx, step in enumerate(episode):
                states.append(step["state"])
                candidates.append(step["candidates"])
                masks.append(step["mask"])
                actions.append(step["action"])
                old_log_probs.append(step["log_prob"])
                advantages.append(group_adv[ep_idx].item())
                returns.append(ep_returns[step_idx])

        states = torch.stack(states).to(self.device)
        candidates = torch.stack(candidates).to(self.device)
        masks = torch.stack(masks).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float, device=self.device)

        for _ in range(self.cfg.update_epochs):
            logits, values = self.model(states, candidates)
            dist = self._masked_categorical(logits, masks)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = ((returns - values) ** 2).mean()
            loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
