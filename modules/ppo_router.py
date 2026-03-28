from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    state_dim: int = 7
    candidate_dim: int = 4
    hidden_dim: int = 64
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    max_grad_norm: float = 0.5
    max_candidates: int = 8


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

    def forward(
        self,
        state_tensor: torch.Tensor,
        candidate_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_tensor: [B, state_dim]
            candidate_tensor: [B, A, candidate_dim]

        Returns:
            logits: [B, A]
            values: [B]
        """
        encoded_state = self.state_encoder(state_tensor)  # [B, H]
        batch_size, action_size, _ = candidate_tensor.shape
        repeated_state = encoded_state.unsqueeze(1).repeat(1, action_size, 1)
        action_input = torch.cat([repeated_state, candidate_tensor], dim=-1)
        logits = self.action_head(action_input).squeeze(-1)
        values = self.value_head(encoded_state).squeeze(-1)
        return logits, values


class PPORouter:
    def __init__(self, cfg: PPOConfig, device: Optional[str] = "auto"):
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
        """
        Args:
            state: [state_dim]
            candidates: [A, candidate_dim]
            mask: [A] bool
        """
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

    def update(self, transitions: List[dict]):
        if not transitions:
            return

        states = torch.stack([t["state"] for t in transitions]).to(self.device)
        candidates = torch.stack([t["candidates"] for t in transitions]).to(self.device)
        masks = torch.stack([t["mask"] for t in transitions]).to(self.device)
        actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t["log_prob"] for t in transitions], dtype=torch.float, device=self.device)
        rewards = torch.tensor([t["reward"] for t in transitions], dtype=torch.float, device=self.device)
        dones = torch.tensor([t["done"] for t in transitions], dtype=torch.float, device=self.device)
        old_values = torch.tensor([t["value"] for t in transitions], dtype=torch.float, device=self.device)

        advantages = torch.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(transitions))):
            delta = rewards[t] + self.cfg.gamma * next_value * (1.0 - dones[t]) - old_values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
            next_value = old_values[t]
        returns = advantages + old_values

        adv_std = advantages.std(unbiased=False)
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

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
