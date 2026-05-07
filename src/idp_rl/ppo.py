from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def make_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_size))
        layers.append(nn.Tanh())
        last_dim = hidden_size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.actor_mean = make_mlp(obs_dim, hidden_sizes, action_dim)
        self.critic = make_mlp(obs_dim, hidden_sizes, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(obs)
        std = self.log_std.exp().expand_as(mean)
        distribution = Normal(mean, std)
        if action is None:
            action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(-1)
        entropy = distribution.entropy().sum(-1)
        value = self.get_value(obs)
        return action, log_prob, entropy, value


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
