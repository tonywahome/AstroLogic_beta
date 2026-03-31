"""PyTorch policy network for REINFORCE algorithm.

Multi-head architecture: shared backbone with 6 independent Categorical
output heads, one per sub-action dimension of the MultiDiscrete space.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class REINFORCEPolicy(nn.Module):
    """Multi-head policy network for MultiDiscrete action space.

    Architecture:
        Input (23) -> Shared Backbone (hidden layers + ReLU)
                   -> Head 0: Linear -> Categorical (5 actions: thrust)
                   -> Head 1: Linear -> Categorical (3 actions: pitch)
                   -> Head 2: Linear -> Categorical (3 actions: yaw)
                   -> Head 3: Linear -> Categorical (3 actions: roll)
                   -> Head 4: Linear -> Categorical (4 actions: instrument)
                   -> Head 5: Linear -> Categorical (2 actions: comm)
    """

    def __init__(self, obs_dim: int = 23, action_nvec: list = None,
                 hidden_sizes: list = None):
        super().__init__()

        if action_nvec is None:
            action_nvec = [5, 3, 3, 3, 4, 2]
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.action_nvec = action_nvec

        # Build shared backbone
        layers = []
        prev_size = obs_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        self.backbone = nn.Sequential(*layers)

        # Build output heads (one per sub-action)
        self.heads = nn.ModuleList([
            nn.Linear(prev_size, n) for n in action_nvec
        ])

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass returning logits for each head.

        Args:
            obs: Observation tensor of shape (batch, 23) or (23,)

        Returns:
            List of 6 logit tensors, one per sub-action dimension.
        """
        features = self.backbone(obs)
        return [head(features) for head in self.heads]

    def get_action(self, obs: torch.Tensor) -> tuple:
        """Sample an action from the policy and return with log probability.

        Args:
            obs: Single observation tensor of shape (23,)

        Returns:
            (actions, log_prob) where:
                actions: numpy array of shape (6,) with sampled sub-actions
                log_prob: scalar tensor with sum of log probs across heads
        """
        logits_list = self.forward(obs)

        actions = []
        total_log_prob = torch.tensor(0.0)

        for logits in logits_list:
            dist = Categorical(logits=logits)
            action = dist.sample()
            total_log_prob = total_log_prob + dist.log_prob(action)
            actions.append(action.item())

        import numpy as np
        return np.array(actions, dtype=np.int64), total_log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for given obs-action pairs.

        Args:
            obs: Batch of observations (batch, 23)
            actions: Batch of actions (batch, 6)

        Returns:
            Log probabilities tensor of shape (batch,)
        """
        logits_list = self.forward(obs)
        total_log_prob = torch.zeros(obs.shape[0])

        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            total_log_prob = total_log_prob + dist.log_prob(actions[:, i])

        return total_log_prob
