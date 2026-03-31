"""Custom REINFORCE training agent.

Implements the REINFORCE (Monte Carlo Policy Gradient) algorithm with
optional baseline subtraction for variance reduction.
"""

import numpy as np
import torch
import torch.optim as optim
import csv
import os

from agents.reinforce_policy import REINFORCEPolicy


class REINFORCEAgent:
    """REINFORCE algorithm with multi-head policy for MultiDiscrete actions.

    Supports three baseline modes:
        - 'none': No baseline (vanilla REINFORCE)
        - 'mean': Subtract mean of returns per episode
        - 'running': Subtract exponential moving average of returns
    """

    def __init__(
        self,
        env,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_sizes: list = None,
        baseline: str = "mean",
        seed: int = 42,
    ):
        self.env = env
        self.gamma = gamma
        self.baseline = baseline

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = env.observation_space.shape[0]
        action_nvec = list(env.action_space.nvec)

        self.policy = REINFORCEPolicy(
            obs_dim=obs_dim,
            action_nvec=action_nvec,
            hidden_sizes=hidden_sizes or [128, 64],
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Running baseline state
        self.running_return = 0.0
        self.running_count = 0

    def collect_episode(self) -> tuple:
        """Run one full episode, collecting log_probs and rewards.

        Returns:
            (log_probs, rewards, episode_length, info) where:
                log_probs: list of log_prob tensors
                rewards: list of floats
                episode_length: int
                info: dict from last step
        """
        obs, info = self.env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs)
            action, log_prob = self.policy.get_action(obs_tensor)
            log_probs.append(log_prob)

            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            done = terminated or truncated

        return log_probs, rewards, len(rewards), info

    def compute_returns(self, rewards: list) -> torch.Tensor:
        """Compute discounted returns with optional baseline subtraction.

        Args:
            rewards: List of rewards from an episode.

        Returns:
            Tensor of (optionally normalized) discounted returns.
        """
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        if self.baseline == "mean":
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            else:
                returns = returns - returns.mean()

        elif self.baseline == "running":
            self.running_count += 1
            alpha = 0.05
            self.running_return = (
                (1 - alpha) * self.running_return + alpha * returns.mean().item()
            )
            returns = returns - self.running_return

        return returns

    def update(self, log_probs: list, returns: torch.Tensor):
        """Perform a single policy gradient update.

        REINFORCE loss: -sum(log_pi(a|s) * G_t) / T
        """
        policy_loss = torch.tensor(0.0)
        for log_prob, G in zip(log_probs, returns):
            policy_loss = policy_loss + (-log_prob * G)

        policy_loss = policy_loss / len(log_probs)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()

    def train(
        self,
        num_episodes: int = 5000,
        log_interval: int = 100,
        save_dir: str = None,
    ) -> list:
        """Train the REINFORCE agent.

        Args:
            num_episodes: Number of episodes to train for.
            log_interval: Print stats every N episodes.
            save_dir: Directory to save model and logs. If None, don't save.

        Returns:
            List of (episode_reward, episode_length) tuples.
        """
        history = []
        reward_window = []

        for episode in range(1, num_episodes + 1):
            log_probs, rewards, ep_len, info = self.collect_episode()
            ep_reward = sum(rewards)
            returns = self.compute_returns(rewards)
            loss = self.update(log_probs, returns)

            history.append((ep_reward, ep_len))
            reward_window.append(ep_reward)
            if len(reward_window) > 100:
                reward_window.pop(0)

            if episode % log_interval == 0:
                avg_reward = np.mean(reward_window)
                avg_len = np.mean([h[1] for h in history[-100:]])
                print(
                    f"Episode {episode:5d} | "
                    f"Avg Reward: {avg_reward:10.2f} | "
                    f"Avg Length: {avg_len:8.1f} | "
                    f"Loss: {loss:10.4f}"
                )

        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Save policy
            torch.save(self.policy.state_dict(), os.path.join(save_dir, "policy.pt"))

            # Save reward history
            csv_path = os.path.join(save_dir, "rewards.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "episode_length"])
                for i, (r, l) in enumerate(history, 1):
                    writer.writerow([i, r, l])

            print(f"\nModel saved to {save_dir}/policy.pt")
            print(f"Rewards saved to {csv_path}")

        return history
