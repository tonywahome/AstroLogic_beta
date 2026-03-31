"""AstroExploration Gymnasium Environment Package."""

from gymnasium.envs.registration import register

register(
    id="AstroExploration-v0",
    entry_point="astro_env.astro_exploration_env:AstroExplorationEnv",
    max_episode_steps=100_000,
)
