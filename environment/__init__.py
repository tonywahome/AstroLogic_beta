"""AstroLogic environment package — registers the Gymnasium env on import."""

from gymnasium.envs.registration import register

register(
    id="AstroExploration-v0",
    entry_point="environment.custom_env:AstroExplorationEnv",
    max_episode_steps=10_000,
)
