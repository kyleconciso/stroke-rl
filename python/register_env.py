import gymnasium as gym
from stroke.env import StrokeEnv

gym.register(
    id="Stroke-v0",
    entry_point=StrokeEnv
)
