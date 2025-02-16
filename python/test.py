import gymnasium as gym
import gymnasium as gym
from stable_baselines3 import A2C
from stroke.env import StrokeEnv
import os
import cv2

# colormath numpy asscalar patch
import numpy
def patch_asscalar(a):
    return a.item()
setattr(numpy, "asscalar", patch_asscalar)

# prepare images for training
images = []
for fn in os.listdir("data/clean"):
    im = cv2.imread("data/clean/"+fn)
    images.append(im)


# register and train
gym.register(
    id="Stroke-v0",
    entry_point=StrokeEnv
)

env = gym.make("Stroke-v0", images=images, render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()