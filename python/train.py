import gymnasium as gym
from stable_baselines3 import PPO
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
env = gym.make("Stroke-v0", images=images)

model = PPO("CnnPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=500000)
model.save("model")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
