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

# prepare images
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
model = PPO("CnnPolicy", env, verbose=1, device="cuda")
model = model.load("weights/checkpoint",env=env,print_system_info=True)
print(model.policy.state_dict)
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, dones, info = vec_env.step(action)
    vec_env.render()
