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

print(images[1].shape)


# register and train
gym.register(
    id="Stroke-v0",
    entry_point=StrokeEnv
)
env = gym.make("Stroke-v0", images=images)

model = A2C("MultiInputPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=1)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
