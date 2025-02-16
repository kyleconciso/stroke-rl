import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import cv2
from skimage.color import rgb2lab, deltaE_cie76

def canvas_delta(c1, c2):
    lab1 = rgb2lab(c1.astype(np.float32) / 255.0)
    lab2 = rgb2lab(c2.astype(np.float32) / 255.0)
    delta = deltaE_cie76(lab1, lab2)
    return np.mean(delta) / 100

class StrokeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, max_steps=50, images=None, render_mode=None):
        super().__init__()

        self.observation_space = spaces.Box(0, 255, (128, 128, 6), dtype=np.uint8)

        self.action_space = spaces.Box(0,1,(8,),dtype=float)

        self.max_steps = max_steps

        self.images = images

        self._prev_delta = None

    def _get_obs(self):
        return np.concatenate((self._target_canvas, self._agent_canvas),axis=2,dtype=np.uint8)
    
    def _get_info(self):
        return {"delta": canvas_delta(self._target_canvas, self._agent_canvas)}

    def reset(self, seed=None, target_img=None, options=None):
        super().reset(seed=seed)

        self._agent_canvas = np.ones((128, 128, 3), dtype=np.uint8)
        self._target_canvas = random.choice(self.images)

        observation = self._get_obs()
        info = self._get_info()

        self._step = 0

        return observation, info
    
    def step(self, action):
        action = {
            "line_start":action[:2],
            "line_end":action[2:4],
            "line_thickness":action[4:5],
            "color":action[5:8]
        }

        pt1 = tuple(map(int, action['line_start']*128))
        pt2 = tuple(map(int, action['line_end']*128))
        thickness = int(action['line_thickness']*128)
        color = tuple(map(int, action['color']*255))
        if thickness > 0:
            cv2.line(self._agent_canvas, pt1, pt2, color, thickness)

        new_delta = canvas_delta(self._target_canvas, self._agent_canvas)

        if not self._prev_delta:
            self._prev_delta = 1
                
        reward = -(new_delta-self._prev_delta)
        observation = self._get_obs()
        info = self._get_info()

        self._prev_delta = new_delta

        truncated = False
        self._step += 1
        if self._step >= self.max_steps:
            print(reward)
            truncated = True
            self._step = 0

        return observation, reward, False, truncated, info
    
    def render(self, mode="human"):
        if mode == "human":
            cv2.imshow("Agent Canvas", self._agent_canvas)
            cv2.waitKey(1)
