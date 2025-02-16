import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import cv2
from colour import delta_E
from colour.models import RGB_to_XYZ, XYZ_to_Lab
from colour.colorimetry import CCS_ILLUMINANTS

def sRGB_to_Lab(srgb_image):
    srgb = srgb_image.astype(np.float32) / 255.0
    linear_rgb = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    xyz = RGB_to_XYZ(linear_rgb, 'sRGB')
    illuminant_D65 = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    lab = XYZ_to_Lab(xyz, illuminant_D65)
    return lab

def canvas_delta(c1, c2):
    lab1 = sRGB_to_Lab(c1)
    lab2 = sRGB_to_Lab(c2)
    delta = delta_E(lab1.reshape(-1, 3), lab2.reshape(-1, 3))
    return (np.sum(delta)/(100*255*255))


class StrokeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, max_steps=50, images=None, render_mode=None):
        super().__init__()

        self.observation_space = spaces.Dict({
            "target_canvas": spaces.Box(0, 255, (256, 256, 3), dtype=float),
            "agent_canvas": spaces.Box(0, 255, (256, 256, 3), dtype=float)
        })

        self.action_space = spaces.Box(0,1,(8,),dtype=float)

        self.max_steps = max_steps

        self.images = images

        self._prev_delta = None

    def _get_obs(self):
        return {"target_canvas":self._target_canvas, "agent_canvas":self._agent_canvas}
    
    def _get_info(self):
        return {"delta": canvas_delta(self._target_canvas, self._agent_canvas)}

    def reset(self, seed=None, target_img=None, options=None):
        super().reset(seed=seed)

        self._agent_canvas = np.ones((256, 256, 3), dtype=np.uint8)
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

        pt1 = tuple(map(int, action['line_start']*256))
        pt2 = tuple(map(int, action['line_end']*256))
        thickness = int(action['line_thickness']*256)
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
