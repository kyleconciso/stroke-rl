import colormath.color_conversions
import colormath.color_diff
import colormath.color_objects
import gymnasium as gym
from gymnasium import spaces
import colormath
from colormath.color_objects import sRGBColor
import numpy as np
import random
import cv2

def canvas_delta(c1, c2):
    res = 0
    for y in range(256):
        for x in range(256):
            color1 = sRGBColor(c1[y][x][0],c1[y][x][1],c1[y][x][2])
            color2 = sRGBColor(c2[y][x][0],c2[y][x][1],c2[y][x][2])
            delta = colormath.color_diff.delta_e_cie2000(
                colormath.color_conversions.convert_color(color1,colormath.color_objects.LabColor),
                colormath.color_conversions.convert_color(color2,colormath.color_objects.LabColor)
            )
            res+=delta

    return res


class StrokeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, max_steps=50, images=None):
        super().__init__()

        self.observation_space = spaces.Dict({
            "target_canvas": spaces.Box(0, 256, (256, 256, 3), dtype=float),
            "agent_canvas": spaces.Box(0, 256, (256, 256, 3), dtype=float)
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

        pt1 = tuple(map(int, action['line_start']))
        pt2 = tuple(map(int, action['line_end']))
        thickness = int(action['line_thickness'])
        color = tuple(map(int, action['color']))
        print(thickness)
        if thickness > 0:
            cv2.line(self._agent_canvas, pt1*256, pt2*256, color*256, thickness*256)

        if not self._prev_delta:
            self._prev_delta = 0
            
        new_delta = canvas_delta(self._target_canvas, self._agent_canvas)
        
        reward = 1-(new_delta-self._prev_delta)
        observation = self._get_obs()
        info = self._get_info()

        self._prev_delta = new_delta

        truncated = False
        self._step += 1
        if self._step >= self.max_steps:
            truncated = True

        return observation, reward, False, truncated, info
    
    def render(self):
        cv2.imshow(self._agent_canvas)
