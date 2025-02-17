import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import cv2
from skimage.color import rgb2lab, deltaE_cie76

def canvas_delta(c1, c2):
    lab1 = rgb2lab(c1 / 255.0)
    lab2 = rgb2lab(c2 / 255.0)
    delta = deltaE_cie76(lab1, lab2)
    return np.mean(delta) / 100

class StrokeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, max_steps=256, images=None, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = spaces.Box(0, 255, (128, 128, 6), dtype=np.uint8)
        self.action_space = spaces.Box(0,1,(8,),dtype=float)
        self.max_steps = max_steps
        self.images = images
        self._prev_thickness = None

    def _get_obs(self):
        return np.concatenate((self._target_canvas, self._agent_canvas),axis=2,dtype=np.uint8)
    
    def _get_info(self):
        return {"delta": canvas_delta(self._target_canvas, self._agent_canvas)}

    def reset(self, seed=None, target_img=None, options=None):
        super().reset(seed=seed)

        self._agent_canvas = np.ones((128, 128, 3), dtype=np.uint8)
        self._target_canvas = random.choice(self.images)
        self._prev_delta = canvas_delta(self._target_canvas, self._agent_canvas)

        observation = self._get_obs()
        info = self._get_info()

        self._step = 0

        return observation, info
    
    def step(self, action):
        action = {
            "line_start":action[:2],
            "line_end":action[2:4],
            "line_thickness":action[4:5][0],
            "color":action[5:8]
        }
        pt1 = tuple(map(int, action['line_start']*128))
        pt2 = tuple(map(int, action['line_end']*128))
        thickness = int(action['line_thickness']*24)+1
        color = tuple(map(int, action['color']*255))
        # print(action)
        # print(pt1,pt2,thickness,color)
        self._agent_canvas = cv2.line(self._agent_canvas, pt1, pt2, color, thickness)

        new_delta = canvas_delta(self._target_canvas, self._agent_canvas)
        delta_change = (new_delta-self._prev_delta)

        terminated = False
        reward = 0
        if delta_change > 0:
            reward = (-delta_change*100)
            # terminated = True
        elif delta_change ==0 and self._step>0:
            reward = -100
            terminated = True
        elif delta_change < 0:
            reward = -delta_change*100

        self._prev_delta = new_delta

        # if not self._prev_thickness:
        #     self._prev_thickness = thickness

        # if thickness >= self._prev_thickness:
        #     reward -= 2
        # else:
        #     reward += 2

        self._prev_thickness = thickness

        observation = self._get_obs()
        info = self._get_info()

        truncated = False
        self._step += 1

        # self.render(override=True)

        if self._step >= self.max_steps:
            print(reward)
            truncated = True

        return observation, reward, terminated, truncated, info
    
    def render(self, override=False):
        if self.render_mode == "human" or override:
            # cv2.line(self._agent_canvas, (0,0), (128,128), (255,0,0), 2)
            frame = cv2.addWeighted(self._target_canvas, 0.5, self._agent_canvas, 0.5, 0)
            cv2.imshow("Agent Canvas", cv2.resize(frame, (500,500)))
            cv2.waitKey(1)
