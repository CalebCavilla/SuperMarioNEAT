import numpy as np
import matplotlib.pyplot as plt
from gym import Env, spaces, Wrapper, ObservationWrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

class SkipFrame(Wrapper):
    """
    Class to combine 4 steps into one, this is done as there is much redundancy between frames,
    Thus we can reduce redundancy and computation power by combining steps.
    """
    # Override constructor to take the number of frames to skip as a parameter
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    # Over-ride the step function
    def step(self, action):
        total_reward = 0.0
        done = False
        # Whenever step is called, we take 'skip' number of steps. This consolidates 4 steps into one
        for _ in range(self.skip):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward # sum up the rewards from each step
            if done:
                break 
        return next_state, total_reward, done, info 


class CropObservation(ObservationWrapper):
    """
    Class to to crop the frame around mario. There is allot of useless information such as the sky, background,
    HUD that is not relevant to making decisions. Thus we only focus on the most important play area.
    """
    # Override constructor, default size is 240x256. 
    def __init__(self, env, top=80, bottom=224, left=16, right=176):
        super().__init__(env)
        self.top = top # 80 cuts off sky/HUD
        self.bottom = bottom # 224 cuts off ground blocks
        self.left = left # 16 cuts off a little redundant space behind mario
        self.right = right # 276 cuts off stuff in distance in front of mario

        cropped_height = bottom - top
        cropped_width = right - left
        cropped_channels = env.observation_space.shape[2]

        self.observation_space = spaces.Box(low=0, high=255, shape=(cropped_height, cropped_width, cropped_channels))

    def observation(self, observation):
        return observation[self.top:self.bottom, self.left:self.right]

def apply_wrappers(env):
    """
    Helper function to apply wrappers one by one, preprocessing the frames
    Has the effect of combining 16 frames into one state
    """
    env = SkipFrame(env, skip=4) # number of frames to apply one action to
    env = CropObservation(env, top=80, bottom=224, left=32, right=192) # crops the frame around mario, reducing computational load
    env = GrayScaleObservation(env) # changes frame from rgb channels to just one, grey scale. Reduces computational load
    env = ResizeObservation(env, shape=20) # changes the dimensions of a frame to 20x20, reduces computational load
    env = FrameStack(env, num_stack=2, lz4_compress=True) # stack frames to capture motion
    return env