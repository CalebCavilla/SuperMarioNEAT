import numpy as np
from gym import Env, Wrapper
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
    
def apply_wrappers(env):
    """
    Helper function to apply wrappers one by one, preprocessing the frames
    Has the effect of combining 16 frames into one state
    """
    env = SkipFrame(env, skip=4) # number of frames to apply one action to
    env = ResizeObservation(env, shape=16) # changes the dimensions of a frame to 16x16, reduces computational load
    env = GrayScaleObservation(env) # changes frame from rgb channels to just one, grey scale. Reduces computational load
    return env