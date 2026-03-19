import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Create environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")

# Simplify action space
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Reset environment
state = env.reset()

done = False
total_reward = 0

while True:
    # Random action (just testing environment)
    action = env.action_space.sample()

    state, reward, done, info = env.step(action)
    total_reward += reward

    # Render the game
    env.render()

    if done:
        print("Episode finished!")
        print("Total reward:", total_reward)
        total_reward = 0
        state = env.reset()

env.close()