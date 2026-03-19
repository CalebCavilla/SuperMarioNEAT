# OS imports
import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# gym environment imports
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

# wrapper import
from wrapper import apply_wrappers

# Set up the environment
ENV_NAME = "SuperMarioBros-1-1-v0"
env = gym_super_mario_bros.make(ENV_NAME)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

NUM_OF_EPISODES = 5

for episode in range(NUM_OF_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    print(f"Episode {episode + 1} started")
    print("State shape:", state.shape)

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    print("Episode finished!")
    print("Total reward:", total_reward)

env.close()