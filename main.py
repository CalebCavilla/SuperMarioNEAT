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

# utility imports
import neat
import numpy as np

# Set up the environment
ENV_NAME = "SuperMarioBros-1-1-v0"
def env_setup(evn_name):
    env = gym_super_mario_bros.make(evn_name)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    return env

NUM_OF_EPISODES = 5

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        env = env_setup(ENV_NAME)

        state = env.reset()
        done = False
        total_reward = 0
        max_x_pos = 0

        while not done:
            inputs = np.array(state).flatten()
            outputs = network.activate(inputs)
            action = int(np.argmax(outputs))

            state, reward, done, info = env.step(action)

            total_reward += reward
            x_pos = info.get("x_pos", 0)
            if (x_pos > max_x_pos):
                max_x_pos = x_pos
        
        genome.fitness = max_x_pos + total_reward
        env.close()