import neat
import os
import pickle
import time
import numpy as np
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from src.wrapper import apply_wrappers


ENV_NAME = "SuperMarioBros-1-1-v0"

"""
Runs a trained genome, allowing it to play mario without any evolutionary pressure
"""
def run_trained_genome(config_path, genome_path):
    # load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # load the genome, rebuild the NN
    try:
        with open(genome_path, "rb") as f:
            trained_genome = pickle.load(f)
    except:
        print("Unable to load genome correctly! Check path or file type (should be plk!)")
        exit()
    net = neat.nn.FeedForwardNetwork.create(trained_genome, config)

    # Initialize variables and environment
    env = gym_super_mario_bros.make(ENV_NAME)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    state = env.reset()
    done = False

    # main gameplay loop, environment state is input into NN, which (deterministically) calculates action to apply to mario
    while not done:
        inputs = np.array(state).flatten()
        outputs = net.activate(inputs)
        action = int(np.argmax(outputs))

        state, reward, done, info = env.step(action)

        env.render() # render the environment
        time.sleep(0.02) # slow down playback so its visible

    env.close()