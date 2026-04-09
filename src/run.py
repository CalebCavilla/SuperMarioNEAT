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


def env_setup(env_name):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    return env


def run_trained_genome(config_path, genome_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    try:
        with open(genome_path, "rb") as f:
            trained_genome = pickle.load(f)
    except:
        print("Unable to load genome correctly! Check path or file type (should be plk!)")
        exit()
    

    net = neat.nn.FeedForwardNetwork.create(trained_genome, config)

    env = gym_super_mario_bros.make(ENV_NAME)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    state = env.reset()
    done = False

    while not done:
        inputs = np.array(state).flatten()
        outputs = net.activate(inputs)
        action = int(np.argmax(outputs))

        state, reward, done, info = env.step(action)

        env.render()
        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    genome_path = os.path.join(local_dir, "best_genomes/gen_2235_best.pkl")
    run_trained_genome(config_path, genome_path)