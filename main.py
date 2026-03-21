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
import pickle

# Set up the environment
ENV_NAME = "SuperMarioBros-1-1-v0"
def env_setup(evn_name):
    env = gym_super_mario_bros.make(evn_name)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    return env

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        env = env_setup(ENV_NAME)

        state = env.reset()
        done = False
        total_reward = 0
        max_x_pos = 0
        idle_timer = 150

        while not done:
            inputs = np.array(state).flatten()
            outputs = network.activate(inputs)
            action = int(np.argmax(outputs))

            state, reward, done, info = env.step(action)

            total_reward += reward
            x_pos = info.get("x_pos", 0)
            if (x_pos > max_x_pos):
                max_x_pos = x_pos
                idle_timer = 150
            else:
                idle_timer -= 1

            if idle_timer <= 0:
                break
            
            env.render()
        
        genome.fitness = max_x_pos
        env.close()

def train_neat(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 15)

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome saved!")

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    train_neat(config_path)