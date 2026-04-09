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
from src.wrapper import apply_wrappers, printObservation

# utility imports
import neat
from neat.parallel import ParallelEvaluator
import random
import numpy as np
import pickle
from src.bestGenomeReporter import BestGenomeReporter
from neat import Checkpointer


# Set up the environment
ENV_NAME = "SuperMarioBros-1-1-v0"
def env_setup(evn_name):

    env = gym_super_mario_bros.make(evn_name)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    return env

def eval_genome(genome, config):

    network = neat.nn.FeedForwardNetwork.create(genome, config)
    env = env_setup(ENV_NAME)

    state = env.reset()
    #print(np.array(state).shape)
    done = False

    max_x_pos = 40
    prev_x_pos = 0
    idle_timer = 100
    fitness = 0.0
    step = 0
    checkpoints = {200:50, 400:75, 600:100, 800:150, 1200:250, 1600:400, 2000:600, 2500:1000, 3000:1500}
    awarded_checkpoints = set()

    while not done:
        inputs = np.array(state).flatten()
        outputs = network.activate(inputs)
        action = int(np.argmax(outputs))

        state, reward, done, info = env.step(action)
        step += 1

        # if step % 100 == 0:
        #     printObservation(state)

        x_pos = info.get("x_pos")

        # reward forward movement
        progress = x_pos - prev_x_pos
        if progress > 0:
            fitness += progress * 1.5 # reward for moving forward
        else:
            fitness += progress * 0.5 # slightly less reward for moving backwards
        prev_x_pos = x_pos

        if (x_pos > max_x_pos):
            max_x_pos = x_pos
            idle_timer = 100
        else:
            idle_timer -= 1

        if idle_timer <= 0:
            fitness -= 50
            break

        # time penalty to encourage fast movement
        fitness -= 0.2
    
    env.close()

    fitness += max_x_pos * 0.5

    # # Penalize dying before finishing
    # if not info.get("flag_get", False):
    #     fitness -= 100

    for cp, bonus in checkpoints.items():
        if x_pos >= cp and cp not in awarded_checkpoints:
            fitness += bonus
            awarded_checkpoints.add(cp)


    return fitness

def train_neat(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    #population = neat.Population(config)
    population = neat.Checkpointer.restore_checkpoint("neat_checkpoints/neat-checkpoint-521")

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(BestGenomeReporter())

    os.makedirs("neat_checkpoints", exist_ok=True)
    population.add_reporter(Checkpointer(
        generation_interval=100,
        filename_prefix="neat_checkpoints/neat-checkpoint-"
    ))


    num_workers = max(1, os.cpu_count() - 1)

    with ParallelEvaluator(num_workers, eval_genome) as pe:
        winner = population.run(pe.evaluate, 5000)

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome saved!")

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    train_neat(config_path)