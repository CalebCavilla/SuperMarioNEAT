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
from src.wrapper import apply_wrappers

# utility imports
import neat
from neat.parallel import ParallelEvaluator
import random
import numpy as np
import pickle
from src.bestGenomeReporter import BestGenomeReporter
from neat import Checkpointer


"""
This is the fitness function. It evaluates a given genome (Neural Network) on its ability to play Super Mario Bros.
Criteria impacting fitness are:
    - Overall change in xpos (deltaX),
    - reaching notable distance milestones (checkpoints), 
    - Time spent on run (deltaT)
"""
def eval_genome(genome, config):

    # Reconstruct the NN from the genome, define the environment
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    ENV_NAME = "SuperMarioBros-1-1-v0"
    env = gym_super_mario_bros.make(ENV_NAME)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)

    # control variable initialization
    state = env.reset()
    done = False
    max_x_pos = 40
    prev_x_pos = 0
    idle_timer = 100
    fitness = 0.0
    step = 0
    checkpoints = {200:50, 400:75, 600:100, 800:150, 1200:250, 1600:400, 2000:600, 2500:1000, 3000:1500}
    awarded_checkpoints = set()

    # main gameplay loop, genomes fitness is evaluated/updated on every loop
    while not done:
        # input the environment (state) into the NN, get a controller output from the network
        inputs = np.array(state).flatten()
        outputs = network.activate(inputs)
        action = int(np.argmax(outputs))

        # apply the calculated controller output above to mario
        state, reward, done, info = env.step(action)

        # Fitness reward for movement (deltaX), favouring forward movement.
        x_pos = info.get("x_pos")
        progress = x_pos - prev_x_pos
        if progress > 0:
            fitness += progress * 1.5 # reward for moving forward
        else:
            fitness += progress * 0.5 # slightly less reward for moving backwards
        prev_x_pos = x_pos

        # Checking if Mario has stagnated, if he has end the run early to save processing time
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
    
    # Final fitness updates (max_x_pos reached + checkpoints reached)
    env.close()
    fitness += max_x_pos * 0.5
    for cp, bonus in checkpoints.items():
        if x_pos >= cp and cp not in awarded_checkpoints:
            fitness += bonus
            awarded_checkpoints.add(cp)

    # fitness is sent back to the neat algorithm for evolution calculations
    return fitness

"""
This function is the primary pilot of the NEAT algorithm, applies parallel processing to evolve an eventual winner genome via evolution.
Parameters are specified in main.py
"""
def train_neat(config_path, generations, resume_training, report_stats, save_checkpoints, checkpoint_interval, save_genomes):
    # initialize the NEAT config file
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # If the user specifies they want to resume training from an old training population do so, otherwise create a new population
    if resume_training != None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        population = neat.Checkpointer.restore_checkpoint(os.path.join(root, resume_training))
    else:
        population = neat.Population(config)

    # Stat reporter prints information to the console during training such as best_fitness and speciation per generation
    if report_stats == True:
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

    # Allow users to save training checkpoints to allow pausing/resuming of training
    if save_checkpoints != None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_directory = os.path.join(root, save_checkpoints)
        os.makedirs(os.path.join(root, save_checkpoints), exist_ok=True)
        population.add_reporter(Checkpointer(
            generation_interval=checkpoint_interval,
            filename_prefix= os.path.join(checkpoint_directory, "neat-checkpoint-")
        ))

    # Allow users to save genomes during training, providing a means to see progress overtime via run.py
    if save_genomes != None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        population.add_reporter(BestGenomeReporter(os.path.join(root, save_genomes)))

    # This is the NEAT algorithm call via ParallelEvaluator, using cpu_cores-1 for processing.
    num_workers = max(1, os.cpu_count() - 1)
    with ParallelEvaluator(num_workers, eval_genome) as pe:
        winner = population.run(pe.evaluate, generations)

    # At this point, training has completed.
    # Regardless of if the user specified they want to save genomes, the best genome is always saved after training ends.
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("\nBest genome saved!")