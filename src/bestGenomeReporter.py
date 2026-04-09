from neat.reporting import BaseReporter
import os
import pickle
import copy

"""
Reporter class for saving genomes during training. A genome is saved whenever a new max_fitness is achieved.
"""
class BestGenomeReporter(BaseReporter):

    def __init__(self, save_path):
        self.current_generation = None
        self.best_fitness = 1
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    # automatically called at the start of every generation by NEAT, retrieves the current gen number
    def start_generation(self, generation):
        self.current_generation = generation

    # automatically called after a generation completes. Checks if new fitness PB is achieved, if so, saves the genome
    def post_evaluate(self, config, population, species, best_genome):
        
        if (best_genome.fitness > self.best_fitness):
            self.best_fitness = best_genome.fitness

            file_path = os.path.join(self.save_path, f"gen_{self.current_generation}_best.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(best_genome, f)
            print(f"Saved new best genome from generation {self.current_generation} to {file_path}")