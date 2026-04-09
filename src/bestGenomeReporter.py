from neat.reporting import BaseReporter
import os
import pickle
import copy

class BestGenomeReporter(BaseReporter):

    def __init__(self, save_path):
        self.current_generation = None
        self.best_fitness = 1
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        
        if (best_genome.fitness > self.best_fitness):
            self.best_fitness = best_genome.fitness
            file_path = os.path.join(self.save_path, f"gen_{self.current_generation}_best.pkl")

            with open(file_path, "wb") as f:
                pickle.dump(best_genome, f)
            print(f"Saved new best genome from generation {self.current_generation} to {file_path}")
    
    def start_generation(self, generation):
        self.current_generation = generation