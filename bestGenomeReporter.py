from neat.reporting import BaseReporter
import os
import pickle
import copy

class BestGenomeReporter(BaseReporter):

    def __init__(self):
        self.current_generation = None
        os.makedirs("best_genomes", exist_ok=True)

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        
        if (self.current_generation % 10 == 0):

            save_path = os.path.join("best_genomes", f"gen_{self.current_generation}_best.pkl")

            with open(save_path, "wb") as f:
                pickle.dump({
                "generation": self.current_generation,
                "fitness": best_genome.fitness,
                "genome": copy.deepcopy(best_genome)
            }, f)
            print(f"Saved best genome for generation {self.current_generation}")
    
    def start_generation(self, generation):
        self.current_generation = generation