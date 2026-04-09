import argparse
import os
import pickle
import sys
import neat

from src.train import train_neat
from src.run import run_trained_genome

# Train Function
def train(args):
    train_neat(
        config_path = args.config,
        generations = args.generations,
        resume_training= args.resume_training,
        report_stats = args.report_stats,
        save_checkpoints = args.save_checkpoints,
        checkpoint_interval = args.checkpoint_interval,
        save_genomes = args.save_genomes
    )

# Run Function
def run(args):
    run_trained_genome(
        config_path = args.config,
        genome_path = args.genome
    )

def create_parser():
    parser = argparse.ArgumentParser(description="Command Line interface for interacting with Super Mario NEAT bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Training Parser
    train_parser = subparsers.add_parser("train", help="Train the bot from scratch or resume from checkpoint")
    train_parser.add_argument("--config", type=str, default="src/config.txt", help="path to NEAT config file")
    train_parser.add_argument("--generations", type=int, default=2500, help="Number of generations to train for")
    train_parser.add_argument("--resume_training", type=str, default=None, help="The path to the checkpoint file to resume training from (If not set, training will begin from fresh population)")
    train_parser.add_argument("--report_stats", action="store_true", default=True, help="Reports per generation stats to the console like best_fitness and speciation")
    train_parser.add_argument("--no_report_stats", action="store_false", dest="report_stats", help="Prevents stat reporting in console")
    train_parser.add_argument("--save_checkpoints", type=str, default=None, help="The Path to the folder where checkpoints are saved (if not set, checkpoints are NOT saved)")
    train_parser.add_argument("--checkpoint_interval", type=int, default=10, help="How many generations per checkpoint save")
    train_parser.add_argument("--save_genomes", type=str, default=None, help="Saves genomes for replay (if not set, genomes are NOT saved)")
    train_parser.set_defaults(func=train)

    # Run Parser
    run_parser = subparsers.add_parser("run", help="Run a pre-trained model to see how it performs")
    run_parser.add_argument("--config", type=str, default="src/config.txt", help="path to NEAT config file")
    run_parser.add_argument("--genome", type=str, required=True, help="path to trained genome to be run")
    run_parser.set_defaults(func=run)

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()