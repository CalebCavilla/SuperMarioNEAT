# Super Mario Bro's NEAT

This is a university project for learning neuroevolution by programming and training a bot to complete the first level of the original Super Mario Bros. for the NES using the NEAT (Neuro Evolution of Augmenting Topologies) algorithm (neat-python docs: https://neat-python.readthedocs.io/en/latest/index.html).

# Results

https://github.com/user-attachments/assets/17d98a88-48c2-4cd0-a056-747a950ec7b7

After training the bot for ~10 hrs over 2235 generations, the NEAT algorithm produces a top performer Network capable of completing world 1-1 with 100% certainty. The entire genome history, neat-checkpoint histroy, final video, and final genome can be found in the Solution folder. If you wish to reproduce my exact setup deterministcally, run train.py on neat-checkpoint-371 (we lost the first 370 generations :< ).

# Environment Setup (Windows)

Requires Python 3.8.2

### Clone the repository
```
cd <desired-folder>
git clone https://github.com/CalebCavilla/SuperMarioNEAT
cd SuperMarioNEAT
```
### Create virtual environment
```python -3.8 -m venv venv```

### Activate the Environment
```venv\Scripts\activate```

### Install dependencies
```pip install -r requirements.txt```

# Usage
This project utalizes a command line interface for executing three primary scripts:
  - train.py
  - run.py
  - runHuman.py

## Training the Bot

To train the bot from scratch with no options (default config, 2500 generations, stat reporting, no checkpoint/genome saving):

``` python main.py train ```

### Optional Arguments:
```
-- config <config_path>         # defaults to path already present in the repo, can ignore unless imported custom config.
-- generations <int>            # The number of generations to train for, defaults to 2500.
-- resume_training <load_path>  # The path to the checkpoint file to resume training from. (If not set, training will begin from fresh population)
-- no_report_stats              # Prevents stat reporting in console during training. (Enabled by default)
-- save_checkpoints <save_path> # The Path to the folder where checkpoints are to be saved, creates folder if path does not exist. (if not set, checkpoints are NOT saved)
-- checkpoint_interval <int>    # How many generations per checkpoint save, defaults to 100
-- save_genomes <save_path>     # The path to the folder to save genomes for replay, creates folder if path does not exist (if not set, genomes are NOT saved)
```
### Examples:

#### Resume training from checkpoint:
```python main.py train --resume_training checkpoints/neat-checkpoint-50```

### Solution setup
``` python main.py train --save_checkpoints checkpoints --save_genomes genomes ```

## Running Trained Models

This function allows viewing of a pre-trained genome completing level 1-1 without evolutionary pressure

``` python main.py run ```

### Required Argument
```
--genome <save_path> # path to trained genome to be run
```
### Example

```python main.py run --genome genome/best_genome.plk```

## Playing has a Human

This function allows a user to play Super Mario Bros themselves with their own fitness score for compairison and testing against the bots

### Example

``` python main.py runHuman ```

# Config.txt

The config file is located in src. This is the main controller of the NEAT algorithm and is very sensitive to change. Take caution in making any modifications
to this file. The parameters are already well optimized for this task so you shouldnt need to touch anything. If you must, there are only 5 settings you will ever need to modify:

```
pop_size                = 150   # Defines the number of genomes per generation. Massive impact on training results/speed.
conn_add_prob           = 0.3   # The probability of a connection being made between nodes in the network per generation. Moderates network complexity, large impact on speed.
node_add_prob           = 0.10  # The probability of a node being added to a hidden layer per generation. Morderates network complexity, large impact on speed.
compatibility_threshold = 2.2   # Controls speciation, if two species of genomes are less diverse than this threshold they are combined, otherwise kept seperate.
max_stagnation          = 15    # The limit on how long a species can survive without improvement in best_fitness.
```
See https://neat-python.readthedocs.io/en/latest/config_file.html for more details on the config.


