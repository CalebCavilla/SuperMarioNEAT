# Super Mario Bro's NEAT

This is a university project for learning neuroevolution by programming and training a bot to complete the first level of the original Super Mario Bros. for the NES using the NEAT (Neuro Evolution of Augmenting Topologies) algorithm.

# Results

https://github.com/user-attachments/assets/17d98a88-48c2-4cd0-a056-747a950ec7b7

After training the bot for ~10 hrs over 2235 generations, the NEAT algorithm produces a top performer Network capable of completing world 1-1 with 100% certainty. The entire genome history, neat-checkpoint histroy, final video, and final genome can be found in the Solution folder. If you wish to reproduce my exact setup deterministcally, run train.py on neat-checkpoint-371 (we lost the first 370 generations :< ).

# Environment Setup (Windows)

Requires Python 3.8.2

## Clone the repository
```
cd <desired-folder>
git clone https://github.com/CalebCavilla/SuperMarioNEAT
cd SuperMarioNEAT
```
## Create virtual environment
```python -3.8 -m venv venv```

## Activate the Environment
```venv\Scripts\activate```

## Install dependencies
```pip install -r requirements.txt```

# Usage
