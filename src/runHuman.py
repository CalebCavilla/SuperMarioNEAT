import gym
import gym_super_mario_bros
import numpy as np
import pygame
from pygame.locals import *
import json
from datetime import datetime
import time

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Initialize the environment and game for Level 1-1
ENV_NAME = "SuperMarioBros-1-1-v0"
env = gym_super_mario_bros.make(ENV_NAME)
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()

# Initialize Pygame
pygame.init()
window_width, window_height = 800, 600
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Play Super Mario Bros (NES) - Level 1-1")

# Pygame clock to limit FPS
clock = pygame.time.Clock()

key_map = {
    K_a: 8,        # A button (Jump) at index 8
    K_s: 0,        # B button (Run/Fireball) at index 0
    #K_BACKSPACE: 2,  # Select button at index 2
    #K_RETURN: 3,   # Start button at index 3
    #K_UP: 4,       # Up at index 4
    #K_DOWN: 5,     # Down at index 5
    #K_LEFT: 6,     # Left at index 6
    K_RIGHT: 7     # Right at index 7
}

# Define COMPLEX_MOVEMENT options
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]

# Initial action state (all buttons released)
action = np.array([0] * 9)

# Limit to a specific number of episodes
current_episode = 0
prev_lives = 2  # Start with 2 lives
prev_level = 0  # Start at level 0

def convert_action_to_complex(action_array):
    """Convert the current action array to one of the COMPLEX_MOVEMENT options."""
    active_keys = []
    
    # Check which keys are pressed
    if action_array[7] == 1:  # Right
        active_keys.append('right')
    # if action_array[6] == 1:  # Left
    #     active_keys.append('left')
    # if action_array[4] == 1:  # Up
    #     active_keys.append('up')
    # if action_array[5] == 1:  # Down
    #     active_keys.append('down')
    if action_array[8] == 1:  # A (Jump)
        active_keys.append('A')
    if action_array[0] == 1:  # B (Run/Fireball)
        active_keys.append('B')

    # Match active buttons to COMPLEX_MOVEMENT options
    for movement in COMPLEX_MOVEMENT:
        if sorted(movement) == sorted(active_keys):
            return movement
    return ['NOOP']  # Default to 'NOOP' if no match

start_time = time.time()
prev_x_pos = 0
fitness = 0
max_x_pos = 0
done = False
while not done:
    # Limit the frame rate to 90 FPS
    clock.tick(90)

    # Get the current frame from the environment
    frame = env.render(mode='rgb_array')

    # Convert the frame to Pygame format and scale it to fit the window
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    scaled_surface = pygame.transform.scale(frame_surface, (window_width, window_height))

    # Display the frame
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()

    # Event handling for keys
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        elif event.type == KEYDOWN:
            if event.key in key_map:
                action[key_map[event.key]] = 1  # Press button

        elif event.type == KEYUP:
            if event.key in key_map:
                action[key_map[event.key]] = 0  # Release button

    # Convert current action to a COMPLEX_MOVEMENT option and store it
    complex_action = convert_action_to_complex(action)

    # Take a step in the environment
    complex_action = convert_action_to_complex(action)
    action_index = COMPLEX_MOVEMENT.index(complex_action)
    obs, rew, done, info = env.step(action_index)

    # reward forward movement
    x_pos = info.get("x_pos", 0)
    progress = x_pos - prev_x_pos
    if progress > 0:
        fitness += progress * 1.5 # reward for moving forward
    else:
        fitness += progress * 0.5 # slightly less reward for moving backwards
    prev_x_pos = x_pos

    if (x_pos > max_x_pos):
        max_x_pos = x_pos
    

    # time penalty to encourage fast movement
    fitness -= 0.2
    print("Fitness:", fitness)

    # Check current lives
    current_lives = info.get('lives', prev_lives)
    current_level = info.get('levelLo', prev_level)


env.close()
fitness += max_x_pos * 0.5
checkpoints = {200:50, 400:75, 600:100, 800:150, 1200:250, 1600:400, 2000:600, 2500:1000, 3000:1500}
awarded_checkpoints = set()
for cp, bonus in checkpoints.items():
    if x_pos >= cp and cp not in awarded_checkpoints:
        fitness += bonus
        awarded_checkpoints.add(cp)

print("Final fitness:", fitness)
