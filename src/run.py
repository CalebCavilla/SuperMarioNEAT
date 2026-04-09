import neat
import os
import pickle
import time
import numpy as np
import gym_super_mario_bros
import cv2

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from src.wrapper import apply_wrappers


# Set up the environment
ENV_NAME = "SuperMarioBros-1-1-v0"
def env_setup(env_name):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    return env


# creates a hud for showing mario's progress through the level on the replay
def stat_display(frame, genome_name, info):
    h, w, _ = frame.shape
    hud_height = 30  # even smaller

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, h - hud_height),
        (w, h),
        (0, 0, 0),
        -1
    )

    alpha = 0.4  # subtle
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # text
    x_pos = info.get("x_pos", 0)
    progress = (x_pos / 3161) * 100
    progress = max(0, min(progress, 100))
    progress_text = f"{progress:.1f}%"
    text = f"{genome_name} | {progress_text}"
    cv2.putText(frame,text,(8, h - 10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255, 255, 255),1,cv2.LINE_AA)

    return frame


def run_trained_genome(config_path, genome_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    with open(genome_path, "rb") as f:
        trained_genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(trained_genome, config)

    env = env_setup(ENV_NAME)
    state = env.reset()
    done = False

    genome_name = os.path.basename(genome_path)
    frame_count = 0

    cv2.namedWindow("NEAT Mario Playback", cv2.WINDOW_NORMAL)

    local_dir = os.path.dirname(__file__)
    video_path = os.path.join(local_dir, "mario_replay.mp4")
    fps = 30
    scale = 4
    video_writer = None

    try:
        while not done:
            frame_count += 1

            inputs = np.array(state).flatten()
            outputs = net.activate(inputs)
            action = int(np.argmax(outputs))

            step_result = env.step(action)
            if len(step_result) == 4:
                state, reward, done, info = step_result
            else:
                state, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            # draw the frame
            frame = env.render(mode="rgb_array")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = stat_display(frame, genome_name, info)

            frame = cv2.resize(
                frame,
                (frame.shape[1] * scale, frame.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST
            )

            # initialize recorder once we know frame size
            if video_writer is None:
                height, width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            cv2.imshow("NEAT Mario Playback", frame)
            video_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.02)

    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    genome_path = os.path.join(local_dir, "best_genomes/gen_2235_best.pkl")
    run_trained_genome(config_path, genome_path)