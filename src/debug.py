import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

import cv2
import numpy as np
import matplotlib.pyplot as plt

ENV_NAME = "SuperMarioBros-1-1-v3"

# =========================
# EASY-TO-CHANGE SETTINGS
# =========================
SETTINGS = {
    "skip": 4,
    "top": 96,
    "bottom": 224,
    "left": 32,
    "right": 192,
    "resize_shape": 50,
    "middle_step": 60,
    "action": 1,              # 1 = move right for RIGHT_ONLY
    "show_stack": True,
    "stack_size": 2,
}


def crop_frame(frame, top, bottom, left, right):
    return frame[top:bottom, left:right, :]


def grayscale_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def resize_frame(frame, shape):
    return cv2.resize(frame, (shape, shape), interpolation=cv2.INTER_AREA)


def make_stack(frames, stack_size):
    if len(frames) < stack_size:
        while len(frames) < stack_size:
            frames.insert(0, frames[0])
    return np.stack(frames[-stack_size:], axis=0)


def skip_step(env, action, skip):
    obs = None
    done = False
    info = {}
    total_reward = 0

    for _ in range(skip):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    return obs, total_reward, done, info


def process_frame(raw_frame, settings):
    cropped = crop_frame(
        raw_frame,
        settings["top"],
        settings["bottom"],
        settings["left"],
        settings["right"]
    )
    gray = grayscale_frame(cropped)
    resized = resize_frame(gray, settings["resize_shape"])
    return {
        "raw": raw_frame,
        "crop": cropped,
        "gray": gray,
        "resize": resized
    }


def plot_pipeline(prev_raw, curr_raw, title, settings):
    prev_processed = process_frame(prev_raw, settings)
    curr_processed = process_frame(curr_raw, settings)

    stages = [
        ("Raw", curr_processed["raw"]),
        ("Crop", curr_processed["crop"]),
        ("Gray", curr_processed["gray"]),
        ("Resize", curr_processed["resize"]),
    ]

    if settings["show_stack"]:
        stacked = make_stack(
            [prev_processed["resize"], curr_processed["resize"]],
            settings["stack_size"]
        )
        stages.append(("Stack", stacked[-1]))

    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))
    if len(stages) == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, stages):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)

        ax.set_title(f"{name}\n{np.array(img).shape}")
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def print_settings(settings):
    print("\n===== CURRENT DEBUG SETTINGS =====")
    for key, value in settings.items():
        print(f"{key}: {value}")
    print("==================================\n")


def main():
    settings = SETTINGS.copy()
    print_settings(settings)

    env = gym_super_mario_bros.make(ENV_NAME)
    env = JoypadSpace(env, RIGHT_ONLY)

    raw0 = env.reset()
    raw1, _, done, _ = skip_step(env, settings["action"], settings["skip"])

    plot_pipeline(raw0, raw1, "INITIAL SCENE", settings)

    prev_frame = raw1
    curr_frame = raw1

    for _ in range(settings["middle_step"]):
        prev_frame = curr_frame
        curr_frame, _, done, _ = skip_step(env, settings["action"], settings["skip"])
        if done:
            break

    plot_pipeline(
        prev_frame,
        curr_frame,
        f"LATER SCENE (~step {settings['middle_step']})",
        settings
    )

    env.close()


if __name__ == "__main__":
    main()