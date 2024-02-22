import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gymnasium.wrappers import RecordVideo
from IPython import display
from moviepy.editor import VideoFileClip
from stable_baselines3 import PPO

from .train import make_env, read_from_yaml
from .utils import NotVecNormalize

# ----- Notebook utilities ----------------------------------------------------


def evaluate_ares_ea_agent(run_name, n=200, include_position=False):
    loaded_model = PPO.load(f"utils/models/{run_name}/model")
    loaded_config = read_from_yaml(f"utils/models/{run_name}/config")

    env = make_env(loaded_config)
    env = NotVecNormalize(env, f"utils/models/{run_name}/normalizer")

    maes = []
    step_indicies = []
    err_sigma_xs = []
    err_sigma_ys = []
    for _ in range(n):
        terminated = False
        truncated = False
        observation, info = env.reset()
        i = 0
        while not (terminated or truncated):
            i += 1
            action, _ = loaded_model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            err_sigma_xs.append(info["err_sigma_x"])
            err_sigma_ys.append(info["err_sigma_y"])
            step_indicies.append(i)
        maes.append(info["mae_all"] if include_position else info["mae_focus"])
    env.close()

    mean_mae = np.mean(maes)
    rmse = np.sqrt(np.mean(np.square(np.concatenate([err_sigma_xs, err_sigma_ys]))))
    mean_steps = len(err_sigma_xs) / n

    print(f"Evaluation results ({n} evaluations)")
    print("----------------------------------------")
    print(f"==> Mean MAE = {mean_mae}")
    print(f"==> RMSE = {rmse}")
    print(f"==> Mean no. of steps = {mean_steps}")

    sns.lineplot(
        x=step_indicies * 2,
        y=err_sigma_xs + err_sigma_ys,
        hue=[r"$\sigma_x$"] * len(err_sigma_xs) + [r"$\sigma_y$"] * len(err_sigma_ys),
    )
    plt.xlim(0, None)
    plt.xlabel("Step")
    plt.ylabel("Error (m)")
    plt.show()


def make_ares_ea_training_videos():
    recdir = "utils/recordings/ml_workshop"
    imgdir = "img"

    for eval_episode in [0, 27, 343]:
        filename = f"rl-video-episode-{eval_episode}"
        clip = VideoFileClip(f"{recdir}/{filename}.mp4")
        clip.write_gif(f"{imgdir}/ares-ea-{filename}.gif", logger=None)


def make_lunar_lander_gif(model_path, gif_path):
    env = gym.make("LunarLander-v2")
    model = PPO.load(model_path)

    images = []
    obs = env.reset()
    img = env.render(mode="rgb_array")
    done = False
    while not done:
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        img = env.render(mode="rgb_array")

    imageio.mimsave(
        f"{gif_path}.gif",
        [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
        fps=29,
    )
    env.close()


def make_lunar_lander_training_gifs():
    for steps in [1e5, 3e5, 1e6]:
        make_lunar_lander_gif(
            model_path=f"utils/rl/lunar_lander/checkpoints/rl_model_{int(steps)}_steps",
            gif_path=f"img/lunar_lander_trainig_{int(steps)}_steps",
        )


def plot_ares_ea_training_history(run_name):
    monitor = pd.read_csv(
        f"utils/monitors/{run_name}/0.monitor.csv",
        index_col=False,
        skiprows=1,
    )

    plt.figure(figsize=(13, 4))
    plt.subplot(121)
    plt.title("Episode Rewards")
    plt.plot(monitor["t"] / 60, monitor["r"])
    plt.xlabel("Wall time (min)")
    plt.ylabel("Reward")
    plt.grid()
    plt.subplot(122)
    plt.title("Episode Lengths")
    plt.plot(monitor["t"] / 60, monitor["l"], c="tab:orange")
    plt.xlabel("Wall time (min)")
    plt.ylabel("Steps")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_lunar_lander_training_history():
    monitor = pd.read_csv(
        "utils/rl/lunar_lander/monitors/0.monitor.csv", index_col=False, skiprows=1
    )

    plt.figure(figsize=(13, 4))
    plt.subplot(121)
    plt.title("Episode Rewards")
    plt.axhline(-100, color="tab:red", ls="--")
    plt.axhspan(-1000, -100, color="tab:red", alpha=0.2)
    plt.text(120 / 60, -700, "Crashing", color="tab:red")
    plt.axhline(140, color="tab:green", ls="--")
    plt.axhspan(140, 400, color="tab:green", alpha=0.2)
    plt.text(15 / 60, 250, "Landing", color="tab:green")
    plt.plot(monitor["t"] / 60, monitor["r"])
    plt.ylim(-950, 350)
    plt.xlabel("Wall time (min)")
    plt.ylabel("Reward")
    plt.grid()
    plt.subplot(122)
    plt.title("Episode Lengths")
    plt.plot(monitor["t"] / 60, monitor["l"], c="tab:orange")
    plt.xlabel("Wall time (min)")
    plt.ylabel("Steps")
    plt.grid()
    plt.tight_layout()
    plt.show()


def record_video(env):
    return RecordVideo(
        env,
        video_folder="utils/rl/lunar_lander_recordings",
        episode_trigger=lambda i: (i % 60) == 0,
    )


def show_video(filename):
    return display.Video(filename)
