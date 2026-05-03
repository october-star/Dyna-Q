import os
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np

from agents.tabular_dyna_q import TabularMountainCarDynaQAgent
from utils.discretization import UniformDiscretizer
from utils.result_save_util import create_experiment_dir, save_json, save_numpy

try:
    import gymnasium as gym
except ModuleNotFoundError:
    try:
        import gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MountainCar experiments require gymnasium or gym to be installed."
        ) from exc


MOUNTAIN_CAR_LOW = np.array([-1.2, -0.07], dtype=float)
MOUNTAIN_CAR_HIGH = np.array([0.6, 0.07], dtype=float)


def make_env():
    return gym.make("MountainCar-v0")


def reset_env(env):
    state = env.reset()
    if isinstance(state, tuple):
        return state[0]
    return state


def step_env(env, action):
    outcome = env.step(action)
    if len(outcome) == 5:
        next_state, reward, terminated, truncated, info = outcome
        done = terminated or truncated
        return next_state, reward, done, terminated, truncated, info
    next_state, reward, done, info = outcome
    terminated = bool(next_state[0] >= 0.5)
    truncated = done and not terminated
    return next_state, reward, done, terminated, truncated, info


def run_tabular_mountaincar_experiment(
    bins_per_dim=(10, 10),
    planning_steps=10,
    episodes=300,
    max_steps=200,
    runs=3,
    seed=0,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
):
    episode_steps = np.zeros(episodes)
    episode_returns = np.zeros(episodes)
    success_rate = np.zeros(episodes)

    for run in range(runs):
        run_seed = seed + run
        np.random.seed(run_seed)
        random.seed(run_seed)

        env = make_env()
        try:
            env.reset(seed=run_seed)
        except TypeError:
            pass

        discretizer = UniformDiscretizer(
            low=MOUNTAIN_CAR_LOW,
            high=MOUNTAIN_CAR_HIGH,
            bins_per_dim=np.array(bins_per_dim, dtype=int),
        )
        agent = TabularMountainCarDynaQAgent(
            actions=env.action_space.n,
            discretizer=discretizer,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            planning_steps=planning_steps,
        )

        for episode in range(episodes):
            state = reset_env(env)
            total_reward = 0.0
            success = 0

            for step in range(1, max_steps + 1):
                action = agent.choose_action(state, training=True)
                next_state, reward, done, terminated, truncated, _ = step_env(env, action)
                agent.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    success = int(terminated)
                    episode_steps[episode] += step
                    break
            else:
                episode_steps[episode] += max_steps

            episode_returns[episode] += total_reward
            success_rate[episode] += success

        env.close()

    return {
        "steps": episode_steps / runs,
        "returns": episode_returns / runs,
        "success_rate": success_rate / runs,
    }


def rolling_mean(values, window=5):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def plot_metric(series_by_label, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))
    for label, values in series_by_label.items():
        plt.plot(values, label=label)
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_bucket_sweep(
    bucket_configs=((5, 5), (10, 10), (20, 20)),
    planning_steps=10,
    episodes=300,
    max_steps=200,
    runs=3,
    seed=0,
):
    all_results = {}
    for bucket_config in bucket_configs:
        label = f"{bucket_config[0]}x{bucket_config[1]}"
        print(f"Running bucket config {label}")
        all_results[label] = run_tabular_mountaincar_experiment(
            bins_per_dim=bucket_config,
            planning_steps=planning_steps,
            episodes=episodes,
            max_steps=max_steps,
            runs=runs,
            seed=seed,
        )
    return all_results


if __name__ == "__main__":
    bucket_configs = ((5, 5), (10, 10), (20, 20))
    planning_steps = 10
    episodes = 300
    max_steps = 200
    runs = 3
    seed = 0

    results = run_bucket_sweep(
        bucket_configs=bucket_configs,
        planning_steps=planning_steps,
        episodes=episodes,
        max_steps=max_steps,
        runs=runs,
        seed=seed,
    )

    save_dir = create_experiment_dir(name="mountaincar_tabular_dyna")
    save_json(
        save_dir,
        "config.json",
        {
            "bucket_configs": [list(config) for config in bucket_configs],
            "planning_steps": planning_steps,
            "episodes": episodes,
            "max_steps": max_steps,
            "runs": runs,
            "seed": seed,
            "env": "MountainCar-v0",
        },
    )

    save_numpy(
        save_dir,
        "data.npz",
        **{
            f"steps_{label}": metrics["steps"]
            for label, metrics in results.items()
        },
        **{
            f"returns_{label}": metrics["returns"]
            for label, metrics in results.items()
        },
        **{
            f"success_rate_{label}": metrics["success_rate"]
            for label, metrics in results.items()
        },
    )

    plot_metric(
        {label: metrics["steps"] for label, metrics in results.items()},
        ylabel="Steps to goal",
        title="MountainCar Tabular Dyna-Q: Episodes vs Steps-to-goal",
        save_path=os.path.join(save_dir, "steps_to_goal.png"),
    )

    plot_metric(
        {label: metrics["returns"] for label, metrics in results.items()},
        ylabel="Return",
        title="MountainCar Tabular Dyna-Q: Episodes vs Return",
        save_path=os.path.join(save_dir, "returns.png"),
    )

    plot_metric(
        {
            label: rolling_mean(metrics["returns"], window=5)
            for label, metrics in results.items()
        },
        ylabel="Return (rolling mean, window=5)",
        title="MountainCar Tabular Dyna-Q: Episodes vs Rolling Return",
        save_path=os.path.join(save_dir, "returns_rolling_mean.png"),
    )

    plot_metric(
        {label: metrics["success_rate"] for label, metrics in results.items()},
        ylabel="Success Rate",
        title="MountainCar Tabular Dyna-Q: Episodes vs Success Rate",
        save_path=os.path.join(save_dir, "success_rate.png"),
    )
