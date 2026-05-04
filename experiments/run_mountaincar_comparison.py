import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np

from experiments.run_mountaincar_tabular_dyna import run_tabular_mountaincar_experiment
from experiments.run_mountaincar_dqn import run_dqn_mountaincar_experiment
from experiments.run_mountaincar_deep_dyna import run_deep_dyna_mountaincar_experiment
from utils.result_save_util import create_experiment_dir, save_json, save_numpy


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


def run_mountaincar_comparison(
    episodes=300,
    max_steps=200,
    runs=3,
    seed=0,
    tabular_bins=(10, 10),
    tabular_planning_steps=10,
    dqn_hidden_dims=(128, 128),
    dqn_reward_shaping_scale=0.1,
    dqn_warmup_steps=1000,
    deep_hidden_dims=(128, 128),
    deep_model_hidden_dims=(128, 128),
    deep_reward_shaping_scale=0.1,
    deep_warmup_steps=1000,
    deep_planning_steps=5,
    deep_planning_batch_size=64,
    deep_planning_start_size=1000,
    deep_model_train_steps=2,
):
    print("Running Tabular Dyna-Q baseline")
    tabular_results = run_tabular_mountaincar_experiment(
        bins_per_dim=tabular_bins,
        planning_steps=tabular_planning_steps,
        episodes=episodes,
        max_steps=max_steps,
        runs=runs,
        seed=seed,
    )

    print("Running DQN baseline")
    dqn_results = run_dqn_mountaincar_experiment(
        episodes=episodes,
        max_steps=max_steps,
        runs=runs,
        seed=seed,
        hidden_dims=dqn_hidden_dims,
        reward_shaping_scale=dqn_reward_shaping_scale,
        warmup_steps=dqn_warmup_steps,
    )

    print("Running Deep Dyna-Q")
    deep_results = run_deep_dyna_mountaincar_experiment(
        episodes=episodes,
        max_steps=max_steps,
        runs=runs,
        seed=seed,
        hidden_dims=deep_hidden_dims,
        model_hidden_dims=deep_model_hidden_dims,
        reward_shaping_scale=deep_reward_shaping_scale,
        warmup_steps=deep_warmup_steps,
        planning_steps=deep_planning_steps,
        planning_batch_size=deep_planning_batch_size,
        planning_start_size=deep_planning_start_size,
        model_train_steps=deep_model_train_steps,
    )

    return {
        "tabular_dyna_q": tabular_results,
        "dqn_baseline": dqn_results,
        "deep_dyna_q": deep_results,
    }


def parse_hidden_dims(raw_value):
    return tuple(int(value) for value in raw_value.split(",") if value)


def parse_bucket_config(raw_value):
    left, right = raw_value.lower().split("x")
    return (int(left), int(right))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare MountainCar tabular, DQN, and Deep Dyna-Q agents.")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--runs", type=int, default=3, help="Number of random seeds / runs.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--tabular-bins", type=str, default="10x10", help="Tabular bucket config, e.g. 10x10.")
    parser.add_argument("--tabular-planning-steps", type=int, default=10, help="Tabular planning steps.")
    parser.add_argument("--dqn-hidden-dims", type=str, default="128,128", help="DQN hidden layer sizes.")
    parser.add_argument("--dqn-reward-shaping-scale", type=float, default=0.1, help="DQN reward shaping coefficient.")
    parser.add_argument("--dqn-warmup-steps", type=int, default=1000, help="DQN warmup steps.")
    parser.add_argument("--deep-hidden-dims", type=str, default="128,128", help="Deep Dyna-Q hidden layer sizes.")
    parser.add_argument("--deep-model-hidden-dims", type=str, default="128,128", help="Deep Dyna-Q world-model hidden layer sizes.")
    parser.add_argument("--deep-reward-shaping-scale", type=float, default=0.1, help="Deep Dyna-Q reward shaping coefficient.")
    parser.add_argument("--deep-warmup-steps", type=int, default=1000, help="Deep Dyna-Q warmup steps.")
    parser.add_argument("--deep-planning-steps", type=int, default=5, help="Deep Dyna-Q planning steps.")
    parser.add_argument("--deep-planning-batch-size", type=int, default=64, help="Deep Dyna-Q planning batch size.")
    parser.add_argument("--deep-planning-start-size", type=int, default=1000, help="Replay size before Deep Dyna-Q planning starts.")
    parser.add_argument("--deep-model-train-steps", type=int, default=2, help="Deep Dyna-Q model updates per real step.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    episodes = args.episodes
    max_steps = args.max_steps
    runs = args.runs
    seed = args.seed

    tabular_bins = parse_bucket_config(args.tabular_bins)
    tabular_planning_steps = args.tabular_planning_steps

    dqn_hidden_dims = parse_hidden_dims(args.dqn_hidden_dims)
    dqn_reward_shaping_scale = args.dqn_reward_shaping_scale
    dqn_warmup_steps = args.dqn_warmup_steps

    deep_hidden_dims = parse_hidden_dims(args.deep_hidden_dims)
    deep_model_hidden_dims = parse_hidden_dims(args.deep_model_hidden_dims)
    deep_reward_shaping_scale = args.deep_reward_shaping_scale
    deep_warmup_steps = args.deep_warmup_steps
    deep_planning_steps = args.deep_planning_steps
    deep_planning_batch_size = args.deep_planning_batch_size
    deep_planning_start_size = args.deep_planning_start_size
    deep_model_train_steps = args.deep_model_train_steps

    results = run_mountaincar_comparison(
        episodes=episodes,
        max_steps=max_steps,
        runs=runs,
        seed=seed,
        tabular_bins=tabular_bins,
        tabular_planning_steps=tabular_planning_steps,
        dqn_hidden_dims=dqn_hidden_dims,
        dqn_reward_shaping_scale=dqn_reward_shaping_scale,
        dqn_warmup_steps=dqn_warmup_steps,
        deep_hidden_dims=deep_hidden_dims,
        deep_model_hidden_dims=deep_model_hidden_dims,
        deep_reward_shaping_scale=deep_reward_shaping_scale,
        deep_warmup_steps=deep_warmup_steps,
        deep_planning_steps=deep_planning_steps,
        deep_planning_batch_size=deep_planning_batch_size,
        deep_planning_start_size=deep_planning_start_size,
        deep_model_train_steps=deep_model_train_steps,
    )

    save_dir = create_experiment_dir(name="mountaincar_comparison")
    save_json(
        save_dir,
        "config.json",
        {
            "episodes": episodes,
            "max_steps": max_steps,
            "runs": runs,
            "seed": seed,
            "agents": {
                "tabular_dyna_q": {
                    "bins": list(tabular_bins),
                    "planning_steps": tabular_planning_steps,
                },
                "dqn_baseline": {
                    "hidden_dims": list(dqn_hidden_dims),
                    "reward_shaping_scale": dqn_reward_shaping_scale,
                    "warmup_steps": dqn_warmup_steps,
                },
                "deep_dyna_q": {
                    "hidden_dims": list(deep_hidden_dims),
                    "model_hidden_dims": list(deep_model_hidden_dims),
                    "reward_shaping_scale": deep_reward_shaping_scale,
                    "warmup_steps": deep_warmup_steps,
                    "planning_steps": deep_planning_steps,
                    "planning_batch_size": deep_planning_batch_size,
                    "planning_start_size": deep_planning_start_size,
                    "model_train_steps": deep_model_train_steps,
                },
            },
            "env": "MountainCar-v0",
            "notes": "Training return curves include exploration; deterministic eval is only tracked in the standalone Deep Dyna-Q experiment.",
        },
    )

    save_numpy(
        save_dir,
        "data.npz",
        tabular_steps=results["tabular_dyna_q"]["steps"],
        tabular_returns=results["tabular_dyna_q"]["returns"],
        tabular_success_rate=results["tabular_dyna_q"]["success_rate"],
        dqn_steps=results["dqn_baseline"]["steps"],
        dqn_returns=results["dqn_baseline"]["returns"],
        dqn_success_rate=results["dqn_baseline"]["success_rate"],
        dqn_loss=results["dqn_baseline"]["loss"],
        dqn_epsilon=results["dqn_baseline"]["epsilon"],
        deep_steps=results["deep_dyna_q"]["steps"],
        deep_returns=results["deep_dyna_q"]["returns"],
        deep_success_rate=results["deep_dyna_q"]["success_rate"],
        deep_direct_q_loss=results["deep_dyna_q"]["direct_q_loss"],
        deep_model_loss=results["deep_dyna_q"]["model_loss"],
        deep_planning_q_loss=results["deep_dyna_q"]["planning_q_loss"],
        deep_epsilon=results["deep_dyna_q"]["epsilon"],
    )

    plot_metric(
        {
            "Tabular Dyna-Q": results["tabular_dyna_q"]["steps"],
            "DQN Baseline": results["dqn_baseline"]["steps"],
            "Deep Dyna-Q": results["deep_dyna_q"]["steps"],
        },
        ylabel="Steps to goal",
        title="MountainCar: Episodes vs Steps-to-goal",
        save_path=os.path.join(save_dir, "steps_to_goal_comparison.png"),
    )

    plot_metric(
        {
            "Tabular Dyna-Q": results["tabular_dyna_q"]["returns"],
            "DQN Baseline": results["dqn_baseline"]["returns"],
            "Deep Dyna-Q": results["deep_dyna_q"]["returns"],
        },
        ylabel="Return",
        title="MountainCar: Episodes vs Return",
        save_path=os.path.join(save_dir, "returns_comparison.png"),
    )

    plot_metric(
        {
            "Tabular Dyna-Q": rolling_mean(results["tabular_dyna_q"]["returns"], window=5),
            "DQN Baseline": rolling_mean(results["dqn_baseline"]["returns"], window=5),
            "Deep Dyna-Q": rolling_mean(results["deep_dyna_q"]["returns"], window=5),
        },
        ylabel="Return (rolling mean, window=5)",
        title="MountainCar: Episodes vs Rolling Return",
        save_path=os.path.join(save_dir, "returns_rolling_mean_comparison.png"),
    )

    plot_metric(
        {
            "Tabular Dyna-Q": results["tabular_dyna_q"]["success_rate"],
            "DQN Baseline": results["dqn_baseline"]["success_rate"],
            "Deep Dyna-Q": results["deep_dyna_q"]["success_rate"],
        },
        ylabel="Success Rate",
        title="MountainCar: Episodes vs Success Rate",
        save_path=os.path.join(save_dir, "success_rate_comparison.png"),
    )
