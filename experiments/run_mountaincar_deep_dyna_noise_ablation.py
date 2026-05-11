import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np

from experiments.run_mountaincar_deep_dyna import run_deep_dyna_mountaincar_experiment
from utils.result_save_util import create_experiment_dir, save_json, save_numpy


def parse_hidden_dims(raw_value):
    return tuple(int(value) for value in raw_value.split(",") if value)


def parse_noise_stds(raw_value):
    return tuple(float(value) for value in raw_value.split(",") if value)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Deep Dyna-Q model-noise ablation on MountainCar."
    )
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--runs", type=int, default=3, help="Number of random seeds / runs.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--noise-stds",
        type=str,
        default="0.0,0.3,0.5,1.0",
        help="Comma-separated Gaussian noise standard deviations for planning.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon.")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon.")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Episode-wise epsilon decay.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Q-network learning rate.")
    parser.add_argument("--model-learning-rate", type=float, default=1e-3, help="World-model learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Direct RL batch size.")
    parser.add_argument("--planning-batch-size", type=int, default=64, help="Planning batch size.")
    parser.add_argument("--planning-steps", type=int, default=5, help="Planning updates per real step.")
    parser.add_argument(
        "--planning-start-size",
        type=int,
        default=500,
        help="Replay buffer size required before planning starts.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=200,
        help="Number of optimizer steps between target-network syncs.",
    )
    parser.add_argument("--replay-capacity", type=int, default=50000, help="Replay buffer size.")
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="64,64",
        help="Comma-separated hidden layer sizes for the Q-network.",
    )
    parser.add_argument(
        "--model-hidden-dims",
        type=str,
        default="64,64",
        help="Comma-separated hidden layer sizes for the world model.",
    )
    parser.add_argument(
        "--reward-shaping-scale",
        type=float,
        default=0.1,
        help="Velocity-based reward shaping coefficient used for training only.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of random-action steps collected before training starts.",
    )
    parser.add_argument(
        "--model-train-steps",
        type=int,
        default=2,
        help="Number of world-model updates per real environment step.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Run deterministic evaluation every N episodes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    noise_stds = parse_noise_stds(args.noise_stds)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model_hidden_dims = parse_hidden_dims(args.model_hidden_dims)

    all_results = {}
    for noise_std in noise_stds:
        label = f"sigma={noise_std}"
        print(f"Running noise ablation with {label}")
        all_results[label] = run_deep_dyna_mountaincar_experiment(
            episodes=args.episodes,
            max_steps=args.max_steps,
            runs=args.runs,
            seed=args.seed,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            learning_rate=args.learning_rate,
            model_learning_rate=args.model_learning_rate,
            batch_size=args.batch_size,
            planning_batch_size=args.planning_batch_size,
            planning_steps=args.planning_steps,
            planning_start_size=args.planning_start_size,
            target_update_interval=args.target_update_interval,
            replay_capacity=args.replay_capacity,
            hidden_dims=hidden_dims,
            model_hidden_dims=model_hidden_dims,
            reward_shaping_scale=args.reward_shaping_scale,
            warmup_steps=args.warmup_steps,
            model_train_steps=args.model_train_steps,
            eval_interval=args.eval_interval,
            planning_noise_std=noise_std,
        )

    save_dir = create_experiment_dir(name="mountaincar_deep_dyna_noise_ablation")
    save_json(
        save_dir,
        "config.json",
        {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "runs": args.runs,
            "seed": args.seed,
            "noise_stds": list(noise_stds),
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "learning_rate": args.learning_rate,
            "model_learning_rate": args.model_learning_rate,
            "batch_size": args.batch_size,
            "planning_batch_size": args.planning_batch_size,
            "planning_steps": args.planning_steps,
            "planning_start_size": args.planning_start_size,
            "target_update_interval": args.target_update_interval,
            "replay_capacity": args.replay_capacity,
            "hidden_dims": list(hidden_dims),
            "model_hidden_dims": list(model_hidden_dims),
            "reward_shaping_scale": args.reward_shaping_scale,
            "warmup_steps": args.warmup_steps,
            "model_train_steps": args.model_train_steps,
            "eval_interval": args.eval_interval,
            "env": "MountainCar-v0",
            "agent": "Deep Dyna-Q",
            "ablation": "Gaussian noise added to simulated next states during planning only",
        },
    )

    save_numpy(
        save_dir,
        "data.npz",
        **{
            f"steps_sigma_{str(noise_std).replace('.', '_')}": metrics["steps"]
            for noise_std, metrics in zip(noise_stds, all_results.values())
        },
        **{
            f"returns_sigma_{str(noise_std).replace('.', '_')}": metrics["returns"]
            for noise_std, metrics in zip(noise_stds, all_results.values())
        },
        **{
            f"success_rate_sigma_{str(noise_std).replace('.', '_')}": metrics["success_rate"]
            for noise_std, metrics in zip(noise_stds, all_results.values())
        },
        **{
            f"eval_steps_sigma_{str(noise_std).replace('.', '_')}": metrics["eval_steps"]
            for noise_std, metrics in zip(noise_stds, all_results.values())
        },
        **{
            f"eval_returns_sigma_{str(noise_std).replace('.', '_')}": metrics["eval_returns"]
            for noise_std, metrics in zip(noise_stds, all_results.values())
        },
        **{
            f"eval_success_rate_sigma_{str(noise_std).replace('.', '_')}": metrics["eval_success_rate"]
            for noise_std, metrics in zip(noise_stds, all_results.values())
        },
        **{
            f"model_loss_sigma_{str(noise_std).replace('.', '_')}": metrics["model_loss"]
            for noise_std, metrics in zip(noise_stds, all_results.values())
        },
        **{
            f"planning_q_loss_sigma_{str(noise_std).replace('.', '_')}": metrics["planning_q_loss"]
            for noise_std, metrics in zip(noise_stds, all_results.values())
        },
    )

    plot_metric(
        {label: metrics["steps"] for label, metrics in all_results.items()},
        ylabel="Steps to goal",
        title="Deep Dyna-Q Noise Ablation: Episodes vs Steps-to-goal",
        save_path=os.path.join(save_dir, "steps_to_goal.png"),
    )

    plot_metric(
        {label: metrics["returns"] for label, metrics in all_results.items()},
        ylabel="Return",
        title="Deep Dyna-Q Noise Ablation: Episodes vs Return",
        save_path=os.path.join(save_dir, "returns.png"),
    )

    plot_metric(
        {label: rolling_mean(metrics["returns"], window=5) for label, metrics in all_results.items()},
        ylabel="Return (rolling mean, window=5)",
        title="Deep Dyna-Q Noise Ablation: Episodes vs Rolling Return",
        save_path=os.path.join(save_dir, "returns_rolling_mean.png"),
    )

    plot_metric(
        {label: metrics["success_rate"] for label, metrics in all_results.items()},
        ylabel="Success Rate",
        title="Deep Dyna-Q Noise Ablation: Episodes vs Success Rate",
        save_path=os.path.join(save_dir, "success_rate.png"),
    )

    plot_metric(
        {label: metrics["eval_steps"] for label, metrics in all_results.items()},
        ylabel="Eval Steps to goal",
        title="Deep Dyna-Q Noise Ablation: Deterministic Eval Steps",
        save_path=os.path.join(save_dir, "eval_steps.png"),
    )

    plot_metric(
        {label: metrics["eval_returns"] for label, metrics in all_results.items()},
        ylabel="Eval Return",
        title="Deep Dyna-Q Noise Ablation: Deterministic Eval Return",
        save_path=os.path.join(save_dir, "eval_returns.png"),
    )

    plot_metric(
        {label: metrics["eval_success_rate"] for label, metrics in all_results.items()},
        ylabel="Eval Success Rate",
        title="Deep Dyna-Q Noise Ablation: Deterministic Eval Success",
        save_path=os.path.join(save_dir, "eval_success_rate.png"),
    )

    plot_metric(
        {label: metrics["model_loss"] for label, metrics in all_results.items()},
        ylabel="MSE Loss",
        title="Deep Dyna-Q Noise Ablation: World Model Loss",
        save_path=os.path.join(save_dir, "model_loss.png"),
    )

    plot_metric(
        {label: metrics["planning_q_loss"] for label, metrics in all_results.items()},
        ylabel="Q Loss",
        title="Deep Dyna-Q Noise Ablation: Planning Q Loss",
        save_path=os.path.join(save_dir, "planning_q_loss.png"),
    )
