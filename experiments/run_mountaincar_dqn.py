import argparse
import os
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.dqn import DQNAgent
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


def run_dqn_mountaincar_experiment(
    episodes=300,
    max_steps=200,
    runs=3,
    seed=0,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    learning_rate=1e-3,
    batch_size=64,
    target_update_interval=200,
    replay_capacity=50000,
    hidden_dims=(128, 128),
    device=None,
    reward_shaping_scale=0.1,
    warmup_steps=1000,
):
    episode_steps = np.zeros(episodes)
    episode_returns = np.zeros(episodes)
    success_rate = np.zeros(episodes)
    episode_losses = np.zeros(episodes)
    episode_epsilons = np.zeros(episodes)

    for run in range(runs):
        run_seed = seed + run
        np.random.seed(run_seed)
        random.seed(run_seed)
        torch.manual_seed(run_seed)
        global_step = 0

        env = make_env()
        try:
            env.reset(seed=run_seed)
        except TypeError:
            pass

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            learning_rate=learning_rate,
            batch_size=batch_size,
            target_update_interval=target_update_interval,
            replay_capacity=replay_capacity,
            hidden_dims=hidden_dims,
            device=device,
        )

        for episode in range(episodes):
            state = reset_env(env)
            total_reward = 0.0
            success = 0
            losses = []

            for step in range(1, max_steps + 1):
                if global_step < warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.choose_action(state, training=True)

                next_state, reward, done, terminated, truncated, _ = step_env(env, action)
                shaped_reward = reward + reward_shaping_scale * abs(float(next_state[1]))
                agent.store_transition(state, action, shaped_reward, next_state, done)
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

                total_reward += reward
                state = next_state
                global_step += 1

                if done:
                    success = int(terminated)
                    episode_steps[episode] += step
                    break
            else:
                episode_steps[episode] += max_steps

            episode_returns[episode] += total_reward
            success_rate[episode] += success
            if losses:
                episode_losses[episode] += float(np.mean(losses))
            episode_epsilons[episode] += agent.epsilon

            agent.decay_epsilon()

        env.close()

    return {
        "steps": episode_steps / runs,
        "returns": episode_returns / runs,
        "success_rate": success_rate / runs,
        "loss": episode_losses / runs,
        "epsilon": episode_epsilons / runs,
    }


def parse_hidden_dims(raw_value):
    return tuple(int(value) for value in raw_value.split(",") if value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a DQN baseline on MountainCar.")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--runs", type=int, default=3, help="Number of random seeds / runs.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon.")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon.")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Episode-wise epsilon decay.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Q-network learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
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
        default="128,128",
        help="Comma-separated hidden layer sizes for the Q-network.",
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
        default=1000,
        help="Number of random-action steps collected before training starts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    episodes = args.episodes
    max_steps = args.max_steps
    runs = args.runs
    seed = args.seed
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    reward_shaping_scale = args.reward_shaping_scale
    warmup_steps = args.warmup_steps

    results = run_dqn_mountaincar_experiment(
        episodes=episodes,
        max_steps=max_steps,
        runs=runs,
        seed=seed,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        replay_capacity=args.replay_capacity,
        hidden_dims=hidden_dims,
        reward_shaping_scale=reward_shaping_scale,
        warmup_steps=warmup_steps,
    )

    save_dir = create_experiment_dir(name="mountaincar_dqn")
    save_json(
        save_dir,
        "config.json",
        {
            "episodes": episodes,
            "max_steps": max_steps,
            "runs": runs,
            "seed": seed,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "target_update_interval": args.target_update_interval,
            "replay_capacity": args.replay_capacity,
            "hidden_dims": list(hidden_dims),
            "env": "MountainCar-v0",
            "agent": "DQN baseline",
            "reward_shaping_scale": reward_shaping_scale,
            "warmup_steps": warmup_steps,
        },
    )

    save_numpy(
        save_dir,
        "data.npz",
        steps=results["steps"],
        returns=results["returns"],
        success_rate=results["success_rate"],
        loss=results["loss"],
        epsilon=results["epsilon"],
    )

    plot_metric(
        {"DQN": results["steps"]},
        ylabel="Steps to goal",
        title="MountainCar DQN Baseline: Episodes vs Steps-to-goal",
        save_path=os.path.join(save_dir, "steps_to_goal.png"),
    )

    plot_metric(
        {"DQN": results["returns"]},
        ylabel="Return",
        title="MountainCar DQN Baseline: Episodes vs Return",
        save_path=os.path.join(save_dir, "returns.png"),
    )

    plot_metric(
        {"DQN": rolling_mean(results["returns"], window=5)},
        ylabel="Return (rolling mean, window=5)",
        title="MountainCar DQN Baseline: Episodes vs Rolling Return",
        save_path=os.path.join(save_dir, "returns_rolling_mean.png"),
    )

    plot_metric(
        {"DQN": results["success_rate"]},
        ylabel="Success Rate",
        title="MountainCar DQN Baseline: Episodes vs Success Rate",
        save_path=os.path.join(save_dir, "success_rate.png"),
    )

    plot_metric(
        {"DQN": results["loss"]},
        ylabel="Loss",
        title="MountainCar DQN Baseline: Episodes vs Mean Training Loss",
        save_path=os.path.join(save_dir, "loss.png"),
    )

    plot_metric(
        {"DQN": results["epsilon"]},
        ylabel="Epsilon",
        title="MountainCar DQN Baseline: Episodes vs Epsilon",
        save_path=os.path.join(save_dir, "epsilon.png"),
    )
