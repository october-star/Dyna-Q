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

from agents.deep_dyna_q import DeepDynaQAgent
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


MOUNTAIN_CAR_LOW = np.array([-1.2, -0.07], dtype=np.float32)
MOUNTAIN_CAR_HIGH = np.array([0.6, 0.07], dtype=np.float32)
MOUNTAIN_CAR_GOAL_POSITION = 0.5


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
    terminated = bool(next_state[0] >= MOUNTAIN_CAR_GOAL_POSITION)
    truncated = done and not terminated
    return next_state, reward, done, terminated, truncated, info


def evaluate_agent(agent, env, max_steps=200):
    state = reset_env(env)
    total_reward = 0.0

    for step in range(1, max_steps + 1):
        action = agent.choose_action(state, training=False)
        next_state, reward, done, terminated, truncated, _ = step_env(env, action)
        total_reward += reward
        state = next_state

        if done:
            return {
                "return": total_reward,
                "steps": step,
                "success": int(terminated),
            }

    return {
        "return": total_reward,
        "steps": max_steps,
        "success": 0,
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


def make_planning_reward_fn(reward_shaping_scale=0.1):
    def planning_reward_fn(predicted_next_states):
        rewards = -torch.ones(
            (predicted_next_states.shape[0], 1),
            dtype=torch.float32,
            device=predicted_next_states.device,
        )
        if reward_shaping_scale != 0.0:
            rewards = rewards + reward_shaping_scale * torch.abs(predicted_next_states[:, 1:2])

        dones = (predicted_next_states[:, 0:1] >= MOUNTAIN_CAR_GOAL_POSITION).float()
        return rewards, dones

    return planning_reward_fn


def run_deep_dyna_mountaincar_experiment(
    episodes=300,
    max_steps=200,
    runs=3,
    seed=0,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    learning_rate=1e-3,
    model_learning_rate=1e-3,
    batch_size=64,
    planning_batch_size=64,
    planning_steps=5,
    planning_start_size=1000,
    target_update_interval=200,
    replay_capacity=50000,
    hidden_dims=(128, 128),
    model_hidden_dims=(128, 128),
    device=None,
    reward_shaping_scale=0.1,
    warmup_steps=1000,
    model_train_steps=2,
    eval_interval=10,
):
    episode_steps = np.zeros(episodes)
    episode_returns = np.zeros(episodes)
    success_rate = np.zeros(episodes)
    direct_q_losses = np.zeros(episodes)
    model_losses = np.zeros(episodes)
    planning_q_losses = np.zeros(episodes)
    episode_epsilons = np.zeros(episodes)
    eval_returns = np.full(episodes, np.nan)
    eval_steps = np.full(episodes, np.nan)
    eval_success_rate = np.full(episodes, np.nan)

    for run in range(runs):
        run_seed = seed + run
        np.random.seed(run_seed)
        random.seed(run_seed)
        torch.manual_seed(run_seed)
        global_step = 0

        env = make_env()
        eval_env = make_env()
        try:
            env.reset(seed=run_seed)
        except TypeError:
            pass
        try:
            eval_env.reset(seed=run_seed + 10_000)
        except TypeError:
            pass

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        planning_reward_fn = make_planning_reward_fn(reward_shaping_scale=reward_shaping_scale)

        agent = DeepDynaQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            learning_rate=learning_rate,
            model_learning_rate=model_learning_rate,
            batch_size=batch_size,
            planning_batch_size=planning_batch_size,
            planning_steps=planning_steps,
            planning_start_size=planning_start_size,
            target_update_interval=target_update_interval,
            replay_capacity=replay_capacity,
            hidden_dims=hidden_dims,
            model_hidden_dims=model_hidden_dims,
            model_train_steps=model_train_steps,
            state_low=MOUNTAIN_CAR_LOW,
            state_high=MOUNTAIN_CAR_HIGH,
            device=device,
        )

        for episode in range(episodes):
            state = reset_env(env)
            total_reward = 0.0
            success = 0
            episode_direct_losses = []
            episode_model_losses = []
            episode_planning_losses = []

            for step in range(1, max_steps + 1):
                if global_step < warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.choose_action(state, training=True)

                next_state, reward, done, terminated, truncated, _ = step_env(env, action)
                shaped_reward = reward + reward_shaping_scale * abs(float(next_state[1]))
                agent.store_transition(state, action, shaped_reward, next_state, done)
                if global_step >= warmup_steps:
                    loss_info = agent.update(planning_reward_fn)
                else:
                    loss_info = None
                if loss_info is not None:
                    episode_direct_losses.append(loss_info["direct_q_loss"])
                    episode_model_losses.append(loss_info["model_loss"])
                    episode_planning_losses.append(loss_info["planning_q_loss"])

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
            if episode_direct_losses:
                direct_q_losses[episode] += float(np.mean(episode_direct_losses))
            if episode_model_losses:
                model_losses[episode] += float(np.mean(episode_model_losses))
            if episode_planning_losses:
                planning_q_losses[episode] += float(np.mean(episode_planning_losses))
            episode_epsilons[episode] += agent.epsilon

            if (episode + 1) % eval_interval == 0:
                eval_metrics = evaluate_agent(agent, eval_env, max_steps=max_steps)
                eval_returns[episode] += eval_metrics["return"]
                eval_steps[episode] += eval_metrics["steps"]
                eval_success_rate[episode] += eval_metrics["success"]

            agent.decay_epsilon()

        env.close()
        eval_env.close()

    return {
        "steps": episode_steps / runs,
        "returns": episode_returns / runs,
        "success_rate": success_rate / runs,
        "direct_q_loss": direct_q_losses / runs,
        "model_loss": model_losses / runs,
        "planning_q_loss": planning_q_losses / runs,
        "epsilon": episode_epsilons / runs,
        "eval_returns": eval_returns / runs,
        "eval_steps": eval_steps / runs,
        "eval_success_rate": eval_success_rate / runs,
    }


def parse_hidden_dims(raw_value):
    return tuple(int(value) for value in raw_value.split(",") if value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep Dyna-Q on MountainCar.")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--runs", type=int, default=3, help="Number of random seeds / runs.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
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
        default=1000,
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
        default="128,128",
        help="Comma-separated hidden layer sizes for the Q-network.",
    )
    parser.add_argument(
        "--model-hidden-dims",
        type=str,
        default="128,128",
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
        default=1000,
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
    episodes = args.episodes
    max_steps = args.max_steps
    runs = args.runs
    seed = args.seed
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model_hidden_dims = parse_hidden_dims(args.model_hidden_dims)
    reward_shaping_scale = args.reward_shaping_scale
    warmup_steps = args.warmup_steps
    planning_steps = args.planning_steps
    planning_batch_size = args.planning_batch_size
    planning_start_size = args.planning_start_size
    model_train_steps = args.model_train_steps
    eval_interval = args.eval_interval

    results = run_deep_dyna_mountaincar_experiment(
        episodes=episodes,
        max_steps=max_steps,
        runs=runs,
        seed=seed,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        model_learning_rate=args.model_learning_rate,
        batch_size=args.batch_size,
        hidden_dims=hidden_dims,
        model_hidden_dims=model_hidden_dims,
        reward_shaping_scale=reward_shaping_scale,
        warmup_steps=warmup_steps,
        planning_steps=planning_steps,
        planning_batch_size=planning_batch_size,
        planning_start_size=planning_start_size,
        target_update_interval=args.target_update_interval,
        replay_capacity=args.replay_capacity,
        model_train_steps=model_train_steps,
        eval_interval=eval_interval,
    )

    save_dir = create_experiment_dir(name="mountaincar_deep_dyna")
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
            "model_learning_rate": args.model_learning_rate,
            "batch_size": args.batch_size,
            "planning_batch_size": planning_batch_size,
            "planning_steps": planning_steps,
            "planning_start_size": planning_start_size,
            "target_update_interval": args.target_update_interval,
            "replay_capacity": args.replay_capacity,
            "hidden_dims": list(hidden_dims),
            "model_hidden_dims": list(model_hidden_dims),
            "env": "MountainCar-v0",
            "agent": "Deep Dyna-Q",
            "reward_shaping_scale": reward_shaping_scale,
            "warmup_steps": warmup_steps,
            "model_train_steps": model_train_steps,
            "eval_interval": eval_interval,
            "q_loss": "SmoothL1Loss",
            "model_loss_fn": "SmoothL1Loss",
            "double_dqn": True,
            "world_model_target": "state_delta",
            "state_normalization": "[-1, 1]",
        },
    )

    save_numpy(
        save_dir,
        "data.npz",
        steps=results["steps"],
        returns=results["returns"],
        success_rate=results["success_rate"],
        direct_q_loss=results["direct_q_loss"],
        model_loss=results["model_loss"],
        planning_q_loss=results["planning_q_loss"],
        epsilon=results["epsilon"],
        eval_returns=results["eval_returns"],
        eval_steps=results["eval_steps"],
        eval_success_rate=results["eval_success_rate"],
    )

    plot_metric(
        {"Deep Dyna-Q": results["steps"]},
        ylabel="Steps to goal",
        title="MountainCar Deep Dyna-Q: Episodes vs Steps-to-goal",
        save_path=os.path.join(save_dir, "steps_to_goal.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": results["returns"]},
        ylabel="Return",
        title="MountainCar Deep Dyna-Q: Episodes vs Return",
        save_path=os.path.join(save_dir, "returns.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": rolling_mean(results["returns"], window=5)},
        ylabel="Return (rolling mean, window=5)",
        title="MountainCar Deep Dyna-Q: Episodes vs Rolling Return",
        save_path=os.path.join(save_dir, "returns_rolling_mean.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": results["success_rate"]},
        ylabel="Success Rate",
        title="MountainCar Deep Dyna-Q: Episodes vs Success Rate",
        save_path=os.path.join(save_dir, "success_rate.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": results["direct_q_loss"]},
        ylabel="Loss",
        title="MountainCar Deep Dyna-Q: Episodes vs Direct Q Loss",
        save_path=os.path.join(save_dir, "direct_q_loss.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": results["model_loss"]},
        ylabel="Smooth L1 Loss",
        title="MountainCar Deep Dyna-Q: Episodes vs World Model Loss",
        save_path=os.path.join(save_dir, "model_loss.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": results["planning_q_loss"]},
        ylabel="Loss",
        title="MountainCar Deep Dyna-Q: Episodes vs Planning Q Loss",
        save_path=os.path.join(save_dir, "planning_q_loss.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": results["epsilon"]},
        ylabel="Epsilon",
        title="MountainCar Deep Dyna-Q: Episodes vs Epsilon",
        save_path=os.path.join(save_dir, "epsilon.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": results["eval_returns"]},
        ylabel="Eval Return",
        title="MountainCar Deep Dyna-Q: Episodes vs Deterministic Eval Return",
        save_path=os.path.join(save_dir, "eval_return.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": results["eval_steps"]},
        ylabel="Eval Steps to goal",
        title="MountainCar Deep Dyna-Q: Episodes vs Deterministic Eval Steps",
        save_path=os.path.join(save_dir, "eval_steps.png"),
    )

    plot_metric(
        {"Deep Dyna-Q": results["eval_success_rate"]},
        ylabel="Eval Success Rate",
        title="MountainCar Deep Dyna-Q: Episodes vs Deterministic Eval Success",
        save_path=os.path.join(save_dir, "eval_success_rate.png"),
    )
