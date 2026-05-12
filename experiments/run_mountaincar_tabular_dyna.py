import argparse
import os
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np

from agents.tabular_dyna_q import TabularMountainCarDynaQAgent
from utils.discretization import UniformDiscretizer
from utils.plotting import plot_metric, plot_with_rolling
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


def evaluate_agent(agent, env, max_steps=200, eval_episodes=5):
    total_steps = 0.0
    total_return = 0.0
    total_success = 0.0

    for _ in range(eval_episodes):
        state = reset_env(env)
        episode_return = 0.0
        steps_this_episode = max_steps
        success = 0

        for step in range(1, max_steps + 1):
            # Greedy policy: no epsilon exploration
            action = agent.choose_action(state, training=False)

            next_state, reward, done, terminated, truncated, _ = step_env(env, action)

            episode_return += reward
            state = next_state

            if done:
                steps_this_episode = step
                success = int(terminated)
                break

        total_steps += steps_this_episode
        total_return += episode_return
        total_success += success

    return {
        "eval_steps": total_steps / eval_episodes,
        "eval_returns": total_return / eval_episodes,
        "eval_success_rate": total_success / eval_episodes,
    }


def run_tabular_mountaincar_experiment(
    bins_per_dim=(10, 10),
    planning_steps=10,
    episodes=300,
    max_steps=200,
    runs=3,
    seed=0,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.2,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    eval_episodes=5,
):
    episode_steps = np.zeros(episodes)
    episode_returns = np.zeros(episodes)
    success_rate = np.zeros(episodes)

    eval_steps = np.zeros(episodes)
    eval_returns = np.zeros(episodes)
    eval_success_rate = np.zeros(episodes)

    for run in range(runs):
        run_seed = seed + run
        np.random.seed(run_seed)
        random.seed(run_seed)

        env = make_env()
        try:
            env.reset(seed=run_seed)
            env.action_space.seed(run_seed)
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
            steps_this_episode = max_steps

            for step in range(1, max_steps + 1):
                action = agent.choose_action(state, training=True)
                next_state, reward, done, terminated, truncated, _ = step_env(env, action)

                # Use terminated for learning terminal flag.
                # truncated means time limit, not true terminal state.
                agent.update(state, action, reward, next_state, terminated)

                total_reward += reward
                state = next_state

                if done:
                    steps_this_episode = step
                    success = int(terminated)
                    break

            episode_steps[episode] += steps_this_episode
            episode_returns[episode] += total_reward
            success_rate[episode] += success

            agent.decay_epsilon(
                min_epsilon=epsilon_min,
                decay_rate=epsilon_decay,
            )

            eval_metrics = evaluate_agent(
                agent=agent,
                env=env,
                max_steps=max_steps,
                eval_episodes=eval_episodes,
            )

            eval_steps[episode] += eval_metrics["eval_steps"]
            eval_returns[episode] += eval_metrics["eval_returns"]
            eval_success_rate[episode] += eval_metrics["eval_success_rate"]

        env.close()

    return {
        "steps": episode_steps / runs,
        "returns": episode_returns / runs,
        "success_rate": success_rate / runs,
        "eval_steps": eval_steps / runs,
        "eval_returns": eval_returns / runs,
        "eval_success_rate": eval_success_rate / runs,
    }


def run_bucket_sweep(
    bucket_configs=((5, 5), (10, 10), (20, 20)),
    planning_steps=10,
    episodes=300,
    max_steps=200,
    runs=3,
    seed=0,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.2,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    eval_episodes=5,
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
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            eval_episodes=eval_episodes,
        )

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Tabular Dyna-Q bucket sweep on MountainCar."
    )

    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--planning-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument(
        "--bucket-configs",
        type=str,
        default="5x5,10x10,20x20",
        help="Comma-separated bucket configs such as '5x5,10x10,20x20'.",
    )

    return parser.parse_args()


def parse_bucket_configs(raw_value):
    configs = []

    for item in raw_value.split(","):
        left, right = item.lower().split("x")
        configs.append((int(left), int(right)))

    return tuple(configs)


if __name__ == "__main__":
    args = parse_args()

    bucket_configs = parse_bucket_configs(args.bucket_configs)

    results = run_bucket_sweep(
        bucket_configs=bucket_configs,
        planning_steps=args.planning_steps,
        episodes=args.episodes,
        max_steps=args.max_steps,
        runs=args.runs,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        eval_episodes=args.eval_episodes,
    )

    save_dir = create_experiment_dir(name="mountaincar_tabular_dyna_bucket_sweep")

    save_json(
        save_dir,
        "config.json",
        {
            "env": "MountainCar-v0",
            "algorithm": "Tabular Dyna-Q with bucketing",
            "bucket_configs": [list(config) for config in bucket_configs],
            "planning_steps": args.planning_steps,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "runs": args.runs,
            "seed": args.seed,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "eval_episodes": args.eval_episodes,
            "note": "Training uses epsilon-greedy exploration. Evaluation uses greedy actions with training=False. Steps-to-goal is capped at max_steps for episodes that do not reach the goal.",
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
        **{
            f"eval_steps_{label}": metrics["eval_steps"]
            for label, metrics in results.items()
        },
        **{
            f"eval_returns_{label}": metrics["eval_returns"]
            for label, metrics in results.items()
        },
        **{
            f"eval_success_rate_{label}": metrics["eval_success_rate"]
            for label, metrics in results.items()
        },
    )

    # Training curves
    plot_metric(
        {
            label: metrics["steps"]
            for label, metrics in results.items()
        },
        ylabel=f"Training steps to goal / capped at {args.max_steps}",
        title="MountainCar Tabular Dyna-Q: Training Episodes vs Steps-to-goal",
        save_path=os.path.join(save_dir, "training_steps_to_goal.png"),
    )

    plot_with_rolling(
        {
            label: metrics["returns"]
            for label, metrics in results.items()
        },
        window=5,
        ylabel="Training return rolling mean, window=5",
        title="MountainCar Tabular Dyna-Q: Training Episodes vs Return",
        save_path=os.path.join(save_dir, "training_returns_rolling_mean.png"),
    )

    plot_metric(
        {
            label: metrics["success_rate"]
            for label, metrics in results.items()
        },
        ylabel="Training success rate",
        title="MountainCar Tabular Dyna-Q: Training Episodes vs Success Rate",
        save_path=os.path.join(save_dir, "training_success_rate.png"),
    )

    # Greedy evaluation curves
    plot_metric(
        {
            label: metrics["eval_steps"]
            for label, metrics in results.items()
        },
        ylabel=f"Evaluation steps to goal / capped at {args.max_steps}",
        title="MountainCar Tabular Dyna-Q: Greedy Evaluation Steps-to-goal",
        save_path=os.path.join(save_dir, "eval_steps_to_goal.png"),
    )

    plot_with_rolling(
        {
            label: metrics["eval_returns"]
            for label, metrics in results.items()
        },
        window=5,
        ylabel="Evaluation return rolling mean, window=5",
        title="MountainCar Tabular Dyna-Q: Greedy Evaluation Return",
        save_path=os.path.join(save_dir, "eval_returns_rolling_mean.png"),
    )

    plot_metric(
        {
            label: metrics["eval_success_rate"]
            for label, metrics in results.items()
        },
        ylabel="Evaluation success rate",
        title="MountainCar Tabular Dyna-Q: Greedy Evaluation Success Rate",
        save_path=os.path.join(save_dir, "eval_success_rate.png"),
    )

    print(f"Results saved to: {save_dir}")