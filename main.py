"""
main.py
Main entry point for running experiments
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from experiments.run_dyna_maze import run_static_maze_experiment, test_and_visualize
# from experiments.run_blocking_maze import run_blocking_maze_experiment
from env.maze_env import MazeEnv
from utils.plotting import plot_learning_curve


def main():
    parser = argparse.ArgumentParser(description='Run Q-Learning experiments on Dyna Maze')
    parser.add_argument('--experiment', type=str, default='static',
                        choices=['static', 'blocking'],
                        help='Experiment type: static or blocking maze')
    parser.add_argument('--agent', type=str, default='q_learning',
                        choices=['q_learning', 'dyna_q', 'dyna_q_plus'],
                        help='Agent type')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes for static maze')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Number of steps for blocking maze')
    parser.add_argument('--plot', action='store_false',
                        help='Plot learning curves')

    args = parser.parse_args()

    if args.experiment == 'static':
        print("\n" + "=" * 70)
        print("STATIC MAZE EXPERIMENT")
        print("=" * 70)

        env = MazeEnv(blocking=False)

        # Run experiment
        rewards, steps, successes, agent = run_static_maze_experiment(env=env,
            agent_type=args.agent,
            episodes=args.episodes,
            verbose=True
        )

        # Test agent
        test_and_visualize(agent, env, num_episodes=5)

        # Plot learning curve
        if args.plot:
            plot_learning_curve(
                rewards, steps, successes,
                title=f"{args.agent.upper()} on Static Dyna Maze"
            )

    # elif args.experiment == 'blocking':
    #     print("\n" + "=" * 70)
    #     print("BLOCKING MAZE EXPERIMENT")
    #     print("=" * 70)
    #
    #     # Run experiment
    #     success_rates, step_times, phase1, phase2 = run_blocking_maze_experiment(
    #         agent_type=args.agent,
    #         total_steps=args.steps,
    #         verbose=True
    #     )
    #
    #     # Plot results
    #     if args.plot:
    #         from utils.plotting import plot_blocking_maze_performance
    #         plot_blocking_maze_performance(
    #             success_rates, step_times,
    #             change_step=1000
    #         )


if __name__ == "__main__":
    main()