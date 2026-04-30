import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.maze_env import MazeEnv
from agents.dyna_q import DynaQAgent
from agents.dyna_q_plus import DynaQPlusAgent
from experiments.config import AGENT_CONFIGS
import matplotlib.pyplot as plt
from utils.result_save_util import create_experiment_dir, save_numpy, save_json, save_plot
from experiments.run_dyna_maze import run_static_maze_experiment



def experiment_dyna_maze(n_values=[0,5,10,50], runs=30, episodes=50):
    all_results = {n: np.zeros(episodes) for n in n_values}

    for n in n_values:
        print(f"\nRunning n={n}")

        for run in range(runs):
            env = MazeEnv()

            _, steps, _, _ = run_static_maze_experiment(
                env=env,
                agent_type='dyna_q',
                planning_steps=n,
                episodes=episodes,
                verbose=False
            )

            all_results[n] += np.array(steps)

        all_results[n] /= runs

    return all_results

def plot_dyna_maze(results, save_path=None):
    import matplotlib.pyplot as plt

    for n, steps in results.items():
        plt.plot(steps, label=f"n={n}")

    plt.xlabel("Episodes")
    plt.ylabel("Steps to goal")
    plt.title("Dyna Maze (Figure 8.2)")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def experiment_blocking_maze(runs=30, max_steps=3000, seed=0, kappa=None):
    dyna = np.zeros(max_steps)
    dyna_plus = np.zeros(max_steps)
    dyna_config = AGENT_CONFIGS["dyna_q"]
    dyna_plus_config = AGENT_CONFIGS["dyna_q_plus"]
    dyna_plus_kappa = dyna_plus_config["kappa"] if kappa is None else kappa

    for run in range(runs):
        run_seed = seed + run
        np.random.seed(run_seed)
        random.seed(run_seed)

        env1 = MazeEnv()
        env2 = MazeEnv()

        agent1 = DynaQAgent(
            actions=4,
            alpha=dyna_config["alpha"],
            gamma=dyna_config["gamma"],
            epsilon=dyna_config["epsilon"],
            planning_steps=50,
        )
        agent2 = DynaQPlusAgent(
            actions=4,
            alpha=dyna_plus_config["alpha"],
            gamma=dyna_plus_config["gamma"],
            epsilon=dyna_plus_config["epsilon"],
            planning_steps=50,
            kappa=dyna_plus_kappa,
        )

        s1 = env1.reset()
        s2 = env2.reset()

        cum_r1, cum_r2 = 0, 0

        for t in range(max_steps):

            if t == 1000:
                env1._change_blocking_maze()
                env2._change_blocking_maze()

            # ---- Dyna-Q ----
            a1 = agent1.choose_action(s1)
            s1_next, r1, done1 = env1.step(a1)

            agent1.update(s1, a1, r1, s1_next, done1)
            s1 = env1.reset() if done1 else s1_next

            cum_r1 += r1
            dyna[t] += cum_r1

            # ---- Dyna-Q+ ----
            a2 = agent2.choose_action(s2)
            s2_next, r2, done2 = env2.step(a2)

            agent2.update(s2, a2, r2, s2_next, done2)
            s2 = env2.reset() if done2 else s2_next

            cum_r2 += r2
            dyna_plus[t] += cum_r2

    return dyna/runs, dyna_plus/runs

def plot_blocking(dyna, dyna_plus, save_path=None):

    plt.plot(dyna, label="Dyna-Q")
    plt.plot(dyna_plus, label="Dyna-Q+")
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative reward")
    plt.title("Blocking Maze (Figure 8.4)")
    plt.legend()
    plt.axvline(x=1000, linestyle='--')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def compare_q_vs_dyna(runs=30, episodes=50):
    q_res = np.zeros(episodes)
    dyna_res = np.zeros(episodes)

    for run in range(runs):
        env = MazeEnv()

        _, steps_q, _, _ = run_static_maze_experiment(
            env, 'q_learning', episodes=episodes, verbose=False
        )

        _, steps_d, _, _ = run_static_maze_experiment(
            env, 'dyna_q', planning_steps=50, episodes=episodes, verbose=False
        )

        q_res += np.array(steps_q)
        dyna_res += np.array(steps_d)

    return q_res/runs, dyna_res/runs    


if __name__ == "__main__":

    # ---- Figure 8.2 ----
    results = experiment_dyna_maze()
    # save results
    save_dir = create_experiment_dir(name="dyna_maze")
    fixed_results = {f"n_{k}": v for k, v in results.items()}
    save_numpy(save_dir, "data.npz", **fixed_results)
    save_json(save_dir, "config.json", {
    "n_values": [0,5,50],
    "runs": 30,
    "episodes": 50
    })

    plot_dyna_maze(results, save_path=os.path.join(save_dir, "figure_8_2.png"))

    # ---- Figure 8.4 ----
    dyna, dyna_plus = experiment_blocking_maze()
    save_dir = create_experiment_dir(name="blocking_maze")

    save_numpy(save_dir, "data.npz",
           dyna=dyna,
           dyna_plus=dyna_plus)

    save_json(save_dir, "config.json", {
            "runs": 30,
            "max_steps": 3000,
            "change_step": 1000
    })
   
    plot_blocking(dyna, dyna_plus, save_path=os.path.join(save_dir, "figure_8_4.png"))

    # ---- Comparison ----
    q, dyna = compare_q_vs_dyna()

    save_dir = create_experiment_dir(name="q_vs_dyna")

    save_numpy(save_dir, "data.npz",
        q_learning=q,
        dyna_q=dyna)

    save_json(save_dir, "config.json", {
        "episodes": 50,
        "runs": 30,
        "dyna_n": 50
    })

    plt.plot(q, label="Q-Learning")
    plt.plot(dyna, label="Dyna-Q")
    plt.legend()
    save_plot(save_dir, "comparison.png")
