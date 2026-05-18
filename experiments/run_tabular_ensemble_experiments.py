import sys
import os
import random
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from env.maze_env import MazeEnv
from agents.dyna_q import DynaQAgent
from agents.tabular_ensemble_dyna import TabularEnsembleDynaAgent
from utils.result_save_util import create_experiment_dir, save_numpy

# Custom Agent class to test non-averaged independent acting
class IndependentExpertTabularAgent(TabularEnsembleDynaAgent):
    def choose_action(self, state, training=True):
        """Selects actions by querying an isolated, single random Q-table."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.actions)
        
        # Pull a single mind index out of QS to completely drive this step
        n = random.randint(0, self.K - 1)
        expert_q = self.q_tables[n][state]
        
        max_q = np.max(expert_q)
        max_actions = [a for a, q in enumerate(expert_q) if q == max_q]
        return int(np.random.choice(max_actions))

def run_performance_and_disagreement(episodes=50, runs=10):
    """Experiment 1 & 2: Baseline vs Ensemble benchmarks."""
    print("--- Running Track 1: Performance & Disagreement Benchmark (50 Episodes) ---")
    baseline_steps = np.zeros(episodes)
    ensemble_steps = np.zeros(episodes)
    disagreement_curve = np.zeros(episodes)
    
    for r in range(runs):
        seed = 400 + r
        random.seed(seed); np.random.seed(seed)
        
        # Baseline
        env = MazeEnv(blocking=False)
        baseline_agent = DynaQAgent(actions=4, planning_steps=10, alpha=0.1, gamma=0.95, epsilon=0.1)
        for ep in range(episodes):
            state = env.reset()
            steps = 0
            while True:
                action = baseline_agent.choose_action(state)
                next_state, reward, done = env.step(action)
                baseline_agent.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1
                if done or steps > 1000: break
            baseline_steps[ep] += steps
            
        # Ensemble
        env = MazeEnv(blocking=False)
        ensemble_agent = TabularEnsembleDynaAgent(actions=4, K=3, planning_steps=10, alpha=0.1, gamma=0.95, epsilon=0.1)
        for ep in range(episodes):
            state = env.reset()
            steps = 0
            while True:
                action = ensemble_agent.choose_action(state)
                next_state, reward, done = env.step(action)
                ensemble_agent.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1
                if done or steps > 1000: break
            ensemble_steps[ep] += steps
            disagreement_curve[ep] += ensemble_agent.get_ensemble_disagreement()

    return baseline_steps / runs, ensemble_steps / runs, disagreement_curve / runs

def run_noise_ablation(episodes=50, runs=10, noise_levels=[0.3, 0.5, 1.0]):
    """Experiment 3: Performance under transition noise probabilities (sigma)."""
    print("\n--- Running Track 2: Noise Robustness Ablation (50 Episodes) ---")
    results = {}
    
    for sigma in noise_levels:
        print(f"Evaluating transition failure profile noise sigma = {sigma}...")
        ens_steps = np.zeros(episodes)
        
        for r in range(runs):
            seed = 500 + r
            random.seed(seed); np.random.seed(seed)
            env = MazeEnv(blocking=False)
            agent = TabularEnsembleDynaAgent(actions=4, K=3, planning_steps=10)
            
            for ep in range(episodes):
                state = env.reset()
                steps = 0
                while True:
                    action = agent.choose_action(state)
                    next_state, reward, done = env.step(action)
                    
                    # Model corruption ablation
                    if random.random() < sigma:
                        cr = random.randint(0, env.rows - 1)
                        cc = random.randint(0, env.cols - 1)
                        next_state = (cr, cc)
                        
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    steps += 1
                    if done or steps > 1000: break
                ens_steps[ep] += steps
        results[f"noise_sigma_{sigma}"] = ens_steps / runs
        
    return results

def run_q_initialization_ablation(episodes=50, runs=10, ranges=[0.0, 1.0, 5.0]):
    """Ablation Study 4: Impact of uniform Q-table initialization variance."""
    print("\n--- Running Track 3: Random Q-Initialization Ablation (50 Episodes) ---")
    results = {}
    
    for init_scale in ranges:
        print(f"Evaluating initialization noise range scale = {init_scale}...")
        steps_history = np.zeros(episodes)
        disagreement_history = np.zeros(episodes)
        
        for r in range(runs):
            seed = 600 + r
            random.seed(seed); np.random.seed(seed)
            env = MazeEnv(blocking=False)
            agent = TabularEnsembleDynaAgent(actions=4, K=3, planning_steps=10)
            
            if init_scale > 0.0:
                for k in range(agent.K):
                    for row in range(env.rows):
                        for col in range(env.cols):
                            agent.q_tables[k][(row, col)] = np.random.uniform(-init_scale, init_scale, agent.actions)
            
            for ep in range(episodes):
                state = env.reset()
                steps = 0
                while True:
                    action = agent.choose_action(state)
                    next_state, reward, done = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    steps += 1
                    if done or steps > 1000: break
                steps_history[ep] += steps
                disagreement_history[ep] += agent.get_ensemble_disagreement()
                
        results[f"q_init_scale_{init_scale}_steps"] = steps_history / runs
        results[f"q_init_scale_{init_scale}_disagreement"] = disagreement_history / runs
        
    return results

def run_500_episode_long_horizon(episodes=500, runs=5):
    """Extended Track: Policy Selection Ablation & Long Horizon Disagreement Tracking."""
    print(f"\n--- Running Track 4: Extended Horizon Policy Ablation ({episodes} Episodes) ---")
    consensus_steps = np.zeros(episodes)
    independent_steps = np.zeros(episodes)
    extended_disagreement = np.zeros(episodes)
    
    for r in range(runs):
        seed = 700 + r
        random.seed(seed); np.random.seed(seed)
        
        # 1. Averaged Consensus
        env = MazeEnv(blocking=False)
        agent_c = TabularEnsembleDynaAgent(actions=4, K=3, planning_steps=10)
        for ep in range(episodes):
            state = env.reset()
            steps = 0
            while True:
                action = agent_c.choose_action(state)
                next_state, reward, done = env.step(action)
                agent_c.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1
                if done or steps > 1000: break
            consensus_steps[ep] += steps
            extended_disagreement[ep] += agent_c.get_ensemble_disagreement()
            
        # 2. Independent Expert
        env = MazeEnv(blocking=False)
        agent_i = IndependentExpertTabularAgent(actions=4, K=3, planning_steps=10)
        for ep in range(episodes):
            state = env.reset()
            steps = 0
            while True:
                action = agent_i.choose_action(state)
                next_state, reward, done = env.step(action)
                agent_i.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1
                if done or steps > 1000: break
            independent_steps[ep] += steps

    return consensus_steps / runs, independent_steps / runs, extended_disagreement / runs

if __name__ == "__main__":
    # Initialize timestamped folder environment
    save_dir = create_experiment_dir(name="tabular_ensemble_complete_suite")
    
    # Execute full multi-track parameters
    base_perf, ens_perf, disagreement_curve = run_performance_and_disagreement()
    noise_results = run_noise_ablation(noise_levels=[0.3, 0.5, 1.0])
    q_init_results = run_q_initialization_ablation(ranges=[0.0, 1.0, 5.0])
    c_steps_500, i_steps_500, disagreement_500 = run_500_episode_long_horizon(episodes=500, runs=5)
    
    # Package matrices
    all_data = {
        "baseline_performance": base_perf,
        "ensemble_performance": ens_perf,
        "disagreement_50_episodes": disagreement_curve,
        "consensus_steps_500": c_steps_500,
        "independent_steps_500": i_steps_500,
        "disagreement_500_episodes": disagreement_500
    }
    all_data.update(noise_results)
    all_data.update(q_init_results)
    save_numpy(save_dir, "data_complete.npz", **all_data)
    
    # Render Plot 1: Main Efficiency Baseline
    plt.figure(figsize=(10, 5))
    plt.plot(base_perf, label="Single Tabular Dyna-Q Baseline", color="red", linestyle="--")
    plt.plot(ens_perf, label="Tabular Ensemble Dyna-Q (Ours)", color="blue")
    plt.title("Track 1: Learning Efficiency Comparisons")
    plt.xlabel("Episodes"); plt.ylabel("Steps-to-Goal"); plt.legend()
    plt.savefig(os.path.join(save_dir, "01_performance_baseline_vs_ensemble.png"))
    plt.close()
    
    # Render Plot 2: 50-Episode Disagreement Path
    plt.figure(figsize=(10, 5))
    plt.plot(disagreement_curve, color="darkorange")
    plt.title("Track 2: Action-Value Table Variance Divergence (50 Episodes)")
    plt.xlabel("Episodes"); plt.ylabel("Mean Disagreement Metric $u(s)$")
    plt.savefig(os.path.join(save_dir, "02_disagreement_curve_50.png"))
    plt.close()
    
    # Render Plot 3: Transition Noise Tracking
    plt.figure(figsize=(10, 5))
    for key in sorted(noise_results.keys()):
        plt.plot(noise_results[key], label=f"Ensemble System ({key.replace('_', ' ')})")
    plt.title("Track 3: Performance Profiles Across Transition Noise Scales ($\sigma$)")
    plt.xlabel("Episodes"); plt.ylabel("Steps-to-Goal"); plt.legend()
    plt.savefig(os.path.join(save_dir, "03_noise_robustness_ablations.png"))
    plt.close()
    
    # Render Plot 4: Random Q Initializations Bounds
    plt.figure(figsize=(10, 5))
    for key in sorted(q_init_results.keys()):
        if "steps" in key:
            plt.plot(q_init_results[key], label=f"{key.replace('_', ' ').replace(' steps', '')}")
    plt.title("Ablation 4A: Steps-to-Goal Across Initialization Bound Ranges")
    plt.xlabel("Episodes"); plt.ylabel("Steps-to-Goal"); plt.legend()
    plt.savefig(os.path.join(save_dir, "04a_q_init_ablation_steps.png"))
    plt.close()
    
    # Render Plot 5: Random Q Disagreement Decay Lifespan
    plt.figure(figsize=(10, 5))
    for key in sorted(q_init_results.keys()):
        if "disagreement" in key:
            plt.plot(q_init_results[key], label=f"{key.replace('_', ' ').replace(' disagreement', '')}")
    plt.title("Ablation 4B: Disagreement Lifespans Driven by Initialization Ranges")
    plt.xlabel("Episodes"); plt.ylabel("Ensemble Variance $u(s)$"); plt.legend()
    plt.savefig(os.path.join(save_dir, "04b_q_init_ablation_disagreement.png"))
    plt.close()
    
    # Render Plot 6: 500-Episode Extended Disagreement Curve (Looking for the Peak-and-Decay)
    plt.figure(figsize=(10, 5))
    plt.plot(disagreement_500, color="darkmagenta", linewidth=2)
    plt.title("Track 5: Long-Horizon Extended Disagreement Profile (500 Episodes)")
    plt.xlabel("Episodes"); plt.ylabel("Ensemble Variance $u(s)$")
    plt.grid(True, linestyle="--")
    plt.savefig(os.path.join(save_dir, "05_extended_disagreement_curve_500.png"))
    plt.close()
    
    # Render Plot 7: 500-Episode Policy Ablation (Consensus vs Independent Expert)
    plt.figure(figsize=(10, 5))
    plt.plot(c_steps_500, label="Averaged Consensus Selection (QS Pooling)", color="blue")
    plt.plot(i_steps_500, label="Independent Expert Selection (No Averaging)", color="green", alpha=0.6)
    plt.title("Ablation 6: Long-Horizon Policy Selection Performance (500 Episodes)")
    plt.xlabel("Episodes"); plt.ylabel("Steps-to-Goal"); plt.legend()
    plt.grid(True, linestyle="--")
    plt.savefig(os.path.join(save_dir, "06_policy_ablation_steps_500.png"))
    plt.close()
    
    print(f"\nAll data successfully processed.\nOutputs compiled in folder:\n{save_dir}")