import argparse, os, random, sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from agents.ensemble_deep_dyna_q import EnsembleDeepDynaQAgent
from utils.result_save_util import create_experiment_dir, save_json, save_numpy

MOUNTAIN_CAR_LOW = np.array([-1.2, -0.07], dtype=np.float32)
MOUNTAIN_CAR_HIGH = np.array([0.6, 0.07], dtype=np.float32)
MOUNTAIN_CAR_GOAL = 0.5

try: import gymnasium as gym
except: import gym

def make_env(): return gym.make("MountainCar-v0")
def reset_env(env):
    s = env.reset()
    return s[0] if isinstance(s, tuple) else s

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5: return out[0], out[1], (out[2] or out[3]), out[2]
    return out[0], out[1], out[2], bool(out[0][0] >= MOUNTAIN_CAR_GOAL)

def make_planning_reward_fn(reward_shaping_scale=0.1):
    def planning_reward_fn(next_states):
        rewards = -torch.ones((next_states.shape[0], 1), device=next_states.device)
        rewards += reward_shaping_scale * torch.abs(next_states[:, 1:2])
        dones = (next_states[:, 0:1] >= MOUNTAIN_CAR_GOAL).float()
        return rewards, dones
    return planning_reward_fn

def run_ensemble_experiment(episodes=300, runs=3, K=3, lambda_val=1.0, noise_std=0.0):
    all_returns = np.zeros((runs, episodes))
    all_disagreement = np.zeros((runs, episodes))

    for r in range(runs):
        seed = 42 + r
        np.random.seed(seed); torch.manual_seed(seed)
        env = make_env()
        agent = EnsembleDeepDynaQAgent(
            K=K, lambda_penalty=lambda_val,
            state_dim=env.observation_space.shape[0], 
            action_dim=env.action_space.n,
            state_low=MOUNTAIN_CAR_LOW, state_high=MOUNTAIN_CAR_HIGH,
            planning_start_size=1000
        )
        reward_fn = make_planning_reward_fn(0.1)
        global_step = 0

        for ep in range(episodes):
            state = reset_env(env)
            ep_reward = 0
            ep_disagreements = []
            for step in range(200):
                action = env.action_space.sample() if global_step < 1000 else agent.choose_action(state)
                next_state, reward, done, _ = step_env(env, action)
                if noise_std > 0: # Noise ablation for Extension B[cite: 1]
                    next_state += np.random.normal(0, noise_std, size=next_state.shape)
                agent.store_transition(state, action, reward + 0.1 * abs(next_state[1]), next_state, done)
                if global_step >= 1000:
                    info = agent.update(reward_fn)
                    if info and "avg_disagreement" in info:
                        ep_disagreements.append(info["avg_disagreement"])
                ep_reward += reward
                state = next_state
                global_step += 1
                if done: break
            all_returns[r, ep] = ep_reward
            if ep_disagreements: all_disagreement[r, ep] = np.mean(ep_disagreements)
            agent.decay_epsilon()
        env.close()
    return {"returns": np.mean(all_returns, axis=0), "disagreement": np.mean(all_disagreement, axis=0)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--lambda-val", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    args = parser.parse_args()

    results = run_ensemble_experiment(episodes=args.episodes, lambda_val=args.lambda_val, noise_std=args.noise_std)
    save_dir = create_experiment_dir(name=f"ensemble_L{args.lambda_val}_N{args.noise_std}")
    save_numpy(save_dir, "data.npz", **results)
    plt.plot(results["disagreement"])
    plt.title("Ensemble Disagreement")
    plt.savefig(os.path.join(save_dir, "disagreement.png"))
    print(f"Success. Files saved in {save_dir}")