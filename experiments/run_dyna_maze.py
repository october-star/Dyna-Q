import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
from env.maze_env import MazeEnv
from agents.q_learning import QLearningAgent
from agents.dyna_q import DynaQAgent
from experiments.config import TRAINING_CONFIGS, AGENT_CONFIGS


def run_static_maze_experiment(env, agent_type='q_learning',
                               planning_steps=5, episodes=None,
                               verbose=True):
    # Load configuration
    if episodes is None:
        episodes = TRAINING_CONFIGS['static_maze']['episodes']

    max_steps = TRAINING_CONFIGS['static_maze']['max_steps_per_episode']
    eval_interval = TRAINING_CONFIGS['static_maze']['eval_interval']

    # Create agent based on type
    if agent_type == 'q_learning':
        config = AGENT_CONFIGS['q_learning']
        agent = QLearningAgent(
            actions=4,
            alpha=config['alpha'],
            gamma=config['gamma'],
            epsilon=config['epsilon']
        )
    elif agent_type == 'dyna_q':
        config = AGENT_CONFIGS['dyna_q']
        agent = DynaQAgent(
            actions=4,
            alpha=config['alpha'],
            gamma=config['gamma'],
            epsilon=config['epsilon'],
            planning_steps=planning_steps
        )
    else:
        # TODO: Implement Dyna-Q+ agent
        raise NotImplementedError(f"Agent type {agent_type} not yet implemented")

    if verbose:
        print("=" * 60)
        print(f"STATIC MAZE EXPERIMENT - {agent_type.upper()}")
        print("=" * 60)
        print(f"Environment: 6x9 grid, start=(2,0), goal=(0,8)")
        print(f"Agent: α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon}")
        print(f"Training: {episodes} episodes, max {max_steps} steps/episode")
        print("-" * 60)

    # Training loop
    episode_rewards = []
    episode_steps = []
    success_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        success = False

        for step in range(max_steps):
            action = agent.choose_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                success = True
                break

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        success_history.append(success)

        # Print progress
        if verbose and (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_steps = np.mean(episode_steps[-eval_interval:])
            success_rate = np.mean(success_history[-eval_interval:])
            print(f"Episode {episode + 1:4d} | "
                  f"Success: {success_rate:.2f} | "
                  f"Steps: {avg_steps:.1f} | "
                  f"Reward: {avg_reward:.2f}")

    if verbose:
        print("-" * 60)
        print(f"Final Results:")
        print(f"  Success rate (last 100): {np.mean(success_history[-100:]):.2%}")
        print(f"  Avg steps (last 100): {np.mean(episode_steps[-100:]):.1f}")
        print("=" * 60)

    return episode_rewards, episode_steps, success_history, agent


def test_and_visualize(agent, env, num_episodes=5):
    print("\n" + "=" * 60)
    print("TESTING TRAINED AGENT")
    print("=" * 60)

    test_successes = []
    test_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        steps = 0

        while steps < 200:
            action = agent.choose_action(state, training=False)
            next_state, reward, done = env.step(action)
            steps += 1

            if done:
                test_successes.append(1)
                test_steps.append(steps)
                print(f"  Episode {episode + 1}: ✓ Goal reached in {steps} steps")
                break

            state = next_state

            if steps >= 200:
                test_successes.append(0)
                test_steps.append(200)
                print(f"  Episode {episode + 1}: ✗ Failed to reach goal")
                break

    print(f"\nTest Summary:")
    print(f"  Success rate: {np.mean(test_successes):.2%}")
    print(f"  Average steps: {np.mean(test_steps):.1f}")

    return test_successes, test_steps


if __name__ == "__main__":
    # set env
    env = MazeEnv(blocking=False)

    # Run experiment
    rewards, steps, successes, agent = run_static_maze_experiment(env=env,
        agent_type='dyna_q',
        episodes=500,
        verbose=True
    )

    # Test and visualize
    test_and_visualize(agent, env)