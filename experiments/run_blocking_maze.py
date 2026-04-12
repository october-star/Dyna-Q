import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
from env.maze_env import MazeEnv
from agents.q_learning import QLearningAgent
from agents.dyna_q import DynaQAgent
from experiments.config import AGENT_CONFIGS


def run_blocking_maze_experiment(env, agent_type='q_learning',
                                 planning_steps=5,
                                 episodes_per_phase=500,
                                 max_steps=200,
                                 change_episode=500,
                                 verbose=True):
    # Create agent
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
        raise NotImplementedError(f"Agent type {agent_type} not yet implemented")

    # ===== PHASE 1: Learning short path (before environment change) =====
    if verbose:
        print("\n" + "-" * 50)
        print(f"PHASE 1: Learning Short Path (Episodes 1 to {change_episode})")
        print("-" * 50)

    phase1_success = []
    phase1_steps = []

    for episode in range(1, change_episode + 1):
        state = env.reset()
        success = False
        steps = 0

        for step in range(max_steps):
            action = agent.choose_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            steps += 1
            state = next_state

            if done:
                success = True
                break

        phase1_success.append(1 if success else 0)
        phase1_steps.append(steps)

        # Print progress every 50 episodes
        if verbose and episode % 50 == 0:
            recent_rate = np.mean(phase1_success[-50:])
            avg_steps = np.mean(phase1_steps[-50:])
            print(f"Episode {episode:4d} | Success: {recent_rate:.2%} | Steps: {avg_steps:.1f}")

    phase1_final_rate = np.mean(phase1_success[-100:]) if change_episode >= 100 else np.mean(phase1_success)
    if verbose:
        print(f"\n>>> Phase 1 Complete!")
        print(f"    Final success rate: {phase1_final_rate:.2%}")
        print(f"    Average steps: {np.mean(phase1_steps[-100:]):.1f}")

    # ===== MANUAL ENVIRONMENT CHANGE =====
    if verbose:
        print("\n" + "=" * 50)
        print("MANUAL ENVIRONMENT CHANGE")
        print("=" * 50)
        print("Blocking the left path...")
        print("Opening the right path...")

    # Manually change the environment
    env._change_blocking_maze()

    if verbose:
        print(">>> Environment changed!")
        print("    - Obstacle added at (3, 0) [left path blocked]")
        print("    - Obstacle removed at (0, 7) [right path opened]")

    # ===== PHASE 2: Adapting to long path (after environment change) =====
    if verbose:
        print("\n" + "-" * 50)
        print(
            f"PHASE 2: Adapting to Long Path (Episodes {change_episode + 1} to {change_episode + episodes_per_phase})")
        print("-" * 50)

    phase2_success = []
    phase2_steps = []

    for episode in range(1, episodes_per_phase + 1):
        state = env.reset()
        success = False
        steps = 0

        for step in range(max_steps):
            action = agent.choose_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)

            steps += 1
            state = next_state

            if done:
                success = True
                break

        phase2_success.append(1 if success else 0)
        phase2_steps.append(steps)

        # Print progress every 50 episodes
        if verbose and episode % 50 == 0:
            recent_rate = np.mean(phase2_success[-50:])
            avg_steps = np.mean(phase2_steps[-50:])
            print(f"Episode {episode:4d} | Success: {recent_rate:.2%} | Steps: {avg_steps:.1f}")

    phase2_final_rate = np.mean(phase2_success[-100:]) if episodes_per_phase >= 100 else np.mean(phase2_success)
    if verbose:
        print(f"\n>>> Phase 2 Complete!")
        print(f"    Final success rate: {phase2_final_rate:.2%}")
        print(f"    Average steps: {np.mean(phase2_steps[-100:]):.1f}")

    # ===== RESULTS =====
    if verbose:
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Phase 1 (Short Path - Before Change):")
        print(f"  Success rate: {phase1_final_rate:.2%}")
        print(f"  Avg steps: {np.mean(phase1_steps[-100:]):.1f}")
        print(f"\nPhase 2 (Long Path - After Change):")
        print(f"  Success rate: {phase2_final_rate:.2%}")
        print(f"  Avg steps: {np.mean(phase2_steps[-100:]):.1f}")

        # Calculate performance drop and recovery
        performance_drop = phase1_final_rate - phase2_final_rate
        print(f"\nPerformance Drop: {performance_drop:.2%}")

        # Calculate adaptation speed (episodes to reach 50% of Phase 1 performance)
        if len(phase2_success) >= 50:
            target_rate = phase1_final_rate * 0.5
            recovery_episode = None

            for i in range(50, len(phase2_success) + 1):
                recent_rate = np.mean(phase2_success[i - 50:i])
                if recent_rate >= target_rate:
                    recovery_episode = i
                    break

            if recovery_episode:
                print(f"Adaptation Time: {recovery_episode} episodes to reach {target_rate:.2%}")

        # Model summary for Dyna-Q
        if agent_type == 'dyna_q' and hasattr(agent, 'get_model_size'):
            print(f"\nModel Summary:")
            print(f"  Total transitions stored: {agent.get_model_size()}")
            print(f"  Unique (s,a) pairs visited: {agent.get_visited_pairs_count()}")

        print("=" * 60)

    return phase1_success, phase2_success, phase1_steps, phase2_steps, agent


def test_agent(env, agent, num_episodes=10, max_steps=200, verbose=True):
    """Test the trained agent with greedy policy"""
    successes = 0
    step_list = []

    if verbose:
        print("\n" + "=" * 60)
        print("TESTING TRAINED AGENT (Greedy Policy)")
        print("=" * 60)

    for episode in range(num_episodes):
        state = env.reset()

        for step in range(max_steps):
            action = agent.choose_action(state, training=False)
            next_state, reward, done = env.step(action)

            if done:
                successes += 1
                step_list.append(step + 1)
                if verbose:
                    print(f"  Episode {episode + 1}: ✓ Goal reached in {step + 1} steps")
                break

            state = next_state

        if len(step_list) <= episode and verbose:
            print(f"  Episode {episode + 1}: ✗ Failed to reach goal")
            step_list.append(max_steps)

    if verbose:
        print(f"\nTest Summary:")
        print(f"  Success rate: {successes / num_episodes:.2%}")
        if step_list:
            print(f"  Average steps: {np.mean(step_list):.1f}")
            print(f"  Min steps: {np.min(step_list)}")
            print(f"  Max steps: {np.max(step_list)}")
        print("=" * 60)

    return successes, step_list


if __name__ == "__main__":
    # Create environment with blocking=False (manual change control)
    env = MazeEnv(blocking=False)

    # Run experiment with Q-Learning
    print("\n" + "=" * 70)
    print("TESTING: Q-LEARNING ON BLOCKING MAZE")
    print("=" * 70)

    phase1, phase2, steps1, steps2, agent_q = run_blocking_maze_experiment(
        env=env,
        agent_type='q_learning',
        episodes_per_phase=500,
        max_steps=200,
        change_episode=500,
        verbose=True
    )

    # Test Q-Learning agent
    test_agent(env, agent_q, num_episodes=10)

    # Create fresh environment for Dyna-Q
    env2 = MazeEnv(blocking=False)

    # Run experiment with Dyna-Q
    print("\n" + "=" * 70)
    print("TESTING: DYNA-Q ON BLOCKING MAZE")
    print("=" * 70)

    phase1_d, phase2_d, steps1_d, steps2_d, agent_dyna = run_blocking_maze_experiment(
        env=env2,
        agent_type='dyna_q',
        planning_steps=10,
        episodes_per_phase=500,
        max_steps=200,
        change_episode=500,
        verbose=True
    )

    # Test Dyna-Q agent
    test_agent(env2, agent_dyna, num_episodes=10)