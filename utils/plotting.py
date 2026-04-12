import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(rewards, steps, successes,
                        title="Q-Learning on Dyna Maze",
                        save_path=None):
    """
    Plot learning curves: rewards, steps, and success rate.

    Args:
        rewards: List of episode rewards
        steps: List of episode steps
        successes: List of success booleans
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Smoothing window
    window = min(50, len(rewards) // 10)

    # Plot 1: Episode rewards
    axes[0].plot(rewards, alpha=0.3, color='blue', label='Raw')
    if window > 0:
        smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(smoothed_rewards, color='blue', linewidth=2, label=f'Smoothed (window={window})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Episode steps
    axes[1].plot(steps, alpha=0.3, color='green', label='Raw')
    if window > 0:
        smoothed_steps = np.convolve(steps, np.ones(window) / window, mode='valid')
        axes[1].plot(smoothed_steps, color='green', linewidth=2, label=f'Smoothed')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Goal')
    axes[1].set_title('Steps per Episode')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Success rate
    success_rate = np.convolve(successes, np.ones(window) / window, mode='valid')
    axes[2].plot(success_rate, color='red', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_title(f'Success Rate (window={window})')
    axes[2].set_ylim([-0.05, 1.05])
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()