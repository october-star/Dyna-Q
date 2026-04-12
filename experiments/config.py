# ===== Environment Configuration =====
# MAZE_CONFIG = {
#     'rows': 6,
#     'cols': 9,
#     'start': (2, 0),
#     'goal': (0, 8),
#     'obstacles': {
#         # Vertical wall at column 2
#         'wall_vertical': [(1, 2), (2, 2), (3, 2)],
#         # Right-side wall at column 7
#         'wall_right': [(0, 7), (1, 7), (2, 7)],
#         # Single obstacle
#         'single': [(4, 5)]
#     },
#     'rewards': {
#         'step': 0,
#         'goal': 1
#     }
# }
#
# # ===== Blocking Maze Configuration =====
# BLOCKING_MAZE_CONFIG = {
#     'change_step': 1000,
#     'block_path': (3, 0),      # Add obstacle to block left path
#     'open_path': (0, 7)        # Remove obstacle to open right path
# }

# ===== Agent Configuration =====
AGENT_CONFIGS = {
    'q_learning': {
        'alpha': 0.1,
        'gamma': 0.95,
        'epsilon': 0.1
    },
    'dyna_q': {
        'alpha': 0.1,
        'gamma': 0.95,
        'epsilon': 0.1,
        'planning_steps': 5,     # Number of planning steps per real step
        'model': 'tabular'        # Tabular model for transitions
    },
    'dyna_q_plus': {
        'alpha': 0.1,
        'gamma': 0.95,
        'epsilon': 0.1,
        'planning_steps': 5,
        'kappa': 0.001,           # Exploration bonus scaling factor
        'model': 'tabular'
    }
}

# ===== Training Configuration =====
TRAINING_CONFIGS = {
    'static_maze': {
        'episodes': 500,
        'max_steps_per_episode': 200,
        'eval_interval': 50,      # Print progress every N episodes
        'num_test_episodes': 10
    },
    'blocking_maze': {
        'total_steps': 2000,
        'eval_window': 100,        # Success rate window size
        'num_runs': 10,            # Number of independent runs for statistics
        'change_step': 1000
    }
}

# ===== Visualization =====
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'font_size': 12,
    'line_width': 2,
    'colors': {
        'q_learning': 'blue',
        'dyna_q': 'green',
        'dyna_q_plus': 'red'
    }
}