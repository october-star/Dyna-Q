import copy

class MazeEnv:
    def __init__(self, blocking=False):
        self.rows = 6
        self.cols = 9

        # Fixed start and goal positions 
        self.start = (2, 0)
        self.goal = (0, 8)

        self.blocking = blocking

        self._init_static_maze()
        self.reset()

    # ===== Static Maze（Figure 8.2）=====
    def _init_static_maze(self):
        """
         Standard Dyna Maze:
        - Vertical wall in column 2
        - Partial wall in column 7
        - One isolated obstacle
        """
        self.obstacles = set()

        # Vertical wall at column = 2
        for r in [1, 2, 3]:
            self.obstacles.add((r, 2))

        # Right-side wall at column = 7 (top part)
        for r in [0, 1, 2]:
            self.obstacles.add((r, 7))

        # Single obstacle in the lower middle
        self.obstacles.add((4, 5))

        # Save initial layout for Blocking Maze modification
        self.initial_obstacles = copy.deepcopy(self.obstacles)

    # ===== Reset =====
    def reset(self):
        self.state = self.start
        self.timestep = 0
        return self.state

    # ===== Step =====
    def step(self, action):
        """
        action:
            0 = up
            1 = down
            2 = left
            3 = right

        Returns:  
            next_state: (row, col)
            reward: 1 if goal reached, else 0
            done: True if goal reached, else False  
        """
        self.timestep += 1

        r, c = self.state

        # Apply action
        if action == 0: r -= 1    # Up
        elif action == 1: r += 1  # Down
        elif action == 2: c -= 1  # Left
        elif action == 3: c += 1  # Right

        next_state = (r, c)

        # Boundary check (stay in place if out of bounds)
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            next_state = self.state

        # Obstacle check (cannot move into obstacle)
        if next_state in self.obstacles:
            next_state = self.state

        #  Update state
        self.state = next_state

        # ===== Blocking Maze change (Figure 8.3) =====
        if self.blocking and self.timestep == 1000:
            self._change_blocking_maze()

        # ===== Reward =====
        if self.state == self.goal:
            return self.state, 1, True
        else:
            return self.state, 0, False

    # ===== Blocking Maze（Figure 8.3）=====
    def _change_blocking_maze(self):
        """
        - The original (short) path is blocked
        - A new longer path is opened

        Implementation:
        - Add obstacle at (3, 0) to block left path
        - Remove obstacle at (0, 7) to open right path
        """

        # Reset to initial layout
        self.obstacles = copy.deepcopy(self.initial_obstacles)

        # Block the original shortest path
        self.obstacles.add((3, 0))  

        # Open a new path on the right
        self.obstacles.remove((0, 7))

        print(">>> Blocking Maze changed at timestep 1000")