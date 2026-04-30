import copy

class MazeEnv:
    def __init__(self, blocking=True):
        self.rows = 6
        self.cols = 9
        self.blocking = blocking

        if self.blocking:
            self.start = (5, 3)
            self.goal = (0, 8)
            self._init_blocking_maze()
        else:
            self.start = (2, 0)
            self.goal = (0, 8)
            self._init_static_maze()

        self.reset()

    def _init_static_maze(self):
        """
        Static Dyna Maze used for the Figure 8.2 style experiments.
        """
        self.obstacles = {
            (1, 2), (2, 2), (3, 2),
            (0, 7), (1, 7), (2, 7),
            (4, 5),
        }
        self.initial_obstacles = copy.deepcopy(self.obstacles)

    def _init_blocking_maze(self):
        """
        Blocking Maze:
        A horizontal wall across row 3, with one opening on the right initially.
        After change:
        - right opening is blocked
        - left opening is opened
        """
        self.obstacles = set()

        # Horizontal wall spans the full row. The initial opening near the goal
        # creates the short path used before the blocking change.
        for c in range(9):
            self.obstacles.add((3, c))

        # Initial opening at the right end
        self.obstacles.remove((3, 8))

        self.initial_obstacles = copy.deepcopy(self.obstacles)

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state

        if action == 0:   # up
            r -= 1
        elif action == 1: # down
            r += 1
        elif action == 2: # left
            c -= 1
        elif action == 3: # right
            c += 1

        next_state = (r, c)

        if not (0 <= r < self.rows and 0 <= c < self.cols):
            next_state = self.state

        if next_state in self.obstacles:
            next_state = self.state

        self.state = next_state

        if self.state == self.goal:
            return self.state, 1, True
        return self.state, 0, False

    def _change_blocking_maze(self):
        """
        Block the old right opening and open a new left opening.
        """
        if not self.blocking:
            raise ValueError("Blocking-maze change is only valid when blocking=True")

        # Block old opening near the goal
        self.obstacles.add((3, 8))

        # Open new opening far from the goal
        self.obstacles.remove((3, 0))

        print(">>> Blocking Maze changed")
