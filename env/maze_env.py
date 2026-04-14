import copy

class MazeEnv:
    def __init__(self):
        self.rows = 6
        self.cols = 9

        self.start = (5, 3)
        self.goal = (0, 8)

        self._init_blocking_maze()
        self.reset()

    def _init_blocking_maze(self):
        """
        Blocking Maze:
        A horizontal wall across row 3, with one opening on the left initially.
        After change:
        - left opening is blocked
        - right opening is opened
        """
        self.obstacles = set()

        # horizontal wall at row 3, columns 0..7
        for c in range(8):
            self.obstacles.add((3, c))

        # initial opening at left end
        self.obstacles.remove((3, 0))

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
        Block old opening on the left, open new opening on the right.
        """
        # block old opening
        self.obstacles.add((3, 0))

        # open new opening
        self.obstacles.remove((3, 7))

        print(">>> Blocking Maze changed")