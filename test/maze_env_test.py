import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from env.maze_env import MazeEnv



def test_basic_movement():
    print("=== Test 1: Basic Movement ===")
    env = MazeEnv(blocking=False)
    state = env.reset()

    print("Start:", state)

    # Try moving right 5 steps
    for i in range(5):
        state, reward, done = env.step(3)  # right
        print(f"Step {i+1}: state={state}, reward={reward}, done={done}")


def test_boundary():
    print("\n=== Test 2: Boundary Check ===")
    env = MazeEnv()
    env.reset()

    # Move left at boundary (should stay)
    state, _, _ = env.step(2)  # left
    print("Move left at boundary:", state)


def test_obstacle():
    print("\n=== Test 3: Obstacle Check ===")
    env = MazeEnv()
    env.reset()

    # Move to obstacle position (2,2) from (2,0)
    env.step(3)  # (2,1)
    state, _, _ = env.step(3)  # attempt (2,2) obstacle

    print("Attempt move into obstacle (2,2):", state)


def test_goal():
    print("\n=== Test 4: Goal Reaching ===")
    env = MazeEnv()
    state = env.reset()

    # Manually move to goal (simple path, may hit obstacles)
    actions = [0,0,0,3,3,3,3,3,3,3]  # up + right

    for i, a in enumerate(actions):
        state, reward, done = env.step(a)
        print(f"Step {i+1}: state={state}, reward={reward}, done={done}")
        if done:
            print("Reached goal!")
            break


def test_blocking_maze():
    print("\n=== Test 5: Blocking Maze Change ===")
    env = MazeEnv(blocking=True)
    env.reset()

    for i in range(1001):
        env.step(3)  # keep moving right

        if i == 999:
            print("\nBefore change:")
            print(env.obstacles)

        if i == 1000:
            print("\nAfter change:")
            print(env.obstacles)

    # Check key conditions
    print("\nCheck conditions:")
    print("(3,0) in obstacles:", (3, 0) in env.obstacles)
    print("(0,7) in obstacles:", (0, 7) in env.obstacles)


if __name__ == "__main__":
    test_basic_movement()
    test_boundary()
    test_obstacle()
    test_goal()
    test_blocking_maze()