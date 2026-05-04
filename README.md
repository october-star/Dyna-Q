# Dyna-Q

Course project for model-based reinforcement learning experiments around:

- Q-Learning
- Dyna-Q
- Dyna-Q+
- DQN baseline
- Deep Dyna-Q

The repository contains two experiment families:

- classic tabular maze experiments inspired by Sutton & Barto Chapter 8
- MountainCar experiments for `Extension A`

## Project Layout

```text
Dyna-Q/
├─ agents/
│  ├─ dqn.py
│  ├─ deep_dyna_q.py
│  ├─ dyna_q.py
│  ├─ dyna_q_plus.py
│  ├─ q_learning.py
│  └─ tabular_dyna_q.py
├─ env/
│  └─ maze_env.py
├─ experiments/
│  ├─ run_blocking_maze.py
│  ├─ run_dyna_maze.py
│  ├─ run_experiments.py
│  ├─ run_mountaincar_tabular_dyna.py
│  ├─ run_mountaincar_dqn.py
│  ├─ run_mountaincar_deep_dyna.py
│  └─ run_mountaincar_comparison.py
├─ results/
├─ test/
└─ utils/
   ├─ discretization.py
   ├─ plotting.py
   └─ result_save_util.py
```

## Requirements

Python 3.10+ is recommended.

Maze experiments only need:

- `numpy`
- `matplotlib`

MountainCar experiments additionally need:

- `gymnasium[classic-control]`
- `torch`

Example install:

```bash
pip install numpy matplotlib torch gymnasium[classic-control]
```

If you use the local virtual environment:

```bash
source env/bin/activate
pip install numpy matplotlib torch gymnasium[classic-control]
```

## Tabular Maze Experiments

These reproduce the classic Dyna results on grid mazes.

### Run the bundled Chapter 8 experiment script

```bash
python experiments/run_experiments.py
```

This script generates:

- Dyna Maze results
- Blocking Maze results
- Q-Learning vs Dyna-Q comparison

Outputs are written to timestamped folders under `results/`.

### Run the static maze script directly

```bash
python experiments/run_dyna_maze.py
```

### Run the blocking maze script directly

```bash
python experiments/run_blocking_maze.py
```

## MountainCar Experiments

These scripts are intended for `Extension A`.

### 1. Tabular Dyna-Q with Bucketing

Runs tabular Dyna-Q on discretized MountainCar states.

```bash
python experiments/run_mountaincar_tabular_dyna.py
```

Useful debug configuration for local CPU:

```bash
python experiments/run_mountaincar_tabular_dyna.py \
  --episodes 30 \
  --runs 1 \
  --planning-steps 5 \
  --bucket-configs 10x10
```

### 2. DQN Baseline

This is a DQN baseline, not Deep Dyna-Q.

```bash
python experiments/run_mountaincar_dqn.py
```

Useful debug configuration:

```bash
python experiments/run_mountaincar_dqn.py \
  --episodes 30 \
  --runs 1 \
  --warmup-steps 200 \
  --batch-size 32 \
  --hidden-dims 64,64
```

### 3. Deep Dyna-Q

This script adds:

- neural world model
- direct Q updates
- model updates
- planning updates
- deterministic evaluation every `N` episodes

```bash
python experiments/run_mountaincar_deep_dyna.py
```

Useful debug configuration:

```bash
python experiments/run_mountaincar_deep_dyna.py \
  --episodes 30 \
  --runs 1 \
  --warmup-steps 200 \
  --planning-steps 2 \
  --planning-batch-size 32 \
  --model-train-steps 1 \
  --hidden-dims 64,64 \
  --model-hidden-dims 64,64
```

### 4. Unified Comparison

Runs:

- Tabular Dyna-Q
- DQN baseline
- Deep Dyna-Q

and saves side-by-side comparison plots.

```bash
python experiments/run_mountaincar_comparison.py
```

Useful debug configuration:

```bash
python experiments/run_mountaincar_comparison.py \
  --episodes 20 \
  --runs 1 \
  --tabular-bins 10x10 \
  --tabular-planning-steps 5 \
  --dqn-hidden-dims 64,64 \
  --dqn-warmup-steps 200 \
  --deep-hidden-dims 64,64 \
  --deep-model-hidden-dims 64,64 \
  --deep-planning-steps 2 \
  --deep-planning-batch-size 32 \
  --deep-model-train-steps 1
```

## Output Structure

Every experiment creates a timestamped directory in `results/`.

Typical contents:

- `config.json`: hyperparameters and run settings
- `data.npz`: saved metrics
- `.png` figures

For MountainCar scripts, common metrics include:

- `steps`
- `returns`
- `success_rate`
- `epsilon`

Deep Dyna-Q also records:

- `direct_q_loss`
- `model_loss`
- `planning_q_loss`
- deterministic evaluation curves

## Notes on Interpretation

### MountainCar is hard

If you see:

- return staying near `-200`
- steps staying near `200`
- success rate near `0`

that does not automatically mean the code is broken. It often means:

- training budget is still too small
- exploration is still high
- the agent has not learned a successful policy yet

### Training vs evaluation reward

The DQN and Deep Dyna-Q MountainCar scripts use reward shaping during training:

- buffer reward may include a velocity-based shaping term
- plotted episode return still uses the original environment reward

This keeps evaluation fair while making training easier.

## Tests

Current repository tests cover the maze environment:

```bash
python -m pytest test/maze_env_test.py
```

## Recommended Workflow

For local CPU:

1. Start with a small debug run.
2. Verify that results are saved correctly.
3. Increase `episodes`, `runs`, and planning settings for formal experiments.

For report-quality MountainCar runs:

- use more episodes
- use multiple seeds
- inspect both training curves and deterministic evaluation curves
