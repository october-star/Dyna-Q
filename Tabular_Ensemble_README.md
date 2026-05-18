# Extension B: Discrete Tabular Ensemble Framework

This directory houses the complete, fully decoupled codebase for **Extension B (Discrete Tabular Ensemble Framework)**. This architecture isolates algorithmic model variance from neural network function approximations by operating within a purely discrete, tabular grid-world landscape (`MazeEnv`).

## 1. Core Architecture

The agent maintains an ensemble of $K = 3$ independent tabular transition models paired explicitly with matching action-value look-up structures ($QS$):
* **Model Set:** $MS = \{M_1, M_2, M_3\}$
* **Q-Table Set:** $QS = \{Q_1, Q_2, Q_3\}$

### Stochastic Planning Update Loop
During each environment step, a live transition tuple $(s, a, r, s')$ updates all $K$ tables to maintain a ground-truth baseline. Internal planning is routed stochastically:
1. Loop through a fixed planning budget: `for i in range(10)`.
2. Sample a random ensemble index: $n \sim \text{Uniform}(1, K)$.
3. Query model $M_n$ for an imagined transition outcome.
4. Execute a tabular Bellman update applied **strictly** to action-value table $Q_n$.

---

## 2. Supported Evaluation & Ablation Tracks

The execution script runs four distinct experimental evaluations sequentially:
1. **Track 1: Baseline Comparison** – Compares our multi-table system against a classical, single-table Tabular Dyna-Q baseline under stationary grid parameters.
2. **Track 2: Action-Value Variance Tracking** – Monitors global ensemble disagreement $u(s)$ (mean absolute variance across discovered coordinate fields).
3. **Track 3: Transition Noise Ablation** – Corrupts model prediction outcomes during simulated planning steps with a failure probability $\sigma \in \{0.3, 0.5, 1.0\}$ to evaluate performance degradation profiles.
4. **Track 4: Multi-Agent Policy Ablation (500 Episodes)** – Contrasts **Averaged Consensus Action Selection** ($\sum Q_k$) against **Independent Expert Selection** (where a single random index is fielded to completely guide each step without pooling) over an extended long-horizon window.

---

## 3. Repository File Blueprint

Your implementation uses these specific isolated files:
```text
├── agents/
│   └── tabular_ensemble_dyna.py             # Core K-Model/K-Table ensemble agent logic
├── env/
│   └── maze_env.py                          # Stationary and non-stationary grid layout
└── experiments/
    ├── run_tabular_ensemble_experiments.py  # Comprehensive multi-track ablation runner
    └── README.md                            # This documentation asset
