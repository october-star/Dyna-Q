import numpy as np


class UniformDiscretizer:
    def __init__(self, low, high, bins_per_dim):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.bins_per_dim = np.asarray(bins_per_dim, dtype=int)

        if self.low.shape != self.high.shape or self.low.shape != self.bins_per_dim.shape:
            raise ValueError("low, high, and bins_per_dim must have the same shape")

        self.edges = []
        for dim in range(len(self.low)):
            # Internal edges only; np.digitize returns indices in [0, num_bins-1]
            internal_edges = np.linspace(
                self.low[dim],
                self.high[dim],
                self.bins_per_dim[dim] + 1,
            )[1:-1]
            self.edges.append(internal_edges)

    def discretize(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape != self.low.shape:
            raise ValueError("state shape does not match discretizer dimensions")

        indices = []
        for dim, value in enumerate(state):
            clipped = np.clip(value, self.low[dim], self.high[dim])
            index = np.digitize(clipped, self.edges[dim])
            indices.append(int(index))
        return tuple(indices)
