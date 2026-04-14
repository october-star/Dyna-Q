import os
import json
import numpy as np
from datetime import datetime


def create_experiment_dir(base="results", name="experiment"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"{name}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


def save_numpy(path, filename, **kwargs):
    filepath = os.path.join(path, filename)
    np.savez(filepath, **kwargs)


def save_json(path, filename, data):
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def save_plot(path, filename):
    import matplotlib.pyplot as plt
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")