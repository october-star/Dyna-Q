import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import numpy as np

from experiments.run_mountaincar_deep_dyna import (
    run_deep_dyna_mountaincar_experiment,
)

from experiments.run_mountaincar_ensemble import (
    run_ensemble_experiment,
)

from utils.result_save_util import (
    create_experiment_dir,
    save_json,
    save_numpy,
)


def plot_metric(series_by_label, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))

    for label, values in series_by_label.items():
        plt.plot(values, label=label)

    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def rolling_mean(values, window=5):
    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return values

    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def parse_noise_stds(raw_value):
    return tuple(float(v) for v in raw_value.split(",") if v)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extension B robustness comparison under model noise."
    )

    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument(
        "--noise-stds",
        type=str,
        default="0.3,0.5,1.0",
        help="Comma-separated Gaussian noise std values.",
    )

    parser.add_argument("--lambda-val", type=float, default=1.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    noise_stds = parse_noise_stds(args.noise_stds)

    deep_results_all = {}
    ensemble_results_all = {}

    for noise_std in noise_stds:
        print(f"Running Deep Dyna-Q with noise sigma={noise_std}")

        deep_results = run_deep_dyna_mountaincar_experiment(
            episodes=args.episodes,
            runs=args.runs,
            planning_noise_std=noise_std,
        )

        print(f"Running Ensemble Deep Dyna-Q with noise sigma={noise_std}")

        ensemble_results = run_ensemble_experiment(
            episodes=args.episodes,
            runs=args.runs,
            noise_std=noise_std,
            lambda_val=args.lambda_val,
        )

        deep_results_all[f"deep_sigma_{noise_std}"] = deep_results
        ensemble_results_all[f"ensemble_sigma_{noise_std}"] = ensemble_results

    save_dir = create_experiment_dir(
        name="extension_b_noise_robustness"
    )

    save_json(
        save_dir,
        "config.json",
        {
            "episodes": args.episodes,
            "runs": args.runs,
            "noise_stds": list(noise_stds),
            "lambda_val": args.lambda_val,
            "experiment": "Extension B robustness comparison",
            "agents": [
                "Deep Dyna-Q",
                "Ensemble Deep Dyna-Q",
            ],
        },
    )

    save_numpy(
        save_dir,
        "data.npz",

        **{
            f"deep_returns_sigma_{str(noise).replace('.', '_')}":
                metrics["returns"]
            for noise, metrics in zip(noise_stds, deep_results_all.values())
        },

        **{
            f"ensemble_returns_sigma_{str(noise).replace('.', '_')}":
                metrics["returns"]
            for noise, metrics in zip(noise_stds, ensemble_results_all.values())
        },

        **{
            f"ensemble_disagreement_sigma_{str(noise).replace('.', '_')}":
                metrics["disagreement"]
            for noise, metrics in zip(noise_stds, ensemble_results_all.values())
        },
    )

    plot_metric(
        {
            f"Deep Dyna-Q σ={noise}":
                rolling_mean(
                    deep_results_all[f'deep_sigma_{noise}']["returns"]
                )
            for noise in noise_stds
        },

        ylabel="Rolling Return",
        title="Deep Dyna-Q under Model Noise",
        save_path=os.path.join(
            save_dir,
            "deep_noise_returns.png",
        ),
    )

    plot_metric(
        {
            f"Ensemble σ={noise}":
                rolling_mean(
                    ensemble_results_all[f'ensemble_sigma_{noise}']["returns"]
                )
            for noise in noise_stds
        },

        ylabel="Rolling Return",
        title="Ensemble Deep Dyna-Q under Model Noise",
        save_path=os.path.join(
            save_dir,
            "ensemble_noise_returns.png",
        ),
    )

    plot_metric(
        {
            f"σ={noise}":
                ensemble_results_all[f'ensemble_sigma_{noise}']["disagreement"]
            for noise in noise_stds
        },

        ylabel="Mean disagreement",
        title="Ensemble Disagreement under Noise",
        save_path=os.path.join(
            save_dir,
            "ensemble_disagreement.png",
        ),
    )

    print(f"Success. Results saved in {save_dir}")