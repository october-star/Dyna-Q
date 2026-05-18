#!/bin/bash
#SBATCH --job-name=deep-dyna-noise
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Dyna-Q}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

# If you use a virtual environment on Ubelix, uncomment and adjust:
# source "$HOME/venvs/dyna-q/bin/activate"

${PYTHON_BIN} -u experiments/run_mountaincar_deep_dyna_noise_ablation.py \
  --episodes 300 \
  --runs 3 \
  --noise-stds 0.0,0.3,0.5,1.0 \
  --planning-steps 5 \
  --planning-start-size 1000 \
  --warmup-steps 1000 \
  --planning-batch-size 64 \
  --model-train-steps 2 \
  --hidden-dims 128,128 \
  --model-hidden-dims 128,128 \
  --epsilon-decay 0.997 \
  --reward-shaping-scale 0.1 \
  --eval-interval 10 \
  "$@"
