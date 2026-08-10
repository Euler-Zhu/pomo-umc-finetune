#!/bin/bash
#SBATCH --job-name=mmmmm_maxgrad
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source config/paths.env

PROJECT="${PROJECT_ROOT}"
CHECKPOINT="${POMO_CHECKPOINT}"

SEED=${SEED:-20260717}
ADDITIONAL_EPOCHS=${ADDITIONAL_EPOCHS:-50}
OUT="${PROJECT}/outputs/mmmmm_maxgrad_m8_seed${SEED}"

cd "${PROJECT}"
source "${VENV}/bin/activate"

export PYTHONPATH="${PROJECT}/scripts:${POMO_TSP_ROOT}/POMO:${POMO_TSP_ROOT}:${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

for required_file in \
  "${CHECKPOINT}" \
  "${PROJECT}/scripts/rbf_soft_clustered_generator.py" \
  "${PROJECT}/scripts/train_pomo100_umc.py"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 2
  fi
done

if [[ -e "${OUT}/checkpoint-latest.pt" || -e "${OUT}/training_history.csv" ]]; then
  echo "Output directory already contains a run: ${OUT}" >&2
  exit 3
fi

python -u scripts/train_pomo100_umc.py \
  --source-checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT}" \
  --pomo-tsp-root "${POMO_TSP_ROOT}" \
  --additional-epochs "${ADDITIONAL_EPOCHS}" \
  --train-episodes 100000 \
  --train-batch-size 64 \
  --candidate-batches 8 \
  --cycle-pattern "maxgrad" \
  --cluster-k-min 2 \
  --cluster-k-max 10 \
  --cluster-alpha 0.8 \
  --rbf-rho 0.35 \
  --save-interval-epochs 10 \
  --log-interval-updates 10 \
  --seed "${SEED}" \
  --expected-source-epoch 2000 \
  --expected-initial-lr 1e-6
