#!/bin/bash
#SBATCH --job-name=pomo_eval
#SBATCH --partition=common-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source config/paths.env

PROJECT="${PROJECT_ROOT}"
BASE_CHECKPOINT="${POMO_CHECKPOINT}"

MODEL_NAME=${MODEL_NAME:?Please provide MODEL_NAME}
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:?Please provide MODEL_CHECKPOINT}

SEED=${SEED:-20260719}
START_ROW=${START_ROW:-0}
NUM_INSTANCES=${NUM_INSTANCES:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
BOOTSTRAP_REPS=${BOOTSTRAP_REPS:-5000}

OUT="${PROJECT}/outputs/eval_${MODEL_NAME}_seed${SEED}"

cd "${PROJECT}"
source "${VENV}/bin/activate"

export PYTHONPATH="${PROJECT}/scripts:${POMO_TSP_ROOT}/POMO:${POMO_TSP_ROOT}:${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

for required_file in \
  "${BASE_CHECKPOINT}" \
  "${MODEL_CHECKPOINT}" \
  "${PROJECT}/scripts/eval_std_pgd_generalization.py"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 2
  fi
done

if [[ ! -d "${PROJECT}/data/data_farm" ]]; then
  echo "Missing data directory." >&2
  exit 3
fi

if [[ ! -d "${PROJECT}/data/solution_farm" ]]; then
  echo "Missing solution directory." >&2
  exit 3
fi

echo "=============================================="
echo "POMO evaluation"
echo "Model name:       ${MODEL_NAME}"
echo "Model checkpoint: ${MODEL_CHECKPOINT}"
echo "Base checkpoint:  ${BASE_CHECKPOINT}"
echo "Output:           ${OUT}"
echo "=============================================="

python -u scripts/eval_std_pgd_generalization.py \
  --pomo-dir "${POMO_TSP_ROOT}/POMO" \
  --data-dir "${PROJECT}/data/data_farm" \
  --solution-dir "${PROJECT}/data/solution_farm" \
  --output-dir "${OUT}" \
  --checkpoint "base2000=${BASE_CHECKPOINT}" \
  --checkpoint "${MODEL_NAME}=${MODEL_CHECKPOINT}" \
  --reference-model base2000 \
  --start-row "${START_ROW}" \
  --num-instances "${NUM_INSTANCES}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda:0 \
  --seed "${SEED}" \
  --bootstrap-reps "${BOOTSTRAP_REPS}"

echo "Evaluation complete."
echo "Results written to:"
echo "${OUT}"
