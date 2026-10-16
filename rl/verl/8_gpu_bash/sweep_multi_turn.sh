#!/bin/bash

# Launch a sweep of multi-turn GRPO runs, one per pretrain spec. For each spec
# it resolves the corresponding SFT checkpoint (the RL starting point), then
# calls run_multi_turn.sh with the right env vars.
#
# NOTE: training/eval data and SFT checkpoints are NOT shipped with this code.
# Download them first (see download_sft_models.sh and the project README) or
# point the *_DATA_PATH / SFT_ROOT variables below at your own copies.

set -euo pipefail

# Sync training data from HF if a data preparation script is present (no-op
# otherwise). The data/ directory is not part of this code release.
PREP="$(dirname "$0")/../data/prepare.py"
[[ -f "${PREP}" ]] && python "${PREP}" || echo "[sweep] skipping data prep (${PREP} not found)"

pretrain_specs=(
  "6p5e18|50m|0.100|0.023"
  "6p5e18|50m|1.000|0.023"

  "6p5e19|680m|0.400|0.030"
  "6p5e19|680m|0.750|0.030"
  "6p5e19|680m|0.200|0.030"
)

MODE="multi_turn"
COT_TYPE="${COT_TYPE:-trajectory_sep_no_labels}"

SFT_ROOT="${SFT_ROOT:-../../sft_checkpoints/sft/}"
SAVE_ROOT="${SAVE_ROOT:-../results/rl/${COT_TYPE}}"
RUNNER_SCRIPT="./run_multi_turn.sh"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-../data/train_thinking/train_v4_easy_skewed_multi_turn.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-../data/puzzles/test_multi_turn_final.parquet}"
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-../reward_function_multiturn.py}"
PROJECT_NAME="${PROJECT_NAME:-chess_rl}"

ACTOR_LR="${ACTOR_LR:-1e-5}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-8}"
RES_LENGTH="${RES_LENGTH:-2560}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10000}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2048}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
USE_MULTITURN="${USE_MULTITURN:-True}"
ENABLE_THINKING_MODE="${ENABLE_THINKING_MODE:-True}"

declare -A SFT_CHECKPOINT_SELECTOR_BY_MODEL_ID=(
  # ["C9e18_200m_alpha0.740_beta0.03"]="step_700"
)

model_id_from_spec() {
  printf "C%s_%s_alpha%s_beta%s" "$1" "$2" "$3" "$4"
}

resolve_sft_run_dir() {
  python - "$SFT_ROOT" "$1" "$COT_TYPE" <<'PY'
import pathlib, sys
base = pathlib.Path(sys.argv[1]) / sys.argv[3]
model_id = sys.argv[2]
run_dir = base / model_id
if not run_dir.is_dir():
    raise FileNotFoundError(f"No run dir at {run_dir}")
print(run_dir)
PY
}

resolve_model_path() {
  local run_dir="$1" selector="$2"
  [[ "${selector}" =~ ^[0-9]+$ ]] && selector="step_${selector}"
  local candidate="${run_dir}/${selector}"
  [[ -f "${candidate}/config.json" ]] && { echo "${candidate}"; return 0; }
  [[ -f "${candidate}/hf_model/config.json" ]] && { echo "${candidate}/hf_model"; return 0; }
  echo "[ERROR] No HF model in ${candidate}" >&2
  return 1
}

mkdir -p local_logs

for spec in "${pretrain_specs[@]}"; do
  IFS="|" read -r total_compute modelsize alpha beta <<<"${spec}"
  model_id="$(model_id_from_spec "${total_compute}" "${modelsize}" "${alpha}" "${beta}")"
  selector="${SFT_CHECKPOINT_SELECTOR_BY_MODEL_ID[$model_id]:-final}"
  run_dir="$(resolve_sft_run_dir "${model_id}")"
  base_model="$(resolve_model_path "${run_dir}" "${selector}")"

  hparam_tag="${MODE}_lr${ACTOR_LR}_bs${TRAIN_BATCH_SIZE}_kl${KL_LOSS_COEF}_res${RES_LENGTH}"
  save_dir="${SAVE_ROOT}/${hparam_tag}"
  experiment_name="${model_id}"

  log_file="local_logs/rl_${model_id}_${MODE}.log"

  echo "[RL] launching ${model_id}  steps=${TOTAL_STEPS}  log=${log_file}"

  export MODE BASE_MODEL="${base_model}" SAVE_DIR="${save_dir}" EXPERIMENT_NAME="${experiment_name}" \
    TOTAL_STEPS TRAIN_DATA_PATH EVAL_DATA_PATH CUSTOM_REWARD_PATH PROJECT_NAME \
    ACTOR_LR KL_LOSS_COEF TRAIN_BATCH_SIZE GROUP_SIZE RES_LENGTH SAVE_FREQ TEST_FREQ \
    PPO_MICRO_BATCH_SIZE_PER_GPU MAX_NUM_SEQS ROLLOUT_DTYPE USE_MULTITURN ENABLE_THINKING_MODE

  bash "${RUNNER_SCRIPT}" > "${log_file}" 2>&1 &
done

wait
echo "[RL] All jobs finished."
