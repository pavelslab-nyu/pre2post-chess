#!/bin/bash

# Single 8-GPU multi-turn GRPO run. Normally invoked by sweep_multi_turn.sh,
# which exports the required env vars below. Can also be run standalone after
# exporting them yourself.

# --- required vars (injected via export from sweep_multi_turn.sh) ---
required_vars=(
  MODE BASE_MODEL TRAIN_DATA_PATH EVAL_DATA_PATH CUSTOM_REWARD_PATH
  PROJECT_NAME EXPERIMENT_NAME SAVE_DIR TOTAL_STEPS
)
for v in "${required_vars[@]}"; do
  [[ -z "${!v:-}" ]] && { echo "[ERROR] Missing required env var: ${v}" >&2; exit 1; }
done

# --- optional hyperparams (defaults match sweep_multi_turn.sh) ---
ACTOR_LR="${ACTOR_LR:-1e-5}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-8}"
RES_LENGTH="${RES_LENGTH:-2560}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10000}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2048}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
USE_MULTITURN="${USE_MULTITURN:-True}"
ENABLE_THINKING_MODE="${ENABLE_THINKING_MODE:-True}"

# --- derived paths ---
CHECKPOINT_DIR="${SAVE_DIR}/${EXPERIMENT_NAME}/checkpoints"
LOG_FILE="${SAVE_DIR}/${EXPERIMENT_NAME}/logs/log_${EXPERIMENT_NAME}.log"
MLFLOW_DIR="${SAVE_DIR}/${EXPERIMENT_NAME}/mlflow"
ROLLOUT_DIR="${SAVE_DIR}/${EXPERIMENT_NAME}/rollouts/training"
VALIDATION_DIR="${SAVE_DIR}/${EXPERIMENT_NAME}/rollouts/validation"

# Package root = two levels up from this script (rl/verl/8_gpu_bash -> rl/verl).
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- environment setup ---
source ~/.bashrc
conda activate "${CONDA_ENV:-chess_rl}"

export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export REWARD_MODEL_TYPE="RULE_BASED"
export MLFLOW_TRACKING_URI="file://${MLFLOW_DIR}"

cd "${PKG_ROOT}"
mkdir -p "${SCRATCH}/.triton-cache" "${SCRATCH}/torchinductor"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

mkdir -p "${CHECKPOINT_DIR}" "$(dirname "${LOG_FILE}")" "${MLFLOW_DIR}" "${ROLLOUT_DIR}" "${VALIDATION_DIR}"

echo "Starting RL job (local) for ${EXPERIMENT_NAME}"
echo "Mode=${MODE}  Model=${BASE_MODEL}"
echo "Experiment=${EXPERIMENT_NAME}  total_steps=${TOTAL_STEPS}"

# --- training command ---
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  "data.train_files=${TRAIN_DATA_PATH}" \
  "data.val_files=['${EVAL_DATA_PATH}']" \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=512 \
  "data.max_response_length=${RES_LENGTH}" \
  data.filter_overlong_prompts=True \
  "data.truncation='error'" \
  "+data.use_multiturn=${USE_MULTITURN}" \
  "+data.use_chat_template=False" \
  data.trust_remote_code=True \
  "actor_rollout_ref.model.path=${BASE_MODEL}" \
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  "actor_rollout_ref.rollout.dtype=${ROLLOUT_DTYPE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  "actor_rollout_ref.rollout.n=${GROUP_SIZE}" \
  "actor_rollout_ref.rollout.enforce_eager=False" \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
  "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.interactive_mode.enable=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  reward_model.reward_manager=batch \
  "custom_reward_function.path=${CUSTOM_REWARD_PATH}" \
  custom_reward_function.name=compute_score_batch \
  trainer.critic_warmup=0 \
  trainer.default_hdfs_dir=null \
  "trainer.default_local_dir=${CHECKPOINT_DIR}" \
  "trainer.logger=['console','mlflow','wandb']" \
  "trainer.project_name=${PROJECT_NAME}" \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  "trainer.save_freq=${SAVE_FREQ}" \
  "trainer.test_freq=${TEST_FREQ}" \
  trainer.total_epochs=9999 \
  "trainer.total_training_steps=${TOTAL_STEPS}" \
  "trainer.rollout_data_dir=${ROLLOUT_DIR}" \
  "trainer.validation_data_dir=${VALIDATION_DIR}" \
  "trainer.val_before_train=False" \
  $( [[ "${ENABLE_THINKING_MODE}" == "True" ]] && echo "actor_rollout_ref.rollout.thinking_mode.enable=True" ) \
  2>&1 | tee "${LOG_FILE}"
