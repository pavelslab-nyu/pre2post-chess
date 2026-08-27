#!/bin/bash
# =============================================================================
# Run evaluation across all checkpoints of one PRETRAIN run (non-thinking)
# =============================================================================
#
# Pretrained base models have no reasoning phase and no <T>...</T> answer
# format, so this driver differs from run_eval_all_ckps_rl.sh in three ways:
#
#   1. THINKING=False       -> the rollout dispatches to
#                              generate_multi_turn_sequences_no_stop: the model
#                              plays moves directly, with no thinking phase.
#      MULTI_TURN=True      -> the model still interacts with the chess env,
#                              so a rollout is an interleaved sequence of
#                              player moves and environment replies.
#   2. REWARD_FUNCTION      -> reward_function.py (not reward_function_multiturn.py)
#      REWARD_TYPE=EVAL_ONLY_NONTHINK_BASED
#                           -> scores by extracting every complete move from the
#                              raw output (no <T> parsing) and matching the
#                              player-side moves against the ground truth.
#                              NOTE: reward_function.py reads REWARD_MODEL_TYPE
#                              with no default, so this must be exported.
#   3. EVAL_DATA_DIR        -> the non-thinking eval parquets, whose prompts do
#                              NOT end with <T>.
#
# Checkpoint layout. Pretrain runs write plain `step_<N>` directories (not
# `global_step_<N>`) plus a terminal `final/` directory, so STEP_PREFIX defaults
# to "step_". Two modes:
#
#   sweep mode  CHECKPOINT_BASE=<run_dir>          # holds step_<N>/ and/or final/
#   single mode CHECKPOINT_BASE=<run_dir>/step_N   # a model dir (has config.json)
#
# Single mode is auto-detected; STEPS is then ignored.
#
# In sweep mode, STEPS holds checkpoint labels. A numeric label N resolves to
# <run_dir>/step_<N>; any other label resolves to <run_dir>/<label> verbatim,
# which is how `final` is addressed. With STEPS unset, every step_<N> present is
# evaluated in numeric order, followed by `final` if it exists (set
# INCLUDE_FINAL=False to skip it). Runs that only ever wrote `final/` therefore
# work with no arguments.
#
# Usage:
#   CHECKPOINT_BASE=/path/to/6p5e18_20m_llama31_alpha0.050 \
#     bash run_eval_all_ckps_pretrain.sh                      # every step + final
#
#   CHECKPOINT_BASE=/path/to/6p5e18_20m_llama31_alpha0.050 \
#     STEPS="1006 3018 final" bash run_eval_all_ckps_pretrain.sh
#
#   CHECKPOINT_BASE=/path/to/6p5e18_1000m_alpha0.200/final \
#     bash run_eval_all_ckps_pretrain.sh                      # single model dir
#
#   EVAL_DATASETS="test_B1_multi_turn" N_SAMPLES=8 bash run_eval_all_ckps_pretrain.sh
#
# The data/ directory is not shipped with this code release; point
# EVAL_DATA_DIR at your own copy of the non-thinking eval parquets.
#
# =============================================================================

# Don't use set -e - we want to continue even if one checkpoint fails

# Script directory (defined early for path resolution)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"   # = the verl package root (rl/verl)

# Base path to the pretrain checkpoints to evaluate. Either a run directory
# holding step_<N>/ subdirs, or a single model directory. Override via env var.
CHECKPOINT_BASE=${CHECKPOINT_BASE:?"Set CHECKPOINT_BASE to a pretrain run directory (or a single step_<N> directory)"}
CHECKPOINT_BASE=${CHECKPOINT_BASE%/}

# Export settings for verl_eval.sh (can override via env vars).
# Non-thinking models answer directly, so a short response budget is enough.
export RES_LENGTH=${RES_LENGTH:-100}
export TEMPERATURE=${TEMPERATURE:-1}
export N_SAMPLES=${N_SAMPLES:-32}

# -----------------------------------------------------------------------------
#  Resolve checkpoint mode
# -----------------------------------------------------------------------------
# A directory containing config.json is a model, not a run directory.
STEP_PREFIX=${STEP_PREFIX:-"step_"}
INCLUDE_FINAL=${INCLUDE_FINAL:-True}
FINAL_DIR=${FINAL_DIR:-"final"}

# Map a checkpoint label to its directory. Numeric labels are step numbers;
# anything else (notably "final") names a subdirectory directly.
ckpt_path_for() {
    local label="$1"
    if [[ "$label" =~ ^[0-9]+$ ]]; then
        echo "$CHECKPOINT_BASE/${STEP_PREFIX}${label}"
    else
        echo "$CHECKPOINT_BASE/${label}"
    fi
}

if [ -f "$CHECKPOINT_BASE/config.json" ]; then
    SINGLE_MODEL=True
    # eval_results lands next to the checkpoint, i.e. in the run directory
    EXPERIMENT_BASE=$(dirname "$CHECKPOINT_BASE")
    # Label the run from the directory name (step_7005 -> 7005, final -> final)
    STEPS=$(basename "$CHECKPOINT_BASE" | sed "s/^${STEP_PREFIX}//")
else
    SINGLE_MODEL=False
    EXPERIMENT_BASE="$CHECKPOINT_BASE"
    # Checkpoint labels to evaluate (space-separated). Override via STEPS.
    # Default: every step_<N> directory in numeric order, then final/.
    if [ -z "${STEPS:-}" ]; then
        STEPS=$(find "$CHECKPOINT_BASE" -mindepth 1 -maxdepth 1 -type d \
                    -name "${STEP_PREFIX}*" -printf '%f\n' 2>/dev/null \
                | sed "s/^${STEP_PREFIX}//" | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')
        if [ "$INCLUDE_FINAL" = "True" ] && [ -f "$CHECKPOINT_BASE/$FINAL_DIR/config.json" ]; then
            STEPS="$STEPS $FINAL_DIR"
        fi
    fi
    if [ -z "${STEPS// /}" ]; then
        echo "ERROR: no ${STEP_PREFIX}<N> or ${FINAL_DIR}/ checkpoints found under $CHECKPOINT_BASE"
        echo "       Pass STEPS explicitly, or point CHECKPOINT_BASE at a single model directory."
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
#  Auto-generate output directory name from settings
# -----------------------------------------------------------------------------
# Response length short form: 8192 -> 8k, 2560 -> 2560tok, 100 -> 100tok.
# (Non-thinking budgets are typically far below 1024, so integer-dividing by
# 1024 the way the RL driver does would collapse them all to "0k".)
if [ "$RES_LENGTH" -ge 1024 ] && [ $((RES_LENGTH % 1024)) -eq 0 ]; then
    RES_LEN_SHORT=$((RES_LENGTH / 1024))k
else
    RES_LEN_SHORT=${RES_LENGTH}tok
fi
# Convert temperature to string (1 -> t1, 0.6 -> t0_6)
TEMP_STR=$(echo "$TEMPERATURE" | sed 's/\./_/')
# Build output dir name
export OUTPUT_DIR=${OUTPUT_DIR:-"$EXPERIMENT_BASE/eval_results_nonthink_${RES_LEN_SHORT}_t${TEMP_STR}_n${N_SAMPLES}"}

# -----------------------------------------------------------------------------
#  Evaluation settings
# -----------------------------------------------------------------------------
export EVAL_DATASETS=${EVAL_DATASETS:-"test_B0_multi_turn,test_B1_multi_turn,test_B2_multi_turn,test_B3_multi_turn,test_B4_multi_turn,test_B5_multi_turn,random_games_100,human_intermediate_middlegame,human_intermediate_endgame,human_intermediate_opening"}
# Non-thinking eval prompts (these must NOT end with <T>).
export EVAL_DATA_DIR=${EVAL_DATA_DIR:-"$WORKSPACE_DIR/data/eval_nonthinking/"}

# Non-thinking rollout + matching reward. Both are required together: the
# reward parses raw interleaved player/env moves, which is only what the
# no-thinking multi-turn rollout produces.
export MULTI_TURN=${MULTI_TURN:-True}
export THINKING=${THINKING:-False}
export REWARD_FUNCTION=${REWARD_FUNCTION:-"$WORKSPACE_DIR/reward_function.py"}
export REWARD_TYPE=${REWARD_TYPE:-"EVAL_ONLY_NONTHINK_BASED"}

export GPUS=${GPUS:-"0"}
export N_GPUS=${N_GPUS:-1}
export GPU_MEMORY=${GPU_MEMORY:-0.8}
# Short responses mean a small KV-cache footprint, so batch more per GPU.
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}

# Optional post-hoc aggregation over $OUTPUT_DIR. Not shipped with this code
# release - set AGGREGATE_SCRIPT (and optionally AGGREGATE_PYTHON) to enable.
AGGREGATE_SCRIPT=${AGGREGATE_SCRIPT:-""}
AGGREGATE_PYTHON=${AGGREGATE_PYTHON:-"python3"}

if [ ! -d "$EVAL_DATA_DIR" ]; then
    echo "WARNING: EVAL_DATA_DIR does not exist: $EVAL_DATA_DIR"
    echo "         The data/ directory is not shipped with this code release."
fi
if [ ! -f "$REWARD_FUNCTION" ]; then
    echo "ERROR: reward function not found: $REWARD_FUNCTION"
    exit 1
fi

echo "=============================================="
echo "  Pretrain Evaluation Across All Checkpoints"
echo "=============================================="
echo "Run dir:      $CHECKPOINT_BASE"
if [ "$SINGLE_MODEL" = "True" ]; then
echo "Mode:         single model directory"
else
echo "Mode:         sweep over ${STEP_PREFIX}<N> / ${FINAL_DIR} subdirectories"
fi
echo "Checkpoints:  $STEPS"
echo "Datasets:     $EVAL_DATASETS"
echo "Eval data:    $EVAL_DATA_DIR"
echo "Generation:   ${RES_LENGTH} tokens, temp=${TEMPERATURE}, n=${N_SAMPLES}"
echo "Rollout:      multi_turn=$MULTI_TURN thinking=$THINKING"
echo "Reward:       $REWARD_TYPE ($(basename "$REWARD_FUNCTION"))"
echo "GPUs:         $GPUS ($N_GPUS)"
echo "Output:       $OUTPUT_DIR"
echo "=============================================="
echo ""

# Track progress
TOTAL=$(echo "$STEPS" | wc -w)
CURRENT=0
FAILED_STEPS=""
SUCCEEDED_STEPS=""

for STEP in $STEPS; do
    CURRENT=$((CURRENT + 1))

    if [ "$SINGLE_MODEL" = "True" ]; then
        MODEL_PATH="$CHECKPOINT_BASE"
    else
        MODEL_PATH=$(ckpt_path_for "$STEP")
    fi

    echo ""
    echo "=============================================="
    echo "  [$CURRENT/$TOTAL] Evaluating: $(basename "$MODEL_PATH")"
    echo "=============================================="

    if [ ! -d "$MODEL_PATH" ]; then
        echo "WARNING: Checkpoint not found: $MODEL_PATH"
        echo "Skipping..."
        FAILED_STEPS="$FAILED_STEPS $STEP(not_found)"
        continue
    fi

    # Name the result folder ourselves. verl_eval.sh would otherwise derive it
    # by stripping "global_step_", which leaves "step_7005" intact and yields
    # step_step_7005_... for pretrain checkpoints.
    export EXPERIMENT_NAME="step_${STEP}_${RES_LENGTH}len_t${TEMPERATURE}_n${N_SAMPLES}"

    # Run evaluation (continue even if it fails)
    export MODEL_PATH
    if bash "$SCRIPT_DIR/verl_eval.sh"; then
        echo ""
        echo "Completed: $(basename "$MODEL_PATH")"
        SUCCEEDED_STEPS="$SUCCEEDED_STEPS $STEP"
    else
        echo ""
        echo "FAILED: $(basename "$MODEL_PATH")"
        FAILED_STEPS="$FAILED_STEPS $STEP"
    fi

    # Brief pause to allow GPU memory cleanup between runs
    echo "Waiting 10s for GPU cleanup..."
    sleep 10
    echo ""
done

echo ""
echo "=============================================="
echo "  All evaluations complete!"
echo "=============================================="
echo ""
echo "Succeeded:$SUCCEEDED_STEPS"
if [ -n "$FAILED_STEPS" ]; then
    echo "Failed:$FAILED_STEPS"
fi
echo "Results:  $OUTPUT_DIR"
echo ""

if [ -n "$AGGREGATE_SCRIPT" ]; then
    if [ -f "$AGGREGATE_SCRIPT" ]; then
        echo "Running aggregation..."
        "$AGGREGATE_PYTHON" "$AGGREGATE_SCRIPT" "$OUTPUT_DIR"
    else
        echo "WARNING: AGGREGATE_SCRIPT not found: $AGGREGATE_SCRIPT"
    fi
fi
echo ""
