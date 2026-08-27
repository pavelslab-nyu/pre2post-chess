# Chess RL Training

Reinforcement-learning code for post-training chess reasoning models, built on a
fork of [verl](https://github.com/volcengine/verl) (Volcano Engine RL for LLMs).
It runs multi-turn **GRPO** where the policy interacts with a chess environment:
the model reasons, proposes a move, the environment replies, and training
optimizes the resulting trajectories with a rule-based reward.

This release contains the training/eval code only. Datasets and model
checkpoints are distributed separately (see [Data & checkpoints](#data--checkpoints)).

---

## Installation

Requires Python 3.11, CUDA 12.x, and a recent PyTorch (2.4+).

```bash
# from the repository root (this directory)
pip install -e .[vllm]

# FlashAttention (prebuilt wheel matching cu123 / torch2.4 / cp311)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

Notes:
- `pip install -e .[vllm]` installs `verl` in editable mode plus the vLLM rollout
  backend (`vllm<=0.8.5`). Other extras are available: `[sglang]` (alternative
  rollout backend), `[gpu]` (liger-kernel + flash-attn), `[math]`, `[test]`.
- The FlashAttention wheel above is pinned to **cu123 / torch2.4 / cp311**. If
  your CUDA / PyTorch / Python differs, pick the matching wheel from the
  [FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases)
  page instead.
- A conda env is assumed by the launchers (`conda activate chess_rl`); override
  the name with `CONDA_ENV=<your-env>`.

---

## Repository layout

```
.
├── setup.py, pyproject.toml, requirements.txt   # packaging
├── verl/                       # the verl package (fork)
│   ├── trainer/                # main_ppo entrypoint, PPO/GRPO ray trainers
│   │   └── ppo/ray_trainer.py   # RayPPOTrainer + multi-turn rollout dispatch
│   ├── workers/                # FSDP workers, rollout (vllm / sglang)
│   │   └── fsdp_workers.py     # generate_multi_turn_sequences{,_tts,_no_stop}
│   ├── interactions/, tools/   # chess environment interaction
│   ├── reward_function_multiturn.py   # rule-based multi-turn reward
│   ├── reward_function.py             # single-turn reward
│   ├── 8_gpu_bash/             # training launchers (single node, 8 GPUs)
│   └── eval_bash/              # evaluation launchers
```

---

## Quickstart

All launchers live inside the package and resolve paths relative to themselves,
so run them from `verl/`.

### Train (single node, 8 GPUs)

`8_gpu_bash/sweep_multi_turn.sh` launches one GRPO run per pretrain spec. It
resolves each run's SFT starting checkpoint, then calls `run_multi_turn.sh`:

```bash
cd verl/8_gpu_bash
# optional: download the SFT checkpoints used as RL starting points
HF_REPO=<your-hf-org>/sft_trajectory_no_labels bash download_sft_models.sh
# launch the sweep (override data paths / hyperparams via env vars)
WANDB_ENTITY=<your-wandb-entity> bash sweep_multi_turn.sh
```

To run a single configuration directly, export the required vars and call
`run_multi_turn.sh` (see the `required_vars` list at the top of that file).

After training, convert FSDP checkpoints to HuggingFace format:

```bash
LARGE_DIR=../results/rl/<...>/<hparam_tag> bash convert_to_hf_all.sh
```

#### Time estimate

On 8× H200 with the **50M** model, train batch 256, group size 8 (2048
rollouts/step), 2560-token responses: **≈ 30 s/step** (~50 min per 100 steps,
~8 h per 1000). Step time scales roughly linearly with total rollouts
(`batch × group`) and response length, and drifts up as responses grow toward
`RES_LENGTH`. Time a few steps before extrapolating a different config.

**Speed-up tip.** The multi-turn rollout dominates step time, and its cost grows
with the number of sequential env-interaction rounds (each round is one blocking
`vllm.generate` call that waits for the slowest active sample). Capping the
number of rounds is the cheapest lever — add to `run_multi_turn.sh`:

```bash
actor_rollout_ref.rollout.multi_turn.max_env_calls=4 \
```

This bounds each episode to **at most 5 model-generation rounds** (4 env calls +
the final turn) instead of the default 7, accelerating training. Trade-off:
trajectories that legitimately need more than 4 env interactions get cut short,
so verify it against your task before committing.

### Evaluate

`eval_bash/verl_eval.sh` is the single-checkpoint engine;
`run_eval_all_ckps_rl.sh` drives it over many checkpoint steps;
`submit_eval_rl_single.sbatch` is a SLURM wrapper.

```bash
cd verl/eval_bash
# one checkpoint
MODEL_PATH=/path/to/checkpoints_hf_format/global_step_100 \
  EVAL_DATA_DIR=/path/to/eval_data bash verl_eval.sh

# every checkpoint of a run
CHECKPOINT_BASE=/path/to/checkpoints_hf_format \
  STEPS="0 100 200 300" bash run_eval_all_ckps_rl.sh

# on SLURM (set --account in the sbatch header first)
sbatch --export=ALL,CHECKPOINT_BASE=/path/to/checkpoints_hf_format \
  submit_eval_rl_single.sbatch
```

`run_eval_all_ckps_pretrain.sh` is the counterpart for **pretrained base
models**, which have no reasoning phase: it sets `THINKING=False` (so the
rollout is a direct multi-turn game against the chess env, no `<T>` phase),
points `REWARD_FUNCTION` at `reward_function.py` with
`REWARD_TYPE=EVAL_ONLY_NONTHINK_BASED`, and reads the non-thinking eval
parquets. It sweeps `step_<N>/` and `final/` checkpoint directories.

```bash
# every checkpoint of a pretrain run (step_<N> in order, then final)
CHECKPOINT_BASE=/path/to/<pretrain_run> \
  EVAL_DATA_DIR=/path/to/eval_nonthinking bash run_eval_all_ckps_pretrain.sh

# selected checkpoints only
CHECKPOINT_BASE=/path/to/<pretrain_run> \
  STEPS="1006 3018 final" bash run_eval_all_ckps_pretrain.sh
```

---

## Rollout modes

The chess rollout is controlled by three independent toggles. They map to verl
config keys (`actor_rollout_ref.rollout.*`) and, in the launcher scripts, to env
vars. The trainer dispatches to a different generation routine for each
combination (`verl/trainer/ppo/ray_trainer.py`,
`verl/workers/fsdp_workers.py`).

### 1. `interactive_mode` — multi-turn vs. single-shot

- **Enabled (`MULTI_TURN=True`, `interactive_mode.enable=True`)**: the rollout is
  a **multi-turn conversation with the chess environment**. The model generates
  until it emits an environment-call token `<call_env>`; the environment responds (e.g. with
  board state / move feedback); generation resumes. This repeats for up to
  `max_env_calls` rounds. This is the agentic setting and the default for
  training (`generate_multi_turn_sequences`).
- **Disabled**: each prompt gets a single, one-shot generation with no
  environment interaction (`generate_sequences`).

### 2. `thinking_mode` — reason before answering

Only meaningful inside `interactive_mode`.

- **Enabled (`ENABLE_THINKING_MODE=True` / `THINKING=True`,
  `thinking_mode.enable=True`)**: the model produces an explicit reasoning /
  chain-of-thought phase before committing to a move. Dispatches to
  `generate_multi_turn_sequences`. If the thinking mode is enabled, please make sure the prompts end with `<T>`.
- **Disabled**: the model answers directly with no separate thinking phase.
  Dispatches to `generate_multi_turn_sequences_no_stop`.

### 3. `tts_mode` — test-time scaling (fixed thinking budget)

Primarily an **evaluation** knob; requires `thinking_mode` on.

- **Enabled (`TTS=True`, `tts_mode.enable=True`)**: forces a **fixed thinking
  token budget** at inference time. The model must fill
  `THINKING_TOKEN_BUDGET` tokens in the thinking phase before the answer phase
  begins (`generate_multi_turn_sequences_tts` / `generate_sequences_tts`). This
  lets you measure how accuracy scales with the amount of allotted reasoning.
  In TTS mode `verl_eval.sh` reallocates the context window so that
  `answer_length = MODEL_MAX − PROMPT_LEN − THINKING_TOKEN_BUDGET`, keeping the
  total within the model's hard context limit.
- **Disabled**: thinking length is whatever the model naturally produces, capped
  by `RES_LENGTH`.

| Mode | Config key | Launcher env var | Routine |
|------|-----------|------------------|---------|
| interactive | `rollout.interactive_mode.enable` | `MULTI_TURN` / `USE_MULTITURN` | `generate_multi_turn_sequences*` vs `generate_sequences` |
| thinking | `rollout.thinking_mode.enable` | `THINKING` / `ENABLE_THINKING_MODE` | `generate_multi_turn_sequences` vs `..._no_stop` |
| tts | `rollout.tts_mode.enable` | `TTS` (+ `THINKING_TOKEN_BUDGET`) | `generate_multi_turn_sequences_tts` |

---

## Rollout backend

Both training and eval default to **vLLM**. Evaluation can also use **SGLang**
via `ROLLOUT_NAME=sglang` (install with `pip install -e .[sglang]`). When SGLang
is selected, `verl_eval.sh` prepends the pip `nvidia/*/lib` directories to
`LD_LIBRARY_PATH` so SGLang's subprocesses can locate the CUDA runtime.

---

## Data & checkpoints

The `data/` directory and trained checkpoints are **not** included in this code
release. The launchers reference them via overridable variables:

- Training/eval data: `TRAIN_DATA_PATH`, `EVAL_DATA_PATH`, `EVAL_DATA_DIR`
  (expects `*.parquet` files; default location is `verl/data/...`).
- Reward function: `CUSTOM_REWARD_PATH` / `REWARD_FUNCTION`
  (defaults to the shipped `verl/reward_function_multiturn.py`).
- SFT starting checkpoints: download with `8_gpu_bash/download_sft_models.sh`
  (set `HF_REPO`).

Point these at your own copies before launching.
