# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import json
import logging
import os
import warnings
from dataclasses import asdict
from types import MethodType
from typing import Any, Dict, List, Optional, Union

import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.debug.performance import reduce_timing
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
import numpy as np
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def _left_pad_2d(seqs: list[list[int]], pad_id: int, device: torch.device) -> torch.Tensor:
    """Left-pad a list of token sequences into a 2-D tensor.

    Example (pad_id=0):
        [[1,2,3], [4,5]] -> [[0,1,2,3], [0,0,4,5]]
    """
    max_len = max((len(s) for s in seqs), default=1)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        if s:
            out[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long, device=device)
    return out


def _extract_response_tokens(row: torch.Tensor, eos_id: int, pad_id: int) -> list[int]:
    """Strip leading/trailing pads and stop (inclusive) at first EOS."""
    return _extract_response_tokens_and_logprobs(row, None, eos_id=eos_id, pad_id=pad_id)[0]


def _extract_response_tokens_and_logprobs(
    row: torch.Tensor, logprob_row, eos_id: int, pad_id: int
) -> tuple[list[int], list[float] | None, int]:
    """_extract_response_tokens, carrying each kept token's rollout log-prob along with it.

    Both must filter identically, so _extract_response_tokens is this with logprob_row=None (log-probs then
    come back as None). Also counts *generated* tokens dropped by the pad rule: vLLMRollout pads
    rollout_log_probs with -1, so a skipped position holding a real log-prob was sampled by the model
    (e.g. <bos> when pad_id == bos_id).
    """
    lps = logprob_row.tolist() if logprob_row is not None else None
    tokens, logprobs, n_dropped = [], ([] if lps is not None else None), 0
    for k, t in enumerate(row.tolist()):
        if t == pad_id or t == 0: # patch
            if lps is not None and lps[k] != -1.0:
                n_dropped += 1
            continue  # skip pads (left-padded output may have them at start)
        tokens.append(t)
        if lps is not None:
            logprobs.append(lps[k])
        if t == eos_id:
            break
    return tokens, logprobs, n_dropped


def _head(logprobs: list[float] | None, n: int) -> list[float] | None:
    """logprobs[:n], or None when log-probs are not being collected. Sliced in lockstep with the tokens."""
    return None if logprobs is None else logprobs[:n]


def _rollout_log_probs_tensor(states, policy_mask: torch.Tensor, length: int, device) -> torch.Tensor:
    """Right-pad each sample's per-token rollout log-probs to [B, length] (float32).

    Env/opponent tokens hold 0.0 -- they are masked out of the loss by response_mask (= policy_mask). A model
    token whose log-prob was never recorded holds NaN, which the check below turns into an error instead of a
    silently wrong PPO ratio.
    """
    out = torch.zeros((len(states), length), dtype=torch.float32, device=device)
    for i, s in enumerate(states):
        assert len(s.response_logprobs) == len(s.response_tokens), f"sample {i}: log-probs out of step with tokens"
        lp = s.response_logprobs[:length]
        if lp:
            out[i, : len(lp)] = torch.tensor(lp, dtype=torch.float32, device=device)
    policy = policy_mask.bool()
    assert torch.isfinite(out[policy]).all(), "a model-generated token has no rollout log-prob"
    assert (out[policy] <= 1e-3).all(), "rollout log-probs must be <= 0"
    return out


def _enable_fp32_lm_head(model: torch.nn.Module) -> None:
    """Compute the LM head (output projection) in fp32 on the trainer side.

    dp_actor runs the forward under torch.autocast(bf16), and autocast casts F.linear operands back to bf16,
    so upcasting the operands alone is a no-op (the earlier, unconditional version of this patch was).
    Autocast is therefore disabled inside the head. The weights stay bf16 under FSDP mixed precision and are
    upcast at matmul time, exactly like the vLLM side (verl.utils.vllm_utils.patch_vllm_fp32_lm_head), so
    the head adds no trainer<->rollout mismatch. Only `forward` is replaced: module structure, state-dict
    keys and FSDP wrapping are unchanged, and tied embeddings keep sharing one parameter.
    """
    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear):
        raise ValueError(f"fp32_lm_head expects the output embedding to be nn.Linear, got {type(head).__name__}")

    def _fp32_forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.nn.functional.linear(
                x.float(), self.weight.float(), None if self.bias is None else self.bias.float()
            )

    head.forward = MethodType(_fp32_forward, head)


# ---------------------------------------------------------------------------
# Move-boundary helper (no-stop-token multi-turn)
# ---------------------------------------------------------------------------

def _find_move_token_boundary(tokens: list[int], tokenizer) -> tuple:
    """Find the earliest token boundary that completes a chess move.

    Decodes token prefixes one token at a time and watches for newly-formed
    whitespace-delimited words that satisfy ``_is_complete_move``.  This
    correctly handles moves that span multiple sub-word tokens (e.g. ``Pd2``
    + ``d4`` → ``Pd2d4``).

    Returns:
        (end_idx, move_str) where ``tokens[:end_idx]`` is the shortest prefix
        containing a complete move, or ``(None, None)`` if none found.
    """
    from verl.reward_function import _is_complete_move
    prev_word_set: set = set()
    for end in range(1, len(tokens) + 1):
        text = tokenizer.decode(tokens[:end], skip_special_tokens=False)
        word_set = set(text.split())
        for word in word_set - prev_word_set:   # only newly-completed words
            if _is_complete_move(word):
                return end, word
        prev_word_set = word_set
    return None, None


# ---------------------------------------------------------------------------
# DataProto builder
# ---------------------------------------------------------------------------

def _build_dataproto(
    token_seqs: list[list[int]],
    pad_id: int,
    eos_id: int,
    device: torch.device,
) -> DataProto:
    """Build a left-padded DataProto from raw (unpadded) token sequences."""
    input_ids = _left_pad_2d(token_seqs, pad_id, device)
    B, L = input_ids.shape

    attention_mask = (input_ids != pad_id).long()

    # Position ids: pads get 0, real tokens get 0..seq_len-1
    position_ids = torch.zeros_like(input_ids)
    for i in range(B):
        n_real = int(attention_mask[i].sum().item())
        if n_real:
            position_ids[i, L - n_real:] = torch.arange(n_real, device=device)

    batch = TensorDict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
        batch_size=B,
    )
    dp = DataProto(
        batch=batch,
        non_tensor_batch={"raw_prompt_ids": np.array(token_seqs, dtype=object)},
    )
    dp.meta_info = {"eos_token_id": eos_id, "pad_token_id": pad_id, "do_sample": True}
    return dp


# ---------------------------------------------------------------------------
# Per-sample state
# ---------------------------------------------------------------------------

class _SampleState:
    """Tracks the rolling context and accumulated output for one sample."""

    def __init__(self, prompt_ids: list[int], env_replies=None):
        self.ctx: list[int] = list(prompt_ids)   # full context fed to the model each round
        self.response_tokens: list[int] = []      # final response (model + env tokens)
        self.policy_mask: list[int] = []          # 1=model token, 0=env token
        self.n_model_tokens: int = 0              # model-generated tokens so far
        self.n_env_calls: int = 0
        self.finished: bool = False
        self.prompt_token_count: int = len(prompt_ids)  # prompt size in tokens
        self.env_replies_used: list = []          # env reply texts actually consumed
        self.response_logprobs: list[float] = []  # rollout log-prob per response token (0.0 for env tokens)
        # Pre-recorded env replies queue; popped one-by-one each env call
        if isinstance(env_replies, list):
            self.env_replies: list = list(env_replies)
        elif env_replies is not None:
            self.env_replies = [env_replies]
        else:
            self.env_replies = []

    def append_model(self, tokens: list[int], logprobs: list[float] | None = None) -> None:
        self.ctx.extend(tokens)
        self.response_tokens.extend(tokens)
        self.policy_mask.extend([1] * len(tokens))
        self.n_model_tokens += len(tokens)
        if logprobs is None:
            # Not collected. NaN, so _rollout_log_probs_tensor rejects it if collection was expected.
            self.response_logprobs.extend([float("nan")] * len(tokens))
        else:
            assert len(logprobs) == len(tokens), "one rollout log-prob per model token"
            self.response_logprobs.extend(logprobs)

    def append_env(self, tokens: list[int]) -> None:
        self.ctx.extend(tokens)
        self.response_tokens.extend(tokens)
        self.policy_mask.extend([0] * len(tokens))
        self.response_logprobs.extend([0.0] * len(tokens))  # masked out of the loss
        # env tokens do NOT count toward n_model_tokens
    

def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        Worker.__init__(self)

        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        profiler_config: Optional[ProfilerConfig] = None
        if self._is_actor:
            profiler_config = omega_conf_to_dataclass(config.actor.get("profiler", {}), ProfilerConfig)
        if self._is_rollout:
            profiler_config = omega_conf_to_dataclass(config.rollout.get("profiler", {}), ProfilerConfig)
        if self._is_ref:
            profiler_config = omega_conf_to_dataclass(config.ref.get("profiler", {}), ProfilerConfig)

        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=profiler_config))

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            else:
                actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            # Opt-in fp32 LM head for the actor and the ref model (both built here), matched with the vLLM
            # side. With the flag off nothing is patched, which is numerically identical to the old
            # unconditional patch: under dp_actor's bf16 autocast that one never took effect.
            if self.config.model.get("fp32_lm_head", False):
                if use_fused_kernels:
                    # The fused kernel computes log-probs from the hidden states itself and never calls the head.
                    raise ValueError("actor_rollout_ref.model.fp32_lm_head does not support use_fused_kernels=True")
                _enable_fp32_lm_head(actor_module)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )

        if self._is_rollout and self.config.rollout.name == "hf":
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.config.actor.fsdp_config.get("use_orig_params", False),
                forward_prefetch=self.config.actor.fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
                eps=optim_config.get("eps", 1e-8),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name
        if rollout_name == "hf":
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager.base import BaseShardingManager

            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        elif rollout_name == "vllm":
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
            lora_kwargs = (
                {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}}
                if self._is_lora
                else {}
            )
            # lora_kwargs = {}
            from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

            vllm_rollout_cls = vLLMRollout if self.config.rollout.mode == "sync" else vLLMAsyncRollout
            rollout = vllm_rollout_cls(
                model_path=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=trust_remote_code,
                fp32_lm_head=self.config.model.get("fp32_lm_head", False),
                **lora_kwargs,
            )

            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name == "sglang":
            from verl.workers.rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                multi_stage_wake_up=self.config.rollout.multi_stage_wake_up,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = DataParallelPPOActor(
                config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red")
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                if "kwargs" in prompts.meta_info:
                    kwargs = prompts.meta_info["kwargs"]
                    output = self.rollout.generate_sequences(prompts=prompts, **kwargs)
                else:
                    output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red")
    def generate_multi_turn_sequences(self, prompts: DataProto):
        # Support all hardwares
        print("Generating multi-turn sequences")
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        pad_id: int = (
            self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id
        )
        if pad_id is None:
            pad_id = self.tokenizer.bos_token_id

        eos_id: int = (
        self.generation_config.eos_token_id
        if self.generation_config is not None
        else self.tokenizer.eos_token_id
    )
        call_env_id: int = self.tokenizer._convert_token_to_id(self.tokenizer.env_token)
        assert call_env_id is not None, "call_env_id is not in vocab"
        # print(f"[DEBUG multi-turn] env_token={self.tokenizer.env_token!r}  call_env_id={call_env_id}  eos_id={eos_id}  pad_id={pad_id}")
        # print(f"[DEBUG multi-turn] unk_token_id={self.tokenizer.unk_token_id}  — if call_env_id==unk, env_token is not in vocab!")

        meta_info = {
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
        }

        prompts.meta_info.update(meta_info)

        # ---- hyperparams ------------------------------------------------------
        max_env_calls: int = int(
            self.config.rollout.multi_turn.get("max_env_calls", 6)
        )
        max_model_tokens: int = int(
            self.config.rollout.get("response_length", 1024)
        )
        # round_max_tokens: int = int(
        #     self.config.rollout.multi_turn.get("round_max_tokens", 128)
        # )
        # Carry vLLM's sampled-token log-probs through the turns (rollout.calculate_log_probs).
        collect_logprobs = bool(self.config.rollout.get("calculate_log_probs", False))
        n_dropped_generated = 0
        # vLLM rejects (V0: raises on) a request longer than max_model_len. Opponent replies are added outside
        # the model-token budget, so a long prompt plus a near-full budget can overflow it; such samples stop
        # taking turns instead (same limit as vLLMRollout).
        max_ctx_len: int = int(
            self.config.rollout.get("max_model_len", None) or self.config.rollout.prompt_length + max_model_tokens
        )

        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            # preprocess_data allgathers across TP ranks so every rank sees the
            # full batch. States must be built AFTER this so all TP ranks have
            # identical states and call generate_sequences the same number of
            # times (required for TP-collective correctness).
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            base_kwargs = dict(prompts.meta_info.get("kwargs", {}))

            # Expand by n for training only — validation uses n=1 (no repeat).
            # We do the repeat here manually so that each generate_sequences call
            # inside the multi-turn loop uses n=1 and does NOT double-expand.
            is_validate = prompts.meta_info.get("validate", False)
            n = 1 if is_validate else int(self.config.rollout.get("n", 1))
            if n > 1:
                prompts = prompts.repeat(repeat_times=n, interleave=True)

            # ---- init per-sample state (full allgathered batch) ---------------
            if "raw_prompt_ids" in prompts.non_tensor_batch:
                prompt_ids_list = [list(x) for x in prompts.non_tensor_batch["raw_prompt_ids"]]
            else:
                # Fallback: strip left-padding from input_ids
                prompt_ids_list = []
                for row in prompts.batch["input_ids"]:
                    ids = row.tolist()
                    prompt_ids_list.append([t for t in ids if t != pad_id])

            raw_env_replies = prompts.non_tensor_batch.get("env_replies", None)
            states = [
                _SampleState(ids, env_replies=raw_env_replies[i] if raw_env_replies is not None else None)
                for i, ids in enumerate(prompt_ids_list)
            ]

            # Sort by number of env replies descending so samples with similar
            # trajectory lengths are batched together.  Samples that need more
            # env calls stay active longer; grouping them reduces the number of
            # rounds where only a handful of stragglers keep the loop alive.
            # We unsort before the final assembly so output order matches prompts.
            sort_order = sorted(range(len(states)), key=lambda i: len(states[i].env_replies), reverse=True)
            states = [states[i] for i in sort_order]

            # Pre-tokenize all env replies up-front to avoid repeated tokenizer
            # calls inside the hot generation loop.
            _tok_cache: dict[str, list[int]] = {}
            if raw_env_replies is not None:
                all_replies = [r for replies in raw_env_replies if replies for r in (replies if isinstance(replies, list) else [replies])]
                if all_replies:
                    encoded = self.tokenizer(all_replies, add_special_tokens=False).input_ids
                    for text, ids in zip(all_replies, encoded):
                        _tok_cache[text] = ids

            with simple_timer("generate_multi_turn_sequences", timing_generate):
                while True:
                    active_idx = [
                        i for i, s in enumerate(states)
                        if not s.finished
                        and s.n_env_calls < max_env_calls
                        and s.n_model_tokens < max_model_tokens
                        and len(s.ctx) < max_ctx_len
                    ]
                    # print(f"[DEBUG multi-turn] length of active_idx={len(active_idx)}")
                    if not active_idx:
                        break

                    # Build batch for active samples
                    active_states = [states[i] for i in active_idx]
                    dp_active = _build_dataproto(
                        [s.ctx for s in active_states],
                        pad_id=pad_id,
                        eos_id=eos_id,
                        device=torch.device(get_device_id()),
                    )

                    # Per-round token budget: respect both round cap and remaining global budget
                    round_tokens = max_model_tokens - min(s.n_model_tokens for s in active_states)

                    seg_kwargs = {
                        **base_kwargs,
                        "stop_token_ids": [call_env_id],
                        "max_tokens": max(round_tokens, 1),
                        "n": 1,  # batch already expanded above; prevent vLLM from re-expanding
                    }
                    # print(f"[DEBUG multi-turn] round seg_kwargs={seg_kwargs}  active={len(active_idx)} samples")

                    out = self.rollout.generate_sequences(prompts=dp_active, **seg_kwargs)
                    # print(f"[DEBUG multi-turn] out.batch keys={list(out.batch.keys())}")
                    out_logprobs = out.batch["rollout_log_probs"] if collect_logprobs else None

                    # Process each active sample's output
                    for j, i in enumerate(active_idx):
                        state = states[i]
                        raw_row = out.batch["responses"][j]
                        tokens, logprobs, n_drop = _extract_response_tokens_and_logprobs(
                            raw_row, None if out_logprobs is None else out_logprobs[j], eos_id=eos_id, pad_id=pad_id
                        )
                        n_dropped_generated += n_drop
                        # print(f"[DEBUG multi-turn] tokens={tokens}")
                        # decoded = self.tokenizer.decode(tokens) #, skip_special_tokens=False
                        # print(f"[DEBUG multi-turn] decoded={decoded}")
                        # print(f"[DEBUG multi-turn] sample={i} j={j} n_tokens={len(tokens)} #calls={state.n_env_calls} hit_call_env={call_env_id in tokens} decoded={decoded!r:.200}")

                        if not tokens:
                            state.finished = True
                            continue

                        # Clip to this sample's own remaining budget (round_tokens uses the
                        # minimum across all active samples, which over-budgets samples that
                        # have already used more tokens).
                        remaining = max_model_tokens - state.n_model_tokens
                        # print(f"[DEBUG multi-turn] remaining={remaining}")
                        if remaining < len(tokens):
                            print(f"[warning multi-turn] tokens length={len(tokens)} is greater than remaining={remaining}")
                        tokens = tokens[:remaining]
                        logprobs = _head(logprobs, remaining)
                        if not tokens:
                            state.finished = True
                            continue

                        hit_eos = tokens[-1] == eos_id
                        hit_call = call_env_id in tokens
                        # print(f"[DEBUG multi-turn] hit_eos={hit_eos} hit_call={hit_call}")

                        if hit_call:
                            # Keep everything up to and including <call_env>
                            cut = tokens.index(call_env_id) + 1
                            state.append_model(tokens[:cut], _head(logprobs, cut))
                            state.n_env_calls += 1

                            # Get environment observation
                            # TODO: replace with real env.step(state.ctx) for chess
                            # print(f"[DEBUG multi-turn] state.env_replies={state.env_replies}")
                            obs_text = state.env_replies.pop(0) if state.env_replies else None
                            # print(f"[DEBUG multi-turn] <call_env> triggered (call #{state.n_env_calls}) obs_text (after parsing)={obs_text!r}")
                            if obs_text:
                                state.env_replies_used.append(obs_text)
                            if not obs_text:
                                # print(f"[DEBUG multi-turn] no env reply available — finishing sample {i}")
                                state.finished = True
                            else:
                                if obs_text not in _tok_cache:
                                    _tok_cache[obs_text] = self.tokenizer(obs_text, add_special_tokens=False).input_ids
                                obs_ids = _tok_cache[obs_text]
                                # print(f"[DEBUG multi-turn] obs_text={obs_text} obs_ids={obs_ids}")
                                if obs_ids:
                                    state.append_env(obs_ids)
                                else:
                                    print(f"[DEBUG multi-turn] WARNING: tokenizer returned empty obs_ids for {obs_text!r}")
                                # env call succeeded — continue to next generation round
                                # new_ctx = self.tokenizer.decode(state.ctx, skip_special_tokens=False)
                                # print(f"[DEBUG multi-turn] new_ctx: {new_ctx!r}")
                                # print(f"[DEBUG multi-turn] env reply applied, continuing generation for sample {i}")

                            if state.n_env_calls >= max_env_calls:
                                # print(f"[DEBUG multi-turn] max_env_calls={max_env_calls} reached for sample {i}")
                                state.finished = True
                        else:
                            state.append_model(tokens, logprobs)
                            # if hit_eos:
                                # print(f"[DEBUG multi-turn] EOS hit for sample {i}, finishing")
                            # else:
                                # print(f"[DEBUG multi-turn] no <call_env> and no EOS for sample {i} (length/other stop), finishing")
                            state.finished = True

            log_gpu_memory_usage("After rollout generation", logger=logger)
            if n_dropped_generated:
                print(f"[warning multi-turn] dropped {n_dropped_generated} generated pad-id tokens from responses; "
                      f"tokens after them in the same turn were sampled with them in context")
            n_ctx_limited = sum(1 for s in states if not s.finished and len(s.ctx) >= max_ctx_len)
            if n_ctx_limited:
                print(f"[warning multi-turn] {n_ctx_limited} samples stopped at vLLM max_model_len={max_ctx_len}")

            # Restore original sample order so states aligns with prompts.batch
            restored = [None] * len(states)
            for new_i, old_i in enumerate(sort_order):
                restored[old_i] = states[new_i]
            states = restored

            # ---- assemble final outputs ---------------------------------------
            resp_len = max((len(s.response_tokens) for s in states), default=1)
            # Synchronize resp_len across DP workers so all shards pad to the
            # same length — otherwise DataProto.concat fails on dim-1 mismatch.
            device = prompts.batch["input_ids"].device
            resp_len_tensor = torch.tensor(resp_len, dtype=torch.long, device=device)
            dist.all_reduce(resp_len_tensor, op=dist.ReduceOp.MAX)
            resp_len = int(resp_len_tensor.item())
            # Pad responses and policy masks (right-pad is fine here — these are outputs, not inputs)

            def _right_pad(seqs, pad, length, device):
                out = torch.full((len(seqs), length), pad, dtype=torch.long, device=device)
                for i, s in enumerate(seqs):
                    t = s[:length]
                    if t:
                        out[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
                return out

            responses   = _right_pad([s.response_tokens for s in states], pad_id, resp_len, device)
            policy_mask = _right_pad([s.policy_mask      for s in states], 0,      resp_len, device)
            response_mask = (responses != pad_id).to(prompts.batch["attention_mask"].dtype)
            policy_mask   = policy_mask.to(prompts.batch["attention_mask"].dtype)

            assert not (policy_mask & ~response_mask).any(), \
                "policy_mask has 1s outside valid response tokens"

            assert (policy_mask.sum(dim=-1) > 0).all(), \
                "some samples have no model tokens in policy_mask"

            assert (response_mask.sum(dim=-1) >= policy_mask.sum(dim=-1)).all(), \
                "response_mask should cover at least as many tokens as policy_mask"

            # 4. pads appear only at the end (right-pad contract)
            #    once we see a pad, everything after must also be pad
            for i, s in enumerate(states):
                resp_len_i = len(s.response_tokens)
                if resp_len_i < resp_len:
                    assert (responses[i, resp_len_i:] == pad_id).all(), \
                        f"sample {i}: non-pad token found after sequence end (right-pad violated)"

            # 5. policy_mask values are binary (0 or 1 only)
            assert ((policy_mask == 0) | (policy_mask == 1)).all(), \
                "policy_mask contains values other than 0 and 1"

            # 6. response_mask values are binary
            assert ((response_mask == 0) | (response_mask == 1)).all(), \
                "response_mask contains values other than 0 and 1"

            # 7. env tokens exist only where response_mask=1 and policy_mask=0
            env_mask = response_mask & ~policy_mask
            n_env = env_mask.sum(dim=-1)
            for i, s in enumerate(states):
                expected_env = len(s.response_tokens) - sum(s.policy_mask)
                assert n_env[i].item() == expected_env, \
                    f"sample {i}: env token count mismatch — tensor={n_env[i].item()}, state={expected_env}"
            # ------------------------------------------------------------------

            # Stitch prompt + response
            prompt_ids    = prompts.batch["input_ids"]
            prompt_attn   = prompts.batch["attention_mask"]
            prompt_pos    = prompts.batch["position_ids"]

            input_ids     = torch.cat([prompt_ids, responses], dim=-1)
            attention_mask = torch.cat([prompt_attn, response_mask], dim=-1)

            # Response position ids continue from the last prompt position
            delta = torch.arange(1, resp_len + 1, device=device).unsqueeze(0).expand(len(states), -1)
            if prompt_pos.dim() == 3:  # Qwen2-VL mRoPE
                delta = delta.view(len(states), 1, -1).expand(len(states), 3, -1)
            response_pos = prompt_pos[..., -1:] + delta
            position_ids = torch.cat([prompt_pos, response_pos], dim=-1)

            batch = TensorDict(
                {
                    "prompts":        prompt_ids,
                    "responses":      responses,
                    "input_ids":      input_ids,
                    "attention_mask": attention_mask,
                    "position_ids":   position_ids,
                    "response_mask":  policy_mask,   # mask env reply tokens from loss; policy_mask=1 for model tokens, 0 for env tokens
                    "policy_mask":    policy_mask,
                },
                batch_size=len(states),
            )
            if collect_logprobs:
                batch["rollout_log_probs"] = _rollout_log_probs_tensor(states, policy_mask, resp_len, device)
            non_tensor = dict(prompts.non_tensor_batch)
            non_tensor["n_env_calls"] = np.array([s.n_env_calls for s in states], dtype=np.int32)
            non_tensor["max_env_calls_used"] = np.array([max_env_calls] * len(states), dtype=np.int32)
            non_tensor["env_replies_used"] = np.array([s.env_replies_used for s in states], dtype=object)
            non_tensor["prompt_token_count"] = np.array([s.prompt_token_count for s in states], dtype=np.int32)
            non_tensor["model_token_count"] = np.array([s.n_model_tokens for s in states], dtype=np.int32)
            non_tensor["env_token_count"] = np.array(
                [len(s.response_tokens) - s.n_model_tokens for s in states], dtype=np.int32
            )
            non_tensor["total_response_token_count"] = np.array(
                [len(s.response_tokens) for s in states], dtype=np.int32
            )
            output = DataProto(batch=batch, non_tensor_batch=non_tensor)
            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red")
    def generate_sequences_tts(self, prompts: DataProto):
        """Test-time scaling (TTS) generation.

        Phase 1 — thinking: generate until </T>.  If the token budget is not
        yet exhausted, swap the </T> for <sep> and feed back so the model
        continues thinking.  When the budget is consumed (or the model emits
        </T> at/past the limit), commit </T> and transition to Phase 2.

        Phase 2 — answer: run the same multi-turn generation loop as
        ``generate_multi_turn_sequences`` (env calls via <call_env>) for up
        to ``answer_length`` additional model tokens.

        Config keys (all under ``config.rollout.multi_turn``):
          - thinking_budget  (int, default 1024): max model tokens in thinking phase
          - max_env_calls    (int, default 6):    env-call limit in answer phase
        And ``config.rollout.response_length`` (int, default 1024): answer-phase budget.
        """
        print("Generating sequences with TTS (test-time scaling)")
        prompts = prompts.to(get_device_id())
        assert self._is_rollout

        pad_id: int = (
            self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id
        )
        if pad_id is None:
            pad_id = self.tokenizer.bos_token_id

        eos_id: int = (
            self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id
        )

        # Special tokens used for TTS
        t_end_id: int = self.tokenizer.t_end_id()   # </T>
        sep_id: int   = self.tokenizer.sep_id()     # <sep>
        call_env_id: int = self.tokenizer._convert_token_to_id(self.tokenizer.env_token)
        assert t_end_id is not None, "</T> token not in vocab"
        assert sep_id   is not None, "<sep> token not in vocab"
        assert call_env_id is not None, "<call_env> token not in vocab"

        prompts.meta_info.update({"eos_token_id": eos_id, "pad_token_id": pad_id})

        # Hyperparams
        thinking_budget: int = int(
            self.config.rollout.interactive_mode.get("thinking_budget", 1024)
        )
        answer_length: int = int(
            self.config.rollout.get("response_length", 1024)
        )
        max_env_calls: int = int(
            self.config.rollout.multi_turn.get("max_env_calls", 6)
        )
        # Total model-token cap across both phases
        total_budget: int = thinking_budget + answer_length

        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager (TTS)", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            base_kwargs = dict(prompts.meta_info.get("kwargs", {}))

            is_validate = prompts.meta_info.get("validate", False)
            n = 1 if is_validate else int(self.config.rollout.get("n", 1))
            if n > 1:
                prompts = prompts.repeat(repeat_times=n, interleave=True)

            # Build per-sample state
            if "raw_prompt_ids" in prompts.non_tensor_batch:
                prompt_ids_list = [list(x) for x in prompts.non_tensor_batch["raw_prompt_ids"]]
            else:
                prompt_ids_list = []
                for row in prompts.batch["input_ids"]:
                    ids = row.tolist()
                    prompt_ids_list.append([t for t in ids if t != pad_id])

            raw_env_replies = prompts.non_tensor_batch.get("env_replies", None)
            states = [
                _SampleState(ids, env_replies=raw_env_replies[i] if raw_env_replies is not None else None)
                for i, ids in enumerate(prompt_ids_list)
            ]

            # Per-sample phase flag: True = still in thinking phase
            in_thinking = [True] * len(states)

            with simple_timer("generate_sequences_tts", timing_generate):

                # ----------------------------------------------------------------
                # Phase 1: thinking loop
                # ----------------------------------------------------------------
                while True:
                    active_idx = [
                        i for i, s in enumerate(states)
                        if in_thinking[i]
                        and not s.finished
                        and s.n_model_tokens < thinking_budget
                    ]
                    if not active_idx:
                        break

                    active_states = [states[i] for i in active_idx]
                    dp_active = _build_dataproto(
                        [s.ctx for s in active_states],
                        pad_id=pad_id,
                        eos_id=eos_id,
                        device=torch.device(get_device_id()),
                    )

                    round_tokens = thinking_budget - min(s.n_model_tokens for s in active_states)
                    seg_kwargs = {
                        **base_kwargs,
                        "stop_token_ids": [t_end_id],
                        "max_tokens": max(round_tokens, 1),
                        "n": 1,
                    }

                    out = self.rollout.generate_sequences(prompts=dp_active, **seg_kwargs)

                    for j, i in enumerate(active_idx):
                        state = states[i]
                        raw_row = out.batch["responses"][j]
                        tokens = _extract_response_tokens(raw_row, eos_id=eos_id, pad_id=pad_id)

                        if not tokens:
                            # No output at all — close thinking and move on
                            state.append_model([t_end_id])
                            in_thinking[i] = False
                            continue

                        # Clip to this sample's own remaining thinking budget
                        remaining = thinking_budget - state.n_model_tokens
                        tokens = tokens[:remaining]

                        if not tokens:
                            state.append_model([t_end_id])
                            in_thinking[i] = False
                            continue

                        hit_t_end = tokens[-1] == t_end_id
                        hit_eos   = tokens[-1] == eos_id

                        if hit_eos and not hit_t_end:
                            # Model ended generation before </T> — force-close thinking
                            state.append_model(tokens)
                            state.append_model([t_end_id])
                            in_thinking[i] = False
                        elif hit_t_end:
                            tokens_without_end = tokens[:-1]
                            new_total = state.n_model_tokens + len(tokens)
                            if new_total < thinking_budget:
                                # Budget not exhausted: replace </T> with <sep> to extend thinking
                                state.append_model(tokens_without_end + [sep_id])
                                # Stay in thinking phase — next round will continue
                            else:
                                # Budget exhausted: commit </T> and enter answer phase
                                state.append_model(tokens)
                                in_thinking[i] = False
                        else:
                            # Hit round token limit without </T> — force-close thinking
                            state.append_model(tokens)
                            state.append_model([t_end_id])
                            in_thinking[i] = False

                # Force-close thinking for any samples still in thinking phase
                for i, s in enumerate(states):
                    if in_thinking[i] and not s.finished:
                        if not s.response_tokens or s.response_tokens[-1] != t_end_id:
                            s.append_model([t_end_id])
                        in_thinking[i] = False

                # ----------------------------------------------------------------
                # Phase 2: answer / multi-turn loop
                # ----------------------------------------------------------------
                while True:
                    active_idx = [
                        i for i, s in enumerate(states)
                        if not s.finished
                        and s.n_env_calls < max_env_calls
                        and s.n_model_tokens < total_budget
                    ]
                    if not active_idx:
                        break

                    active_states = [states[i] for i in active_idx]
                    dp_active = _build_dataproto(
                        [s.ctx for s in active_states],
                        pad_id=pad_id,
                        eos_id=eos_id,
                        device=torch.device(get_device_id()),
                    )

                    round_tokens = total_budget - min(s.n_model_tokens for s in active_states)
                    seg_kwargs = {
                        **base_kwargs,
                        "stop_token_ids": [call_env_id],
                        "max_tokens": max(round_tokens, 1),
                        "n": 1,
                    }

                    out = self.rollout.generate_sequences(prompts=dp_active, **seg_kwargs)

                    for j, i in enumerate(active_idx):
                        state = states[i]
                        raw_row = out.batch["responses"][j]
                        tokens = _extract_response_tokens(raw_row, eos_id=eos_id, pad_id=pad_id)

                        if not tokens:
                            state.finished = True
                            continue

                        remaining = total_budget - state.n_model_tokens
                        if remaining < len(tokens):
                            print(f"[warning TTS] tokens length={len(tokens)} > remaining={remaining}")
                        tokens = tokens[:remaining]
                        if not tokens:
                            state.finished = True
                            continue

                        hit_eos  = tokens[-1] == eos_id
                        hit_call = call_env_id in tokens

                        if hit_call:
                            cut = tokens.index(call_env_id) + 1
                            state.append_model(tokens[:cut])
                            state.n_env_calls += 1

                            obs_text = state.env_replies.pop(0) if state.env_replies else None
                            if obs_text:
                                state.env_replies_used.append(obs_text)
                            if not obs_text:
                                state.finished = True
                            else:
                                obs_ids = self.tokenizer(obs_text, add_special_tokens=False).input_ids
                                if obs_ids:
                                    state.append_env(obs_ids)
                                else:
                                    print(f"[warning TTS] empty obs_ids for {obs_text!r}")

                            if state.n_env_calls >= max_env_calls:
                                state.finished = True
                        else:
                            state.append_model(tokens)
                            state.finished = True

            log_gpu_memory_usage("After TTS rollout generation", logger=logger)

            # ---- assemble final outputs ----
            resp_len = max((len(s.response_tokens) for s in states), default=1)
            device = prompts.batch["input_ids"].device
            resp_len_tensor = torch.tensor(resp_len, dtype=torch.long, device=device)
            dist.all_reduce(resp_len_tensor, op=dist.ReduceOp.MAX)
            resp_len = int(resp_len_tensor.item())

            def _right_pad(seqs, pad, length, dev):
                out = torch.full((len(seqs), length), pad, dtype=torch.long, device=dev)
                for i, s in enumerate(seqs):
                    t = s[:length]
                    if t:
                        out[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=dev)
                return out

            responses   = _right_pad([s.response_tokens for s in states], pad_id, resp_len, device)
            policy_mask = _right_pad([s.policy_mask      for s in states], 0,      resp_len, device)
            response_mask = (responses != pad_id).to(prompts.batch["attention_mask"].dtype)
            policy_mask   = policy_mask.to(prompts.batch["attention_mask"].dtype)

            assert not (policy_mask & ~response_mask).any(), \
                "policy_mask has 1s outside valid response tokens"
            assert (policy_mask.sum(dim=-1) > 0).all(), \
                "some samples have no model tokens in policy_mask"
            assert (response_mask.sum(dim=-1) >= policy_mask.sum(dim=-1)).all(), \
                "response_mask should cover at least as many tokens as policy_mask"

            prompt_ids   = prompts.batch["input_ids"]
            prompt_attn  = prompts.batch["attention_mask"]
            prompt_pos   = prompts.batch["position_ids"]

            input_ids      = torch.cat([prompt_ids, responses], dim=-1)
            attention_mask = torch.cat([prompt_attn, response_mask], dim=-1)

            delta = torch.arange(1, resp_len + 1, device=device).unsqueeze(0).expand(len(states), -1)
            if prompt_pos.dim() == 3:  # Qwen2-VL mRoPE
                delta = delta.view(len(states), 1, -1).expand(len(states), 3, -1)
            response_pos = prompt_pos[..., -1:] + delta
            position_ids = torch.cat([prompt_pos, response_pos], dim=-1)

            batch = TensorDict(
                {
                    "prompts":        prompt_ids,
                    "responses":      responses,
                    "input_ids":      input_ids,
                    "attention_mask": attention_mask,
                    "position_ids":   position_ids,
                    "response_mask":  policy_mask,
                    "policy_mask":    policy_mask,
                },
                batch_size=len(states),
            )
            non_tensor = dict(prompts.non_tensor_batch)
            non_tensor["n_env_calls"]               = np.array([s.n_env_calls for s in states], dtype=np.int32)
            non_tensor["max_env_calls_used"]         = np.array([max_env_calls] * len(states), dtype=np.int32)
            non_tensor["env_replies_used"]           = np.array([s.env_replies_used for s in states], dtype=object)
            non_tensor["prompt_token_count"]         = np.array([s.prompt_token_count for s in states], dtype=np.int32)
            non_tensor["model_token_count"]          = np.array([s.n_model_tokens for s in states], dtype=np.int32)
            non_tensor["env_token_count"]            = np.array(
                [len(s.response_tokens) - s.n_model_tokens for s in states], dtype=np.int32
            )
            non_tensor["total_response_token_count"] = np.array(
                [len(s.response_tokens) for s in states], dtype=np.int32
            )
            output = DataProto(batch=batch, non_tensor_batch=non_tensor)
            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red")
    def generate_multi_turn_sequences_no_stop(self, prompts: DataProto):
        """Multi-turn generation without a dedicated stop token.

        Because there is no <call_env> stop token to interrupt generation, each
        round generates up to ``round_max_tokens`` tokens freely, then the
        output is decoded and scanned for the first complete chess move
        (e.g. ``Pd2d4``, ``O-O``).  Generation is truncated at that token
        boundary, the environment reply is injected, and the next round begins.

        If a round produces no recognisable move, the sample is marked finished.

        Config keys:
          config.rollout.multi_turn.round_max_tokens  (int, default 8)
              How many tokens to generate per round when scanning for a move.
          config.rollout.multi_turn.max_env_calls     (int, default 6)
              Maximum number of move/env-reply turns.
          config.rollout.response_length              (int, default 1024)
              Hard cap on total model tokens across all rounds.
        """
        print("Generating multi-turn sequences (no stop token)")
        prompts = prompts.to(get_device_id())
        assert self._is_rollout

        pad_id: int = (
            self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id
        )
        if pad_id is None:
            pad_id = self.tokenizer.bos_token_id

        eos_id: int = (
            self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id
        )

        prompts.meta_info.update({"eos_token_id": eos_id, "pad_token_id": pad_id})

        # Hyperparams
        round_max_tokens: int = int(
            self.config.rollout.multi_turn.get("round_max_tokens", 8)
        )
        max_env_calls: int = int(
            self.config.rollout.multi_turn.get("max_env_calls", 6)
        )
        max_model_tokens: int = int(
            self.config.rollout.get("response_length", 1024)
        )
        # Carry vLLM's sampled-token log-probs through the turns (rollout.calculate_log_probs).
        collect_logprobs = bool(self.config.rollout.get("calculate_log_probs", False))
        n_dropped_generated = 0
        # vLLM rejects (V0: raises on) a request longer than max_model_len. Opponent replies are added outside
        # the model-token budget, so a long prompt plus a near-full budget can overflow it; such samples stop
        # taking turns instead (same limit as vLLMRollout).
        max_ctx_len: int = int(
            self.config.rollout.get("max_model_len", None) or self.config.rollout.prompt_length + max_model_tokens
        )

        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager (no-stop)", logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            base_kwargs = dict(prompts.meta_info.get("kwargs", {}))

            is_validate = prompts.meta_info.get("validate", False)
            n = 1 if is_validate else int(self.config.rollout.get("n", 1))
            if n > 1:
                prompts = prompts.repeat(repeat_times=n, interleave=True)

            # Build per-sample state
            if "raw_prompt_ids" in prompts.non_tensor_batch:
                prompt_ids_list = [list(x) for x in prompts.non_tensor_batch["raw_prompt_ids"]]
            else:
                prompt_ids_list = []
                for row in prompts.batch["input_ids"]:
                    ids = row.tolist()
                    prompt_ids_list.append([t for t in ids if t != pad_id])

            raw_env_replies = prompts.non_tensor_batch.get("env_replies", None)
            states = [
                _SampleState(ids, env_replies=raw_env_replies[i] if raw_env_replies is not None else None)
                for i, ids in enumerate(prompt_ids_list)
            ]

            with simple_timer("generate_multi_turn_sequences_no_stop", timing_generate):
                while True:
                    active_idx = [
                        i for i, s in enumerate(states)
                        if not s.finished
                        and s.n_env_calls < max_env_calls
                        and s.n_model_tokens < max_model_tokens
                        and len(s.ctx) < max_ctx_len
                    ]
                    if not active_idx:
                        break

                    active_states = [states[i] for i in active_idx]
                    dp_active = _build_dataproto(
                        [s.ctx for s in active_states],
                        pad_id=pad_id,
                        eos_id=eos_id,
                        device=torch.device(get_device_id()),
                    )

                    remaining_min = max_model_tokens - min(s.n_model_tokens for s in active_states)
                    this_round_tokens = min(round_max_tokens, max(remaining_min, 1))

                    seg_kwargs = {
                        **base_kwargs,
                        # No stop_token_ids — we parse the move from decoded text
                        "max_tokens": this_round_tokens,
                        "n": 1,
                    }

                    out = self.rollout.generate_sequences(prompts=dp_active, **seg_kwargs)
                    out_logprobs = out.batch["rollout_log_probs"] if collect_logprobs else None

                    for j, i in enumerate(active_idx):
                        state = states[i]
                        raw_row = out.batch["responses"][j]
                        tokens, logprobs, n_drop = _extract_response_tokens_and_logprobs(
                            raw_row, None if out_logprobs is None else out_logprobs[j], eos_id=eos_id, pad_id=pad_id
                        )
                        n_dropped_generated += n_drop

                        if not tokens:
                            state.finished = True
                            continue

                        # Clip to this sample's own remaining global budget
                        remaining = max_model_tokens - state.n_model_tokens
                        tokens = tokens[:remaining]
                        logprobs = _head(logprobs, remaining)
                        if not tokens:
                            state.finished = True
                            continue

                        hit_eos = tokens[-1] == eos_id

                        # Strip EOS so boundary detection works on content tokens only.
                        # The model (pretrained to generate full games) often emits
                        # multiple moves + EOS in one shot; we must truncate to the first move.
                        # Every cut below keeps a prefix of what vLLM generated, so slicing the
                        # log-probs the same way keeps them exact.
                        if hit_eos:
                            tokens_body = tokens[:-1]
                            logprobs_body = _head(logprobs, len(tokens) - 1)
                            if not tokens_body:
                                # Only EOS was generated — keep it so policy_mask is non-empty
                                state.append_model(tokens, logprobs)  # [eos_id]
                                state.finished = True
                                continue
                        else:
                            tokens_body = tokens
                            logprobs_body = logprobs

                        # Scan for the first complete chess move (same path for EOS and non-EOS)
                        move_end_idx, move_str = _find_move_token_boundary(tokens_body, self.tokenizer)

                        if move_end_idx is None:
                            # No recognisable move — append everything (without EOS) and finish
                            print(f"[warning no-stop] no move found in round output for sample {i}, finishing")
                            state.append_model(tokens_body, logprobs_body)
                            state.finished = True
                            continue

                        # Keep only up to (and including) the move boundary
                        move_tokens = tokens_body[:move_end_idx]
                        state.append_model(move_tokens, _head(logprobs_body, move_end_idx))
                        state.n_env_calls += 1

                        # Fetch env reply
                        obs_text = state.env_replies.pop(0) if state.env_replies else None
                        if obs_text:
                            state.env_replies_used.append(obs_text)
                        if not obs_text:
                            state.finished = True
                        else:
                            obs_ids = self.tokenizer(obs_text, add_special_tokens=False).input_ids
                            if obs_ids:
                                state.append_env(obs_ids)
                            else:
                                print(f"[warning no-stop] empty obs_ids for {obs_text!r}")

                        if state.n_env_calls >= max_env_calls:
                            state.finished = True

            log_gpu_memory_usage("After no-stop rollout generation", logger=logger)
            if n_dropped_generated:
                print(f"[warning no-stop] dropped {n_dropped_generated} generated pad-id tokens from responses; "
                      f"tokens after them in the same turn were sampled with them in context")
            n_ctx_limited = sum(1 for s in states if not s.finished and len(s.ctx) >= max_ctx_len)
            if n_ctx_limited:
                print(f"[warning no-stop] {n_ctx_limited} samples stopped at vLLM max_model_len={max_ctx_len}")

            # ---- assemble final outputs ----
            resp_len = max((len(s.response_tokens) for s in states), default=1)
            device = prompts.batch["input_ids"].device
            resp_len_tensor = torch.tensor(resp_len, dtype=torch.long, device=device)
            dist.all_reduce(resp_len_tensor, op=dist.ReduceOp.MAX)
            resp_len = int(resp_len_tensor.item())

            def _right_pad(seqs, pad, length, dev):
                out = torch.full((len(seqs), length), pad, dtype=torch.long, device=dev)
                for i, s in enumerate(seqs):
                    t = s[:length]
                    if t:
                        out[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=dev)
                return out

            responses   = _right_pad([s.response_tokens for s in states], pad_id, resp_len, device)
            policy_mask = _right_pad([s.policy_mask      for s in states], 0,      resp_len, device)
            response_mask = (responses != pad_id).to(prompts.batch["attention_mask"].dtype)
            policy_mask   = policy_mask.to(prompts.batch["attention_mask"].dtype)

            assert not (policy_mask & ~response_mask).any(), \
                "policy_mask has 1s outside valid response tokens"
            assert (policy_mask.sum(dim=-1) > 0).all(), \
                "some samples have no model tokens in policy_mask"
            assert (response_mask.sum(dim=-1) >= policy_mask.sum(dim=-1)).all(), \
                "response_mask should cover at least as many tokens as policy_mask"

            prompt_ids   = prompts.batch["input_ids"]
            prompt_attn  = prompts.batch["attention_mask"]
            prompt_pos   = prompts.batch["position_ids"]

            input_ids      = torch.cat([prompt_ids, responses], dim=-1)
            attention_mask = torch.cat([prompt_attn, response_mask], dim=-1)

            delta = torch.arange(1, resp_len + 1, device=device).unsqueeze(0).expand(len(states), -1)
            if prompt_pos.dim() == 3:  # Qwen2-VL mRoPE
                delta = delta.view(len(states), 1, -1).expand(len(states), 3, -1)
            response_pos = prompt_pos[..., -1:] + delta
            position_ids = torch.cat([prompt_pos, response_pos], dim=-1)

            batch = TensorDict(
                {
                    "prompts":        prompt_ids,
                    "responses":      responses,
                    "input_ids":      input_ids,
                    "attention_mask": attention_mask,
                    "position_ids":   position_ids,
                    "response_mask":  policy_mask,
                    "policy_mask":    policy_mask,
                },
                batch_size=len(states),
            )
            if collect_logprobs:
                batch["rollout_log_probs"] = _rollout_log_probs_tensor(states, policy_mask, resp_len, device)
            non_tensor = dict(prompts.non_tensor_batch)
            non_tensor["n_env_calls"]               = np.array([s.n_env_calls for s in states], dtype=np.int32)
            non_tensor["max_env_calls_used"]         = np.array([max_env_calls] * len(states), dtype=np.int32)
            non_tensor["env_replies_used"]           = np.array([s.env_replies_used for s in states], dtype=object)
            non_tensor["prompt_token_count"]         = np.array([s.prompt_token_count for s in states], dtype=np.int32)
            non_tensor["model_token_count"]          = np.array([s.n_model_tokens for s in states], dtype=np.int32)
            non_tensor["env_token_count"]            = np.array(
                [len(s.response_tokens) - s.n_model_tokens for s in states], dtype=np.int32
            )
            non_tensor["total_response_token_count"] = np.array(
                [len(s.response_tokens) for s in states], dtype=np.int32
            )
            output = DataProto(batch=batch, non_tensor_batch=non_tensor)
            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="blue")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_device_id())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                output, entropys, self_certaintys = self.actor.compute_log_prob(data=data, calculate_entropy=True, 
                                                                              calculate_self_certainty=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys, "self_certaintys": self_certaintys},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="olive")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model
        # Support all hardwares
        data = data.to(get_device_id())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output, _, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config):
        Worker.__init__(self)
        profiler_config = omega_conf_to_dataclass(config.get("profiler", {}), ProfilerConfig)
        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=profiler_config))
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(), init_method=os.environ.get("DIST_INIT_METHOD", None)
            )
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = self.config.model.get("lora_rank", 0) > 0

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation="flash_attention_2",
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "target_modules": convert_to_regular_types(self.config.model.target_modules),
                "bias": "none",
            }
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        warmup_style = config.optim.get("warmup_style", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if warmup_style == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif warmup_style == "cosine":
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
            )
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="cyan")
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="pink")
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)
        profiler_config = omega_conf_to_dataclass(config.get("profiler", {}), ProfilerConfig)
        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=profiler_config))

        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(), init_method=os.environ.get("DIST_INIT_METHOD", None)
            )
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        if is_cuda_available:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        elif is_npu_available:
            from transformers.integrations.npu_flash_attention import (
                index_first_axis,
                pad_input,
                rearrange,
                unpad_input,
            )

        from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], list):
                chat: list = data.non_tensor_batch["raw_prompt"][i]
            else:
                chat: list = data.non_tensor_batch["raw_prompt"][i].tolist()

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares
        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncvLLMServer.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        return self.rollout.execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(self, prompt_ids: List[int], sampling_params: Dict[str, Any], request_id: str) -> List[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        if self.config.rollout.free_cache_engine:
            await self.rollout.wake_up()
        # return something to block the caller
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        if self.config.rollout.free_cache_engine:
            await self.rollout.sleep()
        # return something to block the caller
        return True
