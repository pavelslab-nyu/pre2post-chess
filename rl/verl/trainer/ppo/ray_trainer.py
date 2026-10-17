# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

# Debug flag - controlled by environment variable DEBUG
import os
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

import json
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class _IndexedDataset(Dataset):
    """Thin wrapper that injects the original row index as ``__prompt_idx__``
    into each returned sample. Used by the prompt-filter mechanism to identify
    prompts across steps and epochs (per-step ``uid`` is ephemeral)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item["__prompt_idx__"] = int(i)
        return item


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_rollout_log_prob_metrics(
    rollout_log_probs: torch.Tensor,
    trainer_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> dict:
    """How closely the trainer's log-probs match the rollout engine's for the same sampled tokens.

    Only response_mask tokens count -- for multi-turn chess that is policy_mask, so opponent/env tokens (which
    have no rollout log-prob) are excluded. delta = log pi_trainer - log pi_rollout is the log PPO ratio at the
    start of the update when pi_old is the rollout log-prob (actor.use_rollout_log_probs).
    """
    mask = response_mask.bool()
    if not mask.any():
        return {}
    lp_rollout = rollout_log_probs[mask].double()
    lp_trainer = trainer_log_probs[mask].double()
    delta = lp_trainer - lp_rollout
    ratio = torch.exp(delta)
    prob_abs_diff = (torch.exp(lp_trainer) - torch.exp(lp_rollout)).abs()
    seq_delta = ((trainer_log_probs - rollout_log_probs).double() * mask).sum(dim=-1)[mask.any(dim=-1)]
    metrics = {
        "rollout_corr/log_ratio_abs_mean": delta.abs().mean().item(),
        "rollout_corr/log_ratio_abs_max": delta.abs().max().item(),
        # KL(pi_rollout || pi_trainer) on the rollout's own samples: k1 is unbiased, k3 is >= 0 per token.
        "rollout_corr/kl_k1": (-delta).mean().item(),
        "rollout_corr/kl_k3": (ratio - 1 - delta).mean().item(),
        # Tokens whose ratio is already outside the PPO clip range before any update.
        "rollout_corr/ratio_outside_clip_frac": ((ratio < 1 - clip_ratio_low) | (ratio > 1 + clip_ratio_high))
        .double()
        .mean()
        .item(),
        "rollout_corr/prob_abs_diff_mean": prob_abs_diff.mean().item(),
        "rollout_corr/prob_abs_diff_max": prob_abs_diff.max().item(),
        "rollout_corr/seq_log_ratio_abs_mean": seq_delta.abs().mean().item(),
    }
    if lp_rollout.numel() > 1:
        corr = torch.corrcoef(torch.stack([torch.exp(lp_rollout), torch.exp(lp_trainer)]))[0, 1]
        if torch.isfinite(corr):
            metrics["rollout_corr/prob_pearson"] = corr.item()
    return metrics


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
    tokenizer=None,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        
        # Check if we have INTUITOR self-certainty examples
        reward_methods = data.non_tensor_batch.get("reward_methods", [])
        has_self_certainty = len(reward_methods) > 0 and "SELF_CERTAINTY" in reward_methods
        
        if has_self_certainty and "self_certaintys" in data.batch:
            # INTUITOR mode: Replace self-certainty rewards
            self_certaintys = data.batch["self_certaintys"]
            grpo_calculation_mask = grpo_calculation_mask.to(self_certaintys.dtype)
            
            # Calculate self-certainty sentence-wise mean
            sentence_wise_mean = masked_mean(
                self_certaintys.detach(), mask=grpo_calculation_mask, axis=-1
            )
            

            
            # Create token-level rewards with self-certainty
            lengths = grpo_calculation_mask.sum(dim=-1).long()
            eos_mask_id = lengths - 1
            token_level_rewards = torch.zeros_like(self_certaintys)
            
            # Place rewards: self-certainty for SELF_CERTAINTY, original for others
            batch_size = token_level_rewards.shape[0]
            self_certainty_count = 0
            rule_based_count = 0
            
            for i in range(batch_size):
                if i < len(reward_methods) and reward_methods[i] == "SELF_CERTAINTY":
                    token_level_rewards[i, eos_mask_id[i]] = sentence_wise_mean[i]
                    self_certainty_count += 1
                else:
                    # Keep original reward from reward function
                    token_level_rewards[i, eos_mask_id[i]] = data.batch["token_level_rewards"][i, eos_mask_id[i]]
                    rule_based_count += 1
            

            
            # UPDATE: Store the actual rewards back into the batch for logging
            data.batch["token_level_rewards"] = token_level_rewards
            
            # Use mixed token_level_rewards for GRPO
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        else:
            # Standard GRPO: use rewards from reward function
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
            
    elif adv_estimator == AdvantageEstimator.SELFDISTILL:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        
        reward_methods = data.non_tensor_batch.get("reward_methods", [])

        # Standard GRPO: use rewards from reward function
        advantages, returns = core_algos.compute_selfdistill_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
        )
        
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.CONSTANT:
        grpo_calculation_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_constant_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.POSTIVE_GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_positive_grpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.NEGATIVE_GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_negative_grpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.NEGATIVE_SELFDISTILL:
        grpo_calculation_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_negative_selfdistill_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.TANGO_GRPO:
        # TANGO GRPO: Combine outcome rewards (60%) + step rewards (40%) using RL Tango approach
        grpo_calculation_mask = data.batch["response_mask"]
        
        # Use rewards from reward function (no self-certainty logic needed for TANGO_GRPO)
        token_level_rewards = data.batch["token_level_rewards"]
        
        # Prepare step reward information for TANGO_GRPO
        reward_extra_info = {}
        if "step_rewards" in data.batch:
            reward_extra_info["step_rewards"] = data.batch["step_rewards"]
        if "step_rewards_mask" in data.batch:
            reward_extra_info["step_rewards_mask"] = data.batch["step_rewards_mask"]
        # Add raw_step_rewards for PRM_COT_HYBRID_OUTCOME_DIFF_ADV
        if "raw_step_rewards" in data.non_tensor_batch:
            reward_extra_info["raw_step_rewards"] = data.non_tensor_batch["raw_step_rewards"].tolist()
        # Add step_positions for PRM_COT_HYBRID_OUTCOME_DIFF_ADV
        if "step_positions" in data.non_tensor_batch:
            reward_extra_info["step_positions"] = data.non_tensor_batch["step_positions"].tolist()
        
        
        # Use TANGO GRPO advantage computation
        advantages, returns = core_algos.compute_tango_grpo_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            reward_extra_info=reward_extra_info if reward_extra_info else None,
        )
        
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.TANGO_GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.SELFDISTILL,
            AdvantageEstimator.CONSTANT,
            AdvantageEstimator.NEGATIVE_SELFDISTILL,
            AdvantageEstimator.NEGATIVE_GRPO,
            AdvantageEstimator.POSTIVE_GRPO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
                or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None
            ), (
                "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, "
                "due to no role-playing support"
            )
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                "only GRPO is tested for multi-turn with tool"
            )

        # check rollout log-probs as pi_old and the fp32 LM head
        rollout_cfg = config.actor_rollout_ref.rollout
        if config.actor_rollout_ref.actor.get("use_rollout_log_probs", False):
            if not rollout_cfg.get("calculate_log_probs", False):
                raise ValueError("actor.use_rollout_log_probs=True needs actor_rollout_ref.rollout.calculate_log_probs=True")
            if rollout_cfg.name != "vllm" or rollout_cfg.get("mode", "sync") != "sync":
                raise ValueError("actor.use_rollout_log_probs is implemented for the sync vLLM rollout only")
            # pi_old must be log_softmax(logits / T)[token], which is what the trainer computes. Truncation or
            # penalties change the sampling distribution in ways the trainer does not reproduce (and vLLM V0/V1
            # report them differently), so they are rejected rather than silently mismatched.
            if not rollout_cfg.temperature > 0:
                raise ValueError("actor.use_rollout_log_probs needs rollout.temperature > 0")
            if float(rollout_cfg.get("top_p", 1.0)) != 1.0 or int(rollout_cfg.get("top_k", -1)) > 0:
                raise ValueError("actor.use_rollout_log_probs needs rollout.top_p=1 and rollout.top_k=-1")
            for key, neutral in (
                ("min_p", 0.0),
                ("presence_penalty", 0.0),
                ("frequency_penalty", 0.0),
                ("repetition_penalty", 1.0),
                ("min_tokens", 0),
            ):
                if rollout_cfg.get(key, None) is not None and float(rollout_cfg[key]) != neutral:
                    raise ValueError(f"actor.use_rollout_log_probs needs rollout.{key}={neutral}")
        if config.actor_rollout_ref.model.get("fp32_lm_head", False):
            if rollout_cfg.name not in ("vllm", "hf"):
                raise ValueError("actor_rollout_ref.model.fp32_lm_head is implemented for the vLLM and HF rollouts only")
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                raise ValueError("actor_rollout_ref.model.fp32_lm_head does not support LoRA (vLLM's LoRA logits path)")
            if config.actor_rollout_ref.model.get("use_fused_kernels", False):
                raise ValueError("actor_rollout_ref.model.fp32_lm_head does not support use_fused_kernels=True")

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )

        # Wrap train_dataset so each sample carries its original row index as
        # __prompt_idx__. This stable per-prompt key lets us aggregate per-prompt
        # mean reward across a full epoch even though `uid` is regenerated each step.
        train_dataset = _IndexedDataset(train_dataset)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        # Stash construction parameters so we can rebuild the loader over a
        # Subset later (prompt filtering) and on checkpoint resume.
        self._train_loader_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        self._train_loader_num_workers = self.config.data.get("dataloader_num_workers", 8)
        self._train_loader_collate_fn = collate_fn
        self._filter_active = False
        self._filter_hard_indices = None

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self._train_loader_batch_size,
            num_workers=self._train_loader_num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        # DEBUG: Show what fields are available for saving
        if DEBUG:
            print(f"🔍 ROLLOUT SAVE DEBUG: Available fields in reward_extra_infos_dict:")
            for k, v in reward_extra_infos_dict.items():
                print(f"  - {k}: length={len(v)}, expected={n}")
        
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                # Skip step-level tensor information (too verbose even in DEBUG mode)
                if k in ["step_rewards", "step_rewards_mask"]:
                    continue
                    
                # Always save step_positions and penalty info (user requested)
                # In debug mode, also save advantages
                if not DEBUG and k in ["advantages"]:
                    continue
                    
                # Convert tensor objects to lists for JSON serialization
                if k in ["advantages"]:
                    # Convert tensors to lists
                    converted_v = []
                    for item in v:
                        if hasattr(item, 'tolist'):  # PyTorch tensor
                            converted_v.append(item.tolist())
                        else:
                            converted_v.append(item)
                    base_data[k] = converted_v
                else:
                    base_data[k] = v
                
                # DEBUG: Confirm what gets saved
                if DEBUG and k in ["model_step_count", "prm_step_count", "step_penalty_applied"]:
                    print(f"✅ SAVED TO ROLLOUT: {k} = {v[:3] if len(v) > 3 else v}{'...' if len(v) > 3 else ''}")

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        if DEBUG:
            print("{EVAL_START: Starting validation...}")
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            
            if DEBUG:
                print("{EVAL_PROMPT:", input_texts[0], "}")

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            if "env_replies" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_replies")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )


            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.bos_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            # if not self.async_rollout_mode:
            #     test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # else:
            #     test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.config.actor_rollout_ref.rollout.interactive_mode.enable:
                print("Interactive mode is enabled")
                if self.config.actor_rollout_ref.rollout.thinking_mode.enable:
                    if self.config.actor_rollout_ref.rollout.tts_mode.enable:
                        test_output_gen_batch_padded = self.actor_rollout_wg.generate_multi_turn_sequences_tts(test_gen_batch_padded)
                    else:
                        test_output_gen_batch_padded = self.actor_rollout_wg.generate_multi_turn_sequences(test_gen_batch_padded)
                else:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_multi_turn_sequences_no_stop(test_gen_batch_padded)
            else:
                print("Interactive mode is disabled")
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            
            if DEBUG:
                print("{EVAL_ROLLOUT:", output_texts[0], "}")

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            
            if DEBUG:
                print("{EVAL_REWARD:", scores[0], "}")

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if DEBUG:
            print("{EVAL_END: Validation completed}")
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # save prompt-filter state alongside (only meaningful when filtering active)
        filter_state_path = os.path.join(local_global_step_folder, "prompt_filter.pt")
        torch.save(
            {
                "filter_active": getattr(self, "_filter_active", False),
                "hard_indices": getattr(self, "_filter_hard_indices", None),
            },
            filter_state_path,
        )

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # Restore prompt-filter state BEFORE loading dataloader state so that
        # the dataloader being restored matches the (filtered) dataset that was
        # saved into the checkpoint.
        filter_state_path = os.path.join(global_step_folder, "prompt_filter.pt")
        if os.path.exists(filter_state_path):
            from verl.trainer.main_ppo import create_rl_sampler

            fs = torch.load(filter_state_path, weights_only=False)
            if fs.get("filter_active") and fs.get("hard_indices"):
                from torch.utils.data import Subset

                hard = list(fs["hard_indices"])
                subset = Subset(self.train_dataset, hard)
                # Match original sampler semantics (RandomSampler with Generator
                # if data.shuffle, else SequentialSampler).
                subset_sampler = create_rl_sampler(self.config.data, subset)
                self.train_dataloader = StatefulDataLoader(
                    dataset=subset,
                    batch_size=self._train_loader_batch_size,
                    num_workers=self._train_loader_num_workers,
                    drop_last=True,
                    collate_fn=self._train_loader_collate_fn,
                    sampler=subset_sampler,
                )
                self._filter_active = True
                self._filter_hard_indices = hard
                print(
                    f"[prompt_filter] restored from ckpt: {len(hard)} prompts kept "
                    f"(out of {len(self.train_dataset)} original)"
                )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        repeat_sampling_sglang_grpo = (
            self.config.actor_rollout_ref.rollout.name == "sglang"
            and self.config.actor_rollout_ref.rollout.multi_turn.enable
        )

        # Prompt-filter accumulator: prompt_idx -> list of per-step pass@n mean rewards.
        # Reset at every epoch boundary (or at every rebuild boundary).
        self._epoch_prompt_mean = {}
        prompt_filter_cfg = self.config.data.get("prompt_filter", {}) or {}
        prompt_filter_enabled = bool(prompt_filter_cfg.get("enable", False))

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]

                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                if "env_replies" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_replies")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                if repeat_sampling_sglang_grpo:
                    uids_for_prompts = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch.non_tensor_batch["uid"] = uids_for_prompts
                    gen_batch.non_tensor_batch["uid"] = uids_for_prompts
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    assert np.array_equal(batch.non_tensor_batch["uid"], gen_batch.non_tensor_batch["uid"]), (
                        "UIDs must be identical for SGLang rollout"
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        # Check for TTRL either via config or MAJORITY_VOTE reward model type
                        import os
                        reward_model_type = os.environ.get('REWARD_MODEL_TYPE', '').upper()
                        use_ttrl = self.config.get("ttrl", {}).get("enable", False) or (reward_model_type in ['MAJORITY_VOTE', 'MAJORITY_50_DIFFICULT_RULE_50_EASY', 'REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT'])
                        interactive_mode = self.config.actor_rollout_ref.rollout.get("interactive_mode", {}).get("enable", False)
                        thinking_mode = self.config.actor_rollout_ref.rollout.get("thinking_mode", {}).get("enable", True)

                        if interactive_mode:
                            print("Interactive mode is enabled")
                            if thinking_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_multi_turn_sequences(gen_batch)
                            else:
                                gen_batch_output = self.actor_rollout_wg.generate_multi_turn_sequences_no_stop(gen_batch)
                        else:
                            print("Interactive mode is disabled")
                            if use_ttrl:
                                from verl.trainer.ppo.ttrl_utils import select_top_k_per_prompt, apply_ttrl_gt

                                # Auto-configure TTRL parameters for MAJORITY_VOTE modes
                                if reward_model_type in ['MAJORITY_VOTE', 'MAJORITY_50_DIFFICULT_RULE_50_EASY', 'REVERSE_MAJORITY_50_EASY_RULE_50_DIFFICULT']:
                                    # Read TTRL parameters from environment variables (set in shell script)
                                    n_votes_per_prompt = int(os.environ.get('TTRL_N_VOTES_PER_PROMPT', '32'))
                                    n_samples_per_prompt = int(os.environ.get('TTRL_N_SAMPLES_PER_PROMPT', '8'))
                                else:
                                    # Use existing TTRL config
                                    n_votes_per_prompt = self.config.ttrl.n_votes_per_prompt
                                    n_samples_per_prompt = self.config.ttrl.n_samples_per_prompt

                                gen_batch.meta_info["kwargs"] = {"n": n_votes_per_prompt}
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                                # Debug prints to trace the assertion failure
                                if DEBUG:
                                    print(f"DEBUG: len(batch)={len(batch)}, n_votes_per_prompt={n_votes_per_prompt}")
                                    print(f"DEBUG: Expected responses={len(batch) * n_votes_per_prompt}")
                                    print(f"DEBUG: Actual responses={len(gen_batch_output)}")
                                    print(f"DEBUG: gen_batch_output type: {type(gen_batch_output)}")
                                    if hasattr(gen_batch_output, 'batch'):
                                        print(f"DEBUG: gen_batch_output.batch keys: {list(gen_batch_output.batch.keys())}")
                                        if 'responses' in gen_batch_output.batch:
                                            print(f"DEBUG: gen_batch_output.batch['responses'].shape: {gen_batch_output.batch['responses'].shape}")

                                assert len(gen_batch_output) == len(batch) * n_votes_per_prompt

                                batch = apply_ttrl_gt(batch, gen_batch_output, n_votes_per_prompt, self.tokenizer)
                                
                                # Debug before downsampling
                                if DEBUG:
                                    print(f"DEBUG: Before downsampling - len(gen_batch_output)={len(gen_batch_output)}")
                                    print(f"DEBUG: n_votes_per_prompt={n_votes_per_prompt}, n_samples_per_prompt={n_samples_per_prompt}")
                                
                                gen_batch_output = select_top_k_per_prompt(gen_batch_output, n_votes_per_prompt, n_samples_per_prompt)
                                
                                # Debug after downsampling
                                if DEBUG:
                                    print(f"DEBUG: After downsampling - len(gen_batch_output)={len(gen_batch_output)}")
                                    print(f"DEBUG: Expected after downsampling={len(batch) * n_samples_per_prompt}")

                                assert len(gen_batch_output) == len(batch) * n_samples_per_prompt
                            else:
                                if not self.async_rollout_mode:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                else:
                                    # vllm should set async_rollout_mode to enable async rollout
                                    # sglang turns on async_rollout_mode by default
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        # if DEBUG:
                        #     print("{ROLLOUT:", self.tokenizer.decode(gen_batch_output.batch["responses"][0], skip_special_tokens=True), "}")
                        

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if not repeat_sampling_sglang_grpo:
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            print(reward_tensor[0])
                        
                        ref_solve_none = 0
                        ref_solve_all = 0
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        prompt_idx_arr = batch.non_tensor_batch.get("__prompt_idx__", None)
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)
                            if (uid_rewards == 0).all():
                                ref_solve_none += 1
                            elif (uid_rewards == 1).all():
                                ref_solve_all += 1
                            # Record per-prompt mean reward (pass@n) for the
                            # prompt-filter mechanism. __prompt_idx__ is stable
                            # across steps/epochs unlike `uid`.
                            if prompt_filter_enabled and prompt_idx_arr is not None:
                                pidx = int(prompt_idx_arr[uid_mask][0])
                                self._epoch_prompt_mean.setdefault(pidx, []).append(
                                    float(uid_rewards.float().mean().item())
                                )
                        metrics.update({
                            "training/ref_solve_none_ratio": ref_solve_none / len(unique_uids),
                            "training/ref_solve_all_ratio": ref_solve_all / len(unique_uids),
                        })
                    # old_log_probs = pi_old in the PPO ratio. By default the FSDP actor recomputes them. With
                    # actor.use_rollout_log_probs, pi_old is the rollout engine's own log-prob of each sampled token
                    # (the behavior policy); the FSDP forward then runs only if actor.recompute_log_prob_for_metrics
                    # is on, to log entropy and trainer-vs-rollout agreement (rollout_corr/*), not for the loss.
                    actor_cfg = self.config.actor_rollout_ref.actor
                    use_rollout_log_probs = actor_cfg.get("use_rollout_log_probs", False)
                    has_rollout_log_probs = "rollout_log_probs" in batch.batch.keys()
                    if use_rollout_log_probs and not has_rollout_log_probs:
                        raise RuntimeError(
                            "actor.use_rollout_log_probs=True but this rollout path returned no rollout_log_probs"
                        )
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if not use_rollout_log_probs or actor_cfg.get("recompute_log_prob_for_metrics", False):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if use_rollout_log_probs:
                                trainer_log_probs = old_log_prob.batch.pop("old_log_probs")
                            else:
                                trainer_log_probs = old_log_prob.batch["old_log_probs"]
                            batch = batch.union(old_log_prob)

                            if has_rollout_log_probs:
                                clip_ratio_low = actor_cfg.get("clip_ratio_low", None)
                                clip_ratio_high = actor_cfg.get("clip_ratio_high", None)
                                metrics.update(
                                    compute_rollout_log_prob_metrics(
                                        rollout_log_probs=batch.batch["rollout_log_probs"],
                                        trainer_log_probs=trainer_log_probs,
                                        response_mask=batch.batch["response_mask"],
                                        clip_ratio_low=clip_ratio_low if clip_ratio_low is not None else actor_cfg.clip_ratio,
                                        clip_ratio_high=clip_ratio_high if clip_ratio_high is not None else actor_cfg.clip_ratio,
                                    )
                                )

                        if use_rollout_log_probs:
                            batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"].clone()
                            # update_policy reads meta_info["temperature"]; it normally arrives with the
                            # compute_log_prob output, which is skipped when recompute_log_prob_for_metrics=False.
                            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        if has_rollout_log_probs:
                            # Mean -log pi_rollout of the sampled tokens: an unbiased estimate of the rollout policy's
                            # per-token entropy that needs no forward pass.
                            policy_tokens = batch.batch["response_mask"].bool()
                            if policy_tokens.any():
                                metrics["rollout_corr/rollout_neg_log_prob_mean"] = (
                                    -batch.batch["rollout_log_probs"][policy_tokens].mean().item()
                                )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            # Handle tensor data separately (for TANGO_GRPO step rewards)
                            tensor_keys = ["step_rewards", "step_rewards_mask"]
                            # Keys that should be converted to object arrays directly (lists of lists)
                            object_array_keys = ["step_positions", "prm_response"]
                            
                            for k, v in reward_extra_infos_dict.items():
                                if k in tensor_keys:
                                    # Keep tensor data as-is for later processing
                                    continue
                                elif k in object_array_keys:
                                    # Convert directly to object array (for lists of lists)
                                    batch.non_tensor_batch[k] = np.array(v, dtype=object)
                                else:
                                    # Convert non-tensor data to numpy arrays with error handling
                                    try:
                                        batch.non_tensor_batch[k] = np.array(v)
                                    except ValueError as e:
                                        if DEBUG:
                                            print(f"WARNING: Could not convert {k} to numpy array: {e}")
                                            print(f"  {k} content: {v[:3] if len(v) > 3 else v}...")  # Show first 3 items
                                        # For chunking compatibility, convert to object array if possible
                                        try:
                                            batch.non_tensor_batch[k] = np.array(v, dtype=object)
                                        except:
                                            # Last resort: skip this key entirely
                                            if DEBUG:
                                                print(f"WARNING: Skipping key {k} entirely due to conversion issues")
                                            continue

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # Extract step reward information for TANGO GRPO (if needed)
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.TANGO_GRPO:
                            if reward_extra_infos_dict and "step_rewards" in reward_extra_infos_dict:
                                # Convert list of tensors to batch tensor
                                step_rewards_list = reward_extra_infos_dict["step_rewards"]
                                step_rewards_mask_list = reward_extra_infos_dict["step_rewards_mask"]
                                
                                if step_rewards_list and step_rewards_mask_list:
                                    # Stack into batch tensors
                                    step_rewards_tensor = torch.stack(step_rewards_list)  # (B, L)
                                    step_rewards_mask_tensor = torch.stack(step_rewards_mask_list)  # (B, L)
                                    
                                    # Add to batch for advantage computation
                                    batch.batch["step_rewards"] = step_rewards_tensor
                                    batch.batch["step_rewards_mask"] = step_rewards_mask_tensor
                                    

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                            tokenizer=self.tokenizer,
                        )

                        # COMPREHENSIVE DEBUG OUTPUT - Show full batch information
                        if DEBUG and self.config.algorithm.adv_estimator == AdvantageEstimator.TANGO_GRPO:
                            print("\n" + "="*120)
                            print("🔍 TANGO_GRPO COMPREHENSIVE BATCH DEBUG")
                            print("="*120)
                            
                            # Batch overview
                            batch_size = batch.batch["responses"].shape[0]
                            seq_len = batch.batch["responses"].shape[1]
                            print(f"📊 BATCH OVERVIEW:")
                            print(f"   Batch size: {batch_size}")
                            print(f"   Sequence length: {seq_len}")
                            print(f"   Advantage estimator: {self.config.algorithm.adv_estimator}")
                            
                            # Show first response in detail
                            for i in range(min(1, batch_size)):  # Only show first response
                                print(f"\n🔍 RESPONSE {i} DETAILED ANALYSIS:")
                                
                                # Decode prompt and response
                                prompt_ids = batch.batch["prompts"][i]
                                response_ids = batch.batch["responses"][i]
                                response_mask = batch.batch["response_mask"][i]
                                
                                # Find valid tokens
                                valid_response_len = response_mask.sum().item()
                                valid_response_ids = response_ids[:valid_response_len]
                                
                                prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                                response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                                
                                print(f"📝 PROMPT:\n{prompt_text[:200]}{'...' if len(prompt_text) > 200 else ''}")
                                print(f"\n📝 RESPONSE:\n{response_text}")
                                
                                # Token-level analysis
                                print(f"\n🔢 TOKEN ANALYSIS:")
                                print(f"   Valid response tokens: {valid_response_len}")
                                print(f"   Response IDs: {valid_response_ids.tolist()}")
                                
                                # Reward information
                                if "token_level_rewards" in batch.batch:
                                    token_rewards = batch.batch["token_level_rewards"][i]
                                    print(f"   Token rewards: {token_rewards[:valid_response_len].tolist()}")
                                
                                # Step reward information (if available)
                                if "step_rewards" in batch.batch:
                                    step_rewards = batch.batch["step_rewards"][i]
                                    step_mask = batch.batch["step_rewards_mask"][i]
                                    step_positions = torch.nonzero(step_mask, as_tuple=True)[0]
                                    
                                    print(f"\n⚡ STEP REWARD ANALYSIS:")
                                    print(f"   Step reward positions: {step_positions.tolist()}")
                                    print(f"   Step rewards at positions: {step_rewards[step_positions].tolist()}")
                                    print(f"   Step mask: {step_mask[:valid_response_len].tolist()}")
                                    
                                    # Show which tokens get step rewards
                                    for pos in step_positions:
                                        if pos < valid_response_len:
                                            token_id = valid_response_ids[pos].item()
                                            token_text = self.tokenizer.decode([token_id])
                                            step_reward_val = step_rewards[pos].item()
                                            print(f"   Position {pos}: token='{token_text}' (id={token_id}), step_reward={step_reward_val:.4f}")
                                
                                # Advantage information
                                if "advantages" in batch.batch:
                                    advantages = batch.batch["advantages"][i]
                                    print(f"\n🎯 ADVANTAGE ANALYSIS:")
                                    print(f"   Advantages: {advantages[:valid_response_len].tolist()}")
                                    print(f"   Advantage sum: {advantages.sum().item():.4f}")
                                    print(f"   Advantage mean: {advantages[response_mask.bool()].mean().item():.4f}")
                                
                                # Extra info from reward function
                                if hasattr(batch, 'non_tensor_batch') and "reward_methods" in batch.non_tensor_batch:
                                    reward_method = batch.non_tensor_batch["reward_methods"][i] if i < len(batch.non_tensor_batch["reward_methods"]) else "unknown"
                                    print(f"\n📋 REWARD INFO:")
                                    print(f"   Reward method: {reward_method}")
                                    
                                    # Show ground truth if available
                                    if "ground_truth" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["ground_truth"]):
                                        ground_truth = batch.non_tensor_batch["ground_truth"][i]
                                        print(f"   Ground truth: {ground_truth}")
                                
                                # Show PRM response if available
                                prm_response = None
                                if reward_extra_infos_dict and "prm_response" in reward_extra_infos_dict and i < len(reward_extra_infos_dict["prm_response"]):
                                    prm_response = reward_extra_infos_dict["prm_response"][i]
                                elif "prm_response" in batch.non_tensor_batch and i < len(batch.non_tensor_batch["prm_response"]):
                                    prm_response = batch.non_tensor_batch["prm_response"][i]
                                
                                if prm_response:
                                    print(f"\n🤖 PRM RESPONSE:\n{prm_response[:300]}{'...' if len(prm_response) > 300 else ''}")
                            
                            print("="*120 + "\n")


                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)


                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            
                            # Use actual final rewards (token_level_rewards) instead of original scores
                            if "token_level_rewards" in batch.batch:
                                final_scores = batch.batch["token_level_rewards"].sum(-1).cpu().tolist()
                            else:
                                final_scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            
                            # Add final reward and advantage info to reward_extra_infos_dict
                            enhanced_reward_extra_infos_dict = reward_extra_infos_dict.copy()
                            enhanced_reward_extra_infos_dict["final_reward_used"] = final_scores
                            
                            # Extract 5 specific token advantages per response
                            advantages_tensor = batch.batch["advantages"]  # [batch_size, seq_len]
                            response_mask = batch.batch["response_mask"]   # [batch_size, seq_len]
                            batch_size = advantages_tensor.shape[0]
                            
                            first_token_advantages = []
                            first_mid_token_advantages = []  # Between first and middle
                            middle_token_advantages = []
                            mid_last_token_advantages = []   # Between middle and last
                            last_token_advantages = []
                            
                            for i in range(batch_size):
                                # Find valid token positions for this response
                                valid_positions = torch.nonzero(response_mask[i], as_tuple=True)[0]
                                
                                if len(valid_positions) > 0:
                                    # First valid token
                                    first_pos = valid_positions[0]
                                    first_token_advantages.append(advantages_tensor[i, first_pos].item())
                                    
                                    # Last valid token
                                    last_pos = valid_positions[-1]
                                    last_token_advantages.append(advantages_tensor[i, last_pos].item())
                                    
                                    # Calculate positions based on sequence length
                                    seq_len = len(valid_positions)
                                    
                                    if seq_len == 1:
                                        # Only one valid token - use it for all positions
                                        first_mid_token_advantages.append(advantages_tensor[i, first_pos].item())
                                        middle_token_advantages.append(advantages_tensor[i, first_pos].item())
                                        mid_last_token_advantages.append(advantages_tensor[i, first_pos].item())
                                    elif seq_len == 2:
                                        # Only first and last - use first for first_mid and middle, last for mid_last
                                        first_mid_token_advantages.append(advantages_tensor[i, first_pos].item())
                                        middle_token_advantages.append(advantages_tensor[i, first_pos].item())
                                        mid_last_token_advantages.append(advantages_tensor[i, last_pos].item())
                                    elif seq_len == 3:
                                        # Three tokens: first, middle, last
                                        mid_pos = valid_positions[1]
                                        first_mid_token_advantages.append(advantages_tensor[i, mid_pos].item())  # Use middle as first_mid
                                        middle_token_advantages.append(advantages_tensor[i, mid_pos].item())
                                        mid_last_token_advantages.append(advantages_tensor[i, mid_pos].item())   # Use middle as mid_last
                                    elif seq_len == 4:
                                        # Four tokens: first, first_mid, mid_last, last
                                        first_mid_pos = valid_positions[1]
                                        mid_last_pos = valid_positions[2]
                                        first_mid_token_advantages.append(advantages_tensor[i, first_mid_pos].item())
                                        middle_token_advantages.append(advantages_tensor[i, first_mid_pos].item())  # Use first_mid as middle
                                        mid_last_token_advantages.append(advantages_tensor[i, mid_last_pos].item())
                                    else:
                                        # 5+ tokens: divide into 5 segments (first, first_mid, middle, mid_last, last)
                                        # Calculate positions for 5 evenly distributed points
                                        first_mid_idx = seq_len // 4                    # 25% position
                                        middle_idx = seq_len // 2                       # 50% position  
                                        mid_last_idx = (3 * seq_len) // 4               # 75% position
                                        
                                        first_mid_pos = valid_positions[first_mid_idx]
                                        middle_pos = valid_positions[middle_idx]
                                        mid_last_pos = valid_positions[mid_last_idx]
                                        
                                        first_mid_token_advantages.append(advantages_tensor[i, first_mid_pos].item())
                                        middle_token_advantages.append(advantages_tensor[i, middle_pos].item())
                                        mid_last_token_advantages.append(advantages_tensor[i, mid_last_pos].item())
                            
                            enhanced_reward_extra_infos_dict["advantage_first_token"] = first_token_advantages
                            enhanced_reward_extra_infos_dict["advantage_first_mid_token"] = first_mid_token_advantages
                            enhanced_reward_extra_infos_dict["advantage_middle_token"] = middle_token_advantages
                            enhanced_reward_extra_infos_dict["advantage_mid_last_token"] = mid_last_token_advantages
                            enhanced_reward_extra_infos_dict["advantage_last_token"] = last_token_advantages
                            
                            # Add UIDs for grouping analysis
                            if "uid" in batch.non_tensor_batch:
                                enhanced_reward_extra_infos_dict["uid"] = batch.non_tensor_batch["uid"].tolist()

                            # Add multi-turn env call / token count stats (populated by generate_multi_turn_sequences)
                            for _key in [
                                "n_env_calls",
                                "max_env_calls_used",
                                "env_replies_used",
                                "prompt_token_count",
                                "model_token_count",
                                "env_token_count",
                                "total_response_token_count",
                            ]:
                                if _key in batch.non_tensor_batch:
                                    _val = batch.non_tensor_batch[_key]
                                    enhanced_reward_extra_infos_dict[_key] = (
                                        _val.tolist() if hasattr(_val, "tolist") else list(_val)
                                    )
                            
                            # Add TANGO step rewards for analysis (if available)
                            if self.config.algorithm.adv_estimator == AdvantageEstimator.TANGO_GRPO and "step_rewards" in batch.batch:
                                step_rewards_tensor = batch.batch["step_rewards"]  # [batch_size, seq_len]
                                step_rewards_list = []
                                for i in range(len(inputs)):
                                    if i < step_rewards_tensor.shape[0]:
                                        # Convert tensor to list for JSON serialization
                                        step_reward_array = step_rewards_tensor[i].tolist()
                                        step_rewards_list.append(step_reward_array)
                                    else:
                                        # Fallback for missing step rewards
                                        step_rewards_list.append([])
                                enhanced_reward_extra_infos_dict["step_rewards"] = step_rewards_list
                            
                            # Add token-to-advantage mapping using EXACT existing data (DEBUG mode only)
                            if DEBUG:
                                # Save full advantage array
                                full_advantages = []
                                token_advantage_mappings = []
                                
                                for i in range(len(inputs)):
                                    if i < advantages_tensor.shape[0]:
                                        # Convert tensor to list for JSON serialization
                                        advantage_array = advantages_tensor[i].tolist()
                                        full_advantages.append(advantage_array)
                                        
                                        # Create token-to-advantage mapping using EXISTING data only
                                        response_ids = batch.batch["responses"][i]
                                        response_mask = batch.batch["response_mask"][i]
                                        advantages = advantages_tensor[i]
                                        
                                        # Use existing valid positions from response_mask
                                        valid_positions = torch.nonzero(response_mask, as_tuple=True)[0]
                                        
                                        # Create mapping using exact existing tokens and advantages
                                        token_mapping = []
                                        for pos in valid_positions:
                                            token_id = response_ids[pos].item()
                                            token_text = self.tokenizer.decode([token_id])
                                            advantage_val = advantages[pos].item()
                                            
                                            token_mapping.append({
                                                "pos": pos.item(),
                                                "token_id": token_id, 
                                                "token": token_text,
                                                "adv": advantage_val
                                            })
                                        
                                        token_advantage_mappings.append(token_mapping)
                                    else:
                                        # Fallback for missing data
                                        full_advantages.append([])
                                        token_advantage_mappings.append([])
                                
                                enhanced_reward_extra_infos_dict["advantages"] = full_advantages
                                enhanced_reward_extra_infos_dict["token_adv_map"] = token_advantage_mappings
                            
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=final_scores,  # Use final rewards instead of original scores
                                reward_extra_infos_dict=enhanced_reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1



                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

            # End of epoch. Run prompt-filter logic if configured for this boundary.
            if prompt_filter_enabled and (epoch + 1) == int(prompt_filter_cfg.get("start_epoch", 1)):
                threshold = float(prompt_filter_cfg.get("reward_threshold", 1.0))
                min_keep = prompt_filter_cfg.get("min_prompts_to_continue", None)
                if min_keep is None:
                    min_keep = int(self.config.data.train_batch_size)
                min_keep = int(min_keep)

                hard = []
                n_too_easy = 0
                n_too_hard = 0
                for pidx, vs in self._epoch_prompt_mean.items():
                    m = sum(vs) / len(vs)
                    if m <= 0.0:
                        n_too_hard += 1
                    elif m >= threshold:
                        n_too_easy += 1
                    else:
                        hard.append(pidx)
                hard.sort()

                total = len(self._epoch_prompt_mean)
                print(
                    f"[prompt_filter] epoch {epoch} done: total={total} "
                    f"useful={len(hard)} too_easy={n_too_easy} too_hard={n_too_hard} "
                    f"threshold=(0, {threshold})"
                )
                logger.log(
                    data={
                        "prompt_filter/useful": len(hard),
                        "prompt_filter/too_easy": n_too_easy,
                        "prompt_filter/too_hard": n_too_hard,
                        "prompt_filter/total_seen": total,
                    },
                    step=self.global_steps,
                )

                if len(hard) < min_keep:
                    print(
                        f"[prompt_filter] useful prompts ({len(hard)}) < min_keep ({min_keep}); "
                        f"ending training early at global_step {self.global_steps}"
                    )
                    if self.val_reward_fn is not None:
                        last_val_metrics = self._validate()
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        logger.log(data=last_val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0:
                        self._save_checkpoint()
                    progress_bar.close()
                    return

                # Rebuild train_dataloader over the filtered subset.
                from torch.utils.data import Subset
                from verl.trainer.main_ppo import create_rl_sampler

                subset = Subset(self.train_dataset, hard)
                subset_sampler = create_rl_sampler(self.config.data, subset)
                self.train_dataloader = StatefulDataLoader(
                    dataset=subset,
                    batch_size=self._train_loader_batch_size,
                    num_workers=self._train_loader_num_workers,
                    drop_last=True,
                    collate_fn=self._train_loader_collate_fn,
                    sampler=subset_sampler,
                )
                self._filter_active = True
                self._filter_hard_indices = hard
                self._epoch_prompt_mean.clear()
                print(
                    f"[prompt_filter] rebuilt train_dataloader with {len(hard)} prompts "
                    f"(steps/epoch now ~{len(self.train_dataloader)})"
                )
            elif prompt_filter_enabled:
                # Not the rebuild boundary; just reset for cleanliness so memory
                # doesn't grow across many epochs when only one boundary fires.
                self._epoch_prompt_mean.clear()