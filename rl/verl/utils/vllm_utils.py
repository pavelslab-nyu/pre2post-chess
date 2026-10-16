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


from typing import List

import torch
from msgspec import field
from packaging import version as vs
from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from verl.third_party.vllm import get_version

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass


def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {
        MixtralForCausalLM: "block_sparse_moe",
    }
    DEFAULT_MLP_ATTR = "mlp"

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)):
        return

    model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if model is None:
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    for layer in model.layers:
        mlp_attr = MLP_ATTR_MAPPING.get(type(model), DEFAULT_MLP_ATTR)
        mlp = getattr(layer, mlp_attr)

        param_dict = dict(mlp.named_parameters())
        for name, param in param_dict.items():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = mlp.experts.weight_loader


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors

            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
            load memory-based LoRA tensors.
            """
            try:
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_modules: List[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)

                expected_lora_modules = list(set(expected_lora_modules))

                lora_tensors = None
                from vllm.lora.peft_helper import PEFTHelper

                if isinstance(lora_request, TensorLoRARequest):
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)

                    peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)

                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(self.lora_config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                model = self._adapter_manager.model
                hf_to_vllm_mapper = None
                if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                    hf_to_vllm_mapper = model.hf_to_vllm_mapper

                if isinstance(lora_request, TensorLoRARequest):
                    lora = self._lora_model_cls.from_lora_tensors(
                        lora_model_id=lora_request.lora_int_id,
                        tensors=lora_tensors,
                        peft_helper=peft_helper,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        embeddings=None,
                        target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper,
                    )
                else:
                    lora = self._lora_model_cls.from_local_checkpoint(
                        lora_path,
                        expected_lora_modules,
                        peft_helper=peft_helper,
                        lora_model_id=lora_request.lora_int_id,
                        device="cpu",
                        dtype=self.lora_config.lora_dtype,
                        target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                        embedding_modules=self.embedding_modules,
                        embedding_padding_modules=self.embedding_padding_modules,
                        weights_mapper=hf_to_vllm_mapper,
                    )
            except Exception as e:
                raise e

            if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                raise ValueError(
                    f"LoRA added vocab size {lora.extra_vocab_size} is greater than lora_extra_vocab_size "
                    f"{self.lora_config.lora_extra_vocab_size}."
                )
            return lora

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    """check if the package version is greater than or equal to the minimum version"""
    return vs.parse(get_version(pkg)) >= vs.parse(minver)


def patch_vllm_fp32_lm_head():
    """Run vLLM's LM-head matmul in fp32 (actor_rollout_ref.model.fp32_lm_head=True).

    vLLM has no option for this, so LogitsProcessor._get_logits is patched at class level, in the rollout
    worker before the engine is built. That covers the V0 and V1 engines, sync and async rollout, eager and
    CUDA-graph mode (both engines compute logits outside the captured graph), tied and untied heads, and
    TP > 1 (the per-rank vocab shard is gathered exactly as before). The stored weights stay bf16 and are
    upcast at matmul time -- the same operands as the trainer-side fp32 head in fsdp_workers -- so the head
    adds no trainer<->rollout mismatch. Quantized heads keep vLLM's own path. LoRA wraps the processor in a
    different class (LogitsProcessorWithLoRA) that this does not reach; ray_trainer rejects that combination.
    """
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    if getattr(LogitsProcessor, "_verl_fp32_lm_head", False):
        return
    original_get_logits = LogitsProcessor._get_logits
    logged = False

    def _get_logits(self, hidden_states, lm_head, embedding_bias):
        nonlocal logged
        if not isinstance(getattr(lm_head, "quant_method", None), UnquantizedEmbeddingMethod):
            return original_get_logits(self, hidden_states, lm_head, embedding_bias)
        logits = torch.nn.functional.linear(
            hidden_states.float(),
            lm_head.weight.float(),
            None if embedding_bias is None else embedding_bias.float(),
        )
        # Same as vLLM's _get_logits from here on: gather for TP, drop vocab padding.
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
            if not logged:
                logged = True
                print(f"[fp32_lm_head] vLLM logits dtype={logits.dtype} (head weight dtype={lm_head.weight.dtype})")
        return logits

    LogitsProcessor._get_logits = _get_logits
    LogitsProcessor._verl_fp32_lm_head = True


def patch_vllm_v1_sampler_logprobs_with_temperature():
    """Make the vLLM V1 sampler return the sampled token's log-prob under softmax(logits / T).

    With actor.use_rollout_log_probs the rollout log-prob is pi_old in the PPO ratio, so it must mean what the
    trainer computes: log_softmax(logits / T)[token]. The V0 sampler already returns that (it divides by T
    before log_softmax); the V1 sampler of vLLM < 0.10 returns log-probs of the raw logits, i.e. ignores T.
    Callers only apply this when T != 1 (at T == 1 both engines already agree). Newer vLLM exposes the same
    thing as logprobs_mode="processed_logprobs", which vLLMRollout uses instead when available.
    """
    from vllm.v1.sample.sampler import Sampler

    if getattr(Sampler, "_verl_logprobs_with_temperature", False):
        return
    original_forward = Sampler.forward
    original_compute_logprobs = Sampler.compute_logprobs

    def forward(self, logits, sampling_metadata):
        # compute_logprobs runs first thing inside forward; hand it this batch's per-request temperatures.
        self._verl_temperature = None if sampling_metadata.all_greedy else sampling_metadata.temperature
        try:
            return original_forward(self, logits, sampling_metadata)
        finally:
            self._verl_temperature = None

    def compute_logprobs(self, logits):
        temperature = getattr(self, "_verl_temperature", None)
        if temperature is not None:
            # Greedy rows carry temperature -1 in V1: leave them unscaled. Divide out of place -- the same
            # logits tensor is sampled from afterwards.
            temperature = torch.where(temperature > 0, temperature, torch.ones_like(temperature))
            logits = logits.float() / temperature.unsqueeze(1)
        return original_compute_logprobs(self, logits)

    Sampler.forward = forward
    Sampler.compute_logprobs = compute_logprobs
    Sampler._verl_logprobs_with_temperature = True
