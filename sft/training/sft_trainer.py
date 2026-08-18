# sft_trainer.py
"""
Supervised Fine-Tuning (SFT) Trainer.

This trainer implements SFT with:
- Masked loss on prompt tokens (only train on responses)
- Support for chat format with thinking tokens
- Gradient accumulation and mixed precision training
- Evaluation and checkpointing
"""
import pathlib
import os
import sys
import shutil
from datetime import timedelta
from tqdm import tqdm
from typing import Optional, Dict, Any

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from .sft_data_utils import create_sft_dataloader, create_multi_turn_sft_dataloader
from .sft_loss import SFTLoss
from llm_tokens.chess.tokenizer_factory import init_tokenizer
from .optim_sched import build_optimizer, build_scheduler
from evaluation.example_evaluator import (
    HumanGamesEvaluator, PuzzlesEvaluator, RandomGamesEvaluator,
    HumanGamesEvaluatorSFT, PuzzlesEvaluatorSFT, RandomGamesEvaluatorSFT
)
from evaluation.multiturn_puzzle_evaluator import evaluate_multiturn_puzzle
import torch
from transformers import AutoConfig, AutoModelForCausalLM


class SFTTrainer:
    """
    Trainer for Supervised Fine-Tuning (SFT).
    
    Key features:
    - Masked loss that ignores prompt tokens
    - Support for chat format with thinking tokens
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing and resuming
    """
    
    def __init__(self, cfg, run_config_path: Optional[str] = None):
        """
        Args:
            cfg: Configuration object with model, training, data, and tokenizer configs
            run_config_path: Path to the config file (for snapshotting)
        """
        self.cfg = cfg
        self.run_cfg_path = run_config_path
        
        # Setup accelerator with extended timeout for long evaluations
        backend = cfg.logging.get("backend", "wandb")
        mixed_precision = cfg.training.get("mixed_precision", "no")
        gradient_accumulation_steps = cfg.training.get("gradient_accumulation_steps", 1)
        
        # Increase NCCL timeout to 60 minutes for long evaluation runs
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
        
        self.acc = Accelerator(
            log_with=("wandb" if backend == "wandb" else backend),
            mixed_precision=mixed_precision if mixed_precision != "no" else None,
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[process_group_kwargs]
        )
        
        self._init_all()
    
    def _init_all(self):
        """Initialize all components: tokenizer, model, dataloaders, optimizer, etc."""
        cfg = self.cfg
        self.mcfg = cfg.model
        self.tcfg = cfg.training
        self.dcfg = cfg.data
        self.tokcfg = cfg.tokenizer

        sft_config = self.dcfg.get("sft", {})
        multi_turn = sft_config.get("multi_turn", False)
        self.multi_turn = multi_turn
        # Derive whether the training data uses <T>…</T> thinking from the cot_field.
        # Methods like best_move_only / solution_continuation produce plain moves (no thinking).
        _NON_THINKING_METHODS = {"best_move_only", "solution_continuation"}
        cot_field = sft_config.get("cot_field", "")
        # Extract the method name: supports both "cot_by_method.solution_continuation.cot_format"
        # (take index 1) and bare "solution_continuation" (take index 0).
        _parts = cot_field.split(".")
        _cot_method = _parts[1] if len(_parts) >= 2 else _parts[0]
        self.use_thinking = _cot_method not in _NON_THINKING_METHODS
        # How to handle <call_env> in the response (see SFTDataset.env_mode):
        #   multi_turn  -> 'keep'  (<call_env> is a real vocab token, env replies masked)
        #   single-turn -> 'strip' by default (legacy), or 'mask' when requested.
        #
        # 'strip' DELETES the opponent's reply along with the tag, which makes
        # continuation-style data (solution_continuation) an illegal move sequence
        # from the second move onwards.  Use env_mode='mask' for those: the
        # opponent's replies stay in input_ids (board stays legal) but are excluded
        # from the loss.
        _env_mode_cfg = sft_config.get("env_mode", None)
        if multi_turn:
            if _env_mode_cfg not in (None, "keep"):
                raise ValueError(
                    f"data.sft.env_mode={_env_mode_cfg!r} is incompatible with multi_turn=True "
                    "(multi-turn keeps <call_env> and masks env replies itself)"
                )
            self.env_mode = "keep"
        else:
            self.env_mode = _env_mode_cfg or "strip"
        if self.env_mode not in ("strip", "mask", "keep"):
            raise ValueError(f"data.sft.env_mode must be 'strip'/'mask'/'keep', got {self.env_mode!r}")
        self._strip_env_tokens = (self.env_mode == "strip")
        self.acc.print(f"[SFT] use_thinking={self.use_thinking} env_mode={self.env_mode} (cot_field={cot_field!r})")
        if multi_turn:
            self.acc.print("[SFT] Multi-turn mode enabled: env response tokens after <call_env> will be masked.")
            self.tokcfg["include_env_tokens"] = True

        # ----- Tokenizer -----
        self.tok = init_tokenizer(
            name=self.tokcfg.name,
            config=self.tokcfg
        )
        
        # Get vocab size
        if hasattr(self.tok, "get_vocab_size"):
            self.vocab_size = int(self.tok.get_vocab_size())
        else:
            self.vocab_size = int(len(self.tok.get_vocab()))
        
        self.acc.print(f"[SFT] Vocab size: {self.vocab_size}")
        
        # ----- Data files -----
        train_files = self._get_data_files(self.dcfg.get("train_files", []))
        eval_files = self._get_data_files(self.dcfg.get("eval_files", []))

        # Limit training files if max_train_files is set
        max_train_files = self.dcfg.get("max_train_files", None)
        if max_train_files is not None and len(train_files) > max_train_files:
            train_files = train_files[:max_train_files]
            self.acc.print(f"[SFT] max_train_files={max_train_files}: using {len(train_files)} training files")
        
        # Support eval_holdout: hold out last N files from train_files for evaluation
        eval_holdout = self.dcfg.get("eval_holdout", 0)
        if eval_holdout > 0 and len(train_files) > eval_holdout:
            # Sort to ensure consistent ordering
            train_files = sorted(train_files)
            # Hold out last N files for evaluation
            eval_files_holdout = train_files[-eval_holdout:]
            train_files = train_files[:-eval_holdout]
            # Combine with explicitly specified eval_files
            eval_files = eval_files + eval_files_holdout
            self.acc.print(f"[SFT] Holding out {eval_holdout} files for evaluation: {eval_files_holdout}")
        
        if not train_files:
            raise ValueError("No training files specified in config.data.train_files")
        
        self.acc.print(f"[SFT] Training files: {len(train_files)}")
        if eval_files:
            self.acc.print(f"[SFT] Evaluation files: {len(eval_files)}")
        
        # ----- Dataloaders -----
        _dataloader_fn = create_multi_turn_sft_dataloader if multi_turn else create_sft_dataloader

        self.train_loader = _dataloader_fn(
            data_files=train_files,
            tokenizer=self.tok,
            batch_size=self.tcfg.batch_size,
            seq_len=self.mcfg.block_size,
            num_workers=self.tcfg.get("num_workers", 4),
            # Default True (unchanged). Set training.shuffle_train: false to
            # train in on-disk file/sample order (e.g. ordered rollout SFT).
            shuffle=self.tcfg.get("shuffle_train", True),
            mask_prompt=sft_config.get("mask_prompt", True),
            prefetch_factor=self.tcfg.get("prefetch_factor", 2),
            persistent_workers=self.tcfg.get("persistent_workers", True),
            cot_field=sft_config.get("cot_field", "cot_format"),
            prompt_field=sft_config.get("prompt_field", "pgn"),
            strip_env_tokens=self._strip_env_tokens,
            env_mode=self.env_mode,
        )

        self.eval_loader = None
        if eval_files:
            self.eval_loader = _dataloader_fn(
                data_files=eval_files,
                tokenizer=self.tok,
                batch_size=self.tcfg.get("eval_batch_size", self.tcfg.batch_size),
                seq_len=self.mcfg.block_size,
                num_workers=0,
                shuffle=False,
                mask_prompt=sft_config.get("mask_prompt", True),
                cot_field=sft_config.get("cot_field", "cot_format"),
                prompt_field=sft_config.get("prompt_field", "pgn"),
                strip_env_tokens=self._strip_env_tokens,
                env_mode=self.env_mode,
            )
        
        # ----- Model -----
        if "pretrained_model" in self.mcfg and self.mcfg.pretrained_model:
            # HF model: directly load weights from_pretrained, then resize to new tokenizer vocab
            pretrained_model = self.mcfg.get("pretrained_model", None)

            # If the explicitly-provided path no longer exists (e.g. after a checkpoint
            # layout change) try to resolve the model directory from pretrain_spec fields
            # (total_compute + alpha + beta; modelsize is NOT used in the new layout).
            if pretrained_model and not pathlib.Path(pretrained_model).exists():
                spec = self.mcfg.get("pretrain_spec", {})
                tc   = spec.get("total_compute")
                alp  = spec.get("alpha")
                bet  = spec.get("beta")
                ms = spec.get("modelsize")
                if tc and ms and alp:
                    self.acc.print(
                        f"[SFT] pretrained_model path not found ({pretrained_model}); "
                        f"attempting spec-based resolution (C={tc}, modelsize={ms}, alpha={alp})"
                    )
                    resolved = self._resolve_pretrain_hf_model(tc, ms, alp)
                    if resolved:
                        pretrained_model = resolved
                    else:
                        raise FileNotFoundError(
                            f"[SFT] pretrained_model not found at '{self.mcfg.pretrained_model}' "
                            f"and spec-based resolution also failed for "
                            f"C_{tc}/{tc}_{ms}_alpha{alp}"
                        )
                else:
                    raise FileNotFoundError(
                        f"[SFT] pretrained_model path not found: {pretrained_model}. "
                        "Set model.pretrain_spec.{{total_compute,alpha,beta}} to enable "
                        "automatic path resolution."
                    )

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                trust_remote_code=True,
            )

            # apply new tokenizer special token ids
            self.model.config.bos_token_id = self.tok.bos_id()
            self.model.config.eos_token_id = self.tok.eos_id()
            self.model.config.pad_token_id = self.tok.pad_id()

            # resize token embeddings to match new tokenizer vocab
            new_vocab = int(self.vocab_size)
            old_vocab = int(getattr(self.model.config, "vocab_size", 0) or 0)
            if new_vocab != old_vocab:
                self.acc.print(f"[SFT] Resizing token embeddings: old_vocab={old_vocab} -> new_vocab={new_vocab}")
                # emb = self.model.get_input_embeddings().weight
                # print("[DEBUG] token_emb shape:", emb.shape)
                # print("[DEBUG] token_emb first row:", emb[0][:8])
                # print("[DEBUG] token_emb last row:", emb[-1][:8])
                self.model.resize_token_embeddings(new_vocab)
                self.model.config.vocab_size = new_vocab
                # emb = self.model.get_input_embeddings().weight
                # print("[DEBUG] token_emb shape:", emb.shape)
                # print("[DEBUG] token_emb first row:", emb[0][:8])
                # print("[DEBUG] token_emb last row:", emb[-1][:8])

            # optional: update context length in config (this does NOT resize pos embeddings automatically)
            if "block_size" in self.mcfg and self.mcfg["block_size"] is not None:
                bs = int(self.mcfg["block_size"])
                if hasattr(self.model.config, "max_position_embeddings"):
                    original_max_pos = int(self.model.config.max_position_embeddings)
                    if bs > original_max_pos:
                        self.model.config.max_position_embeddings = bs
                        self.model.config.rope_scaling = {"type": "yarn", "factor": bs / original_max_pos,
                            "original_max_position_embeddings": original_max_pos}
                if hasattr(self.model.config, "n_positions"):
                    self.model.config.n_positions = bs

            self._pretrained_loaded = True

        else:
            raise RuntimeError(
                "[SFT] No pretrained model resolved. From-scratch training was "
                "removed; SFT only fine-tunes pretrained HF checkpoints. Set "
                "model.pretrained_model (or model.pretrain_spec) to a valid checkpoint."
            )


        # Optional: Compile model
        if self.tcfg.get("compile_model", False):
            compile_mode = self.tcfg.get("compile_mode", "reduce-overhead")
            self.acc.print(f"[SFT] Compiling model with mode={compile_mode}")
            try:
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode=compile_mode)
                    self.acc.print("[SFT] Model compiled successfully")
            except Exception as e:
                self.acc.print(f"[SFT] Model compilation failed: {e}")
        
        self.acc.print("=" * 50)
        self.acc.print(f"[SFT] Model: {self.model}")
        self.acc.print("=" * 50)
        
        # ----- Loss function -----
        self.loss_fn = SFTLoss(
            ignore_index=-100,
            reduction='mean',
            label_smoothing=self.tcfg.get("label_smoothing", 0.0),
        )
        
        # ----- Optimizer -----
        self.optimizer = build_optimizer(self.model, {
            "name": self.tcfg.optimizer.name,
            "lr": self.tcfg.optimizer.lr,
            "weight_decay": self.tcfg.optimizer.get("weight_decay", 0.1),
            "betas": tuple(self.tcfg.optimizer.get("betas", (0.9, 0.95))),
            "exclude_bias_and_norm": True,
        })
        
        # ----- Scheduler -----
        self.epochs = self.tcfg.epochs
        steps_per_epoch = len(self.train_loader)
        self.scheduler = build_scheduler(
            self.optimizer,
            {
                "name": self.tcfg.scheduler.name,
                "eta_min": self.tcfg.scheduler.get("eta_min", 0.0),
                "warmup_steps": self.tcfg.scheduler.warmup_steps
            },
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            grad_accum_steps=self.acc.gradient_accumulation_steps,
        )
        
        # ----- Accelerator prep -----
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
            self.eval_loader,
        ) = self.acc.prepare(
            self.model, 
            self.optimizer, 
            self.train_loader, 
            self.scheduler, 
            self.eval_loader
        )
        
        self.acc.print(f"[SFT] Using device: {self.acc.device}")
        
        # ----- Directories & tracking -----
        run_name = self._auto_run_name(cfg)
        save_root = pathlib.Path(
            self.tcfg.get(
                "save_dir",
                "${REPO_ROOT}/LLM-Pretraining/checkpoints/sft_checkpoints",
            )
        )
        cot_type = self._resolve_cot_type(cfg)
        self.save_dir = save_root / cot_type / run_name
        self.acc.print(f"[SFT] save_root={save_root} cot_type={cot_type} run={run_name}")
        self.save_interval = self.tcfg.get("save_interval", 1000)
        self.log_interval = self.tcfg.get("log_interval", 50)
        self.eval_interval = self.tcfg.get("eval_interval", 500)
        self.resume_path = self.tcfg.get("resume", None)
        self.eval_checkpoint = self.tcfg.get("eval_checkpoint", None)
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        
        if self.acc.is_main_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trackers
        project_name = self.cfg.logging.get("project", "llm-sft")
        self.acc.print(f"[SFT] Initializing trackers: project={project_name} run={run_name}")
        self.acc.init_trackers(
            project_name,
            init_kwargs=self._tracker_init_kwargs(run_name, self._tracker_cfg())
        )
        
        if self.acc.is_main_process:
            import wandb
            if wandb.run is not None:
                wandb.run.name = run_name
                self.acc.print(f"[SFT] W&B run: {wandb.run.name} (id={wandb.run.id})")
        
        if self.acc.is_main_process:
            self._snapshot_config()
        
        # Auto-resume: if no explicit resume path, check for latest_ckpt/
        if not self.resume_path:
            latest_ckpt_dir = self.save_dir / "latest_ckpt"
            if latest_ckpt_dir.exists():
                self.resume_path = str(latest_ckpt_dir)
                self.acc.print(f"[SFT] Auto-resume detected: {self.resume_path}")

        # Resume if specified (explicit or auto-detected)
        if self.resume_path:
            self.acc.load_state(self.resume_path)
            self._load_training_state(self.resume_path)
            self.acc.print(f"[SFT] Resumed from {self.resume_path} at step {self.current_step}")
        
        # Load model from hf_model for evaluation (without optimizer)
        if self.eval_checkpoint:
            eval_checkpoint_path = pathlib.Path(self.eval_checkpoint)
            hf_model_path = eval_checkpoint_path / "hf_model"
            
            if not hf_model_path.exists():
                # Try if eval_checkpoint itself is the hf_model directory
                if eval_checkpoint_path.exists() and (eval_checkpoint_path / "config.json").exists():
                    hf_model_path = eval_checkpoint_path
                else:
                    raise ValueError(f"[SFT] eval_checkpoint path does not contain hf_model: {self.eval_checkpoint}")
            
            self.acc.print(f"[SFT] Loading model from hf_model: {hf_model_path}")
            
            # Load model weights only (no optimizer state)
            loaded_model = AutoModelForCausalLM.from_pretrained(
                str(hf_model_path),
                trust_remote_code=True,
            )
            
            # Load state dict into existing model (unwrap DDP before loading)
            state_dict = loaded_model.state_dict()
            missing, unexpected = self.acc.unwrap_model(self.model).load_state_dict(state_dict, strict=False)
            
            if missing:
                self.acc.print(f"[SFT] Missing keys when loading from hf_model: {missing}")
            if unexpected:
                self.acc.print(f"[SFT] Unexpected keys when loading from hf_model: {unexpected}")
            
            # Resize embeddings if needed
            new_vocab = int(self.vocab_size)
            old_vocab = int(getattr(loaded_model.config, "vocab_size", 0) or 0)
            if new_vocab != old_vocab:
                self.acc.print(f"[SFT] Resizing token embeddings: old_vocab={old_vocab} -> new_vocab={new_vocab}")
                self.model.resize_token_embeddings(new_vocab)
                self.model.config.vocab_size = new_vocab
            
            # Update special token ids
            self.model.config.bos_token_id = self.tok.bos_id()
            self.model.config.eos_token_id = self.tok.eos_id()
            self.model.config.pad_token_id = self.tok.pad_id()
            
            self.acc.print(f"[SFT] Successfully loaded model from {hf_model_path} (model weights only, no optimizer)")

    def _resolve_pretrain_hf_model(self, total_compute: str, modelsize: str, alpha: str) -> Optional[str]:
        """Resolve pretrain checkpoint path from spec fields.

        Searches under:
            {pretrain_root}/C_{total_compute}/{total_compute}_{modelsize}_alpha{alpha}/
        in the order: final/hf_model → final → hf_model → root itself.
        Returns the first directory that contains config.json, or None.
        """
        pretrain_root = self.mcfg.get(
            "pretrain_root",
            "${REPO_ROOT}/LLM-Pretraining/checkpoints",
        )
        # Directory layout: {pretrain_root}/C_{total_compute}/{total_compute}_{modelsize}_alpha{alpha}/
        base = pathlib.Path(pretrain_root) / f"C_{total_compute}" / f"{total_compute}_{modelsize}_alpha{alpha}"
        candidates = [
            base / "final" / "hf_model",
            base / "final",
            base / "hf_model",
            base,
        ]
        for candidate in candidates:
            if (candidate / "config.json").exists():
                self.acc.print(f"[SFT] Resolved pretrain HF model from spec: {candidate}")
                return str(candidate)
        self.acc.print(
            f"[SFT] _resolve_pretrain_hf_model: no config.json found under {base} "
            f"(tried {[str(c) for c in candidates]})"
        )
        return None

    def _resolve_cot_type(self, cfg) -> str:
        """Resolve cot_type folder name under save_dir."""
        sft_config = cfg.data.get("sft", {})
        explicit = sft_config.get("cot_type", None)
        if explicit:
            return str(explicit)

        cot_field = sft_config.get("cot_field", "")
        parts = cot_field.split(".")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        if parts and parts[0]:
            return parts[0]
        return "default"
            
    def _get_data_files(self, file_specs):
        """Expand file specifications into list of file paths."""
        from pathlib import Path
        import glob as glob_module
        files = []
        
        if isinstance(file_specs, str):
            file_specs = [file_specs]
        
        for spec in file_specs:
            p = Path(spec)
            if p.is_file():
                files.append(str(p))
            elif p.is_dir():
                # Find all JSONL and JSON files in directory
                files.extend([str(f) for f in p.glob("*.jsonl")])
                files.extend([str(f) for f in p.glob("*.json")])
            else:
                # Try glob pattern (supports absolute paths)
                files.extend(glob_module.glob(str(p)))
        
        return sorted(set(files))
    
    def _legacy_run_name(self, cfg) -> str:
        """Generate legacy run name from config."""
        m = cfg.model
        t = cfg.training
        tok = cfg.tokenizer
        warmup = t.scheduler.warmup_steps
        data_name = self.dcfg.get("data_name", "sft")
        
        pretrained = m.get("pretrained_model", None)
        model_tag = m.get("model_tag", None)
        if model_tag:
            model_name = model_tag
        elif pretrained:
            model_names = pretrained.split("/")
            if len(model_names) > 2:
                model_name = model_names[-3].split("_LanTokenizer")[0]
            else:
                model_name = model_names[-1].replace("-", "_")
        else:
            arch = m.get("architecture", "gpt2")
            model_name = f"scratch_{arch}"

        # Get cot_field from config and format for run name
        sft_config = cfg.data.get("sft", {})
        cot_field = sft_config.get("cot_field", "cot_format")
        # Use last two dot-segments joined by "_":
        # "cot_by_method.trajectory_sep.cot_format_first_move_no_labels" -> "trajectory_sep_cot_format_first_move_no_labels"
        parts = cot_field.split(".")
        cot_short = "_".join(parts[-2:]) if len(parts) >= 2 else cot_field
        
        max_train_files = self.dcfg.get("max_train_files", None)
        files_tag = f"_N{max_train_files}files" if max_train_files is not None else ""

        if pretrained:
            return f"sft_{model_name}_{tok.name}_data_{data_name}_ctx{m.block_size}_bs{t.batch_size}_lr{t.optimizer.lr}_warmup{warmup}_{cot_short}{files_tag}"
        return f"sft_{model_name}_{tok.name}_data_{data_name}_ctx{m.block_size}_L{m.n_layer}H{m.n_head}E{m.n_embed}_bs{t.batch_size}_lr{t.optimizer.lr}_warmup{warmup}_{cot_short}{files_tag}"

    def _pretrain_spec_model_id(self, m) -> str:
        """Build canonical pretrain model identifier from model.pretrain_spec."""
        spec = m.get("pretrain_spec", {})
        required = ("total_compute", "modelsize", "alpha", "beta")
        missing = [k for k in required if not spec.get(k)]
        if missing:
            raise ValueError(
                "[SFT] model.naming_scheme=pretrain_spec requires model.pretrain_spec fields: "
                + ", ".join(missing)
            )
        return (
            f"C{spec['total_compute']}_"
            f"{spec['modelsize']}_"
            f"alpha{spec['alpha']}_"
            f"beta{spec['beta']}"
        )

    def _auto_run_name(self, cfg) -> str:
        """Generate run name from config with explicit naming-scheme support."""
        m = cfg.model

        naming_scheme = m.get("naming_scheme", "legacy")
        if naming_scheme == "pretrain_spec":
            # For pretrain-spec runs, keep the SFT checkpoint name exactly aligned
            # with the canonical checkpoint identifier and avoid extra suffixes.
            return self._pretrain_spec_model_id(m)
        if naming_scheme != "legacy":
            raise ValueError(f"[SFT] Unsupported model.naming_scheme: {naming_scheme}")
        return self._legacy_run_name(cfg)
    
    def _tracker_cfg(self) -> Dict[str, Any]:
        """Get config for tracker."""
        c = self.cfg
        sft_config = c.data.get("sft", {})
        return {
            "epochs": c.training.epochs,
            "batch_size": c.training.batch_size,
            "lr": c.training.optimizer.lr,
            "seq_len": c.model.block_size,
            "vocab_size": self.vocab_size,
            "tokenizer": c.tokenizer.name,
            "training_type": "sft",
            "cot_field": sft_config.get("cot_field", "cot_format"),
            "prompt_field": sft_config.get("prompt_field", "pgn"),
            "mask_prompt": sft_config.get("mask_prompt", True),
        }
    
    def _tracker_init_kwargs(self, run_name: str, cfg: dict) -> dict:
        """Get initialization kwargs for tracker."""
        kinds = self.acc.log_with if isinstance(self.acc.log_with, (list, tuple)) else [self.acc.log_with]
        if "wandb" not in kinds:
            return {}
        
        lg = self.cfg.logging
        return {
            "wandb": {
                "entity": lg.get("entity", None),
                "name": run_name,
                "notes": lg.get("notes", None),
                "tags": lg.get("tags", []) + ["sft"],
                "config": cfg,
            }
        }
    
    def _snapshot_config(self):
        """Save a snapshot of the config file."""
        if not self.run_cfg_path:
            return
        dst = self.save_dir / "config_snapshot" / pathlib.Path(self.run_cfg_path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(self.run_cfg_path, dst)
        except Exception as e:
            self.acc.print(f"[SFT] Config snapshot failed: {e}")
    
    def _save_training_state(self, checkpoint_dir):
        """Save training state for resuming."""
        import json
        if self.acc.is_main_process:
            state_file = pathlib.Path(checkpoint_dir) / "training_state.json"
            state = {
                "step": self.current_step,
                "epoch": self.current_epoch,
            }
            with open(state_file, "w") as f:
                json.dump(state, f)
    
    def _load_training_state(self, checkpoint_dir):
        """Load training state when resuming."""
        import json
        state_file = pathlib.Path(checkpoint_dir) / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            self.current_step = state.get("step", 0)
            self.current_epoch = state.get("epoch", 0)
        else:
            self.acc.print(f"[SFT] No training_state.json found, starting from step 0")
    
    def train(self):
        """Main training loop."""
        self.model.train()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            it = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch}", 
                disable=not self.acc.is_main_process
            )
            
            for batch_idx, batch in enumerate(it):
                # Use accumulate context for gradient accumulation
                with self.acc.accumulate(self.model):
                    # Forward pass
                    input_ids = batch['input_ids']
                    labels = batch['labels']
                    attention_mask = batch['attention_mask']
                    
                    # Get model output
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = output.logits if hasattr(output, 'logits') else output
                    
                    # Compute loss with masking
                    loss, metrics = self.loss_fn(
                        logits=logits,
                        labels=labels,
                        attention_mask=attention_mask,
                        return_metrics=True
                    )
                    
                    # Backward pass
                    self.acc.backward(loss)
                    
                    # Optimizer step (only when gradients are synchronized)
                    if self.acc.sync_gradients:
                        # Gradient clipping
                        max_grad_norm = self.tcfg.get("max_grad_norm", None)
                        if max_grad_norm is not None:
                            self.acc.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.current_step += 1
                        
                        # Logging
                        if self.current_step % self.log_interval == 0:
                            self._log(epoch, self.current_step, metrics)
                        
                        # Evaluation
                        if self.current_step % self.eval_interval == 0 and self.eval_loader is not None:
                            eval_metrics = self._evaluate()
                            if eval_metrics and self.acc.is_main_process:
                                self.acc.print(f"[SFT Eval] step {self.current_step} " + 
                                             " ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()]))
                                self.acc.log(
                                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                                    step=self.current_step
                                )
                        
                        # Checkpointing
                        if self.current_step % self.save_interval == 0:
                            self._save_ckpt(self.current_step)
        
        # Final checkpoint (includes optimizer states)
        self._save_ckpt(self.current_step, is_final=True)
        self.acc.end_training()
    
    @torch.no_grad()
    def _evaluate(self, max_steps: Optional[int] = 50) -> Dict[str, float]:
        """Run comprehensive evaluation including chess-specific metrics."""
        metrics: Dict[str, float] = {}

        # ---- sync: everyone enters eval together ----
        self.acc.wait_for_everyone()

        # Basic loss evaluation on validation set (all ranks can do this safely)
        if self.eval_loader is not None:
            self.model.eval()
            total_loss = 0.0
            total_accuracy = 0.0
            total_tokens = 0
            n_batches = 0

            for i, batch in enumerate(self.eval_loader):
                if max_steps and i >= max_steps:
                    break

                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]

                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits if hasattr(output, "logits") else output

                _, batch_metrics = self.loss_fn(
                    logits=logits,
                    labels=labels,
                    attention_mask=attention_mask,
                    return_metrics=True,
                )

                total_loss += float(batch_metrics["loss"])
                total_accuracy += float(batch_metrics["accuracy"])
                total_tokens += int(batch_metrics["num_valid_tokens"])
                n_batches += 1

            if n_batches > 0:
                metrics["loss"] = total_loss / n_batches
                metrics["accuracy"] = total_accuracy / n_batches
                metrics["avg_tokens_per_batch"] = total_tokens / n_batches

        # ---- chess-specific evaluation + file I/O only on main process ----
        if self.acc.is_main_process:
            base_model = self.acc.unwrap_model(self.model)
            base_model.eval()
            if hasattr(base_model, "gradient_checkpointing_disable"):
                base_model.gradient_checkpointing_disable()

            evaluator_configs = {
                "model": base_model,
                "tokenizer": self.tok,
                "device": self.acc.device,
                "move_format": "uci",
                "batch_size": 32,
                "max_new_tokens": 1536 if self.use_thinking else 100,
                "temperature": 1.0,
                "top_k": None,
            }

            # Pick evaluator classes based on whether training data uses <T>…</T> thinking.
            # Non-thinking (continual-pretrain style) uses plain pretraining evaluators.
            _HumanEvalCls  = HumanGamesEvaluatorSFT  if self.use_thinking else HumanGamesEvaluator
            _RandomEvalCls = RandomGamesEvaluatorSFT  if self.use_thinking else RandomGamesEvaluator
            _PuzzleEvalCls = PuzzlesEvaluatorSFT      if self.use_thinking else PuzzlesEvaluator

            # human_eval_opening = _HumanEvalCls(**evaluator_configs)
            # human_eval_capture = _HumanEvalCls(**evaluator_configs)
            human_eval_random = _HumanEvalCls(**evaluator_configs)
            # human_eval_check = _HumanEvalCls(**evaluator_configs)
            # human_eval_promotion = _HumanEvalCls(**evaluator_configs)
            random_eval = _RandomEvalCls(**evaluator_configs)

            human_eval_configs = [
                # {"evaluator": human_eval_opening, "file": "${REPO_ROOT}/LLM-Pretraining/data/chess/test/human_games_test_opening.parquet", "prefix": "human_opening"},
                # {"evaluator": human_eval_capture, "file": "${REPO_ROOT}/LLM-Pretraining/data/chess/test/human_games_test_capture.parquet", "prefix": "human_capture"},
                # {"evaluator": human_eval_random, "file": "${REPO_ROOT}/LLM-Pretraining/data/chess/test/human_games_test_random.parquet", "prefix": "human_random"},
                # {"evaluator": human_eval_promotion, "file": "${REPO_ROOT}/LLM-Pretraining/data/chess/test/human_games_test_promotion.parquet", "prefix": "human_promotion"},
                # {"evaluator": human_eval_check, "file": "${REPO_ROOT}/LLM-Pretraining/data/chess/test/human_games_test_check.parquet", "prefix": "human_check"},
                # {"evaluator": random_eval, "file": "${REPO_ROOT}/LLM-Pretraining/data/chess/test/random_games_100.parquet", "prefix": "random_games"},
            ]

            puzzle_files = [
                # {"file": "${REPO_ROOT}/LLM-Pretraining/data/chess/test/puzzles_grandmaster.csv", "prefix": "puzzles_grandmaster"},
                # {"file": "${REPO_ROOT}/LLM-Pretraining/data/chess/test/test_final.csv", "prefix": "test_final"},
                # {"file": "${REPO_ROOT}/chess_reasoning/datasets/puzzles_processed_2/puzzles_processed_new.csv", "prefix": "puzzles_train_new"},
            ]

            test_data_dir = self.dcfg.get("test_data_dir", "${REPO_ROOT}/LLM-Pretraining/data")
            _sft_eval_table = [
                # (_PuzzleEvalCls, "chess/test/test_final.csv", "test_final"),
            ]

            for eval_cls, rel_path, prefix in _sft_eval_table:
                file_path = os.path.join(test_data_dir, rel_path)
                if not os.path.exists(file_path):
                    self.acc.print(f"[eval] Test file not found: {file_path}, skipping")
                    continue
                human_eval_configs.append({
                    "evaluator": eval_cls(**evaluator_configs),
                    "file": file_path,
                    "prefix": prefix,
                })

            import json

            # ── Human-games / random eval (same in both modes) ──────────── #
            for config in human_eval_configs:
                pred_dir = f"{self.save_dir}/rollouts/validation/{config['prefix']}"
                os.makedirs(pred_dir, exist_ok=True)

                output_path = f"{pred_dir}/predictions_{self.current_step}.parquet"
                output_metrics_path = f"{pred_dir}/metrics.jsonl"

                eval_metrics = config["evaluator"].evaluate_and_save_predictions(
                    config["file"],
                    output_path=output_path,
                    max_samples=100,
                    verbose=False,
                )

                record = {"step": self.current_step, "prefix": config["prefix"], **eval_metrics}
                with open(output_metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                metrics.update(
                    {f"{config['prefix']}_{k}": float(v) for k, v in eval_metrics.items() if "num" not in k}
                )
                self.acc.print(f"[SFT] Saved predictions for {config['prefix']} to {output_path}")

            # ── Puzzle eval: multi-turn or standard ──────────────────────── #
            for pcfg in puzzle_files:
                pred_dir = f"{self.save_dir}/rollouts/validation/{pcfg['prefix']}"
                os.makedirs(pred_dir, exist_ok=True)
                output_metrics_path = f"{pred_dir}/metrics.jsonl"
                output_path = f"{pred_dir}/predictions_{self.current_step}.parquet"

                if self.multi_turn:
                    eval_metrics = evaluate_multiturn_puzzle(
                        model=base_model,
                        tokenizer=self.tok,
                        puzzle_file=pcfg["file"],
                        device=self.acc.device,
                        max_samples=300,
                        max_cot_tokens=evaluator_configs["max_new_tokens"] if self.use_thinking else 100,
                        temperature=evaluator_configs["temperature"],
                        top_k=evaluator_configs["top_k"],
                        verbose=False,
                        save_path=output_path,
                        use_thinking=self.use_thinking,
                    )
                    # Flatten per_depth_acc and failure_counts into top-level keys
                    flat_metrics = {
                        k: v for k, v in eval_metrics.items()
                        if k not in ("per_puzzle_results", "per_depth_acc", "per_depth_total", "failure_counts")
                    }
                    for d, acc in eval_metrics.get("per_depth_acc", {}).items():
                        flat_metrics[f"depth{d}_acc"] = acc
                    for reason, count in eval_metrics.get("failure_counts", {}).items():
                        flat_metrics[f"fail_{reason}"] = count
                    
                   
                else:
                    puzzles_eval = _PuzzleEvalCls(**evaluator_configs)
                    output_path = f"{pred_dir}/predictions_{self.current_step}.parquet"
                    flat_metrics = puzzles_eval.evaluate_and_save_predictions(
                        pcfg["file"],
                        output_path=output_path,
                        max_samples=500,
                        verbose=False,
                    )
                    self.acc.print(f"[SFT] Saved predictions for {pcfg['prefix']} to {output_path}")

                record = {"step": self.current_step, "prefix": pcfg["prefix"], **flat_metrics}
                with open(output_metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                metrics.update(
                    {f"{pcfg['prefix']}_{k}": float(v) for k, v in flat_metrics.items() if "num" not in k}
                )
                self.acc.print(f"[SFT] Puzzle eval ({pcfg['prefix']}) done (multi_turn={self.multi_turn})")

        # ---- sync: everyone waits until main process finishes chess eval/IO ----
        self.acc.wait_for_everyone()

        # Put model back to train mode on all ranks
        self.model.train()
        return metrics

    @torch.no_grad()
    def evaluate_pass_at_k(self, k: int = 5, max_samples: int = 200):
        """
        Run pass@k evaluation on all configured evaluation datasets.
        
        Args:
            k: Number of samples to generate per state for pass@k evaluation
            max_samples: Maximum number of samples to evaluate per dataset
        
        Returns:
            Dictionary of metrics for all evaluation datasets
        """
        metrics = {}
        
        base_model = self.acc.unwrap_model(self.model)
        base_model.eval()
        if hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
        
        # Create evaluators fresh with current model
        evaluator_configs = {
            "model": base_model, 
            "tokenizer": self.tok,
            "device": self.acc.device,
            "move_format": "uci",
            "batch_size": 16,
            "max_new_tokens": 1536,
            "temperature": 1.0,
            "top_k": None
        }
        
        if not self.acc.is_main_process:
            return metrics
        
        self.acc.print(f"\n{'='*60}")
        self.acc.print(f"Running PASS@{k} Evaluation")
        self.acc.print(f"{'='*60}\n")
        
        test_data_dir = self.dcfg.get("test_data_dir", "${REPO_ROOT}/LLM-Pretraining/data")
        _pass_at_k_table = [
            (HumanGamesEvaluator, "chess/test_large/human_games_test_opening.parquet", "human_opening"),
            (HumanGamesEvaluator, "chess/test_large/human_games_test_capture.parquet", "human_capture"),
            (HumanGamesEvaluator, "chess/test_large/human_games_test_random.parquet", "human_random"),
            (HumanGamesEvaluator, "chess/test_large/human_games_test_promotion.parquet", "human_promotion"),
            (HumanGamesEvaluator, "chess/test_large/human_games_test_check.parquet", "human_check"),
            (PuzzlesEvaluator, "chess/test/puzzles_grandmaster.csv", "puzzles_grandmaster"),
            (PuzzlesEvaluator, "chess/test/puzzles_test.csv", "puzzles_test"),
            (RandomGamesEvaluator, "chess/test/random_games_100.parquet", "random_games"),
        ]

        eval_configs = []
        for eval_cls, rel_path, prefix in _pass_at_k_table:
            file_path = os.path.join(test_data_dir, rel_path)
            if not os.path.exists(file_path):
                continue
            eval_configs.append({
                "evaluator": eval_cls(**evaluator_configs),
                "file": file_path,
                "prefix": prefix,
            })

        if not eval_configs:
            self.acc.print(f"[pass@k] No test files found under {test_data_dir}, skipping")
            return metrics
        
        for config in eval_configs:
            self.acc.print(f"\nEvaluating {config['prefix']}...")
            pred_dir = f"{self.save_dir}/rollouts/pass_at_k/{config['prefix']}"
            os.makedirs(pred_dir, exist_ok=True)

            output_path = f"{pred_dir}/predictions_pass_at_{k}.parquet"
            output_metrics_path = f"{pred_dir}/metrics_pass_at_k.jsonl"
            
            try:
                eval_metrics = config["evaluator"].evaluate_pass_at_k(
                    config["file"],
                    k=k,
                    max_samples=max_samples,
                    verbose=True,
                    output_path=output_path
                )
                
                import json
                record = {"k": k, "prefix": config["prefix"], **eval_metrics}
                with open(output_metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                metrics.update({
                    f"{config['prefix']}_{k}": v
                    for k, v in eval_metrics.items()
                    if "num" not in k
                })
                
                self.acc.print(f"✓ {config['prefix']}: Pass@{k} = {eval_metrics.get('pass_at_k', 0):.2%}")
                self.acc.print(f"  Saved to {output_path}")
                
            except Exception as e:
                self.acc.print(f"✗ Error evaluating {config['prefix']}: {e}")
                import traceback
                traceback.print_exc()
        
        self.acc.print(f"\n{'='*60}")
        self.acc.print(f"PASS@{k} Evaluation Complete")
        self.acc.print(f"{'='*60}\n")
        
        return metrics

    def _log(self, epoch: int, step: int, metrics: Dict[str, Any]):
        """Log training metrics."""
        if self.acc.is_main_process:
            lr = self.scheduler.get_last_lr()[0]
            self.acc.print(
                f"[SFT] epoch {epoch} step {step} "
                f"loss={metrics['loss']:.4f} "
                f"acc={metrics['accuracy']:.4f} "
                f"lr={lr:.2e}"
            )
            self.acc.log({
                "train/loss": metrics['loss'],
                "train/accuracy": metrics['accuracy'],
                "lr": lr,
                "epoch": epoch
            }, step=step)
    
    def _save_ckpt(self, step: int, is_final: bool = False):
        """Save checkpoint.

        Every checkpoint:
          step_{N}/        — HF model (weights + config + tokenizer) for inference/eval
          latest_ckpt/     — full accelerate state (model + optimizer + scheduler),
                             overwritten each time; used for resume
        Final checkpoint additionally:
          step_{final}/optimizer_states/  — archived full accelerate state
          final/                          — final HF model files for parity with pretrain layout
          final/optimizer_states/         — final optimizer/scheduler/etc state
        """
        if self.acc.is_main_process:
            out = self.save_dir / f"step_{step}"
            out.mkdir(parents=True, exist_ok=True)

            # HF model always saved at step_{N}/
            if self.tcfg.get("save_hf_format", True):
                self._save_hf_model(out)
            self._save_training_state(out)

            # Rolling resumable checkpoint (overwritten each time)
            latest_ckpt_dir = self.save_dir / "latest_ckpt"
            latest_ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.acc.save_state(str(latest_ckpt_dir))
            self._save_training_state(latest_ckpt_dir)

            # Final: also archive optimizer states alongside the HF model
            if is_final:
                opt_dir = out / "optimizer_states"
                opt_dir.mkdir(parents=True, exist_ok=True)
                self.acc.save_state(str(opt_dir))
                self._save_training_state(opt_dir)

                # Pretrain-style final export:
                #   final/ + final/optimizer_states
                final_dir = self.save_dir / "final"
                final_opt_dir = final_dir / "optimizer_states"
                final_dir.mkdir(parents=True, exist_ok=True)
                final_opt_dir.mkdir(parents=True, exist_ok=True)

                self._save_hf_model(final_dir)
                self.acc.save_state(str(final_opt_dir))
                self._save_training_state(final_opt_dir)
                self._save_training_state(final_dir)

            # Update latest pointer (tracks which step is current)
            (self.save_dir / "latest").write_text(f"step_{step}")

            label = "Final checkpoint" if is_final else "Checkpoint"
            self.acc.print(f"[SFT] {label} saved → {out}")

    def _save_hf_model(self, path):
        from .hf_tokenizer_utils import save_hf_tokenizer
        import json
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        base_model = self.acc.unwrap_model(self.model)

        if hasattr(base_model, 'save_pretrained'):
            base_model.save_pretrained(str(path), safe_serialization=True)
        else:
            from safetensors.torch import save_file
            save_file(base_model.state_dict(), path / "model.safetensors")

        # Ensure config.json is always written correctly
        if hasattr(base_model, 'config'):
            # Update config fields that may have changed during training
            base_model.config.vocab_size = self.vocab_size
            base_model.config.bos_token_id = self.tok.bos_id()
            base_model.config.eos_token_id = self.tok.eos_id()
            base_model.config.pad_token_id = self.tok.pad_id()
            if self.multi_turn:
                base_model.config.env_token_id = self.tok.call_env_id()
            if self.mcfg.get("block_size"):
                if hasattr(base_model.config, 'max_position_embeddings'):
                    base_model.config.max_position_embeddings = self.mcfg.block_size
            # save_pretrained already wrote it above, but re-save to pick up updates
            base_model.config.save_pretrained(str(path))
        else:
            # Custom nn.Module with no HF config — build a minimal one
            config = {
                "vocab_size": self.vocab_size,
                "block_size": self.mcfg.block_size,
                "n_layer": self.mcfg.n_layer,
                "n_head": self.mcfg.n_head,
                "n_embed": self.mcfg.n_embed,
                "bos_token_id": self.tok.bos_id(),
                "eos_token_id": self.tok.eos_id(),
                "pad_token_id": self.tok.pad_id(),
                "env_token_id": self.tok.call_env_id() if self.multi_turn else None,
            }
            with open(path / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        env_id = self.tok.call_env_id() if self.multi_turn and hasattr(self.tok, "call_env_id") else None
        save_hf_tokenizer(
            tokenizer=self.tok,
            tokcfg=self.tokcfg,
            save_directory=path,
            model_max_length=self.mcfg.get("block_size", 2048),
            env_id=env_id,
        )
        self.acc.print(f"[SFT] HuggingFace model + tokenizer saved → {path}")

