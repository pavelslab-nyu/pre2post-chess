#!/usr/bin/env python
"""
Script to run Supervised Fine-Tuning (SFT) with Chain-of-Thought data.

Usage:
    # Single GPU
    python -m training.run_sft --config config/configs/chess_sft_cot.yaml
    
    # Multi-GPU with accelerate
    accelerate launch --num_processes 4 -m training.run_sft --config config/configs/chess_sft_cot.yaml
    
    # Resume from checkpoint
    python -m training.run_sft --config config/configs/chess_sft_cot.yaml --resume checkpoints/sft/run_name/step1000
"""
import sys, pathlib, argparse
sft_root = pathlib.Path(__file__).resolve().parents[2]   # sft/ (config, training, model, ...)
shared_root = sft_root.parent                            # repo root (holds shared llm_tokens/)
for _p in (shared_root, sft_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


from config import load_config
from training.sft_trainer import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--naming-scheme",
        type=str,
        choices=["legacy", "pretrain_spec"],
        default=None,
        help="Run naming scheme override; pretrain_spec uses C{total_compute}_{modelsize}_alpha{alpha}_beta{beta}",
    )
    parser.add_argument(
        "--total-compute",
        type=str,
        default=None,
        help="Pretrain spec: total compute token used in canonical model ID (for pretrain_spec naming)",
    )
    parser.add_argument(
        "--modelsize",
        type=str,
        default=None,
        help="Pretrain spec: model size token used in canonical model ID (for pretrain_spec naming)",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default=None,
        help="Pretrain spec: alpha token used in canonical model ID (for pretrain_spec naming)",
    )
    parser.add_argument(
        "--beta",
        type=str,
        default=None,
        help="Pretrain spec: beta token used in canonical model ID (for pretrain_spec naming)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from"
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default=None,
        help="Override pretrained weights path from config"
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Override pretrained model path from config"
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Short tag used in auto run name (overrides derived model name)"
    )
    parser.add_argument(
        "--cot-field",
        type=str,
        default=None,
        help="Override data.sft.cot_field from config"
    )
    parser.add_argument(
        "--cot-type",
        type=str,
        default=None,
        help="Override data.sft.cot_type; used as subfolder under sft checkpoint root",
    )
    parser.add_argument(
        "--env-mode",
        type=str,
        choices=["strip", "mask", "keep"],
        default=None,
        help=(
            "Override data.sft.env_mode: how to handle <call_env> in the response. "
            "'strip' deletes the tag AND the opponent's reply (legacy; makes "
            "continuation data an illegal move sequence). 'mask' deletes only the "
            "tag, keeps the opponent's replies in the sequence and excludes them "
            "from the loss. 'keep' leaves the text untouched (multi-turn)."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override model.block_size from config"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, don't train"
    )
    parser.add_argument("--eval-pass-at-k", action="store_true",
                   help="Run pass@k evaluation (requires --eval-only)")
    parser.add_argument(
        "--train-files",
        type=str,
        default=None,
        help="Override data.train_files from config (path to directory or file)"
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default=None,
        help="Short name for the dataset used in the wandb run name (sets data.data_name)"
    )
    parser.add_argument(
        "--max-train-files",
        type=int,
        default=None,
        help="Limit the number of training files used (sets data.max_train_files)"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Override logging.entity (wandb entity/team name)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Optional naming mode + pretrain spec metadata for deterministic run naming
    if args.naming_scheme:
        cfg.model.naming_scheme = args.naming_scheme

    if any(v is not None for v in (args.total_compute, args.modelsize, args.alpha, args.beta)):
        if not cfg.model.get("pretrain_spec"):
            cfg.model.pretrain_spec = {}
        if args.total_compute is not None:
            cfg.model.pretrain_spec.total_compute = args.total_compute
        if args.modelsize is not None:
            cfg.model.pretrain_spec.modelsize = args.modelsize
        if args.alpha is not None:
            cfg.model.pretrain_spec.alpha = args.alpha
        if args.beta is not None:
            cfg.model.pretrain_spec.beta = args.beta

    if args.naming_scheme == "pretrain_spec":
        required = ("total_compute", "modelsize", "alpha", "beta")
        missing = [k for k in required if not cfg.model.get("pretrain_spec", {}).get(k)]
        if missing:
            raise ValueError(
                "[SFT] naming-scheme=pretrain_spec requires pretrain spec fields: "
                + ", ".join(missing)
            )

    # Override settings from command line args
    if args.resume:
        cfg.training.resume = args.resume
    
    if args.pretrained_weights:
        cfg.training.pretrained_weights = args.pretrained_weights

    if args.pretrained_model:
        cfg.model.pretrained_model = args.pretrained_model

    if args.model_tag:
        cfg.model.model_tag = args.model_tag

    if args.cot_field:
        if not cfg.data.get("sft"):
            cfg.data.sft = {}
        cfg.data.sft.cot_field = args.cot_field

    if args.cot_type:
        if not cfg.data.get("sft"):
            cfg.data.sft = {}
        cfg.data.sft.cot_type = args.cot_type

    if args.env_mode:
        if not cfg.data.get("sft"):
            cfg.data.sft = {}
        cfg.data.sft.env_mode = args.env_mode

    if args.block_size is not None:
        cfg.model.block_size = args.block_size

    if args.lr is not None:
        cfg.training.optimizer.lr = args.lr

    if args.train_files:
        cfg.data.train_files = args.train_files

    if args.data_name:
        cfg.data.data_name = args.data_name

    if args.max_train_files is not None:
        cfg.data.max_train_files = args.max_train_files

    if args.wandb_entity:
        cfg.logging.entity = args.wandb_entity

    # Create trainer
    print("=" * 70)
    print(f"[SFT] Starting Supervised Fine-Tuning")
    print(f"[SFT] Config: {args.config}")
    if args.resume:
        print(f"[SFT] Resuming from: {args.resume}")
    if cfg.training.get("pretrained_weights"):
        print(f"[SFT] Loading pretrained weights: {cfg.training.pretrained_weights}")
    if args.lr is not None:
        print(f"[SFT] Learning rate override: {args.lr}")
    print("=" * 70)
    
    if args.eval_only:
        eval_checkpoint = cfg.training.get("eval_checkpoint", None)
        if eval_checkpoint:
            # Override resume path to load checkpoint for evaluation
            # cfg.training.resume = eval_checkpoint
            cfg.training.save_dir = cfg.training.get("eval_save_dir", cfg.training.get("save_dir", "checkpoints/gpt"))
    
    trainer = SFTTrainer(cfg, run_config_path=args.config)
    
    if args.eval_only:
        print("\n[SFT] Running evaluation only...")
        if args.eval_pass_at_k:
            # Run pass@k evaluation
            k = cfg.training.get("pass_at_k", 5)  # Get k from config, default to 5
            max_samples = cfg.training.get("eval_max_samples", 100)
            trainer.acc.print(f"Running pass@{k} evaluation (max_samples={max_samples})")
            metrics = trainer.evaluate_pass_at_k(k=k, max_samples=max_samples)
            trainer.acc.print("\nEvaluation metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    trainer.acc.print(f"  {key}: {value:.4f}")
        else:
            metrics = trainer._evaluate()
        if metrics:
            print("\n[SFT] Evaluation results:")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    else:
        # Start training
        print("\n[SFT] Starting training...")
        trainer.train()
        print("\n[SFT] Training complete!")


if __name__ == "__main__":
    main()

