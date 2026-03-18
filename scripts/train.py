#!/usr/bin/env python3
"""
SmolVLA Fine-Tuning on LIBERO — Training Script

Fine-tunes SmolVLA (450M params) on the LIBERO-Spatial benchmark using LoRA.
Wraps the lerobot-train CLI with configurable hyperparameters.

Usage:
    # Primary run (LoRA r=32)
    python scripts/train.py --run_name smolvla-libero-lora-r32 --lora_rank 32

    # Ablation run (LoRA r=8)
    python scripts/train.py --run_name smolvla-libero-lora-r8 --lora_rank 8

    # Full fine-tuning (no LoRA)
    python scripts/train.py --run_name smolvla-libero-full --no_lora

Requirements:
    pip install "lerobot[smolvla,libero]" wandb
    export MUJOCO_GL=egl  # for headless rendering
"""

import argparse
import os
import subprocess
import sys


def build_train_command(args):
    """Build the lerobot-train CLI command from arguments."""

    cmd = [
        "lerobot-train",
        f"--policy.path={args.base_model}",
        f"--policy.repo_id={args.hf_user}/{args.run_name}",
        f"--dataset.repo_id={args.dataset}",
        f"--env.type=libero",
        f"--env.task={args.task_suite}",
        f"--env.task_ids={args.eval_task_ids}",
        "--rename_map={"
        '"observation.images.image": "observation.images.camera1", '
        '"observation.images.image2": "observation.images.camera2"'
        "}",
        f"--output_dir={args.output_dir}/{args.run_name}",
        f"--job_name={args.run_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--optimizer.lr={args.lr}",
        f"--eval_freq={args.eval_freq}",
        f"--save_freq={args.save_freq}",
        f"--eval.batch_size={args.eval_batch_size}",
        f"--eval.n_episodes={args.eval_n_episodes}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.use_amp).lower()}",
        f"--wandb.enable={str(args.wandb).lower()}",
        f"--wandb.project={args.wandb_project}",
    ]

    # Add LoRA config if enabled
    if not args.no_lora:
        cmd.extend([
            f"--peft.method_type=LORA",
            f"--peft.r={args.lora_rank}",
        ])

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLA on LIBERO with LoRA"
    )

    # Run identification
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name for this training run")
    parser.add_argument("--hf_user", type=str, default="goelshivam1210",
                        help="HuggingFace username for model upload")

    # Model and data
    parser.add_argument("--base_model", type=str, default="lerobot/smolvla_base",
                        help="Base model checkpoint")
    parser.add_argument("--dataset", type=str, default="HuggingFaceVLA/libero",
                        help="Dataset repo ID")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        help="LIBERO task suite for training")

    # Training hyperparameters
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")

    # LoRA config
    parser.add_argument("--no_lora", action="store_true", default=False,
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (default: 32)")

    # Evaluation during training
    parser.add_argument("--eval_freq", type=int, default=2500,
                        help="Evaluate every N steps")
    parser.add_argument("--save_freq", type=int, default=2500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Eval batch size")
    parser.add_argument("--eval_n_episodes", type=int, default=3,
                        help="Number of eval episodes per task")
    parser.add_argument("--eval_task_ids", type=str, default="[0,1,2,3,4]",
                        help="Task IDs to evaluate on")

    # Logging
    parser.add_argument("--wandb", action="store_true", default=True,
                        help="Enable W&B logging")
    parser.add_argument("--no_wandb", action="store_true", default=False,
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="smolvla-libero",
                        help="W&B project name")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Base output directory")

    args = parser.parse_args()

    if args.no_wandb:
        args.wandb = False

    # Ensure MUJOCO_GL is set for headless rendering
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Build and run command
    cmd = build_train_command(args)
    print(f"\n{'='*60}")
    print(f"Starting training: {args.run_name}")
    print(f"LoRA: {'disabled' if args.no_lora else f'rank={args.lora_rank}'}")
    print(f"Steps: {args.steps}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"{'='*60}\n")
    print("Command:", " \\\n  ".join(cmd))
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
