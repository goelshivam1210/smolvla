#!/usr/bin/env python3
"""
SmolVLA Policy Evaluation on LIBERO — Evaluation Script

Evaluates a fine-tuned SmolVLA checkpoint on LIBERO task suites.
Supports both in-distribution (libero_spatial) and out-of-distribution
(libero_object, libero_goal) evaluation.

Usage:
    # In-distribution eval
    python scripts/evaluate.py \
        --checkpoint ./outputs/smolvla-libero-lora-r32/checkpoints/last/pretrained_model \
        --task_suite libero_spatial

    # OOD eval on libero_object
    python scripts/evaluate.py \
        --checkpoint ./outputs/smolvla-libero-lora-r32/checkpoints/last/pretrained_model \
        --task_suite libero_object

    # OOD eval on libero_goal
    python scripts/evaluate.py \
        --checkpoint ./outputs/smolvla-libero-lora-r32/checkpoints/last/pretrained_model \
        --task_suite libero_goal

    # Multi-suite eval
    python scripts/evaluate.py \
        --checkpoint ./outputs/smolvla-libero-lora-r32/checkpoints/last/pretrained_model \
        --task_suite libero_object,libero_goal

Requirements:
    pip install "lerobot[smolvla,libero]"
    export MUJOCO_GL=egl  # for headless rendering
"""

import argparse
import os
import subprocess
import sys


def build_eval_command(args):
    """Build the lerobot-eval CLI command."""

    cmd = [
        "lerobot-eval",
        f"--policy.path={args.checkpoint}",
        f"--env.type=libero",
        f"--env.task={args.task_suite}",
        f"--eval.batch_size={args.eval_batch_size}",
        f"--eval.n_episodes={args.n_episodes}",
        f"--output_dir={args.output_dir}",
    ]

    if args.task_ids:
        cmd.append(f"--env.task_ids={args.task_ids}")

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA on LIBERO task suites"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        help="Task suite(s) to evaluate on (comma-separated)")
    parser.add_argument("--task_ids", type=str, default="[0,1,2,3,4]",
                        help="Task IDs to evaluate (default: first 5)")
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="Number of eval episodes per task")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="Eval batch size")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval",
                        help="Output directory for eval results")

    args = parser.parse_args()

    # Ensure MUJOCO_GL is set
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Build and run
    cmd = build_eval_command(args)
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Task suite: {args.task_suite}")
    print(f"Episodes per task: {args.n_episodes}")
    print(f"{'='*60}\n")
    print("Command:", " \\\n  ".join(cmd))
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()