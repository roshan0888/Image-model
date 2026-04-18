"""
Train a per-user LoRA on 10-20 photos of one person.

This is THE critical training step for production-quality output.
Each trained LoRA captures the user's exact facial details — what
InstantID alone cannot do.

Usage:
    python train_user_lora.py \\
        --user_id john_doe \\
        --photos_dir user_data/john_doe/photos \\
        --steps 1000

On T4 (15GB VRAM), trains in ~20-25 minutes.
Output: user_data/<user_id>/lora/pytorch_lora_weights.safetensors (~50MB)
"""
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent
MODELS = ROOT.parent / "models"
SCRIPTS = ROOT / "scripts"
USER_DATA = ROOT / "user_data"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", required=True,
                    help="unique identifier (e.g. 'john_doe')")
    ap.add_argument("--photos_dir", required=True,
                    help="directory with 10-20 photos of the person")
    ap.add_argument("--base_model", default="realvis-xl-v4",
                    help="base model dir name under /models/")
    ap.add_argument("--instance_token", default="ohwx person",
                    help="rare token to bind user's identity to")
    ap.add_argument("--rank", type=int, default=16,
                    help="LoRA rank (higher=more detail, more VRAM)")
    ap.add_argument("--steps", type=int, default=1000,
                    help="training steps (500-1500 typical)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def count_photos(photos_dir: Path) -> int:
    return sum(1 for _ in photos_dir.glob("*.[jp][pn]g")) + \
           sum(1 for _ in photos_dir.glob("*.jpeg"))


def main():
    args = parse_args()
    photos_dir = Path(args.photos_dir).resolve()
    if not photos_dir.is_dir():
        sys.exit(f"ERROR: photos_dir not found: {photos_dir}")

    n = count_photos(photos_dir)
    if n < 5:
        sys.exit(f"ERROR: need >=5 photos, found {n}")
    if n < 10:
        print(f"WARNING: only {n} photos. 10-20 recommended.")
    print(f"Training on {n} photos from {photos_dir}")

    user_dir = USER_DATA / args.user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = user_dir / "lora"
    lora_dir.mkdir(exist_ok=True)

    base_model = MODELS / args.base_model
    vae_path = MODELS / "sdxl-vae-fp16-fix"

    # Command for diffusers' official train_dreambooth_lora_sdxl.py
    # T4-tuned: fp16, 8bit optim, gradient checkpoint, xformers off
    cmd = [
        "accelerate", "launch", "--mixed_precision=fp16",
        str(SCRIPTS / "train_dreambooth_lora_sdxl.py"),
        f"--pretrained_model_name_or_path={base_model}",
        "--variant=fp16",
        f"--instance_data_dir={photos_dir}",
        f"--output_dir={lora_dir}",
        f"--instance_prompt=a photo of {args.instance_token}",
        f"--resolution={args.resolution}",
        f"--train_batch_size={args.batch_size}",
        "--gradient_accumulation_steps=4",
        "--gradient_checkpointing",
        f"--learning_rate={args.lr}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={args.steps}",
        f"--rank={args.rank}",
        f"--seed={args.seed}",
        "--mixed_precision=fp16",
        "--use_8bit_adam",
        "--enable_xformers_memory_efficient_attention" if False else "",
        "--checkpointing_steps=500",
        "--validation_epochs=25",
        "--report_to=tensorboard",
    ]
    cmd = [c for c in cmd if c]

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"  user:       {args.user_id}")
    print(f"  base:       {args.base_model}")
    print(f"  token:      {args.instance_token}")
    print(f"  rank:       {args.rank}")
    print(f"  steps:      {args.steps}")
    print(f"  photos:     {n}")
    print(f"  output:     {lora_dir}")
    print(f"  expected:   ~20 min on T4")
    print("=" * 60 + "\n")

    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"Training failed with exit code {rc}")

    # Record metadata
    meta = user_dir / "meta.json"
    import json
    meta.write_text(json.dumps({
        "user_id": args.user_id,
        "instance_token": args.instance_token,
        "base_model": args.base_model,
        "rank": args.rank,
        "steps": args.steps,
        "n_photos": n,
    }, indent=2))

    print("\n" + "=" * 60)
    print(f"✓ DONE  Load this LoRA with:")
    print(f"  python inference_with_lora.py --user_id {args.user_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
