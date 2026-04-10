"""
Download InstantID weights + SDXL base model.
Run once: python instantid_pipeline/setup_weights.py
"""
import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

def download_all():
    print("=" * 60)
    print("  INSTANTID WEIGHT DOWNLOAD")
    print("=" * 60)

    # 1. InstantID IP-Adapter checkpoint
    ip_adapter_path = WEIGHTS_DIR / "ip-adapter.bin"
    if not ip_adapter_path.exists():
        print("\n[1/3] Downloading InstantID IP-Adapter...")
        hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ip-adapter.bin",
            local_dir=str(WEIGHTS_DIR),
        )
        print("  Done: ip-adapter.bin")
    else:
        print("[1/3] InstantID IP-Adapter already exists")

    # 2. InstantID ControlNet
    controlnet_path = WEIGHTS_DIR / "ControlNetModel"
    if not controlnet_path.exists():
        print("\n[2/3] Downloading InstantID ControlNet...")
        snapshot_download(
            repo_id="InstantX/InstantID",
            allow_patterns=["ControlNetModel/*"],
            local_dir=str(WEIGHTS_DIR),
        )
        print("  Done: ControlNetModel/")
    else:
        print("[2/3] InstantID ControlNet already exists")

    # 3. SDXL base model (we'll use a lighter variant for T4)
    sdxl_path = WEIGHTS_DIR / "sdxl"
    if not sdxl_path.exists():
        print("\n[3/3] Downloading SDXL base (this takes a while)...")
        snapshot_download(
            repo_id="stabilityai/stable-diffusion-xl-base-1.0",
            allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer/*", "tokenizer_2/*"],
            ignore_patterns=["*.bin", "*.onnx", "*.msgpack"],
            local_dir=str(sdxl_path),
        )
        print("  Done: SDXL base")
    else:
        print("[3/3] SDXL base already exists")

    # 4. antelopev2 for face analysis (we already have this)
    ante_path = Path(__file__).parent.parent / "MagicFace/third_party_files/models/antelopev2"
    if ante_path.exists():
        print(f"\n  antelopev2 found at: {ante_path}")
    else:
        print("\n  WARNING: antelopev2 not found — InstantID needs it for face embedding")

    print("\n" + "=" * 60)
    print("  ALL WEIGHTS READY")
    print("=" * 60)


if __name__ == "__main__":
    download_all()
