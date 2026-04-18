"""Download RealVisXL V4.0 - SDXL photorealistic fine-tune."""
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [dl] %(message)s")
log = logging.getLogger(__name__)

target = Path(__file__).parent / "models" / "realvis-xl-v4"
log.info(f"Downloading RealVisXL V4.0 → {target}")
snapshot_download(
    "SG161222/RealVisXL_V4.0",
    local_dir=str(target),
    allow_patterns=["*.json", "*.txt",
                    "unet/diffusion_pytorch_model.fp16.safetensors",
                    "vae/diffusion_pytorch_model.fp16.safetensors",
                    "text_encoder/model.fp16.safetensors",
                    "text_encoder_2/model.fp16.safetensors",
                    "tokenizer/*", "tokenizer_2/*", "scheduler/*"],
)
log.info("✓ Done")
