"""
Download InstantID + SDXL + IP-Adapter models.
Total size: ~13GB. Runs once. Resumable.
"""
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [dl] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def dl_instantid():
    log.info("Downloading InstantID (1.7GB)...")
    (MODELS / "InstantID").mkdir(exist_ok=True)
    for f in ["ControlNetModel/config.json",
              "ControlNetModel/diffusion_pytorch_model.safetensors",
              "ip-adapter.bin"]:
        hf_hub_download("InstantX/InstantID", f, local_dir=str(MODELS / "InstantID"))
    log.info("  ✓ InstantID")


def dl_antelopev2():
    log.info("Downloading AntelopeV2 face encoder for InstantID...")
    target = MODELS / "antelopev2"
    target.mkdir(exist_ok=True)
    snapshot_download("DIAMONIK7777/antelopev2", local_dir=str(target))
    log.info("  ✓ AntelopeV2")


def dl_sdxl():
    log.info("Downloading SDXL base 1.0 (~7GB)...")
    target = MODELS / "sdxl-base"
    snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
                      local_dir=str(target),
                      allow_patterns=["*.json", "*.txt",
                                      "unet/diffusion_pytorch_model.fp16.safetensors",
                                      "vae/diffusion_pytorch_model.fp16.safetensors",
                                      "text_encoder/model.fp16.safetensors",
                                      "text_encoder_2/model.fp16.safetensors",
                                      "tokenizer/*", "tokenizer_2/*",
                                      "scheduler/*"])
    log.info("  ✓ SDXL base")


def dl_vae():
    log.info("Downloading SDXL VAE fp16 fix (~300MB)...")
    target = MODELS / "sdxl-vae-fp16-fix"
    snapshot_download("madebyollin/sdxl-vae-fp16-fix", local_dir=str(target))
    log.info("  ✓ VAE fp16 fix")


def main():
    log.info(f"Target dir: {MODELS}")
    log.info(f"Total download: ~9GB (fp16 variants)")
    log.info("")

    dl_instantid()
    dl_antelopev2()
    dl_sdxl()
    dl_vae()

    log.info("")
    log.info("=" * 60)
    log.info("ALL MODELS DOWNLOADED")
    log.info(f"  Location: {MODELS}")


if __name__ == "__main__":
    main()
