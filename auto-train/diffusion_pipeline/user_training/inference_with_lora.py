"""
Inference using a user's trained LoRA + hybrid pipeline.

Combines:
  1. User LoRA         → memorized identity (~10-15% identity boost)
  2. InstantID         → structural face conditioning
  3. LivePortrait      → expression geometry
  4. RealVisXL base    → photorealism

Usage:
    python inference_with_lora.py \\
        --user_id john_doe \\
        --source_image user_data/john_doe/photos/neutral.jpg \\
        --prompt "ohwx person with big smile, professional headshot"
"""
import argparse, sys, json, logging, gc
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [infer] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
PIPE_ROOT = ROOT.parent
sys.path.insert(0, str(PIPE_ROOT))
sys.path.insert(0, str(PIPE_ROOT.parent))
sys.path.insert(0, str(PIPE_ROOT.parent.parent / "LivePortrait"))
MODELS = PIPE_ROOT / "models"
USER_DATA = ROOT / "user_data"

LP_DRIVER_SMILE = "/teamspace/studios/this_studio/LivePortrait/assets/examples/driving/d30.jpg"


def build_pipe(base_model: str, lora_path: Path, lora_scale: float = 0.85):
    from diffusers import ControlNetModel, AutoencoderKL
    from instantid_img2img_pipeline import StableDiffusionXLInstantIDImg2ImgPipeline
    from instantid_pipeline import draw_kps

    log.info("Loading base pipeline + InstantID + user LoRA...")
    controlnet = ControlNetModel.from_pretrained(
        str(MODELS / "InstantID/ControlNetModel"), torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(
        str(MODELS / "sdxl-vae-fp16-fix"), torch_dtype=torch.float16)
    pipe = StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained(
        str(MODELS / base_model), controlnet=controlnet, vae=vae,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.load_ip_adapter_instantid(str(MODELS / "InstantID/ip-adapter.bin"))

    # Load user LoRA
    log.info(f"Loading user LoRA from {lora_path}")
    pipe.load_lora_weights(str(lora_path))
    pipe.fuse_lora(lora_scale=lora_scale)

    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()
    return pipe, draw_kps


def run_lp(source: str, driver: str, multiplier: float = 3.5):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "npipe", str(PIPE_ROOT.parent / "natural_pipeline.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    lp = mod.NaturalExpressionPipeline()
    out = lp._run_lp(source, driver, multiplier=multiplier,
                     region="lip", use_retargeting=False)
    del lp; gc.collect(); torch.cuda.empty_cache()
    return out


def generate(user_id, source_image, prompt, driver=LP_DRIVER_SMILE,
             lp_multiplier=3.5, refine_strength=0.45,
             identity_scale=0.85, controlnet_scale=0.55,
             lora_scale=0.85, seed=42, size=1024):

    user_dir = USER_DATA / user_id
    meta = json.loads((user_dir / "meta.json").read_text())
    lora_path = user_dir / "lora"
    log.info(f"User: {user_id}  token={meta['instance_token']}  "
             f"rank={meta['rank']}")

    # Stage 1: LivePortrait
    log.info("[Stage 1] LivePortrait expression transfer...")
    lp_bgr = run_lp(source_image, driver, lp_multiplier)
    if lp_bgr is None:
        raise RuntimeError("LP failed")

    # Stage 2: SDXL + LoRA + InstantID refinement
    log.info("[Stage 2] SDXL+LoRA+InstantID refinement...")
    pipe, draw_kps = build_pipe(meta["base_model"], lora_path, lora_scale)

    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="antelopev2", root=str(MODELS),
                      providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))

    h0, w0 = lp_bgr.shape[:2]
    scale = size / max(h0, w0)
    nh = (int(h0 * scale) // 8) * 8
    nw = (int(w0 * scale) // 8) * 8
    lp_resized = cv2.resize(lp_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    pil_lp = Image.fromarray(cv2.cvtColor(lp_resized, cv2.COLOR_BGR2RGB))

    bgr_src = cv2.imread(source_image)
    src_resized = cv2.resize(bgr_src, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    src_faces = fa.get(src_resized)
    if not src_faces:
        src_faces = fa.get(lp_resized)
    if not src_faces:
        raise RuntimeError("no face for ID")
    face_emb = max(src_faces,
                   key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])
                   ).embedding

    lp_faces = fa.get(lp_resized)
    face_kps = draw_kps(pil_lp,
        max(lp_faces, key=lambda x:
            (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])).kps)

    # Prepend user token to prompt for LoRA activation
    full_prompt = f"{meta['instance_token']}, {prompt}"
    log.info(f"Prompt: {full_prompt[:100]}")

    pipe.set_ip_adapter_scale(identity_scale)
    out = pipe(
        prompt=full_prompt,
        negative_prompt="low quality, blurry, deformed, different person, "
                        "watermark, cartoon, plastic skin",
        image=pil_lp,
        control_image=face_kps,
        image_embeds=face_emb,
        controlnet_conditioning_scale=controlnet_scale,
        strength=refine_strength,
        num_inference_steps=30,
        guidance_scale=5.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", required=True)
    ap.add_argument("--source_image", required=True)
    ap.add_argument("--prompt",
                    default="big warm smile with teeth, happy, photorealistic "
                            "portrait, natural lighting")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_scale", type=float, default=0.85)
    args = ap.parse_args()

    img = generate(args.user_id, args.source_image, args.prompt,
                   seed=args.seed, lora_scale=args.lora_scale)
    out = args.output or f"user_data/{args.user_id}/output_{args.seed}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    log.info(f"✓ Saved: {out}")


if __name__ == "__main__":
    main()
