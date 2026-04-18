"""
HYBRID Expression Editor.

Stage 1: LivePortrait → applies actual smile geometry (clear expression).
Stage 2: SDXL img2img + InstantID → polishes to photoreal quality.

Best of both worlds:
  LP gives the smile (LivePortrait is excellent at expression transfer)
  SDXL fixes the 256x256 quality bottleneck and identity drift
"""
import sys, time, logging, gc
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
AUTO_TRAIN_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTO_TRAIN_ROOT))
sys.path.insert(0, str(AUTO_TRAIN_ROOT.parent / "LivePortrait"))
MODELS = ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [hybrid] %(message)s")
log = logging.getLogger(__name__)

LP_DRIVER_SMILE = "/teamspace/studios/this_studio/LivePortrait/assets/examples/driving/d30.jpg"


class HybridEditor:

    NEGATIVE = (
        "low quality, blurry, distorted face, deformed, bad anatomy, "
        "different person, different hair, different clothes, different background, "
        "watermark, text, plastic skin, 256x256, low resolution, soft focus"
    )

    def __init__(self, base_model: str = "realvis-xl-v4"):
        self.base_path = MODELS / base_model
        self._lp = None
        self._sdxl = None
        self._fa = None

    def _load_face(self):
        if self._fa is not None: return
        from insightface.app import FaceAnalysis
        self._fa = FaceAnalysis(name="antelopev2", root=str(MODELS),
                                providers=["CPUExecutionProvider"])
        self._fa.prepare(ctx_id=0, det_size=(640, 640))

    def _load_lp(self):
        if self._lp is not None: return
        log.info("  Loading LivePortrait...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "npipe", str(AUTO_TRAIN_ROOT / "natural_pipeline.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._lp = mod.NaturalExpressionPipeline()

    def _load_sdxl(self):
        if self._sdxl is not None: return
        from diffusers import ControlNetModel, AutoencoderKL
        from instantid_img2img_pipeline import (
            StableDiffusionXLInstantIDImg2ImgPipeline)
        from instantid_pipeline import draw_kps
        self._draw_kps = draw_kps
        log.info(f"  Loading {self.base_path.name} + InstantID img2img...")
        controlnet = ControlNetModel.from_pretrained(
            str(MODELS / "InstantID/ControlNetModel"), torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(
            str(MODELS / "sdxl-vae-fp16-fix"), torch_dtype=torch.float16)
        pipe = StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained(
            str(self.base_path), controlnet=controlnet, vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        pipe.load_ip_adapter_instantid(str(MODELS / "InstantID/ip-adapter.bin"))
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        self._sdxl = pipe

    def edit(
        self,
        source_image: str,
        driver: str = LP_DRIVER_SMILE,
        lp_multiplier: float = 3.5,    # stronger smile geometry from LP
        prompt: str = "big genuine smile with teeth clearly visible, laughing, "
                      "very happy joyful expression, cheeks lifted, "
                      "crow's feet around eyes from smiling, "
                      "natural lighting, photorealistic portrait, sharp focus, "
                      "color photograph, candid moment, DSLR photo",
        refine_strength: float = 0.50,  # more SDXL regen = stronger smile
        identity_scale: float = 0.90,   # compensate: stronger identity lock
        controlnet_scale: float = 0.50, # loose pose to allow smile geometry
        seed: int = 42,
        size: int = 1024,
    ):
        self._load_face()

        # ── Stage 1: LivePortrait expression transfer ─────────────────────────
        self._load_lp()
        log.info("  [Stage 1] LivePortrait expression transfer...")
        lp_bgr = self._lp._run_lp(source_image, driver,
                                  multiplier=lp_multiplier,
                                  region="lip", use_retargeting=False)
        if lp_bgr is None:
            raise RuntimeError("LivePortrait failed to produce output")

        # Save LP intermediate so we can inspect it
        lp_path = Path("/tmp") / f"lp_intermediate_{Path(source_image).stem}.png"
        cv2.imwrite(str(lp_path), lp_bgr)
        log.info(f"    LP intermediate → {lp_path}")

        # Free LP memory before loading SDXL
        del self._lp
        self._lp = None
        gc.collect()
        torch.cuda.empty_cache()

        # ── Stage 2: SDXL img2img refinement ──────────────────────────────────
        self._load_sdxl()
        log.info("  [Stage 2] SDXL refinement at strength="
                 f"{refine_strength} id={identity_scale}...")

        # Resize LP output to SDXL size (multiple of 8)
        h0, w0 = lp_bgr.shape[:2]
        scale = size / max(h0, w0)
        nh, nw = (int(h0 * scale) // 8) * 8, (int(w0 * scale) // 8) * 8
        lp_resized = cv2.resize(lp_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        pil_lp = Image.fromarray(cv2.cvtColor(lp_resized, cv2.COLOR_BGR2RGB))

        # Use ORIGINAL source for identity (LP may have drifted slightly)
        bgr_src = cv2.imread(source_image)
        src_resized = cv2.resize(bgr_src, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        faces = self._fa.get(src_resized)
        if not faces:
            faces = self._fa.get(lp_resized)
        if not faces:
            raise RuntimeError("No face for ID conditioning")
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        face_emb = f.embedding

        # Use LP output's face keypoints (so ControlNet locks the smiling pose)
        lp_faces = self._fa.get(lp_resized)
        f_lp = max(lp_faces,
                   key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        face_kps = self._draw_kps(pil_lp, f_lp.kps)

        gen = torch.Generator(device="cuda").manual_seed(seed)
        self._sdxl.set_ip_adapter_scale(identity_scale)
        out = self._sdxl(
            prompt=prompt,
            negative_prompt=self.NEGATIVE,
            image=pil_lp,                 # ← refines LP output
            control_image=face_kps,
            image_embeds=face_emb,
            controlnet_conditioning_scale=controlnet_scale,
            strength=refine_strength,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=gen,
        ).images[0]

        return out, lp_path


def build_grid(rows):
    h = 600
    gap = 10
    pad = 50
    sized = []
    for src_p, lp_p, out_p, label, score_lp, score_out in rows:
        src = Image.open(src_p).convert("RGB")
        lp = Image.open(lp_p).convert("RGB")
        out = Image.open(out_p).convert("RGB")
        sw = int(src.width * h / src.height)
        lw = int(lp.width * h / lp.height)
        ow = int(out.width * h / out.height)
        row = Image.new("RGB", (sw + lw + ow + 2*gap, h + pad), (245, 245, 245))
        row.paste(src.resize((sw, h)), (0, pad))
        row.paste(lp.resize((lw, h)), (sw + gap, pad))
        row.paste(out.resize((ow, h)), (sw + lw + 2*gap, pad))
        d = ImageDraw.Draw(row)
        d.text((10, 16), f"SOURCE: {label}", fill=(0, 0, 0))
        d.text((sw + gap + 10, 16),
               f"LP-only  ArcFace={score_lp*100:.1f}%", fill=(140, 60, 0))
        d.text((sw + lw + 2*gap + 10, 16),
               f"LP→SDXL  ArcFace={score_out*100:.1f}%", fill=(0, 140, 0))
        sized.append(row)
    max_w = max(r.width for r in sized)
    grid = Image.new("RGB", (max_w,
                             sum(r.height for r in sized) + gap*(len(sized)-1)),
                     (255, 255, 255))
    y = 0
    for r in sized:
        grid.paste(r, (0, y)); y += r.height + gap
    return grid


def main():
    SRC_DIR = AUTO_TRAIN_ROOT / "raw_data/imgmodels_london/neutral"
    OUT_DIR = ROOT / "outputs/hybrid_edit"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TESTS = [
        "Hailey_Bieber_57d379af.jpg",
        "Freddie_Mackenzie_ddcb8ad1.jpg",
        "Mary_Ukech_7b8bb2ff.jpg",
        "Alex_Consani_62d57d93.jpg",
    ]

    editor = HybridEditor()
    rows = []
    t0 = time.time()

    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="antelopev2", root=str(MODELS),
                      providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))

    def emb(p):
        bgr = cv2.imread(str(p))
        if bgr is None: return None
        faces = fa.get(bgr)
        if not faces: return None
        f = max(faces,
                key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        return f.normed_embedding

    for i, fname in enumerate(TESTS):
        src = SRC_DIR / fname
        if not src.exists(): continue
        log.info(f"\n[{i+1}/{len(TESTS)}] {fname}")
        try:
            out_img, lp_path = editor.edit(str(src), seed=1234 + i)
        except Exception as e:
            log.error(f"  FAILED: {e}")
            continue
        out_path = OUT_DIR / f"{src.stem}_smile.png"
        out_img.save(out_path)
        lp_save = OUT_DIR / f"{src.stem}_LP.png"
        import shutil; shutil.copy2(lp_path, lp_save)
        e_src = emb(src)
        e_lp = emb(lp_save)
        e_out = emb(out_path)
        sim_lp = float(np.dot(e_src, e_lp)) if (
            e_src is not None and e_lp is not None) else 0.0
        sim_out = float(np.dot(e_src, e_out)) if (
            e_src is not None and e_out is not None) else 0.0
        log.info(f"  → {out_path.name} | LP={sim_lp*100:.2f}% "
                 f"hybrid={sim_out*100:.2f}%")
        rows.append((src, lp_save, out_path, fname.rsplit("_", 1)[0],
                     sim_lp, sim_out))

    if rows:
        grid = build_grid(rows)
        grid_path = OUT_DIR / "comparison_grid.jpg"
        grid.save(grid_path, quality=92)
        avg_lp = np.mean([r[4] for r in rows])
        avg_out = np.mean([r[5] for r in rows])
        log.info(f"\n  AVG identity:  LP-only={avg_lp*100:.2f}%  "
                 f"Hybrid={avg_out*100:.2f}%")
        log.info(f"  Grid → {grid_path}")
    log.info(f"  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
