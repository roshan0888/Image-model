"""
Expression-Only Editor.
Keep everything identical except the expression.

Approach: SDXL img2img + InstantID at LOW denoising strength.
  strength=0.25 → only ~25% of pixels change, mostly mouth/face
  identity locked via InstantID embedding
"""
import sys, time, logging
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [edit] %(message)s")
log = logging.getLogger(__name__)


class ExpressionEditor:

    NEGATIVE = (
        "low quality, blurry, distorted face, deformed, bad anatomy, "
        "different person, changed identity, different hair, different clothes, "
        "different background, watermark, text, oversaturated, plastic skin, "
        "closed mouth"
    )

    def __init__(self, base_model: str = "realvis-xl-v4"):
        self.base_path = MODELS / base_model
        self._pipe = None
        self._fa = None

    def _load_face(self):
        if self._fa is not None: return
        from insightface.app import FaceAnalysis
        self._fa = FaceAnalysis(name="antelopev2", root=str(MODELS),
                                providers=["CPUExecutionProvider"])
        self._fa.prepare(ctx_id=0, det_size=(640, 640))

    def _load_pipe(self):
        if self._pipe is not None: return
        from diffusers import ControlNetModel, AutoencoderKL
        from instantid_img2img_pipeline import (
            StableDiffusionXLInstantIDImg2ImgPipeline,
        )
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
        self._pipe = pipe

    def edit(
        self,
        source_image: str,
        expression_prompt: str,
        strength: float = 0.30,
        identity_scale: float = 0.85,
        controlnet_scale: float = 0.85,
        guidance_scale: float = 5.0,
        steps: int = 30,
        seed: Optional[int] = None,
        size: int = 1024,
    ) -> Image.Image:
        self._load_face(); self._load_pipe()

        bgr = cv2.imread(source_image)
        if bgr is None:
            raise ValueError(f"Cannot read: {source_image}")

        # Resize so longest side = size, keep aspect
        h0, w0 = bgr.shape[:2]
        scale = size / max(h0, w0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        # Round to multiples of 8 for SDXL
        nh = (nh // 8) * 8
        nw = (nw // 8) * 8
        bgr_r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        faces = self._fa.get(bgr_r)
        if not faces:
            raise ValueError("No face detected")
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        face_emb = f.embedding
        face_kps = self._draw_kps(pil, f.kps)

        gen = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        log.info(f"  Editing: strength={strength} id={identity_scale} "
                 f"cn={controlnet_scale}  size={nw}x{nh}")
        self._pipe.set_ip_adapter_scale(identity_scale)
        out = self._pipe(
            prompt=expression_prompt,
            negative_prompt=self.NEGATIVE,
            image=pil,                     # ← starting image (img2img)
            control_image=face_kps,
            image_embeds=face_emb,
            controlnet_conditioning_scale=controlnet_scale,
            strength=strength,             # ← key: low = preserve composition
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]
        return out


def build_grid(rows, header_text):
    h = 700
    gap = 12
    pad = 50
    sized = []
    for src_p, out_p, label, score in rows:
        src = Image.open(src_p).convert("RGB")
        out = Image.open(out_p).convert("RGB")
        sw = int(src.width * h / src.height)
        ow = int(out.width * h / out.height)
        row = Image.new("RGB", (sw + ow + gap, h + pad), (245, 245, 245))
        row.paste(src.resize((sw, h)), (0, pad))
        row.paste(out.resize((ow, h)), (sw + gap, pad))
        d = ImageDraw.Draw(row)
        d.text((10, 16), f"SOURCE: {label}", fill=(0, 0, 0))
        score_color = (0, 140, 0) if score >= 0.75 else (
                      (200, 130, 0) if score >= 0.60 else (200, 0, 0))
        d.text((sw + gap + 10, 16),
               f"EXPRESSION-EDITED   |   ArcFace = {score*100:.2f}%",
               fill=score_color)
        sized.append(row)
    max_w = max(r.width for r in sized)
    grid = Image.new("RGB", (max_w,
                             sum(r.height for r in sized) + gap*(len(sized)-1) + 70),
                     (255, 255, 255))
    d = ImageDraw.Draw(grid)
    d.text((12, 22), header_text, fill=(0, 0, 0))
    y = 70
    for r in sized:
        grid.paste(r, (0, y)); y += r.height + gap
    return grid


def main():
    SRC_DIR = ROOT.parent / "raw_data/imgmodels_london/neutral"
    OUT_DIR = ROOT / "outputs/expression_edit"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    TESTS = [
        "Hailey_Bieber_57d379af.jpg",
        "Freddie_Mackenzie_ddcb8ad1.jpg",
        "Mary_Ukech_7b8bb2ff.jpg",
        "Alex_Consani_62d57d93.jpg",
    ]
    PROMPT = ("portrait photograph, person with warm natural smile showing teeth, "
              "happy expression, same lighting same outfit same background, "
              "photorealistic, sharp focus, color photograph")

    editor = ExpressionEditor()

    # Score using same FaceAnalysis instance
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="antelopev2", root=str(MODELS),
                      providers=["CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))

    def emb(p):
        bgr = cv2.imread(str(p))
        if bgr is None: return None
        faces = fa.get(bgr)
        if not faces: return None
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        return f.normed_embedding

    rows = []
    t0 = time.time()
    for i, fname in enumerate(TESTS):
        src = SRC_DIR / fname
        if not src.exists(): continue
        log.info(f"\n[{i+1}/{len(TESTS)}] {fname}")
        try:
            img = editor.edit(
                str(src), PROMPT,
                strength=0.30, identity_scale=0.85,
                controlnet_scale=0.85, seed=1234 + i,
            )
        except Exception as e:
            log.error(f"  FAILED: {e}")
            continue
        out = OUT_DIR / f"{src.stem}_smile.png"
        img.save(out)
        e1 = emb(src); e2 = emb(out)
        sim = float(np.dot(e1, e2)) if e1 is not None and e2 is not None else 0.0
        log.info(f"  → {out.name}  ArcFace={sim*100:.2f}%")
        rows.append((src, out, fname.rsplit("_", 1)[0], sim))

    if rows:
        avg = np.mean([r[3] for r in rows])
        grid = build_grid(rows,
            f"Expression-Only Edit (img2img str=0.30)  |  "
            f"AVG identity = {avg*100:.2f}%")
        grid_path = OUT_DIR / "comparison_grid.jpg"
        grid.save(grid_path, quality=92)
        log.info(f"\n  AVG identity: {avg*100:.2f}%")
        log.info(f"  Grid → {grid_path}")
    log.info(f"  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
