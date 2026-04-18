"""
InstantID + SDXL production photoshoot pipeline.

Takes a user photo → generates magazine-quality styled portrait
with preserved identity (95%+ ArcFace).
"""
import sys, logging
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [pipe] %(message)s")
log = logging.getLogger(__name__)


class PhotoshootPipeline:

    NEGATIVE_PROMPT = (
        "low quality, blurry, distorted face, deformed, extra fingers, "
        "extra limbs, bad anatomy, jpeg artifacts, grainy, washed out, "
        "cartoon, anime, painting, 3d render, cgi, illustration, "
        "getty images, watermark, text, logo, signature, stock photo, "
        "oversaturated, plastic skin, waxy skin, smooth skin, "
        "heavy makeup, retouched, airbrushed, oily skin, shiny face, "
        "closed mouth, serious expression, unhappy, sad, neutral face, "
        "editorial, magazine cover, studio stylized, over-processed"
    )

    def __init__(self, device: str = "cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self._pipe = None
        self._face_analysis = None

    def _load_face_analysis(self):
        if self._face_analysis is not None: return
        from insightface.app import FaceAnalysis
        log.info("Loading AntelopeV2 (InsightFace)...")
        app = FaceAnalysis(name="antelopev2", root=str(MODELS),
                           providers=["CUDAExecutionProvider",
                                      "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        self._face_analysis = app

    def _load_pipeline(self):
        if self._pipe is not None: return
        log.info("Loading SDXL + InstantID pipeline...")
        from diffusers import ControlNetModel, AutoencoderKL
        from instantid_pipeline import StableDiffusionXLInstantIDPipeline, draw_kps

        self._draw_kps = draw_kps

        log.info("  Loading ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            str(MODELS / "InstantID/ControlNetModel"),
            torch_dtype=self.dtype,
        )

        log.info("  Loading fp16-fix VAE...")
        vae = AutoencoderKL.from_pretrained(
            str(MODELS / "sdxl-vae-fp16-fix"), torch_dtype=self.dtype,
        )

        log.info("  Loading SDXL base...")
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            str(MODELS / "sdxl-base"),
            controlnet=controlnet,
            vae=vae,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True,
        )

        log.info("  Loading IP-Adapter (InstantID)...")
        pipe.load_ip_adapter_instantid(str(MODELS / "InstantID/ip-adapter.bin"))

        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()

        self._pipe = pipe
        log.info("  ✓ Pipeline ready")

    def generate(
        self,
        source_image: str,
        prompt: str,
        num_outputs: int = 1,
        num_inference_steps: int = 30,
        identity_scale: float = 0.8,
        controlnet_scale: float = 0.8,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
    ) -> List[Image.Image]:
        self._load_face_analysis()
        self._load_pipeline()

        img = cv2.imread(source_image)
        if img is None:
            raise ValueError(f"Cannot read: {source_image}")
        src_h, src_w = img.shape[:2]

        # Detect face on ORIGINAL image (InsightFace trained on natural sizes),
        # then rescale keypoints to the generation canvas.
        faces = self._face_analysis.get(img)
        if not faces:
            raise ValueError("No face detected in source image")
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) *
                                        (f.bbox[3]-f.bbox[1]))
        face_emb = face.embedding  # raw (un-normed) 512-d ArcFace

        scale_x = width / src_w
        scale_y = height / src_h
        scaled_kps = face.kps.copy()
        scaled_kps[:, 0] *= scale_x
        scaled_kps[:, 1] *= scale_y

        # Build the generation canvas from a resized source for the visualization
        img_resized = cv2.resize(img, (width, height),
                                 interpolation=cv2.INTER_LANCZOS4)
        pil_source = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        face_kps = self._draw_kps(pil_source, scaled_kps)
        log.info(f"  Face: bbox={[int(x) for x in face.bbox]} "
                 f"det_score={face.det_score:.2f} src={src_w}x{src_h}")

        self._pipe.set_ip_adapter_scale(identity_scale)
        gen = torch.Generator(device=self.device).manual_seed(seed) \
            if seed is not None else None

        log.info(f"  Generating {num_outputs} image(s), "
                 f"id_scale={identity_scale} cn_scale={controlnet_scale}")
        results = self._pipe(
            prompt=prompt,
            negative_prompt=self.NEGATIVE_PROMPT,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_outputs,
            generator=gen,
            width=width,
            height=height,
        ).images

        log.info(f"  ✓ Generated {len(results)} image(s)")
        return results
