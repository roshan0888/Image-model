"""
InstantID Expression + Pose Engine
====================================
Completely separate from LivePortrait. Uses SDXL + InstantID for
identity-preserving face generation at 1024x1024.

Capabilities:
  - Change expression (smile, neutral, serious, happy, confident)
  - Change pose (frontal, slight left, slight right, 3/4 view)
  - Change composition (headshot, portrait, upper body)
  - All at 1024x1024 resolution with identity preservation

Usage:
    from instantid_pipeline.instantid_engine import InstantIDEngine

    engine = InstantIDEngine()
    result = engine.generate(
        source_path="photo.jpg",
        expression="warm professional smile",
        pose="slight right turn",
        background="studio gray gradient",
    )
    # result["image"] = PIL Image at 1024x1024
    # result["identity_score"] = ArcFace cosine similarity
"""

import os, sys, cv2, time, logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [InstantID] %(message)s")

WEIGHTS_DIR = Path(__file__).parent / "weights"
OUTPUT_DIR  = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Expression + Pose Prompt Templates ────────────────────────────────────────

EXPRESSION_PROMPTS = {
    "smile": "with a warm natural smile, friendly expression",
    "open_smile": "with a wide happy smile showing teeth, joyful expression",
    "subtle_smile": "with a subtle confident closed-mouth smile",
    "neutral": "with a neutral calm professional expression",
    "confident": "with a confident determined expression, slight smile",
    "warm": "with a warm welcoming friendly smile",
    "serious": "with a serious professional focused expression",
    "happy": "with a genuinely happy joyful expression, bright smile",
}

POSE_PROMPTS = {
    "frontal": "looking directly at camera, frontal view",
    "slight_left": "face turned slightly to the left, three-quarter view",
    "slight_right": "face turned slightly to the right, three-quarter view",
    "tilt_left": "head tilted slightly to the left",
    "tilt_right": "head tilted slightly to the right",
    "look_up": "chin slightly up, looking at camera",
    "three_quarter": "classic three-quarter portrait angle",
}

BACKGROUND_PROMPTS = {
    "studio_white": "plain white studio background",
    "studio_gray": "smooth gray studio background with subtle gradient",
    "studio_dark": "dark charcoal studio background",
    "studio_cream": "warm cream colored studio background",
    "office": "modern professional office background, bokeh",
    "outdoor": "outdoor natural green background, bokeh, soft light",
    "hotel_lobby": "luxury hotel lobby background, elegant, soft lighting",
    "neutral": "clean neutral background",
}

# Photoshoot presets (combined)
PRESETS = {
    "professional_headshot": {
        "expression": "subtle_smile",
        "pose": "frontal",
        "background": "studio_gray",
        "style": "professional corporate headshot photography",
    },
    "friendly_portrait": {
        "expression": "warm",
        "pose": "slight_right",
        "background": "studio_white",
        "style": "friendly approachable portrait photography",
    },
    "confident_executive": {
        "expression": "confident",
        "pose": "three_quarter",
        "background": "studio_dark",
        "style": "executive portrait, power pose, editorial photography",
    },
    "casual_happy": {
        "expression": "happy",
        "pose": "tilt_right",
        "background": "outdoor",
        "style": "casual lifestyle portrait, natural light",
    },
    "hotel_guest": {
        "expression": "smile",
        "pose": "frontal",
        "background": "hotel_lobby",
        "style": "hospitality guest photo, warm welcoming",
    },
}


class InstantIDEngine:

    def __init__(self, device: str = "cuda", low_vram: bool = True):
        """
        Args:
            device: "cuda" or "cpu"
            low_vram: If True, uses CPU offloading to fit on 16GB GPU
        """
        self.device = device
        self.low_vram = low_vram
        self._pipe = None
        self._pipe_img2img = None
        self._face_app = None
        self._face_emb_model = None
        log.info(f"InstantIDEngine initialized (device={device}, low_vram={low_vram})")

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def generate(
        self,
        source_path: str,
        expression: str = "smile",
        pose: str = "frontal",
        background: str = "studio_gray",
        style: str = "",
        num_steps: int = 30,
        guidance_scale: float = 5.0,
        ip_adapter_scale: float = 0.8,
        controlnet_scale: float = 0.8,
        seed: int = -1,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Generate a new image preserving identity with target expression/pose/background.

        Args:
            source_path: Input face photo
            expression: Key from EXPRESSION_PROMPTS or free text
            pose: Key from POSE_PROMPTS or free text
            background: Key from BACKGROUND_PROMPTS or free text
            style: Additional style prompt
            num_steps: Diffusion steps (20-50, higher = better quality)
            guidance_scale: CFG scale (3-8)
            ip_adapter_scale: Identity strength (0.5-1.0, higher = more like source)
            controlnet_scale: Landmark control strength (0.5-1.0)
            seed: Random seed (-1 for random)

        Returns:
            dict with 'image', 'identity_score', 'output_path', 'prompt', 'time'
        """
        import torch

        t0 = time.time()
        log.info(f"Generating: expr={expression}, pose={pose}, bg={background}")

        # Load source face
        source_img = Image.open(source_path).convert("RGB")
        source_cv = cv2.imread(source_path)
        if source_cv is None:
            raise ValueError(f"Cannot read: {source_path}")

        # Get face embedding + landmarks
        face_info = self._analyze_face(source_cv)
        if face_info is None:
            raise ValueError("No face detected in source image")

        face_emb = face_info["embedding"]
        face_kps = face_info["keypoints"]

        # Build prompt
        prompt = self._build_prompt(expression, pose, background, style)
        negative_prompt = (
            "deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, "
            "mutation, mutated, disfigured, poorly drawn face, extra fingers, "
            "low quality, worst quality, watermark, text, signature, "
            "cartoon, anime, illustration, painting, drawing"
        )

        log.info(f"Prompt: {prompt[:100]}...")

        # Generate
        pipe = self._get_pipeline()

        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        generator = torch.Generator(device="cpu").manual_seed(seed)

        result_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            ip_adapter_scale=ip_adapter_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
        ).images[0]

        # Measure identity preservation
        result_cv = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        identity_score = self._measure_identity(source_cv, result_cv)

        # Save
        if output_path is None:
            stem = Path(source_path).stem
            output_path = str(OUTPUT_DIR / f"{stem}_{expression}_{pose}_{seed}.jpg")
        result_image.save(output_path, quality=95)

        total_time = time.time() - t0
        log.info(f"Done in {total_time:.1f}s — identity={identity_score:.4f}")

        return {
            "image": result_image,
            "image_cv": result_cv,
            "output_path": output_path,
            "identity_score": identity_score,
            "prompt": prompt,
            "seed": seed,
            "time": total_time,
        }

    def generate_img2img(
        self,
        source_path: str,
        expression: str = "smile",
        pose: str = "frontal",
        background: str = "studio_gray",
        style: str = "",
        strength: float = 0.45,
        num_steps: int = 30,
        guidance_scale: float = 5.0,
        ip_adapter_scale: float = 0.8,
        controlnet_scale: float = 0.8,
        seed: int = -1,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        img2img mode — starts from the SOURCE IMAGE, not noise.
        Preserves hair, clothing, skin texture. Much higher identity.

        Args:
            strength: How much to change (0.0=keep original, 1.0=fully regenerate)
                      0.3-0.5 = subtle changes (expression only)
                      0.5-0.7 = moderate changes (expression + slight pose)
                      0.7-0.9 = big changes (new pose + expression)
        """
        import torch
        import gc

        t0 = time.time()
        log.info(f"[img2img] expr={expression}, pose={pose}, strength={strength}")

        # Load source
        source_pil = Image.open(source_path).convert("RGB")
        source_cv = cv2.imread(source_path)
        if source_cv is None:
            raise ValueError(f"Cannot read: {source_path}")

        # Resize source for SDXL (must be multiple of 64)
        source_resized = self._resize_for_sdxl(source_pil, max_side=1024, min_side=768)

        # Get face embedding + landmarks
        face_info = self._analyze_face(source_cv)
        if face_info is None:
            raise ValueError("No face detected in source image")

        face_emb = face_info["embedding"]
        # Draw keypoints on the resized source (not original size)
        resized_cv = cv2.cvtColor(np.array(source_resized), cv2.COLOR_RGB2BGR)
        face_info_resized = self._analyze_face(resized_cv)
        if face_info_resized:
            face_kps = face_info_resized["keypoints"]
        else:
            face_kps = face_info["keypoints"]

        # Build prompt
        prompt = self._build_prompt(expression, pose, background, style)
        negative_prompt = (
            "deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, "
            "mutation, mutated, disfigured, poorly drawn face, extra fingers, "
            "low quality, worst quality, watermark, text, signature, "
            "cartoon, anime, illustration, painting, drawing"
        )

        log.info(f"  Prompt: {prompt[:80]}...")
        log.info(f"  Source size: {source_resized.size}, strength: {strength}")

        # Load img2img pipeline
        pipe = self._get_pipeline_img2img()

        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

        result_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source_resized,
            image_embeds=face_emb,
            control_image=face_kps,
            controlnet_conditioning_scale=controlnet_scale,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        ).images[0]

        # Measure identity
        result_cv = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        identity_score = self._measure_identity(source_cv, result_cv)

        # Save
        if output_path is None:
            stem = Path(source_path).stem
            output_path = str(OUTPUT_DIR / f"{stem}_img2img_{expression}_{strength}_{seed}.jpg")
        result_image.save(output_path, quality=95)

        total_time = time.time() - t0
        log.info(f"  Done in {total_time:.1f}s — identity={identity_score:.4f}")

        return {
            "image": result_image,
            "image_cv": result_cv,
            "output_path": output_path,
            "identity_score": identity_score,
            "prompt": prompt,
            "seed": seed,
            "strength": strength,
            "time": total_time,
            "mode": "img2img",
        }

    @staticmethod
    def _resize_for_sdxl(img: Image.Image, max_side=1024, min_side=768) -> Image.Image:
        """Resize image for SDXL, dimensions must be multiples of 64."""
        w, h = img.size
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        w, h = round(ratio * w), round(ratio * h)
        # Round to nearest 64
        w = (w // 64) * 64
        h = (h // 64) * 64
        return img.resize((w, h), Image.LANCZOS)

    def generate_preset(self, source_path: str, preset: str, **kwargs) -> Dict:
        """Generate using a named preset (e.g., 'professional_headshot')."""
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")

        cfg = PRESETS[preset]
        return self.generate(
            source_path=source_path,
            expression=cfg["expression"],
            pose=cfg["pose"],
            background=cfg["background"],
            style=cfg.get("style", ""),
            **kwargs,
        )

    def generate_grid(self, source_path: str, expressions: List[str] = None,
                      poses: List[str] = None, **kwargs) -> List[Dict]:
        """Generate a grid of expression x pose combinations."""
        if expressions is None:
            expressions = ["subtle_smile", "smile", "open_smile"]
        if poses is None:
            poses = ["frontal", "slight_left", "slight_right"]

        results = []
        for expr in expressions:
            for pose in poses:
                try:
                    r = self.generate(source_path, expression=expr, pose=pose, **kwargs)
                    results.append(r)
                    log.info(f"  {expr} + {pose}: identity={r['identity_score']:.4f}")
                except Exception as e:
                    log.error(f"  {expr} + {pose}: FAILED — {e}")
                    results.append({"error": str(e), "expression": expr, "pose": pose})
        return results

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE
    # ══════════════════════════════════════════════════════════════════════════

    def _build_prompt(self, expression: str, pose: str, background: str, style: str) -> str:
        """Build the full generation prompt from components."""
        expr_text = EXPRESSION_PROMPTS.get(expression, expression)
        pose_text = POSE_PROMPTS.get(pose, pose)
        bg_text   = BACKGROUND_PROMPTS.get(background, background)

        parts = [
            "professional portrait photograph of a person",
            expr_text,
            pose_text,
            bg_text,
            style if style else "high quality DSLR photography",
            "sharp focus, professional lighting, 8K, photorealistic",
        ]
        return ", ".join(p for p in parts if p)

    def _analyze_face(self, img: np.ndarray) -> Optional[Dict]:
        """Extract face embedding and keypoints for InstantID."""
        app = self._get_face_app()
        faces = app.get(img)
        if not faces:
            return None

        # Take largest face
        face = max(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]))

        # Get embedding (512-dim ArcFace)
        embedding = face['embedding']

        # Get 2D keypoints
        kps = face['kps']

        # Build keypoint image using official InstantID draw_kps
        repo_path = Path(__file__).parent / "repo"
        sys.path.insert(0, str(repo_path))
        from pipeline_stable_diffusion_xl_instantid import draw_kps

        # Convert source to PIL for draw_kps
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        kps_image = draw_kps(pil_img, kps)

        return {
            "embedding": embedding,
            "keypoints": kps_image,
            "kps_raw": kps,
            "bbox": face['bbox'],
            "det_score": float(face['det_score']),
            "pose": face['pose'] if 'pose' in face else [0, 0, 0],
        }

    def _measure_identity(self, source: np.ndarray, result: np.ndarray) -> float:
        """ArcFace cosine similarity between source and result."""
        try:
            app = self._get_face_app()
            f1 = app.get(source)
            f2 = app.get(result)
            if not f1 or not f2:
                return 0.0
            return float(np.dot(f1[0].normed_embedding, f2[0].normed_embedding))
        except Exception:
            return 0.0

    def _get_face_app(self):
        """Load InsightFace for face analysis."""
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(
                name="antelopev2",
                root=str(ROOT / "MagicFace/third_party_files"),
            )
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_app

    def _get_pipeline(self):
        """Load InstantID pipeline using official InstantID code."""
        if self._pipe is not None:
            return self._pipe

        import torch

        # Add official InstantID repo to path
        repo_path = Path(__file__).parent / "repo"
        sys.path.insert(0, str(repo_path))
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
        from diffusers.models import ControlNetModel

        log.info("Loading InstantID pipeline...")
        t0 = time.time()

        # Load ControlNet
        controlnet_path = WEIGHTS_DIR / "ControlNetModel"
        log.info(f"  Loading ControlNet from {controlnet_path}")
        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_path),
            torch_dtype=torch.float16,
        )

        # Load SDXL + InstantID pipeline
        sdxl_path = WEIGHTS_DIR / "sdxl"
        log.info(f"  Loading SDXL from {sdxl_path}")
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            str(sdxl_path),
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        # Load IP-Adapter for identity
        ip_adapter_path = str(WEIGHTS_DIR / "ip-adapter.bin")
        log.info(f"  Loading IP-Adapter")
        pipe.load_ip_adapter_instantid(ip_adapter_path)

        # Memory optimization for T4 (16GB)
        if self.low_vram:
            log.info("  Enabling CPU offload for low VRAM mode")
            pipe.enable_model_cpu_offload()
        else:
            pipe.cuda()

        self._pipe = pipe
        log.info(f"  Pipeline loaded in {time.time()-t0:.1f}s")
        return self._pipe

    def _get_pipeline_img2img(self):
        """Load InstantID img2img pipeline."""
        if self._pipe_img2img is not None:
            return self._pipe_img2img

        import torch

        repo_path = Path(__file__).parent / "repo"
        sys.path.insert(0, str(repo_path))
        from pipeline_stable_diffusion_xl_instantid_img2img import StableDiffusionXLInstantIDImg2ImgPipeline
        from diffusers.models import ControlNetModel

        log.info("Loading InstantID img2img pipeline...")
        t0 = time.time()

        controlnet_path = WEIGHTS_DIR / "ControlNetModel"
        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_path), torch_dtype=torch.float16,
        )

        sdxl_path = WEIGHTS_DIR / "sdxl"
        pipe = StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained(
            str(sdxl_path),
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        ip_adapter_path = str(WEIGHTS_DIR / "ip-adapter.bin")
        pipe.load_ip_adapter_instantid(ip_adapter_path)

        if self.low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe.cuda()

        self._pipe_img2img = pipe
        log.info(f"  img2img pipeline loaded in {time.time()-t0:.1f}s")
        return self._pipe_img2img


# ── CLI for quick testing ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("source", help="Source face image")
    p.add_argument("--expression", "-e", default="smile")
    p.add_argument("--pose", "-p", default="frontal")
    p.add_argument("--background", "-b", default="studio_gray")
    p.add_argument("--preset", default=None, help="Use a preset instead")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--ip-scale", type=float, default=0.8)
    p.add_argument("--cn-scale", type=float, default=0.8)
    p.add_argument("--output", "-o", default=None)
    args = p.parse_args()

    engine = InstantIDEngine()

    if args.preset:
        result = engine.generate_preset(
            args.source, args.preset,
            num_steps=args.steps, seed=args.seed,
            output_path=args.output,
        )
    else:
        result = engine.generate(
            args.source,
            expression=args.expression,
            pose=args.pose,
            background=args.background,
            num_steps=args.steps,
            guidance_scale=5.0,
            ip_adapter_scale=args.ip_scale,
            controlnet_scale=args.cn_scale,
            seed=args.seed,
            output_path=args.output,
        )

    print(f"\nResult:")
    print(f"  Identity score: {result['identity_score']:.4f} ({result['identity_score']*100:.1f}%)")
    print(f"  Output: {result['output_path']}")
    print(f"  Time: {result['time']:.1f}s")
    print(f"  Seed: {result['seed']}")
