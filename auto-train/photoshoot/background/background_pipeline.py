"""
Background Replacement Pipeline
================================
Takes a portrait photo → removes background → composites onto new background.

Methods (in order of quality):
  1. rembg      — fastest, good for clean portraits (no GPU needed)
  2. MODNet     — best hair/edge detail, GPU accelerated
  3. SAM2       — most accurate segmentation (Meta, GPU)

Background options:
  A. Solid color / gradient (instant)
  B. Studio gradients presets (instant)
  C. User-provided image (compositing)
  D. AI-generated via prompt (Stable Diffusion)

Usage:
    from photoshoot.background.background_pipeline import BackgroundPipeline

    pipeline = BackgroundPipeline()
    result = pipeline.process(
        source_path="photo.jpg",
        background="studio_white",   # or path to image, or text prompt
        output_path="result.jpg"
    )
"""

import os, sys, cv2, logging
import numpy as np
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

# ── Studio background presets ────────────────────────────────────────────────

STUDIO_PRESETS = {
    "studio_white":     (255, 255, 255),
    "studio_gray":      (200, 200, 200),
    "studio_black":     (10,  10,  10),
    "studio_cream":     (245, 240, 230),
    "studio_navy":      (25,  35,  65),
    "studio_charcoal":  (55,  55,  60),
}

GRADIENT_PRESETS = {
    "gradient_white":   [(255,255,255), (220,220,220)],
    "gradient_gray":    [(180,180,180), (120,120,120)],
    "gradient_blue":    [(200,220,255), (100,140,220)],
    "gradient_warm":    [(255,245,235), (220,200,170)],
}


class BackgroundPipeline:

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._segmentor = None
        self._sd_pipe   = None
        log.info("BackgroundPipeline initialized")

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def process(
        self,
        source_path: str,
        background: str = "studio_white",
        output_path: Optional[str] = None,
        feather_px: int = 15,
    ) -> dict:
        """
        Remove background and composite onto new background.

        Args:
            source_path: Input portrait image
            background:  One of:
                         - preset name: "studio_white", "gradient_blue" ...
                         - image path:  "/path/to/bg.jpg"
                         - text prompt: "luxury hotel lobby with soft lighting"
            output_path: Where to save result (auto-generated if None)
            feather_px:  Edge softness in pixels

        Returns:
            dict with 'image' (np array), 'output_path', 'mask_path', 'method'
        """
        img = cv2.imread(source_path)
        if img is None:
            raise ValueError(f"Cannot read image: {source_path}")

        h, w = img.shape[:2]
        log.info(f"Processing {w}x{h} image, background='{background}'")

        # Step 1: Extract person mask
        mask = self._extract_mask(img)

        # Step 2: Feather edges for natural look
        mask = self._feather_mask(mask, feather_px)

        # Step 3: Build background canvas
        bg = self._build_background(background, w, h)

        # Step 4: Composite
        result = self._composite(img, bg, mask)

        # Step 5: Color harmony (match lighting between person and bg)
        result = self._harmonize_lighting(img, result, mask)

        # Save
        if output_path is None:
            stem = Path(source_path).stem
            output_path = str(Path(source_path).parent / f"{stem}_bg_{background.split('/')[-1]}.jpg")

        cv2.imwrite(output_path, result)

        # Save mask for debugging
        mask_path = output_path.replace(".jpg", "_mask.jpg")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        log.info(f"Saved: {output_path}")
        return {
            "image": result,
            "output_path": output_path,
            "mask_path": mask_path,
            "method": self._last_method,
            "size": (w, h),
        }

    def batch_process(self, source_paths: list, background: str = "studio_white",
                      output_dir: str = "output/backgrounds") -> list:
        """Process multiple images."""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for i, src in enumerate(source_paths):
            try:
                out = os.path.join(output_dir, f"{Path(src).stem}_bg.jpg")
                r = self.process(src, background=background, output_path=out)
                results.append(r)
                log.info(f"[{i+1}/{len(source_paths)}] done: {out}")
            except Exception as e:
                log.error(f"Failed {src}: {e}")
                results.append({"error": str(e), "source": src})
        return results

    # ── MASK EXTRACTION ───────────────────────────────────────────────────────

    def _extract_mask(self, img: np.ndarray) -> np.ndarray:
        """Try methods in order: rembg → MODNet → GrabCut fallback."""

        # Method 1: rembg (fastest, best general quality)
        try:
            return self._rembg_mask(img)
        except Exception as e:
            log.debug(f"rembg failed: {e}")

        # Method 2: MODNet (best for hair)
        try:
            return self._modnet_mask(img)
        except Exception as e:
            log.debug(f"MODNet failed: {e}")

        # Method 3: GrabCut fallback (no GPU, no install needed)
        log.warning("Using GrabCut fallback — install rembg for better quality")
        return self._grabcut_mask(img)

    def _rembg_mask(self, img: np.ndarray) -> np.ndarray:
        from rembg import remove
        import io
        from PIL import Image

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        out_bytes = remove(buf.getvalue())
        out_pil = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        alpha = np.array(out_pil)[:, :, 3].astype(np.float32) / 255.0
        self._last_method = "rembg"
        log.info("  Mask: rembg ✓")
        return alpha

    def _modnet_mask(self, img: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        modnet_path = ROOT / "photoshoot/background/modnet.pt"
        if not modnet_path.exists():
            self._download_modnet(modnet_path)

        from photoshoot.background.modnet_model import MODNet
        net = MODNet(backbone_pretrained=False)
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(str(modnet_path), map_location="cpu"))
        net.eval()
        if self.device == "cuda" and torch.cuda.is_available():
            net = net.cuda()

        ref_size = 512
        h, w = img.shape[:2]
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

        # Resize for model
        rh = ref_size if h >= w else int(ref_size * h / w)
        rw = ref_size if w >= h else int(ref_size * w / h)
        rh = rh - rh % 32
        rw = rw - rw % 32
        x_resized = F.interpolate(x, size=(rh, rw), mode="area")

        if self.device == "cuda" and torch.cuda.is_available():
            x_resized = x_resized.cuda()

        with torch.no_grad():
            _, _, matte = net(x_resized, True)

        matte = F.interpolate(matte, size=(h, w), mode="area")
        mask = matte.squeeze().cpu().numpy()
        self._last_method = "modnet"
        log.info("  Mask: MODNet ✓")
        return mask

    def _grabcut_mask(self, img: np.ndarray) -> np.ndarray:
        """OpenCV GrabCut — no dependencies, reasonable quality."""
        h, w = img.shape[:2]

        # Assume face is in center 60% of image
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.05)
        rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

        mask = np.zeros(img.shape[:2], np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgd, fgd, 10, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.float32)
        self._last_method = "grabcut"
        log.info("  Mask: GrabCut fallback ✓")
        return mask2

    @staticmethod
    def _download_modnet(path: Path):
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ZHKKKe/MODNet/releases/download/v0.1/modnet_photographic_portrait_matting.ckpt"
        log.info(f"Downloading MODNet weights to {path}...")
        urllib.request.urlretrieve(url, str(path))

    # ── MASK REFINEMENT ───────────────────────────────────────────────────────

    @staticmethod
    def _feather_mask(mask: np.ndarray, feather_px: int = 15) -> np.ndarray:
        """Soften mask edges for natural compositing."""
        mask_u8 = (mask * 255).astype(np.uint8)
        # Slight erosion to clean edges
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        # Gaussian blur for feathering
        blurred = cv2.GaussianBlur(eroded.astype(np.float32),
                                   (feather_px*2+1, feather_px*2+1), feather_px/3)
        return np.clip(blurred / 255.0, 0, 1)

    # ── BACKGROUND BUILDER ────────────────────────────────────────────────────

    def _build_background(self, background: str, w: int, h: int) -> np.ndarray:
        """Build background canvas matching source image size."""

        # Solid color preset
        if background in STUDIO_PRESETS:
            r, g, b = STUDIO_PRESETS[background]
            bg = np.full((h, w, 3), [b, g, r], dtype=np.uint8)
            return bg

        # Gradient preset
        if background in GRADIENT_PRESETS:
            return self._make_gradient(GRADIENT_PRESETS[background], w, h)

        # Image file path
        if os.path.exists(background):
            bg = cv2.imread(background)
            if bg is not None:
                return cv2.resize(bg, (w, h))

        # Text prompt → Stable Diffusion
        if len(background) > 20:
            return self._generate_background_sd(background, w, h)

        # Fallback: neutral gray
        log.warning(f"Unknown background '{background}', using gray")
        return np.full((h, w, 3), 180, dtype=np.uint8)

    @staticmethod
    def _make_gradient(colors: list, w: int, h: int) -> np.ndarray:
        """Top-to-bottom gradient between two colors."""
        c1 = np.array(colors[0][::-1], dtype=np.float32)  # RGB→BGR
        c2 = np.array(colors[1][::-1], dtype=np.float32)
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            t = y / h
            color = (c1 * (1 - t) + c2 * t).astype(np.uint8)
            bg[y, :] = color
        return bg

    def _generate_background_sd(self, prompt: str, w: int, h: int) -> np.ndarray:
        """Generate background via Stable Diffusion."""
        try:
            import torch
            from diffusers import StableDiffusionPipeline

            if self._sd_pipe is None:
                log.info("Loading Stable Diffusion for background generation...")
                self._sd_pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                )
                if self.device == "cuda" and torch.cuda.is_available():
                    self._sd_pipe = self._sd_pipe.to("cuda")

            full_prompt = f"{prompt}, photographic background, no people, professional photography, 4K"
            neg = "people, faces, text, watermark, blurry"
            out = self._sd_pipe(full_prompt, negative_prompt=neg,
                                width=512, height=512, num_inference_steps=20)
            from PIL import Image
            import io
            pil_bg = out.images[0].resize((w, h))
            bg = cv2.cvtColor(np.array(pil_bg), cv2.COLOR_RGB2BGR)
            log.info(f"SD background generated for prompt: '{prompt[:50]}...'")
            return bg
        except Exception as e:
            log.warning(f"SD generation failed: {e}, using gray fallback")
            return np.full((h, w, 3), 180, dtype=np.uint8)

    # ── COMPOSITE ─────────────────────────────────────────────────────────────

    @staticmethod
    def _composite(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Alpha-composite foreground person onto background."""
        alpha = mask[:, :, np.newaxis]
        fg_f = fg.astype(np.float32)
        bg_f = bg.astype(np.float32)
        result = fg_f * alpha + bg_f * (1 - alpha)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _harmonize_lighting(original: np.ndarray, composited: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
        """
        Subtle color/brightness match between person region and background.
        Prevents the "cut-out pasted on" look.
        """
        # Get background region stats from composited image
        bg_mask = (1 - mask) > 0.5
        fg_mask = mask > 0.5

        if not fg_mask.any() or not bg_mask.any():
            return composited

        result = composited.astype(np.float32)

        # Match mean brightness — shift person brightness to match bg subtly
        for c in range(3):
            bg_mean = result[:, :, c][bg_mask].mean()
            fg_mean = result[:, :, c][fg_mask].mean()
            diff = (bg_mean - fg_mean) * 0.15  # 15% blend — subtle
            result[:, :, c] = np.where(
                mask > 0.5,
                np.clip(result[:, :, c] + diff, 0, 255),
                result[:, :, c]
            )

        return result.astype(np.uint8)
