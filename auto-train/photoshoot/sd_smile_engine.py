#!/usr/bin/env python3
"""
SD Inpainting Smile Engine — Option C Done Properly

Architecture:
  1. LP generates smile geometry (WHERE mouth should move)
  2. Extract mouth mask from LP output
  3. SD Inpainting repaints ONLY the mouth at 512px (not LP's 256px)
  4. Original image stays untouched everywhere except mouth

Result: Original identity + AI-rendered photorealistic smile mouth
"""

import os
import sys
import cv2
import time
import uuid
import torch
import numpy as np
import logging
from typing import Dict, Optional
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [sd-smile] %(levelname)s: %(message)s")
logger = logging.getLogger("sd-smile")

PHOTO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PHOTO_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "MagicFace", "third_party_files")

sys.path.insert(0, LP_DIR)

MOUTH_IDX = list(range(52, 72))


class SDSmileEngine:
    """LP geometry + SD inpainting for photorealistic smile."""

    def __init__(self):
        self.device = "cuda"
        self._load_face_analyzer()
        self._load_liveportrait()
        self._load_sd_inpainting()
        logger.info("SDSmileEngine ready")

    def _load_face_analyzer(self):
        from insightface.app import FaceAnalysis
        self.fa = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2_DIR,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
        logger.info("  InsightFace loaded")

    def _load_liveportrait(self):
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        self.lp = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig(),
        )
        logger.info("  LivePortrait loaded")

    def _load_sd_inpainting(self):
        """Load SD inpainting — the key piece that was blocked before."""
        from diffusers import StableDiffusionInpaintPipeline

        logger.info("  Loading SD Inpainting (first time downloads ~4GB)...")
        self.sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.device)

        # Memory optimization for T4
        self.sd_pipe.enable_attention_slicing()
        logger.info("  SD Inpainting loaded")

    def _get_face(self, img):
        faces = self.fa.get(img)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    def _get_embedding(self, img):
        face = self._get_face(img)
        return face.normed_embedding.reshape(1, -1) if face else None

    def _identity_score(self, src_emb, img):
        emb = self._get_embedding(img)
        if emb is None:
            return 0.0
        return float(cosine_similarity(src_emb, emb)[0][0])

    def _run_lp(self, source_path, driver_path, multiplier=0.7):
        """Run LP to get smile geometry."""
        from src.config.argument_config import ArgumentConfig
        os.makedirs(os.path.join(PHOTO_DIR, "temp"), exist_ok=True)

        args = ArgumentConfig()
        args.source = source_path
        args.driving = driver_path
        args.output_dir = os.path.join(PHOTO_DIR, "temp")
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "all"
        args.driving_option = "expression-friendly"
        args.driving_multiplier = multiplier
        args.source_max_dim = 1920

        inf_keys = {
            'flag_pasteback', 'flag_do_crop', 'flag_stitching',
            'flag_relative_motion', 'animation_region', 'driving_option',
            'driving_multiplier', 'source_max_dim', 'source_division',
            'flag_eye_retargeting', 'flag_lip_retargeting',
        }
        self.lp.live_portrait_wrapper.update_config(
            {k: v for k, v in args.__dict__.items() if k in inf_keys}
        )
        try:
            wfp, _ = self.lp.execute(args)
            return cv2.imread(wfp)
        except:
            return None

    def _create_mouth_mask(self, source_bgr, lp_result):
        """Create mouth mask covering both original and LP mouth positions."""
        h, w = source_bgr.shape[:2]
        lp_resized = cv2.resize(lp_result, (w, h))

        mask = np.zeros((h, w), dtype=np.uint8)

        # LP mouth position
        lp_face = self._get_face(lp_resized)
        if lp_face is not None:
            lp_lmk = getattr(lp_face, 'landmark_2d_106', None)
            if lp_lmk is not None:
                pts = lp_lmk[MOUTH_IDX].astype(np.int32)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(mask, hull, 255)

        # Source mouth position (prevent ghost)
        src_face = self._get_face(source_bgr)
        if src_face is not None:
            src_lmk = getattr(src_face, 'landmark_2d_106', None)
            if src_lmk is not None:
                pts = src_lmk[MOUTH_IDX].astype(np.int32)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(mask, hull, 255)

        # Expand to cover smile lines and chin
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        mask = cv2.dilate(mask, kernel, iterations=3)

        return mask

    def _sd_inpaint_mouth(self, source_bgr, mask, smile_type="photoshoot"):
        """
        Use SD to inpaint the mouth region at 512px resolution.

        This is the KEY improvement over LP:
        - LP renders mouth at 256px → blurry teeth
        - SD renders mouth at 512px → sharp realistic teeth
        - Only the mouth region is AI-generated
        - Everything else is the original photo
        """
        h, w = source_bgr.shape[:2]

        # Find face region for targeted inpainting
        face = self._get_face(source_bgr)
        if face is None:
            return source_bgr

        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1
        pad = int(max(fw, fh) * 0.3)

        # Crop face area
        fx1 = max(0, x1 - pad)
        fy1 = max(0, y1 - pad)
        fx2 = min(w, x2 + pad)
        fy2 = min(h, y2 + pad)

        face_crop = source_bgr[fy1:fy2, fx1:fx2]
        mask_crop = mask[fy1:fy2, fx1:fx2]

        # Resize to 512x512 for SD
        face_512 = cv2.resize(face_crop, (512, 512))
        mask_512 = cv2.resize(mask_crop, (512, 512))

        # Convert to PIL
        face_pil = Image.fromarray(cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_512)

        # Prompts
        prompts = {
            "subtle": "natural gentle closed mouth smile, photorealistic face, same person, sharp skin detail, 8k",
            "photoshoot": "natural happy smile showing teeth, photorealistic face, professional portrait, sharp teeth detail, same person, 8k",
            "natural": "genuine warm smile with teeth, authentic joyful expression, photorealistic, same person, 8k",
        }
        negative = "deformed, distorted, bad teeth, blurry, low quality, cartoon, anime, painting, 3d render, watermark"

        prompt = prompts.get(smile_type, prompts["photoshoot"])

        # Run SD inpainting
        with torch.autocast("cuda"):
            result_pil = self.sd_pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=face_pil,
                mask_image=mask_pil,
                num_inference_steps=30,
                guidance_scale=7.5,
                strength=0.65,
            ).images[0]

        # Convert back to BGR
        result_np = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

        # Resize back to face crop size
        fh_crop, fw_crop = face_crop.shape[:2]
        result_resized = cv2.resize(result_np, (fw_crop, fh_crop), interpolation=cv2.INTER_LANCZOS4)

        # Blend with smooth mask
        mask_float = cv2.resize(mask_512, (fw_crop, fh_crop)).astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (21, 21), 8)
        mask_3ch = mask_float[:, :, np.newaxis]

        blended = (
            result_resized.astype(np.float32) * mask_3ch +
            face_crop.astype(np.float32) * (1 - mask_3ch)
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Paste back into full image
        output = source_bgr.copy()
        output[fy1:fy2, fx1:fx2] = blended

        return output

    def smile(self, source_path, smile_type="photoshoot", output_name=None):
        """Generate smile using LP geometry + SD inpainting."""
        t0 = time.time()

        source_bgr = cv2.imread(source_path)
        if source_bgr is None:
            return {"success": False, "message": "Cannot read image"}

        src_emb = self._get_embedding(source_bgr)
        if src_emb is None:
            return {"success": False, "message": "No face detected"}

        # Step 1: LP generates smile geometry
        driver = os.path.join(LP_DIR, "assets", "examples", "driving", "d12.jpg")
        lp_result = self._run_lp(source_path, driver, multiplier=0.7)
        if lp_result is None:
            return {"success": False, "message": "LP failed"}

        # Step 2: Create mouth mask
        mask = self._create_mouth_mask(source_bgr, lp_result)

        # Step 3: SD inpaints mouth at 512px
        result = self._sd_inpaint_mouth(source_bgr, mask, smile_type)

        # Step 4: Check identity
        score = self._identity_score(src_emb, result)

        # Save
        OUT = os.path.join(PHOTO_DIR, "final_output")
        os.makedirs(OUT, exist_ok=True)
        if output_name is None:
            name = os.path.splitext(os.path.basename(source_path))[0]
            output_name = f"{name}_sd_{smile_type}.png"
        out_path = os.path.join(OUT, output_name)
        cv2.imwrite(out_path, result)

        elapsed = time.time() - t0
        logger.info(f"  {smile_type}: identity={score:.4f} time={elapsed:.1f}s → {output_name}")

        return {
            "success": True,
            "identity_score": round(score, 4),
            "output_path": out_path,
            "smile_type": smile_type,
            "time": round(elapsed, 1),
        }


if __name__ == "__main__":
    engine = SDSmileEngine()

    # Test on 5 men
    men = [
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0008.jpg", "man_asian"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0013.jpg", "man_black"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0015.jpg", "man_suit"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0019.jpg", "man_young"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0009.jpg", "man_older"),
    ]

    OUT = os.path.join(PHOTO_DIR, "final_output")
    os.makedirs(OUT, exist_ok=True)

    print(f"\n{'='*60}")
    print("SD INPAINTING SMILE TEST — 5 men")
    print(f"{'='*60}\n")

    for src_path, name in men:
        # Save original
        src = cv2.imread(src_path)
        if src is None:
            continue
        cv2.imwrite(os.path.join(OUT, f"{name}_original.png"), src)

        # Generate smile
        r = engine.smile(src_path, smile_type="photoshoot", output_name=f"{name}_sd_smile.png")
        ok = "✓" if r["success"] else "✗"
        score = r.get("identity_score", 0)
        print(f"  {ok} {name}: identity={score:.4f} time={r.get('time',0):.1f}s")

        # Also build comparison strip
        if r["success"]:
            result = cv2.imread(r["output_path"])
            h_t = 300
            s1 = h_t / src.shape[0]
            s2 = h_t / result.shape[0]
            o_r = cv2.resize(src, (int(src.shape[1]*s1), h_t))
            r_r = cv2.resize(result, (int(result.shape[1]*s2), h_t))
            cv2.putText(o_r, "Original", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(r_r, f"SD Smile id={score:.3f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            w_max = max(o_r.shape[1], r_r.shape[1])
            if o_r.shape[1] < w_max:
                o_r = np.hstack([o_r, np.zeros((h_t, w_max-o_r.shape[1], 3), dtype=np.uint8)])
            if r_r.shape[1] < w_max:
                r_r = np.hstack([r_r, np.zeros((h_t, w_max-r_r.shape[1], 3), dtype=np.uint8)])
            comp = np.hstack([o_r, r_r])
            cv2.imwrite(os.path.join(OUT, f"{name}_comparison.png"), comp)

    print(f"\nOutput: {OUT}/")
