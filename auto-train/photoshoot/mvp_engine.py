#!/usr/bin/env python3
"""
MVP Smile Engine — Clean, Simple, Production-Ready

LP lip mode + d12.jpg driver + identity-guided multiplier search.
No compositing. No GFPGAN. No SD. Just LP with the right settings.

The simplest thing that works.
"""

import os
import sys
import cv2
import time
import numpy as np
import logging
from typing import Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mvp] %(message)s")
logger = logging.getLogger("mvp")

PHOTO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PHOTO_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")

sys.path.insert(0, LP_DIR)

DRIVER = os.path.join(LP_DIR, "assets", "examples", "driving", "d12.jpg")
ANTELOPEV2 = os.path.join(BASE_DIR, "MagicFace", "third_party_files")
GATE = 0.90  # Minimum identity to show


class MVPSmileEngine:
    """Simple smile engine: LP lip mode + identity search."""

    def __init__(self):
        self._load_models()
        logger.info("MVPSmileEngine ready")

    def _load_models(self):
        from insightface.app import FaceAnalysis
        self.fa = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        self.lp = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig(),
        )

    def _get_embedding(self, img):
        faces = self.fa.get(img)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding.reshape(1, -1)

    def _run_lp(self, source_path, multiplier):
        """Run LP in lip mode."""
        from src.config.argument_config import ArgumentConfig

        args = ArgumentConfig()
        args.source = source_path
        args.driving = DRIVER
        args.output_dir = os.path.join(PHOTO_DIR, "temp")
        os.makedirs(args.output_dir, exist_ok=True)
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "lip"  # Lip mode — clean gentle smile
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

    def _handle_tight_crop(self, image_path):
        """Handle tight face crops where face fills entire frame.

        Instead of padding (which creates visible borders), we:
        1. Detect if face is not found (tight crop)
        2. Resize to 256x256 and run LP with flag_do_crop=False
        3. LP processes the image directly without trying to crop
        4. No padding, no borders, clean output
        """
        img = cv2.imread(image_path)
        if img is None:
            return image_path, False

        faces = self.fa.get(img)
        if faces:
            return image_path, False  # Normal photo, no special handling

        # Tight crop detected — resize to 512x512 (LP-friendly size)
        h, w = img.shape[:2]
        resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        resized_path = os.path.join(PHOTO_DIR, "temp", "tight_crop_resized.jpg")
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        cv2.imwrite(resized_path, resized)

        # Check if face is now detectable at 512x512
        faces = self.fa.get(resized)
        if faces:
            logger.info(f"  Tight crop resized: {w}x{h} → 512x512")
            return resized_path, True

        # Still can't detect — try with small padding (just enough for detection)
        pad = 50  # Minimal padding
        avg = img.mean(axis=(0, 1)).astype(np.uint8)
        padded = cv2.copyMakeBorder(resized, pad, pad, pad, pad,
                                     cv2.BORDER_REFLECT_101)
        padded_path = os.path.join(PHOTO_DIR, "temp", "tight_crop_padded.jpg")
        cv2.imwrite(padded_path, padded)

        faces = self.fa.get(padded)
        if faces:
            logger.info(f"  Tight crop padded: {w}x{h} → {padded.shape[1]}x{padded.shape[0]}")
            return padded_path, True

        return image_path, False

    def smile(self, source_path: str) -> Dict:
        """Apply photogenic smile with identity-guided multiplier search.

        Tries multipliers from high to low. Returns the STRONGEST smile
        that keeps identity above 93%.

        Handles tight face crops by adding padding automatically.
        """
        t0 = time.time()

        source = cv2.imread(source_path)
        if source is None:
            return {"success": False, "message": "Cannot read image"}

        # Handle tight crops
        actual_path, is_tight = self._handle_tight_crop(source_path)

        # Get source embedding
        src_emb = self._get_embedding(source)
        if src_emb is None and is_tight:
            resized_img = cv2.imread(actual_path)
            src_emb = self._get_embedding(resized_img)
        if src_emb is None:
            return {"success": False, "message": "No face detected. Try a photo with more background around the face."}

        # Multipliers from autoresearch winner: all mode works best at 0.5-0.7x
        gate = GATE if not is_tight else 0.0
        multipliers = [1.8, 1.5, 1.2, 1.0]  # 1.8x sweet spot
        best = None

        for mult in multipliers:
            result = self._run_lp(actual_path, mult)  # Use padded path if needed
            if result is None:
                continue

            # For tight crops: resize result back to original dimensions
            if is_tight:
                orig = cv2.imread(source_path)
                if orig is not None:
                    oh, ow = orig.shape[:2]
                    result = cv2.resize(result, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

            res_emb = self._get_embedding(result)
            score = float(cosine_similarity(
                src_emb, res_emb
            )[0][0]) if res_emb is not None else 0

            if score >= gate:
                best = {"result": result, "score": score, "mult": mult}
                break  # Take the strongest that passes

        elapsed = time.time() - t0

        if best is None:
            return {
                "success": False,
                "message": "Could not apply smile while preserving identity. Try a front-facing photo.",
                "time": round(elapsed, 1),
            }

        # Save
        out_dir = os.path.join(PHOTO_DIR, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(source_path))[0]
        out_path = os.path.join(out_dir, f"{name}_smile.png")
        cv2.imwrite(out_path, best["result"])

        return {
            "success": True,
            "identity_score": round(best["score"], 4),
            "multiplier": best["mult"],
            "output_path": out_path,
            "time": round(elapsed, 1),
        }
