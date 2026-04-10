#!/usr/bin/env python3
"""
Pose + Expression Chain Engine

Chains two LP calls:
  1. animation_region="pose" → change head angle
  2. animation_region="lip" → change expression

Each step is independent. Identity gate checks after each step.
"""

import os
import sys
import cv2
import time
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [pose-engine] %(message)s")
logger = logging.getLogger("pose-engine")

POSE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(POSE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ANTELOPEV2 = os.path.join(BASE_DIR, "MagicFace", "third_party_files")
DRIVERS_DIR = os.path.join(POSE_DIR, "drivers")
OUTPUT_DIR = os.path.join(POSE_DIR, "output")

sys.path.insert(0, LP_DIR)

# Default drivers
LP_DRIVING = os.path.join(LP_DIR, "assets", "examples", "driving")
DEFAULT_SMILE_DRIVER = os.path.join(LP_DRIVING, "d12.jpg")
DEFAULT_POSE_DRIVER = os.path.join(LP_DRIVING, "d30.jpg")  # Gentle head angle

IDENTITY_GATE = 0.85  # Lower for combined pose+expression


class PoseExpressionEngine:
    """Chain LP calls: pose first, then expression."""

    def __init__(self):
        self._load_models()
        logger.info("PoseExpressionEngine ready")

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

    def _run_lp(self, source_path, driver_path, region, multiplier, driving_option="expression-friendly"):
        """Run LP with specific animation_region."""
        from src.config.argument_config import ArgumentConfig

        temp = os.path.join(POSE_DIR, "temp")
        os.makedirs(temp, exist_ok=True)

        args = ArgumentConfig()
        args.source = source_path
        args.driving = driver_path
        args.output_dir = temp
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = region
        args.driving_option = driving_option
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
        except Exception:
            return None

    def _find_driver(self, category, subcategory=None):
        """Find best driver for a pose/expression category."""
        if subcategory:
            d = os.path.join(DRIVERS_DIR, category, subcategory)
        else:
            d = os.path.join(DRIVERS_DIR, category)

        if os.path.exists(d):
            files = sorted([f for f in os.listdir(d) if f.endswith('.jpg')])
            if files:
                return os.path.join(d, files[0])
        return None

    def edit(
        self,
        source_path: str,
        pose: Optional[str] = None,
        expression: Optional[str] = None,
        pose_multiplier: float = 1.0,
        expr_multiplier: float = 1.8,
        output_name: Optional[str] = None,
    ) -> Dict:
        """
        Apply pose and/or expression change.

        Args:
            source_path: Input photo
            pose: None, "slight_left", "slight_right", "tilt_left", "tilt_right",
                  "look_up", "look_down", "three_quarter"
            expression: None, "subtle", "photoshoot", "natural"
            pose_multiplier: Pose intensity (0.3-1.5)
            expr_multiplier: Expression intensity (0.8-2.5)

        Returns:
            Dict with success, identity, output_path, etc.
        """
        t0 = time.time()

        source = cv2.imread(source_path)
        if source is None:
            return {"success": False, "message": "Cannot read image"}

        src_emb = self._get_embedding(source)
        if src_emb is None:
            return {"success": False, "message": "No face detected"}

        current_path = source_path
        current_img = source
        steps_done = []

        # ── STEP 1: POSE (if requested) ──
        if pose and pose != "straight":
            pose_driver = self._find_driver("pose", pose)
            if pose_driver is None:
                # Fallback to LP driving video
                pose_driver = os.path.join(LP_DRIVING, "d30.jpg")

            result = self._run_lp(current_path, pose_driver, "pose", pose_multiplier, "pose-friendly")
            if result is not None:
                # Save intermediate result
                inter_path = os.path.join(POSE_DIR, "temp", "pose_intermediate.jpg")
                cv2.imwrite(inter_path, result)
                current_path = inter_path
                current_img = result
                steps_done.append(f"pose={pose} x{pose_multiplier}")

                # Check identity after pose
                pose_score = float(cosine_similarity(
                    src_emb, self._get_embedding(result)
                )[0][0]) if self._get_embedding(result) is not None else 0

                if pose_score < IDENTITY_GATE:
                    logger.info(f"  Pose step dropped identity to {pose_score:.4f}")

        # ── STEP 2: EXPRESSION (if requested) ──
        if expression:
            expr_driver = self._find_driver("expression", expression)
            if expr_driver is None:
                expr_driver = DEFAULT_SMILE_DRIVER

            result = self._run_lp(current_path, expr_driver, "lip", expr_multiplier)
            if result is not None:
                current_img = result
                steps_done.append(f"expr={expression} x{expr_multiplier}")

        # ── CHECK FINAL IDENTITY ──
        final_emb = self._get_embedding(current_img)
        final_score = float(cosine_similarity(src_emb, final_emb)[0][0]) if final_emb is not None else 0

        # ── SAVE ──
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if output_name is None:
            name = os.path.splitext(os.path.basename(source_path))[0]
            parts = []
            if pose: parts.append(pose)
            if expression: parts.append(expression)
            output_name = f"{name}_{'_'.join(parts) if parts else 'original'}.png"

        out_path = os.path.join(OUTPUT_DIR, output_name)
        cv2.imwrite(out_path, current_img)

        elapsed = time.time() - t0

        return {
            "success": final_score >= IDENTITY_GATE or not steps_done,
            "identity_score": round(final_score, 4),
            "output_path": out_path,
            "pose": pose,
            "expression": expression,
            "steps": steps_done,
            "time": round(elapsed, 1),
        }

    def batch_presets(self, source_path: str) -> Dict:
        """Generate all photoshoot presets for one photo."""
        presets = {
            "professional": {"pose": "slight_right", "expression": "subtle", "pose_mult": 0.5, "expr_mult": 1.5},
            "casual_happy": {"pose": "tilt_left", "expression": "natural", "pose_mult": 0.3, "expr_mult": 1.8},
            "confident": {"pose": None, "expression": "subtle", "pose_mult": 0, "expr_mult": 1.5},
            "warm_welcome": {"pose": "slight_left", "expression": "photoshoot", "pose_mult": 0.5, "expr_mult": 1.8},
            "smile_only": {"pose": None, "expression": "photoshoot", "pose_mult": 0, "expr_mult": 1.8},
        }

        results = {}
        for name, cfg in presets.items():
            r = self.edit(
                source_path,
                pose=cfg["pose"],
                expression=cfg["expression"],
                pose_multiplier=cfg["pose_mult"],
                expr_multiplier=cfg["expr_mult"],
                output_name=f"{os.path.splitext(os.path.basename(source_path))[0]}_{name}.png",
            )
            results[name] = r
            ok = "✓" if r["success"] else "✗"
            logger.info(f"  {ok} {name}: id={r['identity_score']:.4f} steps={r['steps']}")

        return results


if __name__ == "__main__":
    engine = PoseExpressionEngine()

    # Test on a face
    test = os.path.join(BASE_DIR, "MagicFace", "test_images", "ros1.jpg")
    if os.path.exists(test):
        print("\nGenerating all photoshoot presets...")
        results = engine.batch_presets(test)
        print(f"\nOutputs: {OUTPUT_DIR}/")
