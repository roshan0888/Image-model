#!/usr/bin/env python3
"""
Production Smile Engine v1

Two innovations inside LivePortrait:
  1. Dynamic region mask — only warp mouth/cheeks, preserve eyes/forehead
  2. Identity-guided search — binary search for max expression at 95%+ identity

No external models. No compositing. No GFPGAN. No SD. Just LP done right.
"""

import os
import sys
import cv2
import time
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [prod] %(levelname)s: %(message)s")
logger = logging.getLogger("prod")

PHOTO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PHOTO_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "MagicFace", "third_party_files")

sys.path.insert(0, LP_DIR)

# Identity gates
GATE_MINIMUM = 0.95   # Never show below this
GATE_TARGET = 0.97    # Try to achieve this


class ProductionSmileEngine:

    def __init__(self):
        self._load_face_analyzer()
        self._load_liveportrait()
        self.driver = os.path.join(LP_DIR, "assets", "examples", "driving", "d12.jpg")
        logger.info("ProductionSmileEngine ready")

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

    def _get_source_landmarks(self, source_path):
        """Get landmarks in LP's 256x256 crop space for dynamic mask."""
        from src.config.crop_config import CropConfig
        crop_cfg = CropConfig()
        source_bgr = cv2.imread(source_path)
        crop_info = self.lp.cropper.crop_source_image(source_bgr, crop_cfg)
        if crop_info is None:
            return None
        # IMPORTANT: use lmk_crop_256x256, NOT lmk_crop
        # lmk_crop is in the pre-resize space (512x512)
        # lmk_crop_256x256 is in the actual 256x256 space that LP processes
        if 'lmk_crop_256x256' in crop_info:
            return crop_info['lmk_crop_256x256']
        # Fallback: scale lmk_crop to 256x256
        lmk = crop_info['lmk_crop']
        img_crop = crop_info.get('img_crop')
        if img_crop is not None:
            scale = 256.0 / img_crop.shape[0]
            return lmk * scale
        return lmk

    def _run_lp(self, source_path, driver_path, multiplier, use_mouth_mask=True, landmarks=None):
        """Run LP at full expression, then mask output to mouth-only at PIXEL level."""
        from src.config.argument_config import ArgumentConfig

        # Use warping-level mask: warp lower 60% of face, preserve upper 40% (eyes/forehead)
        if use_mouth_mask:
            if landmarks is not None:
                self.lp.live_portrait_wrapper.warping_module.set_region_mask(landmarks)
            else:
                self.lp.live_portrait_wrapper.warping_module.set_region_mask("mouth_only")
        else:
            self.lp.live_portrait_wrapper.warping_module.set_region_mask(None)

        args = ArgumentConfig()
        args.source = source_path
        args.driving = driver_path
        args.output_dir = os.path.join(PHOTO_DIR, "temp")
        os.makedirs(args.output_dir, exist_ok=True)
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "all"
        args.driving_option = "expression-friendly"
        args.driving_multiplier = multiplier
        args.source_max_dim = 1920
        args.flag_eye_retargeting = False
        args.flag_lip_retargeting = False

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
            result = cv2.imread(wfp)
            # DON'T reset mask here — identity search calls this multiple times
            return result
        except Exception as e:
            logger.debug(f"LP failed: {e}")
            return None

    def smile(self, source_path, output_dir=None):
        """Generate the best possible smile for this face.

        Strategy:
          1. Get source face landmarks for dynamic mask
          2. Binary search for the MAXIMUM multiplier that keeps identity >= 95%
          3. Return the result with highest expression at 95%+ identity

        Returns dict with success, identity_score, output_path, multiplier_used
        """
        t0 = time.time()

        source_bgr = cv2.imread(source_path)
        if source_bgr is None:
            return {"success": False, "message": "Cannot read image"}

        src_emb = self._get_embedding(source_bgr)
        if src_emb is None:
            return {"success": False, "message": "No face detected"}

        # Get landmarks for dynamic mask
        landmarks = self._get_source_landmarks(source_path)

        # ─── IDENTITY-GUIDED SEARCH ───
        # Binary search: find max multiplier where identity >= GATE_MINIMUM
        # Start from high (1.5) and work down
        #
        # The key insight: every face has a different "smile tolerance"
        # Some faces can handle 1.5x and stay above 95%
        # Some faces drop below 95% at 0.8x
        # We find the sweet spot for THIS specific face

        multipliers_to_try = [1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4]
        best_result = None
        best_score = 0.0
        best_mult = 0.0
        all_attempts = []

        for mult in multipliers_to_try:
            result = self._run_lp(source_path, self.driver, mult,
                                   use_mouth_mask=True, landmarks=landmarks)
            if result is None:
                continue

            score = self._identity_score(src_emb, result)
            all_attempts.append((mult, score))

            if score >= GATE_MINIMUM and mult > best_mult:
                # This multiplier passes the gate AND is stronger than previous best
                best_result = result
                best_score = score
                best_mult = mult

            # If we found a high multiplier that passes, no need to try lower
            if score >= GATE_TARGET and mult >= 1.0:
                break

        elapsed = time.time() - t0

        # Reset mask
        self.lp.live_portrait_wrapper.warping_module.set_region_mask(None)

        if best_result is None:
            return {
                "success": False,
                "message": "Could not achieve 95%+ identity",
                "best_identity": max([s for _, s in all_attempts]) if all_attempts else 0,
                "attempts": all_attempts,
                "time": round(elapsed, 1),
            }

        # Save output
        if output_dir is None:
            output_dir = os.path.join(PHOTO_DIR, "final_output")
        os.makedirs(output_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(source_path))[0]
        out_path = os.path.join(output_dir, f"{name}_smile.png")
        cv2.imwrite(out_path, best_result)

        # Also save original for comparison
        cv2.imwrite(os.path.join(output_dir, f"{name}_original.png"), source_bgr)

        # Build comparison strip
        h_t = 300
        s1 = h_t / source_bgr.shape[0]
        s2 = h_t / best_result.shape[0]
        o_r = cv2.resize(source_bgr, (int(source_bgr.shape[1]*s1), h_t))
        r_r = cv2.resize(best_result, (int(best_result.shape[1]*s2), h_t))
        cv2.putText(o_r, "Original", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(r_r, f"Smile id={best_score:.3f} x{best_mult}", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        w_m = max(o_r.shape[1], r_r.shape[1])
        if o_r.shape[1] < w_m:
            o_r = np.hstack([o_r, np.zeros((h_t, w_m-o_r.shape[1], 3), dtype=np.uint8)])
        if r_r.shape[1] < w_m:
            r_r = np.hstack([r_r, np.zeros((h_t, w_m-r_r.shape[1], 3), dtype=np.uint8)])
        cv2.imwrite(os.path.join(output_dir, f"{name}_comparison.png"), np.hstack([o_r, r_r]))

        logger.info(f"  {name}: id={best_score:.4f} mult={best_mult}x time={elapsed:.1f}s")

        return {
            "success": True,
            "identity_score": round(best_score, 4),
            "multiplier_used": best_mult,
            "output_path": out_path,
            "attempts": all_attempts,
            "time": round(elapsed, 1),
        }


if __name__ == "__main__":
    engine = ProductionSmileEngine()

    men = [
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0008.jpg", "man_asian"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0013.jpg", "man_black"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0015.jpg", "man_suit"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0019.jpg", "man_young"),
        ("../training_engine/dataset/cleaned_scraped/neutral/clean_neutral_0009.jpg", "man_older"),
    ]

    OUT = os.path.join(PHOTO_DIR, "final_output")
    os.makedirs(OUT, exist_ok=True)

    print(f"\n{'='*70}")
    print("PRODUCTION ENGINE — Dynamic Mask + Identity-Guided Search")
    print(f"Target: {GATE_MINIMUM*100:.0f}%+ identity with maximum expression")
    print(f"{'='*70}\n")

    for src_path, name in men:
        r = engine.smile(src_path, output_dir=OUT)
        if r["success"]:
            print(f"  ✓ {name:<12} id={r['identity_score']:.4f}  mult={r['multiplier_used']}x  time={r['time']:.1f}s")
            print(f"    Search: {[(f'{m:.1f}x→{s:.3f}') for m, s in r['attempts']]}")
        else:
            print(f"  ✗ {name:<12} best_id={r.get('best_identity',0):.4f}")
            print(f"    Search: {[(f'{m:.1f}x→{s:.3f}') for m, s in r.get('attempts',[])]}")

    print(f"\n{'='*70}")
    print(f"Output: {OUT}/")
