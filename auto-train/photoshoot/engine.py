#!/usr/bin/env python3
"""
Photoshoot Smile Engine — Two-Pass Architecture

Pass 1: LivePortrait expression transfer (get the smile)
Pass 2: Identity restoration (fix what LP broke)

Result: Visible smile + preserved identity
"""

import os
import sys
import cv2
import json
import time
import uuid
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [engine] %(levelname)s: %(message)s")
logger = logging.getLogger("engine")

PHOTO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PHOTO_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "MagicFace", "third_party_files")

sys.path.insert(0, LP_DIR)

# Identity thresholds
GATE_PERFECT = 0.97   # Show with "Perfect match!"
GATE_GOOD = 0.94      # Show with "Looks great!"
GATE_OK = 0.90        # Show with "Good match"
GATE_REJECT = 0.90    # Below this = never show

# Expression regions in 106-landmark model
MOUTH_IDX = list(range(52, 72))   # Mouth
EYE_IDX = list(range(33, 52))     # Eyes
BROW_IDX = list(range(0, 33))     # Brow + contour


class PhotoshootEngine:
    """Two-pass smile engine for production photo editing."""

    def __init__(self):
        self.face_analyzer = None
        self.lp_pipeline = None
        self._load_models()
        self._load_drivers()
        logger.info("PhotoshootEngine ready")

    def _load_models(self):
        """Load InsightFace + LivePortrait."""
        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2_DIR,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        self.lp_pipeline = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig(),
        )
        logger.info("  Models loaded")

    def _load_drivers(self):
        """Load classified smile driving images."""
        drivers_dir = os.path.join(PHOTO_DIR, "drivers")
        lp_driving = os.path.join(LP_DIR, "assets", "examples", "driving")

        self.drivers = {
            "subtle": [],
            "photoshoot": [],
            "natural": [],
        }

        # Load classified drivers
        for smile_type in self.drivers:
            type_dir = os.path.join(drivers_dir, smile_type)
            if os.path.exists(type_dir):
                for f in sorted(os.listdir(type_dir)):
                    if f.endswith('.jpg'):
                        self.drivers[smile_type].append(os.path.join(type_dir, f))

        # Fallback: LP original drivers
        lp_fallbacks = {
            "subtle": [os.path.join(lp_driving, "d30.jpg")],
            "photoshoot": [os.path.join(lp_driving, "d12.jpg")],
            "natural": [os.path.join(lp_driving, "d12.jpg")],
        }
        for st, paths in lp_fallbacks.items():
            for p in paths:
                if os.path.exists(p) and p not in self.drivers[st]:
                    self.drivers[st].append(p)

        for st, drivers in self.drivers.items():
            logger.info(f"  {st}: {len(drivers)} drivers loaded")

    # ═══════════════════════════════════════════════════
    # FACE ANALYSIS
    # ═══════════════════════════════════════════════════

    def _get_face(self, image_bgr):
        """Detect largest face."""
        faces = self.face_analyzer.get(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    def _get_embedding(self, image_bgr):
        """Get ArcFace embedding."""
        face = self._get_face(image_bgr)
        if face is None:
            return None
        return face.normed_embedding.reshape(1, -1)

    def _identity_score(self, src_emb, image_bgr):
        """Measure identity preservation."""
        emb = self._get_embedding(image_bgr)
        if emb is None:
            return 0.0
        return float(cosine_similarity(src_emb, emb)[0][0])

    def _get_landmarks(self, image_bgr):
        """Get 106 landmarks."""
        face = self._get_face(image_bgr)
        if face is None:
            return None
        return getattr(face, 'landmark_2d_106', None)

    def score_photo(self, image_bgr) -> Dict:
        """Score input photo quality (pre-flight check)."""
        face = self._get_face(image_bgr)
        if face is None:
            return {"score": 0, "stars": 0, "message": "No face detected", "usable": False}

        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1

        checks = {}

        # Face size
        face_ratio = (fw * fh) / (h * w)
        checks["face_size"] = min(face_ratio / 0.15, 1.0)

        # Resolution
        checks["resolution"] = min(min(fw, fh) / 256, 1.0)

        # Sharpness
        face_crop = image_bgr[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        checks["sharpness"] = min(lap / 200, 1.0)

        # Pose (frontal)
        lmk = getattr(face, 'landmark_2d_106', None)
        if lmk is not None and len(lmk) > 86:
            nose_x = lmk[86, 0]
            offset = abs(nose_x - (x1 + x2) / 2) / max(fw, 1)
            checks["frontal"] = max(0, 1.0 - offset * 5)
        else:
            checks["frontal"] = 0.5

        # Lighting
        brightness = gray.mean()
        checks["lighting"] = 1.0 if 60 < brightness < 200 else 0.5

        # Overall score
        score = (
            checks["face_size"] * 0.25 +
            checks["resolution"] * 0.20 +
            checks["sharpness"] * 0.20 +
            checks["frontal"] * 0.25 +
            checks["lighting"] * 0.10
        )

        stars = min(5, max(1, int(score * 5 + 0.5)))

        messages = {
            5: "Perfect for all smile types!",
            4: "Great photo — all smiles will work well",
            3: "Good — subtle and photoshoot smiles will work",
            2: "Fair — only subtle smile recommended",
            1: "Try a clearer, front-facing photo for best results",
        }

        return {
            "score": round(score, 3),
            "stars": stars,
            "message": messages[stars],
            "usable": score >= 0.3,
            "checks": {k: round(v, 2) for k, v in checks.items()},
            "recommended_types": (
                ["subtle", "photoshoot", "natural"] if stars >= 4 else
                ["subtle", "photoshoot"] if stars >= 3 else
                ["subtle"] if stars >= 2 else
                []
            ),
        }

    # ═══════════════════════════════════════════════════
    # PASS 1: LIVEPORTRAIT EXPRESSION TRANSFER
    # ═══════════════════════════════════════════════════

    def _run_lp(self, source_path, driver_path, multiplier=1.0, use_retarget=False):
        """Run LivePortrait."""
        from src.config.argument_config import ArgumentConfig

        out_dir = os.path.join(PHOTO_DIR, "temp")
        os.makedirs(out_dir, exist_ok=True)

        args = ArgumentConfig()
        args.source = source_path
        args.driving = driver_path
        args.output_dir = out_dir
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "all"
        args.driving_option = "expression-friendly"
        args.driving_multiplier = multiplier
        args.source_max_dim = 1920

        if use_retarget:
            args.flag_eye_retargeting = True
            args.flag_lip_retargeting = True

        inf_keys = {
            'flag_pasteback', 'flag_do_crop', 'flag_stitching',
            'flag_relative_motion', 'animation_region', 'driving_option',
            'driving_multiplier', 'source_max_dim', 'source_division',
            'flag_eye_retargeting', 'flag_lip_retargeting',
        }
        self.lp_pipeline.live_portrait_wrapper.update_config(
            {k: v for k, v in args.__dict__.items() if k in inf_keys}
        )
        try:
            wfp, _ = self.lp_pipeline.execute(args)
            return cv2.imread(wfp)
        except:
            return None

    # ═══════════════════════════════════════════════════
    # PASS 2: IDENTITY RESTORATION
    # ═══════════════════════════════════════════════════

    def _restore_identity(self, source_bgr, lp_result, smile_type="photoshoot"):
        """
        Restore identity using MOUTH-ONLY SWAP with seamless blending.

        NEW APPROACH:
          1. Start with 100% source image (perfect identity)
          2. Extract ONLY mouth+chin from LP (the smile)
          3. Seamless clone LP mouth into source (invisible boundary)
          4. Add subtle cheek raise from LP (natural smile look)
          5. Inject source skin texture for photoshoot quality

        Result: Source identity everywhere + LP smile in mouth region only
        """
        h, w = lp_result.shape[:2]
        src = cv2.resize(source_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Get landmarks
        lp_lmk = self._get_landmarks(lp_result)
        src_lmk = self._get_landmarks(src)

        if lp_lmk is None or src_lmk is None:
            return lp_result

        # ── Step 1: Create tight mouth mask from LP landmarks ──
        # Include: lips, teeth, chin dimple, nasolabial folds (smile lines)
        mouth_pts = lp_lmk[MOUTH_IDX].astype(np.int32)
        mouth_hull = cv2.convexHull(mouth_pts)

        # Expand slightly to include smile lines and chin movement
        mouth_mask_tight = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mouth_mask_tight, mouth_hull, 255)

        # Expand mask to include nasolabial folds (smile lines run from nose to mouth corners)
        if smile_type in ("photoshoot", "natural"):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            mouth_mask_tight = cv2.dilate(mouth_mask_tight, kernel, iterations=2)
        else:  # subtle
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mouth_mask_tight = cv2.dilate(mouth_mask_tight, kernel, iterations=1)

        # ── Step 2: Warp source to match LP geometry in mouth region ──
        # The source jaw/chin needs to move slightly to accommodate the smile
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        lp_gray = cv2.cvtColor(lp_result, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            src_gray, lp_gray, None,
            pyr_scale=0.5, levels=5, winsize=13,
            iterations=5, poly_n=7, poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        map_x = np.float32(np.arange(w)[np.newaxis, :] + flow[:, :, 0])
        map_y = np.float32(np.arange(h)[:, np.newaxis] + flow[:, :, 1])
        warped_source = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

        # ── Step 3: Seamless clone LP mouth into warped source ──
        # This is the KEY improvement — cv2.seamlessClone handles:
        # - Lighting matching at boundary
        # - Color blending (no visible seam)
        # - Gradient-domain compositing (mathematically optimal)

        # Find center of mouth mask for seamless clone
        moments = cv2.moments(mouth_mask_tight)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = w // 2, int(h * 0.65)

        # Ensure center is within valid range
        cx = max(1, min(w - 2, cx))
        cy = max(1, min(h - 2, cy))

        try:
            # MIXED_CLONE preserves more texture than NORMAL_CLONE
            result = cv2.seamlessClone(
                lp_result,           # Source (LP with smile)
                warped_source,       # Destination (warped original)
                mouth_mask_tight,    # Mask (mouth region only)
                (cx, cy),            # Center point
                cv2.MIXED_CLONE      # Preserves texture from destination
            )
        except cv2.error:
            # Fallback to alpha blend if seamless clone fails
            mask_f = cv2.GaussianBlur(mouth_mask_tight, (31, 31), 10).astype(np.float32) / 255.0
            mask_3ch = mask_f[:, :, np.newaxis]
            result = (
                lp_result.astype(np.float32) * mask_3ch +
                warped_source.astype(np.float32) * (1 - mask_3ch)
            )
            result = np.clip(result, 0, 255).astype(np.uint8)

        # ── Step 4: Add subtle cheek raise (natural smile involves cheeks) ──
        if smile_type in ("photoshoot", "natural"):
            # Create cheek mask (area between mouth corners and eyes)
            cheek_mask = np.zeros((h, w), dtype=np.float32)

            # Left cheek: between left mouth corner and left eye
            left_mouth = lp_lmk[52].astype(int)
            left_eye = lp_lmk[40].astype(int)
            cheek_center_l = ((left_mouth + left_eye) // 2).astype(int)
            cv2.circle(cheek_mask, tuple(cheek_center_l), int(h * 0.06), 1.0, -1)

            # Right cheek
            right_mouth = lp_lmk[58].astype(int)
            right_eye = lp_lmk[49].astype(int)
            cheek_center_r = ((right_mouth + right_eye) // 2).astype(int)
            cv2.circle(cheek_mask, tuple(cheek_center_r), int(h * 0.06), 1.0, -1)

            cheek_mask = cv2.GaussianBlur(cheek_mask, (41, 41), 15)

            # Blend LP cheeks lightly (15% for photoshoot, 25% for natural)
            cheek_weight = 0.15 if smile_type == "photoshoot" else 0.25
            cheek_3ch = cheek_mask[:, :, np.newaxis] * cheek_weight
            result = (
                result.astype(np.float32) * (1 - cheek_3ch) +
                lp_result.astype(np.float32) * cheek_3ch
            )
            result = np.clip(result, 0, 255).astype(np.uint8)

        # ── Step 5: Inject source skin texture for quality ──
        # Extract high-frequency detail from source → overlay on result
        face_hull = cv2.convexHull(lp_lmk.astype(np.int32))
        face_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillConvexPoly(face_mask, face_hull, 1.0)
        face_mask = cv2.GaussianBlur(face_mask, (31, 31), 10)

        result = self._inject_texture(src, result, face_mask, strength=0.20)

        return result

    def _match_color(self, source, target, mask):
        """Match target face colors to source in LAB space."""
        mask_bool = mask > 0.3
        if mask_bool.sum() < 100:
            return target
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float64)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float64)
        result_lab = tgt_lab.copy()
        mf = mask[:, :, np.newaxis].astype(np.float64)
        for ch in range(3):
            sv = src_lab[:, :, ch][mask_bool]
            tv = tgt_lab[:, :, ch][mask_bool]
            sm, ss = sv.mean(), max(sv.std(), 1e-6)
            tm, ts = tv.mean(), max(tv.std(), 1e-6)
            matched = (tgt_lab[:, :, ch] - tm) * (ss / ts) + sm
            result_lab[:, :, ch] = tgt_lab[:, :, ch] * (1 - mf[:,:,0]) + matched * mf[:,:,0]
        return cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _inject_texture(self, source, target, mask, strength=0.15):
        """Inject source skin texture into target."""
        h, w = target.shape[:2]
        src = cv2.resize(source, (w, h))

        # Extract high-frequency detail from source
        src_float = src.astype(np.float32)
        src_blur = cv2.GaussianBlur(src_float, (0, 0), 3)
        src_detail = src_float - src_blur

        # Warp to match target geometry
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            src_gray, tgt_gray, None,
            pyr_scale=0.5, levels=3, winsize=11,
            iterations=3, poly_n=5, poly_sigma=1.1,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        map_x = np.float32(np.arange(w)[np.newaxis, :] + flow[:, :, 0])
        map_y = np.float32(np.arange(h)[:, np.newaxis] + flow[:, :, 1])
        warped_detail = cv2.remap(src_detail, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

        mask_3ch = mask[:, :, np.newaxis]
        result = target.astype(np.float32) + warped_detail * strength * mask_3ch
        return np.clip(result, 0, 255).astype(np.uint8)

    # ═══════════════════════════════════════════════════
    # MAIN SMILE FUNCTION
    # ═══════════════════════════════════════════════════

    def smile(
        self,
        source_path: str,
        smile_type: str = "photoshoot",
        output_name: Optional[str] = None,
    ) -> Dict:
        """
        Apply photogenic smile to source image.

        Args:
            source_path: Path to source photo (any camera, any size)
            smile_type: "subtle", "photoshoot", or "natural"
            output_name: Custom output filename

        Returns:
            Dict with success, identity_score, output_path, etc.
        """
        t0 = time.time()
        request_id = f"ps_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        # ── Load source ──
        source_bgr = cv2.imread(source_path)
        if source_bgr is None:
            return {"success": False, "message": "Cannot read image", "request_id": request_id}

        # ── Pre-flight score ──
        photo_score = self.score_photo(source_bgr)
        if not photo_score["usable"]:
            return {
                "success": False,
                "message": photo_score["message"],
                "photo_score": photo_score,
                "request_id": request_id,
            }

        if smile_type not in photo_score["recommended_types"]:
            # Still try, but note it
            logger.info(f"  Note: {smile_type} not recommended for this photo (stars={photo_score['stars']})")

        # ── Get source identity ──
        src_emb = self._get_embedding(source_bgr)
        if src_emb is None:
            return {"success": False, "message": "Cannot detect face", "request_id": request_id}

        # ── Get drivers for this smile type ──
        drivers = self.drivers.get(smile_type, [])
        if not drivers:
            return {"success": False, "message": f"No drivers for {smile_type}", "request_id": request_id}

        # ── PASS 1: Try drivers until one works ──
        # Multipliers to try: full → reduced → retarget
        attempts = [
            # (multiplier, use_retarget, description)
            (1.2, False, "standard 1.2x"),
            (1.0, False, "standard 1.0x"),
            (0.8, False, "standard 0.8x"),
            (1.0, True, "retarget 1.0x"),
        ]

        best_result = None
        best_score = 0.0
        best_info = {}
        total_attempts = 0

        for driver_path in drivers[:5]:  # Try top 5 drivers
            driver_name = os.path.basename(driver_path)

            for mult, retarget, desc in attempts:
                total_attempts += 1
                lp_result = self._run_lp(source_path, driver_path, mult, retarget)
                if lp_result is None:
                    continue

                # Resize to match source
                h, w = source_bgr.shape[:2]
                if lp_result.shape[:2] != (h, w):
                    lp_result = cv2.resize(lp_result, (w, h), interpolation=cv2.INTER_LANCZOS4)

                # ── PASS 2: Identity restoration ──
                restored = self._restore_identity(source_bgr, lp_result, smile_type)

                # ── Check identity ──
                score = self._identity_score(src_emb, restored)

                if score > best_score:
                    best_score = score
                    best_result = restored
                    best_info = {
                        "driver": driver_name,
                        "multiplier": mult,
                        "retarget": retarget,
                        "desc": desc,
                    }

                # If we hit perfect, stop searching
                if score >= GATE_PERFECT:
                    break

            if best_score >= GATE_PERFECT:
                break

        # ── Decision: show or reject ──
        elapsed = time.time() - t0

        if best_result is None or best_score < GATE_REJECT:
            return {
                "success": False,
                "identity_score": best_score,
                "message": "Could not achieve good identity match. Try a front-facing photo!",
                "photo_score": photo_score,
                "attempts": total_attempts,
                "time": round(elapsed, 1),
                "request_id": request_id,
            }

        # ── Save output ──
        if output_name is None:
            src_name = os.path.splitext(os.path.basename(source_path))[0]
            output_name = f"{src_name}_{smile_type}.png"

        output_path = os.path.join(PHOTO_DIR, "outputs", output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, best_result)

        # Quality label
        if best_score >= GATE_PERFECT:
            quality = "Perfect match!"
        elif best_score >= GATE_GOOD:
            quality = "Looks great!"
        elif best_score >= GATE_OK:
            quality = "Good match"
        else:
            quality = "Fair"

        logger.info(
            f"  {smile_type}: identity={best_score:.4f} ({quality}) "
            f"driver={best_info.get('driver','—')} "
            f"attempts={total_attempts} time={elapsed:.1f}s"
        )

        return {
            "success": True,
            "identity_score": round(best_score, 4),
            "quality": quality,
            "smile_type": smile_type,
            "output_path": output_path,
            "photo_score": photo_score,
            "driver_used": best_info.get("driver", "—"),
            "multiplier": best_info.get("multiplier", 1.0),
            "attempts": total_attempts,
            "time": round(elapsed, 1),
            "request_id": request_id,
        }


if __name__ == "__main__":
    engine = PhotoshootEngine()

    source = os.path.join(BASE_DIR, "MagicFace", "test_images", "ros1.jpg")

    for st in ["subtle", "photoshoot", "natural"]:
        r = engine.smile(source, smile_type=st)
        ok = "✓" if r["success"] else "✗"
        print(f"  {ok} {st}: identity={r.get('identity_score', 0):.4f} "
              f"quality={r.get('quality', '—')} time={r.get('time', 0):.1f}s")
