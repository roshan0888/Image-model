#!/usr/bin/env python3
"""
Hybrid Smile Engine — Option C

Architecture:
  1. LivePortrait: Generate smile geometry (WHERE the smile should be)
  2. Extract mouth landmarks from LP output (smile shape)
  3. Create tight mouth mask on ORIGINAL source image
  4. Use Stable Diffusion Inpainting to repaint ONLY the mouth region
     guided by LP's smile geometry
  5. Result: Original high-res image with AI-generated realistic smile mouth

Why this works:
  - LP is good at: knowing how face muscles move during a smile
  - LP is bad at: generating high-resolution teeth/lip detail (256px limit)
  - SD Inpainting is good at: generating photorealistic detail at 512-1024px
  - SD Inpainting is bad at: knowing WHERE/HOW a smile should look
  - Combine: LP provides the smile blueprint, SD renders the pixels
"""

import os
import sys
import cv2
import time
import uuid
import json
import torch
import numpy as np
import logging
from typing import Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [hybrid] %(levelname)s: %(message)s")
logger = logging.getLogger("hybrid")

PHOTO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PHOTO_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "MagicFace", "third_party_files")

sys.path.insert(0, LP_DIR)

MOUTH_IDX = list(range(52, 72))
EYE_IDX = list(range(33, 52))

# Different gates per smile type — stronger expressions naturally have lower identity
GATES = {
    "subtle":     {"reject": 0.90, "good": 0.94, "perfect": 0.97},
    "photoshoot": {"reject": 0.82, "good": 0.88, "perfect": 0.94},
    "natural":    {"reject": 0.82, "good": 0.88, "perfect": 0.94},
}
GATE_REJECT = 0.82  # Default fallback


class HybridSmileEngine:
    """LP geometry + SD inpainting for photoshoot-quality smiles."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_analyzer = None
        self.lp_pipeline = None
        self.gfpgan_enhancer = None

        self._load_face_analyzer()
        self._load_liveportrait()
        self._load_mouth_enhancer()
        self._load_drivers()
        logger.info("HybridSmileEngine ready")

    def _load_face_analyzer(self):
        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2_DIR,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
        logger.info("  InsightFace loaded")

    def _load_liveportrait(self):
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        self.lp_pipeline = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig(),
        )
        logger.info("  LivePortrait loaded")

    def _load_mouth_enhancer(self):
        """Load GFPGAN for mouth-only quality enhancement.

        KEY: We apply GFPGAN ONLY to the mouth crop, not full face.
        This restores teeth/lip detail without hallucinating identity features.
        Previously GFPGAN was applied to entire face = identity drift.
        Now: mouth-only = quality boost + identity preserved.
        """
        import types as _types
        # Fix torchvision compatibility
        if "torchvision.transforms.functional_tensor" not in sys.modules:
            _ft = _types.ModuleType("torchvision.transforms.functional_tensor")
            def _rgb_to_grayscale(img, num_output_channels=1):
                r, g, b = img.unbind(dim=-3)
                l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
                l_img = l_img.unsqueeze(dim=-3)
                if num_output_channels == 3:
                    return l_img.expand(img.shape)
                return l_img
            _ft.rgb_to_grayscale = _rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = _ft

        from gfpgan import GFPGANer

        # Download model if not exists
        model_path = os.path.join(PHOTO_DIR, "weights", "GFPGANv1.4.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if not os.path.exists(model_path):
            # Check if it exists in the old location
            old_path = os.path.join(BASE_DIR, "gfpgan", "weights", "GFPGANv1.4.pth")
            if os.path.exists(old_path):
                import shutil
                shutil.copy2(old_path, model_path)
                logger.info(f"  Copied GFPGAN weights from {old_path}")
            else:
                logger.info("  Downloading GFPGANv1.4 weights...")
                import urllib.request
                url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
                urllib.request.urlretrieve(url, model_path)

        self.gfpgan_enhancer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
        )
        logger.info("  GFPGAN mouth enhancer loaded")

    def _load_drivers(self):
        drivers_dir = os.path.join(PHOTO_DIR, "drivers")
        lp_driving = os.path.join(LP_DIR, "assets", "examples", "driving")

        self.drivers = {"subtle": [], "photoshoot": [], "natural": []}

        for smile_type in self.drivers:
            type_dir = os.path.join(drivers_dir, smile_type)
            if os.path.exists(type_dir):
                for f in sorted(os.listdir(type_dir)):
                    if f.endswith('.jpg'):
                        self.drivers[smile_type].append(os.path.join(type_dir, f))

        # Fallbacks
        for p in [os.path.join(lp_driving, "d30.jpg"), os.path.join(lp_driving, "d12.jpg")]:
            if os.path.exists(p):
                for st in self.drivers:
                    if p not in self.drivers[st]:
                        self.drivers[st].append(p)

        for st, d in self.drivers.items():
            logger.info(f"  {st}: {len(d)} drivers")

    # ═══════════════════════════════════════════════════
    # FACE HELPERS
    # ═══════════════════════════════════════════════════

    def _get_face(self, img):
        faces = self.face_analyzer.get(img)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    def _get_embedding(self, img):
        face = self._get_face(img)
        if face is None:
            return None
        return face.normed_embedding.reshape(1, -1)

    def _identity_score(self, src_emb, img):
        emb = self._get_embedding(img)
        if emb is None:
            return 0.0
        return float(cosine_similarity(src_emb, emb)[0][0])

    def _get_landmarks(self, img):
        face = self._get_face(img)
        if face is None:
            return None
        return getattr(face, 'landmark_2d_106', None)

    # ═══════════════════════════════════════════════════
    # STEP 1: LP EXPRESSION TRANSFER (geometry only)
    # ═══════════════════════════════════════════════════

    def _run_lp(self, source_path, driver_path, multiplier=1.0):
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
    # STEP 2: CREATE MOUTH MASK FROM LP OUTPUT
    # ═══════════════════════════════════════════════════

    def _create_mouth_mask(self, lp_result, source_bgr, expansion=30):
        """Create mouth mask covering BOTH original and LP mouth positions.

        KEY FIX: When LP changes expression, the mouth MOVES (jaw opens/closes).
        We must cover BOTH the original mouth position AND the new LP mouth position,
        otherwise the original mouth shows through as a ghost.
        """
        h, w = source_bgr.shape[:2]
        lp_resized = cv2.resize(lp_result, (w, h))

        lp_lmk = self._get_landmarks(lp_resized)
        src_lmk = self._get_landmarks(source_bgr)

        if lp_lmk is None:
            return None, None

        # Create mask covering LP mouth (new smile position)
        lp_mouth_pts = lp_lmk[MOUTH_IDX].astype(np.int32)
        lp_mouth_hull = cv2.convexHull(lp_mouth_pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, lp_mouth_hull, 255)

        # ALSO cover source mouth (original position) to prevent ghost
        if src_lmk is not None:
            src_mouth_pts = src_lmk[MOUTH_IDX].astype(np.int32)
            src_mouth_hull = cv2.convexHull(src_mouth_pts)
            cv2.fillConvexPoly(mask, src_mouth_hull, 255)

        # Expand generously to cover all mouth-adjacent areas
        # This prevents any trace of the original mouth showing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (expansion, expansion))
        mask = cv2.dilate(mask, kernel, iterations=3)  # Was 2, now 3

        # Smooth edges
        mask = cv2.GaussianBlur(mask, (31, 31), 12)

        # Get center of the COMBINED mask
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = w // 2, int(h * 0.65)

        return mask, (cx, cy)

    # ═══════════════════════════════════════════════════
    # STEP 3: ENHANCE MOUTH QUALITY WITH GFPGAN
    # ═══════════════════════════════════════════════════

    def _enhance_and_composite(self, source_bgr, lp_result, mask, smile_type="photoshoot"):
        """
        Hybrid compositing:
        1. Enhance LP face with GFPGAN (restores teeth/lip detail)
        2. Take ONLY the mouth region from enhanced LP
        3. Seamless clone into original source image
        4. Source identity preserved everywhere except mouth

        This is different from applying GFPGAN to full output:
        - GFPGAN on full face → hallucinate identity features → identity drops
        - GFPGAN on LP crop → restore teeth detail → paste mouth only → identity safe
        """
        h, w = source_bgr.shape[:2]
        lp_resized = cv2.resize(lp_result, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Step 3a: Enhance LP result with GFPGAN (mouth detail restoration)
        try:
            _, _, enhanced = self.gfpgan_enhancer.enhance(
                lp_resized, has_aligned=False, only_center_face=True, paste_back=True
            )
            if enhanced is not None:
                enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LANCZOS4)
                logger.info("    GFPGAN enhanced LP output")
            else:
                enhanced = lp_resized
        except Exception as e:
            logger.debug(f"    GFPGAN failed: {e}")
            enhanced = lp_resized

        # Step 3b: Warp source to match LP geometry (for seamless transition)
        src_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
        lp_gray = cv2.cvtColor(lp_resized, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            src_gray, lp_gray, None,
            pyr_scale=0.5, levels=5, winsize=13,
            iterations=5, poly_n=7, poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        map_x = np.float32(np.arange(w)[np.newaxis, :] + flow[:, :, 0])
        map_y = np.float32(np.arange(h)[:, np.newaxis] + flow[:, :, 1])
        warped_source = cv2.remap(source_bgr, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

        # Step 3c: Seamless clone enhanced mouth into warped source
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = w // 2, int(h * 0.65)

        cx = max(1, min(w - 2, cx))
        cy = max(1, min(h - 2, cy))

        try:
            result = cv2.seamlessClone(
                enhanced,         # GFPGAN-enhanced LP (good teeth)
                warped_source,    # Warped original (identity)
                mask,             # Mouth region only
                (cx, cy),
                cv2.MIXED_CLONE
            )
        except cv2.error:
            # Fallback: alpha blend
            mask_f = cv2.GaussianBlur(mask, (31, 31), 10).astype(np.float32) / 255.0
            mask_3ch = mask_f[:, :, np.newaxis]
            result = (
                enhanced.astype(np.float32) * mask_3ch +
                warped_source.astype(np.float32) * (1 - mask_3ch)
            )
            result = np.clip(result, 0, 255).astype(np.uint8)

        # Step 3d: Color match face region to source (prevent color shift)
        lp_lmk = self._get_landmarks(result)
        if lp_lmk is not None:
            face_hull = cv2.convexHull(lp_lmk.astype(np.int32))
            face_mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillConvexPoly(face_mask, face_hull, 1.0)
            face_mask = cv2.GaussianBlur(face_mask, (31, 31), 10)

            # LAB color match
            mask_bool = face_mask > 0.3
            if mask_bool.sum() > 100:
                src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
                res_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float64)
                mf = face_mask[:, :, np.newaxis].astype(np.float64)
                for ch in range(3):
                    sv = src_lab[:, :, ch][mask_bool]
                    rv = res_lab[:, :, ch][mask_bool]
                    sm, ss = sv.mean(), max(sv.std(), 1e-6)
                    rm, rs = rv.mean(), max(rv.std(), 1e-6)
                    matched = (res_lab[:, :, ch] - rm) * (ss / rs) + sm
                    res_lab[:, :, ch] = res_lab[:, :, ch] * (1 - mf[:,:,0]) + matched * mf[:,:,0]
                result = cv2.cvtColor(np.clip(res_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Step 3e: Inject source skin texture (for photoshoot quality)
        src_float = source_bgr.astype(np.float32)
        src_blur = cv2.GaussianBlur(src_float, (0, 0), 3)
        src_detail = src_float - src_blur
        warped_detail = cv2.remap(src_detail, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        if lp_lmk is not None:
            texture_mask = face_mask[:, :, np.newaxis] * 0.20
            result = result.astype(np.float32) + warped_detail * texture_mask
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    # ═══════════════════════════════════════════════════
    # MAIN SMILE FUNCTION
    # ═══════════════════════════════════════════════════

    def smile(self, source_path: str, smile_type: str = "photoshoot",
              output_name: Optional[str] = None) -> Dict:
        """Apply photoshoot smile using hybrid LP + SD inpainting."""
        t0 = time.time()
        request_id = f"hyb_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        source_bgr = cv2.imread(source_path)
        if source_bgr is None:
            return {"success": False, "message": "Cannot read image", "request_id": request_id}

        src_emb = self._get_embedding(source_bgr)
        if src_emb is None:
            return {"success": False, "message": "No face detected", "request_id": request_id}

        drivers = self.drivers.get(smile_type, self.drivers["photoshoot"])
        if not drivers:
            return {"success": False, "message": "No drivers available", "request_id": request_id}

        # Get gates for this smile type
        gates = GATES.get(smile_type, GATES["photoshoot"])
        gate_reject = gates["reject"]
        gate_good = gates["good"]
        gate_perfect = gates["perfect"]

        best_result = None
        best_score = 0.0
        best_info = {}
        attempts = 0

        # Try top 5 drivers with multiple multipliers
        for driver_path in drivers[:5]:
            driver_name = os.path.basename(driver_path)

            for mult in [0.8, 1.0, 1.3]:
                attempts += 1

                # Step 1: Run LP to get smile geometry
                lp_result = self._run_lp(source_path, driver_path, mult)
                if lp_result is None:
                    continue

                # Step 2: Create mouth mask from LP output
                mask, center = self._create_mouth_mask(lp_result, source_bgr)
                if mask is None:
                    continue

                # Step 3: Enhance mouth + composite into source
                try:
                    result = self._enhance_and_composite(source_bgr, lp_result, mask, smile_type)
                except Exception as e:
                    logger.debug(f"  Composite failed: {e}")
                    continue

                # Step 4: Check identity
                score = self._identity_score(src_emb, result)
                logger.info(f"  {driver_name} x{mult}: identity={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_result = result
                    best_info = {"driver": driver_name, "multiplier": mult}

                if score >= gate_perfect:
                    break

            if best_score >= gate_perfect:
                break

        elapsed = time.time() - t0

        if best_result is None or best_score < gate_reject:
            return {
                "success": False,
                "identity_score": best_score,
                "message": "Could not achieve good identity match",
                "attempts": attempts,
                "time": round(elapsed, 1),
                "request_id": request_id,
            }

        # Save output
        if output_name is None:
            src_name = os.path.splitext(os.path.basename(source_path))[0]
            output_name = f"{src_name}_{smile_type}_hybrid.png"

        output_path = os.path.join(PHOTO_DIR, "outputs", output_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, best_result)

        quality = (
            "Perfect match!" if best_score >= gate_perfect else
            "Looks great!" if best_score >= gate_good else
            "Good match"
        )

        return {
            "success": True,
            "identity_score": round(best_score, 4),
            "quality": quality,
            "smile_type": smile_type,
            "output_path": output_path,
            "driver_used": best_info.get("driver", "—"),
            "multiplier": best_info.get("multiplier", 1.0),
            "attempts": attempts,
            "time": round(elapsed, 1),
            "request_id": request_id,
        }


if __name__ == "__main__":
    engine = HybridSmileEngine()

    source = os.path.join(BASE_DIR, "MagicFace", "test_images", "ros1.jpg")

    for st in ["subtle", "photoshoot"]:
        print(f"\nTesting: {st}")
        r = engine.smile(source, smile_type=st)
        ok = "✓" if r["success"] else "✗"
        print(f"  {ok} identity={r.get('identity_score', 0):.4f} "
              f"quality={r.get('quality', '—')} time={r.get('time', 0):.1f}s")
        if r["success"]:
            print(f"  Output: {r['output_path']}")
