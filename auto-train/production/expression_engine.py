#!/usr/bin/env python3
"""
Smart Expression Engine — Production

Core logic:
  1. Cascading fallback: try best driver → subtle driver → reduced intensity
  2. Quality gate: NEVER return result with identity < threshold
  3. Multiple driver candidates per expression ranked by quality
  4. Intensity slider support

Usage:
  engine = ExpressionEngine()
  result = engine.edit(source_path, expression="smile", intensity=0.7)

  if result["success"]:
      # Show result["output_path"] to user
  else:
      # Show result["message"] to user
"""

import os
import sys
import cv2
import json
import time
import shutil
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Paths
PROD_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROD_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "MagicFace", "third_party_files")
DRIVERS_DIR = os.path.join(PROD_DIR, "drivers")
OUTPUT_DIR = os.path.join(PROD_DIR, "outputs")
TEMP_DIR = os.path.join(PROD_DIR, "temp")

if LP_DIR not in sys.path:
    sys.path.insert(0, LP_DIR)

# Fix basicsr/torchvision compatibility
import types as _types
if "torchvision.transforms.functional_tensor" not in sys.modules:
    _ft = _types.ModuleType("torchvision.transforms.functional_tensor")
    def _rgb_to_grayscale(img, num_output_channels=1):
        import torch
        r, g, b = img.unbind(dim=-3)
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        l_img = l_img.unsqueeze(dim=-3)
        if num_output_channels == 3:
            return l_img.expand(img.shape)
        return l_img
    _ft.rgb_to_grayscale = _rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _ft


# ═══════════════════════════════════════════════════════════════
# DRIVER CONFIGURATION
# Each expression has multiple driver candidates, ranked by
# identity preservation score. Engine tries them in order.
# ═══════════════════════════════════════════════════════════════

EXPRESSION_DRIVERS = {
    "smile": {
        "description": "Natural smile",
        "multiplier_boost": 1.5,  # Boost for visible smile
        "drivers": [
            ("smile_best_1.jpg", 0.985, "Subtle frontal smile"),
            ("smile_best_2.jpg", 0.983, "Gentle smile"),
            ("smile_best_3.jpg", 0.975, "Natural smile"),
        ],
        "fallback_lp": "d30.jpg",
    },
    "open_smile": {
        "description": "Open mouth smile with teeth",
        "multiplier_boost": 1.8,  # Strong boost — user wants teeth showing
        "drivers": [
            ("open_smile_best_1.jpg", 0.976, "Open smile — toothy grin"),
            ("open_smile_best_2.jpg", 0.975, "Open smile variant"),
            ("open_smile_best_3.jpg", 0.963, "Wide smile"),
        ],
        "fallback_lp": "d12.jpg",
    },
    "surprise": {
        "description": "Surprised expression",
        "multiplier_boost": 1.5,
        "drivers": [
            ("surprise_best_1.jpg", 0.924, "Moderate surprise"),
            ("surprise_best_2.jpg", 0.920, "Surprised look"),
            ("surprise_best_3.jpg", 0.898, "Wider surprise"),
        ],
        "fallback_lp": "d19.jpg",
    },
    "sad": {
        "description": "Sad expression",
        "multiplier_boost": 1.5,
        "drivers": [
            ("sad_best_1.jpg", 0.980, "Subtle sad"),
            ("sad_best_2.jpg", 0.979, "Sad look"),
            ("sad_best_3.jpg", 0.978, "Frown"),
        ],
        "fallback_lp": "d8.jpg",
    },
    "angry": {
        "description": "Angry expression",
        "multiplier_boost": 1.5,
        "drivers": [],
        "fallback_lp": "d38.jpg",
    },
}


class ExpressionEngine:
    """Production expression editing engine with quality gate + RLHF."""

    # Quality thresholds
    IDENTITY_GATE = 0.90       # Minimum identity to show to user
    IDENTITY_EXCELLENT = 0.97  # Above this = definitely show
    MIN_EXPRESSION = 0.015     # Minimum expression change to be actually visible

    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 60)
        print("EXPRESSION ENGINE — Production + RLHF")
        print(f"  Device: {self.device}")
        print(f"  Identity gate: {self.IDENTITY_GATE}")
        print(f"  Expressions: {list(EXPRESSION_DRIVERS.keys())}")
        print("=" * 60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(DRIVERS_DIR, exist_ok=True)

        self._load_models()

        # Initialize RLHF system
        from rlhf import RLHFSystem
        self.rlhf = RLHFSystem()
        self._request_counter = 0

    def _load_models(self):
        """Load InsightFace and LivePortrait."""
        print("  Loading InsightFace...")
        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2_DIR,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        print("  Loading LivePortrait...")
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        self.lp = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig(),
        )
        print("  Models ready!")

    def _get_embedding(self, image_bgr):
        """Extract ArcFace embedding."""
        faces = self.face_analyzer.get(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return face.normed_embedding.reshape(1, -1)

    def _identity_score(self, source_emb, image_bgr):
        """Compute ArcFace cosine similarity."""
        emb = self._get_embedding(image_bgr)
        if emb is None:
            return 0.0
        return float(cosine_similarity(source_emb, emb)[0][0])

    def _run_lp(self, source_path: str, driving_path: str,
                multiplier: float = 1.0,
                use_retargeting: bool = False) -> Optional[np.ndarray]:
        """Run LivePortrait expression transfer.

        use_retargeting=False (default now): Full expression transfer from driving.
            Stronger, visible expression change. May lose some identity.
        use_retargeting=True: Only transfers eye/lip ratios.
            Very subtle expression. High identity but almost invisible change.
        """
        from src.config.argument_config import ArgumentConfig

        out_dir = os.path.join(TEMP_DIR, "lp_prod")
        os.makedirs(out_dir, exist_ok=True)

        args = ArgumentConfig()
        args.source = source_path
        args.driving = driving_path
        args.output_dir = out_dir
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "all"
        args.driving_option = "expression-friendly"
        args.driving_multiplier = multiplier
        args.source_max_dim = 1920

        # Retargeting: subtle but safe. Standard: visible but riskier.
        args.flag_eye_retargeting = use_retargeting
        args.flag_lip_retargeting = use_retargeting

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
            return result
        except Exception as e:
            return None

    def _resolve_driver_path(self, filename: str) -> Optional[str]:
        """Find driver image file."""
        # Check production drivers directory
        path = os.path.join(DRIVERS_DIR, filename)
        if os.path.exists(path):
            return path

        # Check LP driving directory
        lp_path = os.path.join(LP_DIR, "assets", "examples", "driving", filename)
        if os.path.exists(lp_path):
            return lp_path

        return None

    def _detect_issues(self, image_bgr) -> List[str]:
        """Quick pre-flight check."""
        warnings = []
        faces = self.face_analyzer.get(image_bgr)

        if not faces:
            return ["no_face"]
        if len(faces) > 1:
            warnings.append("multiple_faces")

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1

        if fw < 128 or fh < 128:
            warnings.append("low_resolution")

        lmk = getattr(face, 'landmark_2d_106', None)
        if lmk is not None and len(lmk) > 86:
            nose_x = lmk[86, 0]
            face_cx = (x1 + x2) / 2
            if abs(nose_x - face_cx) / max(fw, 1) > 0.15:
                warnings.append("side_pose")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = image_bgr.shape[:2]
        face_gray = gray[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if face_gray.size > 0:
            if cv2.Laplacian(face_gray, cv2.CV_64F).var() < 30:
                warnings.append("blurry")
            mean_bright = face_gray.mean()
            if mean_bright < 40 or mean_bright > 230:
                warnings.append("bad_lighting")

        return warnings

    def _measure_expression(self, source_bgr, result_bgr) -> float:
        """Quick expression displacement measurement."""
        src_faces = self.face_analyzer.get(source_bgr)
        res_faces = self.face_analyzer.get(result_bgr)
        if not src_faces or not res_faces:
            return 0.0

        src_f = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        res_f = max(res_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        src_lmk = getattr(src_f, 'landmark_2d_106', None)
        res_lmk = getattr(res_f, 'landmark_2d_106', None)
        if src_lmk is None or res_lmk is None:
            return 0.0

        face_size = max(src_f.bbox[2]-src_f.bbox[0], src_f.bbox[3]-src_f.bbox[1], 1)
        disp = np.linalg.norm(res_lmk - src_lmk, axis=1) / face_size
        return float(disp.mean())

    # ══════════════════════════════════════════════════════════
    # MAIN EDIT FUNCTION — Cascading Quality Gate
    # ══════════════════════════════════════════════════════════

    def edit(
        self,
        source_path: str,
        expression: str = "smile",
        intensity: float = 1.0,
        identity_threshold: float = None,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Edit expression with cascading quality gate.

        Args:
            source_path: Path to source face image
            expression: "smile", "open_smile", "surprise", "sad", "angry"
            intensity: 0.0 to 1.0 (expression strength)
            identity_threshold: Override default quality gate (0.97)
            output_name: Custom output filename

        Returns:
            {
                "success": bool,
                "output_path": str or None,
                "identity_score": float,
                "expression_change": float,
                "driver_used": str,
                "intensity_used": float,
                "attempts": int,
                "message": str,
                "time": float,
            }
        """
        t0 = time.time()
        threshold = identity_threshold or self.IDENTITY_GATE

        # Validate expression
        if expression not in EXPRESSION_DRIVERS:
            available = list(EXPRESSION_DRIVERS.keys())
            return {
                "success": False,
                "message": f"Unknown expression '{expression}'. Available: {available}",
                "identity_score": 0, "expression_change": 0,
                "driver_used": None, "intensity_used": 0,
                "attempts": 0, "time": 0,
            }

        # Load source
        source_bgr = cv2.imread(source_path)
        if source_bgr is None:
            return self._fail("Cannot read source image", t0)

        # Pre-flight checks
        issues = self._detect_issues(source_bgr)
        if "no_face" in issues:
            return self._fail("No face detected in image", t0)

        source_emb = self._get_embedding(source_bgr)
        if source_emb is None:
            return self._fail("Could not extract face embedding", t0)

        h, w = source_bgr.shape[:2]
        expr_config = EXPRESSION_DRIVERS[expression]
        attempts = 0

        # Generate request ID for RLHF tracking
        self._request_counter += 1
        request_id = f"req_{int(time.time())}_{self._request_counter}"

        # ── ASK RLHF FOR BEST STRATEGY ──
        rlhf_strategy = self.rlhf.get_best_strategy(expression, source_emb)
        rlhf_reordered = self.rlhf.get_driver_order(expression)

        # ── CASCADING FALLBACK STRATEGY ──
        # Priority: VISIBLE expression with identity preserved
        #
        # If RLHF has preferences → reorder drivers by feedback
        # Otherwise → use default order
        #
        # Level 1: Standard mode (full expression) at full intensity
        # Level 2: Standard mode at reduced intensity (0.7x, 0.5x)
        # Level 3: Retargeting mode (subtle but safe) as last resort

        candidates = []  # (path, multiplier, driver_name, description, use_retargeting)

        # LEVEL 1: Standard mode — full expression transfer (VISIBLE changes)
        boost = expr_config.get("multiplier_boost", 1.0)
        effective_intensity = intensity * boost

        # Use RLHF-reordered drivers if available
        if rlhf_reordered:
            # RLHF has learned which drivers users prefer
            for driver_file, rlhf_intensity in rlhf_reordered:
                # Skip drivers that RLHF says are bad
                if self.rlhf.should_demote_driver(expression, driver_file):
                    continue
                path = self._resolve_driver_path(driver_file)
                if path:
                    candidates.append((path, rlhf_intensity, driver_file, "RLHF preferred", False))

        # Add default drivers (if not already added by RLHF)
        rlhf_drivers = set(d for d, _ in rlhf_reordered) if rlhf_reordered else set()
        for driver_file, _, desc in expr_config["drivers"]:
            if driver_file in rlhf_drivers:
                continue  # Already added by RLHF
            if self.rlhf.should_demote_driver(expression, driver_file):
                continue  # RLHF says this driver is bad
            path = self._resolve_driver_path(driver_file)
            if path:
                candidates.append((path, effective_intensity, driver_file, desc, False))

        lp_fallback = expr_config.get("fallback_lp")
        if lp_fallback:
            path = self._resolve_driver_path(lp_fallback)
            if path:
                candidates.append((path, effective_intensity, lp_fallback, "LP original", False))

        # LEVEL 2: Standard mode at reduced intensity
        if intensity > 0.5:
            for driver_file, _, desc in expr_config["drivers"]:
                path = self._resolve_driver_path(driver_file)
                if path:
                    candidates.append((path, intensity * 0.7, driver_file, f"{desc} (70%)", False))

            for driver_file, _, desc in expr_config["drivers"]:
                path = self._resolve_driver_path(driver_file)
                if path:
                    candidates.append((path, intensity * 0.5, driver_file, f"{desc} (50%)", False))

        # LEVEL 3: Retargeting mode — subtle expression, maximum identity safety
        for driver_file, _, desc in expr_config["drivers"]:
            path = self._resolve_driver_path(driver_file)
            if path:
                candidates.append((path, intensity, driver_file, f"{desc} (retarget)", True))

        best_result = None
        best_score = 0.0
        best_expr_change = 0.0
        best_driver = None
        best_intensity = 0.0

        for drv_path, mult, drv_name, desc, retarget in candidates:
            attempts += 1

            result = self._run_lp(source_path, drv_path, multiplier=mult,
                                  use_retargeting=retarget)
            if result is None:
                continue

            if result.shape[:2] != (h, w):
                result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)

            score = self._identity_score(source_emb, result)
            expr_change = self._measure_expression(source_bgr, result)

            # Track best result: prefer VISIBLE expression over high identity
            # A result with good expression + passing identity beats
            # a result with invisible expression + perfect identity
            is_better = False
            if score >= threshold and expr_change > best_expr_change:
                is_better = True  # Better expression AND passes gate
            elif score > best_score and best_score < threshold:
                is_better = True  # Neither passes, but this is closer

            if is_better:
                best_score = score
                best_result = result
                best_driver = drv_name
                best_intensity = mult
                best_expr_change = expr_change

            # Accept first result that passes gate AND has visible expression
            if score >= threshold and expr_change >= 0.01:
                best_result = result
                best_score = score
                best_driver = drv_name
                best_intensity = mult
                best_expr_change = expr_change
                break

        # ── RECORD ATTEMPT IN RLHF ──
        result_emb = self._get_embedding(best_result) if best_result is not None else None
        self.rlhf.record_attempt(
            request_id=request_id,
            expression=expression,
            driver_used=best_driver or "none",
            intensity_used=best_intensity,
            identity_score=best_score,
            expression_change=best_expr_change,
            source_embedding=source_emb.flatten().tolist() if source_emb is not None else None,
            result_embedding=result_emb.flatten().tolist() if result_emb is not None else None,
        )

        # ── SAVE RESULT ──
        if best_result is not None and best_score >= threshold:
            if output_name is None:
                src_base = os.path.splitext(os.path.basename(source_path))[0]
                output_name = f"{src_base}_{expression}.png"

            output_path = os.path.join(OUTPUT_DIR, output_name)
            cv2.imwrite(output_path, best_result)

            rlhf_info = ""
            if rlhf_strategy.get("has_preference"):
                rlhf_info = f" | RLHF: {rlhf_strategy['reason'][:80]}"

            return {
                "success": True,
                "output_path": output_path,
                "identity_score": round(best_score, 4),
                "expression_change": round(best_expr_change, 4),
                "driver_used": best_driver,
                "intensity_used": round(best_intensity, 2),
                "attempts": attempts,
                "request_id": request_id,
                "message": f"Expression '{expression}' applied successfully{rlhf_info}",
                "time": round(time.time() - t0, 2),
                "warnings": issues,
                "rlhf_active": rlhf_strategy.get("has_preference", False),
            }

        # All attempts failed quality gate
        if best_result is not None:
            debug_name = f"REJECTED_{os.path.basename(source_path)}_{expression}.png"
            debug_path = os.path.join(OUTPUT_DIR, "rejected", debug_name)
            os.makedirs(os.path.join(OUTPUT_DIR, "rejected"), exist_ok=True)
            cv2.imwrite(debug_path, best_result)

        return {
            "success": False,
            "output_path": None,
            "identity_score": round(best_score, 4),
            "expression_change": round(best_expr_change, 4),
            "driver_used": best_driver,
            "intensity_used": round(best_intensity, 2),
            "attempts": attempts,
            "request_id": request_id,
            "message": (
                f"Could not achieve {expression} while preserving your identity "
                f"(best: {best_score:.1%}, need: {threshold:.1%}). "
                f"Try a different photo with better lighting and a frontal angle."
            ),
            "time": round(time.time() - t0, 2),
            "warnings": issues,
            "rlhf_active": rlhf_strategy.get("has_preference", False),
        }

    def _fail(self, message: str, t0: float) -> Dict:
        return {
            "success": False, "output_path": None,
            "identity_score": 0, "expression_change": 0,
            "driver_used": None, "intensity_used": 0,
            "attempts": 0, "message": message,
            "time": round(time.time() - t0, 2),
        }

    def list_expressions(self) -> Dict[str, str]:
        """Return available expressions."""
        return {k: v["description"] for k, v in EXPRESSION_DRIVERS.items()}

    def batch_edit(self, source_path: str,
                   expressions: List[str] = None,
                   intensity: float = 1.0) -> Dict[str, Dict]:
        """Edit source with multiple expressions."""
        if expressions is None:
            expressions = list(EXPRESSION_DRIVERS.keys())

        results = {}
        for expr in expressions:
            results[expr] = self.edit(source_path, expression=expr, intensity=intensity)

        return results


def setup_drivers():
    """Copy best driving images to production drivers directory."""
    os.makedirs(DRIVERS_DIR, exist_ok=True)

    cleaned_dir = os.path.join(BASE_DIR, "training_engine", "dataset", "cleaned_scraped")

    driver_map = {
        # Smile drivers (ranked by identity score)
        "smile_best_1.jpg": os.path.join(cleaned_dir, "smile", "clean_smile_0002.jpg"),
        "smile_best_2.jpg": os.path.join(cleaned_dir, "smile", "clean_smile_0003.jpg"),
        "smile_best_3.jpg": os.path.join(cleaned_dir, "smile", "clean_smile_0000.jpg"),
        # Open smile drivers
        "open_smile_best_1.jpg": os.path.join(cleaned_dir, "open_smile_drivers", "clean_opensmile_0001.jpg"),
        "open_smile_best_2.jpg": os.path.join(cleaned_dir, "open_smile_drivers", "clean_opensmile_0002.jpg"),
        "open_smile_best_3.jpg": os.path.join(cleaned_dir, "open_smile_drivers", "clean_opensmile_0003.jpg"),
        # Surprise drivers
        "surprise_best_1.jpg": os.path.join(cleaned_dir, "surprise", "clean_surprise_0001.jpg"),
        "surprise_best_2.jpg": os.path.join(cleaned_dir, "surprise", "clean_surprise_0000.jpg"),
        "surprise_best_3.jpg": os.path.join(cleaned_dir, "surprise", "clean_surprise_0002.jpg"),
        # Sad drivers
        "sad_best_1.jpg": os.path.join(cleaned_dir, "sad", "clean_sad_0001.jpg"),
        "sad_best_2.jpg": os.path.join(cleaned_dir, "sad", "clean_sad_0003.jpg"),
        "sad_best_3.jpg": os.path.join(cleaned_dir, "sad", "clean_sad_0002.jpg"),
    }

    copied = 0
    for target_name, source_path in driver_map.items():
        target_path = os.path.join(DRIVERS_DIR, target_name)
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            copied += 1
            print(f"  ✓ {target_name}")
        else:
            print(f"  ✗ {target_name} — source not found: {source_path}")

    print(f"\n  Copied {copied}/{len(driver_map)} drivers to {DRIVERS_DIR}")


if __name__ == "__main__":
    # Setup drivers first
    print("Setting up production drivers...")
    setup_drivers()

    # Quick test
    print("\nInitializing engine...")
    engine = ExpressionEngine()

    test_source = os.path.join(BASE_DIR, "MagicFace", "test_images", "ros1.jpg")
    if os.path.exists(test_source):
        print(f"\nTesting on: {test_source}")
        results = engine.batch_edit(test_source, ["smile", "open_smile", "surprise", "sad"])

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"  {'Expression':<14} {'Success':>8} {'Identity':>10} {'ExprChg':>10} {'Driver':<25} {'Time':>6}")
        print(f"  {'─'*14} {'─'*8} {'─'*10} {'─'*10} {'─'*25} {'─'*6}")
        for expr, r in results.items():
            ok = "✓" if r["success"] else "✗"
            drv = r.get("driver_used", "—") or "—"
            print(f"  {expr:<14} {ok:>8} {r['identity_score']:>10.4f} "
                  f"{r['expression_change']:>10.4f} {drv:<25} {r['time']:>5.1f}s")
        print(f"{'='*70}")
