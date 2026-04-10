"""
Master Photoshoot Pipeline
===========================
Orchestrates all 4 pillars:
  1. Analysis       — face detect, pose measure, quality check
  2. Pose           — correct to frontal if needed
  3. Expression     — apply photogenic smile via LivePortrait+LoRA
  4. Background     — remove + replace with studio/custom/AI background

Single entry point for production use.

Usage:
    from photoshoot.pipeline.photoshoot_pipeline import PhotoshootPipeline

    pipeline = PhotoshootPipeline()

    result = pipeline.process(
        source_path   = "user_photo.jpg",
        expression    = "smile",          # smile | open_smile | neutral
        background    = "studio_white",   # preset | image path | text prompt
        target_pose   = "frontal",        # frontal | keep | custom
        output_path   = "output.jpg",
    )
    print(result["identity_score"])   # e.g. 0.9921
    print(result["stages_applied"])   # ["expression", "background"]
"""

import os, sys, cv2, time, logging
import numpy as np
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


class PhotoshootPipeline:

    IDENTITY_GATE = 0.97    # minimum ArcFace score to pass each stage

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._expression_pipeline = None
        self._background_pipeline = None
        self._pose_pipeline       = None
        self._face_app            = None
        log.info("PhotoshootPipeline initialized")

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC
    # ═══════════════════════════════════════════════════════════════════════════

    def process(
        self,
        source_path: str,
        expression: str = "smile",
        background: str = "studio_white",
        target_pose: str = "frontal",
        output_path: Optional[str] = None,
        output_dir: str = "output/photoshoot",
    ) -> dict:
        """
        Full pipeline: pose → expression → background.

        Returns dict:
            image           np.ndarray final result
            output_path     str
            identity_score  float (ArcFace vs source)
            stages_applied  list of stages that ran
            stage_scores    dict of identity score at each stage
            total_time      float seconds
            quality_passed  bool (>= IDENTITY_GATE throughout)
        """
        t0 = time.time()
        os.makedirs(output_dir, exist_ok=True)

        log.info(f"\n{'='*55}")
        log.info(f"PHOTOSHOOT PIPELINE")
        log.info(f"  source     : {source_path}")
        log.info(f"  expression : {expression}")
        log.info(f"  background : {background}")
        log.info(f"  pose       : {target_pose}")
        log.info(f"{'='*55}")

        # ── Load source ────────────────────────────────────────────────────────
        img = cv2.imread(source_path)
        if img is None:
            raise ValueError(f"Cannot read: {source_path}")

        current_img  = img.copy()
        stages       = []
        stage_scores = {}
        quality_ok   = True

        # ── Stage 1: ANALYSIS ──────────────────────────────────────────────────
        analysis = self._analyze(current_img, source_path)
        if not analysis["face_detected"]:
            return self._fail("No face detected in source image")

        baseline_score = analysis["baseline_score"]
        log.info(f"  Analysis: pose=({analysis['yaw']:.1f}°, {analysis['pitch']:.1f}°) "
                 f"identity_baseline={baseline_score:.4f}")

        # ── Stage 2: POSE CORRECTION ───────────────────────────────────────────
        if target_pose == "frontal" and analysis["needs_pose_correction"]:
            log.info("  Running pose correction...")
            try:
                pose_result = self._get_pose_pipeline().process(
                    source_path=source_path,
                    target_yaw=0.0,
                    target_pitch=0.0,
                )
                if pose_result["corrected"]:
                    score = pose_result.get("identity_score", 0)
                    if score >= self.IDENTITY_GATE:
                        current_img = pose_result["image"]
                        stages.append("pose")
                        stage_scores["pose"] = score
                        log.info(f"  Pose corrected: identity={score:.4f} ✓")
                    else:
                        log.warning(f"  Pose correction dropped identity to {score:.4f} — skipped")
            except Exception as e:
                log.warning(f"  Pose correction failed: {e} — continuing without")

        # ── Stage 3: EXPRESSION ────────────────────────────────────────────────
        if expression != "neutral":
            log.info(f"  Running expression: {expression}...")
            expr_path = str(Path(output_dir) / "stage_expression.jpg")
            cv2.imwrite(expr_path, current_img)
            try:
                expr_result = self._get_expression_pipeline().process(expr_path, expression)
                expr_img    = expr_result.get("image")
                expr_score  = expr_result.get("identity_score", 0)

                if expr_img is not None and expr_score >= self.IDENTITY_GATE:
                    current_img = expr_img if isinstance(expr_img, np.ndarray) \
                                  else np.array(expr_img)
                    stages.append("expression")
                    stage_scores["expression"] = expr_score
                    log.info(f"  Expression '{expression}': identity={expr_score:.4f} ✓")
                else:
                    log.warning(f"  Expression identity {expr_score:.4f} < gate {self.IDENTITY_GATE} — skipped")
                    quality_ok = False
            except Exception as e:
                log.warning(f"  Expression failed: {e} — continuing without")

        # ── Stage 4: BACKGROUND ────────────────────────────────────────────────
        if background != "keep":
            log.info(f"  Running background: {background}...")
            bg_path = str(Path(output_dir) / "stage_after_expression.jpg")
            cv2.imwrite(bg_path, current_img)
            try:
                bg_result = self._get_background_pipeline().process(
                    source_path=bg_path,
                    background=background,
                )
                bg_img    = bg_result["image"]
                bg_score  = self._measure_identity(img, bg_img)

                if bg_score >= self.IDENTITY_GATE:
                    current_img = bg_img
                    stages.append("background")
                    stage_scores["background"] = bg_score
                    log.info(f"  Background '{background}': identity={bg_score:.4f} ✓")
                else:
                    log.warning(f"  Background dropped identity to {bg_score:.4f} — skipped")
            except Exception as e:
                log.warning(f"  Background failed: {e} — continuing without")

        # ── Save final output ──────────────────────────────────────────────────
        if output_path is None:
            stem = Path(source_path).stem
            output_path = str(Path(output_dir) / f"{stem}_{expression}_{background.split('/')[-1]}.jpg")

        cv2.imwrite(output_path, current_img)

        # Final identity measurement
        final_score = self._measure_identity(img, current_img)
        total_time  = time.time() - t0

        log.info(f"\n{'─'*55}")
        log.info(f"DONE in {total_time:.1f}s")
        log.info(f"  Stages applied : {stages}")
        log.info(f"  Stage scores   : {stage_scores}")
        log.info(f"  Final identity : {final_score:.4f}")
        log.info(f"  Quality gate   : {'PASS ✓' if final_score >= self.IDENTITY_GATE else 'FAIL ✗'}")
        log.info(f"  Output         : {output_path}")
        log.info(f"{'─'*55}\n")

        return {
            "image":          current_img,
            "output_path":    output_path,
            "identity_score": final_score,
            "stage_scores":   stage_scores,
            "stages_applied": stages,
            "total_time":     total_time,
            "quality_passed": final_score >= self.IDENTITY_GATE,
            "analysis":       analysis,
        }

    def batch_process(self, source_paths: list, **kwargs) -> list:
        """Process multiple photos with same settings."""
        results = []
        for i, src in enumerate(source_paths):
            log.info(f"\n[{i+1}/{len(source_paths)}] {Path(src).name}")
            try:
                r = self.process(src, **kwargs)
                results.append(r)
            except Exception as e:
                log.error(f"Failed {src}: {e}")
                results.append({"error": str(e), "source": src})
        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE
    # ═══════════════════════════════════════════════════════════════════════════

    def _analyze(self, img: np.ndarray, source_path: str) -> dict:
        app = self._get_face_app()
        faces = app.get(img)
        if not faces:
            return {"face_detected": False, "needs_pose_correction": False,
                    "baseline_score": 0.0, "yaw": 0, "pitch": 0}

        face  = faces[0]
        pose  = face.pose if hasattr(face, "pose") and face.pose is not None else [0, 0, 0]
        pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])
        needs_pose = abs(yaw) > 15 or abs(pitch) > 15

        return {
            "face_detected":       True,
            "pitch":               pitch,
            "yaw":                 yaw,
            "roll":                roll,
            "needs_pose_correction": needs_pose,
            "baseline_score":      1.0,
            "det_score":           float(face.det_score),
        }

    def _measure_identity(self, source: np.ndarray, result: np.ndarray) -> float:
        try:
            app = self._get_face_app()
            f1  = app.get(source)
            f2  = app.get(result)
            if not f1 or not f2:
                return 0.0
            return float(np.dot(f1[0].normed_embedding, f2[0].normed_embedding))
        except Exception:
            return 0.0

    def _get_face_app(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(
                name="antelopev2",
                root=str(ROOT / "MagicFace/third_party_files")
            )
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_app

    def _get_expression_pipeline(self):
        if self._expression_pipeline is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "natural_pipeline", ROOT / "natural_pipeline.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._expression_pipeline = mod.NaturalExpressionPipeline()
        return self._expression_pipeline

    def _get_background_pipeline(self):
        if self._background_pipeline is None:
            from photoshoot.background.background_pipeline import BackgroundPipeline
            self._background_pipeline = BackgroundPipeline(device=self.device)
        return self._background_pipeline

    def _get_pose_pipeline(self):
        if self._pose_pipeline is None:
            from photoshoot.pose.pose_pipeline import PosePipeline
            self._pose_pipeline = PosePipeline()
        return self._pose_pipeline

    @staticmethod
    def _fail(reason: str) -> dict:
        log.error(f"Pipeline failed: {reason}")
        return {
            "image": None, "output_path": None,
            "identity_score": 0.0, "quality_passed": False,
            "stages_applied": [], "error": reason,
        }
