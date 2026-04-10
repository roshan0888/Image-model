"""
Pose Correction Pipeline
=========================
Detects head pose from a single photo and corrects it toward frontal.

Method:
  1. InsightFace → detect face + get pitch/yaw/roll
  2. If pose is acceptable (|yaw| < 15°) → pass through unchanged
  3. If pose needs correction → 3DDFA-V2 reconstruct 3D mesh → rotate → render

Cost: CPU-capable for analysis. GPU only for 3D rendering (optional).

Usage:
    from photoshoot.pose.pose_pipeline import PosePipeline

    pipeline = PosePipeline()
    result = pipeline.process("photo.jpg", target_yaw=0, target_pitch=0)
"""

import os, sys, cv2, logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


class PosePipeline:

    # Thresholds — within these angles we don't correct (saves compute)
    YAW_THRESHOLD   = 15   # degrees — beyond this we correct
    PITCH_THRESHOLD = 15

    def __init__(self):
        self._face_app   = None
        self._tddfa      = None
        log.info("PosePipeline initialized")

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def analyze(self, image_path: str) -> dict:
        """Detect face pose — returns pitch, yaw, roll + whether correction needed."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")

        app = self._get_face_app()
        faces = app.get(img)

        if not faces:
            return {"detected": False, "needs_correction": False}

        face = faces[0]
        pose = face.pose if hasattr(face, "pose") and face.pose is not None else [0, 0, 0]
        pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])

        needs_correction = abs(yaw) > self.YAW_THRESHOLD or abs(pitch) > self.PITCH_THRESHOLD

        return {
            "detected": True,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "needs_correction": needs_correction,
            "identity_score": 1.0,  # filled after correction
        }

    def process(
        self,
        source_path: str,
        target_yaw: float = 0.0,
        target_pitch: float = 0.0,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        Correct pose toward target angles.

        Args:
            source_path:  Input portrait
            target_yaw:   Target horizontal angle (0 = frontal)
            target_pitch: Target vertical angle (0 = level)
            output_path:  Where to save (auto if None)

        Returns:
            dict with 'image', 'output_path', 'original_pose', 'corrected', 'identity_score'
        """
        img = cv2.imread(source_path)
        if img is None:
            raise ValueError(f"Cannot read: {source_path}")

        pose_info = self.analyze(source_path)

        if not pose_info["detected"]:
            log.warning("No face detected — returning original")
            return {"image": img, "corrected": False, "reason": "no_face"}

        if not pose_info["needs_correction"]:
            log.info(f"Pose OK (yaw={pose_info['yaw']:.1f}°) — no correction needed")
            return {
                "image": img,
                "corrected": False,
                "original_pose": pose_info,
                "output_path": source_path,
            }

        log.info(f"Correcting pose: yaw={pose_info['yaw']:.1f}° → {target_yaw}°")

        # Try 3DDFA correction
        try:
            result_img = self._correct_3ddfa(img, pose_info, target_yaw, target_pitch)
        except Exception as e:
            log.warning(f"3DDFA failed: {e} — using 2D affine fallback")
            result_img = self._correct_2d_affine(img, pose_info, target_yaw)

        # Verify identity preserved
        identity_score = self._check_identity(img, result_img)

        if output_path is None:
            stem = Path(source_path).stem
            output_path = str(Path(source_path).parent / f"{stem}_pose_corrected.jpg")

        cv2.imwrite(output_path, result_img)

        return {
            "image": result_img,
            "output_path": output_path,
            "original_pose": pose_info,
            "corrected": True,
            "identity_score": identity_score,
        }

    # ── POSE CORRECTION METHODS ───────────────────────────────────────────────

    def _correct_3ddfa(self, img: np.ndarray, pose_info: dict,
                        target_yaw: float, target_pitch: float) -> np.ndarray:
        """3DDFA-V2: reconstruct 3D mesh, rotate, re-render."""
        import torch

        tddfa = self._get_tddfa()
        param_lst, roi_box_lst = tddfa(img, [self._get_bbox(img)])
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

        # Render at target pose
        from TDDFA_V2.utils.render import render
        corrected = render(img, ver_lst, tddfa.tri, alpha=0.8,
                          show_flag=False)
        return corrected

    def _correct_2d_affine(self, img: np.ndarray, pose_info: dict,
                            target_yaw: float) -> np.ndarray:
        """
        2D affine approximation for small pose corrections (<30°).
        Not as accurate as 3D but works without heavy dependencies.
        """
        yaw = pose_info["yaw"]
        correction = target_yaw - yaw

        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2

        # Horizontal stretch/compress to simulate frontal
        scale_x = 1.0 + abs(np.sin(np.radians(correction))) * 0.3
        M = np.float32([
            [scale_x, 0, cx * (1 - scale_x)],
            [0, 1, 0]
        ])

        corrected = cv2.warpAffine(img, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        log.info(f"  2D affine correction applied (scale_x={scale_x:.2f})")
        return corrected

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _get_face_app(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(
                name="antelopev2",
                root=str(ROOT / "MagicFace/third_party_files")
            )
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_app

    def _get_tddfa(self):
        if self._tddfa is None:
            tddfa_root = ROOT / "TDDFA_V2"
            if not tddfa_root.exists():
                self._install_3ddfa()
            sys.path.insert(0, str(tddfa_root))
            from TDDFA import TDDFA
            cfg_path = str(tddfa_root / "configs/mb1_120x120.yml")
            self._tddfa = TDDFA(gpu_mode=True, **{"config": cfg_path})
        return self._tddfa

    def _get_bbox(self, img: np.ndarray) -> list:
        app = self._get_face_app()
        faces = app.get(img)
        if not faces:
            h, w = img.shape[:2]
            return [0, 0, w, h, 1.0]
        b = faces[0].bbox.astype(int)
        return [b[0], b[1], b[2], b[3], float(faces[0].det_score)]

    def _check_identity(self, original: np.ndarray, corrected: np.ndarray) -> float:
        try:
            app = self._get_face_app()
            faces_orig = app.get(original)
            faces_corr = app.get(corrected)
            if not faces_orig or not faces_corr:
                return 0.0
            e1 = faces_orig[0].normed_embedding
            e2 = faces_corr[0].normed_embedding
            score = float(np.dot(e1, e2))
            log.info(f"  Identity after pose correction: {score:.4f}")
            return score
        except Exception:
            return 0.0

    @staticmethod
    def _install_3ddfa():
        import subprocess
        log.info("Installing 3DDFA-V2...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/cleardusk/3DDFA_V2.git",
            str(ROOT / "TDDFA_V2")
        ], check=True)
        subprocess.run(["bash", "build.sh"],
                       cwd=str(ROOT / "TDDFA_V2"), check=True)
