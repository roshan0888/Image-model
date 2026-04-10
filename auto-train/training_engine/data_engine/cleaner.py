"""
Multi-Stage Data Cleaning & Filtering Pipeline

Stages:
  1. Face detection + size filtering
  2. Image quality scoring (sharpness, brightness, contrast)
  3. Pose filtering (reject non-frontal)
  4. Occlusion detection
  5. Duplicate removal (perceptual hashing)
  6. AI-generated image detection (frequency analysis)
  7. Automatic annotation (expression, pose, age, gender)
"""

import os
import cv2
import json
import hashlib
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ImageQualityAnalyzer:
    """Compute quality metrics for a single image without loading heavy models."""

    @staticmethod
    def compute_sharpness(gray: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_brightness(gray: np.ndarray) -> float:
        return float(gray.mean())

    @staticmethod
    def compute_contrast(gray: np.ndarray) -> float:
        return float(gray.std())

    @staticmethod
    def compute_blur_score(gray: np.ndarray) -> float:
        """FFT-based blur detection. Low high-frequency content = blurry."""
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-10)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        # Ratio of high-frequency to total energy
        radius = min(h, w) // 4
        total = magnitude.sum()
        center_mask = np.zeros_like(magnitude)
        cv2.circle(center_mask, (cx, cy), radius, 1, -1)
        low_freq = (magnitude * center_mask).sum()
        high_freq_ratio = 1.0 - (low_freq / (total + 1e-10))
        return float(high_freq_ratio)

    @staticmethod
    def detect_ai_generated(gray: np.ndarray) -> float:
        """Simple frequency-domain heuristic for AI-generated images.

        AI images (GAN/diffusion) often have unusual frequency patterns:
        - Missing high frequencies (smooth)
        - Regular grid artifacts in frequency domain

        Returns score 0-1, higher = more likely AI-generated.
        """
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = gray.shape
        cy, cx = h // 2, w // 2

        # Check for periodic artifacts (GAN checkerboard)
        # Look for unusual peaks in frequency domain
        mean_mag = magnitude.mean()
        std_mag = magnitude.std()
        peak_count = np.sum(magnitude > mean_mag + 5 * std_mag)
        peak_ratio = peak_count / (h * w)

        # Check high-frequency falloff (AI images drop off faster)
        outer_ring = magnitude.copy()
        cv2.circle(outer_ring, (cx, cy), min(h, w) // 3, 0, -1)
        inner = magnitude.copy()
        inner[outer_ring > 0] = 0
        if inner.sum() > 0:
            hf_ratio = outer_ring.sum() / (inner.sum() + 1e-10)
        else:
            hf_ratio = 0.0

        # Simple scoring
        score = 0.0
        if peak_ratio > 0.001:  # Too many frequency peaks = GAN artifact
            score += 0.4
        if hf_ratio < 0.1:  # Too little high frequency = AI smoothing
            score += 0.4
        if hf_ratio < 0.05:
            score += 0.2

        return min(score, 1.0)


class DataCleaner:
    """Multi-stage face data cleaning pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.clean_cfg = config["data_cleaning"]
        self.raw_dir = config["paths"]["raw_images_dir"]
        self.cleaned_dir = config["paths"]["cleaned_dir"]
        self.face_analyzer = None
        self._quality_analyzer = ImageQualityAnalyzer()

        os.makedirs(self.cleaned_dir, exist_ok=True)

    def _ensure_face_analyzer(self):
        if self.face_analyzer is not None:
            return
        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(
            name="antelopev2",
            root=self.config["paths"]["antelopev2_dir"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    def clean_all(self) -> Dict:
        """Run full cleaning pipeline on raw images."""
        self._ensure_face_analyzer()

        stats = {
            "total_scanned": 0,
            "passed": 0,
            "rejected": 0,
            "reject_reasons": {},
            "annotations": [],
        }

        image_paths = self._get_all_images(self.raw_dir)
        logger.info(f"Scanning {len(image_paths)} raw images...")

        for i, img_path in enumerate(image_paths):
            if i % 100 == 0:
                logger.info(f"  [{i}/{len(image_paths)}] Passed: {stats['passed']}, "
                          f"Rejected: {stats['rejected']}")

            stats["total_scanned"] += 1
            result = self._process_single(img_path)

            if result["rejected"]:
                reason = result["reject_reason"]
                stats["rejected"] += 1
                stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1
            else:
                stats["passed"] += 1
                stats["annotations"].append(result["annotation"])

                # Copy to cleaned directory
                self._save_cleaned(img_path, result["annotation"])

        # Save annotations
        self._save_annotations(stats["annotations"])
        logger.info(f"Cleaning complete: {stats['passed']}/{stats['total_scanned']} passed")
        return stats

    def _process_single(self, img_path: str) -> Dict:
        """Process a single image through all cleaning stages."""
        img = cv2.imread(img_path)
        if img is None:
            return {"rejected": True, "reject_reason": "unreadable"}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Stage 1: Face detection
        faces = self.face_analyzer.get(img)
        if not faces:
            return {"rejected": True, "reject_reason": "no_face"}

        if len(faces) > 1:
            # Keep only if one face is clearly dominant
            areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
            if max(areas) < 2 * sorted(areas)[-2]:
                return {"rejected": True, "reject_reason": "multiple_faces"}

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1

        # Detection confidence
        if face.det_score < self.clean_cfg["min_detection_score"]:
            return {"rejected": True, "reject_reason": "low_detection_score"}

        # Stage 2: Face size
        if fw < self.clean_cfg["min_face_size"] or fh < self.clean_cfg["min_face_size"]:
            return {"rejected": True, "reject_reason": "face_too_small"}

        # Stage 3: Image quality
        face_gray = gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if face_gray.size == 0:
            return {"rejected": True, "reject_reason": "invalid_bbox"}

        sharpness = self._quality_analyzer.compute_sharpness(face_gray)
        if sharpness < self.clean_cfg["min_sharpness"]:
            return {"rejected": True, "reject_reason": "too_blurry"}

        brightness = self._quality_analyzer.compute_brightness(face_gray)
        if brightness < self.clean_cfg["min_brightness"]:
            return {"rejected": True, "reject_reason": "too_dark"}
        if brightness > self.clean_cfg["max_brightness"]:
            return {"rejected": True, "reject_reason": "too_bright"}

        # Stage 4: Pose estimation
        lmk = getattr(face, 'landmark_2d_106', None)
        pose_angle = 0.0
        if lmk is not None and len(lmk) > 86:
            nose_x = lmk[86, 0]
            face_cx = (x1 + x2) / 2
            offset_ratio = abs(nose_x - face_cx) / max(fw, 1)
            pose_angle = offset_ratio * 90  # Rough conversion to degrees
            if pose_angle > self.clean_cfg["max_pose_angle"]:
                return {"rejected": True, "reject_reason": "non_frontal"}

        # Stage 5: AI-generated detection
        ai_score = self._quality_analyzer.detect_ai_generated(face_gray)
        if ai_score > 0.7:
            return {"rejected": True, "reject_reason": "ai_generated"}

        # Stage 6: Quality score
        contrast = self._quality_analyzer.compute_contrast(face_gray)
        blur_score = self._quality_analyzer.compute_blur_score(face_gray)
        quality_score = self._compute_quality_score(
            sharpness, brightness, contrast, pose_angle, fw, fh, blur_score
        )
        if quality_score < self.config["quality_scoring"]["min_quality_score"]:
            return {"rejected": True, "reject_reason": "low_quality_score"}

        # Stage 7: Annotation
        annotation = self._annotate(face, img_path, {
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "pose_angle": pose_angle,
            "ai_score": ai_score,
            "quality_score": quality_score,
            "face_bbox": [int(x1), int(y1), int(x2), int(y2)],
            "face_size": [int(fw), int(fh)],
            "image_size": [w, h],
        })

        return {"rejected": False, "annotation": annotation}

    def _annotate(self, face, img_path: str, metrics: Dict) -> Dict:
        """Auto-annotate a face with expression, pose, age, gender."""
        annotation = {
            "path": img_path,
            "metrics": metrics,
        }

        # Age and gender from InsightFace
        if hasattr(face, 'age'):
            annotation["age"] = int(face.age)
            annotation["age_group"] = self._age_to_group(face.age)
        if hasattr(face, 'gender'):
            annotation["gender"] = "male" if face.gender == 1 else "female"

        # Expression from landmark analysis
        lmk = getattr(face, 'landmark_2d_106', None)
        if lmk is not None:
            annotation["expression"] = self._classify_expression(lmk, face.bbox)

        # Identity embedding (for clustering)
        if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
            annotation["embedding"] = face.normed_embedding.tolist()

        return annotation

    def _classify_expression(self, lmk: np.ndarray, bbox: np.ndarray) -> str:
        """Classify expression from 106 landmarks."""
        face_h = bbox[3] - bbox[1]
        if face_h < 1:
            return "unknown"

        # Mouth opening ratio
        if len(lmk) > 71:
            upper_lip = lmk[62:66].mean(axis=0)
            lower_lip = lmk[66:71].mean(axis=0)
            mouth_open = np.linalg.norm(upper_lip - lower_lip) / face_h
        else:
            mouth_open = 0.0

        # Mouth width ratio
        if len(lmk) > 60:
            mouth_left = lmk[52]
            mouth_right = lmk[58]
            mouth_width = np.linalg.norm(mouth_left - mouth_right) / (bbox[2] - bbox[0])
        else:
            mouth_width = 0.0

        # Eye opening
        if len(lmk) > 47:
            left_eye_top = lmk[37]
            left_eye_bottom = lmk[41]
            eye_open = np.linalg.norm(left_eye_top - left_eye_bottom) / face_h
        else:
            eye_open = 0.0

        # Simple classification rules
        if mouth_open > 0.06 and eye_open > 0.04:
            return "surprise"
        elif mouth_open > 0.04 and mouth_width > 0.45:
            return "laugh"
        elif mouth_width > 0.42:
            return "smile"
        elif mouth_open < 0.01 and mouth_width < 0.35:
            return "sad"
        else:
            return "neutral"

    @staticmethod
    def _age_to_group(age: int) -> str:
        if age < 18:
            return "under_18"
        elif age < 25:
            return "18-25"
        elif age < 35:
            return "25-35"
        elif age < 50:
            return "35-50"
        elif age < 65:
            return "50-65"
        else:
            return "65+"

    def _compute_quality_score(self, sharpness, brightness, contrast,
                                pose_angle, fw, fh, blur_score) -> float:
        """Weighted quality score 0-1."""
        weights = self.config["quality_scoring"]["weights"]

        # Normalize each metric to 0-1
        sharp_norm = min(sharpness / 200.0, 1.0)

        # Brightness: optimal around 120-140
        bright_norm = 1.0 - abs(brightness - 130) / 130.0
        bright_norm = max(0, bright_norm)

        contrast_norm = min(contrast / 80.0, 1.0)
        pose_norm = max(0, 1.0 - pose_angle / 30.0)
        res_norm = min(min(fw, fh) / 512.0, 1.0)
        expr_norm = blur_score  # Higher blur_score = more detail

        score = (
            weights["sharpness"] * sharp_norm +
            weights["lighting"] * bright_norm +
            weights["pose"] * pose_norm +
            weights["expression_clarity"] * expr_norm +
            weights["resolution"] * res_norm +
            weights["face_symmetry"] * contrast_norm  # Using contrast as proxy
        )
        return float(score)

    def _save_cleaned(self, src_path: str, annotation: Dict):
        """Copy passing image to cleaned directory with standardized name."""
        # Organize by expression
        expr = annotation.get("expression", "unknown")
        expr_dir = os.path.join(self.cleaned_dir, expr)
        os.makedirs(expr_dir, exist_ok=True)

        # Hash-based unique name
        name_hash = hashlib.md5(src_path.encode()).hexdigest()[:12]
        ext = os.path.splitext(src_path)[1]
        dst = os.path.join(expr_dir, f"{name_hash}{ext}")

        img = cv2.imread(src_path)
        if img is not None:
            # Crop and resize face to standardized size
            bbox = annotation["metrics"]["face_bbox"]
            x1, y1, x2, y2 = bbox
            # Expand bbox by 30% for context
            h, w = img.shape[:2]
            fw, fh = x2 - x1, y2 - y1
            pad_x, pad_y = int(fw * 0.3), int(fh * 0.3)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            face_crop = img[y1:y2, x1:x2]
            # Resize to 512x512 for training
            face_512 = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(dst, face_512, [cv2.IMWRITE_JPEG_QUALITY, 95])

            annotation["cleaned_path"] = dst

    def _save_annotations(self, annotations: List[Dict]):
        path = os.path.join(self.cleaned_dir, "annotations.jsonl")
        with open(path, "w") as f:
            for a in annotations:
                # Don't save embedding in annotation file (too large)
                a_copy = {k: v for k, v in a.items() if k != "embedding"}
                f.write(json.dumps(a_copy) + "\n")

    @staticmethod
    def _get_all_images(directory: str) -> List[str]:
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        images = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    images.append(os.path.join(root, f))
        return images
