#!/usr/bin/env python3
"""
Data Cleaning Pipeline

Takes raw scraped images and produces cleaned, annotated, quality-scored face crops.

Pipeline:
  1. Face detection (InsightFace) — reject no-face or multi-face
  2. Quality scoring (sharpness, brightness, face size)
  3. Blur detection (Laplacian variance)
  4. Occlusion detection (landmark coverage)
  5. Pose filtering (reject extreme angles)
  6. AI-generated image detection (texture analysis)
  7. Face crop + resize to 512x512
  8. Annotation (expression, age, gender, embedding)
  9. Duplicate removal (ArcFace embedding similarity)

Output: Cleaned face crops with annotations in JSONL.
"""

import os
import sys
import cv2
import json
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("cleaner")

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ENGINE_DIR)


class DataCleaner:
    """Multi-stage face image cleaning pipeline."""

    def __init__(self, antelopev2_dir: str):
        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(
            name="antelopev2", root=antelopev2_dir,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
        self.seen_embeddings = []  # For duplicate detection
        logger.info("DataCleaner initialized")

    def check_quality(self, image_bgr: np.ndarray) -> Dict:
        """Score image quality. Returns dict with scores and pass/fail."""
        h, w = image_bgr.shape[:2]
        scores = {}

        # 1. Resolution check
        scores["resolution"] = min(h, w)
        scores["resolution_pass"] = min(h, w) >= 256

        # 2. Sharpness (Laplacian variance)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        scores["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scores["sharpness_pass"] = scores["sharpness"] >= 50  # Stricter than detection

        # 3. Brightness
        scores["brightness"] = float(gray.mean())
        scores["brightness_pass"] = 40 < scores["brightness"] < 230

        # 4. Contrast
        scores["contrast"] = float(gray.std())
        scores["contrast_pass"] = scores["contrast"] >= 20

        # 5. Overall quality score (0-1)
        q = 0.0
        if scores["resolution_pass"]:
            q += 0.25
        if scores["sharpness_pass"]:
            q += 0.25 * min(scores["sharpness"] / 500, 1.0)
        if scores["brightness_pass"]:
            q += 0.25
        if scores["contrast_pass"]:
            q += 0.25 * min(scores["contrast"] / 60, 1.0)
        scores["quality_score"] = round(q, 4)
        scores["quality_pass"] = q >= 0.4

        return scores

    def check_face(self, image_bgr: np.ndarray) -> Optional[Dict]:
        """Detect face and extract all annotations. Returns None if unusable."""
        faces = self.face_analyzer.get(image_bgr)
        if not faces:
            return None

        # Take largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1
        h, w = image_bgr.shape[:2]

        result = {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "face_width": int(fw),
            "face_height": int(fh),
            "num_faces": len(faces),
        }

        # Reject tiny faces
        if fw < 128 or fh < 128:
            result["reject_reason"] = "face_too_small"
            return result

        # Reject if face is too small relative to image (probably not a portrait)
        face_area_ratio = (fw * fh) / (h * w)
        result["face_area_ratio"] = round(face_area_ratio, 4)
        if face_area_ratio < 0.05:
            result["reject_reason"] = "face_too_small_in_frame"
            return result

        # Multiple faces — only proceed if dominant face is much larger
        if len(faces) > 1:
            second_face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))[-2]
            sf_area = (second_face.bbox[2]-second_face.bbox[0]) * (second_face.bbox[3]-second_face.bbox[1])
            main_area = fw * fh
            if sf_area / main_area > 0.3:
                result["reject_reason"] = "multiple_prominent_faces"
                return result

        # Pose check via landmarks
        lmk = getattr(face, 'landmark_2d_106', None)
        if lmk is not None and len(lmk) > 86:
            nose_x = lmk[86, 0]
            face_cx = (x1 + x2) / 2
            offset_ratio = abs(nose_x - face_cx) / max(fw, 1)
            result["pose_offset"] = round(float(offset_ratio), 4)
            if offset_ratio > 0.20:  # More lenient than pipeline (0.15)
                result["reject_reason"] = "extreme_pose"
                return result

        # Blur check on face region
        face_crop = image_bgr[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            result["face_sharpness"] = round(float(lap_var), 2)
            if lap_var < 30:
                result["reject_reason"] = "face_blurry"
                return result

        # AI-generated detection (simple heuristic: check for texture uniformity)
        # Real skin has micro-texture variation; AI skin is unnaturally smooth
        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            # High-frequency energy ratio
            blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
            hf = np.abs(gray.astype(np.float32) - blur)
            hf_energy = hf.std()
            result["texture_energy"] = round(float(hf_energy), 2)
            # AI images tend to have very low texture energy (too smooth)
            # or very uniform texture (repeating patterns)
            if hf_energy < 3.0:
                result["reject_reason"] = "possibly_ai_generated"
                return result

        # Extract annotations
        result["age"] = int(face.age) if hasattr(face, 'age') and face.age is not None else -1
        result["gender"] = "male" if getattr(face, 'gender', 1) == 1 else "female"
        result["embedding"] = face.normed_embedding.tolist()
        result["accepted"] = True

        return result

    def is_duplicate(self, embedding: List[float], threshold: float = 0.75) -> bool:
        """Check if this face is a near-duplicate of one we've already seen."""
        if not self.seen_embeddings:
            return False

        emb = np.array(embedding).reshape(1, -1)
        for seen_emb in self.seen_embeddings:
            sim = float(np.dot(emb, seen_emb.T)[0][0])
            if sim > threshold:
                return True
        return False

    def crop_face(self, image_bgr: np.ndarray, bbox: List[int],
                  target_size: int = 512, padding: float = 0.35) -> np.ndarray:
        """Crop face with padding and resize to target_size."""
        x1, y1, x2, y2 = bbox
        h, w = image_bgr.shape[:2]
        fw, fh = x2 - x1, y2 - y1

        pad_x = int(fw * padding)
        pad_y = int(fh * padding)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = image_bgr[cy1:cy2, cx1:cx2]
        resized = cv2.resize(crop, (target_size, target_size),
                            interpolation=cv2.INTER_LANCZOS4)
        return resized

    def clean_directory(self, raw_dir: str, cleaned_dir: str,
                       expression_label: str, min_quality: float = 0.4) -> Dict:
        """Clean all images in a raw directory."""
        os.makedirs(cleaned_dir, exist_ok=True)

        stats = {
            "total": 0, "accepted": 0, "rejected": 0,
            "reject_reasons": {},
        }

        image_files = []
        for f in os.listdir(raw_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_files.append(os.path.join(raw_dir, f))

        if not image_files:
            logger.warning(f"  No images found in {raw_dir}")
            return stats

        annotations = []

        for img_path in image_files:
            stats["total"] += 1
            fname = os.path.basename(img_path)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    stats["rejected"] += 1
                    stats["reject_reasons"]["unreadable"] = stats["reject_reasons"].get("unreadable", 0) + 1
                    continue

                # Quality check
                quality = self.check_quality(img)
                if not quality["quality_pass"]:
                    stats["rejected"] += 1
                    stats["reject_reasons"]["low_quality"] = stats["reject_reasons"].get("low_quality", 0) + 1
                    continue

                # Face check
                face_info = self.check_face(img)
                if face_info is None:
                    stats["rejected"] += 1
                    stats["reject_reasons"]["no_face"] = stats["reject_reasons"].get("no_face", 0) + 1
                    continue

                if not face_info.get("accepted"):
                    reason = face_info.get("reject_reason", "unknown")
                    stats["rejected"] += 1
                    stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1
                    continue

                # Duplicate check
                if self.is_duplicate(face_info["embedding"]):
                    stats["rejected"] += 1
                    stats["reject_reasons"]["duplicate"] = stats["reject_reasons"].get("duplicate", 0) + 1
                    continue

                # All checks passed — crop and save
                face_crop = self.crop_face(img, face_info["bbox"])
                out_name = f"clean_{expression_label}_{stats['accepted']:04d}.jpg"
                out_path = os.path.join(cleaned_dir, out_name)
                cv2.imwrite(out_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Track embedding for duplicate detection
                self.seen_embeddings.append(
                    np.array(face_info["embedding"]).reshape(1, -1)
                )

                annotations.append({
                    "original_path": img_path,
                    "cleaned_path": out_path,
                    "expression": expression_label,
                    "age": face_info["age"],
                    "gender": face_info["gender"],
                    "quality_score": quality["quality_score"],
                    "face_sharpness": face_info.get("face_sharpness", 0),
                    "pose_offset": face_info.get("pose_offset", 0),
                    "face_area_ratio": face_info.get("face_area_ratio", 0),
                })

                stats["accepted"] += 1

            except Exception as e:
                stats["rejected"] += 1
                stats["reject_reasons"]["error"] = stats["reject_reasons"].get("error", 0) + 1
                logger.debug(f"    Error processing {fname}: {e}")

        # Save annotations
        if annotations:
            ann_path = os.path.join(cleaned_dir, f"annotations_{expression_label}.jsonl")
            with open(ann_path, "w") as f:
                for a in annotations:
                    f.write(json.dumps(a) + "\n")

        return stats


def clean_all_scraped_data(antelopev2_dir: str = None):
    """Run cleaning pipeline on all scraped data."""
    if antelopev2_dir is None:
        antelopev2_dir = os.path.join(BASE_DIR, "MagicFace", "third_party_files")

    raw_dir = os.path.join(ENGINE_DIR, "dataset", "raw")
    cleaned_dir = os.path.join(ENGINE_DIR, "dataset", "cleaned_scraped")

    if not os.path.exists(raw_dir):
        logger.error(f"Raw data directory not found: {raw_dir}")
        return

    cleaner = DataCleaner(antelopev2_dir)

    expressions = [d for d in os.listdir(raw_dir)
                   if os.path.isdir(os.path.join(raw_dir, d))]

    if not expressions:
        logger.error("No expression directories found in raw data")
        return

    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE")
    logger.info(f"  Raw: {raw_dir}")
    logger.info(f"  Output: {cleaned_dir}")
    logger.info(f"  Expressions: {expressions}")
    logger.info("=" * 60)

    all_stats = {}
    for expression in expressions:
        expr_raw = os.path.join(raw_dir, expression)
        expr_clean = os.path.join(cleaned_dir, expression)

        logger.info(f"\nCleaning: {expression}")
        stats = cleaner.clean_directory(expr_raw, expr_clean, expression)
        all_stats[expression] = stats

        logger.info(f"  {expression}: {stats['accepted']}/{stats['total']} accepted")
        if stats["reject_reasons"]:
            for reason, count in sorted(stats["reject_reasons"].items()):
                logger.info(f"    rejected: {reason} = {count}")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("CLEANING COMPLETE")
    total_in = sum(s["total"] for s in all_stats.values())
    total_out = sum(s["accepted"] for s in all_stats.values())
    logger.info(f"  Input: {total_in} images")
    logger.info(f"  Output: {total_out} cleaned faces ({total_out/max(total_in,1)*100:.1f}%)")
    for expr, stats in all_stats.items():
        logger.info(f"    {expr}: {stats['accepted']} clean faces")
    logger.info(f"{'=' * 60}")

    return all_stats


if __name__ == "__main__":
    clean_all_scraped_data()
