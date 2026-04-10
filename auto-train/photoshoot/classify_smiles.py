#!/usr/bin/env python3
"""
Classify scraped smile images into 3 types using facial landmarks:
  SUBTLE:     mouth barely open, eyes relaxed
  PHOTOSHOOT: mouth open (teeth), eyes still open
  NATURAL:    mouth open (teeth), eyes crinkle (Duchenne)

Also cleans: rejects blurry, no-face, extreme pose, etc.
"""

import os
import sys
import cv2
import json
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [classify] %(levelname)s: %(message)s")
logger = logging.getLogger("classify")

PHOTO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PHOTO_DIR)
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "MagicFace", "third_party_files")


class SmileClassifier:
    """Classify smile type from facial landmarks."""

    def __init__(self):
        from insightface.app import FaceAnalysis
        self.fa = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2_DIR,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
        self.seen_embeddings = []

    def analyze_face(self, image_bgr) -> Optional[Dict]:
        """Detect face, measure smile characteristics."""
        faces = self.fa.get(image_bgr)
        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1
        h, w = image_bgr.shape[:2]

        # Basic quality checks
        if fw < 128 or fh < 128:
            return {"reject": "face_too_small"}

        if len(faces) > 1:
            second = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))[-2]
            sf_area = (second.bbox[2]-second.bbox[0]) * (second.bbox[3]-second.bbox[1])
            if sf_area / (fw * fh) > 0.3:
                return {"reject": "multiple_faces"}

        # Pose check
        lmk = getattr(face, 'landmark_2d_106', None)
        if lmk is None or len(lmk) < 100:
            return {"reject": "no_landmarks"}

        nose_x = lmk[86, 0]
        face_cx = (x1 + x2) / 2
        if abs(nose_x - face_cx) / max(fw, 1) > 0.18:
            return {"reject": "extreme_pose"}

        # Blur check
        face_crop = image_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
                return {"reject": "blurry"}

        # Duplicate check
        emb = face.normed_embedding.reshape(1, -1)
        for seen in self.seen_embeddings:
            if float(np.dot(emb, seen.T)[0][0]) > 0.7:
                return {"reject": "duplicate"}
        self.seen_embeddings.append(emb)

        # ═══════════════════════════════════════════
        # SMILE CLASSIFICATION using 106 landmarks
        # ═══════════════════════════════════════════

        # Mouth landmarks (indices 52-71 in 106-point model)
        # Upper lip top: ~61, Lower lip bottom: ~65
        # Left corner: ~52, Right corner: ~58
        upper_lip = lmk[61]  # Upper lip center top
        lower_lip = lmk[65]  # Lower lip center bottom
        left_corner = lmk[52]   # Left mouth corner
        right_corner = lmk[58]  # Right mouth corner

        # Mouth openness ratio
        mouth_height = np.linalg.norm(lower_lip - upper_lip)
        mouth_width = np.linalg.norm(right_corner - left_corner)
        mouth_open_ratio = mouth_height / max(fh, 1)
        mouth_aspect = mouth_height / max(mouth_width, 1)

        # Eye landmarks
        # Left eye: ~33-42, Right eye: ~43-51
        # Upper lid: ~35,37 Lower lid: ~40,41
        left_eye_upper = lmk[35]
        left_eye_lower = lmk[40]
        right_eye_upper = lmk[44]
        right_eye_lower = lmk[49]

        left_eye_height = np.linalg.norm(left_eye_lower - left_eye_upper)
        right_eye_height = np.linalg.norm(right_eye_lower - right_eye_upper)
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        eye_openness = avg_eye_height / max(fh, 1)

        # Cheek raise estimation (distance between lower eye landmark and mouth corner)
        # When cheeks raise (AU6), lower eye landmarks move UP
        left_cheek_dist = lmk[40, 1] - lmk[52, 1]  # vertical distance
        right_cheek_dist = lmk[49, 1] - lmk[58, 1]
        avg_cheek = (left_cheek_dist + right_cheek_dist) / 2
        cheek_ratio = avg_cheek / max(fh, 1)

        # Classify smile type
        smile_metrics = {
            "mouth_open_ratio": round(float(mouth_open_ratio), 4),
            "mouth_aspect": round(float(mouth_aspect), 4),
            "eye_openness": round(float(eye_openness), 4),
            "cheek_ratio": round(float(cheek_ratio), 4),
        }

        # Classification using RELATIVE thresholds
        # mouth_open_ratio ranges 0.20-0.43 (includes lip thickness)
        # eye_openness ranges 0.05-0.09
        # Use percentile-based splits:
        #   Bottom 33% mouth → SUBTLE (least mouth open)
        #   Top 33% eye squint (lowest eye_openness) → NATURAL (Duchenne)
        #   Rest → PHOTOSHOOT
        #
        # Also respect the scraped directory as a signal

        if mouth_open_ratio < 0.29:
            smile_type = "subtle"
        elif eye_openness < 0.068:
            # Small eye aperture = eyes crinkle = Duchenne/natural
            smile_type = "natural"
        else:
            smile_type = "photoshoot"

        return {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "face_size": (int(fw), int(fh)),
            "smile_type": smile_type,
            "smile_metrics": smile_metrics,
            "embedding": face.normed_embedding.tolist(),
            "age": int(face.age) if hasattr(face, 'age') and face.age else -1,
            "gender": "male" if getattr(face, 'gender', 1) == 1 else "female",
            "accepted": True,
        }

    def crop_face(self, image_bgr, bbox, target_size=512, padding=0.4):
        """Crop face with padding."""
        x1, y1, x2, y2 = bbox
        h, w = image_bgr.shape[:2]
        fw, fh = x2 - x1, y2 - y1
        pad = int(max(fw, fh) * padding)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        size = max(fw, fh) + pad * 2
        half = size // 2

        cx1 = max(0, cx - half)
        cy1 = max(0, cy - half)
        cx2 = min(w, cx + half)
        cy2 = min(h, cy + half)

        crop = image_bgr[cy1:cy2, cx1:cx2]
        return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def classify_all():
    """Classify all scraped smile images."""
    raw_dir = os.path.join(PHOTO_DIR, "raw_smiles")
    drivers_dir = os.path.join(PHOTO_DIR, "drivers")

    classifier = SmileClassifier()

    stats = {"subtle": 0, "photoshoot": 0, "natural": 0, "rejected": 0}
    all_results = []

    for smile_dir in ["subtle", "photoshoot", "natural"]:
        src_dir = os.path.join(raw_dir, smile_dir)
        if not os.path.exists(src_dir):
            continue

        logger.info(f"\nProcessing: {smile_dir}")
        files = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for fname in files:
            fpath = os.path.join(src_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            result = classifier.analyze_face(img)
            if result is None or result.get("reject"):
                stats["rejected"] += 1
                continue

            # Use the CLASSIFIED type (may differ from scrape directory)
            classified_type = result["smile_type"]
            stats[classified_type] += 1

            # Crop and save to classified directory
            crop = classifier.crop_face(img, result["bbox"])
            out_dir = os.path.join(drivers_dir, classified_type)
            os.makedirs(out_dir, exist_ok=True)

            out_name = f"{classified_type}_{stats[classified_type]:04d}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            all_results.append({
                "source": fpath,
                "output": out_path,
                "scraped_as": smile_dir,
                "classified_as": classified_type,
                **result.get("smile_metrics", {}),
            })

    # Save classification results
    results_path = os.path.join(PHOTO_DIR, "classification_results.jsonl")
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"\n{'=' * 60}")
    logger.info("CLASSIFICATION COMPLETE")
    logger.info(f"  Subtle:     {stats['subtle']} drivers")
    logger.info(f"  Photoshoot: {stats['photoshoot']} drivers")
    logger.info(f"  Natural:    {stats['natural']} drivers")
    logger.info(f"  Rejected:   {stats['rejected']}")
    logger.info(f"{'=' * 60}")

    return stats


if __name__ == "__main__":
    classify_all()
