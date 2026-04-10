#!/usr/bin/env python3
"""
10-Stage Quality Filter for Pose Studio

Filters scraped images and classifies by pose angle + expression.
Outputs cleaned 512x512 crops with full annotations.
"""

import os
import sys
import cv2
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [filter] %(message)s")
logger = logging.getLogger("filter")

STUDIO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANTELOPEV2 = os.path.join(STUDIO_DIR, "MagicFace", "third_party_files")
LP_DIR = os.path.join(os.path.dirname(STUDIO_DIR), "LivePortrait")

sys.path.insert(0, LP_DIR)

# Pose angle buckets
YAW_BUCKETS = {
    "far_left": (-45, -20), "slight_left": (-20, -8), "straight": (-8, 8),
    "slight_right": (8, 20), "far_right": (20, 45),
}
PITCH_BUCKETS = {
    "look_up": (-30, -10), "level": (-10, 10), "look_down": (10, 30),
}
ROLL_BUCKETS = {
    "tilt_left": (-30, -8), "straight": (-8, 8), "tilt_right": (8, 30),
}


class PoseQualityFilter:
    """10-stage quality filter with pose + expression classification."""

    def __init__(self):
        from insightface.app import FaceAnalysis
        self.fa = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        # Load LP motion extractor for precise pose angles
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        self.lp = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig()
        )

        self.seen_embeddings = []
        logger.info("PoseQualityFilter ready")

    def _get_pose_from_lp(self, image_bgr):
        """Get precise pose angles using LP's motion extractor."""
        try:
            crop_info = self.lp.cropper.crop_source_image(image_bgr, self.lp.cropper.crop_cfg)
            if crop_info is None:
                return None
            img_256 = crop_info['img_crop_256x256']
            I_s = self.lp.live_portrait_wrapper.prepare_source(img_256)
            kp_info = self.lp.live_portrait_wrapper.get_kp_info(I_s)
            return {
                "pitch": float(kp_info['pitch'].item()),
                "yaw": float(kp_info['yaw'].item()),
                "roll": float(kp_info['roll'].item()),
            }
        except Exception:
            return None

    def _classify_pose(self, yaw, pitch, roll):
        """Classify pose into buckets."""
        yaw_label = "unknown"
        for label, (lo, hi) in YAW_BUCKETS.items():
            if lo <= yaw < hi:
                yaw_label = label
                break

        pitch_label = "unknown"
        for label, (lo, hi) in PITCH_BUCKETS.items():
            if lo <= pitch < hi:
                pitch_label = label
                break

        roll_label = "unknown"
        for label, (lo, hi) in ROLL_BUCKETS.items():
            if lo <= roll < hi:
                roll_label = label
                break

        return yaw_label, pitch_label, roll_label

    def _classify_expression(self, face):
        """Classify expression from landmarks."""
        lmk = getattr(face, 'landmark_2d_106', None)
        if lmk is None or len(lmk) < 72:
            return "unknown"

        bbox = face.bbox
        face_h = max(bbox[3] - bbox[1], 1)

        upper_lip = lmk[61]
        lower_lip = lmk[65]
        mouth_open = np.linalg.norm(lower_lip - upper_lip) / face_h

        left_eye = lmk[35:42]
        right_eye = lmk[44:51]
        eye_h = (np.linalg.norm(left_eye[2] - left_eye[5]) + np.linalg.norm(right_eye[2] - right_eye[5])) / 2
        eye_open = eye_h / face_h

        if mouth_open < 0.04:
            return "neutral" if eye_open > 0.03 else "subtle_smile"
        elif eye_open < 0.025:
            return "natural_smile"
        else:
            return "photoshoot_smile"

    def process_image(self, image_path):
        """Run 10-stage filter on one image. Returns annotation dict or None."""
        # Stage 1: Readable
        img = cv2.imread(image_path)
        if img is None:
            return None, "unreadable"

        # Stage 2: Face detected
        faces = self.fa.get(img)
        if not faces:
            return None, "no_face"

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        # Stage 3: Single face
        if len(faces) > 1:
            second = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))[-2]
            if (second.bbox[2]-second.bbox[0])*(second.bbox[3]-second.bbox[1]) / ((face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1])) > 0.3:
                return None, "multi_face"

        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2-x1, y2-y1
        h, w = img.shape[:2]

        # Stage 4: Resolution
        if fw < 128 or fh < 128:
            return None, "low_res"

        # Stage 5: Sharpness
        face_crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < 50:
            return None, "blurry"

        # Stage 6: Lighting
        brightness = gray.mean()
        contrast = gray.std()
        if brightness < 40 or brightness > 230 or contrast < 20:
            return None, "bad_lighting"

        # Stage 7: AI detection (texture energy)
        blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
        hf_energy = np.abs(gray.astype(np.float32) - blur).std()
        if hf_energy < 3.0:
            return None, "ai_generated"

        # Stage 8: Duplicate
        emb = face.normed_embedding.reshape(1, -1)
        for seen in self.seen_embeddings:
            if float(np.dot(emb, seen.T)[0][0]) > 0.70:
                return None, "duplicate"
        self.seen_embeddings.append(emb)

        # Stage 9: Pose measurable
        pose = self._get_pose_from_lp(img)
        if pose is None:
            return None, "pose_unmeasurable"

        # Stage 10: Quality score
        quality = (
            min(sharpness / 300, 1.0) * 0.3 +
            min(fw / 300, 1.0) * 0.2 +
            (1.0 if 60 < brightness < 200 else 0.5) * 0.2 +
            min(contrast / 50, 1.0) * 0.15 +
            min(hf_energy / 10, 1.0) * 0.15
        )
        if quality < 0.4:
            return None, "low_quality"

        # Classify pose
        yaw_label, pitch_label, roll_label = self._classify_pose(
            pose["yaw"], pose["pitch"], pose["roll"]
        )

        # Classify expression
        expression = self._classify_expression(face)

        # Crop face 512x512
        pad = int(max(fw, fh) * 0.35)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        sz = max(fw, fh) + pad*2
        half = sz // 2
        crop = img[max(0,cy-half):min(h,cy+half), max(0,cx-half):min(w,cx+half)]
        crop_512 = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        return {
            "image": crop_512,
            "yaw": round(pose["yaw"], 1),
            "pitch": round(pose["pitch"], 1),
            "roll": round(pose["roll"], 1),
            "yaw_label": yaw_label,
            "pitch_label": pitch_label,
            "roll_label": roll_label,
            "expression": expression,
            "quality": round(quality, 4),
            "sharpness": round(sharpness, 1),
            "age": int(face.age) if hasattr(face, 'age') and face.age else -1,
            "gender": "male" if getattr(face, 'gender', 1) == 1 else "female",
            "embedding": emb.tolist(),
        }, "accepted"


def filter_all():
    """Filter all scraped images."""
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    clean_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cleaned")
    os.makedirs(clean_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        logger.error(f"No raw data at {raw_dir}")
        return

    fltr = PoseQualityFilter()

    stats = {"accepted": 0, "rejected": 0, "reasons": {}}
    annotations = []

    for category in sorted(os.listdir(raw_dir)):
        cat_dir = os.path.join(raw_dir, category)
        if not os.path.isdir(cat_dir):
            continue

        logger.info(f"\nProcessing: {category}")
        files = [f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for fname in files:
            fpath = os.path.join(cat_dir, fname)
            result, reason = fltr.process_image(fpath)

            if result is None:
                stats["rejected"] += 1
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                continue

            stats["accepted"] += 1
            out_name = f"clean_{stats['accepted']:04d}.jpg"
            out_path = os.path.join(clean_dir, out_name)
            cv2.imwrite(out_path, result["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])

            ann = {k: v for k, v in result.items() if k != "image"}
            ann["path"] = out_path
            ann["source"] = fpath
            ann["source_category"] = category
            annotations.append(ann)

        logger.info(f"  {category}: {stats['accepted']} total accepted")

    # Save annotations
    ann_path = os.path.join(clean_dir, "annotations.jsonl")
    with open(ann_path, "w") as f:
        for a in annotations:
            f.write(json.dumps(a, default=str) + "\n")

    logger.info(f"\n{'='*60}")
    logger.info(f"FILTER COMPLETE")
    logger.info(f"  Accepted: {stats['accepted']}")
    logger.info(f"  Rejected: {stats['rejected']}")
    for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
        logger.info(f"    {reason}: {count}")
    logger.info(f"  Annotations: {ann_path}")


if __name__ == "__main__":
    filter_all()
