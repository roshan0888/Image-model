"""
Face Quality Gate
=================
Routes input images to the right engine based on face characteristics.

  yaw <  20°  → "frontal"       → LP region='lip' (95%+ identity, fast)
  yaw <  35°  → "near_frontal"  → LP region='lip' (lower confidence, may fall back)
  yaw <  60°  → "three_quarter" → InstantID img2img (only option)
  yaw >= 60°  → "profile"       → REJECT (no engine handles this well)

Also rejects:
  - No face detected
  - Multiple faces (ambiguous)
  - Face too small (<100px bbox)
  - Extreme tilt (pitch > 30°)
  - Blurry input (Laplacian variance < 50)
  - Eye occlusion (sunglasses, hand)
"""

import cv2, numpy as np
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class FaceClassification:
    category: str          # "frontal" | "near_frontal" | "three_quarter" | "profile"
    engine:   str          # "lp_lip" | "instantid_img2img" | "reject"
    confidence: float      # 0.0-1.0
    yaw:      float        # degrees
    pitch:    float
    roll:     float
    face_bbox: tuple
    sharpness: float
    reject_reason: Optional[str] = None


class FaceQualityGate:

    def __init__(self, insightface_root: str = "MagicFace/third_party_files"):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="antelopev2", root=insightface_root)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def classify(self, image_path: str) -> FaceClassification:
        img = cv2.imread(image_path)
        if img is None:
            return FaceClassification("invalid", "reject", 0.0, 0, 0, 0, (0,0,0,0), 0.0,
                                      reject_reason="unreadable")

        faces = self.app.get(img)

        # No face
        if not faces:
            return FaceClassification("none", "reject", 0.0, 0, 0, 0, (0,0,0,0), 0.0,
                                      reject_reason="no_face")

        # Multiple faces — pick largest but warn
        if len(faces) > 1:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            multi_face_warning = True
        else:
            face = faces[0]
            multi_face_warning = False

        # Extract pose
        if hasattr(face, "pose") and face.pose is not None:
            pitch, yaw, roll = float(face.pose[0]), float(face.pose[1]), float(face.pose[2])
        else:
            pitch = yaw = roll = 0.0

        # Bbox & size check
        bbox = face.bbox.astype(int)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if min(bw, bh) < 100:
            return FaceClassification("small", "reject", 0.0, yaw, pitch, roll,
                                      tuple(bbox), 0.0, reject_reason="face_too_small")

        # Detection confidence
        if face.det_score < 0.65:
            return FaceClassification("low_conf", "reject", float(face.det_score),
                                      yaw, pitch, roll, tuple(bbox), 0.0,
                                      reject_reason="low_detection_confidence")

        # Sharpness check on face crop
        crop = img[max(0,bbox[1]):bbox[3], max(0,bbox[0]):bbox[2]]
        if crop.size == 0:
            return FaceClassification("invalid", "reject", 0.0, yaw, pitch, roll,
                                      tuple(bbox), 0.0, reject_reason="bad_crop")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < 50:
            return FaceClassification("blurry", "reject", float(face.det_score),
                                      yaw, pitch, roll, tuple(bbox), sharpness,
                                      reject_reason="blurry")

        # Pitch (head up/down) check — too extreme either way is bad
        if abs(pitch) > 30:
            return FaceClassification("extreme_pitch", "reject", float(face.det_score),
                                      yaw, pitch, roll, tuple(bbox), sharpness,
                                      reject_reason=f"pitch_{abs(pitch):.0f}deg")

        # ── Yaw-based routing ──
        abs_yaw = abs(yaw)
        if abs_yaw < 20:
            cat = "frontal"
            engine = "lp_lip"
            conf = 1.0 - (abs_yaw / 20) * 0.1   # 0.9-1.0
        elif abs_yaw < 35:
            cat = "near_frontal"
            engine = "lp_lip"
            conf = 0.8 - ((abs_yaw - 20) / 15) * 0.2   # 0.6-0.8
        elif abs_yaw < 60:
            cat = "three_quarter"
            engine = "instantid_img2img"
            conf = 0.7 - ((abs_yaw - 35) / 25) * 0.2   # 0.5-0.7
        else:
            return FaceClassification("profile", "reject", float(face.det_score),
                                      yaw, pitch, roll, tuple(bbox), sharpness,
                                      reject_reason=f"profile_{abs_yaw:.0f}deg_yaw")

        return FaceClassification(
            category=cat,
            engine=engine,
            confidence=conf,
            yaw=yaw, pitch=pitch, roll=roll,
            face_bbox=tuple(bbox),
            sharpness=sharpness,
            reject_reason="multiple_faces" if multi_face_warning else None,
        )

    def batch_classify(self, image_dir: str) -> Dict[str, list]:
        """Classify all images in a directory. Returns dict by category."""
        results = {"frontal": [], "near_frontal": [], "three_quarter": [], "rejected": []}
        for img_path in sorted(Path(image_dir).glob("*.jpg")):
            r = self.classify(str(img_path))
            if r.engine == "reject":
                results["rejected"].append((str(img_path), r.reject_reason))
            else:
                results[r.category].append((str(img_path), r.confidence, r.yaw))
        return results


# ── CLI for testing ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, argparse
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Image file or directory")
    args = p.parse_args()

    gate = FaceQualityGate()

    if Path(args.path).is_dir():
        results = gate.batch_classify(args.path)
        print(f"\n{'='*60}")
        print(f"  FACE QUALITY GATE — Batch Results")
        print(f"{'='*60}")
        for cat, items in results.items():
            print(f"\n  {cat.upper()}: {len(items)}")
            for item in items[:5]:
                print(f"    {item}")
            if len(items) > 5:
                print(f"    ... and {len(items)-5} more")
    else:
        r = gate.classify(args.path)
        print(f"\n{'='*60}")
        print(f"  {Path(args.path).name}")
        print(f"{'='*60}")
        print(f"  Category:    {r.category}")
        print(f"  Engine:      {r.engine}")
        print(f"  Confidence:  {r.confidence:.2f}")
        print(f"  Yaw:         {r.yaw:+.1f}°")
        print(f"  Pitch:       {r.pitch:+.1f}°")
        print(f"  Roll:        {r.roll:+.1f}°")
        print(f"  Sharpness:   {r.sharpness:.1f}")
        print(f"  Bbox:        {r.face_bbox}")
        if r.reject_reason:
            print(f"  Reject:      {r.reject_reason}")
