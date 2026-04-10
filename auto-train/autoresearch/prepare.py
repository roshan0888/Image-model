#!/usr/bin/env python3
"""
prepare.py — FROZEN. Do not modify.

Contains:
  - LivePortrait model loading
  - InsightFace ArcFace evaluation
  - Fixed test set (20 diverse faces)
  - Evaluation function (the ground truth metric)
  - LP execution function

This is the equivalent of Karpathy's prepare.py:
  "fixed constants, one-time data prep, and runtime utilities"
"""

import os
import sys
import cv2
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ANTELOPEV2 = os.path.join(BASE_DIR, "MagicFace", "third_party_files")
NEUTRAL_DIR = os.path.join(BASE_DIR, "training_engine", "dataset", "cleaned_scraped", "neutral")
DRIVER_DIR = os.path.join(LP_DIR, "assets", "examples", "driving")

sys.path.insert(0, LP_DIR)

# ═══════════════════════════════════════════════════
# FIXED TEST SET — 20 diverse neutral faces
# ═══════════════════════════════════════════════════

def get_test_faces():
    """Return list of 20 test face paths. FIXED — never changes."""
    faces = []
    for f in sorted(os.listdir(NEUTRAL_DIR)):
        if f.startswith("clean_") and f.endswith(".jpg"):
            faces.append(os.path.join(NEUTRAL_DIR, f))
            if len(faces) >= 20:
                break
    return faces

# ═══════════════════════════════════════════════════
# MODEL LOADING — called once at startup
# ═══════════════════════════════════════════════════

_fa = None
_lp = None

def load_models():
    """Load InsightFace + LivePortrait. Call once."""
    global _fa, _lp

    from insightface.app import FaceAnalysis
    _fa = FaceAnalysis(
        name="antelopev2", root=ANTELOPEV2,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    _lp = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())

    return _fa, _lp

def get_models():
    global _fa, _lp
    if _fa is None or _lp is None:
        load_models()
    return _fa, _lp

# ═══════════════════════════════════════════════════
# EVALUATION — the ground truth metric (DO NOT MODIFY)
# ═══════════════════════════════════════════════════

def get_embedding(fa, img):
    """Get ArcFace embedding from image."""
    faces = fa.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return face.normed_embedding.reshape(1, -1)

def get_landmarks(fa, img):
    """Get 106 landmarks."""
    faces = fa.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return getattr(face, 'landmark_2d_106', None)

def evaluate(config):
    """
    Run one experiment with given config on all 20 test faces.
    Returns metrics dict.

    This is the GROUND TRUTH evaluation — equivalent to evaluate_bpb in autoresearch.
    """
    fa, lp = get_models()
    test_faces = get_test_faces()

    from src.config.argument_config import ArgumentConfig

    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    scores = []
    mouth_disps = []
    eye_disps = []
    failures = 0

    t0 = time.time()

    for face_path in test_faces:
        src = cv2.imread(face_path)
        if src is None:
            failures += 1
            continue

        src_emb = get_embedding(fa, src)
        if src_emb is None:
            failures += 1
            continue

        src_lmk = get_landmarks(fa, src)

        # Handle tight crops
        actual_path = face_path
        if src_emb is None:
            h, w = src.shape[:2]
            pad = int(max(h, w) * 0.3)
            avg = src.mean(axis=(0, 1)).astype(np.uint8)
            padded = cv2.copyMakeBorder(src, pad, pad, pad, pad,
                                         cv2.BORDER_CONSTANT, value=avg.tolist())
            actual_path = os.path.join(temp_dir, "padded_eval.jpg")
            cv2.imwrite(actual_path, padded)
            src_emb = get_embedding(fa, padded)
            if src_emb is None:
                failures += 1
                continue

        # Run LP with config
        args = ArgumentConfig()
        args.source = actual_path
        args.driving = config.get("driver", os.path.join(DRIVER_DIR, "d12.jpg"))
        args.output_dir = temp_dir
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = config.get("flag_stitching", True)
        args.flag_relative_motion = config.get("flag_relative_motion", True)
        args.animation_region = config.get("animation_region", "lip")
        args.driving_option = config.get("driving_option", "expression-friendly")
        args.driving_multiplier = config.get("multiplier", 1.5)
        args.source_max_dim = 1920
        args.flag_eye_retargeting = config.get("flag_eye_retargeting", False)
        args.flag_lip_retargeting = config.get("flag_lip_retargeting", False)

        inf_keys = {
            'flag_pasteback', 'flag_do_crop', 'flag_stitching',
            'flag_relative_motion', 'animation_region', 'driving_option',
            'driving_multiplier', 'source_max_dim', 'source_division',
            'flag_eye_retargeting', 'flag_lip_retargeting',
        }
        lp.live_portrait_wrapper.update_config(
            {k: v for k, v in args.__dict__.items() if k in inf_keys}
        )

        try:
            wfp, _ = lp.execute(args)
            result = cv2.imread(wfp)
            if result is None:
                failures += 1
                continue
        except Exception:
            failures += 1
            continue

        # Measure identity
        res_emb = get_embedding(fa, result)
        if res_emb is None:
            failures += 1
            continue

        score = float(cosine_similarity(src_emb, res_emb)[0][0])
        scores.append(score)

        # Measure expression displacement
        res_lmk = get_landmarks(fa, result)
        if src_lmk is not None and res_lmk is not None:
            face_size = max(1, src.shape[1])
            mouth = np.linalg.norm(res_lmk[52:72] - src_lmk[52:72], axis=1).mean() / face_size
            eyes = np.linalg.norm(res_lmk[33:52] - src_lmk[33:52], axis=1).mean() / face_size
            mouth_disps.append(mouth)
            eye_disps.append(eyes)

    elapsed = time.time() - t0

    # Compute metrics
    avg_identity = np.mean(scores) if scores else 0.0
    min_identity = min(scores) if scores else 0.0
    avg_mouth = np.mean(mouth_disps) if mouth_disps else 0.0
    avg_eyes = np.mean(eye_disps) if eye_disps else 0.0
    pass_rate = sum(1 for s in scores if s >= 0.93) / max(len(scores), 1)

    # Composite score: identity weighted by expression visibility
    # Penalize if mouth_disp < 0.01 (no visible smile)
    expression_bonus = min(avg_mouth / 0.02, 1.0)  # 0-1, 1.0 if mouth moves enough
    composite = avg_identity * 0.8 + expression_bonus * 0.2

    return {
        "avg_identity": round(avg_identity, 6),
        "min_identity": round(min_identity, 6),
        "avg_mouth_disp": round(avg_mouth, 6),
        "avg_eye_disp": round(avg_eyes, 6),
        "pass_rate_93": round(pass_rate, 4),
        "composite": round(composite, 6),
        "failures": failures,
        "num_faces": len(scores),
        "elapsed_seconds": round(elapsed, 1),
    }
