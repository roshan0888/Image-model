#!/usr/bin/env python3
"""
Self-Pair Generator — Phase 1B

Takes existing 305 cleaned images and creates training pairs by running
LP lip mode on each. The pair is (original, LP_output).

Training objective: make LP produce output that preserves identity BETTER
than its current output. This is bootstrapping — using the model's own
output as weak supervision while we wait for VoxCeleb2 real pairs.
"""

import os
import sys
import cv2
import json
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [self-pair] %(message)s")
logger = logging.getLogger("self-pair")

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(ENGINE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")

sys.path.insert(0, LP_DIR)


def generate_self_pairs(cleaned_dir=None, output_dir=None, driver_path=None):
    """Generate training pairs by running LP on each cleaned image."""
    if cleaned_dir is None:
        cleaned_dir = os.path.join(ENGINE_DIR, "dataset", "cleaned_scraped")
    if output_dir is None:
        output_dir = os.path.join(ENGINE_DIR, "dataset", "self_pairs")
    if driver_path is None:
        driver_path = os.path.join(LP_DIR, "assets", "examples", "driving", "d12.jpg")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "source"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "target"), exist_ok=True)
    os.makedirs(os.path.join(ENGINE_DIR, "temp_selfpair"), exist_ok=True)

    # Load LP
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    from src.config.argument_config import ArgumentConfig

    lp = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())

    # Load InsightFace for identity verification
    from insightface.app import FaceAnalysis
    from sklearn.metrics.pairwise import cosine_similarity

    fa = FaceAnalysis(
        name="antelopev2",
        root=os.path.join(BASE_DIR, "MagicFace", "third_party_files"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    # Collect all cleaned images
    all_images = []
    for expr_dir in os.listdir(cleaned_dir):
        d = os.path.join(cleaned_dir, expr_dir)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("clean_") and f.endswith(".jpg"):
                all_images.append(os.path.join(d, f))

    logger.info(f"Found {len(all_images)} cleaned images")
    logger.info(f"Driver: {os.path.basename(driver_path)}")
    logger.info(f"Mode: lip, multiplier: 1.5x")

    pairs = []
    skipped = 0

    for i, img_path in enumerate(all_images):
        fname = os.path.splitext(os.path.basename(img_path))[0]

        # Read source
        src = cv2.imread(img_path)
        if src is None:
            skipped += 1
            continue

        # Get source embedding
        faces = fa.get(src)
        if not faces:
            skipped += 1
            continue

        src_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        src_emb = src_face.normed_embedding.reshape(1, -1)

        # Run LP in lip mode at 1.5x
        args = ArgumentConfig()
        args.source = img_path
        args.driving = driver_path
        args.output_dir = os.path.join(ENGINE_DIR, "temp_selfpair")
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "lip"  # LIP MODE — our best setting
        args.driving_option = "expression-friendly"
        args.driving_multiplier = 1.5  # 1.5x — our best multiplier
        args.source_max_dim = 1920
        args.flag_eye_retargeting = False
        args.flag_lip_retargeting = False

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
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        # Check identity
        res_faces = fa.get(result)
        if not res_faces:
            skipped += 1
            continue

        res_face = max(res_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        score = float(cosine_similarity(src_emb, res_face.normed_embedding.reshape(1, -1))[0][0])

        # Save pair
        src_path = os.path.join(output_dir, "source", f"{fname}.jpg")
        tgt_path = os.path.join(output_dir, "target", f"{fname}_smile.jpg")
        cv2.imwrite(src_path, src)
        cv2.imwrite(tgt_path, result)

        pairs.append({
            "source": src_path,
            "target": tgt_path,
            "identity_score": round(score, 4),
            "expression": "smile",
            "driver": os.path.basename(driver_path),
            "multiplier": 1.5,
            "mode": "lip",
        })

        if (i + 1) % 20 == 0:
            logger.info(f"  {i+1}/{len(all_images)} processed, {len(pairs)} pairs, {skipped} skipped")

    # Save pairs manifest
    manifest_path = os.path.join(output_dir, "self_pairs.jsonl")
    with open(manifest_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    logger.info(f"\nDone: {len(pairs)} pairs created, {skipped} skipped")
    logger.info(f"Manifest: {manifest_path}")

    # Stats
    if pairs:
        scores = [p["identity_score"] for p in pairs]
        logger.info(f"Identity: avg={np.mean(scores):.4f} min={min(scores):.4f} max={max(scores):.4f}")

    return pairs


if __name__ == "__main__":
    generate_self_pairs()
