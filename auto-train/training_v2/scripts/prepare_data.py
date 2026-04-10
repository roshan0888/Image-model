#!/usr/bin/env python3
"""
Data Preparation for LP Training v2

Creates paired training data: same person neutral → smiling.

Sources (in priority order):
  1. CelebA dataset — has identity labels + "Smiling" attribute
  2. Self-pairs from existing images — LP's own output as weak target
  3. VoxCeleb2 — video frames (needs large download)

This script handles source 1 and 2.
"""

import os
import sys
import cv2
import json
import numpy as np
import logging
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [data] %(message)s")
logger = logging.getLogger("data")

TRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TRAIN_DIR, "data")
BASE_DIR = os.path.dirname(TRAIN_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")

sys.path.insert(0, LP_DIR)


def create_self_pairs():
    """Create training pairs from existing scraped images using LP."""
    logger.info("Creating self-pairs from existing images...")

    cleaned_dir = os.path.join(BASE_DIR, "training_engine", "dataset", "cleaned_scraped")
    pairs_dir = os.path.join(DATA_DIR, "self_pairs")
    os.makedirs(os.path.join(pairs_dir, "source"), exist_ok=True)
    os.makedirs(os.path.join(pairs_dir, "target"), exist_ok=True)

    # Check if already done
    manifest = os.path.join(pairs_dir, "pairs.jsonl")
    if os.path.exists(manifest):
        count = sum(1 for _ in open(manifest))
        if count > 100:
            logger.info(f"  Self-pairs already exist: {count} pairs")
            return count

    # Load LP
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    from src.config.argument_config import ArgumentConfig

    lp = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())

    # Load InsightFace
    from insightface.app import FaceAnalysis
    from sklearn.metrics.pairwise import cosine_similarity

    fa = FaceAnalysis(
        name="antelopev2",
        root=os.path.join(BASE_DIR, "MagicFace", "third_party_files"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    DRIVER = os.path.join(LP_DIR, "assets", "examples", "driving", "d12.jpg")
    temp_dir = os.path.join(TRAIN_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Collect all cleaned images
    all_images = []
    if os.path.exists(cleaned_dir):
        for expr_dir in os.listdir(cleaned_dir):
            d = os.path.join(cleaned_dir, expr_dir)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.startswith("clean_") and f.endswith(".jpg"):
                        all_images.append(os.path.join(d, f))

    logger.info(f"  Found {len(all_images)} cleaned images")

    pairs = []
    for i, img_path in enumerate(all_images):
        fname = os.path.splitext(os.path.basename(img_path))[0]

        src = cv2.imread(img_path)
        if src is None:
            continue

        faces = fa.get(src)
        if not faces:
            continue

        # Run LP in lip mode at 1.8x (same as production)
        args = ArgumentConfig()
        args.source = img_path
        args.driving = DRIVER
        args.output_dir = temp_dir
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "lip"
        args.driving_option = "expression-friendly"
        args.driving_multiplier = 1.8
        args.source_max_dim = 1920

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
                continue
        except Exception:
            continue

        # Save
        src_path = os.path.join(pairs_dir, "source", f"{fname}.jpg")
        tgt_path = os.path.join(pairs_dir, "target", f"{fname}_smile.jpg")
        cv2.imwrite(src_path, src)
        cv2.imwrite(tgt_path, result)

        pairs.append({
            "source": src_path,
            "target": tgt_path,
            "type": "self_pair",
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(all_images)} processed, {len(pairs)} pairs")

    # Save manifest
    with open(manifest, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    logger.info(f"  Created {len(pairs)} self-pairs")
    return len(pairs)


def download_celeba_pairs():
    """
    Download CelebA attribute file and create neutral→smile pairs.
    CelebA has 202,599 images with "Smiling" attribute and identity labels.
    """
    logger.info("Setting up CelebA pairs...")

    celeba_dir = os.path.join(DATA_DIR, "celeba")
    os.makedirs(celeba_dir, exist_ok=True)

    # CelebA attributes file (small download, has identity + smile labels)
    attr_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U"
    attr_file = os.path.join(celeba_dir, "list_attr_celeba.txt")

    if not os.path.exists(attr_file):
        logger.info("  CelebA attribute file needs manual download from:")
        logger.info("  https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8")
        logger.info("  Download 'list_attr_celeba.txt' and 'identity_CelebA.txt'")
        logger.info("  Place in: " + celeba_dir)
        return 0

    logger.info("  CelebA attributes found, processing...")
    # Parse attributes and create pairs
    # (Implementation depends on having the actual CelebA images)
    return 0


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DATA PREPARATION FOR TRAINING v2")
    logger.info("=" * 60)

    n_self = create_self_pairs()
    n_celeba = download_celeba_pairs()

    logger.info(f"\nTotal pairs: {n_self + n_celeba}")
    logger.info(f"  Self-pairs: {n_self}")
    logger.info(f"  CelebA: {n_celeba}")
