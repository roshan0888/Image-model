#!/usr/bin/env python3
"""
Bootstrap Dataset Creator

Creates an initial paired training dataset from existing test images
using the current LivePortrait pipeline to generate expression variants.

This gives the training loop data to work with immediately,
before the full internet scraping pipeline runs.
"""

import os
import sys
import cv2
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("bootstrap")

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ENGINE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")

sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, LP_DIR)


def create_bootstrap_dataset():
    """Create training pairs from existing test images using LP."""
    import yaml

    config_path = os.path.join(ENGINE_DIR, "configs", "pipeline_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create dirs
    cleaned_dir = config["paths"]["cleaned_dir"]
    paired_dir = config["paths"]["paired_dir"]
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(paired_dir, exist_ok=True)

    # Find test images
    test_dir = os.path.join(BASE_DIR, "MagicFace", "test_images")
    test_images = []
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(test_dir, f))

    if not test_images:
        logger.error(f"No test images found in {test_dir}")
        return

    logger.info(f"Found {len(test_images)} test images")

    # Load InsightFace for embeddings
    from insightface.app import FaceAnalysis
    face_analyzer = FaceAnalysis(
        name="antelopev2",
        root=config["paths"]["antelopev2_dir"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    # Load LP pipeline for generating expression variants
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    from src.config.argument_config import ArgumentConfig

    lp = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())

    driving_dir = os.path.join(LP_DIR, "assets", "examples", "driving")
    expressions = {
        "smile": "d30.jpg",
        "big_smile": "d12.jpg",
        "surprise": "d19.jpg",
        "angry": "d38.jpg",
        "sad": "d8.jpg",
    }

    annotations = []
    pairs = []

    for img_path in test_images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = face_analyzer.get(img)
        if not faces:
            continue

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        src_name = os.path.splitext(os.path.basename(img_path))[0]

        # Save source as "neutral"
        neutral_dir = os.path.join(cleaned_dir, "neutral")
        os.makedirs(neutral_dir, exist_ok=True)
        src_cleaned = os.path.join(neutral_dir, f"{src_name}.jpg")

        # Crop face with padding
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = img.shape[:2]
        fw, fh = x2-x1, y2-y1
        pad = int(max(fw, fh) * 0.3)
        cx1 = max(0, x1-pad)
        cy1 = max(0, y1-pad)
        cx2 = min(w, x2+pad)
        cy2 = min(h, y2+pad)
        face_crop = img[cy1:cy2, cx1:cx2]
        face_512 = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(src_cleaned, face_512, [cv2.IMWRITE_JPEG_QUALITY, 95])

        annotations.append({
            "path": img_path,
            "cleaned_path": src_cleaned,
            "expression": "neutral",
            "embedding": face.normed_embedding.tolist(),
            "age": int(face.age) if hasattr(face, 'age') else 30,
            "gender": "male" if getattr(face, 'gender', 1) == 1 else "female",
        })

        logger.info(f"Source: {src_name}")

        # Generate expression variants using LP
        for expr_name, driving_file in expressions.items():
            driving_path = os.path.join(driving_dir, driving_file)
            if not os.path.exists(driving_path):
                continue

            try:
                out_dir = os.path.join(ENGINE_DIR, "temp_bootstrap")
                os.makedirs(out_dir, exist_ok=True)

                args = ArgumentConfig()
                args.source = img_path
                args.driving = driving_path
                args.output_dir = out_dir
                args.flag_pasteback = True
                args.flag_do_crop = True
                args.flag_stitching = True
                args.flag_relative_motion = True
                args.animation_region = "all"
                args.driving_option = "expression-friendly"
                args.driving_multiplier = 1.0
                args.source_max_dim = 1920
                args.flag_eye_retargeting = True
                args.flag_lip_retargeting = True

                inf_keys = {
                    'flag_pasteback', 'flag_do_crop', 'flag_stitching',
                    'flag_relative_motion', 'animation_region', 'driving_option',
                    'driving_multiplier', 'source_max_dim', 'source_division',
                    'flag_eye_retargeting', 'flag_lip_retargeting',
                }
                lp.live_portrait_wrapper.update_config(
                    {k: v for k, v in args.__dict__.items() if k in inf_keys}
                )

                wfp, _ = lp.execute(args)
                result = cv2.imread(wfp)

                if result is None:
                    continue

                # Save to expression dir
                expr_dir = os.path.join(cleaned_dir, expr_name)
                os.makedirs(expr_dir, exist_ok=True)
                expr_cleaned = os.path.join(expr_dir, f"{src_name}_{expr_name}.jpg")

                # Crop face from result
                result_faces = face_analyzer.get(result)
                if result_faces:
                    rf = max(result_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    rx1, ry1, rx2, ry2 = rf.bbox.astype(int)
                    rh, rw = result.shape[:2]
                    rfw, rfh = rx2-rx1, ry2-ry1
                    rpad = int(max(rfw, rfh) * 0.3)
                    result_crop = result[
                        max(0,ry1-rpad):min(rh,ry2+rpad),
                        max(0,rx1-rpad):min(rw,rx2+rpad)
                    ]
                    result_512 = cv2.resize(result_crop, (512, 512),
                                           interpolation=cv2.INTER_LANCZOS4)
                else:
                    result_512 = cv2.resize(result, (512, 512),
                                           interpolation=cv2.INTER_LANCZOS4)

                cv2.imwrite(expr_cleaned, result_512, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Compute identity score
                result_faces2 = face_analyzer.get(result_512)
                emb = None
                if result_faces2:
                    emb = max(result_faces2, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding
                    from sklearn.metrics.pairwise import cosine_similarity
                    score = float(cosine_similarity(
                        face.normed_embedding.reshape(1,-1),
                        emb.reshape(1,-1)
                    )[0][0])
                else:
                    score = 0.0

                annotations.append({
                    "path": img_path,
                    "cleaned_path": expr_cleaned,
                    "expression": expr_name,
                    "embedding": emb.tolist() if emb is not None else [],
                    "age": int(face.age) if hasattr(face, 'age') else 30,
                    "gender": "male" if getattr(face, 'gender', 1) == 1 else "female",
                })

                pairs.append({
                    "source_path": src_cleaned,
                    "target_path": expr_cleaned,
                    "identity_group": 0,
                    "source_expression": "neutral",
                    "target_expression": expr_name,
                })

                logger.info(f"  {expr_name}: identity={score:.4f} → {expr_cleaned}")

            except Exception as e:
                logger.warning(f"  {expr_name}: FAILED — {e}")

    # Also create synthetic pairs (for training without targets)
    for expr_name in expressions:
        for ann in annotations:
            if ann["expression"] == "neutral":
                pairs.append({
                    "source_path": ann["cleaned_path"],
                    "target_path": None,
                    "identity_group": 0,
                    "source_expression": "neutral",
                    "target_expression": expr_name,
                    "synthetic": True,
                })

    # Save annotations
    ann_path = os.path.join(cleaned_dir, "annotations.jsonl")
    with open(ann_path, "w") as f:
        for a in annotations:
            f.write(json.dumps(a) + "\n")

    # Save pairs
    pairs_path = os.path.join(paired_dir, "training_pairs.jsonl")
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    logger.info(f"\nBootstrap complete:")
    logger.info(f"  Annotations: {len(annotations)}")
    logger.info(f"  Training pairs: {len(pairs)}")
    logger.info(f"  Saved to: {cleaned_dir}")

    # Cleanup temp
    import shutil
    temp_dir = os.path.join(ENGINE_DIR, "temp_bootstrap")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return {"annotations": len(annotations), "pairs": len(pairs)}


if __name__ == "__main__":
    create_bootstrap_dataset()
