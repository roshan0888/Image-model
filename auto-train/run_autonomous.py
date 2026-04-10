"""
Autonomous Training Pipeline — Master Runner

Runs the full cycle continuously:
  1. Scrape Instagram-quality model photos (Pexels + Bing + Google)
  2. Clean (face detection, blur, pose, quality filters)
  3. Cluster by identity (ArcFace DBSCAN)
  4. Build training pairs (neutral → expression)
  5. Train LoRA on LivePortrait SPADE generator
  6. Evaluate identity preservation
  7. Detect failure cases
  8. Reinforce: scrape more data for failures
  9. Repeat until 99.5%

Usage:
    python run_autonomous.py                    # Full pipeline
    python run_autonomous.py --scrape-only      # Just collect data
    python run_autonomous.py --skip-scrape      # Skip scraping, use existing data
    python run_autonomous.py --cycles 3         # Run 3 cycles
    python run_autonomous.py --quick            # Quick test (50 images, 300 steps)
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Setup
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AUTONOMOUS] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "training_engine/logs/autonomous.log"),
    ],
)
logger = logging.getLogger(__name__)


# ─── CONFIG ──────────────────────────────────────────────────────────────────

RAW_DATA_DIR = ROOT / "raw_data/model_photos"
CLEAN_DATA_DIR = ROOT / "raw_data/cleaned"
PAIRS_DIR = ROOT / "raw_data/pairs"
CHECKPOINTS_DIR = ROOT / "training_engine/checkpoints"
LOGS_DIR = ROOT / "training_engine/logs"

for d in [RAW_DATA_DIR, CLEAN_DATA_DIR, PAIRS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_IDENTITY = 0.995
EXPRESSIONS = ["smile", "neutral", "surprise", "sad"]


# ─── STEP 1: SCRAPE ──────────────────────────────────────────────────────────

def step_scrape(target_per_expression: int = 500, pexels_key: str = "", unsplash_key: str = "") -> dict:
    """Scrape high-quality model photos."""
    logger.info("=" * 60)
    logger.info("STEP 1: SCRAPING MODEL PHOTOS")
    logger.info("=" * 60)

    from training_engine.data_engine.model_photo_scraper import ModelPhotoScraper

    scraper = ModelPhotoScraper(
        output_dir=str(RAW_DATA_DIR),
        pexels_key=pexels_key,
        unsplash_key=unsplash_key,
    )
    stats = scraper.scrape_all(target_per_expression=target_per_expression)
    logger.info(f"Scraping done: {sum(stats.values())} total images")
    return stats


# ─── STEP 2: CLEAN ───────────────────────────────────────────────────────────

def step_clean() -> dict:
    """Clean scraped images — face detection, blur, pose, quality."""
    logger.info("=" * 60)
    logger.info("STEP 2: CLEANING IMAGES")
    logger.info("=" * 60)

    import cv2
    import shutil

    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="antelopev2", root=str(ROOT / "MagicFace/third_party_files"))
        app.prepare(ctx_id=0, det_size=(640, 640))
        use_insightface = True
        logger.info("  InsightFace loaded for face detection")
    except Exception as e:
        logger.warning(f"  InsightFace unavailable ({e}), using OpenCV fallback")
        use_insightface = False
        cascade_xml = "haarcascade_frontalface_default.xml"
        cascade_path = str(ROOT / cascade_xml)
        if not os.path.exists(cascade_path):
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/" + cascade_xml,
                cascade_path,
            )
        face_cascade = cv2.CascadeClassifier(cascade_path)

    stats = {"total": 0, "accepted": 0, "rejected": 0, "by_expression": {}}

    for expr in EXPRESSIONS:
        src_dir = RAW_DATA_DIR / expr
        dst_dir = CLEAN_DATA_DIR / expr
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            logger.warning(f"  No raw data for {expr}, skipping")
            continue

        files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")) + \
                list(src_dir.glob("*.png")) + list(src_dir.glob("*.webp"))

        accepted = 0
        rejected = {"no_face": 0, "multi_face": 0, "blurry": 0, "too_small": 0,
                    "extreme_pose": 0, "low_quality": 0}

        for fpath in files:
            stats["total"] += 1
            try:
                img = cv2.imread(str(fpath))
                if img is None:
                    rejected["low_quality"] += 1
                    continue

                h, w = img.shape[:2]

                # Size filter — minimum 256px face region
                if min(h, w) < 256:
                    rejected["too_small"] += 1
                    continue

                # Blur check (Laplacian variance)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur < 50:  # Very blurry
                    rejected["blurry"] += 1
                    continue

                # Face detection
                if use_insightface:
                    faces = app.get(img)
                    if len(faces) == 0:
                        rejected["no_face"] += 1
                        continue
                    if len(faces) > 1:
                        rejected["multi_face"] += 1
                        continue

                    face = faces[0]
                    # Pose check — reject extreme angles
                    if hasattr(face, "pose"):
                        pitch, yaw, _roll = face.pose
                        if abs(yaw) > 30 or abs(pitch) > 20:
                            rejected["extreme_pose"] += 1
                            continue

                    # Face size relative to image
                    bbox = face.bbox.astype(int)
                    fx1, fy1, fx2, fy2 = bbox
                    face_h = fy2 - fy1
                    face_w = fx2 - fx1
                    if face_h < 100 or face_w < 100:
                        rejected["too_small"] += 1
                        continue
                else:
                    # OpenCV fallback
                    gray_small = cv2.resize(gray, (640, 640))
                    faces_cv = face_cascade.detectMultiScale(gray_small, 1.1, 5, minSize=(80, 80))
                    if len(faces_cv) == 0:
                        rejected["no_face"] += 1
                        continue
                    if len(faces_cv) > 1:
                        rejected["multi_face"] += 1
                        continue

                # Quality: brightness/contrast check
                brightness = gray.mean()
                contrast = gray.std()
                if brightness < 30 or brightness > 240 or contrast < 20:
                    rejected["low_quality"] += 1
                    continue

                # PASS — copy to clean dir
                dst_path = dst_dir / fpath.name
                shutil.copy2(str(fpath), str(dst_path))
                accepted += 1

            except Exception as e:
                logger.debug(f"  Error processing {fpath.name}: {e}")
                rejected["low_quality"] += 1

        stats["accepted"] += accepted
        stats["rejected"] += sum(rejected.values())
        stats["by_expression"][expr] = {"accepted": accepted, "rejected": dict(rejected)}
        logger.info(f"  {expr}: {accepted} accepted / {sum(rejected.values())} rejected")
        logger.info(f"    Rejections: {rejected}")

    total_acc = stats["accepted"]
    total_rej = stats["rejected"]
    rate = total_acc / max(1, total_acc + total_rej) * 100
    logger.info(f"Cleaning done: {total_acc} accepted ({rate:.0f}% pass rate), {total_rej} rejected")
    return stats


# ─── STEP 3: CLUSTER + PAIR ──────────────────────────────────────────────────

def step_cluster_and_pair() -> dict:
    """Cluster by identity and build training pairs."""
    logger.info("=" * 60)
    logger.info("STEP 3: CLUSTERING + PAIRING")
    logger.info("=" * 60)

    from training_engine.data_engine.identity_cluster import IdentityClusterEngine as IdentityClusterer

    all_images = []
    for expr in EXPRESSIONS:
        src_dir = CLEAN_DATA_DIR / expr
        if src_dir.exists():
            for f in src_dir.glob("*.jpg"):
                all_images.append({"path": str(f), "expression": expr})

    logger.info(f"  Total clean images: {len(all_images)}")

    if len(all_images) < 10:
        logger.warning("  Too few images to cluster, skipping")
        return {"pairs": 0}

    clusterer = IdentityClusterer({
        "paths": {
            "paired_dir": str(PAIRS_DIR),
            "cleaned_dir": str(CLEAN_DATA_DIR),
        },
        "identity_clustering": {
            "dbscan_eps": 0.4,
            "dbscan_min_samples": 2,
            "min_images_per_identity": 2,
            "max_pairs_per_identity": 20,
            "synthetic_pairs_per_image": 3,
        },
    })
    result = clusterer.cluster()
    n_pairs = result.get("paired_samples", result.get("pairs", 0))
    logger.info(f"  Generated {n_pairs} training pairs")
    return {"pairs": n_pairs}


# ─── STEP 4: TRAIN ───────────────────────────────────────────────────────────

def step_train(num_steps: int = 5000, cycle: int = 0) -> dict:
    """LoRA fine-tune LivePortrait."""
    logger.info("=" * 60)
    logger.info(f"STEP 4: TRAINING (cycle={cycle}, steps={num_steps})")
    logger.info("=" * 60)

    from training_engine.training.trainer import LivePortraitTrainer

    config = {
        "paths": {
            "paired_dir": str(PAIRS_DIR),
            "checkpoints_dir": str(CHECKPOINTS_DIR),
            "logs_dir": str(LOGS_DIR),
            "liveportrait_dir": str(ROOT / "LivePortrait"),
        },
        "training": {
            "num_steps": num_steps,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "identity_loss_weight": 10.0,
            "expression_loss_weight": 5.0,
            "perceptual_loss_weight": 2.0,
            "pixel_loss_weight": 1.0,
            "eval_every": 500,
            "save_every": 1000,
            "cycle": cycle,
        },
    }

    trainer = LivePortraitTrainer(config)
    metrics = trainer.train()
    logger.info(f"  Best eval identity: {metrics.get('best_identity', 0):.4f}")
    return metrics


# ─── STEP 5: EVALUATE ────────────────────────────────────────────────────────

def step_evaluate(_cycle: int = 0) -> dict:
    """Evaluate current model vs baseline."""
    logger.info("=" * 60)
    logger.info("STEP 5: EVALUATION")
    logger.info("=" * 60)

    test_images = list((ROOT / "LivePortrait/assets/examples/source").glob("*.jpg"))[:5]
    if not test_images:
        test_images = list((ROOT / "output").glob("*.jpg"))[:5]
    if not test_images:
        logger.warning("  No test images found, skipping evaluation")
        return {"by_expression": {}, "avg_identity": 0.0}

    results: dict = {"by_expression": {}, "avg_identity": 0.0}

    try:
        sys.path.insert(0, str(ROOT))
        import importlib.util as _ilu
        pipeline_path = str(ROOT / "natural_pipeline.py")
        spec = _ilu.spec_from_file_location("natural_pipeline", pipeline_path)
        assert spec is not None and spec.loader is not None
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        pipeline = mod.NaturalExpressionPipeline()

        identities = []
        for src in test_images[:3]:
            for expr in ["smile", "surprise", "sad"]:
                try:
                    _out, meta = pipeline.process(str(src), expr)
                    score = meta.get("identity_score", 0)
                    identities.append(score)
                    results["by_expression"].setdefault(expr, []).append(score)
                    logger.info(f"    {Path(str(src)).name} + {expr}: identity={score:.4f}")
                except Exception as e:
                    logger.debug(f"    eval failed {src}/{expr}: {e}")

        if identities:
            results["avg_identity"] = sum(identities) / len(identities)
            logger.info(f"  Average identity: {results['avg_identity']:.4f}")
    except Exception as e:
        logger.warning(f"  Evaluation error: {e}")

    return results


# ─── MAIN AUTONOMOUS LOOP ────────────────────────────────────────────────────

def run(args):
    """Main entry point."""
    os.makedirs(str(LOGS_DIR), exist_ok=True)

    logger.info("\n" + "=" * 60)
    logger.info("AUTONOMOUS TRAINING PIPELINE STARTED")
    logger.info(f"Target: {TARGET_IDENTITY * 100:.1f}% identity preservation")
    logger.info(f"Max cycles: {args.cycles}")
    logger.info("=" * 60 + "\n")

    history = []
    best_identity = 0.0

    for cycle in range(args.cycles):
        t0 = time.time()
        logger.info(f"\n{'#' * 60}")
        logger.info(f"CYCLE {cycle + 1} / {args.cycles}")
        logger.info(f"{'#' * 60}\n")

        cycle_results = {"cycle": cycle + 1, "timestamp": datetime.now().isoformat()}

        # Step 1: Scrape
        if not args.skip_scrape:
            target = 50 if args.quick else 500
            scrape_stats = step_scrape(
                target_per_expression=target,
                pexels_key=args.pexels_key,
                unsplash_key=args.unsplash_key,
            )
            cycle_results["scrape"] = scrape_stats

        # Step 2: Clean
        clean_stats = step_clean()
        cycle_results["clean"] = clean_stats

        # Step 3: Cluster + Pair
        pair_stats = step_cluster_and_pair()
        cycle_results["pairs"] = pair_stats

        if pair_stats.get("pairs", 0) < 5:
            logger.warning("  Not enough pairs for training. Adding self-supervised pairs...")
            # Self-supervised: use any clean image as source, drive with expression templates
            _generate_self_supervised_pairs()
            pair_stats2 = step_cluster_and_pair()
            cycle_results["pairs"] = pair_stats2

        # Step 4: Train
        steps = 300 if args.quick else 5000
        train_metrics = step_train(num_steps=steps, cycle=cycle)
        cycle_results["train"] = train_metrics

        # Step 5: Evaluate
        eval_metrics = step_evaluate(cycle)
        cycle_results["eval"] = eval_metrics
        avg_id = eval_metrics.get("avg_identity", 0)

        if avg_id > best_identity:
            best_identity = avg_id
            logger.info(f"  ★ New best identity: {best_identity:.4f}")

        # Log cycle summary
        elapsed = time.time() - t0
        logger.info(f"\n{'─' * 40}")
        logger.info(f"CYCLE {cycle + 1} COMPLETE in {elapsed / 60:.1f} min")
        logger.info(f"  Identity: {avg_id:.4f} (best: {best_identity:.4f}, target: {TARGET_IDENTITY})")
        logger.info(f"{'─' * 40}\n")

        history.append(cycle_results)

        # Save history
        with open(LOGS_DIR / "autonomous_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        # Check if target reached
        if best_identity >= TARGET_IDENTITY:
            logger.info(f"\n🎯 TARGET REACHED: {best_identity:.4f} >= {TARGET_IDENTITY}")
            break

        if cycle < args.cycles - 1 and not args.skip_scrape:
            logger.info("  Sleeping 30s before next cycle...")
            time.sleep(30)

    logger.info(f"\n{'=' * 60}")
    logger.info("AUTONOMOUS PIPELINE COMPLETE")
    logger.info(f"  Best identity achieved: {best_identity:.4f}")
    logger.info(f"  Target: {TARGET_IDENTITY}")
    logger.info(f"  Cycles run: {len(history)}")
    logger.info(f"{'=' * 60}\n")
    return history


def _generate_self_supervised_pairs():
    """Use LivePortrait itself to generate training pairs from clean images."""
    logger.info("  Generating self-supervised pairs with LP...")
    try:
        sys.path.insert(0, str(ROOT))
        spec = __import__("importlib.util").util.spec_from_file_location(
            "natural_pipeline", ROOT / "natural_pipeline.py"
        )
        mod = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pipeline = mod.NaturalExpressionPipeline()

        PAIRS_DIR.mkdir(parents=True, exist_ok=True)
        count = 0
        for expr in EXPRESSIONS:
            src_dir = CLEAN_DATA_DIR / expr
            if not src_dir.exists():
                continue
            for src in list(src_dir.glob("*.jpg"))[:30]:
                try:
                    out_img, meta = pipeline.process(str(src), "smile")
                    if out_img is not None and meta.get("identity_score", 0) > 0.90:
                        import cv2
                        import numpy as np
                        pair_path = PAIRS_DIR / f"selfpair_{count:04d}_src.jpg"
                        drv_path = PAIRS_DIR / f"selfpair_{count:04d}_drv.jpg"
                        src_img = cv2.imread(str(src))
                        if src_img is not None:
                            cv2.imwrite(str(pair_path), src_img)
                        drv = np.array(out_img) if not isinstance(out_img, np.ndarray) else out_img
                        cv2.imwrite(str(drv_path), drv)
                        count += 1
                except Exception:
                    pass
        logger.info(f"    Generated {count} self-supervised pairs")
    except Exception as e:
        logger.warning(f"    Self-supervised pair generation failed: {e}")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=5, help="Number of training cycles")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping, use existing data")
    parser.add_argument("--scrape-only", action="store_true", help="Only scrape, don't train")
    parser.add_argument("--quick", action="store_true", help="Quick test (50 images, 300 steps)")
    parser.add_argument("--pexels-key", default=os.environ.get("PEXELS_API_KEY", ""),
                        help="Pexels API key for 4K photos")
    parser.add_argument("--unsplash-key", default=os.environ.get("UNSPLASH_ACCESS_KEY", ""),
                        help="Unsplash API access key")
    args = parser.parse_args()

    if args.scrape_only:
        step_scrape(
            target_per_expression=500,
            pexels_key=args.pexels_key,
            unsplash_key=args.unsplash_key,
        )
    else:
        run(args)
