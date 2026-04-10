#!/usr/bin/env python3
"""
Autonomous Training Pipeline v2

Runs continuously in background:
  Phase 1: Scrape more diverse face images (200/expression)
  Phase 2: Clean & filter
  Phase 3: Auto-discover best driving images (test ALL scraped faces as drivers)
  Phase 4: Deploy best drivers to production
  Phase 5: Train LoRA with differentiable identity loss
  Phase 6: Evaluate and deploy if improved
  Phase 7: Analyze RLHF feedback → scrape more for weak areas → repeat

This runs ALONGSIDE the Gradio server — user keeps testing while this improves.
"""

import os
import sys
import cv2
import json
import time
import shutil
import hashlib
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [auto-v2] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "logs", "autonomous_v2.log")),
    ],
)
logger = logging.getLogger("auto-v2")

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ENGINE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
PROD_DIR = os.path.join(BASE_DIR, "production")
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "MagicFace", "third_party_files")

sys.path.insert(0, LP_DIR)
sys.path.insert(0, ENGINE_DIR)

os.makedirs(os.path.join(ENGINE_DIR, "logs"), exist_ok=True)


class AutonomousPipeline:
    """Full autonomous training + driver discovery pipeline."""

    def __init__(self):
        self.face_analyzer = None
        self.lp_pipeline = None
        self._load_models()

    def _load_models(self):
        """Load InsightFace + LivePortrait."""
        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2_DIR,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        self.lp_pipeline = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig(),
        )
        logger.info("Models loaded")

    def get_embedding(self, image_bgr):
        """Get ArcFace embedding."""
        faces = self.face_analyzer.get(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return face.normed_embedding.reshape(1, -1)

    def run_lp(self, source_path, driver_path, multiplier=1.0, use_retarget=False):
        """Run LivePortrait and return result image."""
        from src.config.argument_config import ArgumentConfig

        out_dir = os.path.join(ENGINE_DIR, "temp_auto")
        os.makedirs(out_dir, exist_ok=True)

        args = ArgumentConfig()
        args.source = source_path
        args.driving = driver_path
        args.output_dir = out_dir
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "all"
        args.driving_option = "expression-friendly"
        args.driving_multiplier = multiplier
        args.source_max_dim = 1920

        if use_retarget:
            args.flag_eye_retargeting = True
            args.flag_lip_retargeting = True

        inf_keys = {
            'flag_pasteback', 'flag_do_crop', 'flag_stitching',
            'flag_relative_motion', 'animation_region', 'driving_option',
            'driving_multiplier', 'source_max_dim', 'source_division',
            'flag_eye_retargeting', 'flag_lip_retargeting',
        }
        self.lp_pipeline.live_portrait_wrapper.update_config(
            {k: v for k, v in args.__dict__.items() if k in inf_keys}
        )
        try:
            wfp, _ = self.lp_pipeline.execute(args)
            return cv2.imread(wfp)
        except Exception as e:
            logger.debug(f"LP failed: {e}")
            return None

    def identity_score(self, src_emb, result_bgr):
        """Measure identity preservation."""
        from sklearn.metrics.pairwise import cosine_similarity
        res_emb = self.get_embedding(result_bgr)
        if res_emb is None:
            return 0.0
        return float(cosine_similarity(src_emb, res_emb)[0][0])

    def expression_change(self, source_bgr, result_bgr):
        """Measure how much expression changed (landmark displacement)."""
        src_faces = self.face_analyzer.get(source_bgr)
        res_faces = self.face_analyzer.get(result_bgr)
        if not src_faces or not res_faces:
            return 0.0
        sf = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        rf = max(res_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        sl = getattr(sf, 'landmark_2d_106', None)
        rl = getattr(rf, 'landmark_2d_106', None)
        if sl is None or rl is None:
            return 0.0
        face_size = max(sf.bbox[2]-sf.bbox[0], sf.bbox[3]-sf.bbox[1])
        if face_size < 1:
            return 0.0
        disp = np.linalg.norm(rl - sl, axis=1) / face_size
        return float(disp.mean())

    # ══════════════════════════════════════════════════════════
    # PHASE 1: SCRAPE
    # ══════════════════════════════════════════════════════════

    def phase_scrape(self, target_per_expression=200):
        """Scrape diverse face images."""
        logger.info(f"PHASE 1: Scraping {target_per_expression} images per expression")
        from scraper import scrape_all
        return scrape_all(target_per_expression=target_per_expression)

    # ══════════════════════════════════════════════════════════
    # PHASE 2: CLEAN
    # ══════════════════════════════════════════════════════════

    def phase_clean(self):
        """Clean scraped images."""
        logger.info("PHASE 2: Cleaning images")
        from data_cleaner import clean_all_scraped_data
        return clean_all_scraped_data(ANTELOPEV2_DIR)

    # ══════════════════════════════════════════════════════════
    # PHASE 3: AUTO-DISCOVER BEST DRIVERS
    # ══════════════════════════════════════════════════════════

    def phase_discover_drivers(self, test_faces_count=5, drivers_per_expression=10):
        """
        Test ALL scraped face images as potential driving images.
        Score each by: avg identity preservation across test faces.
        Keep top N per expression.
        """
        logger.info("PHASE 3: Auto-discovering best driving images")

        cleaned_dir = os.path.join(ENGINE_DIR, "dataset", "cleaned_scraped")
        neutral_dir = os.path.join(cleaned_dir, "neutral")

        # Get test source faces (neutral expressions)
        test_faces = []
        if os.path.exists(neutral_dir):
            for f in sorted(os.listdir(neutral_dir)):
                if f.startswith("clean_") and f.endswith(".jpg"):
                    path = os.path.join(neutral_dir, f)
                    img = cv2.imread(path)
                    emb = self.get_embedding(img)
                    if emb is not None:
                        test_faces.append((path, emb))
                        if len(test_faces) >= test_faces_count:
                            break

        if len(test_faces) < 3:
            logger.warning("Not enough test faces for driver discovery")
            return {}

        logger.info(f"  Testing drivers against {len(test_faces)} source faces")

        expression_dirs = {
            "smile": [os.path.join(cleaned_dir, "smile"),
                      os.path.join(cleaned_dir, "open_smile_drivers")],
            "open_smile": [os.path.join(cleaned_dir, "open_smile_drivers"),
                           os.path.join(cleaned_dir, "smile")],
            "surprise": [os.path.join(cleaned_dir, "surprise"),
                         os.path.join(cleaned_dir, "surprise_v2")],
            "sad": [os.path.join(cleaned_dir, "sad")],
        }

        best_drivers = {}  # expression → [(path, avg_identity, avg_expression_change)]

        for expression, dirs in expression_dirs.items():
            logger.info(f"\n  Discovering best {expression} drivers...")

            # Collect all candidate drivers
            candidates = []
            for d in dirs:
                if not os.path.exists(d):
                    continue
                for f in os.listdir(d):
                    if f.startswith("clean_") and f.endswith(".jpg"):
                        candidates.append(os.path.join(d, f))

            if not candidates:
                logger.warning(f"    No candidates for {expression}")
                continue

            # Test each candidate (limit to 30 for speed)
            candidates = candidates[:30]
            results = []

            for ci, driver_path in enumerate(candidates):
                scores = []
                expr_changes = []

                for src_path, src_emb in test_faces:
                    # Test with standard mode (stronger expression)
                    for mult in [1.0, 1.5]:
                        result = self.run_lp(src_path, driver_path,
                                           multiplier=mult, use_retarget=False)
                        if result is not None:
                            score = self.identity_score(src_emb, result)
                            src_img = cv2.imread(src_path)
                            ec = self.expression_change(src_img, result)
                            scores.append(score)
                            expr_changes.append(ec)

                if scores:
                    avg_id = np.mean(scores)
                    min_id = min(scores)
                    avg_ec = np.mean(expr_changes)

                    # Combined score: want HIGH identity AND visible expression
                    # Penalize if expression change < 0.015 (too subtle)
                    combined = avg_id * 0.7 + min(avg_ec / 0.05, 1.0) * 0.3

                    results.append({
                        "path": driver_path,
                        "name": os.path.basename(driver_path),
                        "avg_identity": round(avg_id, 4),
                        "min_identity": round(min_id, 4),
                        "avg_expression_change": round(avg_ec, 4),
                        "combined_score": round(combined, 4),
                        "num_tests": len(scores),
                    })

                if (ci + 1) % 5 == 0:
                    logger.info(f"    Tested {ci+1}/{len(candidates)} candidates")

            # Sort by combined score (identity + expression balance)
            results.sort(key=lambda x: x["combined_score"], reverse=True)
            best_drivers[expression] = results[:drivers_per_expression]

            # Log top 5
            logger.info(f"    Top 5 {expression} drivers:")
            for i, r in enumerate(results[:5]):
                logger.info(
                    f"      #{i+1}: {r['name']:<30} "
                    f"id={r['avg_identity']:.4f} "
                    f"expr={r['avg_expression_change']:.4f} "
                    f"combined={r['combined_score']:.4f}"
                )

        # Save discovery results
        results_path = os.path.join(ENGINE_DIR, "logs", "driver_discovery.json")
        with open(results_path, "w") as f:
            json.dump(best_drivers, f, indent=2)

        return best_drivers

    # ══════════════════════════════════════════════════════════
    # PHASE 4: DEPLOY BEST DRIVERS
    # ══════════════════════════════════════════════════════════

    def phase_deploy_drivers(self, best_drivers: Dict):
        """Copy best drivers to production/drivers/ directory."""
        logger.info("PHASE 4: Deploying best drivers to production")

        prod_drivers_dir = os.path.join(PROD_DIR, "drivers")
        os.makedirs(prod_drivers_dir, exist_ok=True)

        deployed = {}
        for expression, drivers in best_drivers.items():
            for i, drv in enumerate(drivers[:3]):  # Top 3 per expression
                src = drv["path"]
                dst_name = f"{expression}_best_{i+1}.jpg"
                dst = os.path.join(prod_drivers_dir, dst_name)

                # Remove old pkl cache
                pkl = dst.replace(".jpg", ".pkl")
                if os.path.exists(pkl):
                    os.remove(pkl)

                shutil.copy2(src, dst)
                deployed[dst_name] = {
                    "source": src,
                    "avg_identity": drv["avg_identity"],
                    "avg_expression_change": drv["avg_expression_change"],
                    "combined_score": drv["combined_score"],
                }
                logger.info(
                    f"  Deployed: {dst_name} "
                    f"(id={drv['avg_identity']:.4f}, "
                    f"expr={drv['avg_expression_change']:.4f})"
                )

        # Save deployment manifest
        manifest_path = os.path.join(prod_drivers_dir, "deployment_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "deployed": deployed,
            }, f, indent=2)

        logger.info(f"  Deployed {len(deployed)} drivers to production")
        return deployed

    # ══════════════════════════════════════════════════════════
    # PHASE 5: DIFFERENTIABLE LORA TRAINING
    # ══════════════════════════════════════════════════════════

    def phase_train_lora(self, steps=500, lr=5e-5, rank=8, eval_every=50):
        """Train LoRA with real differentiable gradients using scraped data."""
        logger.info(f"PHASE 5: Differentiable LoRA Training ({steps} steps)")

        from differentiable_trainer import DifferentiableTrainer, build_training_pairs

        # Build pairs from all cleaned data
        pairs = build_training_pairs()
        if len(pairs) < 10:
            logger.warning(f"  Only {len(pairs)} pairs — need more data, skipping training")
            return 0.0

        logger.info(f"  Training pairs: {len(pairs)}")

        # We need to release LP model to free GPU memory for training
        # Delete current LP pipeline
        del self.lp_pipeline
        del self.face_analyzer
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        logger.info("  Released inference models to free GPU for training")

        try:
            trainer = DifferentiableTrainer(lora_rank=rank, learning_rate=lr)
            best_identity = trainer.train(
                pairs, num_steps=steps, eval_every=eval_every,
            )

            # Deploy best checkpoint to production
            ckpt_dir = os.path.join(ENGINE_DIR, "checkpoints_diff")
            deploy_dir = os.path.join(PROD_DIR, "lora_weights")
            os.makedirs(deploy_dir, exist_ok=True)

            for fname in ["lora_spade_diff_best.pt", "lora_warping_diff_best.pt", "training_meta.json"]:
                src = os.path.join(ckpt_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(deploy_dir, fname))

            logger.info(f"  Training complete. Best identity: {best_identity:.4f}")
            logger.info(f"  Deployed to {deploy_dir}")

        except Exception as e:
            logger.error(f"  Training failed: {e}")
            best_identity = 0.0

        # Reload inference models
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        self._load_models()

        return best_identity

    # ══════════════════════════════════════════════════════════
    # PHASE 6: ANALYZE RLHF FEEDBACK
    # ══════════════════════════════════════════════════════════

    def phase_analyze_feedback(self) -> Dict:  # noqa: C901
        """
        Analyze RLHF feedback to find weak areas.
        Returns recommendations for targeted data collection.
        """
        logger.info("PHASE 5: Analyzing RLHF feedback")

        feedback_file = os.path.join(PROD_DIR, "rlhf_data", "feedback_rich.jsonl")
        stats_file = os.path.join(PROD_DIR, "rlhf_data", "driver_stats.json")

        feedbacks = []
        if os.path.exists(feedback_file):
            with open(feedback_file) as f:
                for line in f:
                    try:
                        feedbacks.append(json.loads(line))
                    except:
                        pass

        if not feedbacks:
            logger.info("  No RLHF feedback yet")
            return {"weak_expressions": [], "recommendations": []}

        # Analyze per expression
        expr_stats = defaultdict(lambda: {"total": 0, "likes": 0, "avg_identity": []})
        for fb in feedbacks:
            expr = fb.get("expression", "unknown")
            expr_stats[expr]["total"] += 1
            if fb.get("liked"):
                expr_stats[expr]["likes"] += 1
            if "identity_score" in fb:
                expr_stats[expr]["avg_identity"].append(fb["identity_score"])

        weak_expressions = []
        recommendations = []

        for expr, stats in expr_stats.items():
            total = stats["total"]
            like_rate = stats["likes"] / total if total > 0 else 0
            avg_id = np.mean(stats["avg_identity"]) if stats["avg_identity"] else 0

            logger.info(
                f"  {expr}: {like_rate:.0%} liked ({stats['likes']}/{total}), "
                f"avg_identity={avg_id:.4f}"
            )

            if like_rate < 0.4 and total >= 3:
                weak_expressions.append(expr)
                if avg_id < 0.93:
                    recommendations.append(
                        f"Scrape more {expr} faces with SUBTLE expression "
                        f"(current avg identity {avg_id:.2f} too low)"
                    )
                else:
                    recommendations.append(
                        f"Expression {expr} has good identity but users dislike results. "
                        f"Try different expression style/intensity."
                    )

        result = {
            "total_feedbacks": len(feedbacks),
            "weak_expressions": weak_expressions,
            "recommendations": recommendations,
            "per_expression": {
                k: {
                    "total": v["total"],
                    "likes": v["likes"],
                    "like_rate": v["likes"]/v["total"] if v["total"] > 0 else 0,
                    "avg_identity": float(np.mean(v["avg_identity"])) if v["avg_identity"] else 0,
                }
                for k, v in expr_stats.items()
            },
        }

        logger.info(f"  Weak expressions: {weak_expressions}")
        for rec in recommendations:
            logger.info(f"  Recommendation: {rec}")

        return result

    # ══════════════════════════════════════════════════════════
    # PHASE 6: TARGETED SCRAPING FOR WEAK AREAS
    # ══════════════════════════════════════════════════════════

    def phase_targeted_scrape(self, weak_expressions: List[str], count=50):
        """Scrape more images specifically for weak expressions."""
        if not weak_expressions:
            logger.info("PHASE 6: No weak expressions — skipping targeted scrape")
            return

        logger.info(f"PHASE 6: Targeted scraping for: {weak_expressions}")

        from icrawler.builtin import BingImageCrawler

        targeted_queries = {
            "smile": [
                "person gentle smile frontal portrait studio white background",
                "subtle smile headshot professional photo",
                "closed mouth smile portrait face forward high resolution",
                "person slightly smiling passport photo style",
            ],
            "open_smile": [
                "person laughing teeth showing frontal portrait",
                "big smile teeth portrait headshot studio",
                "toothy grin face forward professional photo",
                "genuine laugh portrait closeup natural",
            ],
            "surprise": [
                "person slightly surprised raised eyebrows portrait",
                "mild surprise expression frontal face studio",
                "curious amazed look portrait headshot",
                "eyebrows raised face forward portrait photo",
            ],
            "sad": [
                "person sad frown portrait frontal studio",
                "melancholy face expression portrait closeup",
                "unhappy face frontal portrait high resolution",
                "person upset downturned lips portrait photo",
            ],
            "angry": [
                "person angry frown portrait frontal studio",
                "stern angry expression portrait headshot",
                "furious face forward portrait photo",
                "person grimacing angry portrait closeup",
            ],
        }

        for expr in weak_expressions:
            queries = targeted_queries.get(expr, [])
            if not queries:
                continue

            out_dir = os.path.join(ENGINE_DIR, "dataset", "raw", f"{expr}_targeted")
            os.makedirs(out_dir, exist_ok=True)

            per_query = max(5, count // len(queries) + 1)

            for i, q in enumerate(queries):
                try:
                    qdir = os.path.join(out_dir, f"q{i}")
                    os.makedirs(qdir, exist_ok=True)
                    c = BingImageCrawler(
                        storage={"root_dir": qdir},
                        log_level=logging.WARNING,
                    )
                    c.crawl(keyword=q, max_num=per_query, min_size=(512, 512))

                    for f in os.listdir(qdir):
                        fp = os.path.join(qdir, f)
                        if os.path.isfile(fp):
                            with open(fp, 'rb') as fh:
                                h = hashlib.md5(fh.read()).hexdigest()[:10]
                            ext = os.path.splitext(f)[1]
                            new = os.path.join(out_dir, f"targeted_{expr}_{h}{ext}")
                            if not os.path.exists(new):
                                os.rename(fp, new)
                            else:
                                os.remove(fp)
                    shutil.rmtree(qdir, ignore_errors=True)
                except Exception as e:
                    logger.debug(f"    Query failed: {e}")

            scraped = len([f for f in os.listdir(out_dir) if f.endswith(('.jpg','.png'))])
            logger.info(f"  Scraped {scraped} targeted {expr} images")

    # ══════════════════════════════════════════════════════════
    # FULL AUTONOMOUS CYCLE
    # ══════════════════════════════════════════════════════════

    def run_cycle(self, scrape_count=100, skip_scrape=False,
                  train_steps=500, train_lr=5e-5, train_rank=8):
        """Run one full autonomous cycle."""
        t0 = time.time()

        logger.info("=" * 70)
        logger.info("AUTONOMOUS PIPELINE v2 — STARTING CYCLE")
        logger.info("=" * 70)

        # Phase 1: Scrape (or skip if data exists)
        raw_dir = os.path.join(ENGINE_DIR, "dataset", "raw")
        existing_count = 0
        if os.path.exists(raw_dir):
            for d in os.listdir(raw_dir):
                dp = os.path.join(raw_dir, d)
                if os.path.isdir(dp):
                    existing_count += len([
                        f for f in os.listdir(dp)
                        if f.endswith(('.jpg', '.png'))
                    ])

        if skip_scrape or existing_count >= scrape_count * 3:
            logger.info(f"Phase 1: SKIP (already have {existing_count} raw images)")
        else:
            self.phase_scrape(scrape_count)

        # Phase 2: Clean
        cleaned_dir = os.path.join(ENGINE_DIR, "dataset", "cleaned_scraped")
        existing_clean = 0
        if os.path.exists(cleaned_dir):
            for d in os.listdir(cleaned_dir):
                dp = os.path.join(cleaned_dir, d)
                if os.path.isdir(dp):
                    existing_clean += len([
                        f for f in os.listdir(dp)
                        if f.startswith("clean_") and f.endswith(".jpg")
                    ])

        if existing_clean >= 50:
            logger.info(f"Phase 2: SKIP (already have {existing_clean} cleaned images)")
        else:
            self.phase_clean()

        # Phase 3: Discover best drivers
        best_drivers = self.phase_discover_drivers(
            test_faces_count=5, drivers_per_expression=5
        )

        # Phase 4: Deploy best drivers
        if best_drivers:
            self.phase_deploy_drivers(best_drivers)

        # Phase 5: Differentiable LoRA Training
        best_identity = self.phase_train_lora(
            steps=train_steps, lr=train_lr, rank=train_rank, eval_every=50
        )

        # Phase 6: Analyze RLHF feedback
        feedback_analysis = self.phase_analyze_feedback()

        # Phase 6: Targeted scraping for weak areas
        weak = feedback_analysis.get("weak_expressions", [])
        if weak:
            self.phase_targeted_scrape(weak, count=30)
            # Re-clean targeted data
            self.phase_clean()
            # Re-discover drivers with new data
            new_drivers = self.phase_discover_drivers(
                test_faces_count=5, drivers_per_expression=5
            )
            if new_drivers:
                self.phase_deploy_drivers(new_drivers)

        elapsed = time.time() - t0
        logger.info(f"\n{'=' * 70}")
        logger.info(f"CYCLE COMPLETE — {elapsed/60:.1f} minutes")
        logger.info(f"  Drivers deployed: {sum(len(v) for v in best_drivers.values()) if best_drivers else 0}")
        logger.info(f"  Weak expressions: {weak}")
        logger.info(f"  RLHF feedbacks: {feedback_analysis.get('total_feedbacks', 0)}")
        logger.info(f"{'=' * 70}")

        return {
            "elapsed_minutes": round(elapsed / 60, 1),
            "best_drivers": {
                k: [d["name"] for d in v[:3]]
                for k, v in (best_drivers or {}).items()
            },
            "feedback_analysis": feedback_analysis,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrape-count", type=int, default=100)
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--train-lr", type=float, default=5e-5)
    parser.add_argument("--train-rank", type=int, default=8)
    parser.add_argument("--continuous", action="store_true",
                       help="Run in continuous loop (every 30 min)")
    args = parser.parse_args()

    pipeline = AutonomousPipeline()

    if args.continuous:
        cycle = 1
        while True:
            logger.info(f"\n\n=== AUTONOMOUS CYCLE {cycle} ===\n")
            try:
                result = pipeline.run_cycle(
                    scrape_count=args.scrape_count,
                    skip_scrape=(cycle > 1),
                    train_steps=args.train_steps,
                    train_lr=args.train_lr,
                    train_rank=args.train_rank,
                )
                logger.info(f"Cycle {cycle} complete. Sleeping 30 min...")
            except Exception as e:
                logger.error(f"Cycle {cycle} failed: {e}")
            cycle += 1
            time.sleep(1800)
    else:
        pipeline.run_cycle(
            scrape_count=args.scrape_count,
            skip_scrape=args.skip_scrape,
            train_steps=args.train_steps,
            train_lr=args.train_lr,
            train_rank=args.train_rank,
        )


if __name__ == "__main__":
    main()
