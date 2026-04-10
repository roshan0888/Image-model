#!/usr/bin/env python3
"""
Run script for the Autonomous Training Pipeline.

Usage:
    # Full pipeline (collect data + train + evaluate + loop)
    python run.py

    # Skip data collection (use existing data)
    python run.py --skip-data

    # Quick test (1 cycle, skip data collection)
    python run.py --quick-test

    # Resume from checkpoint
    python run.py --resume checkpoints/

    # Dry run (check everything works without training)
    python run.py --dry-run

    # Only run data collection
    python run.py --data-only

    # Only run training (skip data and evaluation loop)
    python run.py --train-only

    # Custom number of cycles
    python run.py --cycles 5
"""

import os
import sys
import argparse
import logging

# Add training_engine to path
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ENGINE_DIR)

# Add LivePortrait to path
LP_DIR = os.path.join(os.path.dirname(ENGINE_DIR), "LivePortrait")
if os.path.exists(LP_DIR):
    sys.path.insert(0, LP_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run")


def get_config_path():
    return os.path.join(ENGINE_DIR, "configs", "pipeline_config.yaml")


def run_data_only():
    """Run only the data collection pipeline."""
    import yaml
    config_path = get_config_path()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create directories
    for key in ["dataset_dir", "raw_images_dir", "cleaned_dir", "paired_dir"]:
        os.makedirs(config["paths"][key], exist_ok=True)

    from data_engine.collector import FaceDataCollector
    from data_engine.cleaner import DataCleaner
    from data_engine.identity_cluster import IdentityClusterEngine, DatasetBalancer

    logger.info("=" * 60)
    logger.info("DATA PIPELINE ONLY")
    logger.info("=" * 60)

    # Collect
    logger.info("\n[1/4] Collecting...")
    collector = FaceDataCollector(config)
    stats = collector.collect(num_queries=50, images_per_query=30)
    logger.info(f"  Downloaded: {stats['total_downloaded']}")

    # Clean
    logger.info("\n[2/4] Cleaning...")
    cleaner = DataCleaner(config)
    clean_stats = cleaner.clean_all()
    logger.info(f"  Passed: {clean_stats['passed']}/{clean_stats['total_scanned']}")

    # Cluster
    logger.info("\n[3/4] Clustering...")
    cluster = IdentityClusterEngine(config)
    cluster_stats = cluster.cluster()
    logger.info(f"  Clusters: {cluster_stats['valid_clusters']}")

    # Balance
    logger.info("\n[4/4] Balance analysis...")
    balancer = DatasetBalancer(config)
    balance = balancer.analyze_balance()
    logger.info(f"  Distribution: {balance}")


def run_train_only():
    """Run only the training pipeline (assumes data exists)."""
    import yaml
    config_path = get_config_path()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for key in ["checkpoints_dir", "logs_dir", "experiments_dir"]:
        os.makedirs(config["paths"][key], exist_ok=True)

    from training.trainer import LivePortraitTrainer

    logger.info("=" * 60)
    logger.info("TRAINING ONLY")
    logger.info("=" * 60)

    trainer = LivePortraitTrainer(config)
    trainer.setup()
    results = trainer.train()
    logger.info(f"\nTraining complete: {results}")


def run_dry_run():
    """Check all components initialize correctly."""
    import yaml
    config_path = get_config_path()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("DRY RUN — Component Check")
    logger.info("=" * 60)

    # Check paths
    logger.info("\n[Paths]")
    for key, path in config["paths"].items():
        exists = os.path.exists(path)
        status = "OK" if exists else "WILL CREATE"
        logger.info(f"  {key}: {path} [{status}]")

    # Check LivePortrait
    logger.info("\n[LivePortrait]")
    lp_dir = config["paths"]["liveportrait_dir"]
    if os.path.exists(lp_dir):
        logger.info(f"  Directory: {lp_dir} [OK]")
        weights_dir = os.path.join(lp_dir, "pretrained_weights", "liveportrait")
        if os.path.exists(weights_dir):
            for d in os.listdir(weights_dir):
                full = os.path.join(weights_dir, d)
                if os.path.isdir(full):
                    files = os.listdir(full)
                    logger.info(f"  {d}/: {', '.join(files)}")
    else:
        logger.error(f"  LivePortrait NOT FOUND at {lp_dir}")

    # Check InsightFace
    logger.info("\n[InsightFace]")
    antelope_dir = config["paths"]["antelopev2_dir"]
    if os.path.exists(antelope_dir):
        logger.info(f"  Directory: {antelope_dir} [OK]")
    else:
        logger.error(f"  Antelopev2 NOT FOUND at {antelope_dir}")

    # Check GPU
    logger.info("\n[GPU]")
    import torch
    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("  No CUDA GPU available!")

    # Check packages
    logger.info("\n[Packages]")
    packages = ["torch", "cv2", "insightface", "lpips", "sklearn", "yaml",
                "icrawler", "numpy", "onnxruntime"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            logger.info(f"  {pkg}: {ver} [OK]")
        except ImportError:
            logger.error(f"  {pkg}: NOT INSTALLED")

    # Try loading LP modules
    logger.info("\n[LivePortrait Modules]")
    try:
        sys.path.insert(0, lp_dir)
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        wrapper = LivePortraitWrapper(InferenceConfig())

        for name in ["appearance_feature_extractor", "motion_extractor",
                     "warping_module", "spade_generator"]:
            module = getattr(wrapper, name, None)
            if module is not None:
                params = sum(p.numel() for p in module.parameters())
                logger.info(f"  {name}: {params:,} params [OK]")
            else:
                logger.error(f"  {name}: NOT LOADED")

        # Test LoRA injection
        logger.info("\n[LoRA Injection Test]")
        from training.lora_modules import inject_lora, count_lora_parameters
        targets = config["training"]["lora"]["target_modules"].get("spade_generator", [])
        if targets:
            wrapper.spade_generator, injected = inject_lora(
                wrapper.spade_generator, targets,
                rank=config["training"]["lora"]["rank"],
                alpha=config["training"]["lora"]["alpha"],
            )
            trainable, total = count_lora_parameters(wrapper.spade_generator)
            logger.info(f"  SPADE Generator: {len(injected)} LoRA layers, "
                       f"{trainable:,} trainable / {total:,} total "
                       f"({trainable/total*100:.2f}%)")

        logger.info("\n" + "=" * 60)
        logger.info("DRY RUN COMPLETE — All systems ready!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"  LivePortrait load failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Autonomous Training Pipeline Runner")
    parser.add_argument("--dry-run", action="store_true",
                       help="Check all components without running")
    parser.add_argument("--data-only", action="store_true",
                       help="Run only data pipeline")
    parser.add_argument("--train-only", action="store_true",
                       help="Run only training (assumes data exists)")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data collection in full pipeline")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--cycles", type=int, default=None,
                       help="Number of training cycles")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test: 1 cycle, skip data")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    if args.data_only:
        run_data_only()
        return

    if args.train_only:
        run_train_only()
        return

    # Full autonomous pipeline
    from autonomous_loop import AutonomousTrainingPipeline

    config_path = get_config_path()
    pipeline = AutonomousTrainingPipeline(config_path)

    if args.quick_test:
        pipeline.loop_cfg["max_cycles"] = 1
        args.skip_data = True

    if args.cycles:
        pipeline.loop_cfg["max_cycles"] = args.cycles

    pipeline.run(
        skip_data_collection=args.skip_data,
        resume_training=args.resume,
    )


if __name__ == "__main__":
    main()
