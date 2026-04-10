#!/usr/bin/env python3
"""
autoloop.py — The Autonomous Experiment Loop

Inspired by Karpathy's autoresearch:
  "LOOP FOREVER. NEVER STOP. The human might be asleep."

Each iteration:
  1. Pick next experiment config (systematic search)
  2. Write to experiment.py
  3. Run evaluation on 20 test faces
  4. If improved → keep. If worse → discard.
  5. Log to results.tsv
  6. Repeat forever

Usage:
  python autoloop.py              # Run forever
  python autoloop.py --max 50     # Run 50 experiments
"""

import os
import sys
import json
import time
import logging
import argparse
import importlib
from copy import deepcopy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [autoloop] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoloop.log")),
    ],
)
logger = logging.getLogger("autoloop")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

LP_DRIVER_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "LivePortrait", "assets", "examples", "driving")
PHOTO_DRIVER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "photoshoot", "drivers")

# ═══════════════════════════════════════════════════
# EXPERIMENT SPACE — all configurations to try
# ═══════════════════════════════════════════════════

def get_all_drivers():
    """Collect all available driver paths."""
    drivers = []

    # LP built-in
    for f in ["d12.jpg", "d30.jpg"]:
        p = os.path.join(LP_DRIVER_DIR, f)
        if os.path.exists(p):
            drivers.append(p)

    # Scraped drivers (top 5 per category)
    for cat in ["subtle", "photoshoot", "natural"]:
        cat_dir = os.path.join(PHOTO_DRIVER_DIR, cat)
        if os.path.exists(cat_dir):
            files = sorted([f for f in os.listdir(cat_dir) if f.endswith(".jpg")])[:5]
            for f in files:
                drivers.append(os.path.join(cat_dir, f))

    return drivers


def generate_experiment_queue():
    """Generate a systematic queue of experiments to try.

    Strategy: vary ONE parameter at a time from baseline, then combine winners.
    """
    drivers = get_all_drivers()
    baseline = {
        "driver": os.path.join(LP_DRIVER_DIR, "d12.jpg"),
        "multiplier": 1.5,
        "animation_region": "lip",
        "driving_option": "expression-friendly",
        "flag_stitching": True,
        "flag_relative_motion": True,
        "flag_eye_retargeting": False,
        "flag_lip_retargeting": False,
    }

    experiments = []

    # 0. Baseline
    experiments.append((deepcopy(baseline), "baseline: lip mode, d12.jpg, 1.5x"))

    # 1. Vary multiplier (keep everything else baseline)
    for mult in [0.5, 0.8, 1.0, 1.2, 2.0, 2.5, 3.0]:
        cfg = deepcopy(baseline)
        cfg["multiplier"] = mult
        experiments.append((cfg, f"multiplier={mult}"))

    # 2. Vary animation_region
    for region in ["all", "exp"]:
        cfg = deepcopy(baseline)
        cfg["animation_region"] = region
        experiments.append((cfg, f"region={region}, mult=1.5"))

    # 3. Vary animation_region with adjusted multiplier
    for region, mult in [("all", 0.7), ("all", 0.5), ("exp", 1.0), ("exp", 2.0)]:
        cfg = deepcopy(baseline)
        cfg["animation_region"] = region
        cfg["multiplier"] = mult
        experiments.append((cfg, f"region={region}, mult={mult}"))

    # 4. Try retargeting modes
    for eye, lip in [(True, True), (True, False), (False, True)]:
        cfg = deepcopy(baseline)
        cfg["flag_eye_retargeting"] = eye
        cfg["flag_lip_retargeting"] = lip
        experiments.append((cfg, f"retarget eye={eye} lip={lip}"))

    # 5. Vary driver (keep lip mode, 1.5x)
    for drv in drivers:
        if drv == baseline["driver"]:
            continue
        cfg = deepcopy(baseline)
        cfg["driver"] = drv
        dname = os.path.basename(drv)
        experiments.append((cfg, f"driver={dname}, lip 1.5x"))

    # 6. Try best drivers with different multipliers
    for drv in drivers[:5]:
        for mult in [0.8, 1.0, 2.0]:
            cfg = deepcopy(baseline)
            cfg["driver"] = drv
            cfg["multiplier"] = mult
            dname = os.path.basename(drv)
            experiments.append((cfg, f"driver={dname}, mult={mult}"))

    # 7. Try driving_option="pose-friendly"
    cfg = deepcopy(baseline)
    cfg["driving_option"] = "pose-friendly"
    experiments.append((cfg, "pose-friendly mode"))

    # 8. Try no stitching
    cfg = deepcopy(baseline)
    cfg["flag_stitching"] = False
    experiments.append((cfg, "no stitching"))

    # 9. Try no relative motion
    cfg = deepcopy(baseline)
    cfg["flag_relative_motion"] = False
    experiments.append((cfg, "absolute motion"))

    # 10. Retargeting + lip mode combos
    for mult in [1.0, 1.5, 2.0]:
        cfg = deepcopy(baseline)
        cfg["flag_eye_retargeting"] = True
        cfg["flag_lip_retargeting"] = True
        cfg["multiplier"] = mult
        experiments.append((cfg, f"retarget+lip, mult={mult}"))

    return experiments


# ═══════════════════════════════════════════════════
# RESULTS LOGGING
# ═══════════════════════════════════════════════════

RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.tsv")

def init_results():
    """Create results.tsv with header if not exists."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("experiment\tavg_identity\tmin_identity\tmouth_disp\teye_disp\tcomposite\tpass_rate\tstatus\tdescription\n")

def log_result(exp_num, metrics, status, description):
    """Append one result row."""
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{exp_num}\t"
            f"{metrics['avg_identity']:.6f}\t"
            f"{metrics['min_identity']:.6f}\t"
            f"{metrics['avg_mouth_disp']:.6f}\t"
            f"{metrics['avg_eye_disp']:.6f}\t"
            f"{metrics['composite']:.6f}\t"
            f"{metrics['pass_rate_93']:.4f}\t"
            f"{status}\t"
            f"{description}\n"
        )


# ═══════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="Max experiments (0=infinite)")
    args = parser.parse_args()

    # Load models once
    from prepare import load_models, evaluate
    logger.info("Loading models...")
    load_models()
    logger.info("Models ready")

    # Generate experiment queue
    experiments = generate_experiment_queue()
    logger.info(f"Generated {len(experiments)} experiments to try")

    # Init results
    init_results()

    # Track best
    best_composite = 0.0
    best_config = None
    best_description = ""

    logger.info("\n" + "=" * 70)
    logger.info("AUTONOMOUS EXPERIMENT LOOP — STARTING")
    logger.info(f"Experiments: {len(experiments)}")
    logger.info(f"Test faces: 20")
    logger.info(f"NEVER STOP until interrupted")
    logger.info("=" * 70 + "\n")

    for i, (config, description) in enumerate(experiments):
        if args.max > 0 and i >= args.max:
            break

        exp_num = i + 1
        logger.info(f"\n{'─'*60}")
        logger.info(f"EXPERIMENT {exp_num}/{len(experiments)}: {description}")
        logger.info(f"{'─'*60}")

        # Run evaluation
        try:
            metrics = evaluate(config)
        except Exception as e:
            logger.error(f"  CRASH: {e}")
            log_result(exp_num, {
                "avg_identity": 0, "min_identity": 0, "avg_mouth_disp": 0,
                "avg_eye_disp": 0, "composite": 0, "pass_rate_93": 0,
            }, "crash", description)
            continue

        # Check validity
        valid = True
        if metrics["avg_mouth_disp"] < 0.010 and "retarget" not in description:
            valid = False
            reason = "mouth_disp too low (no visible smile)"
        elif metrics["failures"] > 5:
            valid = False
            reason = f"{metrics['failures']} failures out of 20"

        # Compare with best
        if valid and metrics["composite"] > best_composite:
            status = "keep"
            best_composite = metrics["composite"]
            best_config = config
            best_description = description
            logger.info(f"  ✓ NEW BEST: composite={metrics['composite']:.6f}")
        elif valid:
            status = "discard"
            logger.info(f"  ✗ No improvement (composite={metrics['composite']:.6f} vs best={best_composite:.6f})")
        else:
            status = "invalid"
            logger.info(f"  ✗ Invalid: {reason}")

        # Log
        log_result(exp_num, metrics, status, description)

        logger.info(
            f"  identity={metrics['avg_identity']:.4f} "
            f"mouth={metrics['avg_mouth_disp']:.4f} "
            f"eyes={metrics['avg_eye_disp']:.4f} "
            f"composite={metrics['composite']:.4f} "
            f"pass_rate={metrics['pass_rate_93']:.0%} "
            f"time={metrics['elapsed_seconds']:.0f}s "
            f"→ {status}"
        )

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT LOOP COMPLETE")
    logger.info(f"  Total experiments: {len(experiments)}")
    logger.info(f"  Best composite: {best_composite:.6f}")
    logger.info(f"  Best config: {best_description}")
    if best_config:
        logger.info(f"  Best settings: {json.dumps(best_config, indent=2, default=str)}")
    logger.info(f"  Results: {RESULTS_FILE}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
