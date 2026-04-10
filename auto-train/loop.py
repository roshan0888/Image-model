"""
Autonomous Production Loop
==========================
Runs forever: SCRAPE → CLEAN → PAIR → TRAIN → VALIDATE
If validation < target, scrapes more data and repeats.
Stops only when identity >= 99.5% (production level).

Usage:
    python loop.py                    # Full loop, runs until 99.5%
    python loop.py --target 0.98      # Lower target for quick test
    python loop.py --skip-scrape      # Use existing data, skip scraping
    python loop.py --scrape-only      # Just scrape, don't train
"""

import os, sys, cv2, json, time, shutil, logging, argparse, importlib.util
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Directories ───────────────────────────────────────────────────────────────
RAW   = ROOT / "raw_data/model_photos"
CLEAN = ROOT / "raw_data/cleaned"
PAIRS = ROOT / "raw_data/pairs"
CKPTS = ROOT / "training_engine/checkpoints"
LOGS  = ROOT / "training_engine/logs"
for d in [RAW, CLEAN, PAIRS, CKPTS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LOOP] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS / "loop.log"),
    ],
)
log = logging.getLogger(__name__)

EXPRESSIONS = ["smile", "open_smile", "neutral"]
HISTORY_FILE = LOGS / "loop_history.json"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SCRAPE
# ═══════════════════════════════════════════════════════════════════════════════

def step_scrape(target: int = 500) -> dict:
    log.info(f"SCRAPE → target {target} images per expression")
    from training_engine.data_engine.model_photo_scraper import ModelPhotoScraper
    scraper = ModelPhotoScraper(output_dir=str(RAW))
    stats = scraper.scrape_all(target_per_expression=target)
    total = sum(stats.values())
    log.info(f"SCRAPE done: {total} total images")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN
# ═══════════════════════════════════════════════════════════════════════════════

def step_clean() -> dict:
    log.info("CLEAN → face detect, blur, pose, quality filter")

    # Load InsightFace
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="antelopev2",
                           root=str(ROOT / "MagicFace/third_party_files"))
        app.prepare(ctx_id=0, det_size=(640, 640))
        use_if = True
        log.info("  InsightFace loaded")
    except Exception as e:
        log.warning(f"  InsightFace unavailable ({e}), using OpenCV")
        use_if = False
        cascade_xml = str(ROOT / "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_xml):
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                cascade_xml)
        face_cas = cv2.CascadeClassifier(cascade_xml)

    accepted_total = rejected_total = 0

    for expr in EXPRESSIONS:
        src = RAW / expr
        dst = CLEAN / expr
        dst.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            continue

        files = list(src.glob("*.jpg")) + list(src.glob("*.jpeg")) + \
                list(src.glob("*.png")) + list(src.glob("*.webp"))
        acc = 0
        rej: dict = {}

        for fpath in files:
            dst_path = dst / fpath.name
            if dst_path.exists():       # already cleaned from prior run
                acc += 1
                accepted_total += 1
                continue
            reason = None
            try:
                img = cv2.imread(str(fpath))
                if img is None:
                    reason = "unreadable"
                else:
                    h, w = img.shape[:2]
                    if min(h, w) < 200:
                        reason = "too_small"
                    else:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
                            reason = "blurry"
                        elif use_if:
                            faces = app.get(img)
                            if len(faces) == 0:       reason = "no_face"
                            elif len(faces) > 1:      reason = "multi_face"
                            else:
                                face = faces[0]
                                if face.det_score < 0.6: reason = "low_conf"
                                else:
                                    b = face.bbox.astype(int)
                                    if (b[2]-b[0]) < 80 or (b[3]-b[1]) < 80:
                                        reason = "face_small"
                                    elif hasattr(face, "age") and face.age is not None and face.age < 18:
                                        reason = "underage"
                                    elif hasattr(face, "pose") and face.pose is not None:
                                        p, y, _ = face.pose
                                        # FRONTAL ONLY — yaw < 25° required for lip-mode training
                                        if abs(y) > 25 or abs(p) > 25:
                                            reason = "non_frontal"
                        else:
                            gs = cv2.resize(gray, (640, 640))
                            fc = face_cas.detectMultiScale(gs, 1.1, 5, minSize=(80, 80))
                            if len(fc) == 0:  reason = "no_face"
                            elif len(fc) > 1: reason = "multi_face"

                        if reason is None:
                            brt = gray.mean()
                            if brt < 30 or brt > 240 or gray.std() < 20:
                                reason = "low_quality"

            except Exception:
                reason = "error"

            if reason:
                rej[reason] = rej.get(reason, 0) + 1
                rejected_total += 1
            else:
                shutil.copy2(str(fpath), str(dst_path))
                acc += 1
                accepted_total += 1

        log.info(f"  {expr}: {acc} kept | rejected={rej}")

    rate = accepted_total / max(1, accepted_total + rejected_total) * 100
    log.info(f"CLEAN done: {accepted_total} kept ({rate:.0f}% pass rate)")
    return {"accepted": accepted_total, "rejected": rejected_total}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PAIR  (self-supervised via LP)
# ═══════════════════════════════════════════════════════════════════════════════

def step_pair(max_per_expr: int = 40) -> int:
    """Run LivePortrait on clean images to create (source, expression) pairs.

    PRODUCTION MODE: Uses region='lip' + use_retargeting=False with strong drivers.
    This actually produces visible smile changes (verified in LP_ALL_MODES_COMPARISON).
    Identity threshold lowered to 0.88 because real expression changes cost ~3% identity.
    """
    log.info("PAIR → generating LIP-MODE training pairs (production config)")

    # Load pipeline
    spec = importlib.util.spec_from_file_location("npipe", ROOT / "natural_pipeline.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    pipeline = mod.NaturalExpressionPipeline()

    # Production driver library — verified to produce visible expression changes
    LP_DIR = Path("/teamspace/studios/this_studio/LivePortrait/assets/examples/driving")
    DRIVERS = {
        "smile":      (LP_DIR / "d12.jpg", 1.3),   # big_smile, strong but balanced
        "open_smile": (LP_DIR / "d12.jpg", 1.5),   # bigger smile via higher multiplier
    }

    PAIRS.mkdir(parents=True, exist_ok=True)
    existing = len(list(PAIRS.glob("*_src.jpg")))
    count = existing

    for expr in ["smile", "open_smile"]:
        if expr not in DRIVERS:
            continue
        driver_path, mult = DRIVERS[expr]
        if not driver_path.exists():
            log.warning(f"  driver {driver_path} missing, skipping {expr}")
            continue

        src_dir = CLEAN / expr if (CLEAN / expr).exists() else CLEAN / "smile"
        if not src_dir.exists():
            continue

        candidates = list(src_dir.glob("*.jpg"))[:max_per_expr]
        for src in candidates:
            pair_src = PAIRS / f"p{count:05d}_src.jpg"
            pair_drv = PAIRS / f"p{count:05d}_drv_{expr}.jpg"
            if pair_src.exists():
                count += 1
                continue
            try:
                # Use production lip-mode: region='lip', use_retargeting=False
                out_img = pipeline._run_lp(
                    str(src), str(driver_path),
                    multiplier=mult, region="lip", use_retargeting=False,
                )
                if out_img is None:
                    continue

                # Measure identity manually (lip mode bypasses pipeline.process metrics)
                src_img = cv2.imread(str(src))
                if src_img is None:
                    continue
                f1 = pipeline.face_analyzer.get(src_img)
                f2 = pipeline.face_analyzer.get(out_img)
                if not f1 or not f2:
                    continue
                score = float(np.dot(f1[0].normed_embedding, f2[0].normed_embedding))

                # Threshold: 0.88 (lip mode trades a bit of identity for real expression)
                if score >= 0.88:
                    cv2.imwrite(str(pair_src), src_img)
                    cv2.imwrite(str(pair_drv), out_img)
                    count += 1
                    if count % 10 == 0:
                        log.info(f"  {count} pairs so far (identity={score:.3f}, lip-mode)")
            except Exception as e:
                log.debug(f"  pair failed {src.name}/{expr}: {e}")

    new_pairs = count - existing
    log.info(f"PAIR done: {count} total pairs ({new_pairs} new, lip-mode)")

    # Write training_pairs.jsonl manifest for the trainer's dataloader
    manifest_path = PAIRS / "training_pairs.jsonl"
    written = 0
    with open(manifest_path, "w") as mf:
        for src_file in sorted(PAIRS.glob("p*_src.jpg")):
            stem = src_file.stem.replace("_src", "")
            for expr_name in ["smile", "open_smile"]:
                drv_file = PAIRS / f"{stem}_drv_{expr_name}.jpg"
                if drv_file.exists():
                    entry = {
                        "source_path":       str(src_file),
                        "target_path":       str(drv_file),
                        "target_expression": expr_name,
                    }
                    mf.write(json.dumps(entry) + "\n")
                    written += 1
    log.info(f"PAIR manifest: {manifest_path} ({written} entries)")
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

def step_train(num_steps: int = 5000, cycle: int = 0) -> dict:
    log.info(f"TRAIN → {num_steps} steps (cycle {cycle})")
    from training_engine.training.trainer import LivePortraitTrainer

    cfg = {
        "paths": {
            "paired_dir":      str(PAIRS),
            "checkpoints_dir": str(CKPTS),
            "logs_dir":        str(LOGS),
            "experiments_dir": str(LOGS / "experiments"),
            "liveportrait_dir": str(ROOT / "LivePortrait"),
            "antelopev2_dir":  str(ROOT / "MagicFace/third_party_files"),
        },
        "training": {
            "cycle": cycle,
            "schedule": {
                "learning_rate":          1e-4,
                "total_steps":            num_steps,
                "lr_scheduler":           "cosine",
                "mixed_precision":        "fp16",
                "warmup_steps":           int(num_steps * 0.03),
                "batch_size":             1,    # T4 OOM at >1, use grad accum
                "eval_every":             50,   # log to TB every 50 steps for live view
                "save_every":             500,
                "gradient_accumulation":  16,   # effective batch = 16
                "max_grad_norm":          1.0,
            },
            "lora": {
                "rank":    8,    # reduced from 16 for T4 VRAM
                "alpha":   16,
                "dropout": 0.05,
                "target_modules": {
                    "warping_network":       ["to_q", "to_k", "to_v", "to_out.0"],
                    "spade_generator":       ["conv_0", "conv_1", "conv_s"],
                    "motion_extractor":      [],   # disabled — frees ~2GB
                    "stitching_retargeting": [],   # disabled — frees ~1GB
                },
            },
            "losses": {
                "identity_loss":     5.0,
                "expression_loss":   4.0,
                "perceptual_loss":   1.5,
                "pixel_loss":        1.0,
                "regularization":    1e-4,
            },
            "early_stopping": {
                "target_identity": 0.98,    # raise bar so it doesn't trigger at 0.95
                "min_delta":       0.0005,  # smaller threshold for "improvement"
                "patience":        30,      # 30 evals = 1500 steps before giving up
            },
        },
    }
    trainer = LivePortraitTrainer(cfg)
    trainer.setup()
    metrics = trainer.train()
    best = metrics.get("best_identity", 0)
    log.info(f"TRAIN done: best_identity={best:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VALIDATE
# ═══════════════════════════════════════════════════════════════════════════════

def step_validate() -> dict:
    """Run pipeline on held-out test images and return avg identity score."""
    log.info("VALIDATE → running on test images")

    spec = importlib.util.spec_from_file_location("npipe", ROOT / "natural_pipeline.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    pipeline = mod.NaturalExpressionPipeline()

    # Use LP bundled examples as held-out test set
    test_dir = ROOT / "LivePortrait/assets/examples/source"
    test_imgs = list(test_dir.glob("*.jpg"))[:5] if test_dir.exists() else []
    if not test_imgs:
        test_imgs = list((ROOT / "output").glob("*.jpg"))[:5]

    scores: dict = {"smile": [], "open_smile": []}
    for img in test_imgs[:4]:
        for expr in ["smile", "open_smile"]:
            try:
                result = pipeline.process(str(img), expr)
                s = result.get("identity_score", 0)
                scores[expr].append(s)
                log.info(f"  {img.name} + {expr}: {s:.4f}")
            except Exception as e:
                log.debug(f"  validate failed {img.name}/{expr}: {e}")

    avg_by_expr = {e: (sum(v)/len(v) if v else 0) for e, v in scores.items()}
    all_scores  = [s for v in scores.values() for s in v]
    avg = sum(all_scores) / max(1, len(all_scores))

    log.info(f"VALIDATE done: avg_identity={avg:.4f} | {avg_by_expr}")
    return {"avg_identity": avg, "by_expression": avg_by_expr, "n_samples": len(all_scores)}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run(args):
    target      = args.target
    max_cycles  = args.max_cycles
    skip_scrape = args.skip_scrape

    history = []
    best_ever   = 0.0
    cycle       = 0

    log.info("=" * 60)
    log.info("AUTONOMOUS PRODUCTION LOOP STARTED")
    log.info(f"  Target identity : {target*100:.1f}%")
    log.info(f"  Max cycles      : {max_cycles}")
    log.info("=" * 60)

    while cycle < max_cycles:
        cycle += 1
        t0 = time.time()

        log.info(f"\n{'#'*60}")
        log.info(f"CYCLE {cycle} / {max_cycles}")
        log.info(f"{'#'*60}")

        rec: dict = {"cycle": cycle, "ts": datetime.now().isoformat()}

        # ── SCRAPE ────────────────────────────────────────────────────────────
        if not skip_scrape:
            # First cycle: 500/expr. Each retry: +200 more
            target_imgs = 500 + (cycle - 1) * 200
            rec["scrape"] = step_scrape(target=target_imgs)

        # ── CLEAN ─────────────────────────────────────────────────────────────
        rec["clean"] = step_clean()

        # ── PAIR ──────────────────────────────────────────────────────────────
        n_pairs = step_pair(max_per_expr=50 + (cycle - 1) * 20)
        rec["pairs"] = n_pairs

        if n_pairs < 5:
            log.warning("  Too few pairs — skipping training this cycle")
            history.append(rec)
            _save_history(history)
            continue

        # ── TRAIN ─────────────────────────────────────────────────────────────
        # Bigger per-cycle budget so plateaus get crushed by sheer training time
        steps = min(15000 + (cycle - 1) * 5000, 50000)
        rec["train"] = step_train(num_steps=steps, cycle=cycle)

        # ── VALIDATE ──────────────────────────────────────────────────────────
        val = step_validate()
        rec["validate"] = val
        avg_id = val["avg_identity"]

        if avg_id > best_ever:
            best_ever = avg_id
            log.info(f"  ★ New best: {best_ever:.4f}")

        elapsed = (time.time() - t0) / 60
        log.info(f"\n{'─'*50}")
        log.info(f"CYCLE {cycle} DONE in {elapsed:.1f} min")
        log.info(f"  Identity  : {avg_id:.4f}")
        log.info(f"  Best ever : {best_ever:.4f}")
        log.info(f"  Target    : {target}")
        log.info(f"  Gap       : {(target - avg_id)*100:.2f}%")
        log.info(f"{'─'*50}\n")

        history.append(rec)
        _save_history(history)

        # ── DECISION ──────────────────────────────────────────────────────────
        if avg_id >= target:
            log.info(f"\n{'='*60}")
            log.info(f"TARGET REACHED: {avg_id:.4f} >= {target}")
            log.info(f"Production-ready after {cycle} cycles.")
            log.info(f"{'='*60}")
            break

        gap = target - avg_id
        log.info(f"  Not there yet ({gap*100:.1f}% gap). Scraping more data...")
        time.sleep(5)

    log.info(f"\nFINAL RESULT: best_identity={best_ever:.4f} | cycles={cycle}")
    return history


def _save_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target",       type=float, default=0.995,
                   help="Identity score target (default 0.995 = production)")
    p.add_argument("--max-cycles",   type=int,   default=20,
                   help="Max loop iterations before giving up")
    p.add_argument("--skip-scrape",  action="store_true",
                   help="Skip scraping, use already-downloaded images")
    p.add_argument("--scrape-only",  action="store_true",
                   help="Only scrape, then exit")
    args = p.parse_args()

    if args.scrape_only:
        step_scrape(target=500)
    else:
        run(args)
