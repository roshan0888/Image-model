"""
Image Feeder — Watches scrape directories and feeds new images into training pipeline.
Runs in background. Every 30 min, picks up new scraped images, runs quality gate,
copies passing images into cleaned/smile/ so the training loop uses them next cycle.

Run: python feed_new_images.py &
"""
import sys, time, shutil, logging, cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [FEEDER] %(message)s")
log = logging.getLogger(__name__)

# Watch these directories for new images
WATCH_DIRS = [
    ROOT / "raw_data/starnow_ugc_models/bulk_raw",
    ROOT / "raw_data/bulk_scrape/raw_downloads",
]

# Feed into this directory (the training loop reads from here)
FEED_DIR  = ROOT / "raw_data/cleaned/smile"
FEED_DIR.mkdir(parents=True, exist_ok=True)

SEEN_FILE = ROOT / ".feeder_seen.txt"
CHECK_EVERY = 1800  # 30 minutes


def load_seen() -> set:
    if SEEN_FILE.exists():
        return set(SEEN_FILE.read_text().splitlines())
    return set()


def save_seen(seen: set):
    SEEN_FILE.write_text("\n".join(seen))


def quick_quality_check(img_path: Path, face_app) -> bool:
    """Fast quality check — only frontal, single face, sharp."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        h, w = img.shape[:2]
        if min(h, w) < 256:
            return False

        faces = face_app.get(img)
        if len(faces) != 1:
            return False

        face = faces[0]
        if face.det_score < 0.65:
            return False

        # Frontal only (yaw < 25°)
        if hasattr(face, 'pose') and face.pose is not None:
            yaw = abs(float(face.pose[1]))
            if yaw > 25:
                return False

        # Face big enough
        bbox = face.bbox.astype(int)
        if min(bbox[2]-bbox[0], bbox[3]-bbox[1]) < 100:
            return False

        # Sharpness
        crop = img[max(0,bbox[1]):bbox[3], max(0,bbox[0]):bbox[2]]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
            return False

        return True
    except Exception:
        return False


def main():
    # Load InsightFace
    from insightface.app import FaceAnalysis
    log.info("Loading InsightFace for quality checks...")
    face_app = FaceAnalysis(name='antelopev2', root=str(ROOT / 'MagicFace/third_party_files'))
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    log.info("Ready. Watching for new images every 30 min...")

    while True:
        seen = load_seen()
        added = 0
        checked = 0

        for watch_dir in WATCH_DIRS:
            if not watch_dir.exists():
                continue

            for img_path in watch_dir.glob("*"):
                if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                    continue
                if str(img_path) in seen:
                    continue

                seen.add(str(img_path))
                checked += 1

                if quick_quality_check(img_path, face_app):
                    dst = FEED_DIR / f"scraped_{img_path.stem}{img_path.suffix}"
                    if not dst.exists():
                        shutil.copy2(str(img_path), str(dst))
                        added += 1

        save_seen(seen)

        if checked > 0:
            total_clean = len(list(FEED_DIR.glob("*")))
            log.info(f"Checked {checked} new images → {added} passed quality → {total_clean} total in training pool")
        else:
            log.info(f"No new images found yet. Training pool: {len(list(FEED_DIR.glob('*')))} images")

        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    main()
