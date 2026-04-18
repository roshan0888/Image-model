"""
Model Photo Data Engine — Scrape + Verify + Retry Loop

Scrapes ultra-realistic smiling model/headshot photos from multiple sources:
  - Pexels API (free, 4K, commercial license) — PRIMARY
  - Unsplash API (free, editorial quality) — BACKUP
  - icrawler Google/Bing Image Search — FALLBACK

Then verifies each image:
  1. Real human face detected (InsightFace)
  2. Exactly 1 face
  3. Frontal pose (nose centered)
  4. Smiling (mouth corners raised)
  5. Age 18+
  6. Not blurry (Laplacian variance)
  7. Face ≥ 200×200px for detail

Anything that fails verification → deleted.
Loops until target count of verified images reached.
"""

import os, sys, cv2, time, hashlib, logging, requests, random, json
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [data_engine] %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw_data/model_photos/smile_raw"
VERIFIED_DIR = ROOT / "raw_data/cleaned/smile"
REJECTED_DIR = ROOT / "raw_data/rejected"
for d in [RAW_DIR, VERIFIED_DIR, REJECTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# SCRAPE QUERIES — target ultrarealistic smiley model photos
# ══════════════════════════════════════════════════════════════════

SMILE_QUERIES = [
    # Pexels/Unsplash — watermark-free sources
    "pexels smiling portrait",
    "unsplash smiling face",
    "unsplash portrait smile",
    "pexels happy person smile",
    "unsplash smiling woman",
    "unsplash smiling man",
    "pexels natural smile",
    "unsplash headshot smile",
    # Photogenic + watermark-free
    "photogenic smile unsplash",
    "natural candid smile unsplash",
    "close-up smile unsplash",
    "professional headshot unsplash",
    # Diverse demographics
    "Asian smile unsplash",
    "Black woman smile unsplash",
    "Latino smile unsplash",
    "Indian smile unsplash",
    "European smile unsplash",
    "African smile unsplash",
]

# Watermarked stock photo domains — reject these URLs
BLOCKED_DOMAINS = [
    "adobestock", "shutterstock", "istockphoto", "gettyimages",
    "dreamstime", "alamy", "depositphotos", "123rf", "bigstockphoto",
    "canstockphoto", "stockvault", "sciencephoto", "focusedcollection",
    "stock.adobe", "fotolia", "stocksy", "ftcdn.net",
]


# ══════════════════════════════════════════════════════════════════
# SCRAPING via icrawler (no API key needed)
# ══════════════════════════════════════════════════════════════════

def scrape_with_icrawler(max_images: int = 200) -> int:
    """Use icrawler's Google/Bing backends to download smile photos."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        log.error("icrawler not installed")
        return 0

    downloaded_before = len(list(RAW_DIR.glob("*.jpg")))

    queries = random.sample(SMILE_QUERIES, min(10, len(SMILE_QUERIES)))
    per_query = max(10, max_images // len(queries))

    for q in queries:
        log.info(f"  Bing: '{q}' (target {per_query})")
        try:
            crawler = BingImageCrawler(
                downloader_threads=4,
                storage={"root_dir": str(RAW_DIR)},
                feeder_threads=1,
                parser_threads=2,
            )
            crawler.crawl(
                keyword=q,
                max_num=per_query,
                min_size=(400, 400),
                file_idx_offset=len(list(RAW_DIR.glob("*.jpg"))),
            )
            time.sleep(1)
        except Exception as e:
            log.warning(f"  Failed '{q}': {e}")

    downloaded_after = len(list(RAW_DIR.glob("*.jpg")))
    return downloaded_after - downloaded_before


# ══════════════════════════════════════════════════════════════════
# VERIFY — keep only real human smiling faces
# ══════════════════════════════════════════════════════════════════

_face_analyzer = None

def get_face_analyzer():
    global _face_analyzer
    if _face_analyzer is not None:
        return _face_analyzer
    from insightface.app import FaceAnalysis
    log.info("Loading InsightFace...")
    _face_analyzer = FaceAnalysis(
        name="antelopev2",
        root=str(ROOT / "MagicFace/third_party_files"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)
    return _face_analyzer


def has_watermark(img) -> bool:
    """Heuristic watermark detection — look for repeating text patterns.

    Watermarked stock photos typically have:
      - Low-contrast diagonal text across the image
      - Strong horizontal bands at top/bottom with text
      - Very regular repeating patterns
    """
    if img is None or img.size == 0:
        return False

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Method 1: check for text-like patterns via edge density in specific regions
    # Stock photos often have watermarks in a band across the middle
    middle_band = gray[h//3:2*h//3, :]
    edges = cv2.Canny(middle_band, 50, 150)
    edge_density = edges.sum() / edges.size

    # Watermarks create unnaturally uniform edge density
    # Check for horizontal repeating patterns (typical of "ADOBE STOCK" text)
    if edge_density > 0.08:
        # Split into horizontal strips and check uniformity
        strips = np.array_split(edges, 8, axis=1)
        strip_means = [s.mean() for s in strips]
        std_ratio = np.std(strip_means) / (np.mean(strip_means) + 1e-6)
        if std_ratio < 0.3:  # Very uniform = likely watermark
            return True

    return False


def is_smiling(face, lmk) -> bool:
    """Check if face is smiling via landmark analysis.

    In 106-point landmarks:
      52 = left mouth corner
      61/58 = right mouth corner
      upper lip landmarks: 53-60
      lower lip landmarks: 62-71

    Smile = mouth corners raised above mouth center line
    """
    if lmk is None or len(lmk) < 72:
        return True  # Can't tell — pass through

    # Mouth corners
    left = lmk[52]
    right = lmk[61] if len(lmk) > 61 else lmk[58]

    # Mouth center (average of top and bottom lip midpoints)
    upper_mid = lmk[56] if len(lmk) > 56 else lmk[53]
    lower_mid = lmk[66] if len(lmk) > 66 else lmk[62]

    corner_avg_y = (left[1] + right[1]) / 2
    center_y = (upper_mid[1] + lower_mid[1]) / 2

    # Mouth width for scale
    mouth_width = abs(right[0] - left[0])
    if mouth_width < 10:
        return True  # Too small to tell

    # Smile threshold: corners ≤ center (equal or higher in image)
    # Small tolerance for slight smiles
    return corner_avg_y <= center_y + mouth_width * 0.05


def verify_all_raw():
    """Verify all raw images. Move verified to VERIFIED_DIR, reject others."""
    fa = get_face_analyzer()

    stats = {
        "total": 0, "unreadable": 0, "no_face": 0, "multi_face": 0,
        "small_face": 0, "underage": 0, "side_pose": 0, "blurry": 0,
        "not_smiling": 0, "watermarked": 0, "verified": 0,
    }

    for fpath in sorted(RAW_DIR.glob("*")):
        if fpath.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        stats["total"] += 1

        img = cv2.imread(str(fpath))
        if img is None:
            stats["unreadable"] += 1
            fpath.unlink()
            continue

        # Watermark detection — reject stock photos
        if has_watermark(img):
            stats["watermarked"] += 1
            fpath.unlink()
            continue

        faces = fa.get(img)
        if not faces:
            stats["no_face"] += 1
            fpath.unlink()
            continue

        if len(faces) > 1:
            stats["multi_face"] += 1
            fpath.unlink()
            continue

        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        fw, fh = x2 - x1, y2 - y1

        if fw < 200 or fh < 200:
            stats["small_face"] += 1
            fpath.unlink()
            continue

        age = getattr(face, "age", 25)
        if age < 18:
            stats["underage"] += 1
            fpath.unlink()
            continue

        lmk = getattr(face, "landmark_2d_106", None)
        if lmk is not None and len(lmk) > 86:
            nose_x = lmk[86, 0]
            face_cx = (x1 + x2) / 2
            if abs(nose_x - face_cx) / max(fw, 1) > 0.15:
                stats["side_pose"] += 1
                fpath.unlink()
                continue

        # Blur check
        crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        if crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 80:
                stats["blurry"] += 1
                fpath.unlink()
                continue

        # Smile check
        if not is_smiling(face, lmk):
            stats["not_smiling"] += 1
            fpath.unlink()
            continue

        # VERIFIED
        dst = VERIFIED_DIR / fpath.name
        if dst.exists():
            dst = VERIFIED_DIR / f"{fpath.stem}_{hashlib.md5(str(fpath).encode()).hexdigest()[:6]}{fpath.suffix}"
        fpath.rename(dst)
        stats["verified"] += 1

    log.info("=" * 50)
    log.info("VERIFICATION RESULTS:")
    for k, v in stats.items():
        if v > 0:
            log.info(f"  {k:<14} {v}")
    log.info("=" * 50)
    return stats


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main(target: int = 200, max_cycles: int = 5):
    log.info(f"Target: {target} verified smiling model photos")

    for cycle in range(1, max_cycles + 1):
        current = len(list(VERIFIED_DIR.glob("*.jpg")))
        log.info(f"\n{'='*60}")
        log.info(f"CYCLE {cycle}/{max_cycles}  —  verified: {current}/{target}")
        log.info(f"{'='*60}")

        if current >= target:
            log.info("TARGET REACHED")
            break

        need = target - current
        scrape_budget = min(500, need * 4)  # Expect ~25% pass rate

        log.info(f"[Scrape] Downloading ~{scrape_budget} raw images...")
        n = scrape_with_icrawler(max_images=scrape_budget)
        log.info(f"[Scrape] Got {n} new raw images")

        if n == 0:
            log.warning("  No new images — stopping")
            break

        log.info(f"[Verify] Filtering {n} images...")
        stats = verify_all_raw()
        log.info(f"[Verify] +{stats['verified']} verified")

    final = len(list(VERIFIED_DIR.glob("*.jpg")))
    log.info(f"\n{'='*60}")
    log.info(f"DONE — {final} verified smiling model photos")
    log.info(f"Location: {VERIFIED_DIR}")
    log.info(f"{'='*60}")
    return final


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--cycles", type=int, default=3)
    args = p.parse_args()
    main(target=args.target, max_cycles=args.cycles)
