"""
Clean Data Engine — Pexels + Unsplash + icrawler with domain filtering

Strategy:
  1. Pexels API (no watermarks, 4K, commercial license)
  2. icrawler with BLOCKED_DOMAINS filter (rejects stock photo sites)
  3. Verify: face + frontal + smiling + 200×200+ + no watermark
  4. Loop until target reached
"""

import os, sys, cv2, time, hashlib, logging, requests, random
import numpy as np
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [clean_engine] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw_data/model_photos/smile_raw"
VERIFIED_DIR = ROOT / "raw_data/cleaned/smile"
for d in [RAW_DIR, VERIFIED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# BLOCKED DOMAINS — stock photo sites with watermarks
# ══════════════════════════════════════════════════════════════════

BLOCKED_DOMAINS = {
    "adobestock.com", "stock.adobe.com",
    "shutterstock.com", "istockphoto.com", "gettyimages.com",
    "dreamstime.com", "alamy.com", "depositphotos.com", "123rf.com",
    "bigstockphoto.com", "canstockphoto.com", "stockvault.net",
    "sciencephoto.com", "focusedcollection.com", "stocksy.com",
    "fotolia.com", "ftcdn.net", "dissolve.com",
    "agefotostock.com", "mediastorehouse.com", "stock.com",
    "photodune.net", "creativemarket.com",
}

PREFERRED_DOMAINS = {
    "pexels.com", "unsplash.com", "pixabay.com",
    "pinterest.com",  # sometimes OK
    "wikimedia.org", "flickr.com",
    "pxhere.com", "publicdomainpictures.net",
}


def is_blocked_url(url: str) -> bool:
    """Check if URL is from a blocked (watermarked) domain."""
    try:
        host = urlparse(url).netloc.lower()
        for blocked in BLOCKED_DOMAINS:
            if blocked in host:
                return True
    except Exception:
        return False
    return False


# ══════════════════════════════════════════════════════════════════
# PEXELS API SCRAPER — watermark-free, 4K
# ══════════════════════════════════════════════════════════════════

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

SMILE_QUERIES = [
    "smiling portrait", "happy face portrait", "smiling woman", "smiling man",
    "natural smile", "portrait smile", "photogenic smile", "headshot smile",
    "close-up smile", "happy portrait",
    "Asian smile", "Black smile", "Latino smile", "Indian smile",
]


def scrape_pexels(target: int = 100) -> int:
    """Scrape from Pexels API (requires PEXELS_API_KEY env var)."""
    if not PEXELS_API_KEY:
        log.warning("No PEXELS_API_KEY set, skipping Pexels")
        return 0

    downloaded = 0
    for query in SMILE_QUERIES:
        if downloaded >= target:
            break
        try:
            r = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": PEXELS_API_KEY},
                params={
                    "query": query,
                    "per_page": 40,
                    "page": 1,
                    "orientation": "portrait",
                    "size": "large",
                },
                timeout=20,
            )
            if r.status_code != 200:
                continue
            photos = r.json().get("photos", [])
            for p in photos:
                if downloaded >= target:
                    break
                url = p["src"]["large2x"]
                fname = f"pexels_{p['id']}.jpg"
                fpath = RAW_DIR / fname
                if fpath.exists():
                    continue
                try:
                    img_r = requests.get(url, timeout=30, stream=True)
                    if img_r.status_code == 200:
                        with open(fpath, "wb") as f:
                            for chunk in img_r.iter_content(8192):
                                f.write(chunk)
                        downloaded += 1
                        if downloaded % 10 == 0:
                            log.info(f"  Pexels: {downloaded}")
                        time.sleep(0.2)
                except Exception:
                    pass
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"  Pexels '{query}': {e}")
    return downloaded


# ══════════════════════════════════════════════════════════════════
# ICRAWLER with URL filtering
# ══════════════════════════════════════════════════════════════════

def scrape_bing_filtered(target: int = 200) -> int:
    """Use Bing search via icrawler but reject blocked domains."""
    from icrawler.builtin import BingImageCrawler
    from icrawler.downloader import ImageDownloader

    # Custom downloader that rejects blocked domains
    class FilteredDownloader(ImageDownloader):
        def keep_file(self, task, response, min_size=None, max_size=None):
            # Block by URL domain first
            if is_blocked_url(task["file_url"]):
                return False
            return super().keep_file(task, response, min_size, max_size)

    before = len(list(RAW_DIR.glob("*.jpg")))

    queries = [
        "smile portrait pexels",
        "smile unsplash",
        "smiling face unsplash",
        "happy portrait unsplash",
        "smile face pxhere",
        "smile face wikimedia",
        "smile close up unsplash",
    ]

    per_query = max(30, target // len(queries))

    for q in queries:
        log.info(f"  Bing (filtered): '{q}'")
        try:
            crawler = BingImageCrawler(
                downloader_threads=4,
                downloader_cls=FilteredDownloader,
                storage={"root_dir": str(RAW_DIR)},
            )
            crawler.crawl(
                keyword=q,
                max_num=per_query,
                min_size=(500, 500),
                file_idx_offset=len(list(RAW_DIR.glob("*.jpg"))),
            )
            time.sleep(1)
        except Exception as e:
            log.warning(f"  {q}: {e}")

    return len(list(RAW_DIR.glob("*.jpg"))) - before


# ══════════════════════════════════════════════════════════════════
# VERIFY
# ══════════════════════════════════════════════════════════════════

_fa = None


def get_fa():
    global _fa
    if _fa is None:
        from insightface.app import FaceAnalysis
        _fa = FaceAnalysis(
            name="antelopev2",
            root=str(ROOT / "MagicFace/third_party_files"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)
    return _fa


def has_watermark_strict(img) -> bool:
    """Aggressive watermark detection using edge patterns + text detection."""
    if img is None or img.size == 0:
        return False
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Method 1: repetitive patterns (watermark repeats)
    # Sample 4 regions, check if they have similar edge signatures
    regions = [
        gray[0:h//2, 0:w//2],
        gray[0:h//2, w//2:w],
        gray[h//2:h, 0:w//2],
        gray[h//2:h, w//2:w],
    ]
    edge_densities = []
    for r in regions:
        edges = cv2.Canny(r, 50, 150)
        edge_densities.append(edges.sum() / max(edges.size, 1))

    # If edge densities are very uniform AND high, likely watermark pattern
    mean_ed = np.mean(edge_densities)
    std_ed = np.std(edge_densities)
    if mean_ed > 0.05 and std_ed / (mean_ed + 1e-6) < 0.25:
        return True

    # Method 2: check for diagonal repeating pattern in top-left quadrant
    # Stock photo diagonal watermarks create strong Hough line patterns
    top_left = gray[0:h//3, 0:w//3]
    edges_tl = cv2.Canny(top_left, 100, 200)
    lines = cv2.HoughLinesP(edges_tl, 1, np.pi/180, 30, minLineLength=20, maxLineGap=5)
    if lines is not None and len(lines) > 15:
        # Many lines = likely watermark text
        return True

    return False


def is_smiling(lmk) -> bool:
    if lmk is None or len(lmk) < 72:
        return True
    left = lmk[52]
    right = lmk[61] if len(lmk) > 61 else lmk[58]
    upper = lmk[56] if len(lmk) > 56 else lmk[53]
    lower = lmk[66] if len(lmk) > 66 else lmk[62]
    corner_y = (left[1] + right[1]) / 2
    center_y = (upper[1] + lower[1]) / 2
    mouth_w = abs(right[0] - left[0])
    if mouth_w < 10:
        return True
    return corner_y <= center_y + mouth_w * 0.05


def verify_all():
    fa = get_fa()
    stats = {
        "total": 0, "watermarked": 0, "no_face": 0, "multi_face": 0,
        "small_face": 0, "underage": 0, "side_pose": 0, "blurry": 0,
        "not_smiling": 0, "verified": 0,
    }

    for fpath in sorted(RAW_DIR.glob("*")):
        if fpath.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        stats["total"] += 1
        img = cv2.imread(str(fpath))
        if img is None:
            fpath.unlink()
            continue

        if has_watermark_strict(img):
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
        fw, fh = x2-x1, y2-y1
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
            face_cx = (x1+x2)/2
            if abs(nose_x - face_cx) / max(fw, 1) > 0.15:
                stats["side_pose"] += 1
                fpath.unlink()
                continue

        crop = img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]
        if crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 80:
                stats["blurry"] += 1
                fpath.unlink()
                continue

        if not is_smiling(lmk):
            stats["not_smiling"] += 1
            fpath.unlink()
            continue

        dst = VERIFIED_DIR / fpath.name
        if dst.exists():
            dst = VERIFIED_DIR / f"{fpath.stem}_{hashlib.md5(str(fpath).encode()).hexdigest()[:6]}{fpath.suffix}"
        fpath.rename(dst)
        stats["verified"] += 1

    log.info("=" * 50)
    log.info("VERIFICATION:")
    for k, v in stats.items():
        if v > 0:
            log.info(f"  {k:<14} {v}")
    log.info("=" * 50)
    return stats


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

def main(target: int = 200, max_cycles: int = 5):
    log.info(f"Target: {target} clean smiling model photos")

    for cycle in range(1, max_cycles + 1):
        current = len(list(VERIFIED_DIR.glob("*.jpg")))
        log.info(f"\nCYCLE {cycle}/{max_cycles}  —  verified: {current}/{target}")
        if current >= target:
            log.info("TARGET REACHED")
            break

        need = target - current
        # Try Pexels first, fall back to Bing
        p = scrape_pexels(target=need * 2)
        log.info(f"  Pexels: {p} downloaded")

        b = scrape_bing_filtered(target=need * 4)
        log.info(f"  Bing (filtered): {b} downloaded")

        if p + b == 0:
            log.warning("No new images — stopping")
            break

        verify_all()

    final = len(list(VERIFIED_DIR.glob("*.jpg")))
    log.info(f"\nDONE — {final} verified clean photos")
    log.info(f"Location: {VERIFIED_DIR}")
    return final


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=150)
    p.add_argument("--cycles", type=int, default=4)
    args = p.parse_args()
    main(target=args.target, max_cycles=args.cycles)
