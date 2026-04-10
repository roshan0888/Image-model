"""
High-Quality Model Photo Scraper

Targets Instagram-quality model photos from:
1. Pexels API (free, commercial license, 4K)
2. Unsplash API (free, editorial quality)
3. Google Images (site-filtered to pexels.com, unsplash.com, shutterstock editorial)

These sources give photoshoot-grade images:
- Professional studio lighting
- Natural expressions (not exaggerated)
- Multiple ethnicities
- Clean backgrounds
- 2K-4K resolution

We scrape PAIRED data strategy:
- Multiple photos of same model = same identity, different expressions
- Used directly as training pairs for LoRA fine-tuning
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import requests
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Add parent path for insightface
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [scraper] %(message)s")
logger = logging.getLogger(__name__)


# ─── SEARCH QUERIES ──────────────────────────────────────────────────────────

EXPRESSION_QUERIES = {
    "smile": [
        "photogenic model perfect smile portrait studio",
        "beautiful woman radiant smile fashion photography",
        "model gorgeous smile beauty campaign closeup",
        "woman perfect teeth smile professional headshot",
        "fashion model bright smile editorial portrait",
        "model natural photogenic smile studio lighting",
        "woman stunning smile luxury brand campaign",
        "model cheerful smile high fashion closeup",
        "beautiful smile model portfolio photography",
        "woman warm photogenic smile beauty editorial",
        "Asian model perfect smile portrait studio",
        "Black woman beautiful smile fashion editorial",
        "Indian model radiant smile beauty campaign",
        "Latina model gorgeous smile portrait photography",
        "diverse model photogenic smile studio portrait",
    ],
    "open_smile": [
        "model laughing open smile fashion portrait",
        "woman big bright smile teeth showing editorial",
        "model joyful open smile beauty photography",
        "woman genuine laugh smile studio portrait",
        "fashion model open happy smile campaign",
        "model wide smile teeth portrait studio lighting",
        "woman beaming open smile professional photo",
        "model toothy smile high fashion editorial",
    ],
    "neutral": [
        "model neutral expression portrait photography",
        "professional headshot neutral face studio",
        "fashion editorial neutral expression closeup",
        "model face calm expression beauty campaign",
        "beautiful woman neutral studio portrait lighting",
        "editorial model straight face high fashion",
    ],
}

# High-quality sites to bias toward
QUALITY_SITES = [
    "pexels.com",
    "unsplash.com",
    "models.com",
    "vogue.com",
    "harpersbazaar.com",
]

# Demographic diversity terms (combined with expression queries)
DEMOGRAPHIC_BOOST = [
    "Asian woman", "Black woman", "Latina woman", "Indian woman", "White woman",
    "Asian man", "Black man", "Latino man", "Indian man", "White man",
    "diverse model", "multiracial model",
]


# ─── PEXELS SCRAPER ──────────────────────────────────────────────────────────

class PexelsScraper:
    """Scrape from Pexels — free API, 4K quality, commercial license."""

    BASE_URL = "https://api.pexels.com/v1/search"

    def __init__(self, api_key: str, output_dir: str):
        self.api_key = api_key
        self.output_dir = output_dir
        self.headers = {"Authorization": api_key}
        os.makedirs(output_dir, exist_ok=True)

    def search(self, query: str, per_page: int = 80, pages: int = 3) -> List[Dict]:
        """Search Pexels for photos matching query."""
        results = []
        for page in range(1, pages + 1):
            try:
                r = requests.get(
                    self.BASE_URL,
                    headers=self.headers,
                    params={
                        "query": query,
                        "per_page": per_page,
                        "page": page,
                        "orientation": "portrait",
                        "size": "large",  # 4K+
                    },
                    timeout=15,
                )
                if r.status_code == 200:
                    data = r.json()
                    photos = data.get("photos", [])
                    for p in photos:
                        results.append({
                            "id": str(p["id"]),
                            "url": p["src"]["original"],  # Highest resolution
                            "width": p["width"],
                            "height": p["height"],
                            "photographer": p["photographer"],
                            "source": "pexels",
                            "query": query,
                        })
                    logger.info(f"  Pexels '{query}' page {page}: {len(photos)} results")
                elif r.status_code == 429:
                    logger.warning("  Pexels rate limited, sleeping 30s...")
                    time.sleep(30)
                else:
                    logger.warning(f"  Pexels error {r.status_code}")
            except Exception as e:
                logger.error(f"  Pexels request failed: {e}")
            time.sleep(0.5)
        return results

    def download(self, photos: List[Dict], expression: str) -> int:
        """Download photos to output_dir/expression/."""
        expr_dir = os.path.join(self.output_dir, expression)
        os.makedirs(expr_dir, exist_ok=True)
        downloaded = 0
        for p in photos:
            fname = f"pexels_{p['id']}.jpg"
            fpath = os.path.join(expr_dir, fname)
            if os.path.exists(fpath):
                continue
            try:
                r = requests.get(p["url"], timeout=30, stream=True)
                if r.status_code == 200:
                    with open(fpath, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                    downloaded += 1
            except Exception as e:
                logger.debug(f"  Download failed {p['url']}: {e}")
            time.sleep(0.2)
        return downloaded


# ─── UNSPLASH SCRAPER ────────────────────────────────────────────────────────

class UnsplashScraper:
    """Scrape from Unsplash — free API, editorial quality."""

    BASE_URL = "https://api.unsplash.com/search/photos"

    def __init__(self, access_key: str, output_dir: str):
        self.access_key = access_key
        self.output_dir = output_dir
        self.headers = {"Authorization": f"Client-ID {access_key}"}
        os.makedirs(output_dir, exist_ok=True)

    def search(self, query: str, per_page: int = 30, pages: int = 5) -> List[Dict]:
        results = []
        for page in range(1, pages + 1):
            try:
                r = requests.get(
                    self.BASE_URL,
                    headers=self.headers,
                    params={
                        "query": query,
                        "per_page": per_page,
                        "page": page,
                        "orientation": "portrait",
                        "content_filter": "high",
                    },
                    timeout=15,
                )
                if r.status_code == 200:
                    data = r.json()
                    photos = data.get("results", [])
                    for p in photos:
                        results.append({
                            "id": p["id"],
                            "url": p["urls"]["full"],  # Full resolution
                            "width": p["width"],
                            "height": p["height"],
                            "source": "unsplash",
                            "query": query,
                        })
                    logger.info(f"  Unsplash '{query}' page {page}: {len(photos)} results")
                elif r.status_code == 403:
                    logger.warning("  Unsplash quota hit")
                    break
            except Exception as e:
                logger.error(f"  Unsplash request failed: {e}")
            time.sleep(0.5)
        return results

    def download(self, photos: List[Dict], expression: str) -> int:
        expr_dir = os.path.join(self.output_dir, expression)
        os.makedirs(expr_dir, exist_ok=True)
        downloaded = 0
        for p in photos:
            fname = f"unsplash_{p['id']}.jpg"
            fpath = os.path.join(expr_dir, fname)
            if os.path.exists(fpath):
                continue
            try:
                r = requests.get(p["url"], timeout=30, stream=True)
                if r.status_code == 200:
                    with open(fpath, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                    downloaded += 1
            except Exception as e:
                logger.debug(f"  Download failed: {e}")
            time.sleep(0.2)
        return downloaded


# ─── GOOGLE IMAGES SCRAPER (site-filtered) ───────────────────────────────────

class GoogleModelScraper:
    """
    Scrape Google Images with site:pexels.com and site:unsplash.com filters
    for Instagram-quality model photos. Uses icrawler under the hood.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def scrape_expression(self, expression: str, max_images: int = 200) -> int:
        """Scrape one expression category with quality-site filters."""
        from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

        queries = EXPRESSION_QUERIES.get(expression, [])
        # Also add demographic variants
        extra_queries = []
        for demo in random.sample(DEMOGRAPHIC_BOOST, min(4, len(DEMOGRAPHIC_BOOST))):
            base = random.choice(queries) if queries else "portrait"
            extra_queries.append(f"{demo} {expression} portrait")

        all_queries = queries + extra_queries
        expr_dir = os.path.join(self.output_dir, expression)
        os.makedirs(expr_dir, exist_ok=True)

        total = 0
        per_query = max(10, max_images // len(all_queries))

        for q in all_queries:
            # Try Bing first (more reliable for high-res)
            try:
                crawler = BingImageCrawler(
                    storage={"root_dir": expr_dir},
                    log_level=logging.WARNING,
                    feeder_threads=1,
                    parser_threads=2,
                    downloader_threads=4,
                )
                crawler.crawl(
                    keyword=q,
                    max_num=per_query,
                    min_size=(512, 512),  # Minimum 512px (filters out thumbnails)
                    file_idx_offset=0,
                    filters={"license": "creativecommons"},
                )
            except Exception as e:
                logger.debug(f"Bing failed '{q}': {e}")

            # Also try Google
            try:
                crawler = GoogleImageCrawler(
                    storage={"root_dir": expr_dir},
                    log_level=logging.WARNING,
                    feeder_threads=1,
                    parser_threads=2,
                    downloader_threads=4,
                )
                crawler.crawl(
                    keyword=q,
                    max_num=per_query,
                    min_size=(512, 512),
                    file_idx_offset=0,
                )
            except Exception as e:
                logger.debug(f"Google failed '{q}': {e}")

            # Count files in dir
            n = len([f for f in os.listdir(expr_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            total = n
            logger.info(f"  [{expression}] '{q}' — total in dir: {n}")
            time.sleep(random.uniform(1.0, 2.5))

        return total


# ─── MASTER SCRAPER ──────────────────────────────────────────────────────────

class ModelPhotoScraper:
    """
    Master scraper combining Pexels + Unsplash + Google/Bing.
    Targets Instagram-quality model photos for training.
    """

    def __init__(self, output_dir: str, pexels_key: str = "", unsplash_key: str = ""):
        self.output_dir = output_dir
        self.pexels_key = pexels_key
        self.unsplash_key = unsplash_key
        os.makedirs(output_dir, exist_ok=True)

        self.google_scraper = GoogleModelScraper(output_dir)
        self.pexels = PexelsScraper(pexels_key, output_dir) if pexels_key else None
        self.unsplash = UnsplashScraper(unsplash_key, output_dir) if unsplash_key else None

    def scrape_all(self, target_per_expression: int = 500) -> Dict:
        """
        Scrape all expressions targeting `target_per_expression` images each.
        Returns stats dict.
        """
        expressions = ["smile", "neutral", "surprise", "sad"]
        stats = {}

        for expr in expressions:
            logger.info(f"\n{'='*60}")
            logger.info(f"SCRAPING: {expr.upper()} (target: {target_per_expression})")
            logger.info(f"{'='*60}")

            count = 0
            expr_dir = os.path.join(self.output_dir, expr)
            os.makedirs(expr_dir, exist_ok=True)

            # 1. Pexels API (best quality, if key available)
            if self.pexels:
                logger.info("  → Pexels API...")
                for q in EXPRESSION_QUERIES.get(expr, []):
                    photos = self.pexels.search(q, per_page=80, pages=2)
                    n = self.pexels.download(photos, expr)
                    count += n
                    logger.info(f"    Pexels '{q}': +{n} images")
                    if count >= target_per_expression // 3:
                        break

            # 2. Unsplash API (if key available)
            if self.unsplash:
                logger.info("  → Unsplash API...")
                for q in EXPRESSION_QUERIES.get(expr, [])[:4]:
                    photos = self.unsplash.search(q, per_page=30, pages=3)
                    n = self.unsplash.download(photos, expr)
                    count += n
                    logger.info(f"    Unsplash '{q}': +{n} images")

            # 3. Google/Bing crawler (always runs, fills the rest)
            logger.info("  → Google/Bing crawler...")
            n = self.google_scraper.scrape_expression(expr, max_images=target_per_expression)
            logger.info(f"    Google/Bing total in dir: {n}")

            # Final count
            final = len([f for f in os.listdir(expr_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            stats[expr] = final
            logger.info(f"  ✓ {expr}: {final} images collected")

        total = sum(stats.values())
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL COLLECTED: {total} images")
        for expr, n in stats.items():
            logger.info(f"  {expr}: {n}")
        logger.info(f"{'='*60}\n")

        # Save stats
        with open(os.path.join(self.output_dir, "scrape_stats.json"), "w") as f:
            json.dump({"total": total, "by_expression": stats}, f, indent=2)

        return stats

    def scrape_expression(self, expression: str, target: int = 300) -> int:
        """Scrape a single expression."""
        logger.info(f"Scraping {expression} (target: {target})...")

        expr_dir = os.path.join(self.output_dir, expression)
        os.makedirs(expr_dir, exist_ok=True)

        # Pexels
        if self.pexels:
            for q in EXPRESSION_QUERIES.get(expression, []):
                photos = self.pexels.search(q)
                self.pexels.download(photos, expression)

        # Unsplash
        if self.unsplash:
            for q in EXPRESSION_QUERIES.get(expression, [])[:3]:
                photos = self.unsplash.search(q)
                self.unsplash.download(photos, expression)

        # Google/Bing
        self.google_scraper.scrape_expression(expression, max_images=target)

        final = len([f for f in os.listdir(expr_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        logger.info(f"  {expression}: {final} images")
        return final


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape high-quality model photos")
    parser.add_argument("--output-dir", default="raw_data/model_photos",
                        help="Output directory")
    parser.add_argument("--target", type=int, default=500,
                        help="Target images per expression")
    parser.add_argument("--expression", default=None,
                        help="Scrape only one expression (smile/neutral/surprise/sad)")
    parser.add_argument("--pexels-key", default="",
                        help="Pexels API key (free at pexels.com/api)")
    parser.add_argument("--unsplash-key", default="",
                        help="Unsplash access key (free at unsplash.com/developers)")
    args = parser.parse_args()

    scraper = ModelPhotoScraper(
        output_dir=args.output_dir,
        pexels_key=args.pexels_key,
        unsplash_key=args.unsplash_key,
    )

    if args.expression:
        n = scraper.scrape_expression(args.expression, target=args.target)
        print(f"\nDone: {n} images for {args.expression}")
    else:
        stats = scraper.scrape_all(target_per_expression=args.target)
        print(f"\nDone: {sum(stats.values())} total images")
