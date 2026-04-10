#!/usr/bin/env python3
"""
Scrape professional smile photos classified into 3 types:
  1. SUBTLE — closed mouth, editorial, confidence
  2. PHOTOSHOOT — teeth showing, eyes open, commercial
  3. NATURAL — genuine Duchenne smile, eyes crinkle
"""

import os
import sys
import hashlib
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [scrape] %(levelname)s: %(message)s")
logger = logging.getLogger("scrape")

PHOTO_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PHOTO_DIR, "raw_smiles")

SMILE_QUERIES = {
    "subtle": [
        "celebrity subtle smile closeup portrait studio",
        "model closed mouth smile editorial portrait",
        "professional headshot confident slight smile",
        "actress gentle smile portrait red carpet closeup",
        "LinkedIn professional headshot subtle smile",
        "fashion model smirk portrait studio lighting",
        "corporate headshot woman slight smile studio",
        "actor confident smile portrait high resolution",
        "elegant closed mouth smile portrait photography",
        "Bollywood actress subtle smile portrait closeup",
    ],
    "photoshoot": [
        "model smile teeth showing commercial portrait studio",
        "professional headshot toothy smile corporate",
        "actress photoshoot smile teeth portrait studio",
        "commercial model smile upper teeth portrait",
        "toothpaste ad smile portrait frontal studio",
        "Hollywood celebrity controlled smile portrait",
        "business professional smile teeth headshot",
        "Bollywood actor smile teeth portrait studio",
        "Korean celebrity smile teeth portrait closeup",
        "beauty campaign smile teeth portrait studio lighting",
    ],
    "natural": [
        "genuine happy smile portrait natural light candid",
        "person laughing naturally portrait closeup",
        "real Duchenne smile eyes crinkle portrait",
        "candid joy smile portrait photography",
        "wedding smile genuine portrait closeup",
        "natural warm smile portrait studio",
        "person truly happy smile eyes squint portrait",
        "authentic laugh smile portrait high resolution",
        "warm genuine smile Indian portrait photography",
        "real happiness smile portrait diverse",
    ],
}


def scrape_smile_type(smile_type, queries, target=80):
    """Scrape images for one smile type."""
    from icrawler.builtin import BingImageCrawler

    out_dir = os.path.join(RAW_DIR, smile_type)
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if existing >= target:
        logger.info(f"  {smile_type}: already have {existing}/{target}, skipping")
        return existing

    per_query = max(5, (target - existing) // len(queries) + 1)
    total = existing

    for i, q in enumerate(queries):
        if total >= target:
            break
        qdir = os.path.join(out_dir, f"q{i}")
        os.makedirs(qdir, exist_ok=True)
        try:
            c = BingImageCrawler(storage={"root_dir": qdir}, log_level=logging.WARNING)
            c.crawl(keyword=q, max_num=per_query, min_size=(512, 512))

            for f in os.listdir(qdir):
                fp = os.path.join(qdir, f)
                if not os.path.isfile(fp):
                    continue
                with open(fp, 'rb') as fh:
                    h = hashlib.md5(fh.read()).hexdigest()[:10]
                ext = os.path.splitext(f)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png'):
                    os.remove(fp)
                    continue
                new = os.path.join(out_dir, f"{smile_type}_{h}{ext}")
                if not os.path.exists(new):
                    os.rename(fp, new)
                    total += 1
                else:
                    os.remove(fp)
            shutil.rmtree(qdir, ignore_errors=True)
        except Exception as e:
            logger.debug(f"  Query failed: {e}")

        logger.info(f"  {smile_type} query {i+1}/{len(queries)}: {total} total")

    logger.info(f"  {smile_type}: {total} images scraped")
    return total


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SCRAPING PROFESSIONAL SMILE PHOTOS")
    logger.info("=" * 60)

    for smile_type, queries in SMILE_QUERIES.items():
        logger.info(f"\nScraping: {smile_type}")
        scrape_smile_type(smile_type, queries, target=80)

    # Count totals
    for st in SMILE_QUERIES:
        d = os.path.join(RAW_DIR, st)
        if os.path.exists(d):
            c = len([f for f in os.listdir(d) if f.endswith(('.jpg', '.jpeg', '.png'))])
            logger.info(f"  {st}: {c} images")


if __name__ == "__main__":
    main()
