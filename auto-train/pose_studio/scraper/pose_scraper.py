#!/usr/bin/env python3
"""
Pose-Aware Face Scraper

Scrapes photogenic face images organized by POSE and EXPRESSION.
8 pose categories × 4 expression categories = 12 query groups.
"""

import os
import sys
import hashlib
import shutil
import logging
from icrawler.builtin import BingImageCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [pose-scraper] %(message)s")
logger = logging.getLogger("pose-scraper")

STUDIO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

POSE_QUERIES = {
    "straight": [
        "portrait frontal face straight ahead studio professional headshot",
        "professional headshot face forward studio lighting",
        "passport photo style neutral face front view",
        "corporate headshot frontal face clean background",
        "celebrity portrait frontal face high resolution",
    ],
    "slight_left": [
        "portrait face turned slightly left professional headshot",
        "three quarter left face portrait studio",
        "person looking slightly left portrait photography",
        "headshot slight left angle professional",
    ],
    "slight_right": [
        "portrait face turned slightly right professional headshot",
        "three quarter right face portrait studio",
        "person looking slightly right portrait photography",
        "headshot slight right angle professional",
    ],
    "tilt_left": [
        "portrait head tilted left studio photography elegant",
        "person head tilt left portrait professional",
        "elegant head tilt left portrait high resolution",
    ],
    "tilt_right": [
        "portrait head tilted right studio photography elegant",
        "person head tilt right portrait professional",
        "elegant head tilt right portrait high resolution",
    ],
    "look_up": [
        "portrait face looking up slightly studio professional",
        "person looking upward portrait photography",
        "chin up portrait professional headshot",
    ],
    "look_down": [
        "portrait chin down eyes looking camera professional",
        "person looking down slightly portrait photography",
        "modest look down portrait headshot studio",
    ],
    "three_quarter": [
        "portrait three quarter view face studio professional",
        "three quarter profile portrait photography",
        "classic three quarter portrait high resolution",
    ],
}

EXPRESSION_QUERIES = {
    "subtle_smile": [
        "closed mouth smile portrait professional studio",
        "gentle smile headshot professional photography",
        "confident subtle smile portrait high resolution",
        "celebrity closed mouth smile portrait",
    ],
    "photoshoot_smile": [
        "teeth showing smile portrait professional studio",
        "big smile headshot professional photography",
        "toothy grin portrait high resolution studio",
        "happy smile teeth portrait photography",
    ],
    "natural_smile": [
        "genuine happy smile portrait candid natural",
        "authentic Duchenne smile portrait photography",
        "warm natural smile portrait high resolution",
        "real joy smile portrait candid",
    ],
    "neutral": [
        "neutral face portrait professional headshot",
        "serious face portrait studio photography",
        "neutral expression headshot high resolution",
    ],
}


def scrape_category(category, queries, target=60, output_dir=None):
    """Scrape images for one category."""
    if output_dir is None:
        output_dir = os.path.join(RAW_DIR, category)
    os.makedirs(output_dir, exist_ok=True)

    existing = len([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if existing >= target:
        logger.info(f"  {category}: already have {existing}/{target}")
        return existing

    per_query = max(5, (target - existing) // len(queries) + 1)
    total = existing

    for i, q in enumerate(queries):
        if total >= target:
            break
        qdir = os.path.join(output_dir, f"q{i}")
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
                new = os.path.join(output_dir, f"{category}_{h}{ext}")
                if not os.path.exists(new):
                    os.rename(fp, new)
                    total += 1
                else:
                    os.remove(fp)
            shutil.rmtree(qdir, ignore_errors=True)
        except Exception as e:
            logger.debug(f"  Query failed: {e}")

        logger.info(f"  {category} query {i+1}/{len(queries)}: {total} total")

    logger.info(f"  {category}: {total} images")
    return total


def scrape_all(pose_target=60, expr_target=60):
    """Scrape all pose and expression categories."""
    os.makedirs(RAW_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("POSE STUDIO SCRAPER")
    logger.info(f"  Pose target: {pose_target}/category")
    logger.info(f"  Expression target: {expr_target}/category")
    logger.info("=" * 60)

    # Scrape poses
    logger.info("\nScraping POSE images:")
    for cat, queries in POSE_QUERIES.items():
        scrape_category(f"pose_{cat}", queries, pose_target)

    # Scrape expressions
    logger.info("\nScraping EXPRESSION images:")
    for cat, queries in EXPRESSION_QUERIES.items():
        scrape_category(f"expr_{cat}", queries, expr_target)

    # Count totals
    total = 0
    for d in os.listdir(RAW_DIR):
        dp = os.path.join(RAW_DIR, d)
        if os.path.isdir(dp):
            c = len([f for f in os.listdir(dp) if f.endswith(('.jpg', '.jpeg', '.png'))])
            total += c
    logger.info(f"\nTotal scraped: {total} images")


if __name__ == "__main__":
    scrape_all(pose_target=60, expr_target=60)
