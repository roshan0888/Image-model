#!/usr/bin/env python3
"""
Face Image Scraper — Phase 1 (Small Scale)

Scrapes high-quality face images from the internet organized by expression.
Targets: smile, sad, surprise, neutral
Strategy: Celebrity/portrait queries → guaranteed real human faces, high resolution.

Starts small (50 per expression = 200 total), scales up later.
"""

import os
import sys
import json
import time
import logging
import hashlib
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("scraper")

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ENGINE_DIR, "dataset", "raw")


# ═══════════════════════════════════════════════════════════════
# SEARCH QUERIES — designed to find REAL high-quality face photos
# ═══════════════════════════════════════════════════════════════

EXPRESSION_QUERIES = {
    "smile": [
        "celebrity smiling portrait high resolution photo",
        "person smiling headshot professional photography",
        "happy face portrait closeup studio lighting",
        "actor smiling red carpet portrait",
        "business professional smiling headshot",
        "woman smiling portrait natural light photography",
        "man smiling portrait studio photo",
        "young person smiling face closeup photo",
        "elderly person smiling portrait photography",
        "candid smile portrait high quality",
    ],
    "sad": [
        "person sad face portrait photography",
        "actor crying movie still closeup",
        "sad expression portrait studio photo",
        "person upset face closeup photography",
        "melancholy portrait face high resolution",
        "emotional sad portrait closeup photo",
        "person frowning portrait photography",
        "sad face closeup studio lighting",
    ],
    "surprise": [
        "person surprised face portrait photography",
        "shocked expression portrait closeup photo",
        "surprised face reaction high resolution",
        "person amazed expression portrait studio",
        "wide eyes open mouth surprised portrait",
        "surprise reaction face closeup photography",
        "shocked face portrait high quality photo",
        "astonished expression portrait closeup",
    ],
    "neutral": [
        "passport photo face neutral expression",
        "professional headshot neutral face",
        "portrait neutral expression studio lighting",
        "mugshot style neutral face photo",
        "ID photo face straight forward",
        "person neutral face closeup studio",
        "corporate headshot neutral expression",
        "face portrait no expression high resolution",
    ],
}

# Demographic diversity queries (mixed into each expression)
DIVERSITY_MODIFIERS = [
    "asian", "african", "european", "indian", "latino",
    "middle eastern", "young adult", "middle aged", "elderly",
    "male", "female",
]


def scrape_expression(expression: str, queries: List[str], target_count: int = 50,
                      output_dir: str = None):
    """Scrape images for a single expression."""
    from icrawler.builtin import BingImageCrawler

    if output_dir is None:
        output_dir = os.path.join(RAW_DIR, expression)
    os.makedirs(output_dir, exist_ok=True)

    existing = len([f for f in os.listdir(output_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if existing >= target_count:
        logger.info(f"  {expression}: already have {existing}/{target_count}, skipping")
        return existing

    remaining = target_count - existing
    per_query = max(5, remaining // len(queries) + 1)

    logger.info(f"  {expression}: need {remaining} more images, {per_query} per query")

    total_scraped = existing
    for i, query in enumerate(queries):
        if total_scraped >= target_count:
            break

        # Use a subdirectory per query to avoid overwrites
        query_dir = os.path.join(output_dir, f"q{i:02d}")
        os.makedirs(query_dir, exist_ok=True)

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": query_dir},
                log_level=logging.WARNING,
            )
            crawler.crawl(
                keyword=query,
                max_num=per_query,
                min_size=(512, 512),  # Minimum 512x512 for quality
                file_idx_offset=0,
            )

            # Move files to parent dir with unique names
            for fname in os.listdir(query_dir):
                fpath = os.path.join(query_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue

                # Hash-based unique name to avoid duplicates
                with open(fpath, 'rb') as fh:
                    file_hash = hashlib.md5(fh.read()).hexdigest()[:12]
                ext = os.path.splitext(fname)[1].lower()
                if ext == '.webp':
                    ext = '.jpg'
                new_name = f"{expression}_{file_hash}{ext}"
                new_path = os.path.join(output_dir, new_name)

                if not os.path.exists(new_path):
                    os.rename(fpath, new_path)
                    total_scraped += 1
                else:
                    os.remove(fpath)  # Duplicate

            # Cleanup query subdir
            try:
                os.rmdir(query_dir)
            except OSError:
                import shutil
                shutil.rmtree(query_dir, ignore_errors=True)

        except Exception as e:
            logger.warning(f"    Query '{query[:40]}...' failed: {e}")
            continue

        logger.info(f"    Query {i+1}/{len(queries)}: '{query[:50]}...' → {total_scraped} total")

    logger.info(f"  {expression}: scraped {total_scraped} images total")
    return total_scraped


def scrape_all(target_per_expression: int = 50):
    """Scrape all expressions."""
    os.makedirs(RAW_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FACE IMAGE SCRAPER — Phase 1")
    logger.info(f"  Target: {target_per_expression} images per expression")
    logger.info(f"  Expressions: {list(EXPRESSION_QUERIES.keys())}")
    logger.info(f"  Output: {RAW_DIR}")
    logger.info("=" * 60)

    results = {}
    for expression, queries in EXPRESSION_QUERIES.items():
        logger.info(f"\nScraping: {expression}")
        count = scrape_expression(expression, queries, target_per_expression)
        results[expression] = count

    # Save scrape manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_per_expression": target_per_expression,
        "results": results,
        "total": sum(results.values()),
    }
    manifest_path = os.path.join(RAW_DIR, "scrape_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("SCRAPE COMPLETE")
    for expr, count in results.items():
        logger.info(f"  {expr}: {count} images")
    logger.info(f"  TOTAL: {sum(results.values())} images")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50,
                        help="Images per expression (default: 50)")
    args = parser.parse_args()
    scrape_all(target_per_expression=args.count)
