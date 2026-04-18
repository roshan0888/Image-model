"""Scrape more model smile photos with expanded queries."""
import os, time, logging
from pathlib import Path
from icrawler.builtin import BingImageCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [scrape] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("raw_data/model_photos/smile_raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    # Pexels/Unsplash focus
    "unsplash smiling portrait close",
    "pexels smiling woman portrait",
    "pexels smiling man portrait",
    "unsplash beautiful smile woman",
    "unsplash beautiful smile man",
    "unsplash smile face professional",
    "pexels smile face beauty",
    "unsplash young smile portrait",
    "unsplash smile confident portrait",
    "pexels smile studio portrait",
    "unsplash smile outdoor portrait",
    "pexels smile natural light",
    "unsplash smile white background",
    "pexels smile dark background",
    # Diverse demographics
    "unsplash African American smile",
    "pexels Hispanic smile portrait",
    "unsplash Asian smile portrait",
    "pexels South Asian smile",
    "unsplash Korean smile portrait",
    "unsplash Japanese smile portrait",
    "pexels Middle Eastern smile",
    "unsplash mixed race smile portrait",
    # Ages
    "unsplash young adult smile 20s",
    "pexels 30s smile portrait",
    "unsplash 40s smile professional",
    "pexels 50s smile mature",
    # Professional contexts
    "unsplash businessman smile",
    "unsplash businesswoman smile",
    "pexels corporate smile headshot",
    "unsplash creative professional smile",
]

total_before = len(list(RAW_DIR.glob("*")))
log.info(f"Starting with {total_before} existing raw images")

for i, q in enumerate(QUERIES):
    log.info(f"  [{i+1}/{len(QUERIES)}] '{q}'")
    try:
        c = BingImageCrawler(
            downloader_threads=4,
            storage={"root_dir": str(RAW_DIR)},
        )
        c.crawl(
            keyword=q,
            max_num=40,
            min_size=(500, 500),
            file_idx_offset=len(list(RAW_DIR.glob("*"))),
        )
        time.sleep(1)
    except Exception as e:
        log.warning(f"  failed: {e}")

total_after = len(list(RAW_DIR.glob("*")))
log.info(f"DONE — {total_after - total_before} new raw images, {total_after} total")
