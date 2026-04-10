"""
Autonomous Internet Data Collection Engine

Collects diverse face images from the internet using icrawler.
Generates targeted search queries based on demographic distributions.
Ensures geographic, expression, and identity diversity.
"""

import os
import json
import hashlib
import random
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from itertools import product

logger = logging.getLogger(__name__)


class QueryGenerator:
    """Generates diverse search queries for face data collection."""

    ETHNICITY_TERMS = {
        "east_asian": ["Chinese", "Japanese", "Korean", "East Asian"],
        "south_asian": ["Indian", "Pakistani", "Bangladeshi", "South Asian"],
        "african": ["African", "Nigerian", "Kenyan", "Ethiopian", "Black"],
        "caucasian": ["European", "Caucasian", "American", "British"],
        "latino": ["Latin American", "Mexican", "Brazilian", "Hispanic"],
        "middle_eastern": ["Middle Eastern", "Arab", "Turkish", "Persian"],
        "southeast_asian": ["Filipino", "Thai", "Vietnamese", "Indonesian"],
        "mixed": ["mixed ethnicity", "multiracial", "diverse"],
    }

    AGE_TERMS = {
        "18-25": ["young adult", "college age", "twenty something", "young"],
        "25-35": ["adult", "thirty something", "young professional"],
        "35-50": ["middle aged", "mature adult", "forty something"],
        "50-65": ["older adult", "senior professional", "fifty something"],
        "65+": ["elderly", "senior citizen", "older person", "grandparent"],
    }

    GENDER_TERMS = {
        "male": ["man", "male", "guy", "gentleman"],
        "female": ["woman", "female", "lady"],
    }

    EXPRESSION_TERMS = {
        "neutral": ["neutral face", "straight face", "calm expression", "passport photo"],
        "smile": ["smiling", "gentle smile", "happy face", "grinning"],
        "big_smile": ["laughing", "big smile", "toothy grin", "beaming"],
        "surprise": ["surprised face", "shocked expression", "wide eyes", "amazed"],
        "angry": ["angry face", "frowning", "mad expression", "furious"],
        "sad": ["sad face", "crying", "frowning", "melancholy expression"],
        "laugh": ["laughing hard", "burst of laughter", "hilarious reaction"],
    }

    QUALITY_SUFFIXES = [
        "portrait photography",
        "high resolution face",
        "studio portrait",
        "professional headshot",
        "closeup face photo",
        "4k portrait",
        "DSLR portrait",
    ]

    def __init__(self, config: dict):
        self.config = config
        self.ethnicity_dist = config.get("ethnicity_distribution", {})
        self.age_dist = config.get("age_distribution", {})
        self.gender_dist = config.get("gender_distribution", {})
        self.expression_types = config.get("expression_types", ["neutral", "smile"])

    def generate_queries(self, num_queries: int = 500, bias: Optional[Dict] = None) -> List[Dict]:
        """Generate diverse search queries weighted by demographic distribution.

        Args:
            num_queries: Total number of queries to generate.
            bias: Optional dict to bias toward specific demographics
                  (used by reinforcement engine for failure cases).
                  e.g. {"ethnicity": "african", "age": "65+", "expression": "surprise"}
        """
        queries = []

        for _ in range(num_queries):
            # Sample demographics from distribution (with optional bias)
            if bias and random.random() < 0.7:  # 70% biased, 30% random
                ethnicity = bias.get("ethnicity", self._sample_weighted(self.ethnicity_dist))
                age = bias.get("age", self._sample_weighted(self.age_dist))
                gender = bias.get("gender", self._sample_weighted(self.gender_dist))
                expression = bias.get("expression", random.choice(self.expression_types))
            else:
                ethnicity = self._sample_weighted(self.ethnicity_dist)
                age = self._sample_weighted(self.age_dist)
                gender = self._sample_weighted(self.gender_dist)
                expression = random.choice(self.expression_types)

            # Build query string
            eth_term = random.choice(self.ETHNICITY_TERMS.get(ethnicity, ["person"]))
            age_term = random.choice(self.AGE_TERMS.get(age, ["adult"]))
            gen_term = random.choice(self.GENDER_TERMS.get(gender, ["person"]))
            expr_term = random.choice(self.EXPRESSION_TERMS.get(expression, ["face"]))
            quality = random.choice(self.QUALITY_SUFFIXES)

            # Randomly pick a template style
            templates = [
                f"{eth_term} {gen_term} {age_term} {expr_term} {quality}",
                f"{gen_term} {expr_term} portrait photo {age_term}",
                f"professional headshot {eth_term} {gen_term} {expr_term}",
                f"{age_term} {gen_term} face {expr_term} closeup",
                f"studio portrait {eth_term} {expr_term} high quality",
            ]
            query_text = random.choice(templates)

            queries.append({
                "query": query_text,
                "ethnicity": ethnicity,
                "age": age,
                "gender": gender,
                "expression": expression,
            })

        random.shuffle(queries)
        return queries

    def generate_reinforcement_queries(self, failure_analysis: Dict) -> List[Dict]:
        """Generate targeted queries based on failure analysis.

        Args:
            failure_analysis: Dict with failure breakdowns by demographic.
                e.g. {"high_failure_demographics": [{"ethnicity": "african", "age": "65+", ...}]}
        """
        queries = []
        for demo in failure_analysis.get("high_failure_demographics", []):
            bias = {k: v for k, v in demo.items() if k in ("ethnicity", "age", "gender", "expression")}
            targeted = self.generate_queries(num_queries=20, bias=bias)
            queries.extend(targeted)

        return queries

    @staticmethod
    def _sample_weighted(distribution: dict) -> str:
        keys = list(distribution.keys())
        weights = list(distribution.values())
        total = sum(weights)
        if total == 0:
            return random.choice(keys) if keys else "unknown"
        weights = [w / total for w in weights]
        return random.choices(keys, weights=weights, k=1)[0]


class FaceDataCollector:
    """Collects face images from the internet using icrawler."""

    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = config["paths"]["raw_images_dir"]
        self.target_total = config["data_collection"]["target_total_images"]
        self.query_gen = QueryGenerator(config["data_collection"])
        self._seen_hashes = set()
        self._metadata = []

        os.makedirs(self.raw_dir, exist_ok=True)
        self._load_seen_hashes()

    def _load_seen_hashes(self):
        """Load hashes of already-downloaded images to avoid duplicates."""
        hash_file = os.path.join(self.raw_dir, ".image_hashes.json")
        if os.path.exists(hash_file):
            with open(hash_file) as f:
                self._seen_hashes = set(json.load(f))

    def _save_seen_hashes(self):
        hash_file = os.path.join(self.raw_dir, ".image_hashes.json")
        with open(hash_file, "w") as f:
            json.dump(list(self._seen_hashes), f)

    def _image_hash(self, filepath: str) -> str:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def collect(self, num_queries: int = 100, images_per_query: int = 50,
                bias: Optional[Dict] = None) -> Dict:
        """Run collection cycle.

        Returns dict with collection stats.
        """
        from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

        queries = self.query_gen.generate_queries(num_queries, bias=bias)
        stats = {
            "total_downloaded": 0,
            "duplicates_skipped": 0,
            "queries_run": 0,
            "by_ethnicity": {},
            "by_expression": {},
            "by_age": {},
        }

        for qi, q in enumerate(queries):
            query_dir = os.path.join(
                self.raw_dir,
                f"{q['ethnicity']}_{q['gender']}_{q['age']}_{q['expression']}"
            )
            os.makedirs(query_dir, exist_ok=True)

            logger.info(f"[{qi+1}/{len(queries)}] Scraping: {q['query']}")

            try:
                # Try Bing first (more reliable), fall back to Google
                crawler = BingImageCrawler(
                    storage={"root_dir": query_dir},
                    log_level=logging.WARNING,
                )
                crawler.crawl(
                    keyword=q["query"],
                    max_num=images_per_query,
                    min_size=(256, 256),
                    file_idx_offset="auto",
                )
            except Exception as e:
                logger.warning(f"Bing failed for '{q['query']}': {e}, trying Google...")
                try:
                    crawler = GoogleImageCrawler(
                        storage={"root_dir": query_dir},
                        log_level=logging.WARNING,
                    )
                    crawler.crawl(
                        keyword=q["query"],
                        max_num=images_per_query,
                    )
                except Exception as e2:
                    logger.error(f"Google also failed: {e2}")
                    continue

            # Deduplicate new downloads
            new_files = []
            for fname in os.listdir(query_dir):
                fpath = os.path.join(query_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                h = self._image_hash(fpath)
                if h in self._seen_hashes:
                    os.remove(fpath)
                    stats["duplicates_skipped"] += 1
                else:
                    self._seen_hashes.add(h)
                    new_files.append(fpath)
                    self._metadata.append({
                        "path": fpath,
                        "query": q["query"],
                        "ethnicity": q["ethnicity"],
                        "age": q["age"],
                        "gender": q["gender"],
                        "expression": q["expression"],
                    })

            stats["total_downloaded"] += len(new_files)
            stats["queries_run"] += 1

            # Track distribution
            eth = q["ethnicity"]
            stats["by_ethnicity"][eth] = stats["by_ethnicity"].get(eth, 0) + len(new_files)
            expr = q["expression"]
            stats["by_expression"][expr] = stats["by_expression"].get(expr, 0) + len(new_files)
            age = q["age"]
            stats["by_age"][age] = stats["by_age"].get(age, 0) + len(new_files)

            # Save hashes periodically
            if qi % 10 == 0:
                self._save_seen_hashes()

            # Rate limiting
            time.sleep(random.uniform(0.5, 2.0))

        self._save_seen_hashes()
        self._save_metadata()

        logger.info(f"Collection complete: {stats['total_downloaded']} new images, "
                    f"{stats['duplicates_skipped']} duplicates skipped")
        return stats

    def collect_reinforcement(self, failure_analysis: Dict) -> Dict:
        """Targeted collection based on failure analysis."""
        queries = self.query_gen.generate_reinforcement_queries(failure_analysis)
        logger.info(f"Reinforcement collection: {len(queries)} targeted queries")

        # Run collection with the targeted queries
        return self.collect(
            num_queries=len(queries),
            images_per_query=self.config["reinforcement"]["scrape_per_query"],
        )

    def _save_metadata(self):
        meta_path = os.path.join(self.raw_dir, "collection_metadata.jsonl")
        with open(meta_path, "a") as f:
            for m in self._metadata:
                f.write(json.dumps(m) + "\n")
        self._metadata = []

    def get_stats(self) -> Dict:
        """Get current dataset statistics."""
        total = 0
        by_category = {}
        for root, dirs, files in os.walk(self.raw_dir):
            img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if img_files:
                cat = os.path.basename(root)
                by_category[cat] = len(img_files)
                total += len(img_files)
        return {"total_images": total, "by_category": by_category}
