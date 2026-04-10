#!/usr/bin/env python3
"""
Identity Clustering & Pair Generation

Takes cleaned face images and:
1. Clusters by identity using ArcFace embeddings
2. Creates training pairs (same person, different expressions)
3. For identities with only one expression, creates synthetic pairs
   by using the image as source and a different expression's image as driving template
"""

import os
import sys
import cv2
import json
import numpy as np
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("cluster")

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ENGINE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")


class IdentityClusterer:
    """Cluster face images by identity using ArcFace embeddings."""

    def __init__(self, antelopev2_dir: str):
        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(
            name="antelopev2", root=antelopev2_dir,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Extract ArcFace embedding from image."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        faces = self.face_analyzer.get(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return face.normed_embedding

    def cluster_by_identity(self, image_paths: List[str],
                            threshold: float = 0.55) -> Dict[int, List[str]]:
        """
        Cluster images by identity. Two images are same person if
        ArcFace cosine similarity > threshold.

        Uses simple greedy clustering (good enough for hundreds of images).
        """
        embeddings = {}
        for path in image_paths:
            emb = self.get_embedding(path)
            if emb is not None:
                embeddings[path] = emb

        logger.info(f"  Got embeddings for {len(embeddings)}/{len(image_paths)} images")

        # Greedy clustering
        clusters = {}  # cluster_id → [paths]
        cluster_centroids = {}  # cluster_id → mean_embedding
        next_id = 0

        for path, emb in embeddings.items():
            best_cluster = None
            best_sim = threshold

            for cid, centroid in cluster_centroids.items():
                sim = float(np.dot(emb, centroid))
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cid

            if best_cluster is not None:
                clusters[best_cluster].append(path)
                # Update centroid (running average)
                n = len(clusters[best_cluster])
                cluster_centroids[best_cluster] = (
                    cluster_centroids[best_cluster] * (n - 1) + emb
                ) / n
                # Re-normalize
                cluster_centroids[best_cluster] /= np.linalg.norm(
                    cluster_centroids[best_cluster]
                )
            else:
                clusters[next_id] = [path]
                cluster_centroids[next_id] = emb.copy()
                next_id += 1

        return clusters


def generate_training_pairs(cleaned_dir: str, paired_dir: str,
                           antelopev2_dir: str):
    """
    Generate training pairs from cleaned images.

    Strategy:
    1. Load all cleaned images with their expression labels
    2. Cluster by identity
    3. For clusters with multiple expressions → create real pairs
    4. For single-expression clusters → create synthetic pairs
       (same person as source, different expression image as driving template)
    """
    os.makedirs(paired_dir, exist_ok=True)

    # Collect all cleaned images
    all_images = []  # (path, expression)
    expressions_found = set()

    for expr_dir in os.listdir(cleaned_dir):
        expr_path = os.path.join(cleaned_dir, expr_dir)
        if not os.path.isdir(expr_path):
            continue
        for f in os.listdir(expr_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith("clean_"):
                all_images.append((os.path.join(expr_path, f), expr_dir))
                expressions_found.add(expr_dir)

    logger.info(f"Found {len(all_images)} cleaned images across {len(expressions_found)} expressions")

    if not all_images:
        logger.error("No cleaned images found!")
        return

    # Cluster by identity
    logger.info("Clustering by identity...")
    clusterer = IdentityClusterer(antelopev2_dir)
    paths = [p for p, _ in all_images]
    clusters = clusterer.cluster_by_identity(paths, threshold=0.55)

    # Map path → expression
    path_to_expr = {p: e for p, e in all_images}

    # Analyze clusters
    multi_expr_clusters = 0
    single_expr_clusters = 0
    for cid, cpaths in clusters.items():
        exprs = set(path_to_expr.get(p, "unknown") for p in cpaths)
        if len(exprs) > 1:
            multi_expr_clusters += 1
        else:
            single_expr_clusters += 1

    logger.info(f"  Clusters: {len(clusters)} total")
    logger.info(f"  Multi-expression: {multi_expr_clusters}")
    logger.info(f"  Single-expression: {single_expr_clusters}")

    # Generate pairs
    pairs = []

    # Type 1: Real pairs (same person, different expressions)
    for cid, cpaths in clusters.items():
        path_by_expr = defaultdict(list)
        for p in cpaths:
            expr = path_to_expr.get(p, "unknown")
            path_by_expr[expr].append(p)

        exprs = list(path_by_expr.keys())
        if len(exprs) < 2:
            continue

        # Create pairs between all expression combinations
        for i, expr_a in enumerate(exprs):
            for expr_b in exprs[i+1:]:
                for pa in path_by_expr[expr_a]:
                    for pb in path_by_expr[expr_b]:
                        # Both directions
                        pairs.append({
                            "source_path": pa,
                            "target_path": pb,
                            "source_expression": expr_a,
                            "target_expression": expr_b,
                            "identity_cluster": cid,
                            "pair_type": "real",
                        })
                        pairs.append({
                            "source_path": pb,
                            "target_path": pa,
                            "source_expression": expr_b,
                            "target_expression": expr_a,
                            "identity_cluster": cid,
                            "pair_type": "real",
                        })

    logger.info(f"  Real pairs: {len(pairs)}")

    # Type 2: Synthetic pairs (source + driving template from different person)
    # For each image, pair it with a random image of a different expression
    # The source identity should be preserved, expression comes from driving
    expr_pools = defaultdict(list)
    for p, e in all_images:
        expr_pools[e].append(p)

    synthetic_pairs = []
    for source_path, source_expr in all_images:
        for target_expr in expressions_found:
            if target_expr == source_expr:
                continue
            if not expr_pools[target_expr]:
                continue

            # Pick a random driving template
            driving_path = np.random.choice(expr_pools[target_expr])

            synthetic_pairs.append({
                "source_path": source_path,
                "driving_path": driving_path,
                "target_path": None,  # No ground truth — self-supervised
                "source_expression": source_expr,
                "target_expression": target_expr,
                "identity_cluster": -1,
                "pair_type": "synthetic",
            })

    logger.info(f"  Synthetic pairs: {len(synthetic_pairs)}")

    all_pairs = pairs + synthetic_pairs

    # Save pairs
    pairs_path = os.path.join(paired_dir, "training_pairs.jsonl")
    with open(pairs_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    # Save summary
    summary = {
        "total_images": len(all_images),
        "expressions": list(expressions_found),
        "identity_clusters": len(clusters),
        "real_pairs": len(pairs),
        "synthetic_pairs": len(synthetic_pairs),
        "total_pairs": len(all_pairs),
    }
    summary_path = os.path.join(paired_dir, "pair_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n  Total training pairs: {len(all_pairs)}")
    logger.info(f"  Saved to: {pairs_path}")

    return summary


if __name__ == "__main__":
    antelopev2_dir = os.path.join(BASE_DIR, "MagicFace", "third_party_files")
    cleaned_dir = os.path.join(ENGINE_DIR, "dataset", "cleaned_scraped")
    paired_dir = os.path.join(ENGINE_DIR, "dataset", "paired_scraped")
    generate_training_pairs(cleaned_dir, paired_dir, antelopev2_dir)
