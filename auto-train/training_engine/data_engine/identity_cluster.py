"""
Identity Clustering Engine

Groups images of the same person using ArcFace embedding similarity.
Uses DBSCAN for clustering (no need to specify k).
Builds paired datasets: same person, different expressions.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class IdentityClusterEngine:
    """Cluster face images by identity using ArcFace embeddings."""

    def __init__(self, config: dict):
        self.config = config
        self.cluster_cfg = config["identity_clustering"]
        self.cleaned_dir = config["paths"]["cleaned_dir"]
        self.paired_dir = config["paths"]["paired_dir"]
        os.makedirs(self.paired_dir, exist_ok=True)

    def cluster(self) -> Dict:
        """Run identity clustering on all cleaned images.

        Returns clustering stats and saves paired dataset.
        """
        # Load annotations with embeddings
        annotations = self._load_annotations_with_embeddings()
        if not annotations:
            logger.warning("No annotations found. Run data cleaning first.")
            return {"clusters": 0, "paired_samples": 0}

        logger.info(f"Clustering {len(annotations)} images...")

        embeddings = np.array([a["embedding"] for a in annotations])

        # Compute pairwise cosine similarity
        sim_matrix = cosine_similarity(embeddings)

        # Convert to distance matrix for DBSCAN
        distance_matrix = 1.0 - sim_matrix

        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.cluster_cfg["dbscan_eps"],
            min_samples=self.cluster_cfg["dbscan_min_samples"],
            metric="precomputed",
        ).fit(distance_matrix)

        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Build identity groups
        identity_groups = {}
        noise_count = 0
        for idx, label in enumerate(labels):
            if label == -1:
                noise_count += 1
                continue
            if label not in identity_groups:
                identity_groups[label] = []
            identity_groups[label].append(annotations[idx])

        # Filter by cluster size
        min_size = self.cluster_cfg["min_cluster_size"]
        max_size = self.cluster_cfg["max_cluster_size"]
        valid_groups = {}
        for gid, members in identity_groups.items():
            if len(members) >= min_size:
                # Cap size
                if len(members) > max_size:
                    members = sorted(members, key=lambda x: x.get("metrics", {}).get("quality_score", 0), reverse=True)
                    members = members[:max_size]
                valid_groups[gid] = members

        logger.info(f"Found {n_clusters} clusters, {len(valid_groups)} valid "
                    f"(>={min_size} images), {noise_count} noise points")

        # Build paired dataset
        paired_samples = self._build_paired_dataset(valid_groups)

        # Save cluster info
        self._save_cluster_info(valid_groups)

        stats = {
            "total_images": len(annotations),
            "total_clusters": n_clusters,
            "valid_clusters": len(valid_groups),
            "noise_points": noise_count,
            "paired_samples": len(paired_samples),
            "images_per_cluster": {
                "mean": np.mean([len(v) for v in valid_groups.values()]) if valid_groups else 0,
                "min": min(len(v) for v in valid_groups.values()) if valid_groups else 0,
                "max": max(len(v) for v in valid_groups.values()) if valid_groups else 0,
            },
        }
        return stats

    def _build_paired_dataset(self, identity_groups: Dict) -> List[Dict]:
        """Build training pairs: (neutral_source, expression_target) for same identity.

        For each identity with multiple expressions:
        - Find neutral/near-neutral images as sources
        - Pair with expressive images as targets
        """
        pairs = []

        for gid, members in identity_groups.items():
            # Separate by expression
            by_expression = {}
            for m in members:
                expr = m.get("expression", "unknown")
                if expr not in by_expression:
                    by_expression[expr] = []
                by_expression[expr].append(m)

            # Source images: prefer neutral, or least expressive
            sources = by_expression.get("neutral", [])
            if not sources:
                # Use any image as source — training will use LP to create the pair
                sources = members[:3]

            # Target images: all non-neutral expressions
            for expr, targets in by_expression.items():
                if expr in ("neutral", "unknown"):
                    continue
                for src in sources:
                    for tgt in targets:
                        if src["path"] == tgt.get("path", ""):
                            continue
                        pairs.append({
                            "source_path": src.get("cleaned_path", src["path"]),
                            "target_path": tgt.get("cleaned_path", tgt["path"]),
                            "identity_group": int(gid),
                            "source_expression": src.get("expression", "neutral"),
                            "target_expression": tgt.get("expression", "unknown"),
                        })

        # Also create synthetic pairs using LP
        # For identities with only neutral/similar expressions,
        # we'll use LP to generate the target during training
        for gid, members in identity_groups.items():
            expressions = set(m.get("expression", "unknown") for m in members)
            if len(expressions) <= 1:
                # Only one expression type — mark for synthetic pair generation
                for m in members[:5]:  # Limit synthetic pairs per identity
                    for target_expr in ["smile", "surprise", "angry"]:
                        pairs.append({
                            "source_path": m.get("cleaned_path", m["path"]),
                            "target_path": None,  # Will be generated by LP
                            "identity_group": int(gid),
                            "source_expression": m.get("expression", "neutral"),
                            "target_expression": target_expr,
                            "synthetic": True,
                        })

        # Save pairs
        pairs_path = os.path.join(self.paired_dir, "training_pairs.jsonl")
        with open(pairs_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        logger.info(f"Built {len(pairs)} training pairs "
                    f"({sum(1 for p in pairs if p.get('synthetic'))} synthetic)")
        return pairs

    def _load_annotations_with_embeddings(self) -> List[Dict]:
        """Load annotations that have embeddings."""
        # First check for embeddings file
        emb_path = os.path.join(self.cleaned_dir, "embeddings.npz")
        ann_path = os.path.join(self.cleaned_dir, "annotations.jsonl")

        if not os.path.exists(ann_path):
            return []

        annotations = []
        with open(ann_path) as f:
            for line in f:
                a = json.loads(line.strip())
                annotations.append(a)

        # If embeddings aren't in annotations, compute them
        if annotations and "embedding" not in annotations[0]:
            logger.info("Computing embeddings for clustering...")
            annotations = self._compute_embeddings(annotations)

        return [a for a in annotations if "embedding" in a]

    def _compute_embeddings(self, annotations: List[Dict]) -> List[Dict]:
        """Compute ArcFace embeddings for images that don't have them."""
        from insightface.app import FaceAnalysis

        face_analyzer = FaceAnalysis(
            name="antelopev2",
            root=self.config["paths"]["antelopev2_dir"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        import cv2
        for i, ann in enumerate(annotations):
            path = ann.get("cleaned_path", ann.get("path", ""))
            if not os.path.exists(path):
                continue

            img = cv2.imread(path)
            if img is None:
                continue

            faces = face_analyzer.get(img)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                    ann["embedding"] = face.normed_embedding.tolist()

            if (i + 1) % 100 == 0:
                logger.info(f"  Computed embeddings: {i+1}/{len(annotations)}")

        return annotations

    def _save_cluster_info(self, valid_groups: Dict):
        info_path = os.path.join(self.paired_dir, "cluster_info.json")
        summary = {}
        for gid, members in valid_groups.items():
            summary[str(gid)] = {
                "count": len(members),
                "expressions": list(set(m.get("expression", "?") for m in members)),
                "age_range": [
                    min(m.get("age", 0) for m in members),
                    max(m.get("age", 0) for m in members),
                ],
                "paths": [m.get("cleaned_path", m.get("path", "")) for m in members],
            }
        with open(info_path, "w") as f:
            json.dump(summary, f, indent=2)


class DatasetBalancer:
    """Ensures balanced distribution across demographics."""

    def __init__(self, config: dict):
        self.config = config
        self.balance_cfg = config["dataset_balancing"]
        self.cleaned_dir = config["paths"]["cleaned_dir"]

    def analyze_balance(self) -> Dict:
        """Analyze current dataset distribution."""
        ann_path = os.path.join(self.cleaned_dir, "annotations.jsonl")
        if not os.path.exists(ann_path):
            return {"error": "No annotations found"}

        annotations = []
        with open(ann_path) as f:
            for line in f:
                annotations.append(json.loads(line.strip()))

        distribution = {
            "total": len(annotations),
            "by_expression": {},
            "by_age_group": {},
            "by_gender": {},
        }

        for a in annotations:
            expr = a.get("expression", "unknown")
            distribution["by_expression"][expr] = distribution["by_expression"].get(expr, 0) + 1

            age_group = a.get("age_group", "unknown")
            distribution["by_age_group"][age_group] = distribution["by_age_group"].get(age_group, 0) + 1

            gender = a.get("gender", "unknown")
            distribution["by_gender"][gender] = distribution["by_gender"].get(gender, 0) + 1

        # Compute imbalance ratios
        for category in ["by_expression", "by_age_group", "by_gender"]:
            counts = list(distribution[category].values())
            if counts and min(counts) > 0:
                distribution[f"{category}_imbalance"] = max(counts) / min(counts)
            else:
                distribution[f"{category}_imbalance"] = float('inf')

        return distribution

    def get_underrepresented(self, distribution: Dict) -> List[Dict]:
        """Identify underrepresented demographics for targeted collection."""
        underrep = []
        max_ratio = self.balance_cfg["max_imbalance_ratio"]

        for category in ["by_expression", "by_age_group", "by_gender"]:
            counts = distribution.get(category, {})
            if not counts:
                continue
            avg = np.mean(list(counts.values()))
            for group, count in counts.items():
                if count < avg / max_ratio:
                    underrep.append({
                        "category": category.replace("by_", ""),
                        "group": group,
                        "count": count,
                        "target": int(avg),
                        "deficit": int(avg - count),
                    })

        return underrep
