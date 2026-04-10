"""
Failure Detection & Reinforcement Data Engine

After training, automatically:
  1. Analyze model failures by demographic/expression/pose
  2. Identify systematic weaknesses
  3. Generate targeted data collection queries
  4. Feed back into the training loop

This creates a closed loop:
  Train → Evaluate → Find Failures → Collect More Data → Retrain
"""

import os
import json
import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class FailureDetector:
    """Detects and categorizes model failures after evaluation."""

    def __init__(self, config: dict):
        self.config = config
        self.failure_cfg = config["failure_detection"]
        self.log_dir = config["paths"]["logs_dir"]
        self.failures = []

    def analyze_results(self, eval_results: List[Dict]) -> Dict:
        """Analyze evaluation results to find systematic failures.

        Args:
            eval_results: List of per-sample evaluation dicts with keys:
                - identity_score, expression_change, lpips,
                - metadata (ethnicity, age, gender, expression, pose)

        Returns:
            Analysis dict with failure breakdowns.
        """
        self.failures = []
        failure_counts = defaultdict(int)
        failure_by_category = defaultdict(lambda: defaultdict(list))

        for result in eval_results:
            failure_types = self._classify_failure(result)

            if failure_types:
                self.failures.append({
                    "result": result,
                    "failure_types": failure_types,
                })

                for ft in failure_types:
                    failure_counts[ft] += 1

                    # Track by demographic
                    meta = result.get("metadata", {})
                    for dim in ["ethnicity", "age_group", "gender", "expression"]:
                        val = meta.get(dim, "unknown")
                        failure_by_category[dim][val].append({
                            "failure_type": ft,
                            "score": result.get("identity_score", 0),
                        })

        # Compute failure rates by category
        total = len(eval_results)
        analysis = {
            "total_evaluated": total,
            "total_failures": len(self.failures),
            "failure_rate": len(self.failures) / max(total, 1),
            "failure_counts": dict(failure_counts),
            "failure_by_demographic": {},
            "high_failure_demographics": [],
        }

        # Find demographics with high failure rates
        for dim, groups in failure_by_category.items():
            dim_analysis = {}
            for group, failures in groups.items():
                # Count how many total samples exist for this group
                group_total = sum(
                    1 for r in eval_results
                    if r.get("metadata", {}).get(dim) == group
                )
                failure_rate = len(failures) / max(group_total, 1)
                avg_score = np.mean([f["score"] for f in failures]) if failures else 1.0

                dim_analysis[group] = {
                    "total": group_total,
                    "failures": len(failures),
                    "failure_rate": failure_rate,
                    "avg_identity_score": float(avg_score),
                }

                # Flag high-failure demographics
                if failure_rate > 0.3 and len(failures) >= 3:
                    analysis["high_failure_demographics"].append({
                        dim: group,
                        "failure_rate": failure_rate,
                        "failure_count": len(failures),
                        "avg_score": float(avg_score),
                    })

            analysis["failure_by_demographic"][dim] = dim_analysis

        # Save analysis
        self._save_analysis(analysis)

        logger.info(f"Failure analysis: {len(self.failures)}/{total} failures "
                    f"({analysis['failure_rate']*100:.1f}%)")
        for ft, count in failure_counts.items():
            logger.info(f"  {ft}: {count}")
        for demo in analysis["high_failure_demographics"]:
            logger.info(f"  HIGH FAILURE: {demo}")

        return analysis

    def _classify_failure(self, result: Dict) -> List[str]:
        """Classify what type of failure occurred."""
        failures = []

        identity = result.get("identity_score", 1.0)
        expression = result.get("expression_change", 0.0)
        if isinstance(expression, dict):
            expression = expression.get("total", 0.0)
        lpips = result.get("lpips", 0.0)

        # Identity drift
        if identity < self.failure_cfg["identity_threshold"]:
            failures.append("identity_drift")

        # Weak expression
        if expression < self.failure_cfg["expression_threshold"]:
            failures.append("weak_expression")

        # Artifacts (high LPIPS = very different from source = likely artifacts)
        if lpips > self.failure_cfg["artifact_lpips_threshold"]:
            failures.append("artifacts")

        # Specific sub-failures
        if identity < 0.7:
            failures.append("severe_identity_loss")
        if identity >= self.failure_cfg["identity_threshold"] and expression < 0.003:
            failures.append("identity_preserved_but_no_expression")

        return failures

    def get_reinforcement_queries(self, analysis: Dict) -> List[Dict]:
        """Generate targeted data collection queries from failure analysis.

        Translates failure patterns into search queries that will
        collect more training data for the weak areas.
        """
        queries = []

        for demo in analysis.get("high_failure_demographics", []):
            # Build targeted query
            query_bias = {}
            for key in ["ethnicity", "age_group", "gender", "expression"]:
                if key in demo:
                    query_bias[key.replace("age_group", "age")] = demo[key]

            # Determine which expressions need more data
            failure_types = set()
            for f in self.failures:
                meta = f["result"].get("metadata", {})
                match = all(meta.get(k) == v for k, v in demo.items() if k in meta)
                if match:
                    failure_types.update(f["failure_types"])

            # If identity drift, collect more same-identity pairs
            if "identity_drift" in failure_types:
                queries.append({
                    "bias": query_bias,
                    "priority": "high",
                    "reason": "identity_drift",
                    "images_needed": 100,
                })

            # If weak expression, collect more expressive faces
            if "weak_expression" in failure_types:
                for expr in ["smile", "surprise", "angry"]:
                    query_bias_expr = {**query_bias, "expression": expr}
                    queries.append({
                        "bias": query_bias_expr,
                        "priority": "medium",
                        "reason": "weak_expression",
                        "images_needed": 50,
                    })

        return queries

    def _save_analysis(self, analysis: Dict):
        """Save failure analysis to file."""
        path = os.path.join(self.log_dir, "failure_analysis.json")
        # Make serializable
        serializable = json.loads(json.dumps(analysis, default=str))
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)


class ReinforcementEngine:
    """Coordinates the failure → data → retrain loop."""

    def __init__(self, config: dict, collector, cleaner, cluster_engine):
        self.config = config
        self.collector = collector
        self.cleaner = cleaner
        self.cluster_engine = cluster_engine
        self.failure_detector = FailureDetector(config)

    def run_cycle(self, eval_results: List[Dict]) -> Dict:
        """Run one reinforcement cycle.

        1. Analyze failures
        2. Generate targeted queries
        3. Collect new data
        4. Clean and cluster
        5. Return stats for the next training cycle
        """
        logger.info("\n" + "=" * 60)
        logger.info("REINFORCEMENT DATA ENGINE — CYCLE START")
        logger.info("=" * 60)

        # Step 1: Analyze failures
        analysis = self.failure_detector.analyze_results(eval_results)

        if analysis["failure_rate"] < 0.05:
            logger.info("Failure rate < 5% — skipping reinforcement")
            return {"skipped": True, "reason": "low_failure_rate"}

        # Step 2: Generate queries
        queries = self.failure_detector.get_reinforcement_queries(analysis)
        logger.info(f"Generated {len(queries)} reinforcement queries")

        if not queries:
            return {"skipped": True, "reason": "no_queries"}

        # Step 3: Collect new data
        total_collected = 0
        for q in queries:
            stats = self.collector.collect(
                num_queries=5,
                images_per_query=q.get("images_needed", 50),
                bias=q.get("bias"),
            )
            total_collected += stats["total_downloaded"]

        # Step 4: Clean new data
        clean_stats = self.cleaner.clean_all()

        # Step 5: Re-cluster
        cluster_stats = self.cluster_engine.cluster()

        result = {
            "skipped": False,
            "failure_analysis": analysis,
            "queries_generated": len(queries),
            "images_collected": total_collected,
            "images_after_cleaning": clean_stats["passed"],
            "new_pairs": cluster_stats["paired_samples"],
        }

        logger.info(f"Reinforcement complete: {total_collected} collected → "
                    f"{clean_stats['passed']} cleaned → {cluster_stats['paired_samples']} pairs")

        return result
