#!/usr/bin/env python3
"""
RLHF System — Reinforcement Learning from Human Feedback

3 Layers:
  Layer 1: Rich feedback capture (save everything with like/dislike)
  Layer 2: Adaptive driver selection (statistics-based, works from day 1)
  Layer 3: Reward model (neural network, needs 100+ feedbacks)

Usage:
  rlhf = RLHFSystem()

  # After generating a result:
  rlhf.record_attempt(request_id, expression, driver, intensity,
                       identity_score, expression_change, source_emb, result_emb)

  # When user clicks like/dislike:
  rlhf.record_feedback(request_id, liked=True)

  # When generating next result, ask RLHF for best strategy:
  strategy = rlhf.get_best_strategy(expression, source_emb)
  # strategy = {"driver": "smile_best_2.jpg", "intensity": 1.2, "confidence": 0.85}
"""

import os
import json
import time
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rlhf")

PROD_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_DIR = os.path.join(PROD_DIR, "feedback")
RLHF_DIR = os.path.join(PROD_DIR, "rlhf_data")


class RLHFSystem:
    """
    Reinforcement Learning from Human Feedback.

    Immediately useful: tracks which (driver, intensity) combos get likes
    and steers future generations toward liked configurations.
    """

    def __init__(self):
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        os.makedirs(RLHF_DIR, exist_ok=True)

        self.attempts_file = os.path.join(RLHF_DIR, "attempts.jsonl")
        self.feedback_file = os.path.join(RLHF_DIR, "feedback_rich.jsonl")
        self.stats_file = os.path.join(RLHF_DIR, "driver_stats.json")
        self.reward_model_path = os.path.join(RLHF_DIR, "reward_model.pt")

        # In-memory caches
        self.pending_attempts = {}  # request_id → attempt data
        self.driver_stats = self._load_stats()
        self.reward_model = None

        # Try to load reward model
        self._load_reward_model()

        total_feedback = sum(
            s.get("total", 0) for s in self.driver_stats.values()
        )
        logger.info(f"RLHF System initialized. {total_feedback} total feedbacks loaded.")

    # ══════════════════════════════════════════════════════════
    # LAYER 1: Rich Feedback Capture
    # ══════════════════════════════════════════════════════════

    def record_attempt(
        self,
        request_id: str,
        expression: str,
        driver_used: str,
        intensity_used: float,
        identity_score: float,
        expression_change: float,
        source_embedding: Optional[List[float]] = None,
        result_embedding: Optional[List[float]] = None,
        use_retargeting: bool = False,
    ):
        """Record an attempt (before user gives feedback)."""
        attempt = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "expression": expression,
            "driver_used": driver_used,
            "intensity_used": round(intensity_used, 3),
            "identity_score": round(identity_score, 4),
            "expression_change": round(expression_change, 4),
            "use_retargeting": use_retargeting,
        }

        # Store embeddings for reward model training
        if source_embedding is not None:
            attempt["source_embedding"] = [round(float(x), 6) for x in source_embedding[:32]]  # First 32 dims to save space
        if result_embedding is not None:
            attempt["result_embedding"] = [round(float(x), 6) for x in result_embedding[:32]]

        self.pending_attempts[request_id] = attempt

        # Also persist to disk
        with open(self.attempts_file, "a") as f:
            f.write(json.dumps(attempt) + "\n")

    def record_feedback(self, request_id: str, liked: bool) -> Dict:
        """
        Record user feedback and UPDATE driver statistics.

        This is where the learning happens:
        - Updates per-driver like rates
        - Updates per-expression statistics
        - Triggers reward model retraining if enough data

        Returns feedback summary and any changes made.
        """
        attempt = self.pending_attempts.get(request_id)

        if attempt is None:
            # Try to find in recent attempts file
            attempt = self._find_recent_attempt(request_id)

        if attempt is None:
            logger.warning(f"No attempt found for request_id={request_id}")
            return {"status": "no_attempt_found"}

        # Create rich feedback record
        feedback = {
            **attempt,
            "liked": liked,
            "feedback_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Save to disk
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback) + "\n")

        # UPDATE DRIVER STATISTICS — this is the immediate learning
        changes = self._update_stats(feedback)

        # Clean up pending
        self.pending_attempts.pop(request_id, None)

        # Check if we should retrain reward model
        total = self._total_feedback_count()
        if total > 0 and total % 50 == 0:  # Retrain every 50 feedbacks
            self._train_reward_model()

        return {
            "status": "recorded",
            "liked": liked,
            "changes": changes,
            "total_feedbacks": total,
        }

    # ══════════════════════════════════════════════════════════
    # LAYER 2: Adaptive Driver Selection (Statistics-Based)
    # ══════════════════════════════════════════════════════════

    def get_best_strategy(
        self,
        expression: str,
        source_embedding: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Get the best (driver, intensity) for this expression based on feedback.

        Uses:
        1. Driver like-rate statistics (always available)
        2. Reward model prediction (if trained)

        Returns recommended strategy with confidence.
        """
        expr_stats = {}

        # Collect stats for this expression
        for key, stats in self.driver_stats.items():
            if key.startswith(f"{expression}:"):
                expr_stats[key] = stats

        if not expr_stats:
            # No feedback yet — return default (no preference)
            return {
                "has_preference": False,
                "driver_ranking": [],
                "recommended_intensity": None,
                "confidence": 0.0,
                "reason": "No feedback data yet",
            }

        # Rank drivers by like rate (with confidence weighting)
        rankings = []
        for key, stats in expr_stats.items():
            total = stats.get("total", 0)
            likes = stats.get("likes", 0)

            if total == 0:
                continue

            like_rate = likes / total

            # Wilson score for confidence interval
            # More data = tighter interval = higher confidence
            z = 1.96  # 95% confidence
            n = total
            phat = like_rate
            confidence = (
                (phat + z*z/(2*n) - z * np.sqrt((phat*(1-phat) + z*z/(4*n))/n))
                / (1 + z*z/n)
            )

            # Parse key: "expression:driver:intensity"
            parts = key.split(":")
            driver = parts[1] if len(parts) > 1 else "unknown"
            intensity = float(parts[2]) if len(parts) > 2 else 1.0

            rankings.append({
                "driver": driver,
                "intensity": intensity,
                "like_rate": round(like_rate, 3),
                "total": total,
                "likes": likes,
                "confidence_score": round(float(confidence), 3),
            })

        # Sort by confidence-weighted like rate
        rankings.sort(key=lambda x: x["confidence_score"], reverse=True)

        # Use reward model if available
        reward_prediction = None
        if self.reward_model is not None and source_embedding is not None:
            reward_prediction = self._predict_reward(expression, source_embedding)

        best = rankings[0] if rankings else None

        return {
            "has_preference": len(rankings) > 0,
            "driver_ranking": rankings[:5],
            "recommended_driver": best["driver"] if best else None,
            "recommended_intensity": best["intensity"] if best else None,
            "confidence": best["confidence_score"] if best else 0.0,
            "reason": self._explain_ranking(rankings),
            "reward_prediction": reward_prediction,
            "total_feedback_for_expression": sum(r["total"] for r in rankings),
        }

    def get_driver_order(self, expression: str) -> Optional[List[Tuple[str, float]]]:
        """
        Get re-ordered driver list for this expression based on feedback.

        Returns list of (driver_name, intensity) tuples, best first.
        Returns None if not enough data to have a preference.
        """
        strategy = self.get_best_strategy(expression)

        if not strategy["has_preference"]:
            return None

        if strategy["total_feedback_for_expression"] < 5:
            return None  # Not enough data to be confident

        # Return drivers ordered by like rate
        result = []
        for r in strategy["driver_ranking"]:
            result.append((r["driver"], r["intensity"]))

        return result

    def should_demote_driver(self, expression: str, driver: str) -> bool:
        """Check if a driver should be deprioritized due to poor feedback."""
        for key, stats in self.driver_stats.items():
            if key.startswith(f"{expression}:{driver}:"):
                total = stats.get("total", 0)
                likes = stats.get("likes", 0)
                if total >= 5 and likes / total < 0.3:
                    return True  # Less than 30% like rate with 5+ samples
        return False

    # ══════════════════════════════════════════════════════════
    # LAYER 3: Reward Model
    # ══════════════════════════════════════════════════════════

    def _train_reward_model(self):
        """
        Train a simple reward model from accumulated feedback.

        Input features:
          - identity_score
          - expression_change
          - intensity_used
          - expression (one-hot)
          - use_retargeting (bool)

        Output: probability of like (0-1)
        """
        import torch
        import torch.nn as nn

        # Load all feedback
        feedbacks = self._load_all_feedback()
        if len(feedbacks) < 30:
            logger.info(f"Only {len(feedbacks)} feedbacks — need 30+ for reward model")
            return

        logger.info(f"Training reward model on {len(feedbacks)} feedbacks...")

        # Build features
        expressions_list = ["smile", "open_smile", "surprise", "sad", "angry"]

        features = []
        labels = []
        for fb in feedbacks:
            feat = [
                fb.get("identity_score", 0.9),
                fb.get("expression_change", 0.01),
                fb.get("intensity_used", 1.0),
                1.0 if fb.get("use_retargeting", False) else 0.0,
            ]
            # One-hot expression
            expr = fb.get("expression", "smile")
            for e in expressions_list:
                feat.append(1.0 if expr == e else 0.0)

            features.append(feat)
            labels.append(1.0 if fb.get("liked", False) else 0.0)

        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

        # Simple 2-layer network
        input_dim = X.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Train
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred = model(X)
            accuracy = ((pred > 0.5).float() == y).float().mean()

        logger.info(f"  Reward model trained. Accuracy: {accuracy:.1%}")

        # Save
        torch.save({
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "expressions": expressions_list,
            "num_feedbacks": len(feedbacks),
            "accuracy": float(accuracy),
        }, self.reward_model_path)

        self.reward_model = model
        self._reward_expressions = expressions_list

        logger.info(f"  Saved to {self.reward_model_path}")

    def _load_reward_model(self):
        """Load pre-trained reward model if exists."""
        import torch
        import torch.nn as nn

        if not os.path.exists(self.reward_model_path):
            return

        try:
            checkpoint = torch.load(self.reward_model_path, weights_only=True)
            input_dim = checkpoint["input_dim"]

            model = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            self.reward_model = model
            self._reward_expressions = checkpoint["expressions"]
            logger.info(
                f"  Loaded reward model (trained on {checkpoint['num_feedbacks']} feedbacks, "
                f"accuracy={checkpoint['accuracy']:.1%})"
            )
        except Exception as e:
            logger.warning(f"  Could not load reward model: {e}")

    def _predict_reward(self, expression: str, source_embedding: np.ndarray) -> Optional[Dict]:
        """Use reward model to predict best settings for this expression."""
        if self.reward_model is None:
            return None

        import torch

        expressions_list = self._reward_expressions

        # Try different configurations and predict which will be liked
        configs = []
        for identity in [0.90, 0.93, 0.95, 0.97, 0.99]:
            for expr_change in [0.01, 0.02, 0.03, 0.05]:
                for intensity in [0.5, 1.0, 1.5, 1.8]:
                    for retarget in [False, True]:
                        feat = [identity, expr_change, intensity, 1.0 if retarget else 0.0]
                        for e in expressions_list:
                            feat.append(1.0 if expression == e else 0.0)
                        configs.append({
                            "features": feat,
                            "identity": identity,
                            "expr_change": expr_change,
                            "intensity": intensity,
                            "retarget": retarget,
                        })

        X = torch.tensor([c["features"] for c in configs], dtype=torch.float32)

        with torch.no_grad():
            predictions = self.reward_model(X).squeeze().numpy()

        # Find best configuration
        best_idx = np.argmax(predictions)
        best = configs[best_idx]

        return {
            "predicted_like_probability": round(float(predictions[best_idx]), 3),
            "recommended_intensity": best["intensity"],
            "recommended_retarget": best["retarget"],
            "optimal_identity_range": f"{best['identity']:.0%}+",
            "optimal_expression_change": f"{best['expr_change']:.3f}+",
        }

    # ══════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════

    def _update_stats(self, feedback: Dict) -> Dict:
        """Update driver statistics from a feedback record."""
        expression = feedback.get("expression", "unknown")
        driver = feedback.get("driver_used", "unknown")
        intensity = feedback.get("intensity_used", 1.0)
        liked = feedback.get("liked", False)

        # Key: expression:driver:intensity_bucket
        intensity_bucket = round(intensity * 2) / 2  # Round to nearest 0.5
        key = f"{expression}:{driver}:{intensity_bucket}"

        if key not in self.driver_stats:
            self.driver_stats[key] = {"total": 0, "likes": 0, "dislikes": 0}

        self.driver_stats[key]["total"] += 1
        if liked:
            self.driver_stats[key]["likes"] += 1
        else:
            self.driver_stats[key]["dislikes"] += 1

        # Save stats
        self._save_stats()

        # Check if this triggers any changes
        changes = {}
        stats = self.driver_stats[key]
        total = stats["total"]
        like_rate = stats["likes"] / total if total > 0 else 0

        if total >= 5:
            if like_rate < 0.3:
                changes["action"] = "demote"
                changes["reason"] = f"{driver} at {intensity_bucket}x has {like_rate:.0%} like rate ({total} samples) — will be deprioritized"
                logger.info(f"  RLHF: DEMOTING {key} (like_rate={like_rate:.0%})")
            elif like_rate > 0.7:
                changes["action"] = "promote"
                changes["reason"] = f"{driver} at {intensity_bucket}x has {like_rate:.0%} like rate — promoted to top"
                logger.info(f"  RLHF: PROMOTING {key} (like_rate={like_rate:.0%})")

        return changes

    def _load_stats(self) -> Dict:
        """Load driver statistics from disk."""
        if os.path.exists(self.stats_file):
            with open(self.stats_file) as f:
                return json.load(f)
        return {}

    def _save_stats(self):
        """Save driver statistics to disk."""
        with open(self.stats_file, "w") as f:
            json.dump(self.driver_stats, f, indent=2)

    def _load_all_feedback(self) -> List[Dict]:
        """Load all rich feedback records."""
        feedbacks = []
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file) as f:
                for line in f:
                    try:
                        feedbacks.append(json.loads(line))
                    except:
                        pass
        return feedbacks

    def _find_recent_attempt(self, request_id: str) -> Optional[Dict]:
        """Search recent attempts for a request_id."""
        if not os.path.exists(self.attempts_file):
            return None

        # Read last 100 lines
        with open(self.attempts_file) as f:
            lines = f.readlines()

        for line in reversed(lines[-100:]):
            try:
                attempt = json.loads(line)
                if attempt.get("request_id") == request_id:
                    return attempt
            except:
                pass
        return None

    def _total_feedback_count(self) -> int:
        """Total feedbacks across all drivers."""
        return sum(s.get("total", 0) for s in self.driver_stats.values())

    def _explain_ranking(self, rankings: List[Dict]) -> str:
        """Human-readable explanation of why drivers are ranked this way."""
        if not rankings:
            return "No data yet"

        best = rankings[0]
        total = sum(r["total"] for r in rankings)

        if total < 5:
            return f"Only {total} feedbacks — rankings are preliminary"

        parts = []
        parts.append(f"Based on {total} user feedbacks:")
        for r in rankings[:3]:
            parts.append(
                f"  {r['driver']} at {r['intensity']:.1f}x: "
                f"{r['like_rate']:.0%} liked ({r['likes']}/{r['total']})"
            )

        return "\n".join(parts)

    def get_summary(self) -> Dict:
        """Get full RLHF system summary."""
        feedbacks = self._load_all_feedback()
        total = len(feedbacks)
        liked = sum(1 for f in feedbacks if f.get("liked"))

        per_expression = defaultdict(lambda: {"total": 0, "liked": 0})
        for fb in feedbacks:
            expr = fb.get("expression", "unknown")
            per_expression[expr]["total"] += 1
            if fb.get("liked"):
                per_expression[expr]["liked"] += 1

        return {
            "total_feedbacks": total,
            "overall_like_rate": round(liked / total, 3) if total > 0 else 0,
            "per_expression": dict(per_expression),
            "driver_stats": self.driver_stats,
            "reward_model_loaded": self.reward_model is not None,
            "reward_model_trained_on": (
                self._load_all_feedback().__len__()
                if self.reward_model is not None else 0
            ),
        }
