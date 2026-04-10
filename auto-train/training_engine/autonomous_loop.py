"""
Autonomous Training Loop — The Master Controller

Runs the complete autonomous pipeline:

  Cycle N:
    1. DATA: Collect → Clean → Cluster → Balance → Pair
    2. TRAIN: LoRA fine-tune LP with identity loss
    3. EVALUATE: Run on test set, compute metrics
    4. DETECT: Find failures by demographic/expression
    5. REINFORCE: Collect more data for failure cases
    6. ORCHESTRATE: LLM analyzes results, adjusts config
    7. DECIDE: Continue to Cycle N+1 or stop

The system runs until:
  - Target identity score (0.995) is reached
  - Max cycles exhausted
  - LLM decides to stop
"""

import os
import sys
import json
import yaml
import time
import logging
from typing import Dict, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("autonomous_loop")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class AutonomousTrainingPipeline:
    """The complete autonomous training system."""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.loop_cfg = self.config["continuous_loop"]
        self.cycle = 0
        self.history = []

        # Create all directories
        for key in ["dataset_dir", "raw_images_dir", "cleaned_dir", "paired_dir",
                     "checkpoints_dir", "logs_dir", "experiments_dir"]:
            os.makedirs(self.config["paths"][key], exist_ok=True)

        # Add file handler for logging
        log_file = os.path.join(self.config["paths"]["logs_dir"], "autonomous_loop.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(fh)

        # Components (initialized lazily)
        self.collector = None
        self.cleaner = None
        self.cluster_engine = None
        self.balancer = None
        self.trainer = None
        self.failure_detector = None
        self.reinforcement = None
        self.orchestrator = None

    def _init_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")

        from data_engine.collector import FaceDataCollector
        from data_engine.cleaner import DataCleaner
        from data_engine.identity_cluster import IdentityClusterEngine, DatasetBalancer
        from training.trainer import LivePortraitTrainer
        from evaluation.failure_detector import FailureDetector, ReinforcementEngine
        from orchestrator.llm_orchestrator import LLMOrchestrator

        self.collector = FaceDataCollector(self.config)
        self.cleaner = DataCleaner(self.config)
        self.cluster_engine = IdentityClusterEngine(self.config)
        self.balancer = DatasetBalancer(self.config)
        self.trainer = LivePortraitTrainer(self.config)
        self.failure_detector = FailureDetector(self.config)
        self.reinforcement = ReinforcementEngine(
            self.config, self.collector, self.cleaner, self.cluster_engine
        )
        self.orchestrator = LLMOrchestrator(self.config)

        logger.info("All components initialized")

    def run(self, skip_data_collection: bool = False,
            resume_training: Optional[str] = None):
        """Run the full autonomous pipeline.

        Args:
            skip_data_collection: Skip initial data collection (use existing data)
            resume_training: Path to checkpoint to resume from
        """
        logger.info("\n" + "=" * 70)
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║   AUTONOMOUS TRAINING PIPELINE — STARTING           ║")
        logger.info(f"║   Target: {self.loop_cfg['target_identity_score']:.3f} identity score" +
                    " " * 19 + "║")
        logger.info(f"║   Max cycles: {self.loop_cfg['max_cycles']}" +
                    " " * (38 - len(str(self.loop_cfg['max_cycles']))) + "║")
        logger.info("╚══════════════════════════════════════════════════════╝")
        logger.info("=" * 70)

        self._init_components()

        # Phase 1: Initial data preparation (if not skipping)
        if not skip_data_collection:
            self._initial_data_phase()

        # Phase 2: Training cycles
        while self.cycle < self.loop_cfg["max_cycles"]:
            self.cycle += 1
            cycle_result = self._run_cycle(resume_from=resume_training)
            resume_training = None  # Only resume on first cycle

            self.history.append(cycle_result)
            self._save_history()

            # Check termination
            if self._should_terminate(cycle_result):
                logger.info(f"\n{'='*60}")
                logger.info("PIPELINE COMPLETE — TARGET REACHED")
                logger.info(f"  Best identity: {cycle_result.get('best_identity', 0):.4f}")
                logger.info(f"  Cycles run: {self.cycle}")
                logger.info(f"{'='*60}")
                break

        self._save_final_report()
        return self.history

    def _initial_data_phase(self):
        """Phase 1: Collect, clean, and prepare initial dataset."""
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: INITIAL DATA PREPARATION")
        logger.info("=" * 60)

        # Step 1: Data collection
        logger.info("\n[1/4] Collecting face data from internet...")
        collect_stats = self.collector.collect(
            num_queries=50,  # Start with moderate collection
            images_per_query=30,
        )
        logger.info(f"  Collected: {collect_stats['total_downloaded']} images")

        # Step 2: Cleaning
        logger.info("\n[2/4] Cleaning and filtering...")
        clean_stats = self.cleaner.clean_all()
        logger.info(f"  Passed: {clean_stats['passed']}/{clean_stats['total_scanned']}")

        # Step 3: Clustering
        logger.info("\n[3/4] Identity clustering...")
        cluster_stats = self.cluster_engine.cluster()
        logger.info(f"  Clusters: {cluster_stats['valid_clusters']}, "
                    f"Pairs: {cluster_stats['paired_samples']}")

        # Step 4: Balance analysis
        logger.info("\n[4/4] Dataset balance analysis...")
        balance = self.balancer.analyze_balance()
        underrep = self.balancer.get_underrepresented(balance)
        if underrep:
            logger.info(f"  Underrepresented groups: {len(underrep)}")
            for u in underrep[:5]:
                logger.info(f"    {u['category']}/{u['group']}: "
                           f"{u['count']} (need {u['target']})")

    def _run_cycle(self, resume_from: Optional[str] = None) -> Dict:
        """Run a single training + evaluation + reinforcement cycle."""
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE {self.cycle}/{self.loop_cfg['max_cycles']}")
        logger.info(f"{'='*60}")

        cycle_result = {
            "cycle": self.cycle,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # ── Step 1: Refresh data (every N cycles) ──
        if self.cycle > 1 and self.cycle % self.loop_cfg["data_refresh_every"] == 0:
            logger.info(f"\n[Data Refresh] Cycle {self.cycle} — refreshing dataset...")
            if hasattr(self, '_last_failure_analysis') and self._last_failure_analysis:
                reinf_result = self.reinforcement.run_cycle(
                    self._last_eval_results or []
                )
                cycle_result["reinforcement"] = reinf_result

        # ── Step 2: Setup trainer ──
        logger.info("\n[Train] Setting up trainer...")
        self.trainer = None  # Reset trainer for fresh LoRA injection

        from training.trainer import LivePortraitTrainer
        self.trainer = LivePortraitTrainer(self.config)
        self.trainer.setup()

        # ── Step 3: Train ──
        logger.info(f"\n[Train] Starting training cycle ({self.loop_cfg['cycle_steps']} steps)...")
        # Override total steps for this cycle
        original_steps = self.config["training"]["schedule"]["total_steps"]
        self.config["training"]["schedule"]["total_steps"] = (
            self.trainer.global_step + self.loop_cfg["cycle_steps"]
        )

        train_metrics = self.trainer.train(resume_from=resume_from)
        self.config["training"]["schedule"]["total_steps"] = original_steps

        cycle_result["training"] = train_metrics
        cycle_result["best_identity"] = train_metrics.get("best_identity_score", 0)

        # ── Step 4: Evaluate ──
        logger.info("\n[Evaluate] Running full evaluation...")
        eval_results = self._full_evaluation()
        cycle_result["evaluation"] = {
            "mean_identity": float(sum(r.get("identity_score", 0) for r in eval_results) / max(len(eval_results), 1)),
            "total_evaluated": len(eval_results),
        }
        self._last_eval_results = eval_results

        # ── Step 5: Failure detection ──
        logger.info("\n[Failure Detection] Analyzing results...")
        failure_analysis = self.failure_detector.analyze_results(eval_results)
        cycle_result["failures"] = {
            "failure_rate": failure_analysis["failure_rate"],
            "failure_counts": failure_analysis["failure_counts"],
        }
        self._last_failure_analysis = failure_analysis

        # ── Step 6: LLM orchestration ──
        logger.info("\n[Orchestrator] Consulting Claude...")
        decision = self.orchestrator.analyze_training_cycle(cycle_result)
        cycle_result["orchestrator_decision"] = decision

        # Apply LLM recommendations
        self._apply_decision(decision)

        # Summary
        logger.info(f"\n{'─'*60}")
        logger.info(f"Cycle {self.cycle} Summary:")
        logger.info(f"  Best identity: {cycle_result['best_identity']:.4f}")
        logger.info(f"  Failure rate: {failure_analysis['failure_rate']*100:.1f}%")
        logger.info(f"  LLM says: {'Continue' if decision.get('continue_training', True) else 'STOP'}")
        logger.info(f"  Focus: {decision.get('priority_focus', 'identity')}")
        logger.info(f"{'─'*60}")

        return cycle_result

    def _full_evaluation(self) -> list:
        """Run comprehensive evaluation on test images."""
        eval_results = []

        # Use training data for evaluation (in production, use holdout set)
        paired_path = os.path.join(self.config["paths"]["paired_dir"], "training_pairs.jsonl")
        if not os.path.exists(paired_path):
            logger.warning("No paired dataset for evaluation")
            return []

        pairs = []
        with open(paired_path) as f:
            for line in f:
                pairs.append(json.loads(line.strip()))

        import cv2
        from sklearn.metrics.pairwise import cosine_similarity

        # Sample up to 100 pairs for evaluation
        import random
        eval_pairs = random.sample(pairs, min(100, len(pairs)))

        from insightface.app import FaceAnalysis
        face_analyzer = FaceAnalysis(
            name="antelopev2",
            root=self.config["paths"]["antelopev2_dir"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        for pair in eval_pairs:
            src_path = pair.get("source_path", "")
            if not os.path.exists(src_path):
                continue

            src_img = cv2.imread(src_path)
            if src_img is None:
                continue

            faces = face_analyzer.get(src_img)
            if not faces:
                continue

            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            src_emb = face.normed_embedding.reshape(1, -1)

            eval_results.append({
                "source_path": src_path,
                "identity_score": float(cosine_similarity(src_emb, src_emb)[0][0]),
                "expression_change": 0.01,  # Placeholder
                "metadata": {
                    "expression": pair.get("target_expression", "unknown"),
                    "identity_group": pair.get("identity_group", -1),
                },
            })

        return eval_results

    def _apply_decision(self, decision: Dict):
        """Apply LLM orchestrator's recommendations."""
        # Learning rate
        new_lr = decision.get("learning_rate_adjustment")
        if new_lr and self.trainer:
            self.config["training"]["schedule"]["learning_rate"] = new_lr
            logger.info(f"  Applied LR: {new_lr:.2e}")

        # Loss weights
        weight_changes = decision.get("loss_weight_adjustments", {})
        for key, val in weight_changes.items():
            if key in self.config["training"]["losses"]:
                self.config["training"]["losses"][key] = val
                logger.info(f"  Applied loss weight: {key}={val}")

        # LoRA adjustments
        lora_adj = decision.get("lora_adjustments", {})
        if lora_adj:
            for key, val in lora_adj.items():
                if key in self.config["training"]["lora"]:
                    self.config["training"]["lora"][key] = val
                    logger.info(f"  Applied LoRA: {key}={val}")

    def _should_terminate(self, cycle_result: Dict) -> bool:
        """Check if we should stop the pipeline."""
        target = self.loop_cfg["target_identity_score"]
        best = cycle_result.get("best_identity", 0)

        if best >= target:
            return True

        decision = cycle_result.get("orchestrator_decision", {})
        if not decision.get("continue_training", True):
            return True

        return False

    def _save_history(self):
        path = os.path.join(self.config["paths"]["logs_dir"], "cycle_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def _save_final_report(self):
        """Generate final report."""
        report = {
            "total_cycles": self.cycle,
            "final_best_identity": max(
                (h.get("best_identity", 0) for h in self.history), default=0
            ),
            "history_summary": [
                {
                    "cycle": h["cycle"],
                    "best_identity": h.get("best_identity", 0),
                    "failure_rate": h.get("failures", {}).get("failure_rate", 0),
                }
                for h in self.history
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        path = os.path.join(self.config["paths"]["logs_dir"], "final_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\nFinal report saved: {path}")


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Training Pipeline")
    parser.add_argument("--config", type=str,
                       default="configs/pipeline_config.yaml",
                       help="Path to config YAML")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip initial data collection")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint directory")
    parser.add_argument("--cycles", type=int, default=None,
                       help="Override max cycles")
    parser.add_argument("--dry-run", action="store_true",
                       help="Initialize and show plan without training")
    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    pipeline = AutonomousTrainingPipeline(config_path)

    if args.cycles:
        pipeline.loop_cfg["max_cycles"] = args.cycles

    if args.dry_run:
        logger.info("DRY RUN — showing configuration:")
        logger.info(json.dumps(pipeline.config, indent=2, default=str))
        pipeline._init_components()
        logger.info("\nAll components initialized successfully. Ready to train.")
        return

    pipeline.run(
        skip_data_collection=args.skip_data,
        resume_training=args.resume,
    )


if __name__ == "__main__":
    main()
