"""
LLM Orchestrator — Claude-Powered Experiment Planner

Uses Claude to:
  1. Analyze training metrics and failure patterns
  2. Recommend hyperparameter adjustments
  3. Generate new data collection strategies
  4. Design experiment variations
  5. Make go/no-go decisions on training cycles

This is the "brain" of the autonomous system — it reads metrics,
understands what's working and what isn't, and makes decisions.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """Claude-powered autonomous experiment orchestrator."""

    def __init__(self, config: dict, api_key: Optional[str] = None):
        self.config = config
        self.orch_cfg = config["orchestrator"]
        self.log_dir = config["paths"]["logs_dir"]
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        self._decisions_log = []

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude API for analysis."""
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY — using rule-based fallback")
            return ""

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.orch_cfg.get("model", "claude-sonnet-4-20250514"),
                max_tokens=self.orch_cfg.get("max_tokens", 4096),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return ""

    def analyze_training_cycle(self, cycle_metrics: Dict) -> Dict:
        """Analyze a completed training cycle and recommend next steps.

        Returns a decision dict with recommended actions.
        """
        system_prompt = """You are an AI research engineer optimizing a face expression editing system.
You analyze training metrics and make precise, data-driven recommendations.

The system uses LivePortrait (pixel warping) with LoRA fine-tuning.
Primary metric: ArcFace identity preservation (target: 0.995).
Secondary metric: Expression change strength (target: > 0.015).

Respond with a JSON object containing your recommendations. No other text."""

        user_prompt = f"""Training cycle results:
{json.dumps(cycle_metrics, indent=2, default=str)}

Current configuration:
- Learning rate: {self.config['training']['schedule']['learning_rate']}
- Identity loss weight: {self.config['training']['losses']['identity_loss']}
- Expression loss weight: {self.config['training']['losses']['expression_loss']}
- LoRA rank: {self.config['training']['lora']['rank']}
- Batch size: {self.config['training']['schedule']['batch_size']}

Analyze the results and recommend:
1. "continue_training": bool — should we continue or stop?
2. "learning_rate_adjustment": float — new LR (or null to keep)
3. "loss_weight_adjustments": dict — any loss weight changes
4. "data_actions": list — data collection/cleaning actions needed
5. "lora_adjustments": dict — rank/alpha changes if needed
6. "reasoning": string — brief explanation of your reasoning
7. "priority_focus": string — what the next cycle should focus on

Return ONLY valid JSON."""

        response = self._call_claude(system_prompt, user_prompt)

        if response:
            try:
                decision = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    try:
                        decision = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        decision = self._rule_based_decision(cycle_metrics)
                else:
                    decision = self._rule_based_decision(cycle_metrics)
        else:
            decision = self._rule_based_decision(cycle_metrics)

        # Log decision
        self._log_decision(cycle_metrics, decision)
        return decision

    def design_experiment(self, current_state: Dict) -> Dict:
        """Design the next experiment variation."""
        system_prompt = """You are designing experiments for a face expression editing system.
Your goal is to find the optimal training configuration to reach 99.5% ArcFace identity preservation.

Respond with a JSON experiment specification. No other text."""

        user_prompt = f"""Current system state:
{json.dumps(current_state, indent=2, default=str)}

Design an experiment that addresses the current weaknesses.
Include:
1. "experiment_name": descriptive name
2. "hypothesis": what you expect to happen
3. "config_changes": specific config changes to make
4. "success_criteria": how to know if it worked
5. "estimated_steps": how many training steps
6. "priority": "high", "medium", or "low"

Return ONLY valid JSON."""

        response = self._call_claude(system_prompt, user_prompt)

        if response:
            try:
                experiment = json.loads(response)
            except (json.JSONDecodeError, Exception):
                experiment = self._default_experiment(current_state)
        else:
            experiment = self._default_experiment(current_state)

        return experiment

    def generate_data_strategy(self, failure_analysis: Dict,
                                dataset_stats: Dict) -> Dict:
        """Generate targeted data collection strategy from failures."""
        system_prompt = """You are a dataset architect for a face expression AI system.
Analyze failure patterns and design targeted data collection strategies.
Respond with a JSON strategy. No other text."""

        user_prompt = f"""Failure analysis:
{json.dumps(failure_analysis, indent=2, default=str)}

Current dataset statistics:
{json.dumps(dataset_stats, indent=2, default=str)}

Design a data collection strategy:
1. "queries": list of search query objects with "query", "ethnicity", "age", "expression"
2. "priority_demographics": which demographics need more data
3. "target_images_per_category": how many images to collect
4. "cleaning_threshold_adjustments": any quality threshold changes
5. "reasoning": why this strategy

Return ONLY valid JSON."""

        response = self._call_claude(system_prompt, user_prompt)

        if response:
            try:
                strategy = json.loads(response)
            except (json.JSONDecodeError, Exception):
                strategy = self._default_data_strategy(failure_analysis)
        else:
            strategy = self._default_data_strategy(failure_analysis)

        return strategy

    # ═══════════════════════════════════════════════════════════
    # RULE-BASED FALLBACK (when Claude API unavailable)
    # ═══════════════════════════════════════════════════════════

    def _rule_based_decision(self, metrics: Dict) -> Dict:
        """Fallback decision logic when Claude API is unavailable."""
        decision = {
            "continue_training": True,
            "learning_rate_adjustment": None,
            "loss_weight_adjustments": {},
            "data_actions": [],
            "lora_adjustments": {},
            "reasoning": "Rule-based decision (Claude API unavailable)",
            "priority_focus": "identity_preservation",
        }

        best_id = metrics.get("best_identity_score", 0)
        current_id = metrics.get("mean_identity", 0)
        failure_rate = metrics.get("failure_rate", 0)

        # Learning rate adjustments
        if current_id > 0.98:
            # Close to target — reduce LR for fine-grained improvements
            current_lr = self.config["training"]["schedule"]["learning_rate"]
            decision["learning_rate_adjustment"] = current_lr * 0.5
            decision["reasoning"] = "Near target — reducing LR for precision"

        elif current_id < 0.93:
            # Far from target — ensure LR isn't too low
            current_lr = self.config["training"]["schedule"]["learning_rate"]
            if current_lr < 1e-4:
                decision["learning_rate_adjustment"] = 1e-4
                decision["reasoning"] = "Identity too low — boosting LR"

        # Loss weight adjustments
        if current_id < 0.95:
            decision["loss_weight_adjustments"] = {
                "identity_loss": 15.0,  # Increase identity weight
                "expression_loss": 3.0,  # Decrease expression weight
            }
            decision["priority_focus"] = "identity_preservation"
        elif current_id > 0.98 and failure_rate < 0.1:
            decision["loss_weight_adjustments"] = {
                "expression_loss": 7.0,  # Now focus on expression
            }
            decision["priority_focus"] = "expression_quality"

        # Data actions
        if failure_rate > 0.2:
            decision["data_actions"].append("run_reinforcement_collection")
        if metrics.get("dataset_size", 0) < 10000:
            decision["data_actions"].append("expand_dataset")

        # Stop condition
        if best_id >= 0.995:
            decision["continue_training"] = False
            decision["reasoning"] = "Target reached!"

        return decision

    def _default_experiment(self, state: Dict) -> Dict:
        """Default experiment design."""
        return {
            "experiment_name": "identity_focus_baseline",
            "hypothesis": "Increasing identity loss weight improves ArcFace score",
            "config_changes": {
                "training.losses.identity_loss": 15.0,
                "training.schedule.learning_rate": 5e-5,
            },
            "success_criteria": {"min_identity_score": 0.97},
            "estimated_steps": 10000,
            "priority": "high",
        }

    def _default_data_strategy(self, failure_analysis: Dict) -> Dict:
        """Default data collection strategy."""
        high_failures = failure_analysis.get("high_failure_demographics", [])
        queries = []
        for demo in high_failures[:5]:
            queries.append({
                "query": f"{demo.get('ethnicity', '')} person portrait",
                **demo,
            })
        return {
            "queries": queries,
            "priority_demographics": high_failures[:3],
            "target_images_per_category": 200,
            "reasoning": "Targeting high-failure demographics",
        }

    def _log_decision(self, metrics: Dict, decision: Dict):
        """Log orchestrator decisions."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_metrics": metrics,
            "decision": decision,
        }
        self._decisions_log.append(record)

        log_path = os.path.join(self.log_dir, "orchestrator_decisions.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
