"""
LivePortrait LoRA Trainer

The core training loop that fine-tunes LivePortrait's modules
with ArcFace identity loss.

Architecture:
  1. Load frozen LP modules (F, M, W, G, S)
  2. Inject LoRA adapters into target layers
  3. Forward pass: source → LP pipeline → output
  4. Compute losses (identity + expression + perceptual)
  5. Backward through LoRA parameters only
  6. Evaluate every N steps against holdout set

Key design: The LP forward pass is made differentiable by
keeping everything in PyTorch tensors. The ArcFace embedding
extraction is the only non-differentiable step (handled by
DifferentiableIdentityLoss which uses a ResNet proxy).
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .lora_modules import (
    inject_lora, get_lora_parameters, count_lora_parameters,
    save_lora_weights, load_lora_weights,
)
from .losses import CombinedLoss
from .dataset import create_dataloader

logger = logging.getLogger(__name__)


class LivePortraitTrainer:
    """Trains LivePortrait with LoRA + ArcFace identity loss."""

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_step = 0
        self.best_identity_score = 0.0
        self.patience_counter = 0

        # Paths
        self.ckpt_dir = config["paths"]["checkpoints_dir"]
        self.log_dir = config["paths"]["logs_dir"]
        self.exp_dir = config["paths"]["experiments_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.exp_dir, exist_ok=True)

        # TensorBoard writer (live monitoring)
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.log_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"TensorBoard logging → {tb_dir}")
        except Exception as e:
            logger.warning(f"TensorBoard disabled: {e}")

        # Training config
        self.train_cfg = config["training"]
        self.schedule = self.train_cfg["schedule"]

        # Models (loaded lazily)
        self.lp_wrapper = None
        self.face_analyzer = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler("cuda") if self.schedule["mixed_precision"] == "fp16" else None

        # Metrics history
        self.metrics_history = []

    def setup(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("LIVEPORTRAIT LORA TRAINER — SETUP")
        logger.info("=" * 60)

        self._load_liveportrait()
        self._inject_lora()
        self._load_face_analyzer()
        self._setup_losses()
        self._setup_optimizer()
        self._setup_dataloader()

        logger.info("=" * 60)
        logger.info("SETUP COMPLETE — Ready to train")
        logger.info("=" * 60)

    def _load_liveportrait(self):
        """Load LivePortrait modules."""
        logger.info("Loading LivePortrait...")
        lp_dir = self.config["paths"]["liveportrait_dir"]
        if lp_dir not in sys.path:
            sys.path.insert(0, lp_dir)

        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper

        self.lp_wrapper = LivePortraitWrapper(InferenceConfig())

        # Set all modules to eval mode (frozen)
        self.lp_wrapper.appearance_feature_extractor.eval()
        self.lp_wrapper.motion_extractor.eval()
        self.lp_wrapper.warping_module.eval()
        self.lp_wrapper.spade_generator.eval()
        if self.lp_wrapper.stitching_retargeting_module:
            stm = self.lp_wrapper.stitching_retargeting_module
            if isinstance(stm, dict):
                for k, v in stm.items():
                    if hasattr(v, 'eval'):
                        v.eval()
            else:
                stm.eval()

        logger.info("LivePortrait loaded")

    def _inject_lora(self):
        """Inject LoRA into target LP modules."""
        logger.info("Injecting LoRA adapters...")
        lora_cfg = self.train_cfg["lora"]
        rank = lora_cfg["rank"]
        alpha = lora_cfg["alpha"]
        dropout = lora_cfg["dropout"]

        total_injected = 0

        # Warping Network
        targets = lora_cfg["target_modules"].get("warping_network", [])
        if targets:
            logger.info("  Warping Network:")
            self.lp_wrapper.warping_module, injected = inject_lora(
                self.lp_wrapper.warping_module, targets, rank, alpha, dropout
            )
            total_injected += len(injected)

        # SPADE Generator
        targets = lora_cfg["target_modules"].get("spade_generator", [])
        if targets:
            logger.info("  SPADE Generator:")
            self.lp_wrapper.spade_generator, injected = inject_lora(
                self.lp_wrapper.spade_generator, targets, rank, alpha, dropout
            )
            total_injected += len(injected)

        # Motion Extractor
        targets = lora_cfg["target_modules"].get("motion_extractor", [])
        if targets:
            logger.info("  Motion Extractor:")
            self.lp_wrapper.motion_extractor, injected = inject_lora(
                self.lp_wrapper.motion_extractor, targets, rank, alpha, dropout
            )
            total_injected += len(injected)

        # Stitching/Retargeting (it's a dict of sub-modules)
        targets = lora_cfg["target_modules"].get("stitching_retargeting", [])
        stm = self.lp_wrapper.stitching_retargeting_module
        if targets and stm:
            logger.info("  Stitching/Retargeting:")
            if isinstance(stm, dict):
                for key, submodule in stm.items():
                    if any(t in key for t in targets) and hasattr(submodule, 'named_modules'):
                        submodule, injected = inject_lora(
                            submodule, [""], rank, alpha, dropout
                        )
                        stm[key] = submodule
                        total_injected += len(injected)
            else:
                stm, injected = inject_lora(stm, targets, rank, alpha, dropout)
                self.lp_wrapper.stitching_retargeting_module = stm
                total_injected += len(injected)

        # Count parameters
        for module_name, module in [
            ("warping", self.lp_wrapper.warping_module),
            ("spade", self.lp_wrapper.spade_generator),
            ("motion", self.lp_wrapper.motion_extractor),
        ]:
            trainable, total = count_lora_parameters(module)
            logger.info(f"  {module_name}: {trainable:,} trainable / {total:,} total "
                       f"({trainable/total*100:.2f}%)")

        logger.info(f"Total LoRA layers injected: {total_injected}")

        # Move all modules (including new LoRA params) to device
        self.lp_wrapper.warping_module.to(self.device)
        self.lp_wrapper.spade_generator.to(self.device)
        self.lp_wrapper.motion_extractor.to(self.device)
        self.lp_wrapper.appearance_feature_extractor.to(self.device)

    def _load_face_analyzer(self):
        """Load InsightFace for identity verification during training."""
        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(
            name="antelopev2",
            root=self.config["paths"]["antelopev2_dir"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    def _setup_losses(self):
        """Initialize loss functions."""
        # Find ArcFace model
        arcface_path = self._find_arcface_model()
        self.loss_fn = CombinedLoss(
            self.config, arcface_path, self.face_analyzer, self.device
        )

    def _find_arcface_model(self) -> str:
        """Find the ArcFace ONNX model."""
        candidates = [
            os.path.join(self.config["paths"]["antelopev2_dir"],
                        "models", "antelopev2", "glintr100.onnx"),
            os.path.join(self.config["paths"]["antelopev2_dir"],
                        "models", "antelopev2", "w600k_r50.onnx"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

        # Search for any .onnx file that looks like a recognition model
        import glob
        for pattern in ["**/glintr100.onnx", "**/w600k*.onnx", "**/*recognition*.onnx"]:
            matches = glob.glob(os.path.join(self.config["paths"]["antelopev2_dir"], pattern),
                              recursive=True)
            if matches:
                return matches[0]

        raise FileNotFoundError("ArcFace ONNX model not found")

    def _setup_optimizer(self):
        """Setup optimizer for LoRA parameters only."""
        # Collect all LoRA parameters
        lora_params = []
        for module in [
            self.lp_wrapper.warping_module,
            self.lp_wrapper.spade_generator,
            self.lp_wrapper.motion_extractor,
        ]:
            lora_params.extend(get_lora_parameters(module))

        stm = self.lp_wrapper.stitching_retargeting_module
        if stm:
            if isinstance(stm, dict):
                for submodule in stm.values():
                    if hasattr(submodule, 'modules'):
                        lora_params.extend(get_lora_parameters(submodule))
            else:
                lora_params.extend(get_lora_parameters(stm))

        if not lora_params:
            raise RuntimeError("No LoRA parameters found! Check target module names.")

        total_params = sum(p.numel() for p in lora_params)
        logger.info(f"Optimizer: {len(lora_params)} parameter groups, {total_params:,} total params")

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.schedule["learning_rate"],
            weight_decay=self.train_cfg["losses"]["regularization"],
        )

        # Learning rate scheduler
        if self.schedule["lr_scheduler"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.schedule["total_steps"],
                eta_min=self.schedule["learning_rate"] * 0.01,
            )

    def _setup_dataloader(self):
        """Create training data loader."""
        self.train_loader = create_dataloader(self.config, split="train")
        logger.info(f"Training dataset: {len(self.train_loader.dataset)} pairs")

    # ═══════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════

    def train(self, resume_from: Optional[str] = None) -> Dict:
        """Run the full training loop.

        Returns final metrics dict.
        """
        if resume_from:
            self._resume(resume_from)

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING START")
        logger.info(f"  Steps: {self.global_step} → {self.schedule['total_steps']}")
        logger.info(f"  Batch size: {self.schedule['batch_size']}")
        logger.info(f"  LR: {self.schedule['learning_rate']}")
        logger.info(f"  Gradient accumulation: {self.schedule['gradient_accumulation']}")
        logger.info("=" * 60)

        total_steps = self.schedule["total_steps"]
        eval_every = self.schedule["eval_every"]
        save_every = self.schedule["save_every"]
        grad_accum = self.schedule["gradient_accumulation"]

        # Set LoRA layers to training mode
        self._set_lora_train_mode(True)

        running_losses = {}
        epoch = 0
        data_iter = iter(self.train_loader)

        while self.global_step < total_steps:
            t0 = time.time()

            # Get batch (cycle through dataset)
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Forward pass
            losses = self._train_step(batch)

            # Accumulate
            for k, v in losses.items():
                if k not in running_losses:
                    running_losses[k] = 0.0
                running_losses[k] += v

            # Step optimizer every grad_accum steps
            if (self.global_step + 1) % grad_accum == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._get_all_lora_params(),
                        self.schedule["max_grad_norm"]
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self._get_all_lora_params(),
                        self.schedule["max_grad_norm"]
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            self.global_step += 1

            # Logging
            if self.global_step % 50 == 0:
                avg = {k: v / 50 for k, v in running_losses.items()}
                lr = self.optimizer.param_groups[0]["lr"]
                step_time = time.time() - t0
                logger.info(
                    f"Step {self.global_step}/{total_steps} | "
                    f"loss={avg.get('total', 0):.4f} | "
                    f"id={avg.get('identity', 0):.4f} | "
                    f"arcface={avg.get('arcface_score', 0):.4f} | "
                    f"expr={avg.get('expression', 0):.4f} | "
                    f"lr={lr:.2e} | {step_time:.1f}s/step"
                )
                self._log_metrics(avg)
                running_losses = {}

            # Evaluation
            if self.global_step % eval_every == 0:
                eval_metrics = self.evaluate()
                self._handle_eval_results(eval_metrics)

            # Save checkpoint
            if self.global_step % save_every == 0:
                self._save_checkpoint()

            # Early stopping check
            if self._should_stop():
                logger.info(f"Early stopping at step {self.global_step}")
                break

        # Final save
        self._save_checkpoint(is_final=True)
        return self._get_final_metrics()

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        source = batch["source"].to(self.device)

        # Forward through LP pipeline (differentiable path)
        with autocast("cuda", enabled=self.scaler is not None):
            output = self._lp_forward(source, batch)

            # Compute losses
            has_target = batch.get("has_target", None)
            target = batch["target"].to(self.device)
            # If no real target in this batch, pass None
            if has_target is not None and has_target.sum() == 0:
                target = None

            losses = self.loss_fn(source, output, target)

            # Scale loss for gradient accumulation
            total_loss = losses["total"] / self.schedule["gradient_accumulation"]

        # Backward
        if self.scaler:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

    def _lp_forward(self, source: torch.Tensor, batch: Dict) -> torch.Tensor:
        """Differentiable LivePortrait forward pass.

        This runs the LP pipeline in a way that gradients flow through LoRA params.

        Pipeline:
        1. Appearance extractor (F): source → feature_3d
        2. Motion extractor (M): source → kp_source; driving → kp_driving
        3. Keypoint transform: apply driving motion to source keypoints
        4. Warping (W): warp feature_3d with keypoint difference → warped_features
        5. Generator (G): warped_features → output image
        """
        B = source.shape[0]

        # Step 1: Extract 3D appearance features
        feature_3d = self.lp_wrapper.appearance_feature_extractor(source)

        # Step 2: Extract motion from source
        # Use motion extractor directly (allows gradient flow through LoRA)
        raw_kp = self.lp_wrapper.motion_extractor(source)

        # Reshape like get_kp_info does
        from src.utils.camera import headpose_pred_to_degree
        bs = raw_kp['kp'].shape[0]
        kp_info = {}
        for k, v in raw_kp.items():
            kp_info[k] = v.float() if torch.is_tensor(v) else v
        kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]
        kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]
        kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]
        kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)
        kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)

        # Step 3: Create driving keypoints
        kp_source = self.lp_wrapper.transform_keypoint(kp_info)
        driving_kp = self._create_driving_keypoints(kp_info, batch)

        # Step 4: Warp features
        warp_out = self.lp_wrapper.warping_module(feature_3d, driving_kp, kp_source)
        warped_features = warp_out["out"]

        # Step 5: Generate output image
        output = self.lp_wrapper.spade_generator(warped_features)

        return output

    def _create_driving_keypoints(self, source_kp_info: Dict,
                                   batch: Dict) -> torch.Tensor:
        """Create driving keypoints for expression transfer.

        Modifies source expression component to target expression.
        Uses predefined expression deltas or learned templates.
        """
        kp_info = {k: v.clone() if torch.is_tensor(v) else v
                   for k, v in source_kp_info.items()}

        # Expression deltas — these define how keypoints move for each expression
        # These are learned from the driving templates during data preparation
        target_expr = batch.get("target_expression", ["smile"])[0]

        # Predefined expression deltas (keypoint offsets for 21 keypoints)
        EXPR_DELTAS = {
            "smile": self._smile_delta(kp_info["exp"]),
            "surprise": self._surprise_delta(kp_info["exp"]),
            "angry": self._angry_delta(kp_info["exp"]),
            "sad": self._sad_delta(kp_info["exp"]),
        }

        delta = EXPR_DELTAS.get(target_expr, self._smile_delta(kp_info["exp"]))
        kp_info["exp"] = kp_info["exp"] + delta

        return self.lp_wrapper.transform_keypoint(kp_info)

    def _smile_delta(self, exp: torch.Tensor) -> torch.Tensor:
        """Smile expression delta — lift mouth corners, squint eyes."""
        delta = torch.zeros_like(exp)
        # Mouth corners up: keypoints 12-15 (mouth region in 21-kp system)
        delta[:, 12:16, 1] = -0.02  # Move up (negative y)
        delta[:, 12:16, 0] = 0.01   # Widen
        # Squint eyes slightly
        delta[:, 8:12, 1] = 0.005   # Lower eyelids slightly
        return delta

    def _surprise_delta(self, exp: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(exp)
        # Open mouth wide
        delta[:, 14:17, 1] = 0.04   # Lower jaw down
        # Raise eyebrows
        delta[:, 0:4, 1] = -0.03    # Brows up
        # Widen eyes
        delta[:, 8:12, 1] = -0.015  # Upper eyelids up
        return delta

    def _angry_delta(self, exp: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(exp)
        # Furrow brows (move inner brows down and together)
        delta[:, 0:2, 1] = 0.02     # Inner brows down
        delta[:, 0:2, 0] = 0.01     # Inner brows together
        # Tighten lips
        delta[:, 14:17, 1] = -0.01  # Lips compressed
        return delta

    def _sad_delta(self, exp: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(exp)
        # Droop mouth corners
        delta[:, 12:14, 1] = 0.02   # Corners down
        # Inner brows up (distressed)
        delta[:, 0:2, 1] = -0.015
        return delta

    # ═══════════════════════════════════════════════════════════
    # EVALUATION
    # ═══════════════════════════════════════════════════════════

    def evaluate(self) -> Dict:
        """Evaluate current model on holdout samples."""
        logger.info(f"\n--- Evaluation at step {self.global_step} ---")
        self._set_lora_train_mode(False)

        eval_metrics = {
            "identity_scores": [],
            "expression_changes": [],
            "lpips_scores": [],
        }

        # Use a subset of training data for eval
        eval_count = 0
        max_eval = 50

        with torch.no_grad():
            for batch in self.train_loader:
                if eval_count >= max_eval:
                    break

                source = batch["source"].to(self.device)
                output = self._lp_forward(source, batch)

                # Compute metrics
                losses = self.loss_fn(source, output)
                eval_metrics["identity_scores"].append(losses["arcface_score"].item())

                eval_count += source.shape[0]

        self._set_lora_train_mode(True)

        # Aggregate
        results = {
            "step": self.global_step,
            "mean_identity": np.mean(eval_metrics["identity_scores"]),
            "min_identity": np.min(eval_metrics["identity_scores"]),
            "max_identity": np.max(eval_metrics["identity_scores"]),
            "std_identity": np.std(eval_metrics["identity_scores"]),
        }

        logger.info(f"  Identity: {results['mean_identity']:.4f} "
                    f"(min={results['min_identity']:.4f}, max={results['max_identity']:.4f})")

        return results

    def _handle_eval_results(self, metrics: Dict):
        """Handle evaluation results — update best, check early stopping."""
        self.metrics_history.append(metrics)

        current_score = metrics["mean_identity"]
        target = self.train_cfg["early_stopping"]["target_identity"]
        min_delta = self.train_cfg["early_stopping"]["min_delta"]

        if current_score > self.best_identity_score + min_delta:
            self.best_identity_score = current_score
            self.patience_counter = 0
            self._save_checkpoint(is_best=True)
            logger.info(f"  New best identity score: {current_score:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"  No improvement. Patience: {self.patience_counter}/"
                       f"{self.train_cfg['early_stopping']['patience']}")

        if current_score >= target:
            logger.info(f"  TARGET REACHED: {current_score:.4f} >= {target}")

        # Save eval metrics
        metrics_path = os.path.join(self.log_dir, "eval_metrics.jsonl")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def _should_stop(self) -> bool:
        """Check early stopping conditions."""
        patience = self.train_cfg["early_stopping"]["patience"]
        target = self.train_cfg["early_stopping"]["target_identity"]

        if self.patience_counter >= patience:
            return True
        if self.best_identity_score >= target:
            return True
        return False

    # ═══════════════════════════════════════════════════════════
    # CHECKPOINTING
    # ═══════════════════════════════════════════════════════════

    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save LoRA checkpoint."""
        if is_best:
            suffix = "best"
        elif is_final:
            suffix = "final"
        else:
            suffix = f"step_{self.global_step}"

        # Save LoRA weights for each module
        for module_name, module in [
            ("warping", self.lp_wrapper.warping_module),
            ("spade", self.lp_wrapper.spade_generator),
            ("motion", self.lp_wrapper.motion_extractor),
        ]:
            path = os.path.join(self.ckpt_dir, f"lora_{module_name}_{suffix}.pt")
            save_lora_weights(module, path)

        stm = self.lp_wrapper.stitching_retargeting_module
        if stm:
            if isinstance(stm, dict):
                for key, submodule in stm.items():
                    path = os.path.join(self.ckpt_dir, f"lora_stitch_{key}_{suffix}.pt")
                    save_lora_weights(submodule, path)
            else:
                path = os.path.join(self.ckpt_dir, f"lora_stitching_{suffix}.pt")
                save_lora_weights(stm, path)

        # Save optimizer state
        opt_path = os.path.join(self.ckpt_dir, f"optimizer_{suffix}.pt")
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "best_identity_score": self.best_identity_score,
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }, opt_path)

        logger.info(f"Checkpoint saved: {suffix} (step {self.global_step})")

    def _resume(self, checkpoint_dir: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from {checkpoint_dir}")

        # Load LoRA weights
        for module_name, module in [
            ("warping", self.lp_wrapper.warping_module),
            ("spade", self.lp_wrapper.spade_generator),
            ("motion", self.lp_wrapper.motion_extractor),
        ]:
            path = os.path.join(checkpoint_dir, f"lora_{module_name}_best.pt")
            if os.path.exists(path):
                load_lora_weights(module, path)

        # Load optimizer
        opt_path = os.path.join(checkpoint_dir, "optimizer_best.pt")
        if os.path.exists(opt_path):
            state = torch.load(opt_path)
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler and state.get("scheduler"):
                self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state["global_step"]
            self.best_identity_score = state["best_identity_score"]
            if self.scaler and state.get("scaler"):
                self.scaler.load_state_dict(state["scaler"])

    # ═══════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════

    def _set_lora_train_mode(self, training: bool):
        """Set LoRA layers to train/eval mode."""
        from .lora_modules import LoRALinear, LoRAConv2d, LoRAConv3d

        for module in [
            self.lp_wrapper.warping_module,
            self.lp_wrapper.spade_generator,
            self.lp_wrapper.motion_extractor,
        ]:
            for m in module.modules():
                if isinstance(m, (LoRALinear, LoRAConv2d, LoRAConv3d)):
                    m.train(training)

    def _get_all_lora_params(self) -> List[nn.Parameter]:
        params = []
        for module in [
            self.lp_wrapper.warping_module,
            self.lp_wrapper.spade_generator,
            self.lp_wrapper.motion_extractor,
        ]:
            params.extend(get_lora_parameters(module))
        stm = self.lp_wrapper.stitching_retargeting_module
        if stm:
            if isinstance(stm, dict):
                for submodule in stm.values():
                    if hasattr(submodule, 'modules'):
                        params.extend(get_lora_parameters(submodule))
            else:
                params.extend(get_lora_parameters(stm))
        return params

    def _log_metrics(self, metrics: Dict):
        """Log metrics to file + TensorBoard."""
        log_path = os.path.join(self.log_dir, "training_metrics.jsonl")
        record = {"step": self.global_step, **metrics}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Live TensorBoard logging
        if self.tb_writer is not None:
            for key, value in metrics.items():
                try:
                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(f"train/{key}", value, self.global_step)
                except Exception:
                    pass
            self.tb_writer.flush()

    def _get_final_metrics(self) -> Dict:
        return {
            "final_step": self.global_step,
            "best_identity_score": self.best_identity_score,
            "metrics_history": self.metrics_history,
        }

    def update_learning_rate(self, new_lr: float):
        """Update learning rate (used by LLM orchestrator)."""
        for pg in self.optimizer.param_groups:
            old_lr = pg["lr"]
            pg["lr"] = new_lr
        logger.info(f"Learning rate updated: {old_lr:.2e} → {new_lr:.2e}")

    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss weights (used by LLM orchestrator)."""
        self.loss_fn.update_weights(new_weights)
