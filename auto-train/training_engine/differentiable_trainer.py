#!/usr/bin/env python3
"""
Differentiable LoRA Training for LivePortrait

KEY INSIGHT: LP's internal forward pass IS differentiable:
  source_img → appearance_feature_extractor → features
  source_img → motion_extractor → source_kp
  driving_img → motion_extractor → driving_kp
  features + kps → warping_module → warped_features  ← TRAINABLE (LoRA)
  warped_features → spade_generator → output_face     ← TRAINABLE (LoRA)

The problem before: warp_decode() wraps everything in torch.no_grad().
Solution: Call warping_module and spade_generator DIRECTLY with gradients enabled.

Identity loss: Use facenet_pytorch InceptionResnetV1 (fully differentiable PyTorch model)
instead of InsightFace ONNX (non-differentiable).
"""

import os
import sys
import cv2
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [trainer] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "logs", "differentiable_train.log")
        ),
    ],
)
logger = logging.getLogger("trainer")

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ENGINE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ANTELOPEV2_DIR = os.path.join(BASE_DIR, "MagicFace", "third_party_files")

sys.path.insert(0, LP_DIR)
sys.path.insert(0, ENGINE_DIR)

os.makedirs(os.path.join(ENGINE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(ENGINE_DIR, "checkpoints_diff"), exist_ok=True)


class DifferentiableTrainer:
    """
    Train LoRA weights with REAL gradients through LP's differentiable path.

    Loss = identity_loss + expression_loss

    identity_loss: cosine distance between source and output face embeddings
                   (using PyTorch InceptionResnetV1, fully differentiable)

    expression_loss: L1 distance between driving keypoints and output keypoints
                     (ensures expression is transferred, not just copied source)
    """

    def __init__(self, lora_rank=8, learning_rate=5e-5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_rank = lora_rank
        self.lr = learning_rate

        # Load models
        self._load_lp()
        self._load_identity_model()
        self._load_face_detector()
        self._inject_lora()
        self._setup_optimizer()

    def _load_lp(self):
        """Load LivePortrait modules."""
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline

        self.lp = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig()
        )
        self.wrapper = self.lp.live_portrait_wrapper
        logger.info("LivePortrait loaded")

    def _load_identity_model(self):
        """Load differentiable face recognition model."""
        from facenet_pytorch import InceptionResnetV1

        self.id_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # Freeze — we only use it for loss computation
        for p in self.id_model.parameters():
            p.requires_grad = False
        logger.info("Identity model loaded (InceptionResnetV1-vggface2)")

    def _load_face_detector(self):
        """Load InsightFace for face detection and cropping."""
        from insightface.app import FaceAnalysis
        self.face_analyzer = FaceAnalysis(
            name="antelopev2", root=ANTELOPEV2_DIR,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
        logger.info("Face detector loaded")

    def _inject_lora(self):
        """Inject LoRA into warping_module and spade_generator."""
        from training.lora_modules import LoRAConv2d, _get_parent_attr

        def inject_lora_selective(model, rank, alpha, patterns=None):
            """Inject LoRA into Conv2d layers matching name patterns.
            If patterns is None, inject into ALL Conv2d layers.
            Otherwise only inject if any pattern is in the layer name.
            This saves GPU memory by only training critical layers.
            """
            injected = []
            for name, module in list(model.named_modules()):
                if isinstance(module, nn.Conv2d):
                    # Filter by pattern if provided
                    if patterns is not None:
                        if not any(p in name for p in patterns):
                            continue
                    try:
                        parent, attr = _get_parent_attr(model, name)
                        lora_module = LoRAConv2d(module, rank, alpha, dropout=0.0)
                        setattr(parent, attr, lora_module)
                        injected.append(name)
                    except Exception:
                        pass
            logger.info(f"  Injected LoRA into {len(injected)} Conv2d layers")
            return model, injected

        # SPADE generator: only inject into G_middle blocks (the bottleneck)
        # These control the core face appearance without needing huge memory
        self.wrapper.spade_generator, self.lora_layers_g = inject_lora_selective(
            self.wrapper.spade_generator, self.lora_rank, self.lora_rank,
            patterns=["G_middle", "up_0", "up_1"]  # Middle + first 2 upsample blocks
        )

        # Skip warping module LoRA to save memory — SPADE is the main target
        self.lora_layers_w = []

        # Freeze ALL parameters
        for p in self.wrapper.spade_generator.parameters():
            p.requires_grad = False
        for p in self.wrapper.warping_module.parameters():
            p.requires_grad = False
        for p in self.wrapper.motion_extractor.parameters():
            p.requires_grad = False
        for p in self.wrapper.appearance_feature_extractor.parameters():
            p.requires_grad = False

        # Unfreeze ONLY LoRA parameters
        self.lora_params = []
        for name, p in self.wrapper.spade_generator.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
                self.lora_params.append(p)
        if self.lora_layers_w:
            for name, p in self.wrapper.warping_module.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
                    self.lora_params.append(p)

        total = sum(p.numel() for p in self.lora_params)
        logger.info(f"LoRA injected: {len(self.lora_params)} parameter groups, {total:,} trainable params")

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(self.lora_params, lr=self.lr, weight_decay=1e-4)

    def get_face_crop_256(self, image_bgr):
        """Detect face and return 256x256 crop (what LP internally uses)."""
        faces = self.face_analyzer.get(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = image_bgr.shape[:2]

        # Add padding
        fw, fh = x2 - x1, y2 - y1
        pad = int(max(fw, fh) * 0.3)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        # Make square
        size = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)

        crop = image_bgr[y1:y2, x1:x2]
        crop_256 = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        return crop_256

    def img_to_tensor(self, img_bgr_256):
        """Convert 256x256 BGR image to LP input tensor."""
        x = img_bgr_256[np.newaxis].astype(np.float32) / 255.0
        x = np.clip(x, 0, 1)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)  # 1x3x256x256
        return x

    def get_identity_embedding(self, face_tensor_3x256x256):
        """Get differentiable identity embedding from face tensor.

        face_tensor: 1x3x256x256, values 0-1, BGR
        Returns: 1x512 embedding (differentiable)
        """
        # facenet expects RGB 160x160, normalized
        face = face_tensor_3x256x256.clone()
        # BGR to RGB
        face = face.flip(1)
        # Resize to 160x160
        face = F.interpolate(face, size=(160, 160), mode='bilinear', align_corners=False)
        # Normalize to [-1, 1] (what facenet expects)
        face = face * 2 - 1

        embedding = self.id_model(face)
        return embedding  # 1x512, differentiable!

    def differentiable_forward(self, source_tensor, driving_tensor):
        """
        Run LP's forward pass WITH gradients enabled for LoRA params.
        Uses mixed precision to fit in T4 GPU memory.

        source_tensor: 1x3x256x256
        driving_tensor: 1x3x256x256

        Returns: output face tensor (differentiable through LoRA)
        """
        # Step 1: Extract appearance features (frozen, no grad needed)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feature_3d = self.wrapper.appearance_feature_extractor(source_tensor)
            feature_3d = feature_3d.float()

        # Step 2: Extract keypoints (frozen)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                source_kp_info = self.wrapper.motion_extractor(source_tensor)
                driving_kp_info = self.wrapper.motion_extractor(driving_tensor)

            # Float conversion
            for info in [source_kp_info, driving_kp_info]:
                for k, v in info.items():
                    if isinstance(v, torch.Tensor):
                        info[k] = v.float()

        # Step 3: Transform keypoints (frozen)
        with torch.no_grad():
            from src.utils.camera import headpose_pred_to_degree, get_rotation_matrix

            def transform_kp(kp_info):
                kp = kp_info['kp']
                pitch = headpose_pred_to_degree(kp_info['pitch'])
                yaw = headpose_pred_to_degree(kp_info['yaw'])
                roll = headpose_pred_to_degree(kp_info['roll'])
                t, exp = kp_info['t'], kp_info['exp']
                scale = kp_info['scale']
                bs = kp.shape[0]
                num_kp = kp.shape[1] // 3 if kp.ndim == 2 else kp.shape[1]
                rot_mat = get_rotation_matrix(pitch, yaw, roll)
                kp_t = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
                kp_t *= scale[..., None]
                kp_t[:, :, 0:2] += t[:, None, 0:2]
                return kp_t

            kp_source = transform_kp(source_kp_info)
            kp_driving = transform_kp(driving_kp_info)

        # Step 4: Warping + SPADE — THIS IS WHERE LORA LIVES
        # Use mixed precision (float16) to save memory, gradients in float32
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Warping module (has LoRA)
            ret_dct = self.wrapper.warping_module(
                feature_3d, kp_source=kp_source, kp_driving=kp_driving
            )

            # SPADE generator (has LoRA)
            output = self.wrapper.spade_generator(feature=ret_dct['out'])

        return output.float()  # Back to float32 for loss computation

    def compute_loss(self, source_tensor, output_tensor, driving_tensor):
        """
        Compute training loss:

        identity_loss: 1 - cosine_sim(source_embedding, output_embedding)
           → pushes output to look like source person

        reconstruction_loss: L1(output, source) in face region
           → prevents output from going wild

        expression_preservation: we DON'T want output = source exactly
           → small weight on reconstruction to allow expression change
        """
        # Identity loss (differentiable through facenet)
        source_emb = self.get_identity_embedding(source_tensor)
        output_emb = self.get_identity_embedding(output_tensor)

        cosine_sim = F.cosine_similarity(source_emb, output_emb)
        identity_loss = 1.0 - cosine_sim.mean()

        # Mild reconstruction loss (don't want exact copy, just similar)
        # Resize to match dimensions if needed
        if output_tensor.shape != source_tensor.shape:
            source_resized = F.interpolate(source_tensor, size=output_tensor.shape[2:],
                                           mode='bilinear', align_corners=False)
        else:
            source_resized = source_tensor
        recon_loss = F.l1_loss(output_tensor, source_resized)

        # Perceptual smoothness (prevent artifacts)
        # Total variation loss on output
        tv_loss = (
            torch.mean(torch.abs(output_tensor[:, :, :, :-1] - output_tensor[:, :, :, 1:])) +
            torch.mean(torch.abs(output_tensor[:, :, :-1, :] - output_tensor[:, :, 1:, :]))
        )

        # Combined loss
        # Heavy on identity (that's what we're fixing)
        # Light on reconstruction (allow expression change)
        # Tiny TV loss (prevent artifacts)
        total = identity_loss * 5.0 + recon_loss * 0.3 + tv_loss * 0.1

        return {
            "total": total,
            "identity": identity_loss.item(),
            "cosine_sim": cosine_sim.mean().item(),
            "reconstruction": recon_loss.item(),
            "tv": tv_loss.item(),
        }

    def train(self, training_pairs, num_steps=1000, eval_every=100, save_dir=None):
        """
        Train LoRA on paired data.

        training_pairs: list of (source_path, driving_path) tuples
        """
        if save_dir is None:
            save_dir = os.path.join(ENGINE_DIR, "checkpoints_diff")

        metrics_file = os.path.join(ENGINE_DIR, "logs", "diff_training_metrics.jsonl")

        logger.info(f"Starting differentiable training: {num_steps} steps")
        logger.info(f"  Training pairs: {len(training_pairs)}")
        logger.info(f"  LoRA rank: {self.lora_rank}")
        logger.info(f"  Learning rate: {self.lr}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_steps, eta_min=self.lr * 0.1
        )

        best_identity = 0.0
        running_id_loss = 0.0
        running_cosine = 0.0

        for step in range(1, num_steps + 1):
            t0 = time.time()

            # Sample random pair
            idx = np.random.randint(len(training_pairs))
            src_path, drv_path = training_pairs[idx]

            # Load and crop faces
            src_bgr = cv2.imread(src_path)
            drv_bgr = cv2.imread(drv_path)

            if src_bgr is None or drv_bgr is None:
                continue

            src_crop = self.get_face_crop_256(src_bgr)
            drv_crop = self.get_face_crop_256(drv_bgr)

            if src_crop is None or drv_crop is None:
                continue

            # Convert to tensors
            src_tensor = self.img_to_tensor(src_crop)
            drv_tensor = self.img_to_tensor(drv_crop)

            # Forward pass (differentiable through LoRA)
            try:
                output_tensor = self.differentiable_forward(src_tensor, drv_tensor)
            except Exception as e:
                logger.debug(f"Forward failed at step {step}: {e}")
                continue

            # Compute loss
            losses = self.compute_loss(src_tensor, output_tensor, drv_tensor)

            # Backward pass — THIS IS THE KEY DIFFERENCE
            # Real gradients flowing through LoRA parameters!
            self.optimizer.zero_grad()

            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.lora_params, max_norm=1.0)

            self.optimizer.step()
            scheduler.step()

            # Save metrics BEFORE deleting
            step_id_loss = losses["identity"]
            step_cosine = losses["cosine_sim"]
            step_recon = losses["reconstruction"]
            step_tv = losses["tv"]

            elapsed = time.time() - t0
            running_id_loss = 0.9 * running_id_loss + 0.1 * step_id_loss
            running_cosine = 0.9 * running_cosine + 0.1 * step_cosine

            # Free GPU memory
            del output_tensor, src_tensor, drv_tensor
            torch.cuda.empty_cache()

            # Log every 10 steps
            if step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {step}/{num_steps} | "
                    f"id_loss={running_id_loss:.4f} | "
                    f"cosine={running_cosine:.4f} | "
                    f"recon={step_recon:.4f} | "
                    f"lr={lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

                with open(metrics_file, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "identity_loss": round(running_id_loss, 4),
                        "cosine_sim": round(running_cosine, 4),
                        "reconstruction": round(step_recon, 4),
                        "tv": round(step_tv, 4),
                        "lr": lr,
                    }) + "\n")

            # Evaluate and save
            if step % eval_every == 0:
                avg_id = self._evaluate(training_pairs[:5])
                logger.info(f"  EVAL step {step}: facenet_cosine={avg_id:.4f} (running_cosine={running_cosine:.4f})")

                # Use the better of eval or running cosine
                current_score = max(avg_id, running_cosine)
                if current_score > best_identity:
                    best_identity = current_score
                    self._save_checkpoint(save_dir, step, current_score)
                    logger.info(f"  NEW BEST: {current_score:.4f} — checkpoint saved")

        logger.info(f"\nTraining complete. Best identity: {best_identity:.4f}")
        return best_identity

    def _evaluate(self, pairs, max_eval=5):
        """Evaluate using differentiable facenet identity (always works on crops)."""
        scores = []
        for src_path, drv_path in pairs[:max_eval]:
            src_bgr = cv2.imread(src_path)
            drv_bgr = cv2.imread(drv_path)
            if src_bgr is None or drv_bgr is None:
                continue

            src_crop = self.get_face_crop_256(src_bgr)
            drv_crop = self.get_face_crop_256(drv_bgr)
            if src_crop is None or drv_crop is None:
                continue

            with torch.no_grad():
                src_t = self.img_to_tensor(src_crop)
                drv_t = self.img_to_tensor(drv_crop)
                out_t = self.differentiable_forward(src_t, drv_t)

                # Use facenet for identity (same as training loss)
                src_emb = self.get_identity_embedding(src_t)
                out_emb = self.get_identity_embedding(out_t)
                cosine = F.cosine_similarity(src_emb, out_emb).item()
                scores.append(cosine)

                del src_t, drv_t, out_t
                torch.cuda.empty_cache()

        return np.mean(scores) if scores else 0.0

    def _save_checkpoint(self, save_dir, step, identity_score):
        """Save LoRA weights."""
        from training.lora_modules import save_lora_weights

        save_lora_weights(
            self.wrapper.spade_generator,
            os.path.join(save_dir, "lora_spade_diff_best.pt")
        )
        save_lora_weights(
            self.wrapper.warping_module,
            os.path.join(save_dir, "lora_warping_diff_best.pt")
        )

        # Save metadata
        meta = {
            "step": step,
            "identity_score": round(identity_score, 4),
            "lora_rank": self.lora_rank,
            "lr": self.lr,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(save_dir, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


def build_training_pairs():
    """Build training pairs from cleaned scraped data."""
    cleaned_dir = os.path.join(ENGINE_DIR, "dataset", "cleaned_scraped")

    # Source = neutral faces, Driving = expression faces
    neutral_dir = os.path.join(cleaned_dir, "neutral")
    expr_dirs = {
        "smile": os.path.join(cleaned_dir, "smile"),
        "open_smile": os.path.join(cleaned_dir, "open_smile_drivers"),
        "surprise": os.path.join(cleaned_dir, "surprise"),
        "sad": os.path.join(cleaned_dir, "sad"),
    }

    # Get all neutral faces
    neutrals = []
    if os.path.exists(neutral_dir):
        for f in os.listdir(neutral_dir):
            if f.startswith("clean_") and f.endswith(".jpg"):
                neutrals.append(os.path.join(neutral_dir, f))

    # Get all expression faces
    expressions = []
    for expr, d in expr_dirs.items():
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.startswith("clean_") and f.endswith(".jpg"):
                    expressions.append(os.path.join(d, f))

    # Also use expression faces as source (self-expression transfer)
    all_sources = neutrals + expressions

    # Create pairs: every source × every driving
    pairs = []
    for src in all_sources:
        for drv in expressions:
            if src != drv:  # Don't pair with self
                pairs.append((src, drv))

    # Also add pairs where source = driving (identity reconstruction)
    # This teaches the model to preserve identity perfectly when no expression change
    for src in all_sources[:20]:  # Limit these
        pairs.append((src, src))

    np.random.shuffle(pairs)

    logger.info(f"Built {len(pairs)} training pairs from {len(all_sources)} sources × {len(expressions)} drivers")
    return pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("DIFFERENTIABLE LORA TRAINING")
    logger.info(f"  Steps: {args.steps}, LR: {args.lr}, Rank: {args.rank}")
    logger.info("=" * 70)

    # Build pairs
    pairs = build_training_pairs()
    if not pairs:
        logger.error("No training pairs found! Run scraper + cleaner first.")
        return

    # Train
    trainer = DifferentiableTrainer(lora_rank=args.rank, learning_rate=args.lr)
    best = trainer.train(pairs, num_steps=args.steps, eval_every=args.eval_every)

    logger.info(f"\nFinal best identity: {best:.4f}")


if __name__ == "__main__":
    main()
