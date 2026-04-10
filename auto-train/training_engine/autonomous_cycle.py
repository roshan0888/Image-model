#!/usr/bin/env python3
"""
Autonomous Training Cycle v3

The correct training pipeline based on everything we learned:
  1. Uses PyTorch ArcFace (same model as production eval)
  2. Region-aware losses (eye preservation + mouth expression)
  3. LoRA rank 4 on SPADE G_middle blocks only
  4. Lip mode training (matches production usage)
  5. Gradient accumulation 8 for stable training on T4

Usage:
  python autonomous_cycle.py                    # Train on self-pairs
  python autonomous_cycle.py --data voxceleb    # Train on VoxCeleb2 pairs
  python autonomous_cycle.py --steps 5000       # Custom steps
"""

import os
import sys
import cv2
import json
import time
import glob
import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [train-v3] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs", "train_v3.log"
        )),
    ],
)
logger = logging.getLogger("train-v3")

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ENGINE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")

sys.path.insert(0, LP_DIR)
sys.path.insert(0, ENGINE_DIR)

os.makedirs(os.path.join(ENGINE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(ENGINE_DIR, "checkpoints_v3"), exist_ok=True)


class AutonomousTrainerV3:
    """
    Correct autonomous training for LP lip mode.

    Key fixes from v1/v2:
      1. ArcFace PyTorch (not FaceNet) → same metric for train and eval
      2. Region losses (eye preservation) → protects identity
      3. LoRA rank 4 on G_middle only → fits T4, stable gradients
      4. Lip mode during training → matches production use case
      5. Gradient accumulation 8 → effective batch size 8
    """

    def __init__(self, lora_rank=4, lr=1e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_rank = lora_rank
        self.lr = lr

        logger.info("=" * 60)
        logger.info("AUTONOMOUS TRAINER v3")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  LoRA rank: {lora_rank}")
        logger.info(f"  LR: {lr}")
        logger.info("=" * 60)

        self._load_lp()
        self._load_arcface()
        self._inject_lora()
        self._setup_optimizer()
        self._setup_losses()

    def _load_lp(self):
        """Load LivePortrait."""
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline

        self.lp = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig()
        )
        self.wrapper = self.lp.live_portrait_wrapper
        logger.info("  LP loaded")

    def _load_arcface(self):
        """Load PyTorch ArcFace (SAME as production eval)."""
        from data_engine.arcface_pytorch import ArcFacePyTorch
        self.arcface = ArcFacePyTorch(device=self.device)
        logger.info("  PyTorch ArcFace loaded (same as production metric)")

    def _inject_lora(self):
        """Inject LoRA rank 4 into SPADE G_middle blocks ONLY."""
        from training.lora_modules import LoRAConv2d, _get_parent_attr
        import torch.nn as nn

        # Freeze everything
        for p in self.wrapper.spade_generator.parameters():
            p.requires_grad = False
        for p in self.wrapper.warping_module.parameters():
            p.requires_grad = False
        for p in self.wrapper.motion_extractor.parameters():
            p.requires_grad = False
        for p in self.wrapper.appearance_feature_extractor.parameters():
            p.requires_grad = False

        # Inject LoRA into G_middle blocks only (conv_0, conv_1)
        # These are the 6 middle SPADE blocks that control face appearance
        self.lora_params = []
        injected = 0

        for name, module in list(self.wrapper.spade_generator.named_modules()):
            # Target: G_middle_0.conv_0, G_middle_0.conv_1, etc.
            if isinstance(module, nn.Conv2d) and "G_middle" in name:
                if any(t in name for t in ["conv_0", "conv_1"]):
                    try:
                        parent, attr = _get_parent_attr(self.wrapper.spade_generator, name)
                        lora = LoRAConv2d(module, self.lora_rank, self.lora_rank, dropout=0.0)
                        setattr(parent, attr, lora)
                        injected += 1
                    except Exception:
                        pass

        # Collect LoRA parameters
        for name, p in self.wrapper.spade_generator.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
                self.lora_params.append(p)

        total = sum(p.numel() for p in self.lora_params)
        logger.info(f"  LoRA injected: {injected} layers, {total:,} params")

    def _setup_optimizer(self):
        """Setup optimizer with gradient accumulation."""
        self.optimizer = torch.optim.AdamW(
            self.lora_params, lr=self.lr, weight_decay=1e-4
        )
        self.grad_accum_steps = 8  # Effective batch size 8

    def _setup_losses(self):
        """Setup combined loss function."""
        from training.region_losses import CombinedTrainingLoss
        self.loss_fn = CombinedTrainingLoss(self.arcface, device=self.device)
        logger.info("  Combined loss ready (ArcFace + eye + region + mouth)")

    def _lp_forward_differentiable(self, source_tensor, driving_tensor):
        """Run LP forward pass with gradients through LoRA.

        source_tensor: (1, 3, 256, 256) normalized [0, 1]
        driving_tensor: (1, 3, 256, 256) normalized [0, 1]

        Returns: output tensor (1, 3, H, W) in [0, 1]
        """
        # Frozen steps (no gradients needed)
        with torch.no_grad():
            feature_3d = self.wrapper.appearance_feature_extractor(source_tensor)
            feature_3d = feature_3d.float()

            source_kp_info = self.wrapper.motion_extractor(source_tensor)
            driving_kp_info = self.wrapper.motion_extractor(driving_tensor)

            for info in [source_kp_info, driving_kp_info]:
                for k, v in info.items():
                    if isinstance(v, torch.Tensor):
                        info[k] = v.float()

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

            # Apply lip-mode keypoint masking (only lip keypoints from driving)
            kp_lip = kp_source.clone()
            for lip_idx in [6, 12, 14, 17, 19, 20]:
                kp_lip[:, lip_idx, :] = kp_driving[:, lip_idx, :]

        # Trainable steps (gradients flow through LoRA)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            ret_dct = self.wrapper.warping_module(
                feature_3d, kp_source=kp_source, kp_driving=kp_lip
            )
            output = self.wrapper.spade_generator(feature=ret_dct['out'])

        return output.float()

    def _load_training_pairs(self, data_source="self_pairs"):
        """Load training pairs."""
        if data_source == "self_pairs":
            pairs_file = os.path.join(ENGINE_DIR, "dataset", "self_pairs", "self_pairs.jsonl")
        else:
            pairs_file = os.path.join(ENGINE_DIR, "dataset", data_source, "pairs.jsonl")

        if not os.path.exists(pairs_file):
            logger.error(f"No training data at {pairs_file}")
            return []

        pairs = []
        with open(pairs_file) as f:
            for line in f:
                pairs.append(json.loads(line))

        logger.info(f"  Loaded {len(pairs)} training pairs from {data_source}")
        return pairs

    def _img_to_tensor(self, img_bgr_256):
        """Convert 256x256 BGR image to LP input tensor."""
        x = img_bgr_256[np.newaxis].astype(np.float32) / 255.0
        x = np.clip(x, 0, 1)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        return x

    def _get_face_crop_256(self, image_path):
        """Load image and crop face to 256x256."""
        img = cv2.imread(image_path)
        if img is None:
            return None

        from insightface.app import FaceAnalysis
        if not hasattr(self, '_fa'):
            self._fa = FaceAnalysis(
                name="antelopev2",
                root=os.path.join(BASE_DIR, "MagicFace", "third_party_files"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        faces = self._fa.get(img)
        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = img.shape[:2]
        fw, fh = x2-x1, y2-y1
        pad = int(max(fw, fh) * 0.3)

        cx, cy = (x1+x2)//2, (y1+y2)//2
        size = max(fw, fh) + pad*2
        half = size // 2

        c1 = max(0, cx-half)
        c2 = min(w, cx+half)
        r1 = max(0, cy-half)
        r2 = min(h, cy+half)

        crop = img[r1:r2, c1:c2]
        return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    def train(self, data_source="self_pairs", num_steps=5000, eval_every=500):
        """Main training loop."""
        pairs = self._load_training_pairs(data_source)
        if not pairs:
            logger.error("No training data! Run self_pair_generator.py first.")
            return 0.0

        # Get a driving image for lip mode
        driver_path = os.path.join(LP_DIR, "assets", "examples", "driving", "d12.jpg")
        driver_crop = self._get_face_crop_256(driver_path)
        if driver_crop is None:
            logger.error("Cannot load driver image")
            return 0.0

        driver_tensor = self._img_to_tensor(driver_crop)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_steps, eta_min=self.lr * 0.01
        )

        metrics_file = os.path.join(ENGINE_DIR, "logs", "train_v3_metrics.jsonl")
        best_identity = 0.0
        running_loss = 0.0
        running_id = 0.0

        logger.info(f"\n  Starting training: {num_steps} steps")
        logger.info(f"  Grad accumulation: {self.grad_accum_steps}")
        logger.info(f"  Effective batch: {self.grad_accum_steps}")

        self.optimizer.zero_grad()

        for step in range(1, num_steps + 1):
            t0 = time.time()

            # Sample random pair
            pair = pairs[np.random.randint(len(pairs))]
            source_path = pair["source"]

            # Load and crop
            source_crop = self._get_face_crop_256(source_path)
            if source_crop is None:
                continue

            source_tensor = self._img_to_tensor(source_crop)

            # Forward pass (differentiable through LoRA)
            try:
                output_tensor = self._lp_forward_differentiable(source_tensor, driver_tensor)
            except Exception as e:
                logger.debug(f"Forward failed: {e}")
                continue

            # Resize output to match source if needed (SPADE outputs 512x512)
            if output_tensor.shape != source_tensor.shape:
                output_tensor = F.interpolate(
                    output_tensor, size=source_tensor.shape[2:],
                    mode='bilinear', align_corners=False
                )

            # Compute loss
            loss, metrics = self.loss_fn(source_tensor, output_tensor)
            loss = loss / self.grad_accum_steps  # Scale for accumulation

            # Backward
            loss.backward()

            # Step optimizer every grad_accum_steps
            if step % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.lora_params, max_norm=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()

            # Track
            running_loss = 0.9 * running_loss + 0.1 * metrics["total"]
            running_id = 0.9 * running_id + 0.1 * metrics["cosine_sim"]

            # Free memory
            del output_tensor, loss
            torch.cuda.empty_cache()

            elapsed = time.time() - t0

            # Log
            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {step}/{num_steps} | "
                    f"loss={running_loss:.4f} | "
                    f"arcface={running_id:.4f} | "
                    f"eye={metrics['eye_loss']:.4f} | "
                    f"lr={lr:.2e} | "
                    f"{elapsed:.1f}s"
                )
                with open(metrics_file, "a") as f:
                    f.write(json.dumps({
                        "step": step, **metrics, "lr": lr
                    }) + "\n")

            # Evaluate
            if step % eval_every == 0:
                avg_id = self._evaluate(pairs[:10])
                logger.info(f"  EVAL step {step}: arcface={avg_id:.4f}")

                if avg_id > best_identity:
                    best_identity = avg_id
                    self._save_checkpoint(step, avg_id)
                    logger.info(f"  NEW BEST: {avg_id:.4f}")

        logger.info(f"\nTraining complete. Best ArcFace: {best_identity:.4f}")
        return best_identity

    def _evaluate(self, pairs):
        """Evaluate on pairs using production InsightFace ArcFace."""
        from sklearn.metrics.pairwise import cosine_similarity

        if not hasattr(self, '_fa'):
            from insightface.app import FaceAnalysis
            self._fa = FaceAnalysis(
                name="antelopev2",
                root=os.path.join(BASE_DIR, "MagicFace", "third_party_files"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        driver_path = os.path.join(LP_DIR, "assets", "examples", "driving", "d12.jpg")
        scores = []

        for pair in pairs[:5]:
            src_path = pair["source"]
            src_img = cv2.imread(src_path)
            if src_img is None:
                continue

            src_faces = self._fa.get(src_img)
            if not src_faces:
                continue

            src_emb = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding.reshape(1, -1)

            # Run LP with current LoRA weights
            source_crop = self._get_face_crop_256(src_path)
            if source_crop is None:
                continue

            src_t = self._img_to_tensor(source_crop)
            drv_crop = self._get_face_crop_256(driver_path)
            drv_t = self._img_to_tensor(drv_crop)

            with torch.no_grad():
                out_t = self._lp_forward_differentiable(src_t, drv_t)

            out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)

            out_faces = self._fa.get(out_np)
            if out_faces:
                out_emb = max(out_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding.reshape(1, -1)
                score = float(cosine_similarity(src_emb, out_emb)[0][0])
                scores.append(score)

        return np.mean(scores) if scores else 0.0

    def _save_checkpoint(self, step, identity_score):
        """Save LoRA weights."""
        from training.lora_modules import save_lora_weights

        ckpt_dir = os.path.join(ENGINE_DIR, "checkpoints_v3")
        save_lora_weights(
            self.wrapper.spade_generator,
            os.path.join(ckpt_dir, "lora_spade_v3_best.pt")
        )

        meta = {
            "step": step,
            "identity_score": round(identity_score, 4),
            "lora_rank": self.lora_rank,
            "lr": self.lr,
            "mode": "lip",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(ckpt_dir, "meta_v3.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"  Checkpoint saved: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="self_pairs", help="Data source: self_pairs or voxceleb")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    args = parser.parse_args()

    trainer = AutonomousTrainerV3(lora_rank=args.rank, lr=args.lr)
    best = trainer.train(
        data_source=args.data,
        num_steps=args.steps,
        eval_every=args.eval_every,
    )
    logger.info(f"Final best ArcFace: {best:.4f}")


if __name__ == "__main__":
    main()
