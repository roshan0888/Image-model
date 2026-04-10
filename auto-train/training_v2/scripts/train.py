#!/usr/bin/env python3
"""
Training Script v2 — SEPARATE from MVP

Trains LoRA on LP's SPADE generator with:
  1. PyTorch ArcFace identity loss (SAME model as production eval)
  2. Region-aware eye preservation loss
  3. Lip mode training (matches production)
  4. Gradient accumulation 8 for stable training on T4

This does NOT touch the production MVP.
Trained weights are saved to training_v2/checkpoints/
Test separately before deploying.

Usage:
  python train.py                   # Default: 5000 steps
  python train.py --steps 10000     # Longer training
  python train.py --rank 8          # Higher LoRA rank
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
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [train-v2] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "train_v2.log"
        )),
    ],
)
logger = logging.getLogger("train-v2")

TRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(TRAIN_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ENGINE_DIR = os.path.join(BASE_DIR, "training_engine")

sys.path.insert(0, LP_DIR)
sys.path.insert(0, ENGINE_DIR)


class TrainerV2:
    """
    Separate training pipeline.
    Does NOT modify any production files.
    Saves checkpoints to training_v2/checkpoints/
    """

    def __init__(self, lora_rank=4, lr=1e-4):
        self.device = "cuda"
        self.rank = lora_rank
        self.lr = lr
        self.grad_accum = 8

        self._load_lp()
        self._load_arcface()
        self._inject_lora()
        self._setup_optimizer()

    def _load_lp(self):
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_pipeline import LivePortraitPipeline
        self.lp = LivePortraitPipeline(
            inference_cfg=InferenceConfig(), crop_cfg=CropConfig()
        )
        logger.info("LP loaded")

    def _load_arcface(self):
        """Load PyTorch ArcFace (converted from ONNX — same as production)."""
        import onnx
        from onnx2torch import convert

        onnx_path = os.path.join(
            BASE_DIR, "MagicFace", "third_party_files",
            "models", "antelopev2", "glintr100.onnx"
        )
        self.arcface = convert(onnx.load(onnx_path)).eval().to(self.device)
        for p in self.arcface.parameters():
            p.requires_grad = False
        logger.info("PyTorch ArcFace loaded (same as production metric)")

    def _inject_lora(self):
        """Inject LoRA into SPADE G_middle blocks."""
        from training.lora_modules import LoRAConv2d, _get_parent_attr

        # Freeze everything
        for p in self.lp.live_portrait_wrapper.spade_generator.parameters():
            p.requires_grad = False
        for p in self.lp.live_portrait_wrapper.warping_module.parameters():
            p.requires_grad = False
        for p in self.lp.live_portrait_wrapper.motion_extractor.parameters():
            p.requires_grad = False
        for p in self.lp.live_portrait_wrapper.appearance_feature_extractor.parameters():
            p.requires_grad = False

        # Inject LoRA into G_middle conv layers only
        self.lora_params = []
        injected = 0

        for name, module in list(self.lp.live_portrait_wrapper.spade_generator.named_modules()):
            if isinstance(module, nn.Conv2d) and "G_middle" in name:
                if any(t in name for t in ["conv_0", "conv_1"]):
                    try:
                        parent, attr = _get_parent_attr(
                            self.lp.live_portrait_wrapper.spade_generator, name
                        )
                        lora = LoRAConv2d(module, self.rank, self.rank, dropout=0.0)
                        setattr(parent, attr, lora)
                        injected += 1
                    except Exception:
                        pass

        for name, p in self.lp.live_portrait_wrapper.spade_generator.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
                self.lora_params.append(p)

        total = sum(p.numel() for p in self.lora_params)
        logger.info(f"LoRA injected: {injected} layers, {total:,} params")

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.lora_params, lr=self.lr, weight_decay=1e-4)

    def _arcface_embedding(self, face_tensor):
        """Get ArcFace embedding from face tensor (B,3,H,W) in [0,1] BGR."""
        x = F.interpolate(face_tensor, size=(112, 112), mode='bilinear', align_corners=False)
        x = x.flip(1)  # BGR → RGB
        x = (x * 255.0 - 127.5) / 127.5
        emb = self.arcface(x)
        return F.normalize(emb, p=2, dim=1)

    def _identity_loss(self, source_tensor, output_tensor):
        """1 - cosine_similarity between source and output."""
        src_emb = self._arcface_embedding(source_tensor)
        out_emb = self._arcface_embedding(output_tensor)
        cosine = F.cosine_similarity(src_emb, out_emb, dim=1)
        return (1.0 - cosine).mean(), cosine.mean()

    def _eye_loss(self, source_tensor, output_tensor):
        """L1 loss on eye region (top 20-45% of face)."""
        h = source_tensor.shape[2]
        y1, y2 = int(h * 0.20), int(h * 0.45)
        return F.l1_loss(output_tensor[:, :, y1:y2, :], source_tensor[:, :, y1:y2, :])

    def _lp_forward(self, source_tensor, driving_tensor):
        """LP forward with gradients through LoRA only."""
        wrapper = self.lp.live_portrait_wrapper

        with torch.no_grad():
            feature_3d = wrapper.appearance_feature_extractor(source_tensor).float()
            source_kp = wrapper.motion_extractor(source_tensor)
            driving_kp = wrapper.motion_extractor(driving_tensor)
            for info in [source_kp, driving_kp]:
                for k, v in info.items():
                    if isinstance(v, torch.Tensor):
                        info[k] = v.float()

            from src.utils.camera import headpose_pred_to_degree, get_rotation_matrix
            def transform(kp_info):
                kp = kp_info['kp']
                pitch = headpose_pred_to_degree(kp_info['pitch'])
                yaw = headpose_pred_to_degree(kp_info['yaw'])
                roll = headpose_pred_to_degree(kp_info['roll'])
                bs = kp.shape[0]
                num_kp = kp.shape[1] // 3 if kp.ndim == 2 else kp.shape[1]
                R = get_rotation_matrix(pitch, yaw, roll)
                kp_t = kp.view(bs, num_kp, 3) @ R + kp_info['exp'].view(bs, num_kp, 3)
                kp_t *= kp_info['scale'][..., None]
                kp_t[:, :, 0:2] += kp_info['t'][:, None, 0:2]
                return kp_t

            kp_s = transform(source_kp)
            kp_d = transform(driving_kp)

            # Lip mode: only copy lip keypoints from driving
            kp_lip = kp_s.clone()
            for idx in [6, 12, 14, 17, 19, 20]:
                kp_lip[:, idx, :] = kp_d[:, idx, :]

        # Trainable path (gradients through LoRA)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            ret = wrapper.warping_module(feature_3d, kp_source=kp_s, kp_driving=kp_lip)
            output = wrapper.spade_generator(feature=ret['out'])

        return output.float()

    def _get_face_crop(self, image_path):
        """Load and crop face to 256x256."""
        from insightface.app import FaceAnalysis
        if not hasattr(self, '_fa'):
            self._fa = FaceAnalysis(
                name="antelopev2",
                root=os.path.join(BASE_DIR, "MagicFace", "third_party_files"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

        img = cv2.imread(image_path)
        if img is None:
            return None

        faces = self._fa.get(img)
        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = img.shape[:2]
        fw, fh = x2-x1, y2-y1
        pad = int(max(fw, fh) * 0.3)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        sz = max(fw, fh) + pad*2
        half = sz//2
        crop = img[max(0,cy-half):min(h,cy+half), max(0,cx-half):min(w,cx+half)]
        return cv2.resize(crop, (256, 256))

    def _to_tensor(self, img_256):
        x = img_256[np.newaxis].astype(np.float32) / 255.0
        return torch.from_numpy(np.clip(x, 0, 1)).permute(0, 3, 1, 2).to(self.device)

    def train(self, data_dir, num_steps=5000, eval_every=500):
        """Main training loop."""
        # Load pairs
        manifest = os.path.join(data_dir, "pairs.jsonl")
        if not os.path.exists(manifest):
            manifest = os.path.join(data_dir, "self_pairs.jsonl")
        if not os.path.exists(manifest):
            logger.error(f"No training data at {manifest}")
            return 0.0

        pairs = [json.loads(l) for l in open(manifest)]
        logger.info(f"Loaded {len(pairs)} training pairs")

        # Driver
        driver_path = os.path.join(LP_DIR, "assets", "examples", "driving", "d12.jpg")
        driver_crop = self._get_face_crop(driver_path)
        driver_tensor = self._to_tensor(driver_crop)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_steps, eta_min=self.lr * 0.01
        )

        ckpt_dir = os.path.join(TRAIN_DIR, "checkpoints")
        log_file = os.path.join(TRAIN_DIR, "logs", "metrics.jsonl")
        best_id = 0.0
        running_loss = 0.0
        running_id = 0.0

        logger.info(f"Training: {num_steps} steps, rank={self.rank}, lr={self.lr}")
        self.optimizer.zero_grad()

        for step in range(1, num_steps + 1):
            t0 = time.time()
            pair = pairs[np.random.randint(len(pairs))]

            src_crop = self._get_face_crop(pair["source"])
            if src_crop is None:
                continue

            src_tensor = self._to_tensor(src_crop)

            try:
                out_tensor = self._lp_forward(src_tensor, driver_tensor)
            except Exception:
                continue

            # Resize output to match source if needed
            if out_tensor.shape != src_tensor.shape:
                out_tensor = F.interpolate(out_tensor, size=src_tensor.shape[2:], mode='bilinear', align_corners=False)

            # Losses
            id_loss, cosine = self._identity_loss(src_tensor, out_tensor)
            eye_loss = self._eye_loss(src_tensor, out_tensor)
            tv_loss = (
                torch.mean(torch.abs(out_tensor[:,:,:,:-1] - out_tensor[:,:,:,1:])) +
                torch.mean(torch.abs(out_tensor[:,:,:-1,:] - out_tensor[:,:,1:,:]))
            )

            total = 10.0 * id_loss + 3.0 * eye_loss + 0.1 * tv_loss
            total = total / self.grad_accum

            total.backward()

            if step % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.lora_params, 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()

            running_loss = 0.9 * running_loss + 0.1 * (total.item() * self.grad_accum)
            running_id = 0.9 * running_id + 0.1 * cosine.item()

            del out_tensor, total
            torch.cuda.empty_cache()

            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(f"Step {step}/{num_steps} | loss={running_loss:.4f} | arcface={running_id:.4f} | eye={eye_loss.item():.4f} | lr={lr:.2e}")
                with open(log_file, "a") as f:
                    f.write(json.dumps({"step": step, "loss": round(running_loss, 4), "arcface": round(running_id, 4), "eye": round(eye_loss.item(), 4)}) + "\n")

            if step % eval_every == 0:
                avg = self._evaluate(pairs[:10])
                logger.info(f"  EVAL step {step}: arcface={avg:.4f}")
                if avg > best_id:
                    best_id = avg
                    self._save(ckpt_dir, step, avg)
                    logger.info(f"  NEW BEST: {avg:.4f}")

        logger.info(f"Training complete. Best: {best_id:.4f}")
        return best_id

    def _evaluate(self, pairs):
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        scores = []
        for pair in pairs[:5]:
            src = cv2.imread(pair["source"])
            if src is None: continue
            faces = self._fa.get(src)
            if not faces: continue
            src_emb = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding.reshape(1,-1)

            src_crop = self._get_face_crop(pair["source"])
            drv_crop = self._get_face_crop(os.path.join(LP_DIR, "assets/examples/driving/d12.jpg"))
            if src_crop is None or drv_crop is None: continue

            with torch.no_grad():
                out = self._lp_forward(self._to_tensor(src_crop), self._to_tensor(drv_crop))
            out_np = out.squeeze(0).permute(1,2,0).cpu().numpy()
            out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)

            out_faces = self._fa.get(out_np)
            if out_faces:
                out_emb = max(out_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding.reshape(1,-1)
                scores.append(float(cos_sim(src_emb, out_emb)[0][0]))

        return np.mean(scores) if scores else 0.0

    def _save(self, ckpt_dir, step, score):
        from training.lora_modules import save_lora_weights
        os.makedirs(ckpt_dir, exist_ok=True)
        save_lora_weights(
            self.lp.live_portrait_wrapper.spade_generator,
            os.path.join(ckpt_dir, "lora_spade_v2_best.pt")
        )
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump({"step": step, "arcface": round(score, 4), "rank": self.rank, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--data", default=os.path.join(TRAIN_DIR, "data", "self_pairs"))
    args = parser.parse_args()

    trainer = TrainerV2(lora_rank=args.rank, lr=args.lr)
    trainer.train(args.data, num_steps=args.steps, eval_every=args.eval_every)


if __name__ == "__main__":
    main()
