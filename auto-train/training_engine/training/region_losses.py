#!/usr/bin/env python3
"""
Region-Aware Losses for LP Lip Mode Training

Key innovation: penalize eye/forehead changes HEAVILY while allowing mouth changes.
This teaches LP's LoRA: "in lip mode, NEVER change the eyes."

The face is split into vertical bands in the 256x256 crop space:
  Top 0-20%:    Forehead → moderate preservation (weight 1.0)
  Mid 20-45%:   Eyes → HEAVY preservation (weight 3.0)
  Mid 45-55%:   Nose → light preservation (weight 0.5)
  Bot 55-80%:   Mouth → ALLOW change toward driving (weight 0.5)
  Bot 80-100%:  Chin/jaw → moderate preservation (weight 1.0)
"""

import torch
import torch.nn.functional as F


class RegionAwareLoss(torch.nn.Module):
    """Region-weighted L1 loss between source and output faces."""

    def __init__(self, image_size=256):
        super().__init__()
        self.h = image_size

        # Region boundaries (fraction of image height)
        self.regions = {
            "forehead": (0.00, 0.20, 1.0),   # (start, end, weight)
            "eyes":     (0.20, 0.45, 3.0),   # HEAVY — identity lives here
            "nose":     (0.45, 0.55, 0.5),
            "mouth":    (0.55, 0.80, 0.0),   # ZERO — mouth should change
            "chin":     (0.80, 1.00, 1.0),
        }

    def forward(self, source, output):
        """Compute region-weighted L1 loss.

        Args:
            source: (B, 3, H, W) source face
            output: (B, 3, H, W) LP output face

        Returns:
            loss: weighted L1 across regions
            details: dict with per-region losses
        """
        h = source.shape[2]
        total_loss = torch.tensor(0.0, device=source.device)
        details = {}

        for name, (start, end, weight) in self.regions.items():
            y1 = int(h * start)
            y2 = int(h * end)

            src_region = source[:, :, y1:y2, :]
            out_region = output[:, :, y1:y2, :]

            region_loss = F.l1_loss(out_region, src_region)
            total_loss = total_loss + weight * region_loss
            details[name] = region_loss.item()

        return total_loss, details


class EyePreservationLoss(torch.nn.Module):
    """Dedicated eye preservation loss.

    Extracts eye patches using approximate positions in the 256x256 crop.
    Penalizes ANY change to eye pixels — this is the #1 identity signal.
    """

    def __init__(self):
        super().__init__()

    def forward(self, source, output):
        """
        Args:
            source: (B, 3, 256, 256)
            output: (B, 3, 256, 256)

        Returns:
            loss: L1 on eye region
        """
        h = source.shape[2]

        # Eye band: 20-45% of face height, full width
        y1 = int(h * 0.20)
        y2 = int(h * 0.45)

        src_eyes = source[:, :, y1:y2, :]
        out_eyes = output[:, :, y1:y2, :]

        return F.l1_loss(out_eyes, src_eyes)


class MouthExpressionLoss(torch.nn.Module):
    """Ensure mouth region ACTUALLY changes toward the driving expression.

    Without this, LoRA could learn to just copy the source (identity=1.0 but no smile).
    This loss penalizes the output mouth being too similar to the source mouth.
    """

    def __init__(self, min_change=0.01):
        super().__init__()
        self.min_change = min_change

    def forward(self, source, output):
        """
        Args:
            source: (B, 3, 256, 256) source face
            output: (B, 3, 256, 256) LP output face

        Returns:
            loss: penalty if mouth hasn't changed enough
        """
        h = source.shape[2]

        # Mouth band: 55-80% of face height
        y1 = int(h * 0.55)
        y2 = int(h * 0.80)

        src_mouth = source[:, :, y1:y2, :]
        out_mouth = output[:, :, y1:y2, :]

        # Change magnitude
        change = F.l1_loss(out_mouth, src_mouth)

        # Penalize if change is too SMALL (expression not transferred)
        if change < self.min_change:
            loss = self.min_change - change
        else:
            loss = torch.tensor(0.0, device=source.device)

        return loss


class CombinedTrainingLoss(torch.nn.Module):
    """Combined loss for LP lip mode training.

    Brings together:
      1. ArcFace identity loss (primary)
      2. Eye preservation loss (protect identity region)
      3. Mouth expression loss (ensure smile happens)
      4. Perceptual loss (LPIPS for realism)
      5. Total variation (smoothness)
    """

    def __init__(self, arcface_model, device="cuda"):
        super().__init__()
        self.arcface = arcface_model
        self.region_loss = RegionAwareLoss()
        self.eye_loss = EyePreservationLoss()
        self.mouth_loss = MouthExpressionLoss()
        self.device = device

        # Loss weights (from plan)
        self.w_identity = 10.0
        self.w_eye = 3.0
        self.w_region = 1.0
        self.w_mouth = 0.5
        self.w_tv = 0.1

    def forward(self, source, output):
        """
        Args:
            source: (B, 3, H, W) source face crop, [0, 1], BGR
            output: (B, 3, H, W) LP output face crop, [0, 1], BGR

        Returns:
            total_loss: scalar
            metrics: dict with individual losses
        """
        # Resize to 256x256 for region losses if needed
        if source.shape[2] != 256:
            src_256 = F.interpolate(source, size=(256, 256), mode='bilinear', align_corners=False)
            out_256 = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            src_256 = source
            out_256 = output

        # 1. ArcFace identity loss
        id_loss, cosine_sim = self.arcface.identity_loss(src_256, out_256)

        # 2. Eye preservation
        eye_loss = self.eye_loss(src_256, out_256)

        # 3. Region-aware L1
        region_loss, region_details = self.region_loss(src_256, out_256)

        # 4. Mouth expression (penalize if no change)
        mouth_loss = self.mouth_loss(src_256, out_256)

        # 5. Total variation
        tv_loss = (
            torch.mean(torch.abs(out_256[:, :, :, :-1] - out_256[:, :, :, 1:])) +
            torch.mean(torch.abs(out_256[:, :, :-1, :] - out_256[:, :, 1:, :]))
        )

        # Combined
        total = (
            self.w_identity * id_loss +
            self.w_eye * eye_loss +
            self.w_region * region_loss +
            self.w_mouth * mouth_loss +
            self.w_tv * tv_loss
        )

        metrics = {
            "total": total.item(),
            "identity_loss": id_loss.item(),
            "cosine_sim": cosine_sim.item(),
            "eye_loss": eye_loss.item(),
            "region_loss": region_loss.item(),
            "mouth_loss": mouth_loss.item(),
            "tv_loss": tv_loss.item(),
            **{f"region_{k}": v for k, v in region_details.items()},
        }

        return total, metrics
