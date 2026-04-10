"""
Training Dataset for LivePortrait LoRA Fine-Tuning

Creates paired training samples:
  - Source image (neutral face)
  - Target image (same person, different expression)
  - Expression label and driving parameters

Supports both real pairs (from identity clustering) and
synthetic pairs (generated on-the-fly using LP).
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging
import random

logger = logging.getLogger(__name__)


class FaceExpressionDataset(Dataset):
    """Paired face expression dataset for training."""

    EXPRESSION_TO_DRIVING = {
        "smile": "d30.jpg",
        "big_smile": "d12.jpg",
        "surprise": "d19.jpg",
        "angry": "d38.jpg",
        "sad": "d8.jpg",
        "laugh": "laugh.pkl",
    }

    def __init__(self, pairs_path: str, lp_driving_dir: str,
                 image_size: int = 256, augment: bool = True):
        """
        Args:
            pairs_path: Path to training_pairs.jsonl
            lp_driving_dir: Path to LP driving templates
            image_size: Training resolution (LP internal = 256)
            augment: Whether to apply data augmentation
        """
        self.pairs = []
        self.lp_driving_dir = lp_driving_dir
        self.image_size = image_size
        self.augment = augment

        if os.path.exists(pairs_path):
            with open(pairs_path) as f:
                for line in f:
                    pair = json.loads(line.strip())
                    # Only include pairs where source exists
                    if os.path.exists(pair.get("source_path", "")):
                        self.pairs.append(pair)

        logger.info(f"Loaded {len(self.pairs)} training pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        # Load source image
        source = cv2.imread(pair["source_path"])
        if source is None:
            return self._get_fallback()

        source = cv2.resize(source, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LANCZOS4)

        # Load target image (real pair) or return info for synthetic generation
        target_expr = pair.get("target_expression", "smile")

        if pair.get("target_path") and os.path.exists(pair["target_path"]):
            # Real pair — same person, different expression
            target = cv2.imread(pair["target_path"])
            if target is not None:
                target = cv2.resize(target, (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_LANCZOS4)
            else:
                target = None
        else:
            target = None

        # Augmentation
        if self.augment:
            source, target = self._augment(source, target)

        # Convert to tensors (CHW, normalized to [0, 1])
        source_tensor = self._to_tensor(source)

        # Always include target key (zeros if no real target)
        if target is not None:
            target_tensor = self._to_tensor(target)
            has_target = True
        else:
            target_tensor = torch.zeros_like(source_tensor)
            has_target = False

        # Driving info for LP
        driving_file = self.EXPRESSION_TO_DRIVING.get(target_expr, "d30.jpg")

        result = {
            "source": source_tensor,
            "target": target_tensor,
            "has_target": 1 if has_target else 0,
            "target_expression": target_expr,
            "identity_group": pair.get("identity_group", -1),
        }

        return result

    def _augment(self, source: np.ndarray,
                 target: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply synchronized augmentation to source and target."""
        # Horizontal flip (50% chance) — must apply to both
        if random.random() < 0.5:
            source = cv2.flip(source, 1)
            if target is not None:
                target = cv2.flip(target, 1)

        # Brightness jitter (same for both)
        if random.random() < 0.3:
            delta = random.uniform(-20, 20)
            source = np.clip(source.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            if target is not None:
                target = np.clip(target.astype(np.float32) + delta, 0, 255).astype(np.uint8)

        # Slight rotation (same for both)
        if random.random() < 0.2:
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D(
                (self.image_size // 2, self.image_size // 2), angle, 1.0
            )
            source = cv2.warpAffine(source, M, (self.image_size, self.image_size))
            if target is not None:
                target = cv2.warpAffine(target, M, (self.image_size, self.image_size))

        return source, target

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        """BGR uint8 → RGB float32 tensor [0, 1]."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return tensor

    def _get_fallback(self) -> Dict:
        """Return a valid item when current pair fails."""
        idx = random.randint(0, max(0, len(self.pairs) - 1))
        if idx == len(self.pairs):
            # Empty dataset fallback
            dummy = torch.zeros(3, self.image_size, self.image_size)
            return {
                "source": dummy, "source_path": "",
                "target_expression": "smile", "identity_group": -1,
                "is_synthetic": True, "driving_path": "",
            }
        return self.__getitem__(idx)


def create_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Create a DataLoader from config."""
    pairs_path = os.path.join(config["paths"]["paired_dir"], "training_pairs.jsonl")
    lp_driving_dir = os.path.join(config["paths"]["liveportrait_dir"],
                                   "assets", "examples", "driving")

    dataset = FaceExpressionDataset(
        pairs_path=pairs_path,
        lp_driving_dir=lp_driving_dir,
        image_size=256,
        augment=(split == "train"),
    )

    batch_size = config["training"]["schedule"]["batch_size"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,  # Avoid multiprocessing issues with small datasets
        pin_memory=True,
        drop_last=False,
    )
    return loader
