"""
LoRA (Low-Rank Adaptation) Modules for LivePortrait

Injects trainable low-rank matrices into frozen LP modules.
Only LoRA parameters are trained — original weights stay frozen.

Target modules:
  - WarpingNetwork: dense_motion hourglass + third/fourth conv layers
  - SPADEGenerator: all SPADE blocks + upsampling convs
  - MotionExtractor: fc_exp and fc_kp heads
  - StitchingRetargetingModule: all MLP layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """LoRA adapter for nn.Linear layers."""

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Ensure LoRA params on same device as input
        if self.lora_A.device != x.device:
            self.lora_A = nn.Parameter(self.lora_A.to(x.device))
            self.lora_B = nn.Parameter(self.lora_B.to(x.device))
        # Original forward (frozen)
        result = self.original(x)
        # LoRA delta
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result + lora_out


class LoRAConv2d(nn.Module):
    """LoRA adapter for nn.Conv2d layers."""

    def __init__(self, original: nn.Conv2d, rank: int = 16, alpha: float = 32.0,
                 dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # For conv2d: treat as matrix multiplication on reshaped weights
        # Weight shape: (out_channels, in_channels, kH, kW)
        out_ch = original.out_channels
        in_ch = original.in_channels * original.kernel_size[0] * original.kernel_size[1]

        self.lora_A = nn.Parameter(torch.zeros(rank, in_ch))
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Store original conv params for the LoRA path
        self._stride = original.stride
        self._padding = original.padding
        self._dilation = original.dilation
        self._groups = original.groups
        self._kernel_size = original.kernel_size

        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Ensure LoRA params on same device as input
        if self.lora_A.device != x.device:
            self.lora_A = nn.Parameter(self.lora_A.to(x.device))
            self.lora_B = nn.Parameter(self.lora_B.to(x.device))
        # Original forward (frozen)
        result = self.original(x)

        # LoRA path: unfold input → multiply by LoRA matrices → fold back
        B, C, H, W = x.shape

        # Apply dropout
        x_drop = self.dropout(x)

        # Unfold to get patches
        x_unfold = F.unfold(
            x_drop,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
        )  # Shape: (B, C*kH*kW, L)

        # LoRA: A projects down, B projects up
        lora_out = self.lora_B @ (self.lora_A @ x_unfold)  # (B, out_ch, L)
        lora_out = lora_out * self.scaling

        # Fold back to spatial
        out_h = (H + 2 * self._padding[0] - self._dilation[0] * (self._kernel_size[0] - 1) - 1) // self._stride[0] + 1
        out_w = (W + 2 * self._padding[1] - self._dilation[1] * (self._kernel_size[1] - 1) - 1) // self._stride[1] + 1
        lora_out = lora_out.view(B, -1, out_h, out_w)

        return result + lora_out


class LoRAConv3d(nn.Module):
    """LoRA adapter for nn.Conv3d layers (used in DenseMotionNetwork)."""

    def __init__(self, original: nn.Conv3d, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        # For 3d conv, use bottleneck approach:
        # 1x1x1 conv down to rank channels, then 1x1x1 conv up
        in_ch = original.in_channels
        out_ch = original.out_channels

        self.lora_down = nn.Conv3d(in_ch, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv3d(rank, out_ch, kernel_size=1, bias=False)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Ensure LoRA params on same device as input
        if self.lora_down.weight.device != x.device:
            self.lora_down = self.lora_down.to(x.device)
            self.lora_up = self.lora_up.to(x.device)
        result = self.original(x)
        lora_out = self.lora_up(self.lora_down(self.dropout(x))) * self.scaling
        # Match spatial dimensions if needed
        if lora_out.shape != result.shape:
            lora_out = F.interpolate(lora_out, size=result.shape[2:], mode='trilinear', align_corners=False)
        return result + lora_out


def inject_lora(model: nn.Module, target_names: List[str], rank: int = 16,
                alpha: float = 32.0, dropout: float = 0.05) -> Tuple[nn.Module, List[str]]:
    """Inject LoRA adapters into specified layers of a model.

    Args:
        model: The model to inject LoRA into.
        target_names: List of layer name patterns to target.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: Dropout rate.

    Returns:
        Modified model and list of injected layer names.
    """
    injected = []

    for target in target_names:
        # Find matching modules
        for name, module in list(model.named_modules()):
            if target not in name:
                continue

            if isinstance(module, nn.Linear):
                parent, attr = _get_parent_attr(model, name)
                lora_module = LoRALinear(module, rank, alpha, dropout)
                setattr(parent, attr, lora_module)
                injected.append(name)
                logger.info(f"  LoRA Linear: {name} (rank={rank})")

            elif isinstance(module, nn.Conv2d):
                parent, attr = _get_parent_attr(model, name)
                lora_module = LoRAConv2d(module, rank, alpha, dropout)
                setattr(parent, attr, lora_module)
                injected.append(name)
                logger.info(f"  LoRA Conv2d: {name} (rank={rank})")

            elif isinstance(module, nn.Conv3d):
                parent, attr = _get_parent_attr(model, name)
                lora_module = LoRAConv3d(module, rank, min(alpha, 16.0), dropout)
                setattr(parent, attr, lora_module)
                injected.append(name)
                logger.info(f"  LoRA Conv3d: {name} (rank={rank})")

    logger.info(f"Injected LoRA into {len(injected)} layers")
    return model, injected


def _get_parent_attr(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """Get parent module and attribute name from a dotted path."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only LoRA parameters (for optimizer)."""
    params = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d, LoRAConv3d)):
            for pname, param in module.named_parameters():
                if "lora_" in pname:
                    params.append(param)
    return params


def count_lora_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable LoRA params vs total params."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def save_lora_weights(model: nn.Module, path: str):
    """Save only LoRA weights (very small file)."""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
        elif isinstance(module, LoRAConv2d):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
        elif isinstance(module, LoRAConv3d):
            lora_state[f"{name}.lora_down.weight"] = module.lora_down.weight.data.cpu()
            lora_state[f"{name}.lora_up.weight"] = module.lora_up.weight.data.cpu()
    torch.save(lora_state, path)
    logger.info(f"Saved LoRA weights to {path} ({len(lora_state)} tensors)")


def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA weights into model."""
    lora_state = torch.load(path, map_location="cpu")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in lora_state:
                module.lora_A.data = lora_state[a_key].to(module.lora_A.device)
                module.lora_B.data = lora_state[b_key].to(module.lora_B.device)
        elif isinstance(module, LoRAConv2d):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in lora_state:
                module.lora_A.data = lora_state[a_key].to(module.lora_A.device)
                module.lora_B.data = lora_state[b_key].to(module.lora_B.device)
        elif isinstance(module, LoRAConv3d):
            down_key = f"{name}.lora_down.weight"
            up_key = f"{name}.lora_up.weight"
            if down_key in lora_state:
                module.lora_down.weight.data = lora_state[down_key].to(module.lora_down.weight.device)
                module.lora_up.weight.data = lora_state[up_key].to(module.lora_up.weight.device)
    logger.info(f"Loaded LoRA weights from {path}")
