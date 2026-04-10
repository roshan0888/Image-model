# coding: utf-8

"""
Warping field estimator(W) defined in the paper, which generates a warping field using the implicit
keypoint representations x_s and x_d, and employs this flow field to warp the source feature volume f_s.

MODIFIED: Added region_mask support for mouth-only warping.
When region_mask is set, deformation is blended with identity grid
so only the mouth region moves while the rest of the face stays unchanged.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .util import SameBlock2d
from .dense_motion import DenseMotionNetwork


class WarpingNetwork(nn.Module):
    def __init__(
        self,
        num_kp,
        block_expansion,
        max_features,
        num_down_blocks,
        reshape_channel,
        estimate_occlusion_map=False,
        dense_motion_params=None,
        **kwargs
    ):
        super(WarpingNetwork, self).__init__()

        self.upscale = kwargs.get('upscale', 1)
        self.flag_use_occlusion_map = kwargs.get('flag_use_occlusion_map', True)

        # Region mask for mouth-only warping (None = full face warping as before)
        self.region_mask = None

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp,
                feature_channel=reshape_channel,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params
            )
        else:
            self.dense_motion_network = None

        self.third = SameBlock2d(max_features, block_expansion * (2 ** num_down_blocks), kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=block_expansion * (2 ** num_down_blocks), out_channels=block_expansion * (2 ** num_down_blocks), kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map

    def set_region_mask(self, mask):
        """Set region mask for selective warping.

        Args:
            mask: None (full face warping, original LP behavior)
                  "mouth_only": fixed mask (backward compatible)
                  numpy array (N, 2): 256x256 crop-space landmarks → dynamic mask
                  torch tensor (1, 1, 64, 64): pre-computed mask
        """
        self.region_mask = mask

    def _create_identity_grid(self, shape, device):
        """Create identity deformation grid (no movement)."""
        bs, d, h, w, _ = shape
        # Create normalized coordinate grid [-1, 1]
        grid_z = torch.linspace(-1, 1, d, device=device)
        grid_y = torch.linspace(-1, 1, h, device=device)
        grid_x = torch.linspace(-1, 1, w, device=device)
        grid = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij'), dim=-1)
        grid = grid.permute(2, 1, 0, 3)  # (d, h, w, 3)
        grid = grid.unsqueeze(0).expand(bs, -1, -1, -1, -1)  # (B, d, h, w, 3)
        return grid

    def _create_dynamic_mask(self, landmarks_256, device):
        """Create lower-face warp mask from ACTUAL face landmarks.

        Strategy: Split face into upper (identity) and lower (expression).
        The dividing line is the NOSE BRIDGE — everything below it warps,
        everything above stays frozen.

        This gives LP full freedom to move mouth/jaw/cheeks/chin
        while locking eyes and forehead in place.
        """
        import numpy as np
        import cv2

        mask = np.zeros((64, 64), dtype=np.float32)
        lmk = landmarks_256 / 4.0  # 256→64

        # Find the dividing line: nose bridge Y position
        # Nose bridge landmarks: ~index 28 in 106-point model (between eyes)
        # We use the midpoint between eye centers and nose tip
        if len(lmk) > 86:
            left_eye_center = lmk[33:43].mean(axis=0)
            right_eye_center = lmk[43:52].mean(axis=0)
            eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
            nose_tip_y = lmk[86, 1]
            # Dividing line: halfway between eye center and nose tip
            divide_y = int((eye_center_y + nose_tip_y) / 2)
        else:
            divide_y = 30  # fallback

        # Everything below divide_y gets full warp (1.0)
        mask[divide_y:, :] = 1.0

        # Gradient transition zone (5 pixels above divide line)
        # So the boundary isn't sharp
        for i in range(5):
            y = divide_y - 5 + i
            if 0 <= y < 64:
                mask[y, :] = i / 5.0

        # PROTECT eyes explicitly — even if they fall below divide line
        for eye_idx in [35, 37, 38, 40, 41, 44, 46, 47, 49, 50]:
            if eye_idx < len(lmk):
                pt = lmk[eye_idx].astype(int)
                y, x = pt[1], pt[0]
                # Zero out 4-pixel radius around each eye landmark
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < 64 and 0 <= nx < 64:
                            mask[ny, nx] = 0.0

        # Smooth slightly (small kernel to keep coverage)
        mask = cv2.GaussianBlur(mask, (5, 5), 2)
        mask = np.clip(mask, 0, 1)

        return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)

    def _create_fixed_mask(self, device):
        """Fallback fixed mouth mask at 64x64."""
        import numpy as np
        import cv2
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[35:58, 12:52] = 1.0
        mask[28:42, 5:15] = 0.4
        mask[28:42, 49:59] = 0.4
        mask[30:45, 15:20] = 0.6
        mask[30:45, 44:49] = 0.6
        mask = cv2.GaussianBlur(mask, (11, 11), 4)
        return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)

    def deform_input(self, inp, deformation):
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, feature_3d, kp_driving, kp_source):
        if self.dense_motion_network is not None:
            # Feature warper, Transforming feature representation according to deformation and occlusion
            dense_motion = self.dense_motion_network(
                feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source
            )
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']  # Bx1x64x64
            else:
                occlusion_map = None

            deformation = dense_motion['deformation']  # Bx16x64x64x3

            # ═══════════════════════════════════════════════════
            # REGION MASKING: Only warp mouth region
            # ═══════════════════════════════════════════════════
            if self.region_mask is not None:
                import numpy as np

                if isinstance(self.region_mask, str) and self.region_mask == "mouth_only":
                    mask_2d = self._create_fixed_mask(deformation.device)
                elif isinstance(self.region_mask, np.ndarray):
                    mask_2d = self._create_dynamic_mask(self.region_mask, deformation.device)
                elif isinstance(self.region_mask, torch.Tensor):
                    mask_2d = self.region_mask.to(deformation.device)
                else:
                    mask_2d = self._create_fixed_mask(deformation.device)

                # Create identity grid (no movement)
                identity_grid = self._create_identity_grid(deformation.shape, deformation.device)

                # Expand mask to match deformation shape: (1,1,64,64) → (B,16,64,64,1)
                mask_5d = mask_2d.unsqueeze(-1).expand_as(deformation[:, :1, :, :, :])
                mask_5d = mask_5d.expand_as(deformation)

                # Blend: mouth region uses LP deformation, rest uses identity (no warp)
                deformation = mask_5d * deformation + (1.0 - mask_5d) * identity_grid

            out = self.deform_input(feature_3d, deformation)  # Bx32x16x64x64

            bs, c, d, h, w = out.shape  # Bx32x16x64x64
            out = out.view(bs, c * d, h, w)  # -> Bx512x64x64
            out = self.third(out)  # -> Bx256x64x64
            out = self.fourth(out)  # -> Bx256x64x64

            if self.flag_use_occlusion_map and (occlusion_map is not None):
                out = out * occlusion_map

        ret_dct = {
            'occlusion_map': occlusion_map,
            'deformation': deformation,
            'out': out,
        }

        return ret_dct
