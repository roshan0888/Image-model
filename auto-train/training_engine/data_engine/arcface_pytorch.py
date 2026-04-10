#!/usr/bin/env python3
"""
PyTorch ArcFace wrapper — loads the converted ONNX→PyTorch model.
Provides differentiable identity embedding and loss computation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
from onnx2torch import convert

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(
    os.path.dirname(ENGINE_DIR), "MagicFace", "third_party_files",
    "models", "antelopev2", "glintr100.onnx"
)
PT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcface_glintr100.pt")


class ArcFacePyTorch(nn.Module):
    """Differentiable ArcFace model — SAME as production InsightFace ONNX.

    Input: face image tensor (B, 3, 112, 112), normalized to [-1, 1]
    Output: embedding (B, 512), L2-normalized
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        # Load from cache or convert
        if os.path.exists(PT_CACHE):
            self.model = convert(onnx.load(ONNX_PATH))
            self.model.load_state_dict(torch.load(PT_CACHE, map_location="cpu"))
        else:
            self.model = convert(onnx.load(ONNX_PATH))
            torch.save(self.model.state_dict(), PT_CACHE)

        self.model = self.model.to(device).eval()

        # Freeze — we only use this for loss, never train it
        for p in self.model.parameters():
            p.requires_grad = False

    def preprocess(self, face_bgr_256):
        """Convert 256x256 BGR face crop to ArcFace input format.

        Args:
            face_bgr_256: tensor (B, 3, 256, 256) in [0, 1] range, BGR order

        Returns:
            tensor (B, 3, 112, 112) normalized for ArcFace
        """
        # Resize to 112x112
        x = F.interpolate(face_bgr_256, size=(112, 112), mode='bilinear', align_corners=False)

        # BGR to RGB
        x = x.flip(1)

        # Normalize: InsightFace uses (pixel - 127.5) / 127.5 → [-1, 1]
        x = (x * 255.0 - 127.5) / 127.5

        return x

    def forward(self, face_tensor):
        """Get L2-normalized embedding.

        Args:
            face_tensor: (B, 3, 112, 112) preprocessed face

        Returns:
            embedding: (B, 512) L2-normalized
        """
        emb = self.model(face_tensor)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def get_embedding(self, face_bgr_256):
        """Convenience: preprocess + forward in one call.

        Args:
            face_bgr_256: (B, 3, 256, 256) in [0, 1], BGR

        Returns:
            embedding: (B, 512) L2-normalized, differentiable
        """
        x = self.preprocess(face_bgr_256)
        return self.forward(x)

    def identity_loss(self, source_face, output_face):
        """Compute identity loss: 1 - cosine_similarity.

        Args:
            source_face: (B, 3, 256, 256) source face in [0, 1], BGR
            output_face: (B, 3, 256, 256) LP output in [0, 1], BGR

        Returns:
            loss: scalar, lower = better identity preservation
            cosine_sim: scalar, higher = more similar
        """
        src_emb = self.get_embedding(source_face)
        out_emb = self.get_embedding(output_face)

        cosine_sim = F.cosine_similarity(src_emb, out_emb, dim=1)
        loss = (1.0 - cosine_sim).mean()

        return loss, cosine_sim.mean()
