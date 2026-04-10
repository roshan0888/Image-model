#!/usr/bin/env python3
"""
experiment.py — THE EDITABLE FILE

This is what the autonomous agent modifies each iteration.
Contains the experiment configuration for one LP run.

Equivalent to Karpathy's train.py — the single file the agent edits.
"""

# ═══════════════════════════════════════════════════
# EXPERIMENT CONFIG — modify these values
# ═══════════════════════════════════════════════════

CONFIG = {
    # Driver image (the smile source)
    "driver": "/teamspace/studios/this_studio/LivePortrait/assets/examples/driving/d12.jpg",

    # Expression multiplier (higher = stronger smile)
    "multiplier": 1.5,

    # Which face region to animate
    # Options: "lip", "all", "exp", "eyes"
    "animation_region": "lip",

    # Driving mode
    # Options: "expression-friendly", "pose-friendly"
    "driving_option": "expression-friendly",

    # Stitching (blends face back into image)
    "flag_stitching": True,

    # Relative motion (expression relative to source)
    "flag_relative_motion": True,

    # Eye/lip retargeting (uses only ratios, not geometry)
    "flag_eye_retargeting": False,
    "flag_lip_retargeting": False,
}

# ═══════════════════════════════════════════════════
# DESCRIPTION — what this experiment is trying
# ═══════════════════════════════════════════════════

DESCRIPTION = "baseline: lip mode, d12.jpg, 1.5x multiplier"
