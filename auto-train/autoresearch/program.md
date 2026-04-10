# Autoresearch — Face Expression Editing

## Goal

Get the **highest avg_identity** (ArcFace cosine similarity) while maintaining **visible smile** (mouth_disp > 0.015) across 20 diverse test faces.

## Setup

The repo has these files:
- `prepare.py` — FROZEN. Contains LP models, ArcFace eval, test faces. Do not modify.
- `experiment.py` — THE ONLY FILE YOU EDIT. Contains the config dict for one LP run.
- `program.md` — These instructions. Read-only.
- `results.tsv` — Experiment log. Append results here.
- `autoloop.py` — The runner. Loads config, calls evaluate(), logs results.

## What you CAN modify

Only `experiment.py`. You can change any value in the CONFIG dict:

- `driver`: path to driving image (any .jpg in LivePortrait/assets/examples/driving/ or photoshoot/drivers/)
- `multiplier`: float, typically 0.3-3.0. Higher = stronger expression.
- `animation_region`: "lip" (lips only), "all" (full face), "exp" (expression keypoints), "eyes" (eyes only)
- `driving_option`: "expression-friendly" or "pose-friendly"
- `flag_stitching`: True/False (blends face back into image)
- `flag_relative_motion`: True/False
- `flag_eye_retargeting`: True/False (uses eye close ratios instead of full geometry)
- `flag_lip_retargeting`: True/False (uses lip close ratios)

## What you CANNOT modify

- `prepare.py` — the evaluation function, test faces, and model loading are fixed.
- Do not install new packages.
- Do not modify LivePortrait source code.

## The metric

**Primary: `avg_identity`** — average ArcFace cosine similarity across 20 test faces. Higher is better. Target: 0.95+

**Constraint: `avg_mouth_disp`** — must be > 0.015 for the smile to be visible. If mouth_disp < 0.015, the experiment is invalid even if identity is high (because there's no visible smile).

**Composite score** = avg_identity × 0.8 + expression_bonus × 0.2
  where expression_bonus = min(avg_mouth_disp / 0.02, 1.0)

## Available drivers

LP built-in:
- d12.jpg — big open smile (teeth showing)
- d30.jpg — gentle closed smile
- d8.jpg — sad expression
- d9.jpg — scream
- d19.jpg — surprise
- d38.jpg — angry

Scraped (in ../photoshoot/drivers/):
- subtle/subtle_0001.jpg through subtle_0028.jpg
- photoshoot/photoshoot_0001.jpg through photoshoot_0079.jpg
- natural/natural_0001.jpg through natural_0014.jpg

## Experiment strategy hints

1. Start with baseline (lip mode, d12.jpg, 1.5x)
2. Try different multipliers: 0.5, 0.8, 1.0, 1.2, 1.5, 2.0
3. Try different drivers (subtle vs photoshoot vs LP built-in)
4. Try animation_region="all" vs "lip" vs "exp"
5. Try retargeting modes (flag_eye_retargeting + flag_lip_retargeting)
6. Try driving_option="pose-friendly" 
7. Try flag_stitching=False
8. Combine best settings from individual experiments

## Keep/discard rules

- If composite score IMPROVES → keep (advance)
- If composite score is EQUAL or WORSE → discard (revert)
- If avg_mouth_disp < 0.015 → discard (no visible smile)
- If failures > 5 (out of 20 faces) → discard (unstable config)
- Simpler is better: if two configs give same score, keep simpler one

## NEVER STOP

Run experiments continuously. If stuck, try more radical changes:
- Very high multipliers (2.5, 3.0)
- Very low multipliers (0.3, 0.4)
- Combining retargeting with lip mode
- Different drivers from the scraped collection
- flag_relative_motion=False
