# Per-User LoRA Training Pipeline

The production path — trains a mini-model per user for 90%+ identity preservation.

## What this does

For each user, trains a tiny LoRA (~50MB) on their 10-20 photos. That LoRA gets loaded at inference time and combined with InstantID + LivePortrait for strong expression change **with** exact identity preservation.

This is the same approach Aragon, HeadshotPro, and Photo AI use.

## Setup (once)

All models already downloaded by `download_models.py` and `download_realvis.py`. No extra setup needed.

## Workflow (per user)

### 1. Collect photos

You need **10-20 photos** of ONE person. They should vary in:
- Angle (frontal, 3/4, side)
- Expression (neutral, smile, laugh — diverse)
- Lighting (indoor, outdoor, different times)
- Clothing / background (don't all be the same)

Put raw photos in any folder, e.g. `~/uploads/john_doe_raw/`.

### 2. Prep photos

Auto-crops to face + context, resizes to 1024×1024, filters blurry/bad-pose:

```bash
python prep_photos.py \
    --user_id john_doe \
    --input_dir ~/uploads/john_doe_raw
```

This saves cleaned photos to `user_data/john_doe/photos/`.

### 3. Train LoRA

```bash
python train_user_lora.py \
    --user_id john_doe \
    --photos_dir user_data/john_doe/photos \
    --steps 1000
```

**Time:** ~20-25 min on T4 (~$0.10)
**Output:** `user_data/john_doe/lora/pytorch_lora_weights.safetensors`

### 4. Generate

```bash
python inference_with_lora.py \
    --user_id john_doe \
    --source_image user_data/john_doe/photos/001_neutral.jpg \
    --prompt "ohwx person with big happy smile, professional headshot"
```

Uses the trained LoRA + InstantID + LivePortrait together.
**Time:** ~60 sec on T4.

## How it works (the full stack)

```
 Source photo
     │
     ▼
┌────────────────────────────────────────────┐
│  Stage 1: LivePortrait                     │
│  Applies smile geometry (mouth/face)       │
└────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│  Stage 2: SDXL img2img                     │
│  ├─ RealVisXL V4.0 (photoreal base)        │
│  ├─ User's LoRA (memorized exact identity) │  ← THE KEY PIECE
│  ├─ InstantID (structural face lock)       │
│  └─ Refines at strength=0.45               │
└────────────────────────────────────────────┘
     │
     ▼
 Output: 90%+ identity + real smile + photoreal
```

## Expected quality

| Approach | Identity | Smile Clarity |
|----------|---------|---------------|
| InstantID alone | 73-80% | Clear |
| InstantID + LP hybrid (no user LoRA) | 77% | Clear |
| **InstantID + LP + user LoRA (this)** | **90-95%** | **Clear** |

## Tuning knobs

- `--rank 16` → LoRA capacity. Higher = more identity detail, more VRAM, slower. Try 16, 24, 32.
- `--steps 1000` → training iterations. 500 is minimum, 1500 is overkill.
- `--lora_scale 0.85` (inference) → how strongly LoRA affects output. 0.7-0.95 is useful range.

## Limitations

- 1 LoRA per person → need to retrain per user (20 min)
- Bad photos in → bad LoRA out (hence prep step)
- Subjects looking very different in different photos (wildly different hair, heavy makeup variance) can confuse training
