# AI Photo Editing System — Full Project Handoff

> **Date:** April 10, 2026  
> **Goal:** Production-level AI photo editing with 99.5% ArcFace identity preservation  
> **Status:** Training infrastructure complete, autonomous loop built, ~9,149 training pairs generated, loop not currently running  

---

## 1. Vision & Architecture

We're building a **4-pillar AI photo editing system**:

| Pillar | Status | Key File |
|--------|--------|----------|
| Expression editing (smile, open_smile) | ✅ Built + training | `natural_pipeline.py`, `loop.py` |
| Background replacement | ✅ Working | `photoshoot/pipeline/photoshoot_pipeline.py` |
| Pose correction | ✅ Working | `photoshoot/pipeline/photoshoot_pipeline.py` |
| Full integrated pipeline + API | ✅ Deployed | `photoshoot/api/server.py` (FastAPI, port 8001) |

**The core challenge:** LivePortrait does great expression transfer but destroys identity (drops to ~0.70 ArcFace similarity). We're fine-tuning it with LoRA to hit 99.5% identity preservation while still producing visible expression changes.

---

## 2. Models We Use

### LivePortrait (Primary — Expression Transfer)
- **Repo:** `/teamspace/studios/this_studio/LivePortrait/`
- **What it does:** Neural warping-based face expression transfer
- **Components:** appearance_feature_extractor → motion_extractor → warping_module → spade_generator → stitching_retargeting_module
- **Mode:** Retargeting (lip region) — `flag_lip_retargeting=True`
- **256×256 bottleneck:** LP internally crops face to 256×256 — this destroys teeth/mouth detail (known critical limitation)

### InsightFace (AntelopeV2 — Identity & Detection)
- **Path:** `auto-train/MagicFace/third_party_files/models/antelopev2/`
- **Used for:** Face detection (640×640 det_size, 0.3 thresh), ArcFace embedding extraction, 106-point landmarks, pose estimation
- **Critical:** Model root must be `MagicFace/third_party_files` (NOT `MagicFace/third_party_files/models`)

### GFPGAN v1.4 (Face Restoration)
- **Path:** `auto-train/gfpgan/weights/GFPGANv1.4.pth`
- **Used for:** Enhancing mouth region detail in the 256×256 crop
- **Important:** Only applied to mouth region to avoid identity drift

### Other
- **rembg** — AI background removal (CPU, no GPU needed)
- **MODNet** — Hair-detail background matting (fallback)
- **3DDFA-V2** — 3D face reconstruction for pose correction
- **LPIPS** — Perceptual loss metric during training

---

## 3. Directory Structure

```
/teamspace/studios/this_studio/auto-train/
├── loop.py                          # MAIN — Production autonomous loop
├── run_autonomous.py                # Batch training (older entry point)
├── autonomous_loop.py               # V3 full pipeline with reinforcement
├── natural_pipeline.py              # Expression preset library (v9, 12 expressions)
├── face_filter.py                   # InsightFace quality filter
├── face_quality_gate.py             # Additional quality gating
├── monitor.py / live_monitor.py     # Terminal dashboards
├── feed_new_images.py               # Background image feeder
│
├── training_engine/
│   ├── training/
│   │   ├── trainer.py               # LivePortraitTrainer — main LoRA trainer
│   │   ├── losses.py                # Identity + expression + perceptual + pixel losses
│   │   ├── dataset.py               # FaceExpressionDataset (loads pairs, augments)
│   │   ├── lora_modules.py          # LoRA injection for Conv2d and Linear layers
│   │   └── region_losses.py         # Region-aware losses for lip mode
│   │
│   ├── data_engine/
│   │   ├── model_photo_scraper.py   # Pexels/Unsplash/Google scraper
│   │   ├── identity_cluster.py      # DBSCAN clustering on ArcFace embeddings
│   │   ├── cleaner.py               # Multi-stage image filter pipeline
│   │   ├── collector.py             # Query generator + download orchestrator
│   │   ├── self_pair_generator.py   # LivePortrait synthetic pair generation
│   │   └── arcface_pytorch.py       # ArcFace embedding helper
│   │
│   ├── evaluation/
│   │   └── failure_detector.py      # Failure categorization by demographic
│   │
│   ├── orchestrator/
│   │   └── llm_orchestrator.py      # Claude-based training decisions
│   │
│   ├── configs/
│   │   └── pipeline_config.yaml     # Central config (LR, LoRA, losses, etc.)
│   │
│   ├── checkpoints/                 # LoRA weights (v1)
│   ├── checkpoints_v2/              # Evolution Strategies variant
│   ├── checkpoints_v3/              # Latest experiments
│   └── logs/                        # TensorBoard + JSON metrics
│
├── photoshoot/
│   ├── app.py                       # Gradio smile editor UI
│   ├── hybrid_engine.py             # LP + GFPGAN mouth enhancement
│   ├── pipeline/photoshoot_pipeline.py  # Master orchestrator (pose→expr→bg)
│   ├── background/background_pipeline.py # rembg/MODNet + compositing
│   ├── pose/pose_pipeline.py        # 3DDFA-V2 pose correction
│   └── api/server.py               # FastAPI production server (port 8001)
│
├── pose_studio/                     # Gradio pose+expression app (port 7860)
│   ├── app.py
│   └── engine/pose_engine.py
│
├── raw_data/
│   ├── model_photos/                # Scraped raw images (~1,184)
│   ├── cleaned/                     # After quality filter (~922)
│   │   ├── smile/
│   │   └── neutral/
│   ├── pairs/                       # Training pairs (~17,827 files = ~9,149 pairs)
│   │   ├── p00000_src.jpg
│   │   ├── p00000_drv_smile.jpg
│   │   └── training_pairs.jsonl
│   ├── bulk_scrape/                 # Staging area
│   ├── starnow_ugc_models/         # UGC model data
│   └── rejected_underage/           # Age filter rejects
│
├── MagicFace/third_party_files/     # InsightFace AntelopeV2 models
└── gfpgan/weights/                  # GFPGAN v1.4 weights
```

---

## 4. How Training Works

### LoRA Fine-Tuning Strategy

We inject **LoRA adapters** into LivePortrait's frozen network — only ~2.35M trainable params out of 1.2B total:

| Module | LoRA Targets | Trainable Params | Why |
|--------|-------------|-----------------|-----|
| SPADE Generator | fc, G_middle_0 through G_middle_5 | ~2.3M | Controls final pixel output — the "paint brush" |
| Warping Network | third.conv, fourth | ~50K | Spatial deformation geometry |
| Motion Extractor | fc_exp, fc_kp | ~900 | How model interprets expressions |
| Stitching Module | *(disabled)* | 0 | Saves VRAM on T4 |

**LoRA Config:**
```yaml
rank: 8
alpha: 16        # scaling = alpha/rank = 2x
dropout: 0.05
```

### Loss Functions

```python
total_loss = (
    5.0 * identity_loss +      # ArcFace cosine similarity (PRIMARY)
    4.0 * expression_loss +    # Landmark displacement toward target
    1.5 * perceptual_loss +    # LPIPS visual quality
    1.0 * pixel_loss +         # L1 in RGB space
    1e-4 * regularization      # LoRA weight decay
)
```

**Identity loss is king** — weight 5.0, because that's our 99.5% target. The ArcFace model is frozen ONNX (no gradients through it).

### Training Hyperparameters

| Param | Value | Notes |
|-------|-------|-------|
| Batch size | 2 | T4 VRAM limited |
| Gradient accumulation | 4 | Effective batch = 8 |
| Learning rate | 1e-4 | Cosine annealing with 500-step warmup |
| Mixed precision | fp16 | VRAM efficiency |
| Max grad norm | 1.0 | Gradient clipping |
| Eval frequency | Every 1000 steps | Identity score on test set |
| Save frequency | Every 5000 steps | Checkpoint LoRA weights |
| Early stopping | Patience 5 evals, min_delta 0.001 | Stop if plateaued |

---

## 5. How Data Pipeline Works

### Step 1: Scraping (`model_photo_scraper.py`)

**Sources:** Pexels API, Unsplash API, Google/Bing Image Search

**Queries:** 15 variants per expression, targeting DSLR-quality portraits:
- `"photogenic model perfect smile portrait studio"`
- `"beautiful woman radiant smile fashion photography"`
- Demographic boost queries for diversity (Asian, Black, Latina, Indian, etc.)

**Rate:** ~500-1000 images per scraping cycle

### Step 2: Cleaning (`cleaner.py`, `face_filter.py`)

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Readable | — | Skip corrupted files |
| Resolution | ≥256×256 face | Need detail |
| Blur (Laplacian) | variance > 50 | Reject motion blur |
| Face count | Exactly 1, confidence ≥0.6 | Single identity only |
| Face size | ≥100×100px | Large enough to process |
| Pose | yaw <25°, pitch <25° | Frontal faces only |
| Brightness | mean 30-240 | No silhouettes/blowouts |
| Age | ≥18 | Legal requirement |

**Pass rate:** 40-60% of scraped images survive cleaning.

### Step 3: Identity Clustering (`identity_cluster.py`)

- Extract ArcFace embedding per image
- DBSCAN clustering (eps=0.45, min_samples=2)
- Same-person threshold: cosine similarity ≥ 0.55
- Cluster size: 3-50 images
- Output: `training_pairs.jsonl` manifest

### Step 4: Pair Generation

**Two types:**

1. **Real pairs** — Different expressions of same person from clusters (neutral → smile)
2. **Synthetic pairs** — LivePortrait generates the target expression from a single neutral image
   - Uses driver images (d30.jpg for smile, d12.jpg for open_smile)
   - Identity gate: pair only saved if ArcFace similarity ≥ 0.88-0.90

**Current dataset:** ~9,149 pairs (17,827 files) after face_filter.py removed 237 bad pairs

---

## 6. The Autonomous Loop

### How It Runs

```bash
# Production loop — runs until 99.5% or 20 cycles
python loop.py --target 0.995 --max-cycles 20

# Quick test
python run_autonomous.py --cycles 3 --quick
```

### Cycle Logic

```
FOR each cycle (1..20):
  1. SCRAPE — Download 500 + (cycle-1)*200 new images
  2. CLEAN — Filter through quality pipeline
  3. PAIR — Cluster identities + generate training pairs
  4. TRAIN — Run 15000 + (cycle-1)*5000 LoRA training steps
  5. VALIDATE — Measure ArcFace identity on test set
  6. DECIDE — Claude LLM analyzes metrics, recommends adjustments
  
  IF avg_identity >= 0.995: STOP (target reached!)
  ELSE: Apply Claude's recommendations, continue next cycle
```

### Claude Orchestrator (`llm_orchestrator.py`)

After each cycle, Claude analyzes training metrics and returns:
- Learning rate adjustments
- Loss weight rebalancing
- Data collection priorities (e.g., "collect more Asian faces aged 50-65")
- LoRA rank changes
- Whether to continue or stop

**Model:** claude-sonnet-4-20250514 (requires `ANTHROPIC_API_KEY`)

---

## 7. Photoshoot Pipeline (Production)

### Full Pipeline (`photoshoot_pipeline.py`)

Stages run sequentially with an **identity gate** (0.97 threshold) — if any stage drops identity below 97%, it's skipped:

```
Input Image
  → Analyze (pose estimation)
  → Pose Correction (3DDFA-V2, if needed)
  → Expression Editing (LivePortrait + optional GFPGAN mouth fix)
  → Background Replacement (rembg + preset/gradient/SD)
  → Output (with quality scores)
```

### API Endpoints (`server.py` — FastAPI, port 8001)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/process` | POST | Full pipeline (pose + expression + background) |
| `/expression` | POST | Expression editing only |
| `/background` | POST | Background replacement only |
| `/analyze` | POST | Face pose analysis (instant, no processing) |
| `/health` | GET | Health check |
| `/presets` | GET | List available background presets |

**Background presets:** studio_white, studio_gray, studio_black, studio_cream, studio_navy, studio_charcoal + gradient variants

### Expression Presets (`natural_pipeline.py`)

12 expressions available: smile, open_smile, surprise, sad, angry, laugh, wink, shy, serious, confident, flirty, thoughtful

Each uses specific LP driver images with calibrated multipliers.

---

## 8. Known Issues & Critical Notes

### The 256×256 Bottleneck (CRITICAL)
LivePortrait internally processes faces at 256×256. This destroys fine details like teeth and mouth texture. GFPGAN mouth-region enhancement partially mitigates this but isn't perfect. This is the main quality blocker for photoshoot-grade output.

### InsightFace Model Path (GOTCHA)
```python
# CORRECT:
root = str(ROOT / "MagicFace/third_party_files")

# WRONG (will fail silently):
root = str(ROOT / "MagicFace/third_party_files/models")
```
The `antelopev2` folder sits directly inside `third_party_files`, not inside a `models` subdirectory.

### Common Errors Fixed During Development

| Error | Cause | Fix |
|-------|-------|-----|
| `pipeline.process()` returns wrong type | Code assumed `(img, meta)` tuple | Returns single dict — use `result.get("image")` |
| `IdentityClusterer` not found | Wrong class name | Actual class: `IdentityClusterEngine` |
| `file_idx_offset="auto"` crashes | icrawler doesn't accept string | Changed to `file_idx_offset=0` |
| Duplicate loop processes | Two loop.py running simultaneously | Kill with `ps aux \| grep loop.py` |
| Carnival/masked photos in dataset | No face quality filter | Added `face_filter.py` with InsightFace |

---

## 9. GPU & Resource Requirements

| Resource | Requirement | Notes |
|----------|------------|-------|
| GPU | NVIDIA T4 (16GB VRAM) minimum | Training with batch=2, grad_accum=4 |
| Mixed precision | fp16 | Required for T4 VRAM |
| Disk | ~10GB for dataset + models | Grows with scraping cycles |
| RAM | 16GB+ | InsightFace + LivePortrait loading |

---

## 10. How to Resume Training

```bash
cd /teamspace/studios/this_studio/auto-train
n
# Option 1: Full autonomous loop
python loop.py --target 0.995 --max-cycles 20

# Option 2: Skip scraping, use existing 9K+ pairs
python loop.py --target 0.995 --max-cycles 20 --skip-scrape

# Option 3: Quick test (50 images, 300 steps)
python run_autonomous.py --cycles 1 --quick

# Monitor progress
python monitor.py
# or
tensorboard --logdir training_engine/logs
```

### Checkpoints Location
- `training_engine/checkpoints/` — V1 LoRA weights
- `training_engine/checkpoints_v2/` — Evolution Strategies variant
- `training_engine/checkpoints_v3/` — Latest experiments
- Format: `.pt` files (LoRA weights only, not full model)

---

## 11. Tech Stack Summary

**Core:** Python 3.8+, PyTorch 2.0+, ONNX Runtime  
**Models:** LivePortrait, InsightFace AntelopeV2, GFPGAN v1.4, 3DDFA-V2  
**Training:** LoRA fine-tuning, ArcFace identity loss, LPIPS perceptual loss  
**Data:** Pexels/Unsplash/Google scraping, DBSCAN clustering, synthetic pair generation  
**Serving:** FastAPI (production), Gradio (dev UIs)  
**Orchestration:** Claude API for autonomous training decisions  
**Infra:** Lightning.ai Studios, NVIDIA T4 GPU  

---

## 12. Current Metrics & Where We Are

- **Training pairs:** ~9,149 (after filtering)
- **Cleaned images:** ~922
- **Raw scraped:** ~1,184
- **Best identity score achieved:** ~0.90-0.95 range (needs more training cycles)
- **Target:** 0.995 (99.5%)
- **Gap:** Need to run more autonomous cycles with the improved dataset

**Next steps to reach 99.5%:**
1. Resume `loop.py` with current 9K pairs
2. Let Claude orchestrator tune hyperparams per cycle
3. If plateauing at ~0.95, consider increasing LoRA rank (8→16) or unfreezing more SPADE layers
4. Use failure detector to identify weak demographics and scrape targeted data
5. Consider full fine-tuning of SPADE if LoRA capacity is exhausted
