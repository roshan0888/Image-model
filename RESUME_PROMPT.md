# Prompt for existing account — just add auto-train code

Copy everything below the line and paste as your first message:

---

I already have LivePortrait and MagicFace set up in this studio. Now I need to pull in the training/pipeline code from my GitHub repo and wire it up. Don't ask questions, just execute.

## Step 1: Pull the auto-train code from GitHub

```bash
cd /teamspace/studios/this_studio
git init 2>/dev/null
git remote add origin https://github.com/roshan0888/Image-model.git 2>/dev/null || git remote set-url origin https://github.com/roshan0888/Image-model.git
git fetch origin main
git checkout origin/main -- auto-train/ PROJECT_HANDOFF.md SETUP_PROMPT.md
```

## Step 2: Install extra dependencies (skip if already installed)

```bash
pip install basicsr gfpgan lpips scikit-learn gradio fastapi uvicorn pexels-api icrawler anthropic rembg rich dill scipy lmdb ffmpeg-python tyro imageio tqdm
```

## Step 3: Download GFPGAN weights

```bash
mkdir -p /teamspace/studios/this_studio/auto-train/gfpgan/weights
wget -q -P /teamspace/studios/this_studio/auto-train/gfpgan/weights \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

## Step 4: Symlink the existing models into auto-train paths

The auto-train code expects InsightFace models at `auto-train/MagicFace/third_party_files/`. We already have them at `Magicface/MagicFace/third_party_files/`. Create symlinks so both paths work:

```bash
# Link InsightFace models
ln -sfn /teamspace/studios/this_studio/Magicface/MagicFace /teamspace/studios/this_studio/auto-train/MagicFace

# Link natural_pipeline.py from Magicface to auto-train (auto-train has its own version, but keep both)
# The auto-train version is the latest (v9), so it takes priority
```

## Step 5: Create data directories

```bash
mkdir -p /teamspace/studios/this_studio/auto-train/raw_data/{model_photos,cleaned/{smile,neutral},pairs,bulk_scrape,rejected_underage}
mkdir -p /teamspace/studios/this_studio/auto-train/training_engine/{checkpoints,checkpoints_v2,checkpoints_v3,logs}
mkdir -p /teamspace/studios/this_studio/auto-train/output
mkdir -p /teamspace/studios/this_studio/auto-train/temp
```

## Step 6: Set API keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."      # For Claude orchestrator
export PEXELS_API_KEY="..."                 # For scraping (free at pexels.com/api)
```

## Step 7: Verify everything connects

```bash
cd /teamspace/studios/this_studio/auto-train
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')

# Check InsightFace models exist
import os
af_path = 'MagicFace/third_party_files/models/antelopev2'
if os.path.exists(af_path):
    files = os.listdir(af_path)
    print(f'AntelopeV2: {len(files)} files — {files}')
else:
    print(f'ERROR: {af_path} not found!')

# Check LivePortrait
lp_path = '/teamspace/studios/this_studio/LivePortrait/pretrained_weights'
if os.path.exists(lp_path):
    print(f'LivePortrait weights: OK')
else:
    print(f'ERROR: LivePortrait weights not found!')

# Check GFPGAN
gfpgan_path = 'gfpgan/weights/GFPGANv1.4.pth'
if os.path.exists(gfpgan_path):
    print(f'GFPGAN: OK')
else:
    print(f'WARNING: GFPGAN not downloaded yet')

print('Setup complete!')
"
```

## Step 8: Start the autonomous training loop

```bash
cd /teamspace/studios/this_studio/auto-train

# Quick test first (verify everything works end-to-end)
python run_autonomous.py --cycles 1 --quick

# Then full training loop
python loop.py --target 0.995 --max-cycles 20
```

## What this project does

Autonomous AI photo editing system — adds natural smiles to faces while preserving 99.5% identity (ArcFace).

**The loop:** Scrape photos → Clean/filter → Cluster by identity → Generate training pairs via LivePortrait → Fine-tune LivePortrait with LoRA → Evaluate → Claude adjusts hyperparams → Repeat

**Key files you just pulled:**
- `auto-train/loop.py` — Main autonomous loop (scrape → clean → pair → train → validate → repeat)
- `auto-train/natural_pipeline.py` — 12 expression presets (smile, surprise, angry, etc.)
- `auto-train/training_engine/training/trainer.py` — LoRA fine-tuning (rank=8, alpha=16, identity_loss=5.0)
- `auto-train/training_engine/data_engine/model_photo_scraper.py` — Pexels/Unsplash/Google scraper
- `auto-train/training_engine/data_engine/identity_cluster.py` — DBSCAN clustering on ArcFace embeddings
- `auto-train/training_engine/configs/pipeline_config.yaml` — All config (LR, losses, LoRA, thresholds)
- `auto-train/photoshoot/api/server.py` — Production FastAPI (port 8001)
- `auto-train/face_filter.py` — Quality filter (blur, pose, face size, age)
- `PROJECT_HANDOFF.md` — Complete technical documentation

**Critical gotcha:** InsightFace root path must be `MagicFace/third_party_files` (NOT `.../third_party_files/models`). The symlink in Step 4 handles this.

**Status:** All infrastructure built. Need to scrape fresh data and run training cycles to reach 99.5% identity target. The loop handles everything autonomously.

Read `PROJECT_HANDOFF.md` for the full deep-dive. Now set it up and verify. Don't ask questions, just execute.
