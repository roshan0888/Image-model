# Prompt to paste in new Claude Code session

Copy everything below the line and paste it as your first message:

---

I have an AI photo editing project on GitHub. Clone it and set everything up so I can resume training. Here's what to do:

## Step 1: Clone the repo

```
git clone https://github.com/roshan0888/Image-model.git /teamspace/studios/this_studio/
```

If the studio already has files, do:
```
cd /teamspace/studios/this_studio
git init && git remote add origin https://github.com/roshan0888/Image-model.git && git pull origin main
```

## Step 2: Install all dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu insightface opencv-python-headless basicsr gfpgan lpips
pip install scikit-learn gradio fastapi uvicorn pexels-api icrawler anthropic
pip install rembg tyro dill scipy imageio ffmpeg-python lmdb tqdm rich
```

## Step 3: Download model weights (these weren't pushed to git — too large)

### LivePortrait pretrained weights
```bash
cd /teamspace/studios/this_studio/LivePortrait
# Download from HuggingFace
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights
```

### InsightFace AntelopeV2
```bash
mkdir -p /teamspace/studios/this_studio/auto-train/MagicFace/third_party_files/models/antelopev2
cd /teamspace/studios/this_studio/auto-train/MagicFace/third_party_files/models/antelopev2
# Download antelopev2 pack from insightface model zoo:
# https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo
# Files needed: 1k3d68.onnx, 2d106det.onnx, det_10g.onnx, genderage.onnx, w600k_r50.onnx
```

### GFPGAN weights
```bash
mkdir -p /teamspace/studios/this_studio/auto-train/gfpgan/weights
wget -P /teamspace/studios/this_studio/auto-train/gfpgan/weights \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

## Step 4: Create data directories
```bash
mkdir -p /teamspace/studios/this_studio/auto-train/raw_data/{model_photos,cleaned/{smile,neutral},pairs,bulk_scrape,rejected_underage}
mkdir -p /teamspace/studios/this_studio/auto-train/training_engine/{checkpoints,logs}
mkdir -p /teamspace/studios/this_studio/auto-train/output
mkdir -p /teamspace/studios/this_studio/auto-train/temp
```

## Step 5: Set API keys
```bash
export ANTHROPIC_API_KEY="sk-ant-..."      # For Claude orchestrator (autonomous training decisions)
export PEXELS_API_KEY="..."                 # For scraping photos (get free key at pexels.com/api)
```

## Step 6: Verify setup
```bash
cd /teamspace/studios/this_studio/auto-train
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import insightface; print(f'InsightFace: {insightface.__version__}')
import cv2; print(f'OpenCV: {cv2.__version__}')
print('All good!')
"
```

## Step 7: Start training from scratch
```bash
cd /teamspace/studios/this_studio/auto-train

# Full autonomous loop — scrapes data, cleans, pairs, trains, evaluates, repeats
python loop.py --target 0.995 --max-cycles 20

# OR quick test first (50 images, 300 steps — just to verify everything works)
python run_autonomous.py --cycles 1 --quick
```

## What this project does

This is an autonomous AI photo editing system. The goal is 99.5% identity preservation (ArcFace cosine similarity) when editing facial expressions (mainly adding smiles).

**How it works:**
1. Scrapes high-quality portrait photos from Pexels/Unsplash/Google
2. Filters them (blur, pose, face size, age, quality)
3. Clusters by identity using DBSCAN on ArcFace embeddings
4. Generates training pairs using LivePortrait (source face → smiled version)
5. Fine-tunes LivePortrait with LoRA adapters (only 2.35M params, fits on T4 GPU)
6. Evaluates identity preservation on test set
7. Claude AI analyzes results and adjusts hyperparameters
8. Repeats until 99.5% identity score is reached

**Key files:**
- `auto-train/loop.py` — Main autonomous training loop
- `auto-train/natural_pipeline.py` — Expression presets (12 expressions)
- `auto-train/training_engine/training/trainer.py` — LoRA trainer
- `auto-train/training_engine/training/losses.py` — Loss functions (identity=5.0, expression=4.0, perceptual=1.5, pixel=1.0)
- `auto-train/training_engine/data_engine/model_photo_scraper.py` — Data scraping
- `auto-train/training_engine/configs/pipeline_config.yaml` — All training config
- `auto-train/photoshoot/api/server.py` — Production FastAPI server (port 8001)
- `PROJECT_HANDOFF.md` — Full detailed documentation of everything

**Important gotcha:** InsightFace model path must be `MagicFace/third_party_files` (NOT `MagicFace/third_party_files/models`). The antelopev2 folder sits directly inside third_party_files.

**Current status:** Infrastructure 100% built. Had ~9,149 training pairs and ~90-95% identity score before credits ran out. Need to regenerate training data and resume the loop to push from 90% → 99.5%.

Read `PROJECT_HANDOFF.md` for the complete technical deep-dive — every model, every threshold, every config value is documented there.

Now set everything up and verify it works. Don't ask questions, just execute.
