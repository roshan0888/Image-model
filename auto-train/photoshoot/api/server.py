"""
Production API Server
======================
FastAPI server exposing the full photoshoot pipeline.

Endpoints:
    POST /process           — Full pipeline (expression + background + pose)
    POST /expression        — Expression only
    POST /background        — Background only
    POST /analyze           — Analyze face pose/quality (no processing)
    GET  /health            — Health check + model status
    GET  /presets           — List available backgrounds and expressions
    GET  /job/{job_id}      — Get job status (async mode)

Pricing hooks:
    Every request logs usage to usage.jsonl for billing integration.

Usage:
    python photoshoot/api/server.py --port 8001
"""

import os, sys, uuid, time, json, logging, argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [API] %(message)s")

# ── Constants ─────────────────────────────────────────────────────────────────
UPLOAD_DIR  = ROOT / "photoshoot/api/uploads"
OUTPUT_DIR  = ROOT / "photoshoot/api/outputs"
USAGE_LOG   = ROOT / "photoshoot/api/usage.jsonl"
for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BACKGROUND_PRESETS = [
    "studio_white", "studio_gray", "studio_black", "studio_cream",
    "studio_navy", "studio_charcoal",
    "gradient_white", "gradient_gray", "gradient_blue", "gradient_warm",
]
EXPRESSION_PRESETS = ["smile", "open_smile", "neutral"]


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Photoshoot AI API",
    description="Production face editing: expression + background + pose",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Lazy-loaded pipeline
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from photoshoot.pipeline.photoshoot_pipeline import PhotoshootPipeline
        _pipeline = PhotoshootPipeline()
    return _pipeline


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "expression": "LivePortrait + LoRA",
            "identity":   "InsightFace ArcFace",
            "background": "rembg / MODNet",
            "pose":       "3DDFA-V2",
        }
    }

@app.get("/presets")
def presets():
    return {
        "expressions": EXPRESSION_PRESETS,
        "backgrounds": BACKGROUND_PRESETS,
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze face pose and quality — no processing, instant."""
    job_id   = str(uuid.uuid4())[:8]
    img_path = str(UPLOAD_DIR / f"{job_id}.jpg")

    contents = await file.read()
    with open(img_path, "wb") as f:
        f.write(contents)

    try:
        from photoshoot.pose.pose_pipeline import PosePipeline
        pose = PosePipeline()
        result = pose.analyze(img_path)
        _log_usage(job_id, "analyze", file.filename)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        try: os.remove(img_path)
        except: pass

@app.post("/expression")
async def expression_only(
    file:       UploadFile = File(...),
    expression: str = Form("smile"),
):
    """Apply expression only — fastest endpoint."""
    if expression not in EXPRESSION_PRESETS:
        raise HTTPException(400, f"expression must be one of {EXPRESSION_PRESETS}")

    job_id   = str(uuid.uuid4())[:8]
    img_path = str(UPLOAD_DIR / f"{job_id}.jpg")
    out_path = str(OUTPUT_DIR / f"{job_id}_{expression}.jpg")

    contents = await file.read()
    with open(img_path, "wb") as f:
        f.write(contents)

    try:
        t0     = time.time()
        result = get_pipeline()._get_expression_pipeline().process(img_path, expression)
        img    = result.get("image")
        score  = result.get("identity_score", 0)

        if img is not None:
            cv2.imwrite(out_path, img if isinstance(img, np.ndarray) else np.array(img))

        _log_usage(job_id, "expression", file.filename, {
            "expression": expression,
            "identity_score": score,
            "time": time.time() - t0,
        })

        if img is None or score < 0.97:
            return JSONResponse({
                "status": "quality_gate_failed",
                "identity_score": score,
                "message": "Could not achieve 97% identity — try a different photo",
            }, status_code=422)

        return FileResponse(out_path, media_type="image/jpeg",
                            headers={"X-Identity-Score": str(round(score, 4))})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/background")
async def background_only(
    file:       UploadFile = File(...),
    background: str = Form("studio_white"),
):
    """Replace background only."""
    job_id   = str(uuid.uuid4())[:8]
    img_path = str(UPLOAD_DIR / f"{job_id}.jpg")
    out_path = str(OUTPUT_DIR / f"{job_id}_bg.jpg")

    contents = await file.read()
    with open(img_path, "wb") as f:
        f.write(contents)

    try:
        t0     = time.time()
        from photoshoot.background.background_pipeline import BackgroundPipeline
        bp     = BackgroundPipeline()
        result = bp.process(img_path, background=background, output_path=out_path)

        _log_usage(job_id, "background", file.filename, {
            "background": background,
            "method": result.get("method"),
            "time": time.time() - t0,
        })
        return FileResponse(out_path, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/process")
async def full_process(
    file:        UploadFile = File(...),
    expression:  str = Form("smile"),
    background:  str = Form("studio_white"),
    target_pose: str = Form("frontal"),
):
    """Full pipeline: pose → expression → background."""
    if expression not in EXPRESSION_PRESETS:
        raise HTTPException(400, f"expression must be one of {EXPRESSION_PRESETS}")

    job_id   = str(uuid.uuid4())[:8]
    img_path = str(UPLOAD_DIR / f"{job_id}.jpg")
    out_dir  = str(OUTPUT_DIR / job_id)
    os.makedirs(out_dir, exist_ok=True)

    contents = await file.read()
    with open(img_path, "wb") as f:
        f.write(contents)

    try:
        t0     = time.time()
        result = get_pipeline().process(
            source_path=img_path,
            expression=expression,
            background=background,
            target_pose=target_pose,
            output_dir=out_dir,
        )

        score = result.get("identity_score", 0)
        _log_usage(job_id, "full_process", file.filename, {
            "expression":     expression,
            "background":     background,
            "stages_applied": result.get("stages_applied", []),
            "identity_score": score,
            "time":           time.time() - t0,
        })

        if not result.get("quality_passed"):
            return JSONResponse({
                "status":         "quality_gate_failed",
                "identity_score": score,
                "stages_applied": result.get("stages_applied", []),
                "message":        "Could not maintain identity — try a clearer frontal photo",
            }, status_code=422)

        out_path = result["output_path"]
        return FileResponse(
            out_path,
            media_type="image/jpeg",
            headers={
                "X-Identity-Score":   str(round(score, 4)),
                "X-Stages-Applied":   ",".join(result.get("stages_applied", [])),
                "X-Processing-Time":  str(round(time.time() - t0, 2)),
            }
        )
    except Exception as e:
        log.error(f"Job {job_id} failed: {e}")
        raise HTTPException(500, str(e))
    finally:
        try: os.remove(img_path)
        except: pass


# ── Usage logging ─────────────────────────────────────────────────────────────

def _log_usage(job_id: str, endpoint: str, filename: str, meta: dict = {}):
    entry = {
        "ts":       datetime.now().isoformat(),
        "job_id":   job_id,
        "endpoint": endpoint,
        "filename": filename,
        **meta,
    }
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
