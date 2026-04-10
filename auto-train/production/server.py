#!/usr/bin/env python3
"""
Production API Server — Face Expression Editing

Endpoints:
  POST /api/edit          — Edit expression (with quality gate)
  POST /api/batch-edit    — Edit multiple expressions at once
  POST /api/feedback      — Submit like/dislike feedback
  GET  /api/expressions   — List available expressions
  GET  /api/health        — Health check
  GET  /result/{filename} — Serve result images

Usage:
  python server.py                    # Start on port 8000
  python server.py --port 8080        # Custom port
  python server.py --host 0.0.0.0     # Public access
"""

import os
import sys
import json
import time
import uuid
import shutil
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("server")

PROD_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(PROD_DIR, "uploads")
OUTPUT_DIR = os.path.join(PROD_DIR, "outputs")
FEEDBACK_DIR = os.path.join(PROD_DIR, "feedback")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Face Expression Editor API",
    description="Edit facial expressions while preserving identity",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve result images
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")

# Serve static UI files
STATIC_DIR = os.path.join(PROD_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)


@app.get("/")
async def serve_ui():
    """Serve the web UI."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# Global engine instance (loaded once at startup)
engine = None


@app.on_event("startup")
async def startup():
    """Load models on server start."""
    global engine
    logger.info("Loading Expression Engine...")

    # Setup drivers
    from expression_engine import setup_drivers, ExpressionEngine
    setup_drivers()
    engine = ExpressionEngine()

    logger.info("Server ready!")


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "gpu": "cuda" if engine and engine.device == "cuda" else "cpu",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/expressions")
async def list_expressions():
    """List available expressions."""
    if engine is None:
        raise HTTPException(503, "Engine not loaded")
    return {
        "expressions": engine.list_expressions(),
        "default_intensity": 1.0,
        "intensity_range": [0.1, 1.0],
        "identity_gate": engine.IDENTITY_GATE,
    }


@app.post("/api/edit")
async def edit_expression(
    photo: UploadFile = File(...),
    expression: str = Form(default="smile"),
    intensity: float = Form(default=1.0),
    identity_threshold: Optional[float] = Form(default=None),
):
    """
    Edit facial expression in uploaded photo.

    Returns the edited image URL if quality gate passes,
    or an error message if identity can't be preserved.
    """
    if engine is None:
        raise HTTPException(503, "Engine not loaded")

    # Validate intensity
    intensity = max(0.1, min(1.0, intensity))

    # Save uploaded file
    request_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(photo.filename)[1] or ".jpg"
    upload_path = os.path.join(UPLOADS_DIR, f"{request_id}{ext}")

    try:
        with open(upload_path, "wb") as f:
            content = await photo.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(400, f"Failed to save upload: {e}")

    # Run expression engine
    output_name = f"{request_id}_{expression}.png"

    try:
        result = engine.edit(
            source_path=upload_path,
            expression=expression,
            intensity=intensity,
            identity_threshold=identity_threshold,
            output_name=output_name,
        )
    except Exception as e:
        logger.error(f"Engine error: {e}")
        raise HTTPException(500, f"Processing error: {e}")

    # Build response
    response = {
        "request_id": request_id,
        "success": result["success"],
        "expression": expression,
        "intensity_requested": intensity,
        "intensity_used": result["intensity_used"],
        "identity_score": result["identity_score"],
        "expression_change": result["expression_change"],
        "attempts": result["attempts"],
        "time_seconds": result["time"],
        "message": result["message"],
    }

    if result["success"]:
        response["result_url"] = f"/results/{output_name}"
        response["result_filename"] = output_name

    if result.get("warnings"):
        response["warnings"] = result["warnings"]

    # Log for analytics
    _log_request(request_id, photo.filename, expression, intensity, result)

    return response


@app.post("/api/batch-edit")
async def batch_edit(
    photo: UploadFile = File(...),
    expressions: str = Form(default="smile,surprise,sad"),
    intensity: float = Form(default=1.0),
):
    """Edit multiple expressions at once. Returns results for each."""
    if engine is None:
        raise HTTPException(503, "Engine not loaded")

    intensity = max(0.1, min(1.0, intensity))
    expr_list = [e.strip() for e in expressions.split(",")]

    # Save upload
    request_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(photo.filename)[1] or ".jpg"
    upload_path = os.path.join(UPLOADS_DIR, f"{request_id}{ext}")

    with open(upload_path, "wb") as f:
        content = await photo.read()
        f.write(content)

    # Process each expression
    results = {}
    for expr in expr_list:
        output_name = f"{request_id}_{expr}.png"
        try:
            r = engine.edit(
                source_path=upload_path,
                expression=expr,
                intensity=intensity,
                output_name=output_name,
            )
            results[expr] = {
                "success": r["success"],
                "identity_score": r["identity_score"],
                "expression_change": r["expression_change"],
                "result_url": f"/results/{output_name}" if r["success"] else None,
                "message": r["message"],
                "time_seconds": r["time"],
            }
        except Exception as e:
            results[expr] = {"success": False, "message": str(e)}

    return {
        "request_id": request_id,
        "results": results,
    }


@app.post("/api/feedback")
async def submit_feedback(
    request_id: str = Form(...),
    expression: str = Form(...),
    liked: bool = Form(...),
    comment: Optional[str] = Form(default=None),
):
    """
    Submit user feedback (like/dislike) for a result.
    This data feeds into the RLHF training loop.
    """
    feedback = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "expression": expression,
        "liked": liked,
        "comment": comment,
    }

    feedback_file = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback) + "\n")

    logger.info(f"  Feedback: {request_id} {expression} {'👍' if liked else '👎'}")

    return {"status": "recorded", "request_id": request_id}


@app.get("/result/{filename}")
async def get_result(filename: str):
    """Serve a result image."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Result not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/stats")
async def get_stats():
    """Get basic usage statistics."""
    feedback_file = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
    request_log = os.path.join(FEEDBACK_DIR, "requests.jsonl")

    stats = {
        "total_requests": 0,
        "successful": 0,
        "failed": 0,
        "total_feedback": 0,
        "likes": 0,
        "dislikes": 0,
        "per_expression": {},
    }

    if os.path.exists(request_log):
        with open(request_log) as f:
            for line in f:
                r = json.loads(line)
                stats["total_requests"] += 1
                if r.get("success"):
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1

                expr = r.get("expression", "unknown")
                if expr not in stats["per_expression"]:
                    stats["per_expression"][expr] = {
                        "total": 0, "successful": 0, "avg_identity": 0,
                        "likes": 0, "dislikes": 0,
                    }
                stats["per_expression"][expr]["total"] += 1
                if r.get("success"):
                    stats["per_expression"][expr]["successful"] += 1

    if os.path.exists(feedback_file):
        with open(feedback_file) as f:
            for line in f:
                fb = json.loads(line)
                stats["total_feedback"] += 1
                if fb.get("liked"):
                    stats["likes"] += 1
                else:
                    stats["dislikes"] += 1

    if stats["total_feedback"] > 0:
        stats["like_rate"] = round(stats["likes"] / stats["total_feedback"], 3)

    return stats


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def _log_request(request_id, filename, expression, intensity, result):
    """Log every request for analytics."""
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "source_file": filename,
        "expression": expression,
        "intensity": intensity,
        "success": result["success"],
        "identity_score": result["identity_score"],
        "expression_change": result["expression_change"],
        "driver_used": result.get("driver_used"),
        "intensity_used": result.get("intensity_used"),
        "attempts": result["attempts"],
        "time": result["time"],
        "warnings": result.get("warnings", []),
    }

    log_file = os.path.join(FEEDBACK_DIR, "requests.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log) + "\n")


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
