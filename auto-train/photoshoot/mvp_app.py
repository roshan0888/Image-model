#!/usr/bin/env python3
"""
MVP Gradio App — Photogenic Smile Editor

Simple UI:
  - Upload photo
  - Click "Apply Smile"
  - See result with identity score
  - Like/Dislike buttons → RLHF feedback
  - RLHF dashboard shows learning progress
"""

import os
import sys
import cv2
import json
import time
import numpy as np
import gradio as gr

PHOTO_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_DIR = os.path.join(os.path.dirname(PHOTO_DIR), "production")
sys.path.insert(0, PHOTO_DIR)
sys.path.insert(0, PROD_DIR)

print("Loading MVP Smile Engine...")
from mvp_engine import MVPSmileEngine
engine = MVPSmileEngine()

# Load RLHF
try:
    from rlhf import RLHFSystem
    rlhf = RLHFSystem()
    print(f"RLHF loaded ({rlhf._total_feedback_count()} feedbacks)")
except Exception as e:
    print(f"RLHF not available: {e}")
    rlhf = None

print("Ready!")

# Track last result for feedback
last = {"request_id": None, "driver": "d12.jpg", "multiplier": 1.0, "score": 0}
counter = [0]


def apply_smile(input_image):
    if input_image is None:
        return None, "Upload a photo first"

    # Save input
    temp = os.path.join(PHOTO_DIR, "temp", "mvp_input.jpg")
    os.makedirs(os.path.dirname(temp), exist_ok=True)
    img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp, img, [cv2.IMWRITE_JPEG_QUALITY, 98])

    # Run
    r = engine.smile(temp)
    counter[0] += 1
    req_id = f"mvp_{int(time.time())}_{counter[0]}"

    last["request_id"] = req_id
    last["multiplier"] = r.get("multiplier", 0)
    last["score"] = r.get("identity_score", 0)

    # Record in RLHF
    if rlhf:
        try:
            rlhf.record_attempt(
                request_id=req_id,
                expression="smile",
                driver_used="d12.jpg",
                intensity_used=r.get("multiplier", 0),
                identity_score=r.get("identity_score", 0),
                expression_change=0,
            )
        except:
            pass

    if r["success"]:
        result = cv2.imread(r["output_path"])
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        total_fb = rlhf._total_feedback_count() if rlhf else 0
        info = (
            f"**Identity: {r['identity_score']*100:.1f}%** | "
            f"Multiplier: {r['multiplier']}x | "
            f"Time: {r['time']}s\n\n"
            f"RLHF: {total_fb} feedbacks collected"
        )
        return result_rgb, info
    else:
        return None, f"**{r['message']}**\n\nTime: {r.get('time', 0)}s"


def feedback(liked):
    req_id = last.get("request_id")
    if not req_id:
        return "Generate a smile first"

    if rlhf:
        result = rlhf.record_feedback(req_id, liked=liked)
        total = result.get("total_feedbacks", 0)
        changes = result.get("changes", {})

        msg = f"{'👍 LIKED' if liked else '👎 DISLIKED'}\n"
        msg += f"Total feedbacks: {total}\n"

        if changes.get("action"):
            msg += f"RLHF: {changes.get('reason', '')}\n"

        if total < 100:
            msg += f"\n{100 - total} more feedbacks → reward model trains"
        else:
            msg += "\nReward model active!"

        return msg
    else:
        fb_dir = os.path.join(PHOTO_DIR, "feedback")
        os.makedirs(fb_dir, exist_ok=True)
        with open(os.path.join(fb_dir, "mvp_feedback.jsonl"), "a") as f:
            f.write(json.dumps({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "liked": liked,
                "identity_score": last.get("score", 0),
                "multiplier": last.get("multiplier", 0),
            }) + "\n")
        return f"{'👍 LIKED' if liked else '👎 DISLIKED'} — saved"


def rlhf_status():
    if not rlhf:
        return "RLHF not loaded"
    s = rlhf.get_summary()
    lines = [
        f"**Total feedbacks:** {s['total_feedbacks']}",
        f"**Like rate:** {s['overall_like_rate']:.0%}",
        f"**Reward model:** {'Active' if s['reward_model_loaded'] else 'Need more feedbacks'}",
    ]
    if s.get("driver_stats"):
        lines.append("\n**Driver Performance:**")
        for key, stats in sorted(s["driver_stats"].items(),
                                  key=lambda x: x[1].get("total", 0), reverse=True)[:5]:
            t = stats.get("total", 0)
            l = stats.get("likes", 0)
            r = l/t if t > 0 else 0
            icon = "✅" if r >= 0.5 else "❌"
            lines.append(f"- {icon} `{key}`: {r:.0%} ({l}/{t})")
    return "\n".join(lines)


# UI
with gr.Blocks(title="Photogenic Smile Editor") as app:
    gr.Markdown("# Photogenic Smile Editor")
    gr.Markdown("Upload any photo → Get a natural smile. **Your feedback trains the system!**")

    with gr.Row():
        with gr.Column():
            photo = gr.Image(label="Upload Photo", type="numpy")
            btn = gr.Button("Apply Smile", variant="primary", size="lg")

        with gr.Column():
            result = gr.Image(label="Result")
            info = gr.Markdown("Result will appear here")

            with gr.Row():
                like = gr.Button("👍 Looks like me!", size="lg")
                dislike = gr.Button("👎 Doesn't look like me", variant="stop", size="lg")

            fb_text = gr.Textbox(label="Feedback", interactive=False, lines=3)

    with gr.Accordion("RLHF Dashboard", open=False):
        rlhf_md = gr.Markdown("Click refresh")
        gr.Button("Refresh").click(fn=rlhf_status, outputs=[rlhf_md])

    btn.click(fn=apply_smile, inputs=[photo], outputs=[result, info])
    like.click(fn=lambda: feedback(True), outputs=[fb_text])
    dislike.click(fn=lambda: feedback(False), outputs=[fb_text])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
