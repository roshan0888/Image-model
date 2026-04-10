#!/usr/bin/env python3
"""
Photoshoot Smile Editor — with RLHF
Mouth-only seamless swap + identity restoration + feedback learning
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

print("Loading Hybrid Smile Engine + RLHF...")
# Try hybrid engine first (GFPGAN mouth enhancement), fallback to basic
try:
    from hybrid_engine import HybridSmileEngine
    engine = HybridSmileEngine()
    ENGINE_TYPE = "hybrid"
    print("Hybrid engine loaded (LP + GFPGAN mouth enhancement)")
except Exception as e:
    print(f"Hybrid engine failed ({e}), falling back to basic engine")
    from engine import PhotoshootEngine
    engine = PhotoshootEngine()
    ENGINE_TYPE = "basic"

# Load RLHF from production system
try:
    from rlhf import RLHFSystem
    rlhf = RLHFSystem()
    print("RLHF loaded!")
except Exception as e:
    print(f"RLHF not available: {e}")
    rlhf = None

print("Ready!")

last_result = {"request_id": None, "expression": None, "driver": None, "intensity": None}


def process_smile(input_image, smile_type):
    global last_result

    if input_image is None:
        return None, "Upload a photo first"

    try:
        temp_in = os.path.join(PHOTO_DIR, "temp", "input.jpg")
        os.makedirs(os.path.dirname(temp_in), exist_ok=True)
        img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_in, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])

        # Score photo (hybrid engine may not have this method)
        try:
            score = engine.score_photo(img_bgr)
        except AttributeError:
            score = {"stars": 3, "message": "Photo received"}

        r = engine.smile(temp_in, smile_type=smile_type)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"ERROR in process_smile: {err}")
        return None, f"**Error:** {str(e)}\n\nPlease try a different photo."

    last_result = {
        "request_id": r.get("request_id"),
        "expression": smile_type,
        "driver": r.get("driver_used"),
        "intensity": r.get("multiplier", 1.0),
        "identity_score": r.get("identity_score", 0),
    }

    # Record attempt in RLHF
    if rlhf and r.get("request_id"):
        try:
            rlhf.record_attempt(
                request_id=r["request_id"],
                expression=smile_type,
                driver_used=r.get("driver_used", "unknown"),
                intensity_used=r.get("multiplier", 1.0),
                identity_score=r.get("identity_score", 0),
                expression_change=0.0,
            )
        except Exception:
            pass

    if r["success"]:
        result_bgr = cv2.imread(r["output_path"])
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        stars = "★" * score["stars"] + "☆" * (5 - score["stars"])
        rlhf_tag = ""
        if rlhf:
            total = rlhf._total_feedback_count()
            rlhf_tag = f"\n\nRLHF: {total} feedbacks collected"
            if total >= 30:
                rlhf_tag += " | Reward model active"

        info = (
            f"**Identity: {r['identity_score']*100:.1f}%** — {r['quality']}\n\n"
            f"Photo: {stars} {score['message']}\n\n"
            f"Driver: {r.get('driver_used', '—')} | "
            f"Attempts: {r['attempts']} | "
            f"Time: {r['time']:.1f}s"
            f"{rlhf_tag}"
        )
        return result_rgb, info
    else:
        stars = "★" * score["stars"] + "☆" * (5 - score["stars"])
        info = (
            f"**Could not apply {smile_type} smile**\n\n"
            f"Best identity: {r.get('identity_score', 0)*100:.1f}% (need 90%+)\n\n"
            f"Photo: {stars} {score['message']}\n\n"
            f"Tip: Try 'Subtle' smile or a front-facing photo"
        )
        return None, info


def send_feedback(liked):
    global last_result

    request_id = last_result.get("request_id")
    if request_id is None:
        return "Generate a smile first, then give feedback."

    if rlhf:
        result = rlhf.record_feedback(request_id, liked=liked)
        total = result.get("total_feedbacks", 0)
        changes = result.get("changes", {})

        msg = f"{'👍 LIKED' if liked else '👎 DISLIKED'} — Recorded!\n"
        msg += f"Total feedbacks: {total}\n"

        if changes.get("action") == "demote":
            msg += f"RLHF: {changes['reason']}\n"
        elif changes.get("action") == "promote":
            msg += f"RLHF: {changes['reason']}\n"

        if total < 30:
            msg += f"{30 - total} more feedbacks until reward model trains"
        else:
            msg += "Reward model active!"

        return msg
    else:
        # Save to simple file
        fb_dir = os.path.join(PHOTO_DIR, "feedback")
        os.makedirs(fb_dir, exist_ok=True)
        with open(os.path.join(fb_dir, "feedback.jsonl"), "a") as f:
            f.write(json.dumps({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "expression": last_result.get("expression"),
                "liked": liked,
                "identity_score": last_result.get("identity_score", 0),
                "driver": last_result.get("driver"),
            }) + "\n")
        return f"{'👍 LIKED' if liked else '👎 DISLIKED'} — Saved"


def get_rlhf_status():
    if not rlhf:
        return "RLHF not loaded"
    summary = rlhf.get_summary()
    lines = [f"**Total feedbacks:** {summary['total_feedbacks']}"]
    lines.append(f"**Like rate:** {summary['overall_like_rate']:.0%}")
    lines.append(f"**Reward model:** {'Active' if summary['reward_model_loaded'] else 'Need 30+ feedbacks'}")
    if summary.get('driver_stats'):
        lines.append("\n**Driver Performance:**")
        for key, stats in sorted(summary['driver_stats'].items(), key=lambda x: x[1].get('total', 0), reverse=True)[:8]:
            t = stats.get('total', 0)
            l = stats.get('likes', 0)
            r = l / t if t > 0 else 0
            icon = "✅" if r >= 0.5 else "❌"
            lines.append(f"- {icon} `{key}`: {r:.0%} ({l}/{t})")
    return "\n".join(lines)


with gr.Blocks(title="Photoshoot Smile Editor") as app:

    gr.Markdown("# 📸 Photoshoot Smile Editor")
    gr.Markdown("Upload any photo → Pick your smile → Get photoshoot quality. **Like/Dislike trains the system!**")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Photo", type="numpy")

            smile_type = gr.Radio(
                choices=["subtle", "photoshoot", "natural"],
                value="photoshoot",
                label="Smile Type",
                info="Subtle = confidence | Photoshoot = teeth showing | Natural = genuine"
            )

            run_btn = gr.Button("✨ Apply Smile", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(label="Result")
            info_text = gr.Markdown("Result will appear here")

            with gr.Row():
                like_btn = gr.Button("👍 Looks like me!", size="lg")
                dislike_btn = gr.Button("👎 Doesn't look like me", variant="stop", size="lg")

            feedback_text = gr.Textbox(label="Feedback", interactive=False, lines=3)

    with gr.Accordion("RLHF Dashboard", open=False):
        rlhf_md = gr.Markdown("Click refresh")
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(fn=get_rlhf_status, outputs=[rlhf_md])

    run_btn.click(fn=process_smile, inputs=[input_image, smile_type], outputs=[output_image, info_text])
    like_btn.click(fn=lambda: send_feedback(True), outputs=[feedback_text])
    dislike_btn.click(fn=lambda: send_feedback(False), outputs=[feedback_text])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
