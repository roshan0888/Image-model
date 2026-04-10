#!/usr/bin/env python3
"""
Gradio Web UI for Face Expression Editor — with RLHF
Like/Dislike buttons now feed into the RLHF system which
adapts driver selection in real-time.
"""

import os
import sys
import cv2
import json
import time
import numpy as np
import gradio as gr

PROD_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROD_DIR)

# Load engine once
print("Loading Expression Engine + RLHF...")
from expression_engine import setup_drivers, ExpressionEngine
setup_drivers()
engine = ExpressionEngine()
print("Ready!")

# Track last result for feedback
last_result = {"request_id": None, "expression": None}


def edit_expression(input_image, expression, intensity):
    """Main function called by Gradio."""
    global last_result

    if input_image is None:
        return None, "Please upload a photo first"

    # Save input to temp file
    temp_in = os.path.join(PROD_DIR, "temp", "gradio_input.jpg")
    os.makedirs(os.path.dirname(temp_in), exist_ok=True)

    # Gradio gives us RGB numpy array
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_in, img_bgr)

    # Map slider to 0-1
    intensity_float = intensity / 100.0

    # Run engine (RLHF-aware)
    result = engine.edit(
        source_path=temp_in,
        expression=expression,
        intensity=intensity_float,
        output_name=f"gradio_{expression}.png",
    )

    # Store for feedback
    last_result = {
        "request_id": result.get("request_id"),
        "expression": expression,
    }

    if result["success"]:
        result_bgr = cv2.imread(result["output_path"])
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        rlhf_tag = " | **RLHF: Active**" if result.get("rlhf_active") else " | RLHF: Learning"

        info = (
            f"**Identity: {result['identity_score']*100:.1f}%** | "
            f"Expression change: {result['expression_change']:.4f} | "
            f"Driver: {result.get('driver_used', '—')} | "
            f"Intensity: {result['intensity_used']:.0%} | "
            f"Attempts: {result['attempts']} | "
            f"Time: {result['time']:.1f}s"
            f"{rlhf_tag}"
        )
        return result_rgb, info
    else:
        info = (
            f"**Quality gate blocked this result** | "
            f"Best identity: {result['identity_score']*100:.1f}% (need 90%+) | "
            f"Attempts: {result['attempts']} | "
            f"{result['message']}"
        )
        return None, info


def send_feedback(liked):
    """Send feedback through RLHF system — this LEARNS."""
    global last_result

    request_id = last_result.get("request_id")
    expression = last_result.get("expression", "unknown")

    if request_id is None:
        return "No result to give feedback on. Generate an expression first."

    # Record through RLHF system
    result = engine.rlhf.record_feedback(request_id, liked=liked)

    status = result.get("status", "unknown")
    total = result.get("total_feedbacks", 0)
    changes = result.get("changes", {})

    msg = f"{'👍 LIKED' if liked else '👎 DISLIKED'} — Recorded in RLHF system\n"
    msg += f"Total feedbacks: {total}\n"

    if changes.get("action") == "demote":
        msg += f"⚠️ RLHF ACTION: {changes['reason']}\n"
    elif changes.get("action") == "promote":
        msg += f"✅ RLHF ACTION: {changes['reason']}\n"

    if total >= 30 and total % 50 == 0:
        msg += f"🧠 Reward model retrained on {total} feedbacks!\n"

    if total < 30:
        msg += f"Need {30 - total} more feedbacks to train reward model"
    else:
        msg += "Reward model active — predictions improving"

    return msg


def get_rlhf_status():
    """Show RLHF system status."""
    summary = engine.rlhf.get_summary()

    lines = ["## RLHF System Status\n"]
    lines.append(f"**Total feedbacks:** {summary['total_feedbacks']}")
    lines.append(f"**Overall like rate:** {summary['overall_like_rate']:.0%}")
    lines.append(f"**Reward model:** {'✅ Trained' if summary['reward_model_loaded'] else '⏳ Need 30+ feedbacks'}\n")

    if summary['per_expression']:
        lines.append("### Per Expression")
        for expr, data in summary['per_expression'].items():
            rate = data['liked'] / data['total'] if data['total'] > 0 else 0
            bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
            lines.append(f"- **{expr}**: {bar} {rate:.0%} liked ({data['liked']}/{data['total']})")

    if summary['driver_stats']:
        lines.append("\n### Driver Performance (from feedback)")
        sorted_stats = sorted(
            summary['driver_stats'].items(),
            key=lambda x: x[1].get('total', 0),
            reverse=True
        )
        for key, stats in sorted_stats[:10]:
            total = stats.get('total', 0)
            likes = stats.get('likes', 0)
            rate = likes / total if total > 0 else 0
            status = "✅" if rate >= 0.5 else "⚠️" if rate >= 0.3 else "❌"
            lines.append(f"- {status} `{key}`: {rate:.0%} liked ({likes}/{total})")

    return "\n".join(lines)


# Build Gradio UI
with gr.Blocks(
    title="Face Expression Editor + RLHF",
    theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
) as app:

    gr.Markdown("# Face Expression Editor + RLHF")
    gr.Markdown(
        "Upload a photo, choose an expression. "
        "**Like/Dislike buttons train the system** — it learns which drivers work best!"
    )

    with gr.Row():
        # Left column — Input
        with gr.Column():
            input_image = gr.Image(label="Upload Photo", type="numpy")

            expression = gr.Radio(
                choices=["smile", "open_smile", "surprise", "sad", "angry"],
                value="smile",
                label="Expression",
            )

            intensity = gr.Slider(
                minimum=10, maximum=100, value=100, step=10,
                label="Intensity (%)",
            )

            run_btn = gr.Button("Generate Expression", variant="primary", size="lg")

        # Right column — Output
        with gr.Column():
            output_image = gr.Image(label="Result")
            info_text = gr.Markdown("Result will appear here")

            with gr.Row():
                like_btn = gr.Button("👍 Looks like me!", variant="secondary", size="lg")
                dislike_btn = gr.Button("👎 Doesn't look like me", variant="stop", size="lg")

            feedback_text = gr.Textbox(label="RLHF Feedback", interactive=False, lines=4)

    # RLHF Dashboard
    with gr.Accordion("📊 RLHF Dashboard", open=False):
        rlhf_status = gr.Markdown("Click refresh to see RLHF stats")
        refresh_btn = gr.Button("Refresh RLHF Stats")
        refresh_btn.click(fn=get_rlhf_status, outputs=[rlhf_status])

    # Connect events
    run_btn.click(
        fn=edit_expression,
        inputs=[input_image, expression, intensity],
        outputs=[output_image, info_text],
    )

    like_btn.click(
        fn=lambda: send_feedback(True),
        outputs=[feedback_text],
    )

    dislike_btn.click(
        fn=lambda: send_feedback(False),
        outputs=[feedback_text],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
