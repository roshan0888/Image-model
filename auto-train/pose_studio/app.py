#!/usr/bin/env python3
"""
Pose Studio — Gradio App

Photoshoot-style editing with pose + expression control.
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr

POSE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(POSE_DIR, "engine"))

print("Loading Pose Studio Engine...")
from pose_engine import PoseExpressionEngine
engine = PoseExpressionEngine()
print("Ready!")


def generate(input_image, expression, pose, expr_intensity, pose_intensity):
    if input_image is None:
        return None, "Upload a photo first"

    temp = os.path.join(POSE_DIR, "temp", "input.jpg")
    os.makedirs(os.path.dirname(temp), exist_ok=True)
    img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp, img, [cv2.IMWRITE_JPEG_QUALITY, 98])

    # Map UI values
    expr_map = {"None": None, "Subtle Smile": "subtle", "Photoshoot Smile": "photoshoot", "Natural Smile": "natural"}
    pose_map = {"None": None, "Slight Left": "slight_left", "Slight Right": "slight_right",
                "Tilt Left": "tilt_left", "Tilt Right": "tilt_right",
                "Look Up": "look_up", "Look Down": "look_down", "3/4 View": "three_quarter"}

    r = engine.edit(
        temp,
        pose=pose_map.get(pose),
        expression=expr_map.get(expression),
        pose_multiplier=pose_intensity / 100.0,
        expr_multiplier=expr_intensity / 100.0 * 2.0,
    )

    if r["success"]:
        result = cv2.imread(r["output_path"])
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        info = (
            f"**Identity: {r['identity_score']*100:.1f}%** | "
            f"Steps: {', '.join(r['steps']) if r['steps'] else 'none'} | "
            f"Time: {r['time']}s"
        )
        return result_rgb, info
    else:
        return None, f"**{r.get('message', 'Failed')}** | Identity: {r.get('identity_score', 0)*100:.1f}%"


with gr.Blocks(title="Pose Studio") as app:
    gr.Markdown("# Pose Studio — Photoshoot Expression & Pose Editor")
    gr.Markdown("Upload a photo → choose expression + pose → get photoshoot-quality result")

    with gr.Row():
        with gr.Column():
            photo = gr.Image(label="Upload Photo", type="numpy")

            expression = gr.Radio(
                choices=["None", "Subtle Smile", "Photoshoot Smile", "Natural Smile"],
                value="Photoshoot Smile",
                label="Expression"
            )

            pose = gr.Radio(
                choices=["None", "Slight Left", "Slight Right", "Tilt Left", "Tilt Right", "Look Up", "Look Down", "3/4 View"],
                value="None",
                label="Pose"
            )

            with gr.Row():
                expr_slider = gr.Slider(30, 100, value=90, step=10, label="Expression Intensity %")
                pose_slider = gr.Slider(10, 100, value=50, step=10, label="Pose Intensity %")

            btn = gr.Button("Generate Photoshoot", variant="primary", size="lg")

        with gr.Column():
            result = gr.Image(label="Result")
            info = gr.Markdown("Result will appear here")

    btn.click(fn=generate, inputs=[photo, expression, pose, expr_slider, pose_slider], outputs=[result, info])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
