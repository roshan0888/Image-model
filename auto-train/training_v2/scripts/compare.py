#!/usr/bin/env python3
"""
Compare trained model vs current MVP.

Loads LoRA weights from training_v2/checkpoints/
Tests on 10 faces and shows side-by-side comparison.
Only deploy to production if trained version is BETTER.
"""

import os
import sys
import cv2
import json
import numpy as np

TRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(TRAIN_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
ENGINE_DIR = os.path.join(BASE_DIR, "training_engine")

sys.path.insert(0, LP_DIR)
sys.path.insert(0, ENGINE_DIR)


def compare():
    from sklearn.metrics.pairwise import cosine_similarity
    from insightface.app import FaceAnalysis
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    from src.config.argument_config import ArgumentConfig

    fa = FaceAnalysis(name='antelopev2',
        root=os.path.join(BASE_DIR, 'MagicFace/third_party_files'),
        providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    fa.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.3)

    DRIVER = os.path.join(LP_DIR, 'assets/examples/driving/d12.jpg')
    CKPT = os.path.join(TRAIN_DIR, 'checkpoints', 'lora_spade_v2_best.pt')

    # Test faces
    neutral_dir = os.path.join(BASE_DIR, 'training_engine/dataset/cleaned_scraped/neutral')
    test_faces = sorted([
        os.path.join(neutral_dir, f) for f in os.listdir(neutral_dir)
        if f.startswith('clean_') and f.endswith('.jpg')
    ])[:10]

    def run_lp(lp, source_path, mult=1.8):
        args = ArgumentConfig()
        args.source = source_path
        args.driving = DRIVER
        args.output_dir = os.path.join(TRAIN_DIR, 'temp')
        os.makedirs(args.output_dir, exist_ok=True)
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = 'lip'
        args.driving_option = 'expression-friendly'
        args.driving_multiplier = mult
        args.source_max_dim = 1920
        inf_keys = {'flag_pasteback','flag_do_crop','flag_stitching',
            'flag_relative_motion','animation_region','driving_option',
            'driving_multiplier','source_max_dim','source_division',
            'flag_eye_retargeting','flag_lip_retargeting'}
        lp.live_portrait_wrapper.update_config({k:v for k,v in args.__dict__.items() if k in inf_keys})
        wfp, _ = lp.execute(args)
        return cv2.imread(wfp)

    # Test 1: BASELINE (no LoRA)
    print("\nLoading BASELINE LP...")
    lp_base = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())

    # Test 2: TRAINED (with LoRA)
    has_lora = os.path.exists(CKPT)

    print(f"\n{'='*60}")
    print(f"COMPARISON: MVP (baseline) vs Trained (LoRA v2)")
    print(f"LoRA checkpoint: {'FOUND' if has_lora else 'NOT FOUND — run train.py first'}")
    print(f"{'='*60}\n")

    if not has_lora:
        print("No trained checkpoint found. Run train.py first.")
        return

    print(f"{'Face':<25} {'Baseline':>10} {'Trained':>10} {'Diff':>10}")
    print(f"{'─'*25} {'─'*10} {'─'*10} {'─'*10}")

    base_scores = []
    train_scores = []

    for face_path in test_faces:
        name = os.path.basename(face_path)[:22]
        src = cv2.imread(face_path)
        faces = fa.get(src)
        if not faces: continue
        src_emb = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding.reshape(1,-1)

        # Baseline
        result_base = run_lp(lp_base, face_path)
        rf = fa.get(result_base)
        b_score = float(cosine_similarity(src_emb, max(rf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding.reshape(1,-1))[0][0]) if rf else 0

        # Trained (load fresh LP with LoRA each time to avoid contamination)
        # For speed, we load once and inject LoRA
        if not hasattr(compare, '_lp_lora'):
            import torch.nn as nn
            from training.lora_modules import LoRAConv2d, _get_parent_attr, load_lora_weights

            compare._lp_lora = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())
            for n, m in list(compare._lp_lora.live_portrait_wrapper.spade_generator.named_modules()):
                if isinstance(m, nn.Conv2d) and "G_middle" in n and any(t in n for t in ["conv_0", "conv_1"]):
                    try:
                        parent, attr = _get_parent_attr(compare._lp_lora.live_portrait_wrapper.spade_generator, n)
                        setattr(parent, attr, LoRAConv2d(m, 4, 4, 0.0))
                    except: pass
            load_lora_weights(compare._lp_lora.live_portrait_wrapper.spade_generator, CKPT)
            compare._lp_lora.live_portrait_wrapper.spade_generator.to("cuda")

        result_train = run_lp(compare._lp_lora, face_path)
        rf2 = fa.get(result_train)
        t_score = float(cosine_similarity(src_emb, max(rf2, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])).normed_embedding.reshape(1,-1))[0][0]) if rf2 else 0

        diff = t_score - b_score
        arrow = "↑" if diff > 0 else "↓"
        print(f"  {name:<23} {b_score:>10.4f} {t_score:>10.4f} {arrow}{abs(diff):>9.4f}")

        base_scores.append(b_score)
        train_scores.append(t_score)

    avg_base = np.mean(base_scores)
    avg_train = np.mean(train_scores)
    avg_diff = avg_train - avg_base

    print(f"\n{'─'*55}")
    print(f"  {'AVERAGE':<23} {avg_base:>10.4f} {avg_train:>10.4f} {'↑' if avg_diff>0 else '↓'}{abs(avg_diff):>9.4f}")
    print(f"\n{'VERDICT: ' + ('TRAINED IS BETTER — deploy!' if avg_diff > 0.005 else 'NO IMPROVEMENT — keep baseline')}")


if __name__ == "__main__":
    compare()
