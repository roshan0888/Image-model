#!/usr/bin/env python3
"""Generate output images using trained LoRA weights to compare before/after."""

import os, sys, cv2, json, numpy as np, torch

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ENGINE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, LP_DIR)

OUTPUT_DIR = os.path.join(ENGINE_DIR, "output_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    import yaml
    from sklearn.metrics.pairwise import cosine_similarity

    config_path = os.path.join(ENGINE_DIR, "configs", "pipeline_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load InsightFace
    from insightface.app import FaceAnalysis
    face_analyzer = FaceAnalysis(
        name="antelopev2",
        root=config["paths"]["antelopev2_dir"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    # Load LP pipeline (standard — no LoRA, for baseline)
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    from src.config.argument_config import ArgumentConfig

    lp = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())

    # Test image
    source_path = os.path.join(BASE_DIR, "MagicFace", "test_images", "ros1.jpg")
    source_bgr = cv2.imread(source_path)

    # Get source embedding
    faces = face_analyzer.get(source_bgr)
    src_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    src_emb = src_face.normed_embedding.reshape(1, -1)

    driving_dir = os.path.join(LP_DIR, "assets", "examples", "driving")
    expressions = {
        "smile": "d30.jpg",
        "big_smile": "d12.jpg",
        "surprise": "d19.jpg",
        "angry": "d38.jpg",
        "sad": "d8.jpg",
    }

    print("=" * 70)
    print("GENERATING OUTPUT IMAGES — BASELINE (no LoRA)")
    print("=" * 70)

    results_baseline = {}

    for expr_name, driving_file in expressions.items():
        driving_path = os.path.join(driving_dir, driving_file)
        if not os.path.exists(driving_path):
            continue

        out_dir = os.path.join(ENGINE_DIR, "temp_gen")
        os.makedirs(out_dir, exist_ok=True)

        args = ArgumentConfig()
        args.source = source_path
        args.driving = driving_path
        args.output_dir = out_dir
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "all"
        args.driving_option = "expression-friendly"
        args.driving_multiplier = 1.0
        args.source_max_dim = 1920
        args.flag_eye_retargeting = True
        args.flag_lip_retargeting = True

        inf_keys = {
            'flag_pasteback', 'flag_do_crop', 'flag_stitching',
            'flag_relative_motion', 'animation_region', 'driving_option',
            'driving_multiplier', 'source_max_dim', 'source_division',
            'flag_eye_retargeting', 'flag_lip_retargeting',
        }
        lp.live_portrait_wrapper.update_config(
            {k: v for k, v in args.__dict__.items() if k in inf_keys}
        )

        wfp, _ = lp.execute(args)
        result = cv2.imread(wfp)

        # Compute identity
        res_faces = face_analyzer.get(result)
        if res_faces:
            res_face = max(res_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            score = float(cosine_similarity(src_emb, res_face.normed_embedding.reshape(1,-1))[0][0])
        else:
            score = 0.0

        # Save baseline
        out_path = os.path.join(OUTPUT_DIR, f"baseline_{expr_name}.png")
        cv2.imwrite(out_path, result)
        results_baseline[expr_name] = {"score": score, "path": out_path}
        print(f"  {expr_name}: identity={score:.4f} → {out_path}")

    # Now load LoRA weights and run again
    print("\n" + "=" * 70)
    print("GENERATING OUTPUT IMAGES — WITH LORA (trained)")
    print("=" * 70)

    # Load LoRA weights into the LP wrapper
    from training.lora_modules import inject_lora, load_lora_weights

    lora_cfg = config["training"]["lora"]
    rank = lora_cfg["rank"]
    alpha = lora_cfg["alpha"]
    dropout = lora_cfg["dropout"]
    ckpt_dir = config["paths"]["checkpoints_dir"]

    # Check if checkpoints exist
    spade_ckpt = os.path.join(ckpt_dir, "lora_spade_best.pt")
    if not os.path.exists(spade_ckpt):
        print("  No LoRA checkpoints found! Train first.")
        return

    # Inject LoRA into SPADE generator
    targets = lora_cfg["target_modules"].get("spade_generator", [])
    if targets:
        lp.live_portrait_wrapper.spade_generator, _ = inject_lora(
            lp.live_portrait_wrapper.spade_generator, targets, rank, alpha, dropout
        )
        load_lora_weights(lp.live_portrait_wrapper.spade_generator, spade_ckpt)
        lp.live_portrait_wrapper.spade_generator.to("cuda")

    # Inject LoRA into warping
    targets = lora_cfg["target_modules"].get("warping_network", [])
    warp_ckpt = os.path.join(ckpt_dir, "lora_warping_best.pt")
    if targets and os.path.exists(warp_ckpt):
        lp.live_portrait_wrapper.warping_module, _ = inject_lora(
            lp.live_portrait_wrapper.warping_module, targets, rank, alpha, dropout
        )
        load_lora_weights(lp.live_portrait_wrapper.warping_module, warp_ckpt)
        lp.live_portrait_wrapper.warping_module.to("cuda")

    # Inject LoRA into motion extractor
    targets = lora_cfg["target_modules"].get("motion_extractor", [])
    motion_ckpt = os.path.join(ckpt_dir, "lora_motion_best.pt")
    if targets and os.path.exists(motion_ckpt):
        lp.live_portrait_wrapper.motion_extractor, _ = inject_lora(
            lp.live_portrait_wrapper.motion_extractor, targets, rank, alpha, dropout
        )
        load_lora_weights(lp.live_portrait_wrapper.motion_extractor, motion_ckpt)
        lp.live_portrait_wrapper.motion_extractor.to("cuda")

    results_lora = {}

    for expr_name, driving_file in expressions.items():
        driving_path = os.path.join(driving_dir, driving_file)
        if not os.path.exists(driving_path):
            continue

        args = ArgumentConfig()
        args.source = source_path
        args.driving = driving_path
        args.output_dir = out_dir
        args.flag_pasteback = True
        args.flag_do_crop = True
        args.flag_stitching = True
        args.flag_relative_motion = True
        args.animation_region = "all"
        args.driving_option = "expression-friendly"
        args.driving_multiplier = 1.0
        args.source_max_dim = 1920
        args.flag_eye_retargeting = True
        args.flag_lip_retargeting = True

        lp.live_portrait_wrapper.update_config(
            {k: v for k, v in args.__dict__.items() if k in inf_keys}
        )

        wfp, _ = lp.execute(args)
        result = cv2.imread(wfp)

        # Compute identity
        res_faces = face_analyzer.get(result)
        if res_faces:
            res_face = max(res_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            score = float(cosine_similarity(src_emb, res_face.normed_embedding.reshape(1,-1))[0][0])
        else:
            score = 0.0

        out_path = os.path.join(OUTPUT_DIR, f"lora_{expr_name}.png")
        cv2.imwrite(out_path, result)
        results_lora[expr_name] = {"score": score, "path": out_path}
        print(f"  {expr_name}: identity={score:.4f} → {out_path}")

    # Create comparison grid
    print("\n" + "=" * 70)
    print("COMPARISON: BASELINE vs LORA")
    print("=" * 70)
    print(f"  {'Expression':<14} {'Baseline':>10} {'LoRA':>10} {'Improvement':>12}")
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*12}")

    for expr in expressions:
        b = results_baseline.get(expr, {}).get("score", 0)
        l = results_lora.get(expr, {}).get("score", 0)
        diff = l - b
        print(f"  {expr:<14} {b:>10.4f} {l:>10.4f} {diff:>+11.4f}")

    # Build visual comparison strip
    source_resized = cv2.resize(source_bgr, (512, 512))
    panels = [source_resized]
    labels = ["Source"]

    for expr in expressions:
        bp = results_baseline.get(expr, {}).get("path")
        lp_path = results_lora.get(expr, {}).get("path")
        if bp and os.path.exists(bp):
            img = cv2.imread(bp)
            img = cv2.resize(img, (512, 512))
            b_score = results_baseline[expr]["score"]
            cv2.putText(img, f"Base {b_score:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            panels.append(img)
            labels.append(f"Base-{expr}")
        if lp_path and os.path.exists(lp_path):
            img = cv2.imread(lp_path)
            img = cv2.resize(img, (512, 512))
            l_score = results_lora[expr]["score"]
            cv2.putText(img, f"LoRA {l_score:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            panels.append(img)
            labels.append(f"LoRA-{expr}")

    comparison = np.hstack(panels)
    comp_path = os.path.join(OUTPUT_DIR, "comparison_baseline_vs_lora.png")
    cv2.imwrite(comp_path, comparison)
    print(f"\n  Comparison image: {comp_path}")

    # Cleanup
    import shutil
    temp_dir = os.path.join(ENGINE_DIR, "temp_gen")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"\n  All outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
