#!/usr/bin/env python3
"""Test expression transfer on unknown faces from scraped data."""

import os, sys, cv2, numpy as np

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ENGINE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
sys.path.insert(0, LP_DIR)

OUTPUT_DIR = os.path.join(ENGINE_DIR, "output_unknown_faces")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    from sklearn.metrics.pairwise import cosine_similarity
    from insightface.app import FaceAnalysis
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    from src.config.argument_config import ArgumentConfig

    # Load models
    face_analyzer = FaceAnalysis(
        name="antelopev2",
        root=os.path.join(BASE_DIR, "MagicFace", "third_party_files"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    lp = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())

    # Unknown faces from scraped neutral data
    neutral_dir = os.path.join(ENGINE_DIR, "dataset", "cleaned_scraped", "neutral")
    test_faces = []
    for f in sorted(os.listdir(neutral_dir)):
        if f.endswith('.jpg') and f.startswith('clean_'):
            test_faces.append(os.path.join(neutral_dir, f))

    # Pick 4 diverse faces
    test_faces = test_faces[:4]

    driving_dir = os.path.join(LP_DIR, "assets", "examples", "driving")
    expressions = {
        "smile": "d30.jpg",
        "surprise": "d19.jpg",
        "sad": "d8.jpg",
    }

    temp_dir = os.path.join(ENGINE_DIR, "temp_unknown")
    os.makedirs(temp_dir, exist_ok=True)

    all_results = []

    for face_idx, source_path in enumerate(test_faces):
        source_bgr = cv2.imread(source_path)
        if source_bgr is None:
            continue

        faces = face_analyzer.get(source_bgr)
        if not faces:
            print(f"  Face {face_idx}: no face detected, skipping")
            continue
        src_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        src_emb = src_face.normed_embedding.reshape(1, -1)

        face_name = f"face{face_idx}"
        print(f"\n{'='*60}")
        print(f"Testing: {os.path.basename(source_path)} ({face_name})")
        print(f"{'='*60}")

        face_results = {"source": source_path, "name": face_name}

        for expr_name, driving_file in expressions.items():
            driving_path = os.path.join(driving_dir, driving_file)

            try:
                args = ArgumentConfig()
                args.source = source_path
                args.driving = driving_path
                args.output_dir = temp_dir
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

                if result is None:
                    print(f"  {expr_name}: LP returned None")
                    continue

                # Compute identity
                res_faces = face_analyzer.get(result)
                if res_faces:
                    res_face = max(res_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    score = float(cosine_similarity(
                        src_emb, res_face.normed_embedding.reshape(1,-1)
                    )[0][0])
                else:
                    score = 0.0

                # Save individual output
                out_path = os.path.join(OUTPUT_DIR, f"{face_name}_{expr_name}.png")
                cv2.imwrite(out_path, result)
                face_results[expr_name] = {"score": score, "path": out_path, "image": result}
                print(f"  {expr_name}: identity={score:.4f}")

            except Exception as e:
                print(f"  {expr_name}: FAILED — {e}")

        all_results.append(face_results)

    # Build comparison grid: each row = one face, columns = source + expressions
    print(f"\n{'='*60}")
    print("Building comparison grid...")
    print(f"{'='*60}")

    cell_size = 300
    rows = []

    for face_results in all_results:
        row_panels = []

        # Source
        src_img = cv2.imread(face_results["source"])
        src_resized = cv2.resize(src_img, (cell_size, cell_size))
        cv2.putText(src_resized, "SOURCE", (5, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        row_panels.append(src_resized)

        # Expression results
        for expr_name in expressions:
            if expr_name in face_results and isinstance(face_results[expr_name], dict):
                img = face_results[expr_name]["image"]
                score = face_results[expr_name]["score"]
                img_resized = cv2.resize(img, (cell_size, cell_size))

                # Color: green if > 0.95, yellow if > 0.90, red otherwise
                if score >= 0.95:
                    color = (0, 255, 0)
                elif score >= 0.90:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                cv2.putText(img_resized, f"{expr_name} {score:.3f}", (5, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                row_panels.append(img_resized)
            else:
                # Empty cell
                empty = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                cv2.putText(empty, f"{expr_name} FAIL", (5, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                row_panels.append(empty)

        rows.append(np.hstack(row_panels))

    if rows:
        grid = np.vstack(rows)
        grid_path = os.path.join(OUTPUT_DIR, "UNKNOWN_FACES_GRID.png")
        cv2.imwrite(grid_path, grid)
        print(f"\nGrid saved: {grid_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("RESULTS — UNKNOWN FACES")
    print(f"{'='*60}")
    print(f"  {'Face':<12} {'Smile':>10} {'Surprise':>10} {'Sad':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    for fr in all_results:
        name = fr["name"]
        smile = fr.get("smile", {}).get("score", 0)
        surprise = fr.get("surprise", {}).get("score", 0)
        sad = fr.get("sad", {}).get("score", 0)
        print(f"  {name:<12} {smile:>10.4f} {surprise:>10.4f} {sad:>10.4f}")

    avg_smile = np.mean([fr.get("smile", {}).get("score", 0) for fr in all_results if "smile" in fr])
    avg_surprise = np.mean([fr.get("surprise", {}).get("score", 0) for fr in all_results if "surprise" in fr])
    avg_sad = np.mean([fr.get("sad", {}).get("score", 0) for fr in all_results if "sad" in fr])
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'AVERAGE':<12} {avg_smile:>10.4f} {avg_surprise:>10.4f} {avg_sad:>10.4f}")
    print(f"{'='*60}")
    print(f"\nAll outputs: {OUTPUT_DIR}/")

    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
