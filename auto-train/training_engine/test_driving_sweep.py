#!/usr/bin/env python3
"""
Driving Image Sweep — Find the BEST driving template for each expression.

Problem: Default LP driving images have bad poses, occlusions, extreme angles.
         These work for LP's test image but FAIL on external faces.

Solution: Use our scraped clean faces as driving templates.
          Test ALL of them and find which gives highest identity on unknown faces.
"""

import os, sys, cv2, numpy as np, json

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ENGINE_DIR)
LP_DIR = os.path.join(os.path.dirname(BASE_DIR), "LivePortrait")
sys.path.insert(0, LP_DIR)

OUTPUT_DIR = os.path.join(ENGINE_DIR, "output_driving_sweep")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_lp_single(lp, source_path, driving_path, temp_dir):
    """Run LP once and return result image."""
    from src.config.argument_config import ArgumentConfig

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
    return cv2.imread(wfp)


def main():
    from sklearn.metrics.pairwise import cosine_similarity
    from insightface.app import FaceAnalysis
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline

    face_analyzer = FaceAnalysis(
        name="antelopev2",
        root=os.path.join(BASE_DIR, "MagicFace", "third_party_files"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    lp = LivePortraitPipeline(inference_cfg=InferenceConfig(), crop_cfg=CropConfig())

    temp_dir = os.path.join(ENGINE_DIR, "temp_sweep")
    os.makedirs(temp_dir, exist_ok=True)

    # Source faces (unknown/external)
    neutral_dir = os.path.join(ENGINE_DIR, "dataset", "cleaned_scraped", "neutral")
    source_faces = []
    for f in sorted(os.listdir(neutral_dir)):
        if f.endswith('.jpg') and f.startswith('clean_'):
            source_faces.append(os.path.join(neutral_dir, f))
    source_faces = source_faces[:3]  # Test on 3 faces for speed

    # Candidate driving images — SCRAPED faces per expression
    cleaned_dir = os.path.join(ENGINE_DIR, "dataset", "cleaned_scraped")
    candidate_drivers = {}  # expr → [paths]

    for expr in ["smile", "surprise", "sad"]:
        expr_dir = os.path.join(cleaned_dir, expr)
        if not os.path.exists(expr_dir):
            continue
        drivers = []
        for f in sorted(os.listdir(expr_dir)):
            if f.endswith('.jpg') and f.startswith('clean_'):
                drivers.append(os.path.join(expr_dir, f))
        candidate_drivers[expr] = drivers[:8]  # Test top 8 per expression

    # Also include original LP driving images for comparison
    driving_dir = os.path.join(LP_DIR, "assets", "examples", "driving")
    original_drivers = {
        "smile": os.path.join(driving_dir, "d30.jpg"),
        "surprise": os.path.join(driving_dir, "d19.jpg"),
        "sad": os.path.join(driving_dir, "d8.jpg"),
    }

    print("=" * 70)
    print("DRIVING IMAGE SWEEP")
    print(f"  Source faces: {len(source_faces)}")
    for expr, drivers in candidate_drivers.items():
        print(f"  {expr} driving candidates: {len(drivers)} scraped + 1 original")
    print("=" * 70)

    # Get source embeddings
    source_data = []
    for sp in source_faces:
        img = cv2.imread(sp)
        faces = face_analyzer.get(img)
        if faces:
            sf = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            source_data.append({
                "path": sp,
                "image": img,
                "embedding": sf.normed_embedding.reshape(1, -1),
                "name": os.path.basename(sp),
            })

    # Sweep: for each expression, try each driving image on all source faces
    results = {}  # expr → [(driver_path, avg_score, scores)]

    for expr in ["smile", "surprise", "sad"]:
        print(f"\n{'='*60}")
        print(f"EXPRESSION: {expr}")
        print(f"{'='*60}")

        expr_results = []

        # Test original LP driver first
        orig_driver = original_drivers[expr]
        orig_scores = []
        for sd in source_data:
            try:
                result = run_lp_single(lp, sd["path"], orig_driver, temp_dir)
                if result is None:
                    continue
                rf = face_analyzer.get(result)
                if rf:
                    rbest = max(rf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    score = float(cosine_similarity(
                        sd["embedding"], rbest.normed_embedding.reshape(1,-1)
                    )[0][0])
                    orig_scores.append(score)
            except:
                pass

        avg_orig = np.mean(orig_scores) if orig_scores else 0
        expr_results.append({
            "driver": orig_driver,
            "name": f"ORIGINAL ({os.path.basename(orig_driver)})",
            "avg_score": avg_orig,
            "scores": orig_scores,
        })
        print(f"  ORIGINAL {os.path.basename(orig_driver)}: avg={avg_orig:.4f} {orig_scores}")

        # Test each scraped driving image
        for drv_path in candidate_drivers.get(expr, []):
            drv_scores = []
            for sd in source_data:
                try:
                    result = run_lp_single(lp, sd["path"], drv_path, temp_dir)
                    if result is None:
                        continue
                    rf = face_analyzer.get(result)
                    if rf:
                        rbest = max(rf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                        score = float(cosine_similarity(
                            sd["embedding"], rbest.normed_embedding.reshape(1,-1)
                        )[0][0])
                        drv_scores.append(score)
                except:
                    pass

            avg_score = np.mean(drv_scores) if drv_scores else 0
            expr_results.append({
                "driver": drv_path,
                "name": os.path.basename(drv_path),
                "avg_score": avg_score,
                "scores": drv_scores,
            })
            marker = " *** BETTER" if avg_score > avg_orig else ""
            print(f"  {os.path.basename(drv_path)}: avg={avg_score:.4f} "
                  f"[{', '.join(f'{s:.3f}' for s in drv_scores)}]{marker}")

        # Sort by avg score
        expr_results.sort(key=lambda x: x["avg_score"], reverse=True)
        results[expr] = expr_results

        print(f"\n  BEST for {expr}: {expr_results[0]['name']} "
              f"(avg={expr_results[0]['avg_score']:.4f})")

    # Generate final comparison with BEST driving images vs ORIGINAL
    print(f"\n{'='*70}")
    print("FINAL COMPARISON: ORIGINAL vs BEST DRIVERS")
    print(f"{'='*70}")

    best_drivers = {}
    for expr, expr_results in results.items():
        best = expr_results[0]
        orig = [r for r in expr_results if "ORIGINAL" in r["name"]][0]
        improvement = best["avg_score"] - orig["avg_score"]
        best_drivers[expr] = best["driver"]
        print(f"  {expr}:")
        print(f"    Original: {orig['name']} → {orig['avg_score']:.4f}")
        print(f"    Best:     {best['name']} → {best['avg_score']:.4f} ({improvement:+.4f})")

    # Save best driver map
    driver_map = {expr: path for expr, path in best_drivers.items()}
    with open(os.path.join(OUTPUT_DIR, "best_drivers.json"), "w") as f:
        json.dump(driver_map, f, indent=2)

    # Generate visual grid with best drivers
    print(f"\nGenerating comparison grid with best drivers...")

    cell_size = 300
    rows = []

    for sd in source_data:
        row_panels = []
        src_resized = cv2.resize(sd["image"], (cell_size, cell_size))
        cv2.putText(src_resized, "SOURCE", (5, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        row_panels.append(src_resized)

        for expr in ["smile", "surprise", "sad"]:
            # Original driver result
            try:
                result_orig = run_lp_single(lp, sd["path"], original_drivers[expr], temp_dir)
                rf = face_analyzer.get(result_orig)
                if rf:
                    rbest = max(rf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    score_orig = float(cosine_similarity(
                        sd["embedding"], rbest.normed_embedding.reshape(1,-1)
                    )[0][0])
                else:
                    score_orig = 0
                img = cv2.resize(result_orig, (cell_size, cell_size))
                cv2.putText(img, f"OLD {expr}", (5, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(img, f"id={score_orig:.3f}", (5, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                row_panels.append(img)
            except:
                row_panels.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

            # Best driver result
            try:
                result_best = run_lp_single(lp, sd["path"], best_drivers[expr], temp_dir)
                rf = face_analyzer.get(result_best)
                if rf:
                    rbest = max(rf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    score_best = float(cosine_similarity(
                        sd["embedding"], rbest.normed_embedding.reshape(1,-1)
                    )[0][0])
                else:
                    score_best = 0
                img = cv2.resize(result_best, (cell_size, cell_size))
                color = (0, 255, 0) if score_best >= 0.95 else (0, 255, 255) if score_best >= 0.90 else (0, 165, 255)
                cv2.putText(img, f"NEW {expr}", (5, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(img, f"id={score_best:.3f}", (5, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Save individual output
                out_name = f"{sd['name'].replace('.jpg','')}_{expr}_best.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), result_best)
                row_panels.append(img)
            except:
                row_panels.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

        rows.append(np.hstack(row_panels))

    if rows:
        grid = np.vstack(rows)
        grid_path = os.path.join(OUTPUT_DIR, "DRIVING_SWEEP_COMPARISON.png")
        cv2.imwrite(grid_path, grid)
        print(f"  Grid saved: {grid_path}")

    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\nAll outputs: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
