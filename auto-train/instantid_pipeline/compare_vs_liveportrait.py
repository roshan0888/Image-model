"""
Head-to-Head Comparison: InstantID vs LivePortrait
====================================================
Run on the same test faces, same expression targets.
Outputs side-by-side comparison grid + metrics CSV.

Usage:
    python instantid_pipeline/compare_vs_liveportrait.py --test-dir raw_data/starnow_ugc_models/verified/
"""

import os, sys, cv2, json, time, csv
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = Path(__file__).parent / "comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_comparison(test_dir: str, max_images: int = 10):
    """Run both engines on same images and compare."""

    test_images = sorted(Path(test_dir).glob("*.jpg"))[:max_images]
    if not test_images:
        test_images = sorted(Path(test_dir).glob("*.png"))[:max_images]
    if not test_images:
        print(f"No images found in {test_dir}")
        return

    print(f"Testing {len(test_images)} images from {test_dir}")

    # Load InsightFace for identity measurement
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(
        name="antelopev2",
        root=str(ROOT / "MagicFace/third_party_files"),
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Load LivePortrait
    print("\nLoading LivePortrait...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("npipe", ROOT / "natural_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lp_pipeline = mod.NaturalExpressionPipeline()

    # Load InstantID
    print("Loading InstantID...")
    from instantid_pipeline.instantid_engine import InstantIDEngine
    iid_engine = InstantIDEngine()

    # Expressions to test
    expressions = ["smile", "subtle_smile"]  # start simple

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, img_path in enumerate(test_images):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(test_images)}] {img_path.name}")
        print(f"{'='*60}")

        source_img = cv2.imread(str(img_path))
        source_faces = face_app.get(source_img)
        if not source_faces:
            print("  No face detected, skipping")
            continue
        source_emb = source_faces[0].normed_embedding

        for expr in expressions:
            row = {
                "image": img_path.name,
                "expression": expr,
                "lp_identity": 0.0,
                "lp_time": 0.0,
                "lp_status": "skip",
                "iid_identity": 0.0,
                "iid_time": 0.0,
                "iid_status": "skip",
            }

            # ── LivePortrait ─────────────────────────────────────────
            try:
                t0 = time.time()
                lp_result = lp_pipeline.process(str(img_path), expr if expr != "subtle_smile" else "smile")
                lp_time = time.time() - t0

                lp_img = lp_result.get("image")
                lp_score = 0.0
                if lp_img is not None:
                    if not isinstance(lp_img, np.ndarray):
                        lp_img = np.array(lp_img)
                    lp_faces = face_app.get(lp_img)
                    if lp_faces:
                        lp_score = float(np.dot(source_emb, lp_faces[0].normed_embedding))

                    # Save LP output
                    lp_out = str(OUTPUT_DIR / f"{img_path.stem}_LP_{expr}.jpg")
                    cv2.imwrite(lp_out, lp_img)

                row["lp_identity"] = round(lp_score, 4)
                row["lp_time"] = round(lp_time, 1)
                row["lp_status"] = "ok"
                print(f"  LP  {expr}: identity={lp_score:.4f}, time={lp_time:.1f}s")

            except Exception as e:
                row["lp_status"] = f"error: {str(e)[:50]}"
                print(f"  LP  {expr}: FAILED — {e}")

            # ── InstantID ────────────────────────────────────────────
            try:
                t0 = time.time()
                iid_result = iid_engine.generate(
                    str(img_path),
                    expression=expr,
                    pose="frontal",
                    background="studio_gray",
                    num_steps=25,
                    ip_adapter_scale=0.8,
                    controlnet_scale=0.8,
                )
                iid_time = time.time() - t0

                iid_score = iid_result.get("identity_score", 0)

                # Save IID output
                iid_out = str(OUTPUT_DIR / f"{img_path.stem}_IID_{expr}.jpg")
                iid_result["image"].save(iid_out, quality=95)

                row["iid_identity"] = round(iid_score, 4)
                row["iid_time"] = round(iid_time, 1)
                row["iid_status"] = "ok"
                print(f"  IID {expr}: identity={iid_score:.4f}, time={iid_time:.1f}s")

            except Exception as e:
                row["iid_status"] = f"error: {str(e)[:50]}"
                print(f"  IID {expr}: FAILED — {e}")

            results.append(row)

    # Save results CSV
    csv_path = OUTPUT_DIR / f"comparison_{timestamp}.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")

    lp_scores = [r["lp_identity"] for r in results if r["lp_status"] == "ok" and r["lp_identity"] > 0]
    iid_scores = [r["iid_identity"] for r in results if r["iid_status"] == "ok" and r["iid_identity"] > 0]
    lp_times = [r["lp_time"] for r in results if r["lp_status"] == "ok"]
    iid_times = [r["iid_time"] for r in results if r["iid_status"] == "ok"]

    if lp_scores:
        print(f"  LivePortrait:")
        print(f"    Avg identity: {np.mean(lp_scores):.4f} ({np.mean(lp_scores)*100:.1f}%)")
        print(f"    Min identity: {min(lp_scores):.4f}")
        print(f"    Max identity: {max(lp_scores):.4f}")
        print(f"    Avg time:     {np.mean(lp_times):.1f}s")

    if iid_scores:
        print(f"\n  InstantID:")
        print(f"    Avg identity: {np.mean(iid_scores):.4f} ({np.mean(iid_scores)*100:.1f}%)")
        print(f"    Min identity: {min(iid_scores):.4f}")
        print(f"    Max identity: {max(iid_scores):.4f}")
        print(f"    Avg time:     {np.mean(iid_times):.1f}s")

    if lp_scores and iid_scores:
        lp_wins = sum(1 for r in results if r["lp_identity"] > r["iid_identity"])
        iid_wins = sum(1 for r in results if r["iid_identity"] > r["lp_identity"])
        print(f"\n  Head-to-head: LP wins {lp_wins}, IID wins {iid_wins}")

    print(f"\n  Results saved to: {csv_path}")
    print(f"  Images saved to:  {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Save JSON summary
    summary = {
        "timestamp": timestamp,
        "test_images": len(test_images),
        "lp_avg_identity": float(np.mean(lp_scores)) if lp_scores else 0,
        "iid_avg_identity": float(np.mean(iid_scores)) if iid_scores else 0,
        "lp_avg_time": float(np.mean(lp_times)) if lp_times else 0,
        "iid_avg_time": float(np.mean(iid_times)) if iid_times else 0,
    }
    with open(OUTPUT_DIR / f"summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", default=str(ROOT / "raw_data/starnow_ugc_models/verified"))
    p.add_argument("--max-images", type=int, default=5)
    args = p.parse_args()
    run_comparison(args.test_dir, args.max_images)
