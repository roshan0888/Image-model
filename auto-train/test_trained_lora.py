"""
Test the trained LoRA on fresh face images.
Compares LP-baseline vs LP+LoRA side by side.
"""
import os, sys, cv2, time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "LivePortrait"))

OUT_DIR = ROOT / "instantid_pipeline" / "outputs" / "trained_lora_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR = ROOT / "training_engine" / "checkpoints"
DRIVER   = "/teamspace/studios/this_studio/LivePortrait/assets/examples/driving/d12.jpg"

LORA_CFG = {
    "rank": 8, "alpha": 16, "dropout": 0.05,
    "target_modules": {
        "warping_network":       ["to_q", "to_k", "to_v", "to_out.0"],
        "spade_generator":       ["conv_0", "conv_1", "conv_s"],
        "motion_extractor":      [],
        "stitching_retargeting": [],
    },
}


def measure_id(face_app, src_img, out_img):
    f1 = face_app.get(src_img); f2 = face_app.get(out_img)
    if not f1 or not f2: return 0.0
    return float(np.dot(f1[0].normed_embedding, f2[0].normed_embedding))


def run_lp(pipeline, src_path, multiplier=1.3):
    return pipeline._run_lp(
        src_path, DRIVER,
        multiplier=multiplier, region="lip", use_retargeting=False,
    )


def main():
    import importlib.util
    from insightface.app import FaceAnalysis

    print("[1/4] Loading InsightFace...")
    face_app = FaceAnalysis(name="antelopev2", root="MagicFace/third_party_files")
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("[2/4] Loading LP baseline pipeline...")
    spec = importlib.util.spec_from_file_location("npipe", "natural_pipeline.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    lp_baseline = mod.NaturalExpressionPipeline()

    # Run baseline on test images
    # Pick fresh images we haven't tested before (skip 1-5)
    all_images = sorted(Path("raw_data/starnow_ugc_models/verified").glob("*.jpg"))
    test_images = all_images[5:13]  # next 8 images (000007 → 000016)
    print(f"[3/4] Testing {len(test_images)} images on BASELINE...")

    baseline_results = []
    for img_path in test_images:
        src_cv = cv2.imread(str(img_path))
        out = run_lp(lp_baseline, str(img_path))
        if out is None:
            baseline_results.append(None); continue
        score = measure_id(face_app, src_cv, out)
        out_path = OUT_DIR / f"{img_path.stem}_BASELINE.jpg"
        cv2.imwrite(str(out_path), out)
        baseline_results.append({"path": out_path, "score": score, "src": img_path})
        print(f"  {img_path.name} → BASELINE identity={score*100:.2f}%")

    # Now load LoRA into a fresh pipeline
    print("\n[4/4] Loading LP with TRAINED LoRA...")
    del lp_baseline
    import gc, torch
    torch.cuda.empty_cache(); gc.collect()

    spec2 = importlib.util.spec_from_file_location("npipe2", "natural_pipeline.py")
    mod2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(mod2)
    lp_lora = mod2.NaturalExpressionPipeline()

    from training_engine.training.lora_modules import inject_lora, load_lora_weights
    wrapper = lp_lora.lp_pipeline.live_portrait_wrapper

    # Inject + load SPADE LoRA
    spade_ckpt = CKPT_DIR / "lora_spade_best.pt"
    if spade_ckpt.exists():
        wrapper.spade_generator, _ = inject_lora(
            wrapper.spade_generator,
            LORA_CFG["target_modules"]["spade_generator"],
            LORA_CFG["rank"], LORA_CFG["alpha"], LORA_CFG["dropout"],
        )
        load_lora_weights(wrapper.spade_generator, str(spade_ckpt))
        wrapper.spade_generator.to("cuda")
        print(f"  ✓ Loaded SPADE LoRA from {spade_ckpt.name}")

    # Inject + load Warping LoRA
    warp_ckpt = CKPT_DIR / "lora_warping_best.pt"
    if warp_ckpt.exists():
        wrapper.warping_module, _ = inject_lora(
            wrapper.warping_module,
            LORA_CFG["target_modules"]["warping_network"],
            LORA_CFG["rank"], LORA_CFG["alpha"], LORA_CFG["dropout"],
        )
        load_lora_weights(wrapper.warping_module, str(warp_ckpt))
        wrapper.warping_module.to("cuda")
        print(f"  ✓ Loaded Warping LoRA from {warp_ckpt.name}")

    # Run LoRA on same images
    lora_results = []
    print(f"\nTesting on TRAINED LoRA model...")
    for img_path in test_images:
        src_cv = cv2.imread(str(img_path))
        out = run_lp(lp_lora, str(img_path))
        if out is None:
            lora_results.append(None); continue
        score = measure_id(face_app, src_cv, out)
        out_path = OUT_DIR / f"{img_path.stem}_LORA.jpg"
        cv2.imwrite(str(out_path), out)
        lora_results.append({"path": out_path, "score": score, "src": img_path})
        print(f"  {img_path.name} → LORA identity={score*100:.2f}%")

    # Build comparison grid
    print("\nBuilding comparison grid...")
    h = 600; gap = 12
    rows = []
    for i, (b, l) in enumerate(zip(baseline_results, lora_results)):
        if b is None or l is None: continue
        src_img = Image.open(b["src"]).convert("RGB")
        b_img = Image.open(b["path"]).convert("RGB")
        l_img = Image.open(l["path"]).convert("RGB")
        sw = int(src_img.width * h / src_img.height)
        bw = int(b_img.width * h / b_img.height)
        lw = int(l_img.width * h / l_img.height)
        src_r = src_img.resize((sw, h))
        b_r = b_img.resize((bw, h))
        l_r = l_img.resize((lw, h))
        row = Image.new("RGB", (sw + bw + lw + gap*2, h + 50), (255, 255, 255))
        row.paste(src_r, (0, 50))
        row.paste(b_r, (sw + gap, 50))
        row.paste(l_r, (sw + gap + bw + gap, 50))
        d = ImageDraw.Draw(row)
        d.text((sw//2 - 30, 15), "SOURCE", fill=(0, 0, 0))
        d.text((sw + gap + bw//2 - 70, 15), f"LP BASELINE ({b['score']*100:.1f}%)", fill=(0, 0, 0))
        d.text((sw + gap + bw + gap + lw//2 - 70, 15), f"LP + LoRA ({l['score']*100:.1f}%)", fill=(0, 0, 0))
        rows.append(row)

    if rows:
        max_w = max(r.width for r in rows)
        total_h = sum(r.height for r in rows) + gap*(len(rows)-1)
        final = Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y = 0
        for r in rows:
            final.paste(r, ((max_w - r.width)//2, y))
            y += r.height + gap
        comp_path = OUT_DIR / "BASELINE_vs_LORA_COMPARISON.jpg"
        final.save(comp_path, quality=92)
        print(f"\nSaved comparison: {comp_path}")

    # Print summary
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(f"  {'Image':<25} {'Baseline':<12} {'+LoRA':<12} {'Δ':<8}")
    deltas = []
    for b, l in zip(baseline_results, lora_results):
        if b is None or l is None: continue
        delta = (l["score"] - b["score"]) * 100
        deltas.append(delta)
        sign = "+" if delta >= 0 else ""
        print(f"  {b['src'].name:<25} {b['score']*100:>6.2f}%      {l['score']*100:>6.2f}%      {sign}{delta:+.2f}")
    if deltas:
        print(f"  {'AVG':<25} {'':<12} {'':<12} {'+' if sum(deltas)/len(deltas)>=0 else ''}{sum(deltas)/len(deltas):+.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
