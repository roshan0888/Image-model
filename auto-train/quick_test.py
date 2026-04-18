"""
Quick test of trained LoRA on cleaned neutral images.
Runs LP + LoRA on 5 images and reports identity preservation.
"""
import os, sys, cv2, gc, torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "LivePortrait"))

OUT_DIR = ROOT / "output" / "quick_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR = ROOT / "training_engine" / "checkpoints"
DRIVER = "/teamspace/studios/this_studio/LivePortrait/assets/examples/driving/d30.jpg"

LORA_CFG = {
    "rank": 8, "alpha": 16, "dropout": 0.05,
    "spade_targets": ["fc", "G_middle_0", "G_middle_1", "G_middle_2",
                      "G_middle_3", "G_middle_4", "G_middle_5"],
    "warp_targets": ["third.conv", "fourth"],
}


def measure_id(face_app, src_img, out_img):
    f1 = face_app.get(src_img); f2 = face_app.get(out_img)
    if not f1 or not f2: return 0.0
    return float(np.dot(f1[0].normed_embedding, f2[0].normed_embedding))


def main():
    import importlib.util
    from insightface.app import FaceAnalysis

    print("[1/4] Loading InsightFace...")
    face_app = FaceAnalysis(name="antelopev2",
                            root=str(ROOT / "MagicFace/third_party_files"))
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("[2/4] Loading LP pipeline + LoRA...")
    spec = importlib.util.spec_from_file_location("npipe", str(ROOT / "natural_pipeline.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pipeline = mod.NaturalExpressionPipeline()

    from training_engine.training.lora_modules import inject_lora, load_lora_weights
    wrapper = pipeline.lp_pipeline.live_portrait_wrapper

    spade_ckpt = CKPT_DIR / "lora_spade_best.pt"
    if spade_ckpt.exists():
        wrapper.spade_generator, _ = inject_lora(
            wrapper.spade_generator, LORA_CFG["spade_targets"],
            LORA_CFG["rank"], LORA_CFG["alpha"], LORA_CFG["dropout"],
        )
        load_lora_weights(wrapper.spade_generator, str(spade_ckpt))
        wrapper.spade_generator.to("cuda")
        print(f"  ✓ Loaded SPADE LoRA ({spade_ckpt.stat().st_size // 1024}KB)")

    warp_ckpt = CKPT_DIR / "lora_warping_best.pt"
    if warp_ckpt.exists():
        wrapper.warping_module, _ = inject_lora(
            wrapper.warping_module, LORA_CFG["warp_targets"],
            LORA_CFG["rank"], LORA_CFG["alpha"], LORA_CFG["dropout"],
        )
        load_lora_weights(wrapper.warping_module, str(warp_ckpt))
        wrapper.warping_module.to("cuda")
        print(f"  ✓ Loaded Warping LoRA")

    print("[3/4] Testing on 5 neutral images...")
    test_imgs = sorted((ROOT / "raw_data/cleaned/neutral").glob("*.jpg"))[:5]
    results = []
    for img_path in test_imgs:
        src_cv = cv2.imread(str(img_path))
        try:
            out = pipeline._run_lp(str(img_path), DRIVER,
                                   multiplier=1.3, region="lip", use_retargeting=False)
        except Exception as e:
            print(f"  {img_path.name} → FAILED: {e}")
            continue
        if out is None:
            print(f"  {img_path.name} → no output")
            continue
        score = measure_id(face_app, src_cv, out)
        out_path = OUT_DIR / f"{img_path.stem}_smile.jpg"
        cv2.imwrite(str(out_path), out)
        results.append({"src": img_path, "out": out_path, "score": score})
        print(f"  {img_path.name} → identity={score*100:.2f}%  saved={out_path.name}")

    print(f"\n[4/4] Building comparison grid...")
    if not results:
        print("  No results to compare"); return

    h = 400; gap = 8
    rows = []
    for r in results:
        src = Image.open(r["src"]).convert("RGB")
        out = Image.open(r["out"]).convert("RGB")
        sw = int(src.width * h / src.height)
        ow = int(out.width * h / out.height)
        row = Image.new("RGB", (sw + ow + gap, h + 30), (255, 255, 255))
        row.paste(src.resize((sw, h)), (0, 30))
        row.paste(out.resize((ow, h)), (sw + gap, 30))
        d = ImageDraw.Draw(row)
        d.text((10, 8), f"SOURCE: {r['src'].name}", fill=(0, 0, 0))
        d.text((sw + gap + 10, 8), f"LP+LoRA SMILE  identity={r['score']*100:.2f}%",
               fill=(0, 0, 0))
        rows.append(row)
    max_w = max(row.width for row in rows)
    grid = Image.new("RGB", (max_w, sum(r.height for r in rows) + (len(rows)-1)*8),
                     (240, 240, 240))
    y = 0
    for row in rows:
        grid.paste(row, (0, y)); y += row.height + 8
    grid_path = OUT_DIR / "comparison_grid.jpg"
    grid.save(grid_path, quality=92)

    avg = np.mean([r["score"] for r in results])
    print(f"\n{'='*60}")
    print(f"  Avg identity preservation: {avg*100:.2f}%")
    print(f"  Comparison grid: {grid_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
