"""
Test diversity: run the best smile prompt on 4 different people.
Saves side-by-side grids: source | output.
"""
import sys, time, shutil
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import PhotoshootPipeline

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs" / "multi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Best-performing prompt from single-image test
PROMPT = (
    "candid photograph of a person laughing joyfully, mouth open big smile, "
    "natural daylight, warm tones, bokeh background, sharp focus, "
    "DSLR photo, photojournalistic, realistic skin texture, color photograph"
)

SOURCE_DIR = ROOT.parent / "raw_data/imgmodels_london/neutral"

# Pick 4 with varied skin tone, gender, age
TEST_IMAGES = [
    "Hailey_Bieber_57d379af.jpg",
    "Freddie_Mackenzie_ddcb8ad1.jpg",
    "Mary_Ukech_7b8bb2ff.jpg",
    "Alex_Consani_62d57d93.jpg",
]


def build_grid(rows):
    h = 768
    gap = 16
    pad_top = 40
    label_pad = 12
    sized = []
    for src_p, out_p, name in rows:
        src = Image.open(src_p).convert("RGB")
        out = Image.open(out_p).convert("RGB")
        sw = int(src.width * h / src.height)
        ow = int(out.width * h / out.height)
        row = Image.new("RGB", (sw + ow + gap, h + pad_top), (240, 240, 240))
        row.paste(src.resize((sw, h)), (0, pad_top))
        row.paste(out.resize((ow, h)), (sw + gap, pad_top))
        d = ImageDraw.Draw(row)
        d.text((label_pad, 10), f"SOURCE: {name}", fill=(0, 0, 0))
        d.text((sw + gap + label_pad, 10), f"GENERATED: smile", fill=(0, 0, 0))
        sized.append(row)
    max_w = max(r.width for r in sized)
    grid_h = sum(r.height for r in sized) + gap * (len(sized) - 1)
    grid = Image.new("RGB", (max_w, grid_h), (255, 255, 255))
    y = 0
    for r in sized:
        grid.paste(r, (0, y))
        y += r.height + gap
    return grid


def main():
    pipe = PhotoshootPipeline()
    rows = []
    t0 = time.time()

    for i, fname in enumerate(TEST_IMAGES):
        src_path = SOURCE_DIR / fname
        if not src_path.exists():
            print(f"  SKIP: {src_path} not found")
            continue
        print(f"\n[{i+1}/{len(TEST_IMAGES)}] {fname}")
        try:
            images = pipe.generate(
                source_image=str(src_path),
                prompt=PROMPT,
                num_outputs=1,
                num_inference_steps=30,
                identity_scale=0.80,
                controlnet_scale=0.55,
                guidance_scale=6.0,
                seed=1234 + i,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        out_path = OUT_DIR / f"{src_path.stem}_smile.png"
        images[0].save(out_path)
        rows.append((src_path, out_path, fname.rsplit("_", 1)[0]))
        shutil.copy2(src_path, OUT_DIR / f"SOURCE_{src_path.stem}.jpg")
        print(f"  → {out_path.name}")

    if rows:
        grid = build_grid(rows)
        grid_path = OUT_DIR / "comparison_grid.jpg"
        grid.save(grid_path, quality=92)
        print(f"\nGrid saved: {grid_path}")

    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
