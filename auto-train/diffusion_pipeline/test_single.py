"""
Quick sanity test: take ONE IMG Models photo, generate a styled portrait.
If this looks real, we proceed. If not, debug before scaling up.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import PhotoshootPipeline

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


PROMPTS = [
    "RAW photo, portrait of a person smiling showing teeth, happy expression, "
    "natural skin texture with visible pores, professional studio lighting, "
    "85mm lens, f/1.8, Canon R5, photojournalism, candid, natural",

    "candid photograph of a person laughing joyfully, mouth open big smile, "
    "natural daylight, warm tones, bokeh background, sharp focus, "
    "DSLR photo, photojournalistic, realistic skin texture",

    "professional headshot, person with genuine warm smile showing teeth, "
    "wearing casual clothing, clean white studio background, "
    "Rembrandt lighting, shot on Kodak Portra 400, film grain, natural",
]


def main():
    # Pick a clear-neutral source image so the identity is obvious
    neutral_dir = ROOT.parent / "raw_data/imgmodels_london/neutral"
    candidates = sorted(neutral_dir.glob("*.jpg"))
    if not candidates:
        print(f"ERROR: no images in {neutral_dir}"); sys.exit(1)

    source = str(candidates[0])
    print(f"Source image: {source}")

    pipe = PhotoshootPipeline()
    t0 = time.time()

    for i, prompt in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] Prompt: {prompt[:70]}...")
        images = pipe.generate(
            source_image=source,
            prompt=prompt,
            num_outputs=1,
            num_inference_steps=30,
            identity_scale=0.75,      # let prompt push expression through
            controlnet_scale=0.55,    # loose pose — allow smile to form
            guidance_scale=6.0,
            seed=42 + i,
        )
        out_path = OUT_DIR / f"test_{i+1}_{Path(source).stem}.png"
        images[0].save(out_path)
        print(f"  → saved: {out_path}")

    # Also save the source for side-by-side comparison
    import shutil
    shutil.copy2(source, OUT_DIR / f"SOURCE_{Path(source).stem}.jpg")
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
