"""
A/B test: SDXL base vs RealVisXL V4.0
Same 4 people, same prompt, same seed, different base model.
"""
import sys, time, shutil, gc
from pathlib import Path
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import PhotoshootPipeline

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs" / "ab_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DIR = ROOT.parent / "raw_data/imgmodels_london/neutral"

TEST_IMAGES = [
    "Hailey_Bieber_57d379af.jpg",
    "Freddie_Mackenzie_ddcb8ad1.jpg",
    "Mary_Ukech_7b8bb2ff.jpg",
    "Alex_Consani_62d57d93.jpg",
]

# Same prompt as before - to isolate the model difference
PROMPT = (
    "candid photograph of a person laughing joyfully, mouth open big smile, "
    "natural daylight, warm tones, bokeh background, sharp focus, "
    "DSLR photo, photojournalistic, realistic skin texture, color photograph"
)

MODELS = {
    "sdxl_base":  ROOT / "models" / "sdxl-base",
    "realvis_v4": ROOT / "models" / "realvis-xl-v4",
}


def run_model(label, model_path):
    print(f"\n{'='*60}\n  RUNNING: {label}\n  Model: {model_path}\n{'='*60}")
    # Patch pipeline to use the specific base model
    import pipeline as pl
    pl.MODELS_SDXL_PATH = str(model_path)

    pipe = PhotoshootPipeline()
    # Monkey-patch the SDXL load path
    orig = pipe._load_pipeline
    def patched_load():
        if pipe._pipe is not None: return
        from diffusers import ControlNetModel, AutoencoderKL
        from instantid_pipeline import StableDiffusionXLInstantIDPipeline, draw_kps
        pipe._draw_kps = draw_kps
        controlnet = ControlNetModel.from_pretrained(
            str(pl.MODELS / "InstantID/ControlNetModel"), torch_dtype=pipe.dtype)
        vae = AutoencoderKL.from_pretrained(
            str(pl.MODELS / "sdxl-vae-fp16-fix"), torch_dtype=pipe.dtype)
        sdxl = StableDiffusionXLInstantIDPipeline.from_pretrained(
            str(model_path), controlnet=controlnet, vae=vae,
            torch_dtype=pipe.dtype, variant="fp16", use_safetensors=True)
        sdxl.load_ip_adapter_instantid(str(pl.MODELS / "InstantID/ip-adapter.bin"))
        sdxl.enable_vae_tiling()
        sdxl.enable_model_cpu_offload()
        pipe._pipe = sdxl
    pipe._load_pipeline = patched_load

    outputs = {}
    for i, fname in enumerate(TEST_IMAGES):
        src = SOURCE_DIR / fname
        if not src.exists(): continue
        print(f"  [{i+1}/{len(TEST_IMAGES)}] {fname}")
        img = pipe.generate(
            source_image=str(src), prompt=PROMPT,
            num_outputs=1, num_inference_steps=30,
            identity_scale=0.80, controlnet_scale=0.55,
            guidance_scale=6.0, seed=1234 + i,
        )[0]
        out = OUT_DIR / f"{label}_{src.stem}.png"
        img.save(out)
        outputs[fname] = out
        print(f"    → {out.name}")

    # Free VRAM before next model
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def build_grid(sdxl_outputs, realvis_outputs):
    h = 512
    gap = 12
    pad_top = 40
    rows = []
    for fname in TEST_IMAGES:
        if fname not in sdxl_outputs or fname not in realvis_outputs:
            continue
        src = Image.open(SOURCE_DIR / fname).convert("RGB")
        sdxl = Image.open(sdxl_outputs[fname]).convert("RGB")
        realvis = Image.open(realvis_outputs[fname]).convert("RGB")
        sw = int(src.width * h / src.height)
        w = h  # sdxl/realvis are square
        row = Image.new("RGB", (sw + 2 * w + 2 * gap, h + pad_top),
                        (240, 240, 240))
        row.paste(src.resize((sw, h)), (0, pad_top))
        row.paste(sdxl.resize((w, h)), (sw + gap, pad_top))
        row.paste(realvis.resize((w, h)), (sw + 2 * gap + w, pad_top))
        d = ImageDraw.Draw(row)
        d.text((8, 10), f"SOURCE: {fname.rsplit('_', 1)[0]}", fill=(0, 0, 0))
        d.text((sw + gap + 8, 10), "SDXL base", fill=(180, 0, 0))
        d.text((sw + 2 * gap + w + 8, 10), "RealVisXL v4", fill=(0, 120, 0))
        rows.append(row)
    max_w = max(r.width for r in rows)
    grid = Image.new("RGB", (max_w,
                             sum(r.height for r in rows) + gap * (len(rows) - 1)),
                     (255, 255, 255))
    y = 0
    for r in rows:
        grid.paste(r, (0, y)); y += r.height + gap
    return grid


def main():
    t0 = time.time()
    sdxl_outs = run_model("sdxl_base", MODELS["sdxl_base"])
    realvis_outs = run_model("realvis_v4", MODELS["realvis_v4"])

    print("\nBuilding comparison grid...")
    grid = build_grid(sdxl_outs, realvis_outs)
    grid_path = OUT_DIR / "AB_comparison.jpg"
    grid.save(grid_path, quality=92)
    print(f"\nGRID: {grid_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
