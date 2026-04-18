"""
Side-by-side: source vs RealVisXL output with ArcFace identity score.
No new generation — just measures what we already have.
"""
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [score] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
SRC_DIR = ROOT.parent / "raw_data/imgmodels_london/neutral"
GEN_DIR = ROOT / "outputs/ab_compare"
OUT_PATH = ROOT / "outputs/realvis_vs_source.jpg"

PAIRS = [
    ("Hailey_Bieber_57d379af.jpg",      "realvis_v4_Hailey_Bieber_57d379af.png"),
    ("Freddie_Mackenzie_ddcb8ad1.jpg",  "realvis_v4_Freddie_Mackenzie_ddcb8ad1.png"),
    ("Mary_Ukech_7b8bb2ff.jpg",         "realvis_v4_Mary_Ukech_7b8bb2ff.png"),
    ("Alex_Consani_62d57d93.jpg",       "realvis_v4_Alex_Consani_62d57d93.png"),
]


def embed(app, path):
    img = cv2.imread(str(path))
    if img is None: return None
    faces = app.get(img)
    if not faces: return None
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    return f.normed_embedding


def main():
    from insightface.app import FaceAnalysis
    log.info("Loading AntelopeV2...")
    app = FaceAnalysis(name="antelopev2", root=str(ROOT / "models"),
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    results = []
    for src_name, gen_name in PAIRS:
        src = SRC_DIR / src_name
        gen = GEN_DIR / gen_name
        if not src.exists() or not gen.exists():
            log.warning(f"missing: {src.exists()=} {gen.exists()=}")
            continue
        e1 = embed(app, src)
        e2 = embed(app, gen)
        if e1 is None or e2 is None:
            log.warning(f"no face in {src_name} or {gen_name}")
            continue
        sim = float(np.dot(e1, e2))
        log.info(f"  {src_name[:30]:30}  ArcFace={sim*100:.2f}%")
        results.append((src, gen, sim, src_name.rsplit("_", 1)[0]))

    if not results: return

    avg = np.mean([r[2] for r in results])
    log.info(f"\n  AVG ArcFace identity: {avg*100:.2f}%")

    # Build grid: source | generated with score label
    h = 640
    gap = 12
    header = 48
    rows = []
    for src_p, gen_p, sim, name in results:
        src = Image.open(src_p).convert("RGB")
        gen = Image.open(gen_p).convert("RGB")
        sw = int(src.width * h / src.height)
        gw = int(gen.width * h / gen.height)
        row = Image.new("RGB", (sw + gw + gap, h + header), (245, 245, 245))
        row.paste(src.resize((sw, h)), (0, header))
        row.paste(gen.resize((gw, h)), (sw + gap, header))
        d = ImageDraw.Draw(row)
        color = (0, 140, 0) if sim >= 0.60 else (
                (200, 130, 0) if sim >= 0.45 else (200, 0, 0))
        d.text((12, 14), f"{name}", fill=(0, 0, 0))
        d.text((sw + gap + 12, 14),
               f"RealVisXL v4   |   ArcFace = {sim*100:.2f}%",
               fill=color)
        rows.append(row)

    max_w = max(r.width for r in rows)
    grid = Image.new("RGB", (max_w,
                             sum(r.height for r in rows) + gap * (len(rows) - 1) + 60),
                     (255, 255, 255))
    d = ImageDraw.Draw(grid)
    d.text((12, 20), f"Source  vs  RealVisXL v4        AVG identity = {avg*100:.2f}%",
           fill=(0, 0, 0))
    y = 60
    for r in rows:
        grid.paste(r, (0, y)); y += r.height + gap
    grid.save(OUT_PATH, quality=92)
    log.info(f"\n  Grid → {OUT_PATH}")


if __name__ == "__main__":
    main()
