"""Simple verification: face + frontal + smile + >= 200px."""
import os, cv2, sys, logging
from pathlib import Path
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [verify] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
RAW = ROOT / "raw_data/model_photos/smile_raw"
OUT = ROOT / "raw_data/cleaned/smile"
OUT.mkdir(parents=True, exist_ok=True)

fa = FaceAnalysis(
    name="antelopev2",
    root=str(ROOT / "MagicFace/third_party_files"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)

stats = {
    "total": 0, "no_face": 0, "multi_face": 0, "small": 0,
    "side_pose": 0, "blurry": 0, "not_smiling": 0, "underage": 0,
    "verified": 0,
}

for fpath in sorted(RAW.glob("*")):
    if fpath.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue
    stats["total"] += 1

    img = cv2.imread(str(fpath))
    if img is None:
        fpath.unlink()
        continue

    faces = fa.get(img)
    if not faces:
        stats["no_face"] += 1
        fpath.unlink()
        continue
    if len(faces) > 1:
        stats["multi_face"] += 1
        fpath.unlink()
        continue

    face = faces[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    fw, fh = x2-x1, y2-y1

    if fw < 200 or fh < 200:
        stats["small"] += 1
        fpath.unlink()
        continue

    if getattr(face, "age", 25) < 18:
        stats["underage"] += 1
        fpath.unlink()
        continue

    lmk = getattr(face, "landmark_2d_106", None)
    if lmk is not None and len(lmk) > 86:
        nose_x = lmk[86, 0]
        if abs(nose_x - (x1+x2)/2) / max(fw, 1) > 0.15:
            stats["side_pose"] += 1
            fpath.unlink()
            continue

    crop = img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]
    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 80:
            stats["blurry"] += 1
            fpath.unlink()
            continue

    # Smile check via mouth corners
    smiling = True
    if lmk is not None and len(lmk) > 72:
        left = lmk[52]
        right = lmk[61] if len(lmk) > 61 else lmk[58]
        upper = lmk[56] if len(lmk) > 56 else lmk[53]
        lower = lmk[66] if len(lmk) > 66 else lmk[62]
        corner_y = (left[1] + right[1]) / 2
        center_y = (upper[1] + lower[1]) / 2
        mouth_w = abs(right[0] - left[0])
        if mouth_w >= 10 and corner_y > center_y + mouth_w * 0.05:
            smiling = False

    if not smiling:
        stats["not_smiling"] += 1
        fpath.unlink()
        continue

    # VERIFIED
    dst = OUT / fpath.name
    fpath.rename(dst)
    stats["verified"] += 1

log.info("=" * 50)
for k, v in stats.items():
    log.info(f"  {k:<14} {v}")
log.info("=" * 50)
