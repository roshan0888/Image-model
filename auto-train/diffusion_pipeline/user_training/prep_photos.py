"""
Prepare user photos for LoRA training.

Takes raw user uploads → crops to face + context, resizes to 1024×1024,
filters blurry / bad-pose shots.

Usage:
    python prep_photos.py \\
        --user_id john_doe \\
        --input_dir ~/uploads/john_doe_raw \\
        --min_face_size 200
"""
import argparse, logging, shutil
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [prep] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MODELS = ROOT.parent / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--min_face_size", type=int, default=200,
                    help="reject photos where face < this size")
    ap.add_argument("--min_sharpness", type=float, default=50.0,
                    help="Laplacian variance threshold")
    ap.add_argument("--max_pose_angle", type=float, default=35.0,
                    help="reject if yaw/pitch > this")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--pad_ratio", type=float, default=1.4,
                    help="padding around face (1.0=tight, 2.0=loose)")
    args = ap.parse_args()

    src_dir = Path(args.input_dir).expanduser().resolve()
    if not src_dir.exists():
        raise SystemExit(f"No such dir: {src_dir}")

    out_dir = ROOT / "user_data" / args.user_id / "photos"
    out_dir.mkdir(parents=True, exist_ok=True)

    from insightface.app import FaceAnalysis
    log.info("Loading face detector...")
    app = FaceAnalysis(name="antelopev2", root=str(MODELS),
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    kept = skipped = 0
    stats = {"small": 0, "blurry": 0, "pose": 0, "noface": 0, "badread": 0}
    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            stats["badread"] += 1; skipped += 1; continue

        faces = app.get(img)
        if not faces:
            stats["noface"] += 1; skipped += 1; continue

        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        x1, y1, x2, y2 = f.bbox
        fw, fh = x2 - x1, y2 - y1
        if min(fw, fh) < args.min_face_size:
            stats["small"] += 1; skipped += 1; continue

        pose = f.pose if hasattr(f, "pose") and f.pose is not None else [0, 0, 0]
        if abs(pose[0]) > args.max_pose_angle or abs(pose[1]) > args.max_pose_angle:
            stats["pose"] += 1; skipped += 1; continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharp < args.min_sharpness:
            stats["blurry"] += 1; skipped += 1; continue

        # Crop around face with padding
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(fw, fh) * args.pad_ratio / 2
        H, W = img.shape[:2]
        crop_x1 = max(0, int(cx - half))
        crop_y1 = max(0, int(cy - half))
        crop_x2 = min(W, int(cx + half))
        crop_y2 = min(H, int(cy + half))
        cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize to square
        sz = args.size
        square = cv2.resize(cropped, (sz, sz), interpolation=cv2.INTER_LANCZOS4)
        out_path = out_dir / f"{kept:03d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), square, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept += 1
        log.info(f"  ✓ {img_path.name} → {out_path.name} "
                 f"(face={int(fw)}x{int(fh)}, sharp={sharp:.0f})")

    log.info("")
    log.info(f"KEPT:    {kept:>3} → {out_dir}")
    log.info(f"SKIPPED: {skipped:>3}  {stats}")
    if kept < 10:
        log.warning(f"Only {kept} photos passed — aim for 10-20 for best results")


if __name__ == "__main__":
    main()
