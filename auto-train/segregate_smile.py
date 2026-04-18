"""
Segregate 488 IMG Models images into smile vs neutral using InsightFace landmarks.

Smile detection via 106-pt landmarks:
  - Mouth corner lift (corners above inner lip baseline)
  - Mouth width / face width ratio
  - Teeth visibility (upper-lower lip gap)
"""
import shutil, logging
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [seg] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "raw_data/imgmodels_london/images"
SMILE_DIR = ROOT / "raw_data/imgmodels_london/smile"
NEUTRAL_DIR = ROOT / "raw_data/imgmodels_london/neutral"
AMBIG_DIR = ROOT / "raw_data/imgmodels_london/ambiguous"
NOFACE_DIR = ROOT / "raw_data/imgmodels_london/no_face"

for d in [SMILE_DIR, NEUTRAL_DIR, AMBIG_DIR, NOFACE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def smile_score(landmarks_106, bbox):
    """Compute smile score from 106 facial landmarks.

    Landmarks layout (antelopev2 106-point):
      52: left mouth corner
      61: right mouth corner
      71: upper lip center (outer)
      77: lower lip center (outer)
      84: upper lip center (inner)
      90: lower lip center (inner)
    """
    lm = np.array(landmarks_106)
    left_corner  = lm[52]
    right_corner = lm[61]
    upper_outer  = lm[71]
    lower_outer  = lm[77]
    upper_inner  = lm[84]
    lower_inner  = lm[90]

    face_height = bbox[3] - bbox[1]
    face_width  = bbox[2] - bbox[0]

    mouth_mid_y = (upper_outer[1] + lower_outer[1]) / 2
    corner_mid_y = (left_corner[1] + right_corner[1]) / 2
    corner_lift = (mouth_mid_y - corner_mid_y) / face_height

    mouth_width = np.linalg.norm(right_corner - left_corner)
    width_ratio = mouth_width / face_width

    teeth_gap = abs(lower_inner[1] - upper_inner[1])
    teeth_ratio = teeth_gap / face_height

    score = (corner_lift * 100) + (width_ratio * 30) + (teeth_ratio * 80)
    # Direct smile signals (calibrated from 488-image distribution):
    #   smile: mouth clearly open (teeth_ratio >= 0.22) AND corners lifted (lift >= -0.08)
    #   neutral: mouth closed (teeth_ratio < 0.18)
    if teeth_ratio >= 0.22 and corner_lift >= -0.08:
        cls = "smile"
    elif teeth_ratio < 0.18:
        cls = "neutral"
    else:
        cls = "ambiguous"
    return float(score), cls, {
        "corner_lift": float(corner_lift),
        "width_ratio": float(width_ratio),
        "teeth_ratio": float(teeth_ratio),
    }


def main():
    from insightface.app import FaceAnalysis

    log.info("Loading InsightFace antelopev2...")
    app = FaceAnalysis(name="antelopev2",
                       root=str(ROOT / "MagicFace/third_party_files"))
    app.prepare(ctx_id=0, det_size=(640, 640))

    imgs = sorted(SRC_DIR.glob("*.jpg"))
    log.info(f"Processing {len(imgs)} images...")

    results = []
    smile_count = neutral_count = ambig_count = noface_count = 0

    for i, img_path in enumerate(imgs, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            noface_count += 1
            shutil.copy2(img_path, NOFACE_DIR / img_path.name)
            continue

        faces = app.get(img)
        if not faces:
            noface_count += 1
            shutil.copy2(img_path, NOFACE_DIR / img_path.name)
            continue

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        if not hasattr(face, "landmark_2d_106") or face.landmark_2d_106 is None:
            noface_count += 1
            shutil.copy2(img_path, NOFACE_DIR / img_path.name)
            continue

        score, cls, metrics = smile_score(face.landmark_2d_106, face.bbox)
        if cls == "smile":
            target = SMILE_DIR; smile_count += 1
        elif cls == "neutral":
            target = NEUTRAL_DIR; neutral_count += 1
        else:
            target = AMBIG_DIR; ambig_count += 1
        shutil.copy2(img_path, target / img_path.name)

        results.append({
            "name": img_path.name,
            "score": score,
            "class": cls,
            **metrics,
        })

        if i % 50 == 0 or i == len(imgs):
            log.info(f"  [{i}/{len(imgs)}] smile={smile_count} "
                     f"neutral={neutral_count} ambig={ambig_count} "
                     f"noface={noface_count}")

    log.info("=" * 60)
    log.info(f"SMILE:     {smile_count:>3} → {SMILE_DIR}")
    log.info(f"NEUTRAL:   {neutral_count:>3} → {NEUTRAL_DIR}")
    log.info(f"AMBIGUOUS: {ambig_count:>3} → {AMBIG_DIR}")
    log.info(f"NOFACE:    {noface_count:>3} → {NOFACE_DIR}")

    # Print top 10 smiles and top 10 most neutral
    if results:
        sorted_r = sorted(results, key=lambda r: r["score"], reverse=True)
        log.info("\nTop 10 smile scores:")
        for r in sorted_r[:10]:
            log.info(f"  {r['score']:6.2f}  {r['name']}")
        log.info("\nTop 10 most neutral (lowest scores):")
        for r in sorted_r[-10:]:
            log.info(f"  {r['score']:6.2f}  {r['name']}")


if __name__ == "__main__":
    main()
