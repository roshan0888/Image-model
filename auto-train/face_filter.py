"""
Face Quality Filter
Scans raw_data/model_photos and removes anything that is not a clean single frontal face.
Rejected images move to raw_data/rejected/
"""
import os, cv2, shutil, sys
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), 'raw_data/model_photos')
TRASH = os.path.join(os.path.dirname(__file__), 'raw_data/rejected')
os.makedirs(TRASH, exist_ok=True)

print("Loading InsightFace...", flush=True)
from insightface.app import FaceAnalysis
app = FaceAnalysis(
    name="antelopev2",
    root=os.path.join(os.path.dirname(__file__), 'MagicFace/third_party_files')
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace loaded ✓", flush=True)

total_kept = total_removed = 0

for expr in ['smile', 'neutral', 'surprise', 'sad']:
    d = os.path.join(BASE, expr)
    if not os.path.exists(d):
        continue
    files = [f for f in os.listdir(d)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    kept = 0
    reasons = {}

    for i, fname in enumerate(files):
        fpath = os.path.join(d, fname)
        reason = None
        try:
            img = cv2.imread(fpath)
            if img is None:
                reason = "unreadable"
            else:
                h, w = img.shape[:2]
                if min(h, w) < 200:
                    reason = "too_small"
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
                        reason = "blurry"
                    else:
                        faces = app.get(img)
                        if len(faces) == 0:
                            reason = "no_face"
                        elif len(faces) > 1:
                            reason = "multi_face"
                        else:
                            face = faces[0]
                            if face.det_score < 0.6:
                                reason = "low_confidence"
                            else:
                                b = face.bbox.astype(int)
                                fw, fh = b[2] - b[0], b[3] - b[1]
                                if fw < 80 or fh < 80:
                                    reason = "face_too_small"
                                elif hasattr(face, 'age') and face.age is not None and face.age < 18:
                                    reason = "underage"
                                elif hasattr(face, 'pose') and face.pose is not None:
                                    p, y, r = face.pose
                                    if abs(y) > 35 or abs(p) > 30:
                                        reason = "extreme_pose"
        except Exception as e:
            reason = "error"

        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            try:
                os.rename(fpath, os.path.join(TRASH, f"{expr}_{fname}"))
            except Exception:
                try:
                    os.remove(fpath)
                except Exception:
                    pass
            total_removed += 1
        else:
            kept += 1
            total_kept += 1

        if (i + 1) % 50 == 0:
            print(f"  {expr}: processed {i+1}/{len(files)}...", flush=True)

    print(f"  {expr:10s}: {kept:3d} KEPT | rejected={reasons}", flush=True)

rate = total_kept / max(1, total_kept + total_removed) * 100
print(f"\nFINAL: {total_kept} clean faces | {total_removed} removed | {rate:.0f}% pass rate",
      flush=True)
