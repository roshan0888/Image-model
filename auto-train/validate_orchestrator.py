"""
Validate the master orchestrator on a batch of test images.
Reports per-category pass rates, identity scores, timings.
"""

import sys, json, time, csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from master_orchestrator import MasterOrchestrator

def validate(test_dir: str, max_images: int = 10):
    orch = MasterOrchestrator()
    test_images = sorted(Path(test_dir).glob("*.jpg"))[:max_images]
    if not test_images:
        print(f"No images in {test_dir}"); return

    print(f"\n{'='*70}")
    print(f"  ORCHESTRATOR VALIDATION — {len(test_images)} images")
    print(f"{'='*70}\n")

    results = []
    for i, img in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] {img.name}")
        r = orch.process(str(img), expression="smile")
        results.append({
            "image": img.name,
            "status": r["status"],
            "engine": r.get("engine_used", "n/a"),
            "category": r.get("category"),
            "yaw": round(r.get("yaw", 0), 1),
            "identity": round(r.get("identity_score", 0), 4),
            "confidence": r.get("confidence"),
            "attempts": r.get("attempts", 0),
            "time": round(r.get("time", 0), 1),
        })

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    total_success = sum(1 for r in results if r["status"] == "success")
    print(f"\n  Overall pass rate: {total_success}/{len(results)} ({total_success/len(results)*100:.0f}%)")

    for cat, items in by_cat.items():
        success = [r for r in items if r["status"] == "success"]
        if not success:
            print(f"\n  {cat.upper()}: {len(items)} total, 0 success")
            continue
        avg_id = sum(r["identity"] for r in success) / len(success)
        avg_time = sum(r["time"] for r in success) / len(success)
        print(f"\n  {cat.upper()}: {len(items)} total, {len(success)} success")
        print(f"    Avg identity:  {avg_id:.4f} ({avg_id*100:.1f}%)")
        print(f"    Avg time:      {avg_time:.1f}s")
        print(f"    Engines used:  {set(r['engine'] for r in success)}")

    # CSV export
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"production/validation_{ts}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"\n  CSV: {csv_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", default=str(ROOT / "raw_data/starnow_ugc_models/verified"))
    p.add_argument("--max", type=int, default=10)
    args = p.parse_args()
    validate(args.test_dir, args.max)
