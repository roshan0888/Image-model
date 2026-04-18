"""
Download IMG Models London directory images (488 models).
Parallel, resumable, progress-tracked.
"""
import time, hashlib, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [dl] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
URLS_FILE = ROOT / "raw_data/imgmodels_london/all_urls.txt"
OUT_DIR = ROOT / "raw_data/imgmodels_london/images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://imgmodels.com/",
}


def download_one(row):
    name, url = row
    short_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    out_path = OUT_DIR / f"{name}_{short_hash}.jpg"
    if out_path.exists() and out_path.stat().st_size > 1024:
        return {"name": name, "ok": True, "reason": "cached"}

    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and len(r.content) > 1024:
                out_path.write_bytes(r.content)
                return {"name": name, "ok": True, "reason": "downloaded",
                        "size": len(r.content)}
            elif r.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                return {"name": name, "ok": False, "reason": f"http_{r.status_code}"}
        except Exception as e:
            if attempt == 2:
                return {"name": name, "ok": False, "reason": str(e)[:80]}
            time.sleep(1)
    return {"name": name, "ok": False, "reason": "max_retries"}


def main():
    rows = []
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line: continue
            name, url = line.split("|", 1)
            rows.append((name.strip(), url.strip()))

    log.info(f"Loaded {len(rows)} URLs → {OUT_DIR}")

    ok = fail = cached = total_bytes = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(download_one, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res["ok"]:
                if res["reason"] == "cached":
                    cached += 1
                else:
                    ok += 1
                    total_bytes += res.get("size", 0)
            else:
                fail += 1
            if i % 40 == 0 or i == len(rows):
                rate = i / (time.time() - t0 + 0.01)
                log.info(f"  [{i}/{len(rows)}] ok={ok} cached={cached} "
                         f"fail={fail} rate={rate:.1f}/s")

    dt = time.time() - t0
    mb = total_bytes / 1024 / 1024
    log.info("=" * 50)
    log.info(f"DONE in {dt:.1f}s  downloaded={ok} ({mb:.1f}MB)  "
             f"cached={cached}  failed={fail}")


if __name__ == "__main__":
    main()
