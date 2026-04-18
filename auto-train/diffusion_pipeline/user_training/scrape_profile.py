"""
Scrape full high-res portfolio for ONE model from imgmodels.com.

Each model profile has 50-150 high-res photos (1024-1500px).
Perfect for per-user LoRA training.

Usage:
    python scrape_profile.py --slug bella-hadid --section london-women
    # → saves to user_data/bella-hadid/raw_photos/
"""
import argparse, hashlib, logging, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [scrape] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
USER_DATA = ROOT / "user_data"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0",
    "Referer": "https://imgmodels.com/",
}

SECTIONS = [
    "london-women", "london-men",
    "new-york-women", "new-york-men",
    "paris-women", "paris-men",
    "milan-women", "milan-men",
    "los-angeles-women", "los-angeles-men",
    "miami-women", "miami-men",
]


def fetch_profile_html(slug: str, section: str) -> str:
    url = f"https://imgmodels.com/{section}/{slug}"
    log.info(f"Fetching {url}")
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    return r.text


def extract_photo_urls(html: str) -> list[str]:
    # Get all mediaslide URLs ending in image extensions
    pattern = r'(mediaslide-us\.storage\.googleapis\.com/imgmodels/pictures/[^"\'\s]+\.(?:jpg|jpeg|png|webp))'
    raw = re.findall(pattern, html)
    # Filter to only "large-" (high-res), skip "profile-" thumbnails
    urls = [f"https://{u}" for u in raw if "/large-" in u]
    # Dedupe
    return list(dict.fromkeys(urls))


def download_one(url, out_dir):
    fname = hashlib.md5(url.encode()).hexdigest()[:10] + ".jpg"
    out = out_dir / fname
    if out.exists() and out.stat().st_size > 1024:
        return {"ok": True, "cached": True, "path": out}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and len(r.content) > 1024:
                out.write_bytes(r.content)
                return {"ok": True, "cached": False, "path": out,
                        "size": len(r.content)}
            elif r.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                return {"ok": False, "reason": f"http_{r.status_code}"}
        except Exception as e:
            if attempt == 2:
                return {"ok": False, "reason": str(e)[:60]}
    return {"ok": False, "reason": "max_retries"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True,
                    help="model slug (e.g. 'bella-hadid')")
    ap.add_argument("--section", default=None,
                    help="profile section. If omitted, tries all sections.")
    args = ap.parse_args()

    sections = [args.section] if args.section else SECTIONS

    html = None
    for sec in sections:
        try:
            html = fetch_profile_html(args.slug, sec)
            urls = extract_photo_urls(html)
            if urls:
                log.info(f"  ✓ Found {len(urls)} photos in section '{sec}'")
                break
        except RuntimeError as e:
            log.info(f"  - {sec}: {e}")
            continue
    else:
        raise SystemExit(f"No profile found for slug '{args.slug}' in any section")

    out_dir = USER_DATA / args.slug / "raw_photos"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading {len(urls)} photos → {out_dir}")
    ok = cached = fail = total_mb = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(download_one, u, out_dir) for u in urls]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["ok"]:
                if r["cached"]: cached += 1
                else:
                    ok += 1
                    total_mb += r.get("size", 0) / 1024 / 1024
            else:
                fail += 1
            if i % 20 == 0 or i == len(urls):
                rate = i / (time.time() - t0 + 0.01)
                log.info(f"  [{i}/{len(urls)}] ok={ok} cached={cached} "
                         f"fail={fail}  {rate:.1f}/s  {total_mb:.1f}MB")

    log.info("=" * 60)
    log.info(f"DONE  ok={ok}  cached={cached}  fail={fail}  "
             f"size={total_mb:.1f}MB  time={time.time()-t0:.1f}s")
    log.info(f"      → {out_dir}")
    log.info(f"      next: python prep_photos.py "
             f"--user_id {args.slug} --input_dir {out_dir}")


if __name__ == "__main__":
    main()
