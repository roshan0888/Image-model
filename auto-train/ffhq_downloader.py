"""
FFHQ Dataset Downloader

Downloads Flickr-Faces-HQ dataset from HuggingFace (Ryan-sjtu/ffhq512-caption).
FFHQ is THE gold standard face dataset:
  - 70,000 high-quality Flickr photos
  - Real humans (no AI-generated)
  - 512×512 aligned faces
  - Diverse ages, ethnicities, poses
  - No watermarks (Flickr user uploads)
  - Used to train StyleGAN, StyleGAN2

Strategy:
  1. Download 1-3 parquet chunks (each has ~1,300 images = ~4,000 total)
  2. Extract images to disk
  3. Filter to smiling + frontal via InsightFace
  4. Use as training pairs source
"""

import os, io, sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ffhq] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
FFHQ_DIR = ROOT / "raw_data/model_photos/ffhq_raw"
FFHQ_DIR.mkdir(parents=True, exist_ok=True)


def download_ffhq_chunks(num_chunks: int = 2):
    """Download N chunks from FFHQ-512-caption parquet dataset."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    repo = "Ryan-sjtu/ffhq512-caption"
    log.info(f"Downloading {num_chunks} chunks from {repo}...")

    total_saved = 0
    for i in range(num_chunks):
        # Find next chunk file
        try:
            from huggingface_hub import list_repo_files
            all_files = list_repo_files(repo, repo_type="dataset")
            parquet_files = sorted([f for f in all_files if f.endswith('.parquet')])
            if i >= len(parquet_files):
                break
            chunk_path = parquet_files[i]
            log.info(f"  Chunk {i+1}/{num_chunks}: {chunk_path}")
        except Exception as e:
            log.error(f"  list fail: {e}")
            break

        try:
            local_file = hf_hub_download(repo, chunk_path, repo_type="dataset")
            table = pq.read_table(local_file)
            log.info(f"    Rows in chunk: {len(table)}")
            log.info(f"    Columns: {table.column_names}")

            # Extract images (stored as bytes in 'image' column)
            if 'image' in table.column_names:
                images = table.column('image').to_pylist()
            elif 'img' in table.column_names:
                images = table.column('img').to_pylist()
            else:
                # Try to find a bytes column
                log.info(f"    Columns: {table.column_names}")
                continue

            for idx, img_data in enumerate(images):
                if isinstance(img_data, dict):
                    img_bytes = img_data.get('bytes')
                else:
                    img_bytes = img_data
                if img_bytes is None:
                    continue

                fname = f"ffhq_c{i:02d}_{idx:05d}.jpg"
                fpath = FFHQ_DIR / fname
                if fpath.exists():
                    continue
                try:
                    with open(fpath, 'wb') as f:
                        f.write(img_bytes)
                    total_saved += 1
                except Exception as e:
                    log.debug(f"    write fail: {e}")

            log.info(f"    Chunk {i+1}: saved {len(images)} images (total: {total_saved})")
        except Exception as e:
            log.error(f"    chunk {i+1} error: {e}")

    return total_saved


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=int, default=2)
    args = p.parse_args()

    n = download_ffhq_chunks(num_chunks=args.chunks)
    log.info(f"DONE — {n} FFHQ images in {FFHQ_DIR}")
