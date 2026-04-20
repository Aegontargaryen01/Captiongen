"""
batch.py — Batch Image Caption Processor
Processes a folder of images (or a list of URLs) and saves results to CSV/JSON.

Usage:
  python batch.py --folder ./images --output results.csv
  python batch.py --urls urls.txt  --output results.json
  python batch.py --folder ./images --output results.json --workers 2
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from caption import caption_from_file, caption_from_url

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# ── Workers ───────────────────────────────────────────────────────────────────

def process_file(path: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        caption = caption_from_file(path, prompt)
        status  = "ok"
        error   = ""
    except Exception as e:
        caption = ""
        status  = "error"
        error   = str(e)
    return {
        "source":     path,
        "caption":    caption,
        "status":     status,
        "error":      error,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
    }


def process_url(url: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        caption = caption_from_url(url, prompt)
        status  = "ok"
        error   = ""
    except Exception as e:
        caption = ""
        status  = "error"
        error   = str(e)
    return {
        "source":     url,
        "caption":    caption,
        "status":     status,
        "error":      error,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
    }


# ── Saving ─────────────────────────────────────────────────────────────────────

def save_csv(results: list[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "caption", "status", "error", "elapsed_ms"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[batch] Saved CSV → {path}")


def save_json(results: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[batch] Saved JSON → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_batch(
    folder:  str | None,
    urls_file: str | None,
    output:  str,
    prompt:  str,
    workers: int,
):
    tasks: list[tuple] = []  # (fn, arg)

    if folder:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        image_files = [
            str(p) for p in sorted(folder_path.iterdir())
            if p.suffix.lower() in SUPPORTED
        ]
        print(f"[batch] Found {len(image_files)} image(s) in {folder}")
        tasks = [(process_file, f) for f in image_files]

    elif urls_file:
        with open(urls_file, encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"[batch] Found {len(urls)} URL(s) in {urls_file}")
        tasks = [(process_url, u) for u in urls]

    if not tasks:
        print("[batch] Nothing to process.")
        return

    results = []
    total   = len(tasks)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn, arg, prompt): arg for fn, arg in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            status_icon = "✓" if result["status"] == "ok" else "✗"
            print(f"  [{i}/{total}] {status_icon} {result['source'][:60]}  →  {result['caption'][:60]}")

    # Sort results back into original order
    order = {arg: idx for idx, (_, arg) in enumerate(tasks)}
    results.sort(key=lambda r: order.get(r["source"], 9999))

    # Save
    ext = Path(output).suffix.lower()
    if ext == ".json":
        save_json(results, output)
    else:
        save_csv(results, output)

    ok    = sum(1 for r in results if r["status"] == "ok")
    fail  = total - ok
    print(f"\n[batch] Done. {ok}/{total} succeeded, {fail} failed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch AI image captioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch.py --folder ./photos --output results.csv
  python batch.py --urls urls.txt   --output results.json --workers 2
  python batch.py --folder ./photos --output results.csv --prompt "a photo of"
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", type=str, help="Folder containing images")
    group.add_argument("--urls",   type=str, help="Text file with one image URL per line")

    parser.add_argument("--output",  type=str, default="results.csv", help="Output file (.csv or .json)")
    parser.add_argument("--prompt",  type=str, default="",            help="Optional conditioning prompt")
    parser.add_argument("--workers", type=int, default=1,             help="Parallel workers (default: 1)")

    args = parser.parse_args()
    run_batch(
        folder=args.folder,
        urls_file=args.urls,
        output=args.output,
        prompt=args.prompt,
        workers=args.workers,
    )
