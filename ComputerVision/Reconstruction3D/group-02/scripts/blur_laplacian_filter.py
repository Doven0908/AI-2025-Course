#!/usr/bin/env python3
"""
Laplacian-Variance Blur Filter
------------------------------
Detect and optionally filter out blurry images using the "Variance of Laplacian" method.

Why it works
------------
The Laplacian operator (a 2nd derivative) responds strongly at edges. Blur suppresses high-frequency content,
so the Laplacian response becomes less variable. We therefore compute Var(∇²I). Lower variance => blurrier.

Usage
-----
# Fixed threshold (example: 200.0). Values are data/scale-dependent.
python blur_laplacian_filter.py --input images --output kept --csv scores.csv --threshold 200

# Percentile-based (mark bottom 20% as blurry; no fixed threshold needed)
python blur_laplacian_filter.py --input images --output kept --csv scores.csv --percentile 20

# Normalize size (to make scores more comparable across resolutions), dry-run only
python blur_laplacian_filter.py -i images --max-size 1600 --csv scores.csv

Notes
-----
- Scores depend on image scale, optics and scene. Prefer percentile on a single capture session.
- If your frames vary a lot in size, use --max-size to normalize before scoring.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def load_grayscale(path: Path, max_size: int = 0) -> np.ndarray:
    """Load as grayscale [0..255] uint8. Optionally resize longest side to max_size (keep aspect)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if max_size and max(img.shape[:2]) > max_size:
        h, w = img.shape[:2]
        scale = max_size / float(max(h, w))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def laplacian_variance(gray: np.ndarray, ksize: int = 3) -> float:
    """Compute variance of Laplacian response."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    # Use population variance for consistency
    var = float(lap.var())
    return var


def score_images(paths: List[Path], max_size: int, ksize: int) -> List[Tuple[Path, float]]:
    out = []
    for p in paths:
        try:
            gray = load_grayscale(p, max_size=max_size)
            score = laplacian_variance(gray, ksize=ksize)
            out.append((p, score))
        except Exception as e:
            print(f"[WARN] {p}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Detect/filter blurry images via Variance of Laplacian.")
    ap.add_argument("-i", "--input", required=True, type=Path, help="Input folder with images")
    ap.add_argument("-o", "--output", type=Path, help="Output folder to copy ONLY sharp images (optional)")
    ap.add_argument("--csv", type=Path, default=Path("lapvar_scores.csv"), help="CSV to write scores")
    ap.add_argument("--threshold", type=float, default=None, help="Absolute threshold; below => blurry")
    ap.add_argument("--percentile", type=float, default=None, help="Mark bottom P%% as blurry (0-100)")
    ap.add_argument("--max-size", type=int, default=0, help="Resize longest side to this many px before scoring")
    ap.add_argument("--ksize", type=int, default=3, choices=[1, 3, 5, 7], help="Laplacian kernel size")
    ap.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"],
                    help="Image extensions to include")
    ap.add_argument("--dry-run", action="store_true", help="Do not copy, just compute and report")
    args = ap.parse_args()

    if args.threshold is None and args.percentile is None and args.output is not None:
        print("[INFO] No threshold or percentile provided; nothing will be filtered (dry scoring only).")

    # Collect files
    paths = [p for p in sorted(args.input.rglob("*")) if p.suffix.lower() in {e.lower() for e in args.extensions}]
    if not paths:
        raise SystemExit(f"No images found in {args.input} with extensions {args.extensions}")

    # Score
    scored = score_images(paths, max_size=args.max_size, ksize=args.ksize)

    # Decide blurry/sharp
    blurry_mask = np.zeros(len(scored), dtype=bool)
    if args.percentile is not None:
        if not (0 <= args.percentile <= 100):
            raise SystemExit("--percentile must be in [0, 100]")
        scores_arr = np.array([s for _, s in scored], dtype=np.float64)
        thr = float(np.percentile(scores_arr, args.percentile))
        blurry_mask = scores_arr < thr
        print(f"[AUTO] Percentile={args.percentile:.1f}% => threshold={thr:.3f}")
    elif args.threshold is not None:
        thr = args.threshold
        blurry_mask = np.array([s < thr for _, s in scored], dtype=bool)
        print(f"[FIXED] threshold={thr:.3f}")
    else:
        thr = None

    # Write CSV
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "laplacian_variance", "is_blurry", "threshold", "max_size", "ksize"])
        for (p, s), is_blur in zip(scored, blurry_mask):
            w.writerow([str(p), f"{s:.6f}", int(is_blur), thr if thr is not None else "", args.max_size, args.ksize])
    print(f"[OK] Wrote scores: {args.csv}")

    # Copy sharp images if requested
    if args.output and not args.dry_run and (args.threshold is not None or args.percentile is not None):
        args.output.mkdir(parents=True, exist_ok=True)
        kept = 0
        for (p, _), is_blur in zip(scored, blurry_mask):
            if not is_blur:
                dest = args.output / p.name
                # Use fast copy
                data = p.read_bytes()
                dest.write_bytes(data)
                kept += 1
        print(f"[OK] Copied {kept} sharp images to {args.output}")

    # Print quick stats
    if thr is not None:
        total = len(scored)
        num_blur = int(blurry_mask.sum())
        print(f"[STATS] total={total} blurry={num_blur} ({num_blur/total*100:.1f}%)")

    # Show top/bottom examples (paths only)
    if scored:
        scored_sorted = sorted(scored, key=lambda x: x[1])
        worst = ", ".join([f"{p.name}:{s:.0f}" for p, s in scored_sorted[:min(5, len(scored_sorted))]])
        best = ", ".join([f"{p.name}:{s:.0f}" for p, s in scored_sorted[-min(5, len(scored_sorted)):]])
        print(f"[EXAMPLES] lowest scores: {worst}")
        print(f"[EXAMPLES] highest scores: {best}")


if __name__ == "__main__":
    main()
