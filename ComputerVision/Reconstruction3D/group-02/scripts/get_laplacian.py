import argparse
from pathlib import Path

from blur_laplacian_filter import load_grayscale, laplacian_variance


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calculate Variance of Laplacian.")
    ap.add_argument("-i", "--input", required=True, type=Path, help="Input an image")
    args = ap.parse_args()
    img = load_grayscale(args.input)
    score = laplacian_variance(img)
    print(f"Laplacian Variance score: {score:.2f}")
