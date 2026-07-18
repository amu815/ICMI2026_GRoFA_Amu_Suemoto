#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src3/build_dataset_neurips_fairface.py
Corruption-dataset builder — FairFace
- CSV-based label reading (train_labels.csv / val_labels.csv)
- 7 races: White, Black, East Asian, Indian, Latino_Hispanic, Middle Eastern, Southeast Asian
- Random sampling of 5000 IDs from FairFace train set
- Same noise augmentation as UTKFace (7 types × 3 levels)
"""

import argparse
import json
import random
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm
import io

# ==========================================
# 1. Optimized Noise Functions
# ==========================================
def to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img).astype(np.float32)

def to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def noise_gaussian(img: Image.Image, severity: int) -> Image.Image:
    c = [0.08, 0.12, 0.18][severity - 1]
    arr = to_numpy(img) / 255.0
    noise = np.random.normal(0, c, arr.shape)
    noisy = np.clip(arr + noise, 0, 1) * 255.0
    return to_image(noisy)

def noise_shot(img: Image.Image, severity: int) -> Image.Image:
    c = [60, 25, 12][severity - 1]
    arr = to_numpy(img) / 255.0
    arr = np.clip(arr, 0, 1)
    noisy = np.random.poisson(arr * c) / float(c)
    return to_image(noisy * 255.0)

def noise_impulse(img: Image.Image, severity: int) -> Image.Image:
    c = [0.03, 0.06, 0.09][severity - 1]
    arr = to_numpy(img)
    mask = np.random.binomial(1, c, arr.shape[:2])
    salt = np.random.binomial(1, 0.5, arr.shape[:2])
    if len(arr.shape) == 3:
        mask = mask[..., np.newaxis]
        salt = salt[..., np.newaxis]
    noisy = arr * (1 - mask) + (salt * 255) * mask
    return to_image(noisy)

def noise_defocus(img: Image.Image, severity: int) -> Image.Image:
    c = [2, 4, 6][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=c))

def noise_glass(img: Image.Image, severity: int) -> Image.Image:
    params = [(0.7, 1, 1), (0.9, 2, 1), (1.5, 2, 2)][severity - 1]
    sigma, max_delta, iterations = params
    arr = to_numpy(img)
    h, w = arr.shape[:2]
    for _ in range(iterations):
        dx = np.random.randint(-max_delta, max_delta, size=(h, w))
        dy = np.random.randint(-max_delta, max_delta, size=(h, w))
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        new_x = np.clip(x + dx, 0, w - 1)
        new_y = np.clip(y + dy, 0, h - 1)
        arr = arr[new_y, new_x]
    return to_image(arr).filter(ImageFilter.GaussianBlur(radius=sigma))

def noise_jpeg(img: Image.Image, severity: int) -> Image.Image:
    c = [75, 40, 20][severity - 1]
    out = io.BytesIO()
    img.save(out, format='JPEG', quality=c)
    out.seek(0)
    return Image.open(out)

def noise_cutout(img: Image.Image, severity: int) -> Image.Image:
    c = [30, 50, 70][severity - 1]
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    if h > c and w > c:
        y = np.random.randint(0, h - c)
        x = np.random.randint(0, w - c)
        arr[y:y+c, x:x+c] = 0
    return Image.fromarray(arr)

NOISE_MAP = {
    "gaussian": noise_gaussian,
    "shot": noise_shot,
    "impulse": noise_impulse,
    "defocus": noise_defocus,
    "glass": noise_glass,
    "jpeg": noise_jpeg,
    "cutout": noise_cutout,
}

# ==========================================
# 2. Main Builder
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=Path, required=True,
                        help="FairFace root dir containing train_labels.csv, val_labels.csv, train/, val/")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "jsonl").mkdir(parents=True, exist_ok=True)

    # --- Load FairFace CSV (train only — sample from train set) ---
    csv_path = args.src_dir / "train_labels.csv"
    if not csv_path.exists():
        print(f"[Fatal] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"[Info] Loaded {len(df)} entries from {csv_path}")

    # --- Normalize labels ---
    # age: "more than 70" → "70+"
    df["age"] = df["age"].replace("more than 70", "70+")
    # gender: "Male" → "male", "Female" → "female"
    df["gender"] = df["gender"].str.lower()
    # race: keep as-is (7 categories)

    # --- Validate image existence ---
    valid_mask = df["file"].apply(lambda f: (args.src_dir / f).exists())
    df = df[valid_mask].reset_index(drop=True)
    print(f"[Info] {len(df)} images verified to exist")

    # --- Sample ---
    if len(df) > args.n_samples:
        print(f"[Info] Sampling {args.n_samples} from {len(df)} ...")
        df = df.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)
    else:
        print(f"[Warning] Only {len(df)} images available (requested {args.n_samples})")

    # --- Train/Test split ---
    indices = list(range(len(df)))
    random.shuffle(indices)
    n_test = int(len(indices) * args.test_ratio)
    test_indices = set(indices[:n_test])
    train_indices = set(indices[n_test:])

    print(f"[Info] Split: Train={len(train_indices)}, Test={len(test_indices)}")

    # --- Print distributions ---
    print(f"\n[Info] Race distribution:")
    print(df["race"].value_counts().to_string())
    print(f"\n[Info] Gender distribution:")
    print(df["gender"].value_counts().to_string())
    print(f"\n[Info] Age distribution:")
    print(df["age"].value_counts().sort_index().to_string())
    print()

    # --- Process each split ---
    for split_name, idx_set in [("train", train_indices), ("test", test_indices)]:
        print(f"Processing {split_name} set ({len(idx_set)} images) ...")
        rows = []

        for idx in tqdm(sorted(idx_set), unit="img", ncols=80):
            row = df.iloc[idx]
            # Extract original filename number as base_id
            orig_file = row["file"]  # e.g. "train/1234.jpg"
            orig_name = Path(orig_file).stem  # e.g. "1234"
            base_id = f"ff_train_{orig_name}"

            attrs = {
                "age": row["age"],
                "gender": row["gender"],
                "race": row["race"],
            }

            img_path = args.src_dir / orig_file
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            # Clean
            out_fname = f"{orig_name}.jpg"
            rel_clean = f"images/{split_name}/clean/{out_fname}"
            dst_clean = args.out_dir / rel_clean
            dst_clean.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_clean)
            rows.append({
                "id": base_id, "image": str(rel_clean), "split": split_name,
                "noise_type": "clean", "noise_level": 0, **attrs
            })

            # Noisy (7 types × 3 levels)
            for n_type, n_func in NOISE_MAP.items():
                for lvl in [1, 2, 3]:
                    noisy = n_func(img, lvl)
                    fname = f"{orig_name}_{n_type}_lv{lvl}.jpg"
                    rel_noisy = f"images/{split_name}/{n_type}/{fname}"
                    dst_noisy = args.out_dir / rel_noisy
                    dst_noisy.parent.mkdir(parents=True, exist_ok=True)
                    noisy.save(dst_noisy)
                    rows.append({
                        "id": f"{base_id}_{n_type}_lv{lvl}", "image": str(rel_noisy),
                        "split": split_name, "noise_type": n_type, "noise_level": lvl,
                        **attrs
                    })

        out_jsonl = args.out_dir / "jsonl" / f"neurips_{split_name}.jsonl"
        with open(out_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {out_jsonl} ({len(rows)} rows)")

    print("[Done]")

if __name__ == "__main__":
    main()
