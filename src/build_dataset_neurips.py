#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src3/build_dataset_neurips.py (v5: Fix Impulse & Add Cutout)
Corruption-dataset builder — UTKFace
- RCIS設定準拠: 無作為抽出 5000 ID
- Robust Walk: ディレクトリ構造を問わず画像を収集
- Universal Noise: 7種類 (Gaussian, Shot, Impulse, Defocus, Glass, Jpeg, Cutout)
"""

import argparse
import json
import random
import os
from pathlib import Path
import numpy as np
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
    """
    [Fixed] 0-255スケールで直接処理するように修正 (黒画像バグの解消)
    """
    c = [0.03, 0.06, 0.09][severity - 1]
    arr = to_numpy(img) # 0-255 float
    
    # S&P Noise mask
    mask = np.random.binomial(1, c, arr.shape[:2])
    salt = np.random.binomial(1, 0.5, arr.shape[:2]) # 1=Salt(White), 0=Pepper(Black)
    
    if len(arr.shape) == 3:
        mask = mask[..., np.newaxis]
        salt = salt[..., np.newaxis]
    
    # mask=1 の場所を salt(0 or 1)*255 に置き換え
    # mask=0 の場所は 元の画素(arr) を維持
    noisy = arr * (1 - mask) + (salt * 255) * mask
    
    return to_image(noisy)

def noise_defocus(img: Image.Image, severity: int) -> Image.Image:
    c = [2, 4, 6][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=c))

def noise_glass(img: Image.Image, severity: int) -> Image.Image:
    # parameters: (sigma, max_delta, iterations)
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
    """
    [Added] 画像の一部を黒く塗りつぶす (遮蔽)
    """
    # 遮蔽サイズ (画像サイズに対する比率や固定サイズ)
    # ここでは固定サイズ: 30, 50, 70 pixel box
    c = [30, 50, 70][severity - 1]
    
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    
    # ランダムな位置 (はみ出さないように)
    if h > c and w > c:
        y = np.random.randint(0, h - c)
        x = np.random.randint(0, w - c)
        arr[y:y+c, x:x+c] = 0 # Black box
    
    return Image.fromarray(arr)

NOISE_MAP = {
    "gaussian": noise_gaussian, 
    "shot": noise_shot, 
    "impulse": noise_impulse,
    "defocus": noise_defocus, 
    "glass": noise_glass, 
    "jpeg": noise_jpeg,
    "cutout": noise_cutout  # Added
}

# ==========================================
# 2. Main Builder
# ==========================================
def find_images_robust(root_dir: Path):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    found_files = []
    print(f"[Info] Searching images under {root_dir} ...")
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in valid_exts:
                found_files.append(p)
    return found_files

# 必要な変換辞書などを定義
def utkface_age_to_fairface_bin(age_int: int) -> str:
    if age_int <= 2: return "0-2"
    if age_int <= 9: return "3-9"
    if age_int <= 19: return "10-19"
    if age_int <= 29: return "20-29"
    if age_int <= 39: return "30-39"
    if age_int <= 49: return "40-49"
    if age_int <= 59: return "50-59"
    if age_int <= 69: return "60-69"
    return "70+"

def parse_utk_filename(fname: str):
    try:
        parts = fname.split('_')
        if len(parts) < 3: return None
        
        age_raw = int(parts[0])
        gender_raw = int(parts[1])
        race_raw = int(parts[2])

        # ★ここを追加：IJCAI同様のフォーマットに変換する
        age_bin = utkface_age_to_fairface_bin(age_raw)
        gender_str = "male" if gender_raw == 0 else "female"
        # 人種マップ (UTKFace仕様)
        race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
        race_str = race_map.get(race_raw, "Others")

        return {
            "age": age_bin,       # 文字列のBinに変更
            "gender": gender_str, # 文字列に変更
            "race": race_str      # 文字列に変更
        }
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "jsonl").mkdir(parents=True, exist_ok=True)

    all_files = find_images_robust(args.src_dir)
    if not all_files:
        print("[Fatal] No images found.")
        return

    valid_files = [f for f in all_files if parse_utk_filename(f.name)]
    print(f"[Info] Found {len(valid_files)} valid UTKFace images total.")

    if len(valid_files) > args.n_samples:
        print(f"[Info] Sampling {args.n_samples} images randomly...")
        selected_files = random.sample(valid_files, args.n_samples)
    else:
        selected_files = valid_files

    random.shuffle(selected_files)
    n_test = int(len(selected_files) * args.test_ratio)
    test_files = selected_files[:n_test]
    train_files = selected_files[n_test:]
    
    print(f"[Info] Final Split: Train={len(train_files)}, Test={len(test_files)}")

    for split_name, file_list in [("train", train_files), ("test", test_files)]:
        print(f"Processing {split_name} set ...")
        rows = []
        # TQDM: unit='img'
        for fpath in tqdm(file_list, unit="img", ncols=80):
            attrs = parse_utk_filename(fpath.name)
            base_id = fpath.stem
            
            try:
                img = Image.open(fpath).convert("RGB")
            except: continue
            
            # Clean
            rel_clean = f"images/{split_name}/clean/{fpath.name}"
            dst_clean = args.out_dir / rel_clean
            dst_clean.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_clean)
            rows.append({"id": base_id, "image": str(rel_clean), "split": split_name, "noise_type": "clean", "noise_level": 0, **attrs})

            # Noisy (7 types)
            for n_type, n_func in NOISE_MAP.items():
                for lvl in [1, 2, 3]:
                    noisy = n_func(img, lvl)
                    fname = f"{base_id}_{n_type}_lv{lvl}.jpg"
                    rel_noisy = f"images/{split_name}/{n_type}/{fname}"
                    dst_noisy = args.out_dir / rel_noisy
                    dst_noisy.parent.mkdir(parents=True, exist_ok=True)
                    noisy.save(dst_noisy)
                    rows.append({"id": f"{base_id}_{n_type}_lv{lvl}", "image": str(rel_noisy), "split": split_name, "noise_type": n_type, "noise_level": lvl, **attrs})

        out_jsonl = args.out_dir / "jsonl" / f"neurips_{split_name}.jsonl"
        with open(out_jsonl, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(f"Saved {out_jsonl}")

    print("[Done]")

if __name__ == "__main__":
    main()