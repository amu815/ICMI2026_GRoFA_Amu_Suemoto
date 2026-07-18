#!/usr/bin/env python3
"""Cross-backbone embedding generator for DART (v60_xback trained models).

Reads config.json next to the checkpoint to pick backbone (blip/clip/dinov2).
Emits train_student.npz and test_student.npz with the same schema as
generate_embeddings_unified.py so the downstream ensemble/WiSE-FT scripts work.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "neurips" / "src3"))

from models import NeurIPSModelV28
from train_neurips_v60_xback import BACKBONE_SPECS, load_backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class JsonlDataset(Dataset):
    def __init__(self, jsonl_path, transform):
        self.jsonl_path = Path(jsonl_path)
        self.data_root = self.jsonl_path.parent.parent
        with open(jsonl_path) as f:
            self.rows = [json.loads(l) for l in f]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        img_path = self.data_root / item["image"]
        if not img_path.exists():
            raise FileNotFoundError(f"missing image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return {
            "image": img,
            "race": item.get("race", "unknown"),
            "gender": item.get("gender", "unknown"),
            "age": item.get("age", "unknown"),
            "noise_type": item.get("noise_type", "clean"),
            "noise_level": int(item.get("noise_level", 0)),
        }


def get_transforms(backbone: str):
    spec = BACKBONE_SPECS[backbone.lower()]
    return T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=spec["norm_mean"], std=spec["norm_std"]),
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, type=Path,
                   help="Directory containing model_best.pt and config.json")
    p.add_argument("--jsonl_train", required=True)
    p.add_argument("--jsonl_test", required=True)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--backbone", type=str, default=None,
                   help="Override backbone name (else read from config.json)")
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    cfg_path = args.model_dir / "config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    backbone = (args.backbone or cfg.get("backbone", "blip")).lower()
    lora_ckpt = cfg.get("lora_ckpt")
    num_race = cfg.get("num_groups", 7)
    num_age = cfg.get("num_ages", 9)
    arf_floor = cfg.get("arf_floor", 0.0)
    gate_ceiling = cfg.get("gate_ceiling", 1.0)
    print(f"[xback-eval] backbone={backbone} num_race={num_race} num_age={num_age}")

    base_model = load_backbone(backbone, lora_ckpt)
    model = NeurIPSModelV28(
        base_model, num_race=num_race, num_gender=2, num_age=num_age,
        arf_floor=arf_floor, gate_ceiling=gate_ceiling,
    ).to(DEVICE)
    ckpt = torch.load(args.model_dir / "model_best.pt", map_location=DEVICE, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    tfm = get_transforms(backbone)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, jsonl in [("train", args.jsonl_train), ("test", args.jsonl_test)]:
        ds = JsonlDataset(jsonl, tfm)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        all_emb, all_base, all_gates, all_r, all_g, all_a, all_nt, all_nl = [], [], [], [], [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"xback-{split}"):
                imgs = batch["image"].to(DEVICE)
                z_out, z_base, lr, lg, la, noise_score = model(imgs)
                gate = torch.clamp(1.0 - noise_score, arf_floor, gate_ceiling)
                all_emb.append(z_out.cpu().numpy())
                all_base.append(z_base.cpu().numpy())
                all_gates.append(gate.cpu().numpy())
                all_r.extend(batch["race"])
                all_g.extend(batch["gender"])
                all_a.extend(batch["age"])
                all_nt.extend(batch["noise_type"])
                all_nl.extend([int(x) for x in batch["noise_level"]])
        emb = np.concatenate(all_emb, axis=0)
        base_emb = np.concatenate(all_base, axis=0)
        gates = np.concatenate(all_gates, axis=0)
        meta = dict(race=np.array(all_r), gender=np.array(all_g), age=np.array(all_a),
                    noise_type=np.array(all_nt), noise_level=np.array(all_nl))
        out_path = args.out_dir / f"{split}_student.npz"
        np.savez_compressed(out_path, embeddings=emb, gates=gates, **meta)
        base_path = args.out_dir / f"base_{split}.npz"
        np.savez_compressed(base_path, embeddings=base_emb, **meta)
        print(f"Saved: {out_path} ({emb.shape}), {base_path} ({base_emb.shape})")


if __name__ == "__main__":
    main()
