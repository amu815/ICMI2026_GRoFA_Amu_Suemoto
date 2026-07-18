#!/usr/bin/env python3
"""Generate embeddings from v60 (DART on IPMix LoRA backbone).

Usage:
  python gen_v60_embeddings.py \
    --jsonl data/processed/neurips_fairface/jsonl/neurips_train.jsonl \
    --out_npz results/neurips_v60_fairface_s0/train_student.npz \
    --model_path results/neurips_v60_fairface_s0/model_best.pt \
    --lora_ckpt models/baselines_v2_fairface/IPMix/ema
"""
import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import BlipForConditionalGeneration
from peft import PeftModel
from models import NeurIPSModelV28

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_lora_backbone(lora_ckpt, lora_scale=1.0):
    """Load BLIP vision model with IPMix LoRA merged."""
    base_id = "Salesforce/blip-image-captioning-base"
    full = BlipForConditionalGeneration.from_pretrained(base_id)
    vision = full.vision_model
    vision = PeftModel.from_pretrained(vision, lora_ckpt)
    if lora_scale != 1.0:
        with torch.no_grad():
            for name, param in vision.named_parameters():
                if "lora_B" in name:
                    param.data *= lora_scale
    vision = vision.merge_and_unload()
    return vision


class EmbedDataset(Dataset):
    def __init__(self, jsonl_path, transform):
        self.data_root = Path(jsonl_path).parent.parent
        with open(jsonl_path) as f:
            self.rows = [json.loads(l) for l in f]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        img_path = self.data_root / item["image"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "race": item.get("race", "unknown"),
            "gender": item.get("gender", "unknown"),
            "age": item.get("age", "unknown"),
            "noise_type": item.get("noise_type", "clean"),
            "noise_level": int(item.get("noise_level", 0)),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_ckpt", required=True)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.481, 0.458, 0.408], std=[0.268, 0.261, 0.275]),
    ])

    # Load config
    config_path = Path(args.model_path).parent / "config.json"
    num_race, num_age = 7, 9
    arf_floor, gate_ceiling = 0.0, 1.0
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        num_race = cfg.get("num_groups", 7)
        num_age = cfg.get("num_ages", 9)
        arf_floor = cfg.get("arf_floor", 0.0)
        gate_ceiling = cfg.get("gate_ceiling", 1.0)

    # Load LoRA backbone
    print(f"[v60] Loading IPMix LoRA backbone: {args.lora_ckpt} (scale={args.lora_scale})")
    base_model = load_lora_backbone(args.lora_ckpt, args.lora_scale)

    # Build model
    model = NeurIPSModelV28(
        base_model, num_race=num_race, num_gender=2, num_age=num_age,
        arf_floor=arf_floor, gate_ceiling=gate_ceiling,
    ).to(DEVICE)

    # Load trained weights
    state = torch.load(args.model_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"[v60] Model loaded from {args.model_path}")

    # Dataset
    ds = EmbedDataset(args.jsonl, transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_emb = []
    all_gates = []
    all_race, all_gender, all_age = [], [], []
    all_nt, all_nl = [], []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Generating embeddings"):
            imgs = batch["image"].to(DEVICE)
            z_out, z_base, logits_r, logits_g, logits_a, noise_score = model(imgs)
            gate = torch.clamp(1.0 - noise_score, arf_floor, gate_ceiling)
            all_emb.append(z_out.cpu().numpy())
            all_gates.append(gate.cpu().numpy())
            all_race.extend(batch["race"])
            all_gender.extend(batch["gender"])
            all_age.extend(batch["age"])
            all_nt.extend(batch["noise_type"])
            all_nl.extend([int(x) for x in batch["noise_level"]])

    emb = np.concatenate(all_emb, axis=0)
    gates = np.concatenate(all_gates, axis=0)

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        embeddings=emb,
        gates=gates,
        race=np.array(all_race),
        gender=np.array(all_gender),
        age=np.array(all_age),
        noise_type=np.array(all_nt),
        noise_level=np.array(all_nl),
    )
    print(f"[v60] Saved {emb.shape} embeddings to {args.out_npz}")


if __name__ == "__main__":
    main()
