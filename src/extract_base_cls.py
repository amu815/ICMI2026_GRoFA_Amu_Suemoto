#!/usr/bin/env python3
"""Extract frozen backbone CLS embeddings (no adapter) for CLIP/DINOv2/BLIP.

Outputs base_train.npz / base_test.npz matching the schema used by
evaluate_neurips_v2.py and the debiaSAE pipeline.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPECS = {
    "clip":   {"hf_id": "openai/clip-vit-base-patch16", "norm_mean": [0.48145466, 0.4578275, 0.40821073], "norm_std": [0.26862954, 0.26130258, 0.27577711]},
    "dinov2": {"hf_id": "facebook/dinov2-base", "norm_mean": [0.485, 0.456, 0.406], "norm_std": [0.229, 0.224, 0.225]},
    "blip":   {"hf_id": "Salesforce/blip-image-captioning-base", "norm_mean": [0.48145466, 0.4578275, 0.40821073], "norm_std": [0.26862954, 0.26130258, 0.27577711]},
}


def load_backbone(name):
    from transformers import AutoModel, CLIPVisionModel
    if name == "clip":
        m = CLIPVisionModel.from_pretrained(SPECS["clip"]["hf_id"]).to(DEVICE).eval()
        def forward(imgs):
            return m(pixel_values=imgs).pooler_output  # (B, 768)
        return forward
    if name == "dinov2":
        m = AutoModel.from_pretrained(SPECS["dinov2"]["hf_id"]).to(DEVICE).eval()
        def forward(imgs):
            return m(pixel_values=imgs).last_hidden_state[:, 0, :]  # CLS token (B, 768)
        return forward
    if name == "blip":
        from transformers import BlipModel
        m = BlipModel.from_pretrained(SPECS["blip"]["hf_id"]).to(DEVICE).eval()
        def forward(imgs):
            return m.vision_model(pixel_values=imgs).pooler_output
        return forward
    raise ValueError(name)


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
        r = self.rows[idx]
        img = Image.open(self.data_root / r["image"]).convert("RGB")
        return {
            "image": self.transform(img),
            "race": r.get("race", "unknown"),
            "gender": r.get("gender", "unknown"),
            "age": r.get("age", "unknown"),
            "noise_type": r.get("noise_type", "clean"),
            "noise_level": int(r.get("noise_level", 0)),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=["clip", "dinov2", "blip"])
    p.add_argument("--jsonl_train", required=True)
    p.add_argument("--jsonl_test", required=True)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    print(f"[extract] backbone={args.backbone}")
    forward = load_backbone(args.backbone)
    spec = SPECS[args.backbone]
    tfm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=spec["norm_mean"], std=spec["norm_std"]),
    ])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, jsonl in [("train", args.jsonl_train), ("test", args.jsonl_test)]:
        ds = JsonlDataset(jsonl, tfm)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        all_emb, all_r, all_g, all_a, all_nt, all_nl = [], [], [], [], [], []
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"{args.backbone}-{split}"):
                imgs = batch["image"].to(DEVICE)
                z = forward(imgs)
                all_emb.append(z.cpu().numpy())
                all_r.extend(batch["race"])
                all_g.extend(batch["gender"])
                all_a.extend(batch["age"])
                all_nt.extend(batch["noise_type"])
                all_nl.extend([int(x) for x in batch["noise_level"]])
        emb = np.concatenate(all_emb, axis=0)
        meta = dict(race=np.array(all_r), gender=np.array(all_g), age=np.array(all_a),
                    noise_type=np.array(all_nt), noise_level=np.array(all_nl))
        path = args.out_dir / f"base_{split}.npz"
        np.savez_compressed(path, embeddings=emb, **meta)
        print(f"Saved: {path} ({emb.shape})")


if __name__ == "__main__":
    main()
