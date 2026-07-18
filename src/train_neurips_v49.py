#!/usr/bin/env python3
"""
LoRA fine-tuning trainer (multi-task + HSIC) used for the augmentation-based
LoRA backbones. Also provides the ArcFace / FocalLoss / GRL modules imported
by the GRoFA trainers (train_neurips_v60.py, train_neurips_v60_xback.py).

Architecture: Frozen BLIP + LoRA (r=16, 28 modules) + 3 classification heads.

7+2 losses:
  NTXent + MSE_distill + CE_race + CE_gender + CE_age + ArcFace_race + Focal_adv
  + HSIC (fairness) + MGP (gap penalty)
"""
from __future__ import annotations
import argparse, json, random, os, copy, io, math, sys
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Dataset
import torchvision.transforms as T
from PIL import Image, ImageFilter
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

from lightly.loss import NTXentLoss

# Import HSIC from neurips losses
sys.path.insert(0, str(Path(__file__).parent))
from losses import HSICLoss

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

# ---- Noise functions (same as v8) ----
def _to_np(img): return np.array(img).astype(np.float32)
def _to_img(arr): return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
def noise_gaussian(img, s):
    c=[0.08,0.12,0.18][s-1]; a=_to_np(img)/255.0
    return _to_img(np.clip(a+np.random.normal(0,c,a.shape),0,1)*255.0)
def noise_shot(img, s):
    c=[60,25,12][s-1]; a=np.clip(_to_np(img)/255.0,0,1)
    return _to_img(np.random.poisson(a*c)/float(c)*255.0)
def noise_impulse(img, s):
    c=[0.03,0.06,0.09][s-1]; a=_to_np(img)
    m=np.random.binomial(1,c,a.shape[:2]); sa=np.random.binomial(1,0.5,a.shape[:2])
    if len(a.shape)==3: m,sa=m[...,np.newaxis],sa[...,np.newaxis]
    return _to_img(a*(1-m)+(sa*255)*m)
def noise_defocus(img, s): return img.filter(ImageFilter.GaussianBlur(radius=[2,4,6][s-1]))
def noise_glass(img, s):
    sig,md,it=[(0.7,1,1),(0.9,2,1),(1.5,2,2)][s-1]; a=_to_np(img); h,w=a.shape[:2]
    for _ in range(it):
        dx,dy=np.random.randint(-md,md,size=(h,w)),np.random.randint(-md,md,size=(h,w))
        x,y=np.meshgrid(np.arange(w),np.arange(h))
        a=a[np.clip(y+dy,0,h-1),np.clip(x+dx,0,w-1)]
    return _to_img(a).filter(ImageFilter.GaussianBlur(radius=sig))
def noise_jpeg(img, s):
    buf=io.BytesIO(); img.save(buf,format='JPEG',quality=[75,40,20][s-1]); buf.seek(0)
    return Image.open(buf).copy()
def noise_cutout(img, s):
    c=[30,50,70][s-1]; a=np.array(img).copy(); h,w=a.shape[:2]
    if h>c and w>c: y,x=np.random.randint(0,h-c),np.random.randint(0,w-c); a[y:y+c,x:x+c]=0
    return Image.fromarray(a)

NOISE_FUNCS = {"gaussian":noise_gaussian,"shot":noise_shot,"impulse":noise_impulse,
               "defocus":noise_defocus,"glass":noise_glass,"jpeg":noise_jpeg,"cutout":noise_cutout}
NOISE_TYPES = list(NOISE_FUNCS.keys())

# ---- Helpers ----
class ArcFace(nn.Module):
    def __init__(self, d, n_cls, m=0.35, s=48.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, d))
        nn.init.xavier_uniform_(self.W)
        self.m, self.s = float(m), float(s)
    def forward(self, x, y):
        x = F.normalize(x); W = F.normalize(self.W)
        cos = F.linear(x, W)
        phi = torch.cos(torch.acos(cos.clamp(-0.9999, 0.9999)) + self.m)
        oh = F.one_hot(y, self.W.shape[0])
        logits = (oh * phi) + ((1.0 - oh) * cos)
        return F.cross_entropy(self.s * logits, y)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0): super().__init__(); self.g = gamma
    def forward(self, logits, tgt):
        ce = F.cross_entropy(logits, tgt, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt)**self.g * ce).mean()

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None

AUG_PIL = T.Compose([
    T.RandomResizedCrop(224, scale=(0.3, 1.0)),
    T.ColorJitter(0.8, 0.8, 0.8, 0.4),
    T.RandomGrayscale(0.5),
    T.RandomHorizontalFlip(),
])

# ---- Data ----
def load_clean_entries(jsonl_path, data_root):
    entries = []
    race_set, gender_set, age_set = set(), set(), set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["noise_type"] != "clean":
                continue
            race = rec.get("race", "Unknown")
            gender = rec.get("gender", "unknown")
            age = rec.get("age", "unknown")
            race_set.add(race); gender_set.add(gender); age_set.add(age)
            entries.append({
                "image": str(Path(data_root) / rec["image"]),
                "race": race, "gender": gender, "age": age,
            })
    r2id = {r: i for i, r in enumerate(sorted(race_set))}
    g2id = {g: i for i, g in enumerate(sorted(gender_set))}
    a2id = {a: i for i, a in enumerate(sorted(age_set))}
    for e in entries:
        e["race_id"] = r2id[e["race"]]
        e["gender_id"] = g2id[e["gender"]]
        e["age_id"] = a2id[e["age"]]
    print(f"Loaded {len(entries)} clean entries.")
    print(f"  Races({len(r2id)}): {r2id}")
    print(f"  Genders({len(g2id)}): {g2id}")
    print(f"  Ages({len(a2id)}): {a2id}")
    return entries, r2id, g2id, a2id


class FRoLAv49Dataset(Dataset):
    def __init__(self, entries, proc):
        self.entries, self.proc = entries, proc
    def __len__(self): return len(self.entries)
    def __getitem__(self, i):
        rec = self.entries[i]
        try: img = Image.open(rec["image"]).convert("RGB")
        except: img = Image.new("RGB", (224, 224))
        noisy = AUG_PIL(NOISE_FUNCS[random.choice(NOISE_TYPES)](img, random.choice([1,2,3])))
        clean = AUG_PIL(img)
        px_c = self.proc(images=clean, return_tensors="pt")["pixel_values"][0]
        px_n = self.proc(images=noisy, return_tensors="pt")["pixel_values"][0]
        return px_c, px_n, rec["race_id"], rec["gender_id"], rec["age_id"]


class V49ValDataset(Dataset):
    """Validation dataset: loads pre-generated noisy images from JSONL."""
    def __init__(self, jsonl_path, data_root, transform):
        self.data_root = Path(data_root)
        self.transform = transform
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        item = self.rows[idx]
        img_path = self.data_root / item["image"]
        try: image = Image.open(img_path).convert("RGB")
        except: image = Image.new("RGB", (224, 224))
        image = self.transform(image)
        return image, item.get("race", "Unknown"), item.get("gender", "unknown"), item.get("age", "unknown")


class BalancedBatchSampler(Sampler):
    def __init__(self, entries, batch_size):
        self.batch_size = batch_size
        self.indices_by_race = {}
        for i, e in enumerate(entries):
            self.indices_by_race.setdefault(e["race_id"], []).append(i)
        self.races = list(self.indices_by_race.keys())
        self.n_batches = max(1, len(entries) // batch_size)
    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            n_per = max(1, self.batch_size // len(self.races))
            for r in self.races:
                batch.extend(random.choices(self.indices_by_race[r], k=n_per))
            batch = batch[:self.batch_size]
            while len(batch) < self.batch_size:
                batch.append(random.choice(batch))
            random.shuffle(batch)
            yield batch
    def __len__(self): return self.n_batches


# ---- Main ----
def main():
    ap = argparse.ArgumentParser()
    # v8 inherited
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4.28e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--w_nce", type=float, default=0.5)
    ap.add_argument("--w_mse", type=float, default=0.3)
    ap.add_argument("--w_ce", type=float, default=1.0)
    ap.add_argument("--w_ce_age_scale", type=float, default=1.0)
    ap.add_argument("--w_arc", type=float, default=0.5)
    ap.add_argument("--w_arc_gender", type=float, default=0.0)
    ap.add_argument("--w_adv", type=float, default=0.2)
    # v49 new
    ap.add_argument("--val_jsonl", type=str, default=None, help="Validation JSONL for early stopping + MGP")
    ap.add_argument("--lambda_hsic", type=float, default=0.0, help="HSIC fairness weight")
    ap.add_argument("--lambda_mgp", type=float, default=0.0, help="MGP penalty weight")
    ap.add_argument("--eps_race", type=float, default=0.04)
    ap.add_argument("--eps_gender", type=float, default=0.15)
    ap.add_argument("--eps_age", type=float, default=0.12)
    ap.add_argument("--mgp_warmup", type=int, default=3, help="Epochs before activating MGP")
    ap.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Data
    entries, r2id, g2id, a2id = load_clean_entries(args.jsonl, args.data_root)
    n_races, n_genders, n_ages = len(r2id), len(g2id), len(a2id)
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    sampler = BalancedBatchSampler(entries, args.batch)
    loader = DataLoader(
        FRoLAv49Dataset(entries, proc),
        batch_sampler=sampler, num_workers=4, pin_memory=True,
    )

    # Validation data (for MGP gap estimation + early stopping)
    val_loader = None
    if args.val_jsonl:
        val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.481, 0.458, 0.408], std=[0.268, 0.261, 0.275]),
        ])
        val_ds = V49ValDataset(args.val_jsonl, args.data_root, val_transform)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                                num_workers=4, pin_memory=True)

    # Model (LoRA)
    vision = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).vision_model.to(DEVICE)
    vision.gradient_checkpointing_enable()

    t_mods = []
    for i in [0, 1, 2, 8, 9, 10, 11]:
        t_mods.append(f"encoder.layers.{i}.self_attn.qkv")
        t_mods.append(f"encoder.layers.{i}.self_attn.projection")
        t_mods.append(f"encoder.layers.{i}.mlp.fc1")
        t_mods.append(f"encoder.layers.{i}.mlp.fc2")

    lcfg = LoraConfig(r=args.rank, target_modules=t_mods, lora_alpha=args.rank * 2,
                       lora_dropout=0.05, bias="none")
    model = get_peft_model(vision, lcfg).to(DEVICE).train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Classification heads + ArcFace + Domain head
    H = 768
    head_race = nn.Linear(H, n_races).to(DEVICE)
    head_gender = nn.Linear(H, n_genders).to(DEVICE)
    head_age = nn.Linear(H, n_ages).to(DEVICE)
    arcface_race = ArcFace(H, n_races).to(DEVICE)
    arcface_gender = ArcFace(H, n_genders, m=0.35, s=48.0).to(DEVICE) if args.w_arc_gender > 0 else None
    dom_head = nn.Linear(H, 2).to(DEVICE)

    all_params = (list(model.parameters()) +
                  list(head_race.parameters()) + list(head_gender.parameters()) +
                  list(head_age.parameters()) + list(arcface_race.parameters()) +
                  (list(arcface_gender.parameters()) if arcface_gender else []) +
                  list(dom_head.parameters()))
    opt = torch.optim.AdamW(all_params, lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    total_steps = args.epochs * len(loader)
    warmup_steps = args.warmup_epochs * len(loader)
    def lr_lambda(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ntxent = NTXentLoss(temperature=0.07)
    mse_fn = nn.MSELoss()
    focal = FocalLoss()
    crit_hsic = HSICLoss() if args.lambda_hsic > 0 else None

    teacher = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).vision_model.to(DEVICE).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters(): p.requires_grad_(False)

    # State
    current_gaps = None
    best_val_metric = -1.0
    patience_counter = 0
    history = []

    print(f"--- v49 FRoLA Multi-Task + Fairness ---")
    print(f"Losses: NTXent({args.w_nce}) + MSE({args.w_mse}) + CE×3({args.w_ce}, age_scale={args.w_ce_age_scale})"
          f" + ArcFace_R({args.w_arc}) + ADV({args.w_adv})")
    print(f"Fairness: HSIC(λ={args.lambda_hsic}) + MGP(λ={args.lambda_mgp},"
          f" ε=[{args.eps_race},{args.eps_gender},{args.eps_age}], warmup={args.mgp_warmup})")
    print(f"Tasks: race({n_races}) + gender({n_genders}) + age({n_ages})")
    print(f"Seed: {args.seed}, Patience: {args.patience}")

    nan_count = 0
    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Ep{ep}")
        ep_loss, ep_ce, ep_hsic, n = 0.0, 0.0, 0.0, 0

        for px_c, px_n, race, gender, age in pbar:
            px_c, px_n = px_c.to(DEVICE), px_n.to(DEVICE)
            race, gender, age = race.to(DEVICE), gender.to(DEVICE), age.to(DEVICE)

            with torch.amp.autocast('cuda') if USE_AMP else nullcontext():
                feat = model(torch.cat([px_c, px_n])).last_hidden_state[:, 1:]
                f_c, f_n = feat.chunk(2)
                z_c, z_n = f_c.mean(1), f_n.mean(1)

                with torch.no_grad():
                    t_c = teacher(px_c).last_hidden_state[:, 1:].mean(1)

                # 1. NTXent
                l_nce = ntxent(z_c, z_n)
                # 2. MSE distillation
                l_mse = (mse_fn(z_c, t_c) + mse_fn(z_n, t_c)) / 2
                # 3-5. Multi-task classification
                z_both = torch.cat([z_c, z_n])
                labels_r = torch.cat([race, race])
                labels_g = torch.cat([gender, gender])
                labels_a = torch.cat([age, age])

                l_ce_r = F.cross_entropy(head_race(z_both), labels_r)
                l_ce_g = F.cross_entropy(head_gender(z_both), labels_g)
                l_ce_a = F.cross_entropy(head_age(z_both), labels_a)
                # 6. ArcFace
                l_arc_r = arcface_race(z_n, race)
                l_arc_g = arcface_gender(z_n, gender) if arcface_gender else torch.tensor(0.0, device=DEVICE)
                # 7. Adversarial domain
                dom_labels = torch.cat([torch.zeros(len(z_c)), torch.ones(len(z_n))]).long().to(DEVICE)
                l_adv = focal(dom_head(GRL.apply(torch.cat([z_c, z_n]), 0.6)), dom_labels)

                # v8 total
                loss_v8 = (args.w_nce * l_nce +
                           args.w_mse * l_mse +
                           args.w_ce * (l_ce_r + l_ce_g + args.w_ce_age_scale * l_ce_a) +
                           args.w_arc * l_arc_r +
                           args.w_arc_gender * l_arc_g +
                           args.w_adv * l_adv)

                # v49: HSIC fairness
                l_hsic = torch.tensor(0.0, device=DEVICE)
                if crit_hsic is not None:
                    l_hsic = crit_hsic(z_both, labels_r) + crit_hsic(z_both, labels_g)

                # v49: MGP (Max-Gap Penalty)
                l_mgp = torch.tensor(0.0, device=DEVICE)
                if args.lambda_mgp > 0 and current_gaps is not None and ep >= args.mgp_warmup:
                    epsilon_t = torch.tensor(
                        [args.eps_race, args.eps_gender, args.eps_age],
                        device=DEVICE
                    )
                    violations = (current_gaps - epsilon_t).clamp(min=0.0)
                    l_mgp = args.lambda_mgp * violations.max()

                loss = loss_v8 + args.lambda_hsic * l_hsic + l_mgp

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count > 20:
                    print(f"[ABORT] NaN={nan_count}")
                    break
                opt.zero_grad(); continue

            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                opt.step()
            scheduler.step()

            # EMA update
            with torch.no_grad():
                for (k1, p1), (k2, p2) in zip(model.named_parameters(), ema_model.named_parameters()):
                    if k1 == k2: p2.data.mul_(0.99).add_(p1.data, alpha=0.01)

            ep_loss += loss.item()
            ep_ce += (l_ce_r + l_ce_g + l_ce_a).item()
            ep_hsic += l_hsic.item()
            n += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}",
                              "ce": f"{(l_ce_r+l_ce_g+l_ce_a).item():.3f}",
                              "hsic": f"{l_hsic.item():.4f}",
                              "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if nan_count > 20: break
        avg_loss = ep_loss / max(n, 1)
        avg_hsic = ep_hsic / max(n, 1)
        mgp_str = f" mgp={l_mgp.item():.4f}" if args.lambda_mgp > 0 and l_mgp.item() > 0 else ""
        print(f"  Ep{ep}  loss={avg_loss:.4f}  ce={ep_ce/max(n,1):.4f}"
              f"  hsic={avg_hsic:.4f}{mgp_str}  nan={nan_count}")

        # ==================================
        # Validation (for MGP + early stopping)
        # ==================================
        val_metric = avg_loss  # fallback: use train loss
        if val_loader is not None:
            model.eval()
            val_cls_correct = {"race": {}, "gender": {}, "age": {}}
            val_cls_count = {}

            with torch.no_grad():
                for img_v, race_v, gender_v, age_v in val_loader:
                    img_v = img_v.to(DEVICE)
                    out_v = model(img_v).last_hidden_state[:, 1:].mean(1)
                    lr_v = head_race(out_v)
                    lg_v = head_gender(out_v)
                    la_v = head_age(out_v)

                    # Map string labels to IDs
                    for i_s in range(len(race_v)):
                        r_str, g_str, a_str = race_v[i_s], gender_v[i_s], age_v[i_s]
                        r_id = r2id.get(r_str, -1)
                        g_id = g2id.get(g_str, -1)
                        a_id = a2id.get(a_str, -1)
                        if r_id < 0: continue

                        val_cls_count[r_id] = val_cls_count.get(r_id, 0) + 1
                        for task, logits, label_id in [
                            ("race", lr_v, r_id), ("gender", lg_v, g_id), ("age", la_v, a_id)
                        ]:
                            if label_id < 0: continue
                            pred = logits[i_s].argmax().item()
                            val_cls_correct[task].setdefault(r_id, 0)
                            if pred == label_id:
                                val_cls_correct[task][r_id] = val_cls_correct[task].get(r_id, 0) + 1

            # Per-group accuracy + gaps
            groups = sorted(val_cls_count.keys())
            if len(groups) > 0:
                race_acc_pg = np.array([val_cls_correct["race"].get(g, 0) / max(val_cls_count.get(g, 1), 1) for g in groups])
                gender_acc_pg = np.array([val_cls_correct["gender"].get(g, 0) / max(val_cls_count.get(g, 1), 1) for g in groups])
                age_acc_pg = np.array([val_cls_correct["age"].get(g, 0) / max(val_cls_count.get(g, 1), 1) for g in groups])

                gap_race = float(race_acc_pg.max() - race_acc_pg.min())
                gap_gender = float(gender_acc_pg.max() - gender_acc_pg.min())
                gap_age = float(age_acc_pg.max() - age_acc_pg.min())

                current_gaps = torch.tensor([gap_race, gap_gender, gap_age],
                                             dtype=torch.float32, device=DEVICE)

                total_n = sum(val_cls_count.values())
                acc_r = sum(val_cls_correct["race"].values()) / max(total_n, 1)
                acc_g = sum(val_cls_correct["gender"].values()) / max(total_n, 1)
                acc_a = sum(val_cls_correct["age"].values()) / max(total_n, 1)
                val_metric = (acc_r + acc_g + acc_a) / 3.0

                print(f"  Val ClsAcc: race={acc_r:.3f} gender={acc_g:.3f} age={acc_a:.3f} | Mean={val_metric:.3f}")
                print(f"  Val Gaps: race={gap_race:.3f} gender={gap_gender:.3f} age={gap_age:.3f}")

                if args.lambda_mgp > 0:
                    epsilon_t = torch.tensor([args.eps_race, args.eps_gender, args.eps_age], device=DEVICE)
                    violations = (current_gaps - epsilon_t).clamp(min=0.0)
                    if ep >= args.mgp_warmup:
                        print(f"  [MGP] max_violation={violations.max().item():.4f}")
                    else:
                        print(f"  [MGP] warmup ({ep+1}/{args.mgp_warmup})")

        # Early stopping
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            # Save best checkpoint
            args.out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out_dir / "best_online")
            ema_model.save_pretrained(args.out_dir / "best_ema")
            torch.save({
                "head_race": head_race.state_dict(),
                "head_gender": head_gender.state_dict(),
                "head_age": head_age.state_dict(),
                "arcface_race": arcface_race.state_dict(),
                "dom_head": dom_head.state_dict(),
            }, args.out_dir / "best_heads.pt")
            print(f"  -> Best model saved (metric={val_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  -> Early stopping at epoch {ep} (patience={args.patience})")
                break

        history.append({
            "epoch": ep, "train_loss": avg_loss, "val_metric": val_metric,
            "hsic": avg_hsic, "mgp": l_mgp.item() if args.lambda_mgp > 0 else 0.0,
            "gaps": [gap_race, gap_gender, gap_age] if current_gaps is not None else None,
        })

    # Save final model
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir / "online")
    ema_model.save_pretrained(args.out_dir / "ema")
    torch.save({
        "head_race": head_race.state_dict(),
        "head_gender": head_gender.state_dict(),
        "head_age": head_age.state_dict(),
        "arcface_race": arcface_race.state_dict(),
        "dom_head": dom_head.state_dict(),
    }, args.out_dir / "heads.pt")

    config = {
        "version": "v49_multitask_fairness",
        "n_races": n_races, "n_genders": n_genders, "n_ages": n_ages,
        "epochs": args.epochs, "actual_epochs": ep + 1,
        "batch_size": args.batch, "lr": args.lr, "rank": args.rank,
        "w_nce": args.w_nce, "w_mse": args.w_mse, "w_ce": args.w_ce,
        "w_ce_age_scale": args.w_ce_age_scale,
        "w_arc": args.w_arc, "w_arc_gender": args.w_arc_gender, "w_adv": args.w_adv,
        "lambda_hsic": args.lambda_hsic, "lambda_mgp": args.lambda_mgp,
        "eps_race": args.eps_race, "eps_gender": args.eps_gender, "eps_age": args.eps_age,
        "mgp_warmup": args.mgp_warmup,
        "seed": args.seed, "nan_count": nan_count,
        "best_val_metric": best_val_metric,
        "history": history,
    }
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved FRoLA v49 to {args.out_dir}")
    print(f"Best val metric: {best_val_metric:.4f}")


if __name__ == "__main__":
    main()
