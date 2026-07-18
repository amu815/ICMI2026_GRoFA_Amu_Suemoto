#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src3/evaluate_protocol.py
GRoFA evaluation script (fixed 66-condition protocol)

Key fixes over the legacy evaluation script:
  1. LogReg trains on CLEAN embeddings only (not all 88k mixed)
  2. Per-seed train/test sub-split (80/20 stratified) -> non-zero std
  3. Fairness metrics: Accuracy Gap, Demographic Parity Difference, F1-macro
  4. Robust AUC computation for class-imbalanced conditions
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ==========================================
# Label Mappings
# ==========================================
RACE_ID_TO_STR = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
GENDER_ID_TO_STR = {0: "Male", 1: "Female"}
AGE_ID_TO_STR = {
    0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39",
    5: "40-49", 6: "50-59", 7: "60-69", 8: "70+",
}


# ==========================================
# Utilities
# ==========================================
def clean_label(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, (float, np.floating)) and np.isnan(raw):
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "unknown":
        return None
    return s


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load NPZ and return dict with embeddings + metadata."""
    print(f"[Info] Loading {path} ...")
    data = np.load(path, allow_pickle=True)
    out = {
        "X": data["embeddings"],
        "noise_type": np.array([str(x) for x in data["noise_type"]]),
        "noise_level": data["noise_level"],
    }
    if "gates" in data:
        out["gates"] = data["gates"]

    for task_key in ["race", "gender", "age"]:
        if task_key in data:
            out[task_key] = data[task_key]

    return out


def get_task_labels(data: Dict, task: str) -> Tuple[List[str], np.ndarray]:
    """Extract cleaned string labels and valid indices for a task."""
    raw = data[task]
    labels: List[str] = []
    keep_idx: List[int] = []
    for i, lbl in enumerate(raw):
        c = clean_label(lbl)
        if c is not None:
            labels.append(c)
            keep_idx.append(i)
    return labels, np.array(keep_idx, dtype=int)


def get_id_to_str(task: str) -> Dict[int, str]:
    if task == "race":
        return RACE_ID_TO_STR
    elif task == "gender":
        return GENDER_ID_TO_STR
    elif task == "age":
        return AGE_ID_TO_STR
    return {}


# ==========================================
# Metrics
# ==========================================
def compute_robust_auc(
    y_true: np.ndarray, proba: np.ndarray, le: LabelEncoder
) -> Optional[float]:
    n_classes = len(le.classes_)
    if len(np.unique(y_true)) < 2:
        return None

    if n_classes == 2:
        try:
            return roc_auc_score(y_true, proba[:, 1])
        except ValueError:
            return None
    else:
        try:
            return roc_auc_score(
                y_true, proba, multi_class="ovr",
                labels=np.arange(n_classes),
            )
        except ValueError:
            # Robust fallback: per-class OvR average
            aucs = []
            for cls_idx in np.unique(y_true):
                y_bin = (y_true == cls_idx).astype(int)
                if len(np.unique(y_bin)) == 2:
                    try:
                        aucs.append(roc_auc_score(y_bin, proba[:, cls_idx]))
                    except (ValueError, IndexError):
                        pass
            return float(np.mean(aucs)) if aucs else None


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    task: str,
) -> Dict[str, float]:
    """Compute fairness metrics across demographic groups."""
    metrics = {}
    unique_groups = np.unique(group_labels)

    if len(unique_groups) < 2:
        return {"accuracy_gap": 0.0, "dp_diff": 0.0}

    # Per-group accuracy
    group_accs = {}
    for g in unique_groups:
        mask = group_labels == g
        if mask.sum() > 0:
            group_accs[g] = accuracy_score(y_true[mask], y_pred[mask])

    if group_accs:
        metrics["accuracy_gap"] = max(group_accs.values()) - min(group_accs.values())
        metrics["worst_group_acc"] = min(group_accs.values())
        metrics["best_group_acc"] = max(group_accs.values())

    # Demographic Parity Difference (for binary tasks like gender)
    if len(np.unique(y_pred)) == 2:
        positive_rates = {}
        for g in unique_groups:
            mask = group_labels == g
            if mask.sum() > 0:
                positive_rates[g] = np.mean(y_pred[mask] == 1)
        if len(positive_rates) >= 2:
            rates = list(positive_rates.values())
            metrics["dp_diff"] = max(rates) - min(rates)
    else:
        metrics["dp_diff"] = np.nan

    # Equalized Odds Difference (per-class TPR gap)
    if len(np.unique(y_true)) >= 2 and len(group_accs) >= 2:
        tpr_gaps = []
        for cls in np.unique(y_true):
            cls_tprs = {}
            for g in unique_groups:
                g_mask = group_labels == g
                cls_mask = y_true == cls
                combined = g_mask & cls_mask
                if combined.sum() > 0:
                    cls_tprs[g] = accuracy_score(y_true[combined], y_pred[combined])
            if len(cls_tprs) >= 2:
                tpr_gaps.append(max(cls_tprs.values()) - min(cls_tprs.values()))
        if tpr_gaps:
            metrics["eq_odds_diff"] = float(np.mean(tpr_gaps))

    return metrics


# ==========================================
# Core Evaluation
# ==========================================
def evaluate_model(
    train_data: Dict,
    test_data: Dict,
    model_name: str,
    tasks: List[str],
    n_seeds: int,
    train_ratio: float,
) -> List[Dict]:
    """
    Train LogReg on CLEAN embeddings only, evaluate per noise condition.
    """
    # --- Filter CLEAN train embeddings ---
    clean_mask_train = train_data["noise_type"] == "clean"
    X_train_clean = train_data["X"][clean_mask_train]
    n_clean = clean_mask_train.sum()
    print(f"  [{model_name}] Clean train samples: {n_clean} / {len(train_data['X'])}")

    # Test data: identify unique (noise_type, noise_level) conditions
    test_nt = test_data["noise_type"]
    test_nl = test_data["noise_level"]
    conditions = sorted(
        set(zip(test_nt, test_nl)),
        key=lambda x: (x[0] != "clean", str(x[0]), int(x[1])),
    )

    all_rows = []

    for task in tasks:
        if task not in train_data or task not in test_data:
            continue

        # Get clean train labels
        train_labels_all = train_data[task]
        train_labels_clean = train_labels_all[clean_mask_train]

        # Clean and encode labels
        labels_str: List[str] = []
        valid_idx: List[int] = []
        for i, lbl in enumerate(train_labels_clean):
            c = clean_label(lbl)
            if c is not None:
                labels_str.append(c)
                valid_idx.append(i)
        valid_idx_arr = np.array(valid_idx, dtype=int)
        X_clean_valid = X_train_clean[valid_idx_arr]

        if len(labels_str) == 0:
            print(f"  [Warning] No valid labels for task={task}")
            continue

        le = LabelEncoder()
        y_clean_enc = le.fit_transform(labels_str)

        # Determine sensitive attribute for fairness metrics
        # Use race as the sensitive attribute for gender/age tasks, gender for race task
        if task == "race":
            sensitive_key = "gender"
        else:
            sensitive_key = "race"

        for seed in range(n_seeds):
            # --- Per-seed sub-split ---
            if train_ratio >= 1.0:
                tr_idx = np.arange(len(y_clean_enc))
            else:
                try:
                    tr_idx, _ = train_test_split(
                        np.arange(len(y_clean_enc)),
                        test_size=1.0 - train_ratio,
                        stratify=y_clean_enc,
                        random_state=seed,
                    )
                except ValueError:
                    tr_idx, _ = train_test_split(
                        np.arange(len(y_clean_enc)),
                        test_size=1.0 - train_ratio,
                        random_state=seed,
                    )

            # Train LogReg on clean sub-split
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000, C=0.1, solver="liblinear",
                    multi_class="auto", random_state=seed,
                ),
            )
            clf.fit(X_clean_valid[tr_idx], y_clean_enc[tr_idx])

            # --- Evaluate per noise condition ---
            for nt, nl in conditions:
                cond_mask = (test_nt == nt) & (test_nl == nl)
                if cond_mask.sum() == 0:
                    continue

                X_test_cond = test_data["X"][cond_mask]
                test_labels_cond = test_data[task][cond_mask]

                # Clean test labels and filter
                test_str: List[str] = []
                test_valid: List[int] = []
                for i, lbl in enumerate(test_labels_cond):
                    c = clean_label(lbl)
                    if c is not None and c in set(le.classes_):
                        test_str.append(c)
                        test_valid.append(i)

                if len(test_str) == 0:
                    continue

                test_valid_arr = np.array(test_valid, dtype=int)
                X_sub = X_test_cond[test_valid_arr]
                y_sub = le.transform(test_str)

                # Predictions
                preds = clf.predict(X_sub)
                proba = clf.predict_proba(X_sub)

                # Core metrics
                acc = accuracy_score(y_sub, preds)
                f1 = f1_score(y_sub, preds, average="macro")
                auc = compute_robust_auc(y_sub, proba, le)

                # Per-class accuracy
                class_accs = {}
                for cls_idx in range(len(le.classes_)):
                    cls_mask = y_sub == cls_idx
                    if cls_mask.sum() > 0:
                        cls_name = le.classes_[cls_idx]
                        class_accs[f"Acc_{cls_name}"] = accuracy_score(
                            y_sub[cls_mask], preds[cls_mask]
                        )

                # Fairness metrics
                fairness = {}
                if sensitive_key in test_data:
                    sens_labels = test_data[sensitive_key][cond_mask][test_valid_arr]
                    sens_clean = np.array([
                        str(s) for s in sens_labels
                    ])
                    fairness = compute_fairness_metrics(
                        y_sub, preds, sens_clean, task
                    )

                row = {
                    "Model": model_name,
                    "Task": task,
                    "Noise_Type": nt,
                    "Noise_Level": int(nl),
                    "Seed": seed,
                    "N_samples": len(y_sub),
                    "Acc": acc,
                    "F1_macro": f1,
                    "AUC": auc,
                    **class_accs,
                    **fairness,
                }
                all_rows.append(row)

    return all_rows


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="GRoFA evaluation (fixed 66-condition protocol)"
    )
    parser.add_argument("--base_train_npz", required=True)
    parser.add_argument("--base_test_npz", required=True)
    parser.add_argument("--student_train_npz", required=True)
    parser.add_argument("--student_test_npz", required=True)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    d_base_train = load_npz(args.base_train_npz)
    d_base_test = load_npz(args.base_test_npz)
    d_stud_train = load_npz(args.student_train_npz)
    d_stud_test = load_npz(args.student_test_npz)

    # Detect available tasks
    tasks = [t for t in ["race", "gender", "age"] if t in d_base_train]
    print(f"[Info] Detected tasks: {tasks}")

    # 2. Evaluate Baseline
    print("\n=== Evaluating Baseline (BLIP CLS) ===")
    rows_base = evaluate_model(
        d_base_train, d_base_test, "Baseline", tasks,
        args.n_seeds, args.train_ratio,
    )

    # 3. Evaluate Ours (Student)
    print("\n=== Evaluating Ours (Student) ===")
    rows_stud = evaluate_model(
        d_stud_train, d_stud_test, "Ours", tasks,
        args.n_seeds, args.train_ratio,
    )

    # 4. Combine and save
    df = pd.DataFrame(rows_base + rows_stud)

    # --- Summary by condition (mean ± std across seeds) ---
    grp_cols = ["Model", "Task", "Noise_Type", "Noise_Level"]
    metric_cols = ["Acc", "F1_macro", "AUC"]

    # Only aggregate numeric columns that exist
    agg_dict = {}
    for col in metric_cols:
        if col in df.columns:
            agg_dict[f"{col}_mean"] = (col, "mean")
            agg_dict[f"{col}_std"] = (col, "std")

    # Fairness columns
    for fcol in ["accuracy_gap", "dp_diff", "worst_group_acc", "best_group_acc", "eq_odds_diff"]:
        if fcol in df.columns:
            agg_dict[f"{fcol}_mean"] = (fcol, "mean")
            agg_dict[f"{fcol}_std"] = (fcol, "std")

    summary = df.groupby(grp_cols).agg(**agg_dict).reset_index()
    summary.to_csv(
        args.out_dir / "summary_by_condition.csv", index=False, float_format="%.4f"
    )
    print(f"\n[Saved] {args.out_dir / 'summary_by_condition.csv'}")

    # --- Fairness metrics ---
    fairness_cols = [c for c in df.columns if c in ["accuracy_gap", "dp_diff"]]
    if fairness_cols:
        # Per-class accuracy columns
        class_acc_cols = [c for c in df.columns if c.startswith("Acc_")]
        fairness_detail_cols = grp_cols + ["Seed"] + fairness_cols + class_acc_cols
        available = [c for c in fairness_detail_cols if c in df.columns]
        df[available].to_csv(
            args.out_dir / "fairness_metrics.csv", index=False, float_format="%.4f"
        )
        print(f"[Saved] {args.out_dir / 'fairness_metrics.csv'}")

    # --- Raw per-seed data ---
    df.to_csv(args.out_dir / "raw_per_seed.csv", index=False, float_format="%.6f")
    print(f"[Saved] {args.out_dir / 'raw_per_seed.csv'}")

    # --- Print quick summary ---
    print("\n" + "=" * 70)
    print("QUICK SUMMARY (Clean condition, mean across seeds)")
    print("=" * 70)
    clean_summary = summary[summary["Noise_Type"] == "clean"]
    for _, row in clean_summary.iterrows():
        wga = f"WGA={row['worst_group_acc_mean']:.3f}" if "worst_group_acc_mean" in row.index else ""
        gap = f"Gap={row['accuracy_gap_mean']:.3f}" if "accuracy_gap_mean" in row.index else ""
        print(
            f"  {row['Model']:10s} | {row['Task']:8s} | "
            f"Acc={row['Acc_mean']:.3f}±{row['Acc_std']:.3f} | "
            f"F1={row['F1_macro_mean']:.3f} | "
            f"{wga} {gap}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
