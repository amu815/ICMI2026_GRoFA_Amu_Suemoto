#!/usr/bin/env python3
"""Gate-response analysis (Sec. 4.6 and Appendix D of the paper).

Computes, from the stored evaluation npz files (key 'gates' = clamp(1 - noise_score),
i.e. the adapter-side mixing weight g of Eq. (3)):
  1. Per-noise-type Spearman rho between gate value and severity {0(clean),1,2,3}
  2. Clean-vs-corrupted AUROC using the gate value as the ranking score
  3. Per-condition mean gate values (clean / per type x severity)

Reference values reported in the paper:
  FF train rho: cutout .68 / defocus .58 / impulse .50 / jpeg .40 / shot .35 /
                gaussian .29 / glass -.02
  AUROC: FF train 0.74 / FF test 0.73 / UTK 0.76

Usage:
  python3 analysis/gate_analysis.py --out gate_analysis.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = Path("results")
SETS = {
    "ff_train": BASE / "neurips_v60_ens5_wiseft0.95_fairface/train_student.npz",
    "ff_test": BASE / "neurips_v60_ens5_wiseft0.95_fairface/test_student.npz",
    "utk_train": BASE / "neurips_v60_ens5_wiseft0.60_utkface/train_student.npz",
    "utk_test": BASE / "neurips_v60_ens5_wiseft0.60_utkface/test_student.npz",
}
NOISE_TYPES = ["gaussian", "shot", "impulse", "defocus", "glass", "jpeg", "cutout"]


def analyze(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    gates = d["gates"].ravel().astype(float)
    ntype = np.asarray(d["noise_type"]).astype(str)
    nlev = np.asarray(d["noise_level"]).astype(int)

    res = {"n": int(len(gates))}
    clean_mask = ntype == "clean"
    res["mean_gate_clean"] = float(gates[clean_mask].mean())

    rho = {}
    cond_means = {"clean": res["mean_gate_clean"]}
    for nt in NOISE_TYPES:
        mask = clean_mask | (ntype == nt)
        sev = np.where(clean_mask[mask], 0, nlev[mask])
        r, p = spearmanr(gates[mask], sev)
        rho[nt] = {"rho": round(float(r), 3), "p": float(p)}
        for lv in (1, 2, 3):
            m = (ntype == nt) & (nlev == lv)
            cond_means[f"{nt}_lv{lv}"] = round(float(gates[m].mean()), 4)
    res["spearman_by_type"] = rho
    res["mean_gate_by_condition"] = cond_means
    res["mean_gate_corrupted"] = float(gates[~clean_mask].mean())

    y = (~clean_mask).astype(int)
    res["auroc_clean_vs_corrupted"] = round(float(roc_auc_score(y, gates)), 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("gate_analysis.json"))
    args = ap.parse_args()

    out = {}
    for name, path in SETS.items():
        if not path.exists():
            out[name] = {"error": f"missing: {path}"}
            continue
        out[name] = analyze(path)
        print(f"[{name}] n={out[name]['n']} "
              f"AUROC={out[name]['auroc_clean_vs_corrupted']} "
              f"clean_g={out[name]['mean_gate_clean']:.3f} "
              f"corrupt_g={out[name]['mean_gate_corrupted']:.3f}")
        print("   rho:", {k: v["rho"] for k, v in out[name]["spearman_by_type"].items()})

    args.out.write_text(json.dumps(out, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
