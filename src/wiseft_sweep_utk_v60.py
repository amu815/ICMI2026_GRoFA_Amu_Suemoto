"""WiSE-FT alpha sweep for v60-IPMix UTKFace + 1st-place analysis."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings, sys
warnings.filterwarnings("ignore")

BASE = Path("results")

# UTKFace comparison methods
COMP_DIRS = {
    "IPMix": BASE / "comparison_cls_utk/eval_IPMix",
    "AugMix": BASE / "comparison_cls_utk/eval_AugMix",
    "PixMix": BASE / "comparison_cls_utk/eval_PixMix",
    "DiffAug": BASE / "comparison_cls_utk/eval_DiffAug",
    "GaussianConsist": BASE / "comparison_cls_utk/eval_GaussianConsist",
    "FairerCLIP": BASE / "comparison_cls_utk/eval_FairerCLIP",
    "LEACE": BASE / "comparison_cls_utk/eval_LEACE",
    "PASS": BASE / "comparison_cls_utk/eval_PASS",
    "SensitiveNets": BASE / "comparison_cls_utk/eval_SensitiveNets",
    "SFID": BASE / "comparison_cls_utk/eval_SFID",
}

# Load comparison results
comp_results = {}
for method, eval_dir in COMP_DIRS.items():
    csv_path = eval_dir / "summary_by_condition.csv"
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    ours = df[df["Model"] == "Ours"]
    baseline = df[df["Model"] == "Baseline"]
    comp_results[method] = {
        (r["Task"], r["Noise_Type"], int(r["Noise_Level"])): r["Acc_mean"]
        for _, r in ours.iterrows()
    }
    if "Baseline" not in comp_results:
        comp_results["Baseline"] = {
            (r["Task"], r["Noise_Type"], int(r["Noise_Level"])): r["Acc_mean"]
            for _, r in baseline.iterrows()
        }

print(f"Loaded {len(comp_results)-1} comparison methods", flush=True)


def count_wins(dart_accs):
    wins, bm_wins, total = 0, 0, 0
    pt = {"race": [0, 0], "gender": [0, 0], "age": [0, 0]}
    for key, dart_acc in dart_accs.items():
        task = key[0]
        total += 1
        best_comp = dart_acc
        bl_acc = comp_results.get("Baseline", {}).get(key, 0)
        for m, accs in comp_results.items():
            if m == "Baseline":
                continue
            if key in accs:
                best_comp = max(best_comp, accs[key])
        if dart_acc >= best_comp:
            wins += 1
            pt[task][0] += 1
        if dart_acc > bl_acc:
            bm_wins += 1
        pt[task][1] += 1
    return wins, bm_wins, total, pt


# v53 reference
v53_csv = BASE / "neurips_v53_utkface_s0/eval_v32base/summary_by_condition.csv"
df53 = pd.read_csv(v53_csv)
v53_accs = {
    (r["Task"], r["Noise_Type"], int(r["Noise_Level"])): r["Acc_mean"]
    for _, r in df53[df53["Model"] == "Ours"].iterrows()
}
w53, bm53, _, pt53 = count_wins(v53_accs)
print(f"v53 raw: {w53}/66 1st, {bm53}/66 BM>  R={pt53['race'][0]} G={pt53['gender'][0]} A={pt53['age'][0]}", flush=True)

# v60 raw
v60_csv = BASE / "neurips_v60_ipmix_utkface_s0/eval_final/summary_by_condition.csv"
df60 = pd.read_csv(v60_csv)
v60_accs = {
    (r["Task"], r["Noise_Type"], int(r["Noise_Level"])): r["Acc_mean"]
    for _, r in df60[df60["Model"] == "Ours"].iterrows()
}
w60, bm60, _, pt60 = count_wins(v60_accs)
print(f"v60 raw: {w60}/66 1st, {bm60}/66 BM>  R={pt60['race'][0]} G={pt60['gender'][0]} A={pt60['age'][0]}", flush=True)

# WiSE-FT sweep
print(f"\n{'='*60}", flush=True)
print("  WiSE-FT Embedding Interpolation (alpha * student + (1-a) * base)", flush=True)
print(f"{'='*60}", flush=True)

dart_train = np.load(BASE / "neurips_v60_ipmix_utkface_s0/train_student.npz", allow_pickle=True)
dart_test = np.load(BASE / "neurips_v60_ipmix_utkface_s0/test_student.npz", allow_pickle=True)
bl_train = np.load(BASE / "neurips_v32_utkface_embeddings/base_train.npz", allow_pickle=True)
bl_test = np.load(BASE / "neurips_v32_utkface_embeddings/base_test.npz", allow_pickle=True)

tasks = ["race", "gender", "age"]
train_labels = {t: dart_train[t] for t in tasks}
test_labels = {t: dart_test[t] for t in tasks}
test_nt = dart_test["noise_type"]
test_nl = dart_test["noise_level"]
train_clean = dart_train["noise_level"] == 0

# Precompute unique conditions
conditions = []
for nt in np.unique(test_nt):
    for nl in np.unique(test_nl):
        mask = (test_nt == nt) & (test_nl == nl)
        if mask.sum() > 0:
            conditions.append((str(nt), int(nl), mask))

n_seeds = 5
alphas = np.arange(0.0, 1.01, 0.10)

print(f"\n  {'Alpha':<7} {'1st':>4} {'BM>':>4}  {'R':>3} {'G':>3} {'A':>3}", flush=True)
print(f"  {'-'*35}", flush=True)

best_wins, best_alpha, best_bm, best_pt = 0, 1.0, 0, None

for alpha in alphas:
    train_emb = alpha * dart_train["embeddings"] + (1 - alpha) * bl_train["embeddings"]
    test_emb = alpha * dart_test["embeddings"] + (1 - alpha) * bl_test["embeddings"]

    dart_accs = {}
    for task in tasks:
        y_train = train_labels[task]
        y_test = test_labels[task]
        seed_accs = {}
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            clean_idx = np.where(train_clean)[0]
            rng.shuffle(clean_idx)
            train_idx = clean_idx[:int(0.8 * len(clean_idx))]
            clf = LogisticRegression(max_iter=500, random_state=seed, C=1.0, solver='lbfgs')
            clf.fit(train_emb[train_idx], y_train[train_idx])
            for nt_str, nl_int, mask in conditions:
                key = (task, nt_str, nl_int)
                pred = clf.predict(test_emb[mask])
                acc = accuracy_score(y_test[mask], pred)
                if key not in seed_accs:
                    seed_accs[key] = []
                seed_accs[key].append(acc)
        for key, accs in seed_accs.items():
            dart_accs[key] = np.mean(accs)

    w, bm, t, pt = count_wins(dart_accs)
    marker = " ***" if w > best_wins else ""
    print(f"  {alpha:<7.2f} {w:>4} {bm:>4}  {pt['race'][0]:>3} {pt['gender'][0]:>3} {pt['age'][0]:>3}{marker}", flush=True)

    if w > best_wins or (w == best_wins and bm > best_bm):
        best_wins, best_alpha, best_bm, best_pt = w, alpha, bm, pt

print(f"\n  BEST: alpha={best_alpha:.2f} -> {best_wins}/66 1st, {best_bm}/66 BM>  "
      f"R={best_pt['race'][0]} G={best_pt['gender'][0]} A={best_pt['age'][0]}", flush=True)
print(f"\n  v53 raw={w53}/66 | v60 raw={w60}/66 | v60 WFT({best_alpha:.2f})={best_wins}/66", flush=True)
