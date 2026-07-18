"""Generate WiSE-FT interpolated embeddings for v60-IPMix UTKFace.

Creates interpolated embedding NPZs at various alphas:
  z_wft = alpha * z_student + (1 - alpha) * z_baseline

Then evaluates each with evaluate_protocol.py protocol.
"""
import numpy as np
from pathlib import Path
import subprocess, sys

BASE = Path("results")
STUDENT_DIR = BASE / "grofa_ipmix_utkface_s0"
BASELINE_DIR = BASE / "baseline_utkface_embeddings"

# Load embeddings
print("Loading embeddings...", flush=True)
stu_train = np.load(STUDENT_DIR / "train_student.npz", allow_pickle=True)
stu_test = np.load(STUDENT_DIR / "test_student.npz", allow_pickle=True)
bl_train = np.load(BASELINE_DIR / "base_train.npz", allow_pickle=True)
bl_test = np.load(BASELINE_DIR / "base_test.npz", allow_pickle=True)

alphas = [0.50, 0.60, 0.70, 0.80, 0.90]

for alpha in alphas:
    out_dir = BASE / f"grofa_wiseft{alpha:.2f}_utkface"
    out_dir.mkdir(exist_ok=True)

    # Interpolate
    train_emb = alpha * stu_train["embeddings"] + (1 - alpha) * bl_train["embeddings"]
    test_emb = alpha * stu_test["embeddings"] + (1 - alpha) * bl_test["embeddings"]

    # Save with all metadata from student (labels, noise info)
    train_data = {k: stu_train[k] for k in stu_train.files}
    train_data["embeddings"] = train_emb
    test_data = {k: stu_test[k] for k in stu_test.files}
    test_data["embeddings"] = test_emb

    np.savez(out_dir / "train_student.npz", **train_data)
    np.savez(out_dir / "test_student.npz", **test_data)
    print(f"  alpha={alpha:.2f}: saved to {out_dir}", flush=True)

    # Evaluate
    eval_dir = out_dir / "eval_final"
    cmd = [
        sys.executable, "src/evaluate_protocol.py",
        "--base_train_npz", str(BASELINE_DIR / "base_train.npz"),
        "--base_test_npz", str(BASELINE_DIR / "base_test.npz"),
        "--student_train_npz", str(out_dir / "train_student.npz"),
        "--student_test_npz", str(out_dir / "test_student.npz"),
        "--out_dir", str(eval_dir),
        "--n_seeds", "5",
    ]
    print(f"  Evaluating alpha={alpha:.2f}...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}", flush=True)
    else:
        # Print summary line
        for line in result.stdout.split("\n"):
            if "Acc=" in line and "Ours" in line:
                print(f"    {line.strip()}", flush=True)
    print(flush=True)

print("Done! Now run 1st-place analysis.", flush=True)
