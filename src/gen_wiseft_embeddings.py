#!/usr/bin/env python3
"""Generate WiSE-FT interpolated embeddings: z = alpha*z_student + (1-alpha)*z_base."""
import numpy as np
import sys
from pathlib import Path

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.60
student_dir = sys.argv[2] if len(sys.argv) > 2 else "results/neurips_v55b_fairface_s0"
base_dir = "results/neurips_v39_fairface_embeddings"
out_dir = f"results/neurips_v55b_wiseft{alpha:.2f}_fairface"

Path(out_dir).mkdir(parents=True, exist_ok=True)

for split in ["train", "test"]:
    base_path = f"{base_dir}/base_{split}.npz"
    student_path = f"{student_dir}/{split}_student.npz"
    out_path = f"{out_dir}/{split}_student.npz"

    base = np.load(base_path, allow_pickle=True)
    student = np.load(student_path, allow_pickle=True)

    z_base = base["embeddings"]
    z_student = student["embeddings"]
    z_wiseft = alpha * z_student + (1 - alpha) * z_base

    # Copy all metadata from student, replace embeddings
    data = {k: student[k] for k in student.files}
    data["embeddings"] = z_wiseft.astype(np.float32)

    np.savez(out_path, **data)
    print(f"Saved {out_path}: shape={z_wiseft.shape}, alpha={alpha}")

print("Done")
