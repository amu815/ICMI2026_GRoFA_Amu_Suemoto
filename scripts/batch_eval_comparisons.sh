#!/bin/bash
# Batch evaluate all comparison method embeddings under 66-condition protocol
set -e

PYTHON=python3
EVAL_SCRIPT=./src/evaluate_neurips_v2.py
BASE_TRAIN=./results/neurips_v39_fairface_embeddings/base_train.npz
BASE_TEST=./results/neurips_v39_fairface_embeddings/base_test.npz
EMB_DIR=./results/comparison_embeddings  # npz embeddings of the post-hoc baselines
OUT_BASE=./results/comparison_eval_fairface

mkdir -p "$OUT_BASE"

eval_method() {
    local name=$1
    local train=$2
    local test=$3
    local out=$4
    echo "=== Evaluating: $name ==="
    $PYTHON $EVAL_SCRIPT \
        --base_train_npz "$BASE_TRAIN" \
        --base_test_npz "$BASE_TEST" \
        --student_train_npz "$train" \
        --student_test_npz "$test" \
        --out_dir "$out" \
        --n_seeds 5
    echo "=== Done: $name ==="
    echo ""
}

# Post-hoc methods
eval_method "LEACE" "$EMB_DIR/LEACE_train.npz" "$EMB_DIR/LEACE_test.npz" "$OUT_BASE/LEACE"
eval_method "PASS" "$EMB_DIR/PASS_train.npz" "$EMB_DIR/PASS_test.npz" "$OUT_BASE/PASS"
eval_method "FairerCLIP" "$EMB_DIR/FairerCLIP_train.npz" "$EMB_DIR/FairerCLIP_test.npz" "$OUT_BASE/FairerCLIP"
eval_method "SFID" "$EMB_DIR/SFID_train.npz" "$EMB_DIR/SFID_test.npz" "$OUT_BASE/SFID"

# LoRA-based methods (raw)
eval_method "AugMix" "$EMB_DIR/AugMix_train.npz" "$EMB_DIR/AugMix_test.npz" "$OUT_BASE/AugMix_raw"
eval_method "PixMix" "$EMB_DIR/PixMix_train.npz" "$EMB_DIR/PixMix_test.npz" "$OUT_BASE/PixMix_raw"
eval_method "IPMix" "$EMB_DIR/IPMix_train.npz" "$EMB_DIR/IPMix_test.npz" "$OUT_BASE/IPMix_raw"

# LoRA-based methods with WiSE-FT
for alpha in 0.6 0.7 0.8 0.9 1.0; do
    eval_method "AugMix_a${alpha}" "$EMB_DIR/AugMix_fairnoise_a${alpha}_train.npz" "$EMB_DIR/AugMix_fairnoise_a${alpha}_test.npz" "$OUT_BASE/AugMix_a${alpha}"
    eval_method "PixMix_a${alpha}" "$EMB_DIR/PixMix_fairnoise_a${alpha}_train.npz" "$EMB_DIR/PixMix_fairnoise_a${alpha}_test.npz" "$OUT_BASE/PixMix_a${alpha}"
    eval_method "IPMix_a${alpha}" "$EMB_DIR/IPMix_fairnoise_a${alpha}_train.npz" "$EMB_DIR/IPMix_fairnoise_a${alpha}_test.npz" "$OUT_BASE/IPMix_a${alpha}"
done

echo "ALL EVALUATIONS COMPLETE"
