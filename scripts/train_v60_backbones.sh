#!/bin/bash
# Train AugMix and PixMix LoRA backbones for 5-seed backbone diversification ablation
set -e
cd "$(dirname "$0")/.."
PY=python3

echo "$(date): Starting backbone diversification training pipeline"

# Seed mapping: s0 -> 42, s1 -> 0, s2 -> 1, s3 -> 2, s4 -> 3
SEEDS=(42 0 1 2 3)
SLOTS=(s0 s1 s2 s3 s4)

run_one () {
    local BK=$1       # AugMix|PixMix
    local DS=$2       # fairface|utk
    local JDIR=$3     # fairface_corrupted | utkface_corrupted
    for i in ${!SEEDS[@]}; do
        SEED=${SEEDS[$i]}
        SLOT=${SLOTS[$i]}
        LC=${BK,,}    # lowercase
        DS_TAG=$( [ "$DS" = "fairface" ] && echo fairface || echo utkface )
        SDIR=results/grofa_${LC}_${DS_TAG}_${SLOT}
        LOG=${SDIR}.log
        LORA_CKPT=models/baselines_v2_${DS}/${BK}/ema
        if [ -f "$SDIR/model_best.pt" ]; then
            echo "$(date): [SKIP] ${BK}-${DS_TAG} ${SLOT} already done"
            continue
        fi
        if [ ! -d "$LORA_CKPT" ]; then
            echo "$(date): [ERR ] missing LoRA $LORA_CKPT, skipping"
            continue
        fi
        echo "$(date): Starting ${BK}-${DS_TAG} seed=${SEED} (${SLOT}) -> $SDIR"
        CUDA_VISIBLE_DEVICES=0 $PY src/train_grofa.py \
            --train_jsonl data/processed/${JDIR}/jsonl/train_views.jsonl \
            --val_jsonl data/processed/${JDIR}/jsonl/test_views.jsonl \
            --out_dir $SDIR \
            --lora_ckpt $LORA_CKPT \
            --seed $SEED \
            > $LOG 2>&1 || { echo "$(date): [FAIL] ${BK}-${DS_TAG} ${SLOT}, continuing"; continue; }
        echo "$(date): ${BK}-${DS_TAG} ${SLOT} done. $(grep 'New best' $LOG | tail -1)"
    done
}

# AugMix first (more likely to produce useful results at single-seed 51/58)
run_one AugMix fairface fairface_corrupted
run_one AugMix utk      utkface_corrupted

# Then PixMix (single-seed was weaker but try ensemble rescue)
run_one PixMix fairface fairface_corrupted
run_one PixMix utk      utkface_corrupted

echo "$(date): ALL BACKBONE TRAINING COMPLETE"
