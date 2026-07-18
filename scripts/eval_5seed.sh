#!/bin/bash
# 5-seed ensemble evaluation for both FairFace and UTKFace
set -e
cd "$(dirname "$0")/.."
PY=python3

echo "$(date): Starting 5-seed evaluation pipeline"

###############################################
# FairFace
###############################################
FF_SEEDS=(results/grofa_fairface_s0 results/grofa_fairface_s1 results/grofa_fairface_s2 results/grofa_fairface_s4 results/grofa_fairface_s5)
FF_LORA=models/baselines_v2_fairface/IPMix/ema
FF_TRAIN=data/processed/fairface_corrupted/jsonl/train_views.jsonl
FF_TEST=data/processed/fairface_corrupted/jsonl/test_views.jsonl
FF_BASE=results/baseline_fairface_embeddings

echo "=== FairFace: Generate embeddings ==="
for SDIR in "${FF_SEEDS[@]}"; do
    for SPLIT in train test; do
        if [ "$SPLIT" = "train" ]; then JSONL=$FF_TRAIN; else JSONL=$FF_TEST; fi
        OUT=$SDIR/${SPLIT}_student.npz
        if [ -f "$OUT" ]; then echo "  [SKIP] $OUT"; continue; fi
        echo "  Generating $OUT"
        CUDA_VISIBLE_DEVICES=0 $PY src/gen_grofa_embeddings.py \
            --jsonl $JSONL --out_npz $OUT \
            --model_path $SDIR/model_best.pt --lora_ckpt $FF_LORA
    done
done

echo "=== FairFace: Create 5-seed ensemble ==="
ENS_FF=results/grofa_ensemble_5seed_fairface
$PY -c "
import numpy as np; from pathlib import Path
seeds = '${FF_SEEDS[0]} ${FF_SEEDS[1]} ${FF_SEEDS[2]} ${FF_SEEDS[3]} ${FF_SEEDS[4]}'.split()
out = '$ENS_FF'
Path(out).mkdir(parents=True, exist_ok=True)
for split in ['train','test']:
    embs = [np.load(f'{s}/{split}_student.npz', allow_pickle=True) for s in seeds]
    z = np.mean([e['embeddings'] for e in embs], axis=0).astype(np.float32)
    d = {k: embs[0][k] for k in embs[0].files}
    d['embeddings'] = z
    np.savez(f'{out}/{split}_student.npz', **d)
    print(f'  Saved {out}/{split}_student.npz shape={z.shape}')
"

echo "=== FairFace: WiSE-FT on ensemble ==="
for ALPHA in 0.90 0.95; do
    WFT_DIR=results/grofa_ens5_wiseft${ALPHA}_fairface
    $PY -c "
import numpy as np; from pathlib import Path
alpha=$ALPHA; sd='$ENS_FF'; bd='$FF_BASE'; od='$WFT_DIR'
Path(od).mkdir(parents=True, exist_ok=True)
for s in ['train','test']:
    b=np.load(f'{bd}/base_{s}.npz',allow_pickle=True)
    t=np.load(f'{sd}/{s}_student.npz',allow_pickle=True)
    z=alpha*t['embeddings']+(1-alpha)*b['embeddings']
    d={k:t[k] for k in t.files}; d['embeddings']=z.astype(np.float32)
    np.savez(f'{od}/{s}_student.npz',**d)
print(f'  WFT alpha={alpha} done')
"
done

echo "=== FairFace: Evaluate ==="
FF_BASE_TRAIN=$FF_BASE/base_train.npz
FF_BASE_TEST=$FF_BASE/base_test.npz

# Individual seeds
for SDIR in "${FF_SEEDS[@]}"; do
    EVAL_DIR=$SDIR/eval_final
    if [ -f "$EVAL_DIR/summary_by_condition.csv" ]; then echo "  [SKIP] $SDIR"; continue; fi
    echo "  Eval $SDIR"
    $PY src/evaluate_protocol.py \
        --base_train_npz $FF_BASE_TRAIN --base_test_npz $FF_BASE_TEST \
        --student_train_npz $SDIR/train_student.npz --student_test_npz $SDIR/test_student.npz \
        --out_dir $EVAL_DIR
done

# Ensemble RAW
echo "  Eval Ensemble RAW"
$PY src/evaluate_protocol.py \
    --base_train_npz $FF_BASE_TRAIN --base_test_npz $FF_BASE_TEST \
    --student_train_npz $ENS_FF/train_student.npz --student_test_npz $ENS_FF/test_student.npz \
    --out_dir $ENS_FF/eval_final

# Ensemble + WiSE-FT
for ALPHA in 0.90 0.95; do
    WFT_DIR=results/grofa_ens5_wiseft${ALPHA}_fairface
    echo "  Eval Ensemble WFT alpha=$ALPHA"
    $PY src/evaluate_protocol.py \
        --base_train_npz $FF_BASE_TRAIN --base_test_npz $FF_BASE_TEST \
        --student_train_npz $WFT_DIR/train_student.npz --student_test_npz $WFT_DIR/test_student.npz \
        --out_dir $WFT_DIR/eval_final
done

###############################################
# UTKFace
###############################################
UT_SEEDS=(results/grofa_ipmix_utkface_s0 results/grofa_ipmix_utkface_s1 results/grofa_ipmix_utkface_s2 results/grofa_ipmix_utkface_s3 results/grofa_ipmix_utkface_s4)
UT_LORA=models/baselines_v2_utk/IPMix/ema
UT_TRAIN=data/processed/utkface_corrupted/jsonl/train_views.jsonl
UT_TEST=data/processed/utkface_corrupted/jsonl/test_views.jsonl
UT_BASE=results/baseline_utkface_embeddings

echo "=== UTKFace: Generate embeddings ==="
for SDIR in "${UT_SEEDS[@]}"; do
    for SPLIT in train test; do
        if [ "$SPLIT" = "train" ]; then JSONL=$UT_TRAIN; else JSONL=$UT_TEST; fi
        OUT=$SDIR/${SPLIT}_student.npz
        if [ -f "$OUT" ]; then echo "  [SKIP] $OUT"; continue; fi
        echo "  Generating $OUT"
        CUDA_VISIBLE_DEVICES=0 $PY src/gen_grofa_embeddings.py \
            --jsonl $JSONL --out_npz $OUT \
            --model_path $SDIR/model_best.pt --lora_ckpt $UT_LORA
    done
done

echo "=== UTKFace: Create 5-seed ensemble ==="
ENS_UT=results/grofa_ensemble_5seed_utkface
$PY -c "
import numpy as np; from pathlib import Path
seeds = '${UT_SEEDS[0]} ${UT_SEEDS[1]} ${UT_SEEDS[2]} ${UT_SEEDS[3]} ${UT_SEEDS[4]}'.split()
out = '$ENS_UT'
Path(out).mkdir(parents=True, exist_ok=True)
for split in ['train','test']:
    embs = [np.load(f'{s}/{split}_student.npz', allow_pickle=True) for s in seeds]
    z = np.mean([e['embeddings'] for e in embs], axis=0).astype(np.float32)
    d = {k: embs[0][k] for k in embs[0].files}
    d['embeddings'] = z
    np.savez(f'{out}/{split}_student.npz', **d)
    print(f'  Saved {out}/{split}_student.npz shape={z.shape}')
"

echo "=== UTKFace: WiSE-FT on ensemble ==="
for ALPHA in 0.60 0.90 0.95; do
    WFT_DIR=results/grofa_ens5_wiseft${ALPHA}_utkface
    $PY -c "
import numpy as np; from pathlib import Path
alpha=$ALPHA; sd='$ENS_UT'; bd='$UT_BASE'; od='$WFT_DIR'
Path(od).mkdir(parents=True, exist_ok=True)
for s in ['train','test']:
    b=np.load(f'{bd}/base_{s}.npz',allow_pickle=True)
    t=np.load(f'{sd}/{s}_student.npz',allow_pickle=True)
    z=alpha*t['embeddings']+(1-alpha)*b['embeddings']
    d={k:t[k] for k in t.files}; d['embeddings']=z.astype(np.float32)
    np.savez(f'{od}/{s}_student.npz',**d)
print(f'  WFT alpha={alpha} done')
"
done

echo "=== UTKFace: Evaluate ==="
UT_BASE_TRAIN=$UT_BASE/base_train.npz
UT_BASE_TEST=$UT_BASE/base_test.npz

# Individual seeds
for SDIR in "${UT_SEEDS[@]}"; do
    EVAL_DIR=$SDIR/eval_final
    if [ -f "$EVAL_DIR/summary_by_condition.csv" ]; then echo "  [SKIP] $SDIR"; continue; fi
    echo "  Eval $SDIR"
    $PY src/evaluate_protocol.py \
        --base_train_npz $UT_BASE_TRAIN --base_test_npz $UT_BASE_TEST \
        --student_train_npz $SDIR/train_student.npz --student_test_npz $SDIR/test_student.npz \
        --out_dir $EVAL_DIR
done

# Ensemble RAW
echo "  Eval Ensemble RAW"
$PY src/evaluate_protocol.py \
    --base_train_npz $UT_BASE_TRAIN --base_test_npz $UT_BASE_TEST \
    --student_train_npz $ENS_UT/train_student.npz --student_test_npz $ENS_UT/test_student.npz \
    --out_dir $ENS_UT/eval_final

# Ensemble + WiSE-FT
for ALPHA in 0.60 0.90 0.95; do
    WFT_DIR=results/grofa_ens5_wiseft${ALPHA}_utkface
    echo "  Eval Ensemble WFT alpha=$ALPHA"
    $PY src/evaluate_protocol.py \
        --base_train_npz $UT_BASE_TRAIN --base_test_npz $UT_BASE_TEST \
        --student_train_npz $WFT_DIR/train_student.npz --student_test_npz $WFT_DIR/test_student.npz \
        --out_dir $WFT_DIR/eval_final
done

###############################################
# Summary
###############################################
echo "=== FINAL SUMMARY ==="
$PY -c "
import pandas as pd, os

def bm_wins(csv):
    df=pd.read_csv(csv)
    bl=df[df.Model=='Baseline']; ours=df[df.Model=='Ours']
    m=bl.merge(ours,on=['Task','Noise_Type','Noise_Level'],suffixes=('_bl','_ours'))
    total=int((m.Acc_mean_ours>m.Acc_mean_bl).sum())
    parts={}
    for t in ['race','gender','age']:
        parts[t]=int((m[m.Task==t].Acc_mean_ours>m[m.Task==t].Acc_mean_bl).sum())
    return total, parts

print('=== FairFace ===')
print(f'{\"Config\":20} {\"Total\":>6} {\"race\":>6} {\"gender\":>7} {\"age\":>5}')
for label, csv in [
    ('s0 (seed=42)', 'results/grofa_fairface_s0/eval_final/summary_by_condition.csv'),
    ('s1 (seed=0)', 'results/grofa_fairface_s1/eval_final/summary_by_condition.csv'),
    ('s2 (seed=1)', 'results/grofa_fairface_s2/eval_final/summary_by_condition.csv'),
    ('s4 (seed=2)', 'results/grofa_fairface_s4/eval_final/summary_by_condition.csv'),
    ('s5 (seed=3)', 'results/grofa_fairface_s5/eval_final/summary_by_condition.csv'),
    ('Ens5 RAW', 'results/grofa_ensemble_5seed_fairface/eval_final/summary_by_condition.csv'),
    ('Ens5+WFT0.90', 'results/grofa_ens5_wiseft0.90_fairface/eval_final/summary_by_condition.csv'),
    ('Ens5+WFT0.95', 'results/grofa_ens5_wiseft0.95_fairface/eval_final/summary_by_condition.csv'),
]:
    if not os.path.exists(csv): print(f'{label:20} NOT FOUND'); continue
    t, p = bm_wins(csv)
    print(f'{label:20} {t:>4}/66 {p[\"race\"]:>4}/22 {p[\"gender\"]:>5}/22 {p[\"age\"]:>3}/22')

print()
print('=== UTKFace ===')
print(f'{\"Config\":20} {\"Total\":>6} {\"race\":>6} {\"gender\":>7} {\"age\":>5}')
for label, csv in [
    ('s0 (existing)', 'results/grofa_ipmix_utkface_s0/eval_final/summary_by_condition.csv'),
    ('s1 (seed=0)', 'results/grofa_ipmix_utkface_s1/eval_final/summary_by_condition.csv'),
    ('s2 (seed=1)', 'results/grofa_ipmix_utkface_s2/eval_final/summary_by_condition.csv'),
    ('s3 (seed=2)', 'results/grofa_ipmix_utkface_s3/eval_final/summary_by_condition.csv'),
    ('s4 (seed=3)', 'results/grofa_ipmix_utkface_s4/eval_final/summary_by_condition.csv'),
    ('Ens5 RAW', 'results/grofa_ensemble_5seed_utkface/eval_final/summary_by_condition.csv'),
    ('Ens5+WFT0.60', 'results/grofa_ens5_wiseft0.60_utkface/eval_final/summary_by_condition.csv'),
    ('Ens5+WFT0.90', 'results/grofa_ens5_wiseft0.90_utkface/eval_final/summary_by_condition.csv'),
    ('Ens5+WFT0.95', 'results/grofa_ens5_wiseft0.95_utkface/eval_final/summary_by_condition.csv'),
]:
    if not os.path.exists(csv): print(f'{label:20} NOT FOUND'); continue
    t, p = bm_wins(csv)
    print(f'{label:20} {t:>4}/66 {p[\"race\"]:>4}/22 {p[\"gender\"]:>5}/22 {p[\"age\"]:>3}/22')
"
echo "$(date): ALL EVALUATION COMPLETE"
