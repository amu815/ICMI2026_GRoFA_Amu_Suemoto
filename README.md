# GRoFA: Noise-Gated Adapters for Jointly Fair and Robust Face Embeddings

Official code for the ICMI 2026 paper
**"GRoFA: Noise-Gated Adapters for Jointly Fair and Robust Face Embeddings"**
(Amu Suemoto, Yutaka Arakawa, Tsunenori Mine — Kyushu University).
DOI: [10.1145/3776574.3831174](https://doi.org/10.1145/3776574.3831174)

GRoFA is a parameter-efficient adapter for frozen vision encoders (BLIP / CLIP / DINOv2)
that jointly improves demographic fairness and noise robustness of face-attribute
embeddings. A noise gate estimates per-input corruption severity and modulates a
bottleneck adapter's contribution; a 99-parameter router (RTDAR) balances 14
distillation/alignment losses.

## Repository layout

```
src/        Training, models, losses, embedding generation, evaluation
scripts/    End-to-end pipelines (5-seed evaluation, LoRA backbones, baseline eval)
analysis/   Gate-response analysis and statistical tests
logs/       Per-seed evaluation logs of the final models (CSV)
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Datasets are not redistributed here. Download FairFace and UTKFace from their
official sources and accept their licenses, then build the fixed corruption
datasets (7 corruption types x 3 severities, pre-generated once):

```bash
python3 src/build_dataset_fairface.py   # FairFace: 4,000 train / 1,000 test pool
python3 src/build_dataset_utkface.py            # UTKFace
```

## Reproduction pipeline

1. **LoRA backbone (optional but used for the main BLIP results).**
   Pre-trained IPMix / AugMix / PixMix LoRA backbones are provided as release
   checkpoints; `scripts/train_v60_backbones.sh` retrains the AugMix / PixMix
   variants from scratch.

2. **Train GRoFA (5 seeds).**
   ```bash
   python3 src/train_grofa.py --ablation_mode rtdar_g \
     --train_jsonl data/processed/fairface_corrupted/jsonl/train_views.jsonl \
     --val_jsonl   data/processed/fairface_corrupted/jsonl/test_views.jsonl \
     --out_dir results/grofa_fairface_s0 \
     --lora_ckpt models/baselines_v2_fairface/IPMix/ema --lora_scale 1.0 \
     --epochs 50 --patience 20 --seed 42 \
     --lambda_hsic_race 0.5 --lambda_hsic_gender 0.3 --lambda_anchor 0.5 \
     --arc_fixed_weight 0.15
   ```
   `src/train_grofa_xback.py` is the same trainer instantiated on
   CLIP ViT-B/16 and DINOv2-base.

3. **5-seed ensemble + WiSE-FT + 66-condition evaluation.**
   ```bash
   bash scripts/eval_5seed.sh
   ```
   This generates per-seed embeddings (`src/gen_grofa_embeddings.py`), averages
   them, applies WiSE-FT interpolation against the frozen baseline
   (`src/gen_wiseft_embeddings*.py`, alpha sweep in `src/wiseft_sweep_utk.py`),
   and evaluates a clean-trained logistic probe under 22 conditions x 3 tasks
   (`src/evaluate_protocol.py`).

4. **Baselines.**
   `src/run_debiasae.py` is our adaptation of SAE-based debiasing (debiaSAE) to
   ViT CLS embeddings. `scripts/batch_eval_comparisons.sh` evaluates all
   post-hoc baseline embeddings under the identical protocol.

5. **Analysis.**
   `analysis/gate_analysis.py` reproduces the gate-response statistics
   (Spearman rho per corruption type, clean-vs-corrupted AUROC);
   `analysis/phase1_statistical_tests.py` reproduces the significance tests
   from the per-seed logs in `logs/`.

## Per-seed logs

`logs/fairface_5seed_wiseft095/` and `logs/utkface_5seed_wiseft060/` contain the
raw per-seed accuracies (`raw_per_seed.csv`), per-condition summaries, and
fairness metrics of the final 5-seed + WiSE-FT models reported in the paper.

## Checkpoints

Trained adapter/gate checkpoints and the LoRA backbones are distributed via the
GitHub Releases page of this repository.

## License

MIT (see `LICENSE`). The FairFace and UTKFace datasets keep their own licenses.

## Citation

```bibtex
@inproceedings{suemoto2026grofa,
  author    = {Suemoto, Amu and Arakawa, Yutaka and Mine, Tsunenori},
  title     = {GRoFA: Noise-Gated Adapters for Jointly Fair and Robust Face Embeddings},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimodal Interaction (ICMI '26)},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3776574.3831174}
}
```
