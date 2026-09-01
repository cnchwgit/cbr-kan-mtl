# CBR-KAN MTL

Multi-task learning (MTL) for transcription factor (TF) binding site prediction using a **CBR-KAN** architecture: a shared CNN–BiLSTM backbone with one independent **KAN** (Kolmogorov–Arnold Network) prediction head per TF.

## Highlights

- **True multi-head MTL** — shared backbone across all TFs, per-TF KAN heads routed by TF identity
- **Weak TFs benefit** from shared features while strong TFs keep specialized heads
- Balanced positive/negative ChIP-seq FASTA per TF (default split 60/20/20, seed 42)
- CPU-friendly demo that runs end-to-end in ~5–10 min

## Architecture

```
Embedding → Multi-CNN (3 parallel Conv1d) → BiLSTM → LayerNorm
    └────────────── shared CBR-KAN backbone ──────────────┘
                    │
        ┌───────────┼───────────┐
    KAN head      KAN head    KAN head      (one per TF, [256, 32, 2])
    (TF A)        (TF B)      (TF C)
```

## Repository Layout

```
├── data/                  # TF cluster lists + FASTA datasets
│   ├── cluster_t098_csvs/ # TF clusters (e.g. cluster_001.csv)
│   ├── demo_fasta/        # tiny demo dataset (300 pos + 300 neg per TF)
│   └── full_fasta/        # full datasets (kept out of git)
├── scripts/
│   ├── train_mtl_cluster.py    # train CBR-KAN MTL on one TF cluster
│   ├── train_single.py         # single-task baseline
│   ├── train_all_clusters.py   # run all clusters
│   ├── compare_mtl_vs_single.py# paired MTL-vs-single comparison
│   └── generate_plan.py        # training plan generator
├── src/
│   ├── models/
│   │   ├── CBRKAN_MTL.py   # MTL model (shared backbone + per-TF heads)
│   │   └── CBR_KAN.py      # single-task CBR-KAN
│   └── efficient_kan/      # KAN implementation
├── results/                # training + comparison outputs (gitignored)
├── requirements.txt
└── run_demo.sh             # end-to-end demo
```

## Quick Start

```bash
pip install -r requirements.txt

# End-to-end demo (CPU, ~5-10 min)
bash run_demo.sh
```

Demo trains MTL on `cluster_001` (SP140 + SP140L), then single-task baselines, and writes the paired comparison to `results/comparison_demo.csv`.

## Training (full experiment)

```bash
# MTL on one TF cluster
python3 scripts/train_mtl_cluster.py \
    --tf-list data/cluster_t098_csvs/cluster_001.csv \
    --output-dir results/cluster_001 \
    --data-root /path/to/fasta_datasets \
    --epochs 30 \
    --device cuda:0

# Single-task baseline
python3 scripts/train_single.py \
    --tf-name SP140 --gsm-id GSM2398967 \
    --output-dir results/single/SP140 \
    --data-root /path/to/fasta_datasets \
    --device cuda:0

# Paired comparison
python3 scripts/compare_mtl_vs_single.py \
    --mtl-dir results/cluster_001 \
    --single-dir results/single \
    --output results/comparison.csv
```

## Requirements

- Python 3.10+
- PyTorch ≥ 2.3.0, NumPy, pandas, scikit-learn, scipy, matplotlib, seaborn

See `requirements.txt` for the full list.
