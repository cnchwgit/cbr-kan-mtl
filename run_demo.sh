#!/usr/bin/env bash
# =============================================================================
# CBR-KAN MTL — end-to-end demo (CPU, ~5-10 min)
# =============================================================================
# Trains on a 2-TF cluster with a tiny demo FASTA dataset (300 pos + 300 neg
# per TF), then trains single-task baselines for the same TFs and runs the
# paired MTL-vs-Single comparison. Everything runs on CPU so it works on any
# laptop.
#
# Usage:
#     bash run_demo.sh
#
# Outputs are written to results/demo_*/ .
# =============================================================================
set -e
cd "$(dirname "$0")"

DATA_ROOT="$(pwd)/data/demo_fasta"
EPOCHS=3            # demo only — use 30 for the full experiment
BATCH=64
MAX_PER_CLASS=300
SEED=42

echo "==> [1/4] Train CBR-KAN MTL on cluster_001 (SP140 + SP140L)"
python3 scripts/train_mtl_cluster.py \
    --tf-list data/cluster_t098_csvs/cluster_001.csv \
    --output-dir results/demo_mtl/cluster_001 \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --max-per-class "$MAX_PER_CLASS" \
    --device cpu

echo "==> [2/4] Train single-task baselines"
for tf_gsm in SP140_GSM2398967 SP140L_GSM5214592; do
    tf="${tf_gsm%%_*}"
    gsm="${tf_gsm##*_}"
    echo "    -> ${tf} (${gsm})"
    python3 scripts/train_single.py \
        --tf-name "$tf" --gsm-id "$gsm" \
        --data-root "$DATA_ROOT" \
        --output "results/demo_single/${tf_gsm}" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --max-per-class "$MAX_PER_CLASS" \
        --device cpu
done

echo "==> [3/4] Paired comparison (MTL vs Single)"
python3 scripts/compare_mtl_vs_single.py \
    --single-dir results/demo_single \
    --mtl-dir results/demo_mtl \
    --cluster-csv-dir data/cluster_t098_csvs \
    --output-dir results \
    --tag demo

echo "==> [4/4] Demo finished."
echo "    MTL results:      results/demo_mtl/cluster_001/"
echo "    Single results:   results/demo_single/"
echo "    Comparison CSV:   results/comparison_demo.csv"
echo "    Comparison table: results/comparison_demo_detailed.csv"
