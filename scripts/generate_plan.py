#!/usr/bin/env python3
"""
Generate the t098 (threshold=0.98) MTL training plan
=====================================================
Reads cluster definition files from a clustering output directory and
writes: (1) one TF-list CSV per cluster (TF_Name, GSM_ID, Subgroup) and
(2) a plan JSON consumed by train_all_clusters.py.

Only clusters with size >= 2 are included (MTL clusters).

Usage:
    python scripts/generate_plan.py \
        --cluster-dir clustering_results/t098 \
        --tf-info <path/to/tf_info.csv> \
        --output-csv-dir data/cluster_csvs \
        --output-json results/cluster_mtl_t098_plan.json
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='Generate an MTL training plan based on t098 clustering results')
    p.add_argument('--cluster-dir', required=True,
                   help='clustering results directory (containing cluster_*.txt, format: Cluster N / Size: n / Members: ...)')
    p.add_argument('--tf-info', required=True, help='TF info CSV (with TF_Name, GSM_ID columns)')
    p.add_argument('--output-csv-dir', required=True, help='output directory for cluster TF list CSVs')
    p.add_argument('--output-json', required=True, help='output path for the training plan JSON')
    return p.parse_args()


def main():
    args = parse_args()
    T098_DIR = args.cluster_dir
    CSV_PATH = args.tf_info
    OUTPUT_JSON = args.output_json
    TEMP_CSV_DIR = args.output_csv_dir

    print("=" * 70)
    print("Generate the MTL training plan based on t098 clustering (threshold=0.98)")
    print("Only MTL clusters with size>=2 are included")
    print("=" * 70)

    # Read TF info
    print(f"\n[1/4] Reading TF info: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Number of TFs in CSV: {len(df)}")

    tf_to_gsm = {}
    for _, row in df.iterrows():
        tf_name = str(row['TF_Name']).strip()
        gsm_id = str(row['GSM_ID']).strip()
        tf_to_gsm[tf_name.lower()] = gsm_id
        tf_to_gsm[tf_name.upper()] = gsm_id
        tf_to_gsm[tf_name.capitalize()] = gsm_id
        tf_to_gsm[tf_name] = gsm_id
    print(f"  TF→GSM mapping: {len(tf_to_gsm)}")

    # Parse cluster files
    print(f"\n[2/4] Parsing cluster files: {T098_DIR}")
    cluster_files = sorted(
        [f for f in os.listdir(T098_DIR) if f.startswith('cluster_') and f.endswith('.txt')],
        key=lambda x: int(re.search(r'cluster_(\d+)', x).group(1))
    )
    print(f"  Found {len(cluster_files)} cluster files")

    cluster_data = []
    for fname in cluster_files:
        fpath = os.path.join(T098_DIR, fname)
        with open(fpath) as f:
            lines = f.read().strip().split('\n')
        cluster_id = int(re.search(r'cluster_(\d+)', fname).group(1))
        size = int([l for l in lines if l.startswith('Size:')][0].split(':')[1].strip())

        members_start = next(i for i, l in enumerate(lines) if 'Members:' in l) + 1
        tf_names = [l.strip() for l in lines[members_start:] if l.strip()]
        cluster_data.append((cluster_id, size, tf_names))

    cluster_data.sort(key=lambda x: x[0])

    total = len(cluster_data)
    mtl_clusters = [(cid, sz, tfs) for cid, sz, tfs in cluster_data if sz >= 2]
    single_clusters = [(cid, sz, tfs) for cid, sz, tfs in cluster_data if sz == 1]
    print(f"\n  Total clusters: {total}")
    print(f"  MTL clusters: {len(mtl_clusters)} (size>=2)")
    print(f"  Skipped single-TF clusters: {len(single_clusters)}")

    Path(TEMP_CSV_DIR).mkdir(parents=True, exist_ok=True)

    # Generate the training plan
    print(f"\n[3/4] Generating the training plan...")
    plan = []
    valid_tfs = 0
    skipped_tfs = []
    seq_idx = 0  # new numbering: assign cluster_001, cluster_002, ... sequentially in numeric order

    for cluster_id, size, tf_names in cluster_data:
        if size < 2:
            continue  # skip single-TF clusters

        tf_gsm_list = []
        for tf in tf_names:
            gsm = None
            for key in [tf, tf.lower(), tf.upper(), tf.capitalize()]:
                if key in tf_to_gsm:
                    gsm = tf_to_gsm[key]
                    break
            if gsm:
                tf_gsm_list.append({
                    'TF_Name': tf,
                    'GSM_ID': gsm,
                    'Subgroup': f't098_cluster_{cluster_id:03d}'
                })
                valid_tfs += 1
            else:
                skipped_tfs.append((cluster_id, tf))

        if len(tf_gsm_list) < 2:
            print(f"  ⚠️ Cluster {cluster_id}: valid TF < 2 (remaining {len(tf_gsm_list)}), skipping")
            continue

        seq_idx += 1
        temp_csv = os.path.join(TEMP_CSV_DIR, f'cluster_{seq_idx:03d}.csv')
        pd.DataFrame(tf_gsm_list).to_csv(temp_csv, index=False)

        plan.append({
            "Cluster_ID": seq_idx,
            "Original_Cluster_ID": cluster_id,
            "Task_Type": "MTL",
            "TF_Count": len(tf_gsm_list),
            "TF_List_CSV": temp_csv,
        })

    print(f"\n  Planned tasks: {len(plan)}")
    print(f"  TFs in plan: {sum(p['TF_Count'] for p in plan)}")

    size_counts = Counter(p['TF_Count'] for p in plan)
    print(f"\nCluster size distribution:")
    for sz in sorted(size_counts.keys(), reverse=True):
        print(f"  Size {sz:3d}: {size_counts[sz]:3d} clusters")

    # Save the JSON
    print(f"\n[4/4] Saving the plan...")
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Plan: {OUTPUT_JSON}")
    print(f"  ✅ CSVs: {TEMP_CSV_DIR}/")

    print(f"\n{'='*70}")
    print(f"Execution estimate:")
    total_min = 0
    for p in plan:
        tc = p['TF_Count']
        if tc >= 100:
            total_min += 60 * 72
        elif tc >= 10:
            total_min += 30
        elif tc >= 6:
            total_min += 20
        elif tc >= 4:
            total_min += 12
        elif tc == 3:
            total_min += 10
        else:
            total_min += 8
    print(f"  Estimated total time: {total_min} minutes ≈ {total_min/60:.1f} hours")
    print(f"{'='*70}")

    if skipped_tfs:
        print(f"\n⚠️ TFs with no GSM_ID found (first 20):")
        for cid, tf in skipped_tfs[:20]:
            print(f"    Cluster {cid}: {tf}")
        if len(skipped_tfs) > 20:
            print(f"    ... {len(skipped_tfs)} in total")


if __name__ == '__main__':
    main()
