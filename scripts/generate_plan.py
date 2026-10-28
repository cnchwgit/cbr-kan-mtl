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
        --output-csv-dir data/cluster_t098_csvs \
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
    p = argparse.ArgumentParser(description='生成基于 t098 聚类结果的 MTL 训练计划')
    p.add_argument('--cluster-dir', required=True,
                   help='聚类结果目录 (含 cluster_*.txt，格式: Cluster N / Size: n / Members: ...)')
    p.add_argument('--tf-info', required=True, help='TF 信息 CSV (含 TF_Name, GSM_ID 列)')
    p.add_argument('--output-csv-dir', required=True, help='簇 TF 列表 CSV 输出目录')
    p.add_argument('--output-json', required=True, help='训练计划 JSON 输出路径')
    return p.parse_args()


def main():
    args = parse_args()
    T098_DIR = args.cluster_dir
    CSV_PATH = args.tf_info
    OUTPUT_JSON = args.output_json
    TEMP_CSV_DIR = args.output_csv_dir

    print("=" * 70)
    print("生成基于 t098 聚类 (threshold=0.98) 的 MTL 训练计划")
    print("仅包含 size>=2 的 MTL 簇")
    print("=" * 70)

    # 读取 TF 信息
    print(f"\n[1/4] 读取 TF 信息: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  CSV 中 TF 数: {len(df)}")

    tf_to_gsm = {}
    for _, row in df.iterrows():
        tf_name = str(row['TF_Name']).strip()
        gsm_id = str(row['GSM_ID']).strip()
        tf_to_gsm[tf_name.lower()] = gsm_id
        tf_to_gsm[tf_name.upper()] = gsm_id
        tf_to_gsm[tf_name.capitalize()] = gsm_id
        tf_to_gsm[tf_name] = gsm_id
    print(f"  TF→GSM 映射: {len(tf_to_gsm)}")

    # 解析聚类文件
    print(f"\n[2/4] 解析聚类文件: {T098_DIR}")
    cluster_files = sorted(
        [f for f in os.listdir(T098_DIR) if f.startswith('cluster_') and f.endswith('.txt')],
        key=lambda x: int(re.search(r'cluster_(\d+)', x).group(1))
    )
    print(f"  找到 {len(cluster_files)} 个簇文件")

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
    print(f"\n  总簇数: {total}")
    print(f"  MTL 簇: {len(mtl_clusters)} (size>=2)")
    print(f"  跳过 Single: {len(single_clusters)}")

    Path(TEMP_CSV_DIR).mkdir(parents=True, exist_ok=True)

    # 生成训练计划
    print(f"\n[3/4] 生成训练计划...")
    plan = []
    valid_tfs = 0
    skipped_tfs = []
    seq_idx = 0  # 新编号: 按数值序依次分配 cluster_001, cluster_002, ...

    for cluster_id, size, tf_names in cluster_data:
        if size < 2:
            continue  # 跳过单TF簇

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
            print(f"  ⚠️ Cluster {cluster_id}: 有效 TF < 2 (剩余 {len(tf_gsm_list)})，跳过")
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

    print(f"\n  计划任务: {len(plan)}")
    print(f"  计划中 TF: {sum(p['TF_Count'] for p in plan)}")

    size_counts = Counter(p['TF_Count'] for p in plan)
    print(f"\n簇大小分布:")
    for sz in sorted(size_counts.keys(), reverse=True):
        print(f"  Size {sz:3d}: {size_counts[sz]:3d} 簇")

    # 保存 JSON
    print(f"\n[4/4] 保存计划...")
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 计划: {OUTPUT_JSON}")
    print(f"  ✅ CSVs: {TEMP_CSV_DIR}/")

    print(f"\n{'='*70}")
    print(f"执行估算:")
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
    print(f"  估计总时间: {total_min} 分钟 ≈ {total_min/60:.1f} 小时")
    print(f"{'='*70}")

    if skipped_tfs:
        print(f"\n⚠️ 找不到 GSM_ID 的 TF (前 20):")
        for cid, tf in skipped_tfs[:20]:
            print(f"    Cluster {cid}: {tf}")
        if len(skipped_tfs) > 20:
            print(f"    ... 共 {len(skipped_tfs)} 个")


if __name__ == '__main__':
    main()
