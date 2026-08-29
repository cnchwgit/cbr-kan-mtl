#!/usr/bin/env python3
"""
Compare CBR-KAN MTL v5 (t098) vs Single-task baselines
=======================================================
Produces the paired comparison CSV (7 metrics per TF) and prints
statistical summaries (overall, by cluster size, by single-task
performance strata, top/bottom transfer).

Usage:
    python scripts/compare_mtl_vs_single.py \
        --single-dir results/single_1043 \
        --mtl-dir results/mtl_v5_t098 \
        --cluster-csv-dir data/cluster_t098_csvs \
        --output-dir results
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def parse_args():
    p = argparse.ArgumentParser(description='CBR-KAN MTL v5 (t098) vs 单任务对比分析')
    p.add_argument('--single-dir', required=True, help='单任务基线结果目录 (每个TF一个子目录, 含 test_metrics.json)')
    p.add_argument('--mtl-dir', required=True, help='MTL v5 结果目录 (每个簇一个子目录, 含 test_tf_metrics.csv)')
    p.add_argument('--cluster-csv-dir', required=True, help='簇定义 CSV 目录 (cluster_*_t098.csv)')
    p.add_argument('--output-dir', default='results')
    p.add_argument('--tag', default='cbrkan_mtl_v5_t098', help='输出文件名标签')
    return p.parse_args()


def main():
    args = parse_args()
    SINGLE_DIR, MTL_DIR, CSV_DIR = args.single_dir, args.mtl_dir, args.cluster_csv_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("CBR-KAN MTL v5 (t098) vs 单任务对比 — Multi-Head MTL")
    print("t098: 更严格聚类 (threshold=0.98)，仅 MTL 簇 (size>=2)")
    print("生成时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)

    # ========== 1. 聚类映射 ==========
    tf_to_cluster = {}
    cluster_to_tfs = defaultdict(list)
    cluster_to_type = {}
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, 'cluster_*_t098.csv')))
    for fpath in csv_files:
        cid = int(re.search(r'cluster_(\d+)_t098\.csv', os.path.basename(fpath)).group(1))
        tdf = pd.read_csv(fpath)
        for _, row in tdf.iterrows():
            tf_to_cluster[(row['TF_Name'], row['GSM_ID'])] = cid
            cluster_to_tfs[cid].append((row['TF_Name'], row['GSM_ID']))
        cluster_to_type[cid] = 'MTL' if len(tdf) >= 2 else 'Single'
    print(f"[1/5] 聚类映射: {len(cluster_to_tfs)} clusters (t098)")

    # ========== 2. 单任务基线 ==========
    print("\n[2/5] 加载单任务基线...")
    single_results = {}
    for tf_dir in sorted(os.listdir(SINGLE_DIR)):
        jp = os.path.join(SINGLE_DIR, tf_dir, 'test_metrics.json')
        if not os.path.exists(jp):
            continue
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        # train_single.py writes metrics at top level (with tf_name/gsm_id),
        # older versions nested them under d['test_metrics'] — accept both.
        m = d.get('test_metrics') or d
        tf_name = d.get('TF_Name') or d.get('tf_name')
        gsm_id = d.get('GSM_ID') or d.get('gsm_id')
        if tf_name is None or gsm_id is None:
            continue
        single_results[(tf_name, gsm_id)] = {
            'roc_auc': m['roc_auc'],
            'pr_auc': m.get('pr_auc', 0),
            'f1': m.get('f1', 0),
            'accuracy': m.get('accuracy', 0),
            'precision': m.get('precision', 0),
            'recall': m.get('recall', 0),
            'mcc': m.get('mcc', 0),
        }
    print(f"   单任务: {len(single_results)} TF")

    # ========== 3. MTL v5 (t098) 结果 ==========
    print("\n[3/5] 加载 CBR-KAN MTL v5 t098 结果...")
    mtl_results = {}
    for cluster_dir in sorted(os.listdir(MTL_DIR)):
        if 'bak' in cluster_dir.lower():
            continue
        tcp = os.path.join(MTL_DIR, cluster_dir, 'test_tf_metrics.csv')
        if not os.path.exists(tcp):
            continue
        try:
            df = pd.read_csv(tcp)
        except Exception:
            continue

        m = re.search(r'cluster_(\d+)_', cluster_dir)
        cid = int(m.group(1)) if m else -1

        csv_path = os.path.join(CSV_DIR, f'cluster_{cid:03d}_t098.csv') if cid > 0 else None
        gsm_map = {}
        if csv_path and os.path.exists(csv_path):
            cdf = pd.read_csv(csv_path)
            gsm_map = dict(zip(cdf['TF_Name'], cdf['GSM_ID']))

        for _, row in df.iterrows():
            tf_name = row['TF_Name']
            gsm = gsm_map.get(tf_name, '')
            key = (tf_name, gsm)
            mtl_results[key] = {
                'roc_auc': row['Test_ROC_AUC'],
                'pr_auc': row.get('Test_PR_AUC', 0),
                'f1': row.get('Test_F1', 0),
                'accuracy': row.get('Test_Accuracy', 0),
                'precision': row.get('Test_Precision', 0),
                'recall': row.get('Test_Recall', 0),
                'mcc': row.get('Test_MCC', 0),
            }
    print(f"   MTL v5 t098: {len(mtl_results)} TF")

    # ========== 4. 合并对比（全面指标） ==========
    print("\n[4/5] 合并对比...")
    METRICS = [
        ('roc_auc', 'ROC_AUC'),
        ('pr_auc', 'PR_AUC'),
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1'),
        ('mcc', 'MCC'),
    ]

    rows = []
    for key, s in single_results.items():
        if key not in mtl_results:
            continue
        m = mtl_results[key]
        cid = tf_to_cluster.get(key, -1)
        ct = cluster_to_type.get(cid, 'Unknown')
        tc = len(cluster_to_tfs.get(cid, []))

        row = {'TF_Name': key[0], 'GSM_ID': key[1],
               'Cluster_ID': cid, 'Cluster_Type': ct,
               'TF_Count_in_Cluster': tc}

        for skey, label in METRICS:
            sv = s.get(skey, 0)
            mv = m.get(skey, 0)
            row[f'Single_{label}'] = sv
            row[f'MTL_{label}'] = mv
            row[f'Delta_{label}'] = mv - sv
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"   共同配对 TF: {len(df)}")

    out_csv = os.path.join(OUTPUT_DIR, f'comparison_{args.tag}.csv')
    df.to_csv(out_csv, index=False)
    print(f"   已保存 CSV: {out_csv}")

    out_csv_detailed = os.path.join(OUTPUT_DIR, f'comparison_{args.tag}_detailed.csv')
    df.to_csv(out_csv_detailed, index=False)
    print(f"   已保存详细 CSV: {out_csv_detailed}")

    # ========== 5. 统计分析 ==========
    print(f"\n{'='*70}")
    print(f"CBR-KAN MTL v5 (t098) vs 单任务 — 对比统计 ({args.tag})")
    print(f"配对 TF 数: {len(df)}")
    print(f"{'='*70}")

    print(f"\n📊 整体统计（ROC-AUC）")
    mean_s = df['Single_ROC_AUC'].mean()
    mean_m = df['MTL_ROC_AUC'].mean()
    mean_d = df['Delta_ROC_AUC'].mean()
    std_d = df['Delta_ROC_AUC'].std()
    n_pos = (df['Delta_ROC_AUC'] > 0).sum()
    n_neg = (df['Delta_ROC_AUC'] <= 0).sum()
    pct_pos = n_pos / len(df) * 100
    print(f"   单任务: {mean_s:.4f} ± {df['Single_ROC_AUC'].std():.4f}")
    print(f"   MTL:    {mean_m:.4f} ± {df['MTL_ROC_AUC'].std():.4f}")
    print(f"   Δ:      {mean_d:.4f} ± {std_d:.4f}")
    print(f"   正向:   {n_pos} ({pct_pos:.1f}%), 负向: {n_neg} ({n_neg/len(df)*100:.1f}%)")
    t, p = sp_stats.ttest_rel(df['Single_ROC_AUC'], df['MTL_ROC_AUC'])
    cd = mean_d / std_d if std_d > 0 else 0
    print(f"   配对 t 检验: t={t:.2f}, p={p:.2e}, d={cd:.2f}")

    print(f"\n📊 ΔROC 分布")
    for q in [5, 10, 25, 50, 75, 90, 95]:
        print(f"   P{q}: {np.percentile(df['Delta_ROC_AUC'], q):+.4f}")

    print(f"\n📊 按集群大小")
    for cat in ['2', '3', '4-5', '6-10', '11-20', '21-50', '51+']:
        if cat == '2':
            sub = df[df['TF_Count_in_Cluster'] == 2]
        elif cat == '3':
            sub = df[df['TF_Count_in_Cluster'] == 3]
        elif '-' in cat:
            lo, hi = map(int, cat.split('-'))
            sub = df[(df['TF_Count_in_Cluster'] >= lo) & (df['TF_Count_in_Cluster'] <= hi)]
        elif cat == '51+':
            sub = df[df['TF_Count_in_Cluster'] >= 51]
        else:
            sub = df[df['TF_Count_in_Cluster'] == int(cat)]
        if len(sub) == 0:
            continue
        pos_pct = (sub['Delta_ROC_AUC'] > 0).mean() * 100
        print(f"   {cat:6s}: N={len(sub):4d}, 单任务={sub['Single_ROC_AUC'].mean():.4f}, "
              f"MTL={sub['MTL_ROC_AUC'].mean():.4f}, Δ={sub['Delta_ROC_AUC'].mean():+.4f}, "
              f"正向={pos_pct:.1f}%")

    print(f"\n📊 按单任务 ROC-AUC 分层")
    for thr in [0.85, 0.90, 0.93, 0.95, 0.97]:
        lo = df[df['Single_ROC_AUC'] < thr]
        hi = df[df['Single_ROC_AUC'] >= thr]
        if len(lo) == 0 or len(hi) == 0:
            continue
        print(f"   <{thr:.2f}: N={len(lo):3d}, Δ={lo['Delta_ROC_AUC'].mean():+.4f}, "
              f"正向={((lo['Delta_ROC_AUC']>0).mean()*100):.1f}%  |  "
              f">={thr:.2f}: N={len(hi):3d}, Δ={hi['Delta_ROC_AUC'].mean():+.4f}, "
              f"正向={((hi['Delta_ROC_AUC']>0).mean()*100):.1f}%")

    print(f"\n📊 全面指标汇总")
    for label in ['ROC_AUC', 'PR_AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']:
        sv = df[f'Single_{label}'].mean()
        mv = df[f'MTL_{label}'].mean()
        dv = df[f'Delta_{label}'].mean()
        sd = df[f'Delta_{label}'].std()
        pos = (df[f'Delta_{label}'] > 0).sum()
        n = len(df)
        print(f"   {label:10s}: 单任务={sv:.4f}, MTL={mv:.4f}, Δ={dv:+.4f}±{sd:.4f}, 正向={pos}/{n}({pos/n*100:.1f}%)")

    print(f"\n📊 最佳正向迁移 (Top 10)")
    for _, r in df.nlargest(10, 'Delta_ROC_AUC').iterrows():
        print(f"   {r['TF_Name']:20s}: Δ={r['Delta_ROC_AUC']:+.4f} (单任务={r['Single_ROC_AUC']:.4f}, MTL={r['MTL_ROC_AUC']:.4f})")

    print(f"\n📊 最差负向迁移 (Bottom 10)")
    for _, r in df.nsmallest(10, 'Delta_ROC_AUC').iterrows():
        print(f"   {r['TF_Name']:20s}: Δ={r['Delta_ROC_AUC']:+.4f} (单任务={r['Single_ROC_AUC']:.4f}, MTL={r['MTL_ROC_AUC']:.4f})")

    print(f"\n{'='*70}")
    print("生成时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()
