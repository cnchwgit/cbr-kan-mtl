#!/usr/bin/env python3
"""
Batch training: run CBR-KAN MTL v5 across all TF clusters (t098, threshold=0.98)
===============================================================================
Serial single-GPU runner. Skips clusters that already have a valid summary.json.

Usage (plan auto-built from the 204 cluster CSVs in data/cluster_t098_csvs/):
    python scripts/train_all_clusters.py \
        --results-base results/mtl_v5_t098 \
        --log-dir logs \
        --data-root /path/to/fasta_datasets

Or with an explicit plan JSON (legacy):
    python scripts/train_all_clusters.py \
        --plan <plan.json> \
        --results-base results/mtl_v5_t098 \
        --data-root /path/to/fasta_datasets
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_plan_from_csvs(csv_dir):
    """Build the training plan directly from the per-cluster TF-list CSVs.

    Each file data/cluster_t098_csvs/cluster_<ID>_t098.csv is one cluster;
    the CSV header is the first row, each data row is one TF.
    """
    plan = []
    for f in sorted(Path(csv_dir).glob('cluster_*_t098.csv')):
        m = re.search(r'cluster_(\d+)_t098\.csv$', f.name)
        if not m:
            continue
        cid = int(m.group(1))
        with open(f) as fh:
            n_tf = sum(1 for _ in csv.reader(fh)) - 1  # minus header
        if n_tf < 2:
            continue  # MTL clusters only (size >= 2)
        plan.append({'Cluster_ID': cid, 'Task_Type': 'MTL',
                     'TF_Count': n_tf, 'TF_List_CSV': str(f)})
    return plan


def parse_args():
    p = argparse.ArgumentParser(description='Batch train CBR-KAN MTL v5 over t098 clusters')
    p.add_argument('--plan', default=None,
                   help='Path to plan JSON (list of {Cluster_ID, TF_List_CSV}); '
                        'if omitted, auto-built from data/cluster_t098_csvs/')
    p.add_argument('--cluster-csv-dir', default=str(REPO_ROOT / 'data' / 'cluster_t098_csvs'),
                   help='Dir of per-cluster TF-list CSVs (used when --plan is omitted)')
    p.add_argument('--results-base', required=True, help='Output root dir')
    p.add_argument('--log-dir', default=str(REPO_ROOT / 'logs'))
    p.add_argument('--data-root', required=True, help='Dir containing {GSM_ID}.pos.fasta/.neg.fasta')
    p.add_argument('--train-script', default=str(REPO_ROOT / 'scripts' / 'train_mtl_cluster.py'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--timeout', type=int, default=7200, help='Per-cluster timeout (s)')
    return p.parse_args()


def main():
    args = parse_args()

    RESULTS_BASE = args.results_base
    LOG_DIR = args.log_dir
    os.makedirs(RESULTS_BASE, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.plan:
        plan = json.load(open(args.plan))
    else:
        plan = build_plan_from_csvs(args.cluster_csv_dir)
        print(f'[train_all_clusters] --plan 未指定，从 {args.cluster_csv_dir} 自动构建计划: '
              f'{len(plan)} 簇')
    total = len(plan)

    success = skipped = failed = 0
    failed_list = []
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f" CBR-KAN MTL v5 批量训练")
    print(f" 架构: True Multi-Head MTL (per-TF KAN heads)")
    print(f" 聚类: t098 (threshold=0.98), 仅 MTL 簇 (size>=2)")
    print(f" 参数: epochs={args.epochs}, batch={args.batch_size}, 60/20/20, seed=42")
    print(f" 总簇数: {total}")
    print(f" 输出: {RESULTS_BASE}")
    print(f" 开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for idx, task in enumerate(plan, 1):
        cid = task['Cluster_ID']
        tf_cnt = task['TF_Count']
        tf_csv = task['TF_List_CSV']
        out_dir = os.path.join(RESULTS_BASE, f'cluster_{cid:03d}_t098')
        summary_file = os.path.join(out_dir, 'summary.json')

        # 跳过已完成的
        if os.path.exists(summary_file):
            try:
                s = json.load(open(summary_file))
                if s.get('test') and s['test'].get('overall_roc_auc', 0) > 0.1:
                    skipped += 1
                    print(f"  [{idx:3d}/{total}] Cluster {cid:03d} ({tf_cnt:3d} TF) ✅ 跳过")
                    continue
            except Exception:
                pass

        progress_file = os.path.join(out_dir, '.training')
        if os.path.exists(progress_file):
            skipped += 1
            print(f"  [{idx:3d}/{total}] Cluster {cid:03d} ({tf_cnt:3d} TF) ⏳ 训练中 (标记存在)")
            continue

        cluster_log = os.path.join(LOG_DIR, f'cluster_{cid:03d}.log')
        os.makedirs(out_dir, exist_ok=True)
        Path(progress_file).touch()

        # 超大簇降低 batch size 防 OOM
        batch_size = args.batch_size
        max_per_class = 15000
        if tf_cnt >= 100:
            batch_size = 32
        if tf_cnt >= 200:
            max_per_class = 10000

        cmd = [
            sys.executable, args.train_script,
            '--tf-list', tf_csv,
            '--output-dir', out_dir,
            '--data-root', args.data_root,
            '--epochs', str(args.epochs),
            '--batch-size', str(batch_size),
            '--max-per-class', str(max_per_class),
            '--device', args.device,
        ]

        timeout = args.timeout
        if tf_cnt >= 100:
            timeout = max(timeout, 86400)   # 24h for huge clusters
        elif tf_cnt >= 10:
            timeout = max(timeout, 14400)   # 4h for medium clusters

        t0 = time.time()
        try:
            with open(cluster_log, 'w') as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                   cwd=str(REPO_ROOT), timeout=timeout)
            sec = time.time() - t0
            if r.returncode == 0 and os.path.exists(summary_file):
                s = json.load(open(summary_file))
                tauc = s.get('test', {}).get('overall_roc_auc', 0)
                avg_pt = s.get('test', {}).get('avg_per_tf_roc_auc', 0)
                epochs = s.get('training', {}).get('epochs_trained', 0)
                print(f"  [{idx:3d}/{total}] Cluster {cid:03d} ✅ {sec:.0f}s  Overall={tauc:.4f} avgPerTF={avg_pt:.4f} (ep={epochs})")
                success += 1
            else:
                print(f"  [{idx:3d}/{total}] Cluster {cid:03d} ❌ (code={r.returncode}) {sec:.0f}s")
                failed += 1
                failed_list.append(cid)
        except subprocess.TimeoutExpired:
            print(f"  [{idx:3d}/{total}] Cluster {cid:03d} ⏰ 超时 >{timeout//3600}h")
            failed += 1
            failed_list.append((cid, 'timeout'))
        except Exception as e:
            print(f"  [{idx:3d}/{total}] Cluster {cid:03d} ❌ {e}")
            failed += 1
            failed_list.append((cid, str(e)))
        finally:
            if os.path.exists(progress_file):
                os.remove(progress_file)

        # 每10个簇打印进度
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - t_start
            rate = idx / elapsed * 60 if elapsed > 0 else 0
            rem = (total - idx) / rate if rate > 0 else 0
            print(f"    📊 {idx}/{total} S={success} F={failed} SK={skipped} "
                  f"{rate:.1f}簇/分 剩余≈{rem:.0f}分 ({rem/60:.1f}h)")

    t_elap = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"✅ 完成! {success} OK, {failed} FAIL, {skipped} SKIP")
    print(f"耗时: {t_elap:.0f}s ({t_elap/3600:.1f}h)")
    if failed_list:
        print(f"失败: {failed_list[:30]}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
