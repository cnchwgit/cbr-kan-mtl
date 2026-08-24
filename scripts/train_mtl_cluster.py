#!/usr/bin/env python3
"""
Train CBR-KAN Multi-Task Learning (MTL) on one TF cluster
==========================================================
True Multi-Head MTL:
  - Shared CBR-KAN backbone (Embedding → Multi-CNN → BiLSTM) across all TFs
  - One independent KAN prediction head per TF (routed by TF identity label)
  - Weak TFs benefit from shared features; strong TFs keep specialized heads

Architecture: CBRKANSharedBackbone(Embed→CNN1+CNN2+CNN3→BiLSTM→Norm)
              → per-TF KAN head ([256, 32, 2])

Data: balanced positive/negative ChIP-seq FASTA per TF
      (default split 60/20/20, seed 42)

Usage:
    python scripts/train_mtl_cluster.py \
        --tf-list data/cluster_t098_csvs/cluster_005_t098.csv \
        --output-dir results/cluster_005 \
        --data-root /path/to/fasta_datasets \
        --device cuda:0
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Make the repo root importable (contains src/ with the vendored efficient-kan)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.CBRKAN_MTL import (  # noqa: E402
    CBRKANSharedBackbone,
    CBRKAN_MTL_MultiHead,
)

# ======================== 路径 ========================
DEFAULT_DATA_ROOT = os.environ.get(
    "TFMTL_DATA_ROOT",
    "/path/to/datasets",  # replace with the dir containing {GSM_ID}.pos.fasta / .neg.fasta
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
    )
    return log_file


# ======================== 词表与分词（与单任务一致） ========================
def build_vocab():
    bases = ['A', 'C', 'G', 'T']
    vocab = {}
    for b1 in bases:
        for b2 in bases:
            for b3 in bases:
                vocab[b1 + b2 + b3] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)
    return vocab


def word_cut(dna, max_len=200):
    tokens = [dna[i:i+3] for i in range(min(len(dna)-2, max_len-2))]
    return tokens


def seq_to_ids(seq, vocab, pad_size=200):
    tokens = word_cut(seq, max_len=pad_size)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    target_len = pad_size - 2
    if len(ids) < target_len:
        ids += [vocab['<PAD>']] * (target_len - len(ids))
    return ids[:target_len]


# ======================== 数据加载（per-TF） ========================
def load_fasta_balanced(gsm_id, max_per_class=15000, data_root=None):
    data_root = Path(data_root or DEFAULT_DATA_ROOT)
    pos_file = data_root / f"{gsm_id}.pos.fasta"
    neg_file = data_root / f"{gsm_id}.neg.fasta"

    def read_fasta(fp, limit):
        seqs = []
        try:
            with open(fp) as f:
                for line in f:
                    if line.startswith('>'):
                        continue
                    s = line.strip().upper()
                    if s:
                        seqs.append(s)
                        if limit > 0 and len(seqs) >= limit:
                            break
        except FileNotFoundError:
            return []
        return seqs

    pos = read_fasta(pos_file, max_per_class)
    neg = read_fasta(neg_file, max_per_class)
    return pos, neg


class MultiTFSequenceDataset(Dataset):
    """
    多TF数据集：每个样本携带TF名称标签
    返回: (seq_ids, tf_name, label)
    """
    def __init__(self, tf_samples, vocab, pad_size=200):
        """
        tf_samples: list of (seq_str, tf_name, label)
        """
        self.vocab = vocab
        self.pad_size = pad_size
        self.samples = []
        for seq, tf_name, label in tf_samples:
            ids = seq_to_ids(seq, vocab, pad_size)
            self.samples.append((ids, tf_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_ids, tf_name, label = self.samples[idx]
        return torch.LongTensor(seq_ids), tf_name, torch.LongTensor([label])


def collate_multi_tf(batch):
    """自定义 collate_fn：处理字符串型 TF name"""
    seqs = torch.stack([item[0] for item in batch])
    tf_names = [item[1] for item in batch]
    labels = torch.cat([item[2] for item in batch])
    return seqs, tf_names, labels


# ======================== 训练与评估 ========================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_tfs = [], [], []
    n_batches = 0
    for batch_x, batch_tf_names, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.squeeze().to(device)
        optimizer.zero_grad()
        outputs = model(batch_x, batch_tf_names)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
        all_preds.extend(probs)
        all_labels.extend(batch_y.cpu().numpy())
        all_tfs.extend(batch_tf_names)
        n_batches += 1
    return total_loss / max(n_batches, 1), np.array(all_preds), np.array(all_labels), all_tfs


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_tfs = [], [], []
    with torch.no_grad():
        for batch_x, batch_tf_names, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.squeeze().to(device)
            outputs = model(batch_x, batch_tf_names)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())
            all_tfs.extend(batch_tf_names)
    return total_loss / max(len(dataloader), 1), np.array(all_preds), np.array(all_labels), all_tfs


def compute_metrics(y_true, y_pred):
    y_pred_bin = (y_pred > 0.5).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y_true, y_pred)),
        'pr_auc': float(average_precision_score(y_true, y_pred)),
        'mcc': float(matthews_corrcoef(y_true, y_pred_bin)),
        'accuracy': float(accuracy_score(y_true, y_pred_bin)),
        'precision': float(precision_score(y_true, y_pred_bin, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred_bin, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred_bin, zero_division=0)),
    }


def compute_per_tf_metrics(all_tfs, all_labels, all_preds):
    """计算每个TF的单独指标"""
    tf_data = defaultdict(lambda: {'labels': [], 'preds': []})
    for tf, lbl, pred in zip(all_tfs, all_labels, all_preds):
        tf_data[tf]['labels'].append(lbl)
        tf_data[tf]['preds'].append(pred)

    per_tf = {}
    for tf, data in tf_data.items():
        per_tf[tf] = compute_metrics(
            np.array(data['labels']),
            np.array(data['preds'])
        )
    return per_tf


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description='CBR-KAN MTL v5 (真 Multi-Head MTL)')
    parser.add_argument('--tf-list', required=True, help='聚类CSV路径 (columns: TF_Name,GSM_ID,Subgroup)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                        help='Dir containing {GSM_ID}.pos.fasta / .neg.fasta')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-per-class', type=int, default=15000)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr-patience', type=int, default=10)
    parser.add_argument('--embed-dim', type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logging.info("=" * 70)
    logging.info("CBR-KAN MTL v5 — 真 Multi-Head MTL")
    logging.info("=" * 70)
    logging.info(f"TF列表: {args.tf_list}")
    logging.info(f"数据目录: {args.data_root}")
    logging.info(f"参数: epochs={args.epochs} batch={args.batch_size} lr={args.lr}")
    logging.info(f"      wd={args.weight_decay} patience={args.patience} seed={args.seed}")
    logging.info(f"      max_per_class={args.max_per_class} embed_dim={args.embed_dim}")

    # 1. 加载TF列表
    tf_list = pd.read_csv(args.tf_list)
    tf_names = sorted(tf_list['TF_Name'].tolist())
    n_tf = len(tf_names)
    logging.info(f"TF数量: {n_tf}")
    logging.info(f"TF名称: {tf_names}")

    # 2. 每个TF单独加载数据
    vocab = build_vocab()
    tf_samples = {}  # {tf_name: {'pos': [...], 'neg': [...]}}
    for _, row in tf_list.iterrows():
        tf_name = row['TF_Name']
        p, n = load_fasta_balanced(row['GSM_ID'], args.max_per_class, data_root=args.data_root)
        tf_samples[tf_name] = {'pos': p, 'neg': n}
        logging.info(f"  {tf_name:12s}: {len(p):5d}正 + {len(n):5d}负 = {len(p)+len(n):5d}")

    total = sum(len(d['pos']) + len(d['neg']) for d in tf_samples.values())
    logging.info(f"总数据: {total} 样本（{n_tf}个TF）")

    # 3. 每个TF独立划分 60/20/20
    train_samples, val_samples, test_samples = [], [], []
    for tf_name, data in tf_samples.items():
        if len(data['pos']) == 0 or len(data['neg']) == 0:
            logging.warning(f"  ⚠️ {tf_name} 数据为空，跳过")
            continue
        # 构建该TF的样本列表 [(seq, tf_name, label)]
        all_s = [(seq, tf_name, 1) for seq in data['pos']]
        all_s += [(seq, tf_name, 0) for seq in data['neg']]
        all_labels = [1] * len(data['pos']) + [0] * len(data['neg'])

        indices = np.arange(len(all_s))
        train_i, temp_i = train_test_split(
            indices, test_size=0.4, random_state=args.seed, stratify=all_labels)
        val_i, test_i = train_test_split(
            temp_i, test_size=0.5, random_state=args.seed,
            stratify=np.array(all_labels)[temp_i])

        train_samples.extend([all_s[i] for i in train_i])
        val_samples.extend([all_s[i] for i in val_i])
        test_samples.extend([all_s[i] for i in test_i])

    logging.info(f"\n划分: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    logging.info(f"      60/20/20, seed={args.seed}")

    # 4. 创建数据集和数据加载器
    train_ds = MultiTFSequenceDataset(train_samples, vocab)
    val_ds = MultiTFSequenceDataset(val_samples, vocab)
    test_ds = MultiTFSequenceDataset(test_samples, vocab)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_multi_tf)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_multi_tf)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_multi_tf)

    # 5. 模型
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = CBRKAN_MTL_MultiHead(
        tf_names=tf_names, len_vocab=len(vocab), embed_dim=args.embed_dim
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.tf_heads.parameters())
    per_head = head_params // max(n_tf, 1)

    logging.info(f"\n设备: {device}")
    logging.info(f"总参数量: {total_params:,}")
    logging.info(f"  backbone: {backbone_params:,}")
    logging.info(f"  per-TF heads: {head_params:,} ({n_tf} heads × {per_head:,}/head)")
    logging.info(f"  平均每TF: {(backbone_params + per_head):,}")

    # 6. 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    logging.info(f"\n🚀 训练开始 ({args.epochs} epochs)...")
    t_start = time.time()

    for epoch in range(args.epochs):
        train_loss, train_preds, train_labels, _ = train_epoch(
            model, train_loader, criterion, optimizer, device)
        train_metrics = compute_metrics(train_labels, train_preds)

        val_loss, val_preds, val_labels, val_tfs = evaluate(
            model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels, val_preds)
        val_per_tf = compute_per_tf_metrics(val_tfs, val_labels, val_preds)
        val_per_tf_avg_auc = np.mean([m['roc_auc'] for m in val_per_tf.values()])

        scheduler.step(val_loss)

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_roc_auc': float(train_metrics['roc_auc']),
            'val_roc_auc': float(val_metrics['roc_auc']),
            'val_per_tf_avg_auc': float(val_per_tf_avg_auc),
        })

        logging.info(
            f"Epoch [{epoch+1:2d}/{args.epochs}] "
            f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
            f"ROC: {train_metrics['roc_auc']:.4f}/{val_metrics['roc_auc']:.4f}  "
            f"perTF: {val_per_tf_avg_auc:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停用 overall val_auc
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            logging.info(f"   ✅ 最佳模型 (val_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f"   ⏰ 早停 (patience={args.patience})")
                break

    elapsed = time.time() - t_start
    logging.info(f"\n训练完成: {elapsed:.0f}s ({elapsed/60:.1f}min, {elapsed/3600:.2f}h)")

    # 7. 测试集评估
    logging.info("\n📊 测试集评估...")
    if os.path.exists(output_dir / 'best_model.pt'):
        model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    test_loss, test_preds, test_labels, test_tfs = evaluate(
        model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_preds)
    test_per_tf = compute_per_tf_metrics(test_tfs, test_labels, test_preds)

    logging.info(f"\n{'='*70}")
    logging.info(f"整体测试指标:")
    logging.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logging.info(f"  PR-AUC:  {test_metrics['pr_auc']:.4f}")
    logging.info(f"  F1:      {test_metrics['f1']:.4f}")
    logging.info(f"  Acc:     {test_metrics['accuracy']:.4f}")
    logging.info(f"  Prec:    {test_metrics['precision']:.4f}")
    logging.info(f"  Recall:  {test_metrics['recall']:.4f}")

    # 打印每个TF的测试指标
    logging.info(f"\n每个TF测试指标:")
    for tf_name in sorted(test_per_tf.keys()):
        m = test_per_tf[tf_name]
        logging.info(f"  {tf_name:12s}: ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  F1={m['f1']:.4f}  "
                      f"Acc={m['accuracy']:.4f}")
    avg_per_tf = np.mean([m['roc_auc'] for m in test_per_tf.values()])
    logging.info(f"  平均 per-TF ROC: {avg_per_tf:.4f}")

    # 8. 保存结果
    # 每TF指标
    tf_metrics_rows = []
    for tf_name in sorted(test_per_tf.keys()):
        m = test_per_tf[tf_name]
        tf_metrics_rows.append({
            'TF_Name': tf_name,
            'Test_ROC_AUC': m['roc_auc'],
            'Test_PR_AUC': m['pr_auc'],
            'Test_F1': m['f1'],
            'Test_Accuracy': m['accuracy'],
            'Test_Precision': m['precision'],
            'Test_Recall': m['recall'],
            'Test_MCC': m['mcc'],
        })
    pd.DataFrame(tf_metrics_rows).to_csv(output_dir / 'test_tf_metrics.csv', index=False)

    # 训练历史
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)

    # 摘要
    summary = {
        'experiment': 'CBRKAN_MTL_v5_true_multi_head',
        'version': 5,
        'date': datetime.now().isoformat(),
        'num_tfs': n_tf,
        'total_params': total_params,
        'backbone_params': backbone_params,
        'per_tf_head_params': per_head,
        'data_split': {'train': 0.6, 'val': 0.2, 'test': 0.2, 'seed': args.seed},
        'training': {
            'epochs_trained': epoch + 1,
            'best_epoch': best_epoch,
            'best_val_auc': float(best_val_auc),
        },
        'test': {
            'overall_roc_auc': float(test_metrics['roc_auc']),
            'overall_pr_auc': float(test_metrics['pr_auc']),
            'avg_per_tf_roc_auc': float(avg_per_tf),
        },
        'per_tf_metrics': {tf: test_per_tf[tf] for tf in sorted(test_per_tf.keys())},
        'args': vars(args),
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logging.info(f"\n{'='*70}")
    logging.info(f"✅ 完成!")
    logging.info(f"  Overall Test ROC: {test_metrics['roc_auc']:.4f}")
    logging.info(f"  Avg Per-TF ROC:  {avg_per_tf:.4f}")
    logging.info(f"  结果保存至: {output_dir}")
    logging.info(f"{'='*70}")


if __name__ == '__main__':
    main()
