#!/usr/bin/env python3
"""
Single-task CBR-KAN training (baseline for the MTL vs Single comparison)
========================================================================
Trains one CBR-KAN model per TF on balanced positive/negative FASTA.
Same data split (60/20/20, seed 42) and hyper-parameters as the MTL run.

Usage:
    python scripts/train_single.py \
        --tf-name SP140 --gsm-id GSM2398967 \
        --data-root /path/to/fasta_datasets \
        --output results/single/SP140_GSM2398967
"""
import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util  # noqa: E402
from src.models.CBR_KAN import Model as CBRKAN  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sequences(pos_fasta, neg_fasta, max_samples=15000):
    def parse_fasta(filepath):
        sequences = []
        with open(filepath, 'r') as f:
            seq = ''
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if seq:
                        sequences.append(seq.upper())
                        if len(sequences) >= max_samples:
                            break
                    seq = ''
                else:
                    seq += line
            if seq and len(sequences) < max_samples:
                sequences.append(seq.upper())
        return sequences

    return parse_fasta(pos_fasta), parse_fasta(neg_fasta)


def seq_to_tokens(seq, kmer=3, max_len=200):
    """将DNA序列转为k-mer token ids (滑动窗口)"""
    bases = ['A', 'C', 'G', 'T']
    vocab = {}
    for b1 in bases:
        for b2 in bases:
            for b3 in bases:
                kmer_str = b1 + b2 + b3
                vocab[kmer_str] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)

    seq = seq[:max_len]
    tokens = []
    for i in range(len(seq) - kmer + 1):
        kmer_str = seq[i:i + kmer]
        tokens.append(vocab.get(kmer_str, vocab['<UNK>']))

    target_len = (max_len - kmer + 1) if max_len >= kmer else 1
    if len(tokens) < target_len:
        tokens.extend([vocab['<PAD>']] * (target_len - len(tokens)))
    elif len(tokens) > target_len:
        tokens = tokens[:target_len]

    return tokens, len(vocab)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, pos_seqs, neg_seqs, config):
        self.seq_len = getattr(config, 'pad_size', 200)
        self.vocab_size = getattr(config, 'len_vocab', 10000)
        self.sequences = []
        self.labels = []
        self.tokens_cache = {}

        for seq in pos_seqs:
            tokens, _ = seq_to_tokens(seq, kmer=3, max_len=self.seq_len)
            self.sequences.append(tokens)
            self.labels.append(1)
        for seq in neg_seqs:
            tokens, _ = seq_to_tokens(seq, kmer=3, max_len=self.seq_len)
            self.sequences.append(tokens)
            self.labels.append(0)

        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, n_batches = 0, 0
    all_preds, all_labels = [], []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.squeeze().to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
        all_preds.extend(probs)
        all_labels.extend(batch_y.cpu().numpy())
        n_batches += 1
    return total_loss / max(n_batches, 1), np.array(all_preds), np.array(all_labels)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, n_batches = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.squeeze().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())
            n_batches += 1
    return total_loss / max(n_batches, 1), np.array(all_preds), np.array(all_labels)


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


def main():
    parser = argparse.ArgumentParser(description='Train Single-Task CBR-KAN for one TF')
    parser.add_argument('--tf-name', required=True, help='TF name')
    parser.add_argument('--gsm-id', required=True, help='GEO series ID')
    parser.add_argument('--data-root', required=True, help='Dir containing {GSM_ID}.pos.fasta/.neg.fasta')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-per-class', type=int, default=15000)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    logger.info(f"开始训练 Single-Task: {args.tf_name} (GSM: {args.gsm_id})")
    logger.info(f"输出目录: {output_dir}")

    # 简单配置对象（与 baseline 配置等价）
    class Config:
        pass

    config = Config()
    config.embed = 100
    config.len_vocab = 10000
    config.pad_size = 200
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.device = args.device
    config.learning_rate = 0.001
    config.weight_decay = 1e-4
    config.embedding_pretrained = None

    logger.info(f"配置: epochs={config.num_epochs}, batch={config.batch_size}, "
                f"pad={config.pad_size}, vocab={config.len_vocab}")

    pos_fasta = Path(args.data_root) / f"{args.gsm_id}.pos.fasta"
    neg_fasta = Path(args.data_root) / f"{args.gsm_id}.neg.fasta"
    pos_seqs, neg_seqs = load_sequences(str(pos_fasta), str(neg_fasta), args.max_per_class)
    dataset = SequenceDataset(pos_seqs, neg_seqs, config)
    logger.info(f"数据: {len(pos_seqs)}正 + {len(neg_seqs)}负 = {len(dataset)}")

    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=args.seed, stratify=dataset.labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed, stratify=dataset.labels[temp_idx])

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=config.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=config.batch_size,
                            shuffle=False, num_workers=0, drop_last=True)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=config.batch_size,
                             shuffle=False, num_workers=0, drop_last=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = CBRKAN(config).to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    best_val_auc, best_epoch, patience_counter = 0.0, 0, 0
    history = []

    for epoch in range(config.num_epochs):
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = compute_metrics(train_labels, train_preds)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels, val_preds)
        scheduler.step(val_loss)

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss), 'val_loss': float(val_loss),
            'train_roc_auc': float(train_metrics['roc_auc']),
            'val_roc_auc': float(val_metrics['roc_auc']),
        })
        logger.info(f"Epoch [{epoch+1:2d}/{config.num_epochs}] Loss: {train_loss:.4f}/{val_loss:.4f}  "
                    f"ROC: {train_metrics['roc_auc']:.4f}/{val_metrics['roc_auc']:.4f}")

        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= 15:
                logger.info("早停")
                break

    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_preds)
    logger.info(f"\n测试: ROC={test_metrics['roc_auc']:.4f} PR={test_metrics['pr_auc']:.4f} "
                f"F1={test_metrics['f1']:.4f} Acc={test_metrics['accuracy']:.4f} MCC={test_metrics['mcc']:.4f}")

    test_metrics['tf_name'] = args.tf_name
    test_metrics['gsm_id'] = args.gsm_id
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)

    summary = {
        'tf_name': args.tf_name, 'gsm_id': args.gsm_id,
        'best_epoch': best_epoch, 'best_val_auc': float(best_val_auc),
        'test': test_metrics, 'args': vars(args),
        'date': datetime.now().isoformat(),
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("完成!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
