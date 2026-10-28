#!/usr/bin/env python3
"""CBR-KAN MTL model.

Shared CBR-KAN backbone (Embedding -> Multi-CNN -> BiLSTM -> LayerNorm)
plus one independent KAN prediction head per TF ([256, 32, 2]), routed by
the TF identity carried by each training sample.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from src.efficient_kan import KAN  # noqa: E402


# ======================== CBR-KAN shared backbone (identical to the single-task version) ========================
class CBRKANSharedBackbone(nn.Module):
    def __init__(self, len_vocab, embed_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(len_vocab, embed_dim, padding_idx=len_vocab - 1)

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=128, kernel_size=11),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool1d(2),
        )

        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=160, kernel_size=9),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=160, out_channels=160, kernel_size=1),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=160, out_channels=160, kernel_size=5),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=160, out_channels=256, kernel_size=8),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool1d(2),
        )

        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=180, kernel_size=1),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=180, out_channels=256, kernel_size=8),
            nn.ReLU(True), nn.Dropout(0.2),
            nn.MaxPool1d(2),
        )

        self.BiLSTM = nn.LSTM(
            input_size=256, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, bias=True, dropout=0.2
        )

        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        out = self.embedding(x).permute(0, 2, 1)
        out1 = self.cnn1(self.dropout(out)).permute(0, 2, 1)
        out2 = self.cnn2(self.dropout(out)).permute(0, 2, 1)
        out3 = self.cnn3(self.dropout(out)).permute(0, 2, 1)
        out = torch.sum(torch.cat([out1, out2, out3], 1), 1).unsqueeze(1)
        lstm_out, _ = self.BiLSTM(out)
        out = out + lstm_out
        out = torch.sum(self.norm(out), 1)
        return out


class CBRKAN_MTL_MultiHead(nn.Module):
    """
    True Multi-Head MTL:
    - shared CBR-KAN backbone (shared by all TFs)
    - per-TF KAN heads (one independent prediction head per TF)
    """
    def __init__(self, tf_names, len_vocab, embed_dim=100):
        super().__init__()
        self.backbone = CBRKANSharedBackbone(len_vocab, embed_dim)
        self.tf_heads = nn.ModuleDict()
        for tf in tf_names:
            self.tf_heads[tf] = KAN([256, 32, 2])
        self.tf_list = list(tf_names)

    def forward(self, x, tf_names):
        """
        x: [B, L] token ids
        tf_names: list[str] of length B
        returns: [B, 2] logits
        """
        features = self.backbone(x)  # [B, 256]

        # if all samples belong to the same TF, process them as a batch
        if len(set(tf_names)) == 1:
            return self.tf_heads[tf_names[0]](features)

        # multiple TFs: group by TF and process in batches
        outputs = torch.zeros(len(tf_names), 2, device=features.device)
        tf_groups = defaultdict(list)
        for i, name in enumerate(tf_names):
            tf_groups[name].append(i)

        for name, indices in tf_groups.items():
            if len(indices) == 1:
                outputs[indices[0]] = self.tf_heads[name](features[indices[0]:indices[0]+1])
            else:
                outputs[indices] = self.tf_heads[name](features[indices])

        return outputs
