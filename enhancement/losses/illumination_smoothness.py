#!/usr/bin/env python3
"""
Module: enhancement/losses/illumination_smoothness.py
Deskripsi: Illumination Smoothness Loss / Total Variation Loss (L_tvA) untuk menjaga kehalusan gradien parameter kurva A(x).
"""

import torch                  # PyTorch untuk penanganan tensor & perhitungan gradien Total Variation (TV)
import torch.nn as nn         # PyTorch Neural Network module


class IlluminationSmoothnessLoss(nn.Module):
    """
    Menghitung Total Variation (TV) horizontal dan vertikal pada peta parameter kurva A(x).
    """
    def __init__(self, tv_loss_weight: float = 1.0):
        super(IlluminationSmoothnessLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        batch_size = A.size(0)                         # Ambil ukuran batch B
        h_x = A.size(2)                                # Ambil tinggi H
        w_x = A.size(3)                                # Ambil lebar W

        count_h = self._tensor_size(A[:, :, 1:, :])    # Hitung total elemen vertikal
        count_w = self._tensor_size(A[:, :, :, 1:])    # Hitung total elemen horizontal

        h_tv = torch.pow((A[:, :, 1:, :] - A[:, :, :h_x - 1, :]), 2).sum()  # Total variation arah vertikal
        w_tv = torch.pow((A[:, :, :, 1:] - A[:, :, :, :w_x - 1]), 2).sum()  # Total variation arah horizontal

        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size  # Normalisasi loss TV

    def _tensor_size(self, t: torch.Tensor) -> int:
        return t.size(1) * t.size(2) * t.size(3)       # Perkalian elemen channel x H x W

