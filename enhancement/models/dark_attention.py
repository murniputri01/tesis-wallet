#!/usr/bin/env python3
"""
Module: enhancement/models/dark_attention.py
Deskripsi: Modifikasi Convolutional Block Attention Module (CBAM) Berbasis Darkness Map.

Komponen:
  1. Darkness Map Extractor: D(x) = 1 - I(x)
  2. Dark-Aware Channel Attention: Pooling berbobot D(x) untuk memprioritaskan fitur daerah gelap.
  3. Dark-Guided Spatial Attention: Spatial attention 7x7 yang dipandu langsung oleh D(x).
"""

import torch                     # PyTorch untuk penanganan tensor, komputasi autograd, & operasi matriks
import torch.nn as nn            # PyTorch Neural Network module (Conv2d, Sequential, Sigmoid, ReLU)
import torch.nn.functional as F  # PyTorch functional utilities (interpolate)


def compute_darkness_map(x: torch.Tensor, threshold: float = None) -> torch.Tensor:
    """
    Formulasi Darkness Map: D(x) = 1 - I(x)
    
    Args:
        x (Tensor): Input citra [B, 3, H, W] di [0.0, 1.0].
        threshold (float, optional): Threshold τ untuk binarisasi/pemotongan area terang jika diperlukan.
    
    Returns:
        D (Tensor): Darkness map [B, 1, H, W] di [0.0, 1.0] (1 = makin gelap, 0 = makin terang).
    """
    # Intensitas rata-rata channel (Grayscale)
    intensity = torch.mean(x, dim=1, keepdim=True)      # Rata-rata 3 channel RGB sepanjang dim=1
    D = 1.0 - intensity                                 # Hitung peta kegelapan D(x) = 1.0 - I(x)

    if threshold is not None:
        # Jika threshold diset, nol-kan bobot area yang terang (D < threshold)
        D = torch.where(D >= threshold, D, torch.zeros_like(D))  # Potong area terang jika D < threshold

    return torch.clamp(D, 0.0, 1.0)                     # Clamp nilai D di rentang [0.0, 1.0]


class DarkAwareChannelAttention(nn.Module):
    """
    1. DARK-AWARE CHANNEL ATTENTION
    Menggunakan darkness map D(x) sebagai bobot pada pooling channel (AvgPool_dark & MaxPool_dark).
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super(DarkAwareChannelAttention, self).__init__()
        self.channels = channels
        reduced_channels = max(4, channels // reduction_ratio)

        # Shared MLP
        self.mlp = nn.Sequential(                       # Shared MLP 1x1 Conv -> ReLU -> 1x1 Conv
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()                    # Aktivasi Sigmoid [0.0, 1.0]

    def forward(self, F_in: torch.Tensor, D: torch.Tensor):
        """
        Args:
            F_in (Tensor): Feature map [B, C, H, W]
            D (Tensor): Darkness Map [B, 1, H, W]
        """
        B, C, H, W = F_in.shape

        # Rescale D jika ukuran spatial berbeda dengan F_in
        if D.shape[2:] != (H, W):
            D = F.interpolate(D, size=(H, W), mode="bilinear", align_corners=False)  # Interpolasi bilinear

        # Dark-Aware Average Pooling: Σ [D(x) * F(x)] / (Σ D(x) + ε)
        D_sum = torch.sum(D, dim=[2, 3], keepdim=True) + 1e-6                   # Sum Darkness Map sepanjang H, W
        avg_dark = torch.sum(F_in * D, dim=[2, 3], keepdim=True) / D_sum        # Average pooling terbobot D(x)

        # Dark-Aware Max Pooling: max [D(x) * F(x)]
        max_dark, _ = torch.max(F_in * D, dim=3, keepdim=True)                  # Max pooling sepanjang W
        max_dark, _ = torch.max(max_dark, dim=2, keepdim=True)                  # Max pooling sepanjang H

        # Channel Attention Map Mc_dark
        mlp_avg = self.mlp(avg_dark)                    # Ekstraksi fitur MLP dari avg_dark
        mlp_max = self.mlp(max_dark)                    # Ekstraksi fitur MLP dari max_dark
        Mc_dark = self.sigmoid(mlp_avg + mlp_max)       # Sigmoid pembobotan channel Mc_dark [B, C, 1, 1]

        # Feature Refinement: Fc = Mc_dark * F_in
        Fc = Mc_dark * F_in                             # Penguatan fitur channel Fc
        return Fc, Mc_dark


class DarkGuidedSpatialAttention(nn.Module):
    """
    2. DARK-GUIDED SPATIAL ATTENTION
    Spatial attention 7x7 yang dipandu oleh darkness map D(x).
    """
    def __init__(self, kernel_size: int = 7):
        super(DarkGuidedSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size harus 3 atau 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False) # Conv 7x7 spatial
        self.sigmoid = nn.Sigmoid()                    # Aktivasi Sigmoid [0.0, 1.0]

    def forward(self, Fc: torch.Tensor, D: torch.Tensor):
        """
        Args:
            Fc (Tensor): Feature hasil channel attention [B, C, H, W]
            D (Tensor): Darkness Map [B, 1, H, W]
        """
        B, C, H, W = Fc.shape

        # Rescale D jika ukuran spatial berbeda
        if D.shape[2:] != (H, W):
            D = F.interpolate(D, size=(H, W), mode="bilinear", align_corners=False)  # Interpolasi bilinear

        # Average & Max pooling sepanjang channel
        avg_out = torch.mean(Fc, dim=1, keepdim=True)   # Rata-rata sepanjang channel warna (dim=1)
        max_out, _ = torch.max(Fc, dim=1, keepdim=True) # Maksimum sepanjang channel warna (dim=1)
        concat = torch.cat([avg_out, max_out], dim=1)   # Gabung fitur avg & max [B, 2, H, W]

        # Spatial Map dipandu oleh D(x): Ms_dark(x) = Sigmoid(Conv7x7([AvgPool_c, MaxPool_c])) * D(x)
        spatial_map = self.sigmoid(self.conv(concat))   # Spatial attention map murni [B, 1, H, W]
        Ms_dark = spatial_map * D                       # Peta atensi dipandu langsung oleh D(x)

        # Feature Refinement Akhir (Residual Gating): Fcs = Fc * (1 + Ms_dark)
        Fcs = Fc * (1.0 + Ms_dark)                      # Residual gating penguatan area gelap
        return Fcs, Ms_dark


class DarkCBAMModule(nn.Module):
    """
    Modul Modifikasi CBAM Lengkap yang menggabungkan Dark-Aware Channel Attention & Dark-Guided Spatial Attention.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super(DarkCBAMModule, self).__init__()
        self.channel_attn = DarkAwareChannelAttention(channels=channels, reduction_ratio=reduction_ratio)   # Modul Channel Attention
        self.spatial_attn = DarkGuidedSpatialAttention(kernel_size=spatial_kernel_size)                     # Modul Spatial Attention

    def forward(self, F_in: torch.Tensor, D: torch.Tensor):
        """
        Returns:
            Fcs (Tensor): Refined feature map [B, C, H, W]
            Mc_dark (Tensor): Channel attention map [B, C, 1, 1]
            Ms_dark (Tensor): Spatial attention map [B, 1, H, W]
        """
        Fc, Mc_dark = self.channel_attn(F_in, D)       # Eksekusi Dark-Aware Channel Attention
        Fcs, Ms_dark = self.spatial_attn(Fc, D)        # Eksekusi Dark-Guided Spatial Attention
        return Fcs, Mc_dark, Ms_dark

