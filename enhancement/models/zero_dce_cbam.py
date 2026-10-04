#!/usr/bin/env python3
"""
Module: enhancement/models/zero_dce_cbam.py
Deskripsi: Model Utama Penelitian (Skenario C): Zero-DCE + Modifikasi CBAM Berbasis Darkness Map.

Metode Utama Penelitian:
Input -> DCE-Net Feature Extraction -> Feature Map F -> Darkness Map D -> Dark-Aware Channel Attention
      -> Dark-Guided Spatial Attention -> Refined Feature Fcs -> Curve Estimation Head -> A(x)
      -> Zero-DCE Iterative Enhancement Curve -> Enhanced Image
"""

import torch                  # PyTorch untuk penanganan tensor & komputasi autograd
import torch.nn as nn         # PyTorch Neural Network module (Conv2d, ReLU, Tanh)
import torch.nn.functional as F  # PyTorch functional utilities

from .dark_attention import compute_darkness_map, DarkCBAMModule  # Impor Darkness Map & Modul CBAM


class ZeroDCE_CBAM(nn.Module):
    """
    Model Zero-DCE dengan Modifikasi CBAM Berbasis Darkness Map (Skenario C).
    """
    def __init__(self, n_iters: int = 8, nf: int = 32, dark_threshold: float = None):
        super(ZeroDCE_CBAM, self).__init__()
        self.n_iters = n_iters
        self.dark_threshold = dark_threshold

        # Feature extractor layers dari DCE-Net
        self.conv1 = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1)     # Layer konvolusi 1
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)    # Layer konvolusi 2
        self.conv3 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)    # Layer konvolusi 3
        self.conv4 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)    # Layer konvolusi 4
        self.conv5 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1) # Layer konvolusi 5 (Concatenation x3 & x4)
        self.conv6 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1) # Layer konvolusi 6 (Concatenation x2 & x5)

        # Modul Modifikasi CBAM dipasang pada fitur layer 6
        self.dark_cbam = DarkCBAMModule(channels=nf, reduction_ratio=16, spatial_kernel_size=7)

        # Curve Estimation Head
        self.conv7 = nn.Conv2d(nf * 2, n_iters * 3, kernel_size=3, stride=1, padding=1) # Estimasi 24 channel parameter kurva A(x)

        self.relu = nn.ReLU(inplace=True)               # Fungsi aktivasi ReLU
        self.tanh = nn.Tanh()                           # Fungsi aktivasi Tanh (-1 s.d. 1)

    def enhance_curve(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        enhanced = x
        for i in range(self.n_iters):                   # Loop iteratif pencerahan n = 1 s.d. 8
            a_n = A[:, i * 3:(i + 1) * 3, :, :]         # Ambil 3 channel parameter kurva per iterasi
            enhanced = enhanced + a_n * enhanced * (1.0 - enhanced)  # Persamaan kurva pencerahan kuadratik
        enhanced = torch.clamp(enhanced, 0.0, 1.0)      # Pembatasan piksel di rentang [0.0, 1.0]
        return enhanced

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x (Tensor): Citra input low-light [B, 3, H, W] di [0.0, 1.0].
            
        Returns:
            enhanced (Tensor): Citra hasil enhancement [B, 3, H, W]
            A (Tensor): Parameter kurva [B, 24, H, W]
            D (Tensor): Darkness map [B, 1, H, W]
            Mc_dark (Tensor): Channel attention map [B, C, 1, 1]
            Ms_dark (Tensor): Spatial attention map [B, 1, H, W]
        """
        # 1. Ekstraksi Darkness Map: D(x) = 1 - I(x)
        D = compute_darkness_map(x, threshold=self.dark_threshold)  # Hitung Darkness Map D(x)

        # 2. Feature Extraction (DCE-Net Layers 1-6)
        x1 = self.relu(self.conv1(x))                   # Ekstraksi fitur layer 1
        x2 = self.relu(self.conv2(x1))                  # Ekstraksi fitur layer 2
        x3 = self.relu(self.conv3(x2))                  # Ekstraksi fitur layer 3
        x4 = self.relu(self.conv4(x3))                  # Ekstraksi fitur layer 4

        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))  # Concat & ekstraksi layer 5
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))  # Feature Map F [B, 32, H, W]

        # 3. Modifikasi CBAM (Dark-Aware Channel & Dark-Guided Spatial Attention)
        Fcs, Mc_dark, Ms_dark = self.dark_cbam(x6, D)   # Aplikasi Dark CBAM pada fitur F6

        # 4. Curve Estimation Head
        A = self.tanh(self.conv7(torch.cat([x1, Fcs], dim=1)))  # Estimasi peta parameter A(x)

        # 5. Zero-DCE Enhancement Curve Iterations
        enhanced = self.enhance_curve(x, A)            # Aplikasi kurva pencerahan iteratif

        # Sanity Checks (Hanya dilakukan saat training untuk menghemat overhead CPU pada mode eval/inferensi)
        if self.training:
            assert not torch.isnan(enhanced).any(), "[ERROR] Output ZeroDCE_CBAM mengandung NaN"
            assert not torch.isinf(enhanced).any(), "[ERROR] Output ZeroDCE_CBAM mengandung Inf"
            assert (enhanced >= 0.0).all() and (enhanced <= 1.0).all(), "[ERROR] Nilai piksel di luar [0, 1]"

        return enhanced, A, D, Mc_dark, Ms_dark

