#!/usr/bin/env python3
"""
Module: enhancement/losses/color_constancy.py
Deskripsi: Color Constancy Loss (L_col) berdasarkan prinsip Gray-World untuk mencegah perubahan warna tak alami.
"""

import torch                  # PyTorch untuk penanganan tensor & perhitungan mean/pow/sqrt
import torch.nn as nn         # PyTorch Neural Network module Base Class


class ColorConstancyLoss(nn.Module):
    """
    Menghitung selisih kuadrat rata-rata antar pasang channel warna RGB (R-G, R-B, G-B).
    """
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()     # Inisialisasi parent class nn.Module

    def forward(self, enhanced_img: torch.Tensor) -> torch.Tensor:
        r_mean = torch.mean(enhanced_img[:, 0, :, :], dim=[1, 2])  # Rata-rata intensitas channel Merah (R)
        g_mean = torch.mean(enhanced_img[:, 1, :, :], dim=[1, 2])  # Rata-rata intensitas channel Hijau (G)
        b_mean = torch.mean(enhanced_img[:, 2, :, :], dim=[1, 2])  # Rata-rata intensitas channel Biru (B)

        d_rg = torch.pow(r_mean - g_mean, 2)            # Jarak selisih kuadrat Merah vs Hijau (R-G)^2
        d_rb = torch.pow(r_mean - b_mean, 2)            # Jarak selisih kuadrat Merah vs Biru (R-B)^2
        d_gb = torch.pow(g_mean - b_mean, 2)            # Jarak selisih kuadrat Hijau vs Biru (G-B)^2

        loss = torch.mean(torch.sqrt(d_rg + d_rb + d_gb + 1e-6))  # Total rata-rata loss konstansi warna Gray-World
        return loss                                    # Mengembalikan tensor loss L_col

