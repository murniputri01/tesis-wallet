#!/usr/bin/env python3
"""
Module: enhancement/losses/exposure_control.py
Deskripsi: Exposure Control Loss (L_exp) dengan pembobotan seimbang untuk mencegah citra silau / overexposure.
"""

import torch                  # PyTorch untuk operasi tensor & perhitungan rata-rata patch
import torch.nn as nn         # PyTorch Neural Network module & AvgPool2d


class ExposureControlLoss(nn.Module):
    """
    Mengukur jarak kuadrat daerah lokal terhadap tingkat pencahayaan target E.
    Mencegah pencahayaan berlebihan (overexposure) dengan menyeimbangkan bobot exposure.

    Nilai E diambil dari configs/zero_dce_cbam.yaml. Default diturunkan ke 0.45 karena
    frame CCTV walet sudah semi-terang; target 0.55 mendorong latar (papan sirip) ke
    highlight dan menurunkan kontras walet terhadap latar.
    """
    def __init__(self, patch_size: int = 16, target_exposure: float = 0.45, use_pixel_weighting: bool = False, gamma: float = 1.0):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)           # Average pooling untuk menghitung rerata patch 16x16
        self.target_exposure = target_exposure
        self.use_pixel_weighting = use_pixel_weighting
        self.gamma = gamma

    def forward(self, enhanced_img: torch.Tensor, original_img: torch.Tensor = None) -> torch.Tensor:
        gray_enh = torch.mean(enhanced_img, dim=1, keepdim=True)  # Rata-rata intensitas citra enhanced (grayscale)
        patch_mean = self.pool(gray_enh)               # Hitung rata-rata kecerahan tiap patch 16x16

        if self.use_pixel_weighting and original_img is not None:
            gray_orig = torch.mean(original_img, dim=1, keepdim=True) # Rata-rata intensitas citra original
            darkness_weight = torch.pow(1.0 - gray_orig, self.gamma)  # Hitung bobot kegelapan patch D(x)^gamma
            patch_weight = self.pool(darkness_weight)  # Pooling bobot kegelapan
            patch_weight = torch.clamp(patch_weight, 0.2, 1.0) # Batasi bobot minimum di 0.2

            loss = torch.mean(patch_weight * torch.pow(patch_mean - self.target_exposure, 2))  # Loss terbobot kegelapan
        else:
            loss = torch.mean(torch.pow(patch_mean - self.target_exposure, 2)) # Loss selisih terhadap target exposure E

        return loss

