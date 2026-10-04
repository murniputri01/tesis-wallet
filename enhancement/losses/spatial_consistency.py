#!/usr/bin/env python3
"""
Module: enhancement/losses/spatial_consistency.py
Deskripsi: Spatial Consistency Loss (L_spa) untuk mempertahankan kontras lokal antara citra input dan output.
"""

import torch                  # PyTorch untuk komputasi tensor & pembuatan kernel diferensiasi spasial
import torch.nn as nn         # PyTorch Neural Network module & nn.Parameter
import torch.nn.functional as F  # PyTorch functional untuk operasi konvolusi conv2d


class SpatialConsistencyLoss(nn.Module):
    """
    Menghitung konsistensi hubungan perbedaan intensitas antar tetangga lokal 4 arah (kiri, kanan, atas, bawah).
    """
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        # Kernel 4 arah diferensiasi spasial (kiri, kanan, atas, bawah)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4, stride=4)          # Average pooling 4x4 untuk reduksi skala patch

    def forward(self, input_img: torch.Tensor, enhanced_img: torch.Tensor) -> torch.Tensor:
        # Konversi ke grayscale (rata-rata channel)
        in_mean = torch.mean(input_img, dim=1, keepdim=True)    # Intensitas rata-rata citra input
        enh_mean = torch.mean(enhanced_img, dim=1, keepdim=True) # Intensitas rata-rata citra enhanced

        in_pool = self.pool(in_mean)                   # Reduksi spasial citra input
        enh_pool = self.pool(enh_mean)                 # Reduksi spasial citra enhanced

        D_org_left = F.conv2d(in_pool, self.weight_left, padding=1)    # Gradien spasial kiri input
        D_org_right = F.conv2d(in_pool, self.weight_right, padding=1)  # Gradien spasial kanan input
        D_org_up = F.conv2d(in_pool, self.weight_up, padding=1)        # Gradien spasial atas input
        D_org_down = F.conv2d(in_pool, self.weight_down, padding=1)    # Gradien spasial bawah input

        D_enh_left = F.conv2d(enh_pool, self.weight_left, padding=1)    # Gradien spasial kiri output
        D_enh_right = F.conv2d(enh_pool, self.weight_right, padding=1)  # Gradien spasial kanan output
        D_enh_up = F.conv2d(enh_pool, self.weight_up, padding=1)        # Gradien spasial atas output
        D_enh_down = F.conv2d(enh_pool, self.weight_down, padding=1)    # Gradien spasial bawah output

        loss_left = torch.pow(D_org_left - D_enh_left, 2)              # Selisih kuadrat arah kiri
        loss_right = torch.pow(D_org_right - D_enh_right, 2)           # Selisih kuadrat arah kanan
        loss_up = torch.pow(D_org_up - D_enh_up, 2)                    # Selisih kuadrat arah atas
        loss_down = torch.pow(D_org_down - D_enh_down, 2)              # Selisih kuadrat arah bawah

        return torch.mean(loss_left + loss_right + loss_up + loss_down) # Total rata-rata loss konsistensi spasial

