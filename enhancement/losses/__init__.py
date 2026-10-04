#!/usr/bin/env python3
"""
Module: enhancement/losses/__init__.py
Deskripsi: Gabungan Fungsi Kerugian (Non-Reference Zero-DCE Loss) dengan Pembagian Pixel-Wise Dark Weighting.
"""

import torch                  # PyTorch untuk penanganan tensor & perhitungan total loss
import torch.nn as nn         # PyTorch Neural Network module
from .spatial_consistency import SpatialConsistencyLoss  # Spatial Consistency Loss (L_spa)
from .exposure_control import ExposureControlLoss        # Exposure Control Loss (L_exp)
from .color_constancy import ColorConstancyLoss          # Color Constancy Loss (L_col)
from .illumination_smoothness import IlluminationSmoothnessLoss  # Illumination Smoothness Loss (L_tvA)


class ZeroDCELoss(nn.Module):
    """
    Total Loss Zero-DCE:
        L_total = λ_spa * L_spa + λ_exp * L_exp + λ_col * L_col + λ_tvA * L_tvA
    """
    def __init__(
        self,
        lambda_spa: float = 5.0,
        lambda_exp: float = 25.0,
        lambda_col: float = 5.0,
        lambda_tvA: float = 200.0,
        patch_size: int = 16,
        target_exposure: float = 0.55,
        use_pixel_weighting: bool = False,
        gamma: float = 1.0
    ):
        super(ZeroDCELoss, self).__init__()
        self.lambda_spa = lambda_spa
        self.lambda_exp = lambda_exp
        self.lambda_col = lambda_col
        self.lambda_tvA = lambda_tvA

        self.loss_spa = SpatialConsistencyLoss()       # Inisialisasi L_spa
        self.loss_exp = ExposureControlLoss(           # Inisialisasi L_exp
            patch_size=patch_size,
            target_exposure=target_exposure,
            use_pixel_weighting=use_pixel_weighting,
            gamma=gamma
        )
        self.loss_col = ColorConstancyLoss()           # Inisialisasi L_col
        self.loss_tvA = IlluminationSmoothnessLoss()   # Inisialisasi L_tvA

    def forward(self, input_img: torch.Tensor, enhanced_img: torch.Tensor, A: torch.Tensor):
        """
        Returns:
            total_loss (Tensor), dict_loss_components (dict)
        """
        l_spa = self.loss_spa(input_img, enhanced_img)  # Hitung loss konsistensi spasial L_spa
        l_exp = self.loss_exp(enhanced_img, input_img) # Hitung loss kontrol exposure L_exp
        l_col = self.loss_col(enhanced_img)            # Hitung loss konsistensi warna L_col
        l_tvA = self.loss_tvA(A)                       # Hitung loss kehalusan kurva L_tvA

        total_loss = (                                 # Kombinasi linier berbobot total loss
            self.lambda_spa * l_spa +
            self.lambda_exp * l_exp +
            self.lambda_col * l_col +
            self.lambda_tvA * l_tvA
        )

        loss_dict = {
            "loss_total": total_loss.item(),           # Ekstrak nilai float loss total
            "loss_spa": l_spa.item(),                  # Ekstrak nilai float L_spa
            "loss_exp": l_exp.item(),                  # Ekstrak nilai float L_exp
            "loss_col": l_col.item(),                  # Ekstrak nilai float L_col
            "loss_tvA": l_tvA.item(),                  # Ekstrak nilai float L_tvA
        }

        return total_loss, loss_dict

