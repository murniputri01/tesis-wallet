#!/usr/bin/env python3
"""
Module: enhancement/dataloader.py
Deskripsi: Custom PyTorch Dataset dan DataLoader untuk membaca citra low-light dari folder dataset (splits/train, val, test).
           Mendukung normalisasi piksel [0, 1] dan penyesuaian ukuran citra (resizing) tanpa menggunakan label YOLO.
"""

from pathlib import Path      # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)
from typing import List, Tuple, Optional  # Penentu tipe data opsional, list, & tuple
from PIL import Image         # PIL (Pillow) untuk pembacaan berkas citra
import torch                  # PyTorch untuk penanganan tensor & PyTorch Dataset/DataLoader
from torch.utils.data import Dataset, DataLoader  # Base class Dataset & kelas loader batch PyTorch
import torchvision.transforms as T  # Transformasi tensor PyTorch (Resize, ToTensor, Compose)


class LowLightDataset(Dataset):
    """
    PyTorch Dataset untuk citra low-light tanpa paired ground-truth.
    """
    def __init__(
        self,
        images_dir: str,
        image_size: Optional[Tuple[int, int]] = (512, 512),
        transform: Optional[T.Compose] = None
    ):
        """
        Args:
            images_dir (str): Path ke folder gambar (misal: 'data/splits/train/images').
            image_size (Tuple[int, int], optional): Ukuran target (Height, Width) untuk resize. Jika None, gunakan ukuran asli.
            transform (T.Compose, optional): Transformasi tambahan PyTorch.
        """
        self.images_dir = Path(images_dir)
        self.image_size = image_size

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths = sorted([           # Mencari seluruh file gambar secara rekursif
            p for p in self.images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts
        ])

        if len(self.image_paths) == 0:
            print(f"[WARNING] DataLoader: Tidak ditemukan file gambar di '{self.images_dir.as_posix()}'")  # Warning jika data kosong

        # Basic transform: PIL Image to Tensor [0, 1]
        transforms_list = []
        if self.image_size is not None:
            transforms_list.append(T.Resize(self.image_size, antialias=True))  # Resize ukuran citra
        transforms_list.append(T.ToTensor())  # Mengubah PIL (0-255) -> FloatTensor [3, H, W] di [0.0, 1.0]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose(transforms_list)  # Penggabungan rangkaian transformasi

    def __len__(self) -> int:
        return len(self.image_paths)           # Mengembalikan total jumlah sampel citra

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = self.image_paths[index]
        try:
            with Image.open(img_path) as img:   # Buka berkas citra dengan PIL
                img_rgb = img.convert("RGB")   # Konversi citra ke mode RGB
                tensor_img = self.transform(img_rgb)  # Terapkan transformasi PyTorch
                return tensor_img
        except Exception as e:
            print(f"[ERROR] DataLoader: Gagal membaca gambar '{img_path.name}': {e}")
            # Fallback tensor 0 jika error
            channels = 3
            h = self.image_size[0] if self.image_size else 512
            w = self.image_size[1] if self.image_size else 512
            return torch.zeros((channels, h, w), dtype=torch.float32)  # Kembalikan tensor nol jika error


def create_dataloader(
    images_dir: str,
    batch_size: int = 8,
    image_size: Optional[Tuple[int, int]] = (512, 512),
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Helper function untuk membuat PyTorch DataLoader.
    """
    dataset = LowLightDataset(images_dir=images_dir, image_size=image_size)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader
