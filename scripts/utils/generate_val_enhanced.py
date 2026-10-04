#!/usr/bin/env python3
import sys                 # Pengendali eksekusi sistem & manipulasi modul sys.path
from pathlib import Path   # Pengelola jalur folder/file lintas sistem operasi (Windows/Linux)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Lokasi akar proyek
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))              # Sisipkan jalur akar proyek ke sys.path

import torch               # PyTorch untuk penanganan tensor & eksekusi model deep learning
import numpy as np         # NumPy untuk komputasi matriks piksel & manipulasi array
from PIL import Image, ImageFilter  # PIL (Pillow) untuk pembacaan citra, resample, & UnsharpMask
import torchvision.transforms as T  # Transformasi tensor PyTorch (ToTensor)
from enhancement.models.zero_dce_cbam import ZeroDCE_CBAM  # Impor arsitektur model Zero-DCE + CBAM

def parse_group_id(filename: str) -> str:
    """Meng-ekstrak ID grup/sumber video dari nama file."""
    if "_frame_" in filename:
        return filename.split("_frame_")[0]            # Memotong string nama file berdasarkan pemisah '_frame_'
    return "default_group"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Deteksi perangkat GPU atau CPU
    ckpt_path = Path("checkpoints/zero_dce_cbam/best_zero_dce_cbam.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)  # Memuat checkpoint memori model .pth
    
    cfg = checkpoint.get("config", {})                 # Membaca dictionary konfigurasi dari checkpoint
    n_iters = cfg.get("model", {}).get("n_iters", 8)   # Jumlah iterasi kurva pencerahan (n = 8)
    nf = cfg.get("model", {}).get("nf", 32)            # Jumlah filter konvolusi awal (nf = 32)
    dark_thresh = cfg.get("model", {}).get("dark_threshold", None)
    
    model = ZeroDCE_CBAM(n_iters=n_iters, nf=nf, dark_threshold=dark_thresh).to(device)  # Inisialisasi model
    model.load_state_dict(checkpoint["model_state_dict"])  # Memuat bobot memori terlatikan
    model.eval()                                       # Set model ke mode evaluasi/inferensi

    val_in = Path("data/splits/val/images")
    val_out = Path("data/enhanced/zero_dce_cbam/val/images")
    val_out.mkdir(parents=True, exist_ok=True)         # Membuat folder tujuan output validasi

    to_tensor = T.ToTensor()                          # Inisialisasi converter PIL Image ke Tensor [0, 1]

    with torch.no_grad():                             # Non-aktifkan komputasi gradien autograd
        for img_p in sorted(val_in.rglob("*.jpg")):    # Loop membaca seluruh citra validasi .jpg
            group_id = parse_group_id(img_p.name)
            sub_val_out = val_out / group_id
            sub_val_out.mkdir(parents=True, exist_ok=True)  # Membuat sub-folder grup video
            out_p = sub_val_out / img_p.name
            with Image.open(img_p) as pil_img:         # Buka file gambar dengan PIL
                rgb = pil_img.convert("RGB")           # Konversi citra ke mode RGB
                orig_w, orig_h = rgb.size
                t_in = to_tensor(rgb).unsqueeze(0).to(device)  # Ubah ke tensor 4D [1, 3, H, W]
                
                enh_t, _, _, _, _ = model(t_in)        # Inferensi model Zero-DCE + CBAM
                enh_f = enh_t[0].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)  # Tensor ke NumPy
                
                final_w, final_h = orig_w * 2, orig_h * 2
                channels = [                           # Resize per channel dengan interpolasi Lanczos
                    np.asarray(
                        Image.fromarray(enh_f[:, :, c], mode="F").resize((final_w, final_h), Image.Resampling.LANCZOS),
                        dtype=np.float32
                    )
                    for c in range(3)
                ]
                resized_f = np.stack(channels, axis=2)  # Tumpuk kembali 3 channel warna
                enh_np = np.clip(resized_f * 255.0, 0, 255).astype(np.uint8)  # Kuantisasi float32 ke uint8 [0, 255]
                
                out_img = Image.fromarray(enh_np).filter(  # Terapkan UnsharpMask untuk penajaman visual
                    ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3)
                )
                out_img.save(out_p)                    # Simpan gambar ter-enhance ke disk
                print(f"[SAVED] {out_p.as_posix()} ({final_w}x{final_h})")

if __name__ == "__main__":                             # Memastikan main() berjalan hanya jika dieksekusi langsung
    main()

