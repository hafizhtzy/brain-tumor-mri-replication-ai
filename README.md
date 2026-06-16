# Tugas Replikasi AI - Brain Tumor MRI Classification

Nama: Muhammad Abdurrahman Hafizhuddin  
NIM: 241730038  
Topik: Klasifikasi tumor otak pada citra MRI menggunakan transfer learning.

## Paper Utama
Judul: MRI-Based Brain Tumor Classification using ResNet-50 and Optimized Softmax Regression  
Jurnal: Jurnal INFOTEL  
Tahun: 2024  
Metode acuan: ResNet-50 / deep feature extraction / optimized softmax regression  
Target replikasi: ResNet50 transfer learning untuk klasifikasi 4 kelas MRI.

## Dataset
Dataset yang disarankan: MRI Brain Tumor Dataset: 4-Class (7023 Images) dari Kaggle.
Kelas:
- Glioma
- Meningioma
- No-Tumor
- Pituitary

Letakkan dataset di folder `Dataset/` dengan struktur seperti ini:

```text
Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Langkah Cepat Eksekusi

1. Download dataset Kaggle dan extract ke folder `Dataset/`.
2. Buka terminal VS Code di folder project ini.
3. Jalankan:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

4. Cek GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

5. Jalankan notebook:

```powershell
jupyter notebook Notebook/brain_tumor_replication_pytorch.ipynb
```

atau jalankan source code:

```powershell
python SourceCode/train_brain_tumor_pytorch.py --data_dir Dataset --epochs 8 --batch_size 32
```

## Output yang Dihasilkan

Hasil eksperimen akan tersimpan di:

- `Hasil_Eksperimen/` untuk CSV classification report, model comparison, dan model weights.
- `Gambar_Grafik/` untuk grafik accuracy/loss, confusion matrix, dan Grad-CAM.
- `Draft_Jurnal/` untuk draft artikel ilmiah.

## Catatan Akademik

Penelitian ini merupakan studi replikasi dan pengembangan model klasifikasi citra MRI berbasis deep learning. Sistem ini tidak ditujukan sebagai alat diagnosis medis, tetapi sebagai eksperimen akademik pada bidang Artificial Intelligence dan Computer Vision.
