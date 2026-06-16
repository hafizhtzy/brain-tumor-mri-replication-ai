# Brain Tumor MRI Classification Replication Study

This repository contains the source code, notebook, experimental results, and draft article for an Artificial Intelligence course assignment: **Replication Study and Scientific Article Draft Writing**.

## Author

**Muhammad Abdurrahman Hafizhuddin**  
Department of Informatics, Faculty of Science  
UIN Sultan Maulana Hasanuddin Banten  
Email: Hafizhthericky123@gmail.com

## Research Title

**Replication and Extension of Brain Tumor MRI Classification Using Transfer Learning with ResNet50, EfficientNetB0, and MobileNetV2**

## Main Article

The main article replicated in this project is:

**MRI-Based Brain Tumor Classification using ResNet-50 and Optimized Softmax Regression**  
Published in **JURNAL INFOTEL**, 2024.

The article was selected because it is related to Artificial Intelligence, Deep Learning, Computer Vision, and Healthcare AI. It also contains an experimental method that can be replicated using a similar public brain tumor MRI dataset.

## Research Objective

The objectives of this replication study are:

1. To understand the methodology of a previous AI research article.
2. To implement a brain tumor MRI classification model using transfer learning.
3. To compare ResNet50 with additional models, namely EfficientNetB0 and MobileNetV2.
4. To evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.
5. To provide visual analysis using training curves, confusion matrix, and Grad-CAM.
6. To prepare a scientific article draft using the JURNAL INFOTEL template.

## Dataset

The dataset used in this project is a public Brain Tumor MRI dataset with four classes:

- Glioma
- Meningioma
- No-tumor
- Pituitary

Dataset summary:

| Split | Number of Images |
|---|---:|
| Training | 5,712 |
| Testing | 1,311 |
| Total | 7,023 |

The full dataset is stored in the Google Drive artifact folder because of its large size.

## Methods

This project uses transfer learning with three pretrained CNN models:

1. **ResNet50**  
   Used as the main replication model because the main article uses ResNet-50.

2. **EfficientNetB0**  
   Used as a comparative model to improve the analysis.

3. **MobileNetV2**  
   Used as a lightweight comparative model.

## Experimental Configuration

| Parameter | Value |
|---|---|
| Image size | 224 x 224 pixels |
| Batch size | 32 |
| Epochs | 8 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Cross Entropy Loss |
| Framework | PyTorch |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |

## Experimental Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| ResNet50 | 80.55% | 80.23% | 80.55% | 80.25% |
| EfficientNetB0 | 82.38% | 82.35% | 82.38% | 81.99% |
| MobileNetV2 | 80.55% | 80.13% | 80.55% | 80.11% |

The best model in this experiment is **EfficientNetB0**, with an accuracy of **82.38%**.

## Visualization

This project includes several result visualizations:

- Dataset distribution
- Sample MRI images
- Accuracy curves
- Loss curves
- Model accuracy comparison
- Confusion matrix
- Grad-CAM visualization

Grad-CAM is used only for academic interpretability and not for clinical diagnosis.

## GitHub Repository

The source code, notebook, experimental results, figures, and draft article are also available in this GitHub repository:

https://github.com/hafizhtzy/brain-tumor-mri-replication-ai

This repository is provided as additional documentation to support reproducibility.

## Repository Structure

```text
brain-tumor-mri-replication-ai/
├── SourceCode/
├── Notebook/
├── Hasil_Eksperimen/
├── Gambar_Grafik/
├── Draft_Jurnal/
├── README.md
└── requirements.txt


