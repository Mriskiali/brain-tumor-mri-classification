# Brain Tumor MRI Classification — Hybrid Autoencoder + CNN

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://brain-tumor-mri-classification-web.streamlit.app)
[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/riskiali/skripsi-brain-tumor-mri-classification)
[![Web App Code](https://img.shields.io/badge/Web_App-Source-181717?logo=github)](https://github.com/Mriskiali/brain-tumor-mri-classification-web)

> **Try it live:** upload an MRI scan and get a prediction with per-class probabilities and a model attention heatmap -> [brain-tumor-mri-classification-web.streamlit.app](https://brain-tumor-mri-classification-web.streamlit.app) (research/educational tool, not a medical device).

Deep learning system that classifies brain tumors from MRI scans into 4 classes — **glioma**, **meningioma**, **pituitary**, and **no tumor** — using a hybrid **Autoencoder + CNN** architecture, with an autoencoder reconstruction-error branch for anomaly detection.

## Research Evolution

This repository covers two stages of the same research line (Informatics, Universitas Gunadarma):

1. **Research Paper (Tulisan Ilmiah, 2025)** — `notebooks/penelitian_ilmiah_autoencoder_cnn.ipynb`
   Baseline pipeline: Autoencoder + CNN, grayscale/resize/normalization preprocessing. **95% test accuracy** (precision/recall/F1 = 0.95, 1,143 test images, MSE 0.00135, SSIM 0.93). Deployed as a Flask web app on Railway.

2. **Undergraduate Thesis (Skripsi, 2026)** — `notebooks/skripsi_hybrid_autoencoder_cnn_clahe.ipynb`
   Full upgrade: **CLAHE preprocessing optimization** (with ablation study), a 3-stage training pipeline (Autoencoder pretraining -> frozen-encoder classifier -> fine-tuning), full determinism controls, and **external validation on BraTS 2020**. Also explores Grad-CAM explainability and an end-to-end data audit (corrupt/duplicate/watermark detection).

## Thesis Results (Kaggle, GPU T4 x2, 1h 30m)

| Metric | Value | Target |
|---|---|---|
| SSIM (Autoencoder) | 0.8872 | > 0.85 |
| MSE (Autoencoder) | 0.003422 | < 0.01 |
| Test Accuracy (internal) | 82.14% (711 samples) | |
| External Accuracy (BraTS 2020, glioma) | 82.92% (240 samples) | |

Internal test-set classification report (thesis version, harder multi-dataset mix):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| glioma | 0.93 | 0.78 | 0.85 | 252 |
| meningioma | 0.80 | 0.65 | 0.71 | 176 |
| notumor | 0.68 | 0.91 | 0.78 | 69 |
| pituitary | 0.80 | 0.99 | 0.88 | 214 |

> The research-paper version (single clean dataset, simpler split) reached **95% accuracy** — see the first notebook. The thesis version is evaluated on a much harder, multi-source dataset mix (7,200 + 10,560 images + BraTS NIfTI volumes) with external validation, which is why absolute numbers differ.

## Pipeline (Thesis Version)

1. **Data** — 3 Kaggle datasets (Brain Tumor MRI, ishans24 Brain Tumor, BraTS 2020 NIfTI) via kagglehub
2. **Cleaning & audit** — corrupt/0-byte scan, duplicate detection, watermark removal, outlier analysis
3. **Preprocessing** — CLAHE optimization (ablation over clip/tile parameters), grayscale, resize 128x128, normalization
4. **Stage 1** — Autoencoder training (50 epochs, reconstruction + anomaly signal)
5. **Stage 2** — Classifier training on frozen encoder (30 epochs)
6. **Stage 3** — Fine-tuning (20 epochs, lr 1e-5)
7. **Evaluation** — confusion matrix, per-class precision/recall/F1, ROC/AUC, Grad-CAM
8. **External validation** — BraTS 2020 glioma subset

## Repository Structure

```
├── notebooks/
│   ├── penelitian_ilmiah_autoencoder_cnn.ipynb     # research paper: baseline AE+CNN (95% acc)
│   └── skripsi_hybrid_autoencoder_cnn_clahe.ipynb  # thesis: hybrid + CLAHE + external validation
├── webapp/
│   ├── app.py                  # Streamlit demo (live: brain-tumor-mri-classification-web.streamlit.app)
│   └── classifier_final.keras  # trained model weights (38 MB)

**Web demo (Streamlit):** [webapp/](https://github.com/Mriskiali/brain-tumor-mri-classification/tree/main/webapp) — live at [brain-tumor-mri-classification-web.streamlit.app](https://brain-tumor-mri-classification-web.streamlit.app). Run locally with `streamlit run webapp/app.py`. Includes Grad-CAM-style attention heatmap and per-class probabilities.
├── requirements.txt
└── README.md
```

## Getting Started

Open the notebooks in Google Colab or Kaggle (GPU recommended):

```bash
git clone https://github.com/Mriskiali/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Tech Stack

Python · TensorFlow/Keras · OpenCV (CLAHE) · scikit-learn · Flask · Railway · Kaggle (T4 GPU) · nibabel (NIfTI)

## Author

**Mu'afa Riski Ali** — [GitHub](https://github.com/Mriskiali) · [LinkedIn](https://www.linkedin.com/in/muafa-riski-ali-3114b536b/)
