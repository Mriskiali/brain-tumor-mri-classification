# Brain Tumor MRI Classification — Hybrid Autoencoder + CNN

Deep learning system that classifies brain tumors from MRI scans into 4 classes — **glioma**, **meningioma**, **pituitary**, and **no tumor** — using a hybrid **Autoencoder + CNN** architecture, with an autoencoder reconstruction-error branch for anomaly detection.

**Test accuracy: 95%** (precision / recall / F1 = 0.95 across all classes) on a held-out set of 1,143 images from a public dataset of 7,022 MRI scans.

> Undergraduate thesis project: *"Brain Tumor Classification on MRI Images Using a Hybrid Autoencoder and CNN Architecture with CLAHE Preprocessing Optimization"* — Informatics, Universitas Gunadarma (2026).

## Results

### Classification report (test set, 1,143 images)

| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| glioma      | 0.96      | 0.94   | 0.95     | 288     |
| meningioma  | 0.91      | 0.89   | 0.90     | 265     |
| notumor     | 0.97      | 0.96   | 0.97     | 291     |
| pituitary   | 0.95      | 0.99   | 0.97     | 299     |
| **accuracy / macro / weighted** | | | **0.95** | 1143 |

### Autoencoder reconstruction quality

| Metric | Value |
|--------|-------|
| MSE  | 0.00135 |
| SSIM | 0.93 |

Reconstruction error doubles as an **anomaly signal**: inputs the autoencoder cannot reconstruct well are flagged for review before the CNN classification is shown.

## Pipeline

1. **Dataset** — public Brain Tumor MRI Dataset (7,022 images). Split: 60% train / 20% validation / 20% test.
2. **Preprocessing** — CLAHE contrast enhancement, grayscale conversion, resize to 128×128, pixel normalization to [0, 1].
3. **Architecture** — convolutional Autoencoder for representation learning and reconstruction-error anomaly detection, followed by a CNN classifier for 4-class softmax output.
4. **Training** — Google Colab (NVIDIA T4), up to 100 epochs, validation-tracked (val accuracy peaked at ~0.95 early in training).
5. **Evaluation** — confusion matrix + per-class precision/recall/F1 on the untouched test set.
6. **Deployment** — the trained model is served as a **Flask** web app on **Railway**: upload an MRI image, get real-time classification plus an anomaly indication from reconstruction error.

## Repository structure

```
├── notebooks/
│   └── penelitian_ilmiah_autoencoder_cnn.ipynb  # full training + evaluation pipeline
├── model/              # saved Keras/TensorFlow weights (.h5) — to be added
├── app/                # Flask web app for inference (deployed on Railway) — to be added
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/Mriskiali/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
pip install -r requirements.txt
python app/app.py   # run the inference web app locally
```

## Tech stack

Python · TensorFlow/Keras · OpenCV (CLAHE) · Flask · Railway · Google Colab (T4 GPU)

## Author

**Mu'afa Riski Ali** — [GitHub](https://github.com/Mriskiali) · [LinkedIn](https://www.linkedin.com/in/muafa-riski-ali-3114b536b/)
