# Intel Image Classification — Computer Vision Final Project

> **Multi-class scene classification** using a custom CNN from scratch and MobileNetV2 transfer learning.
> Dataset: Intel Image Classification (Kaggle) — ~17,000 images, 6 classes.

---

## Repository Structure

```
files/
├── data/                             ← Dataset (see setup below)
│   ├── train/                        ← 12,632 training images
│   │   ├── buildings/
│   │   ├── forest/
│   │   ├── glacier/
│   │   ├── mountain/
│   │   ├── sea/
│   │   └── street/
│   ├── val/                          ← 1,402 validation images
│   │   └── <same 6 folders>
│   └── test/                         ← 3,000 test images
│       └── <same 6 folders>
│
├── notebooks/
│   ├── 01_CNN_from_scratch.ipynb      ← Full CNN pipeline (scratch)
│   └── 02_Transfer_Learning.ipynb     ← MobileNetV2 TL + comparison
│
├── models/                            ← Saved .keras model files (after training)
├── results/                           ← Plots, metrics JSON files
├── app/
│   └── app.py                         ← Streamlit web app (bonus deployment)
│
├── Intel_CV_Project_Report.docx       ← Written report
├── requirements.txt
└── README.md
```

---

## Dataset Setup

1. Download from Kaggle: `kaggle datasets download -d puneet6060/intel-image-classification`
2. Unzip the archive into the `data/` folder so you have `data/train/`, `data/val/`, `data/test/`.
3. Confirm the structure matches the tree above.

**Dataset Stats:**
| Split       | Images   |
|-------------|----------|
| Train       | ~12,632  |
| Validation  | ~1,402   |
| Test        | ~3,000   |
| **Total**   | **~17,034** |

**Classes:** `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebook 1 — CNN from Scratch

```bash
python -m jupyter notebook notebooks/01_CNN_from_scratch.ipynb
```

This notebook covers:
- Dataset exploration & class balance check (train/val/test splits)
- Corrupted image detection
- Data augmentation (rotation, flip, zoom, brightness)
- Custom 4-block CNN architecture (built layer by layer)
- Training with EarlyStopping + ReduceLROnPlateau
- Full evaluation: accuracy, precision, recall, F1, confusion matrix, ROC curves

### 3. Run Notebook 2 — Transfer Learning

```bash
python -m jupyter notebook notebooks/02_Transfer_Learning.ipynb
```

This notebook covers:
- MobileNetV2 backbone loaded **without** top
- Custom head explicitly defined layer by layer
- Phase 1: train head only (backbone frozen)
- Phase 2: fine-tune last 30 backbone layers (low LR)
- Full evaluation + **model comparison table**

### 4. Run the Web App (Bonus Deployment)

```bash
streamlit run app/app.py
```

Opens a browser at `http://localhost:8501` — upload any scene image and get predictions from both models.

---

## Model Architectures

### CNN from Scratch

| Block   | Layers |
|---------|--------|
| Block 1 | Conv2D(32) → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25) |
| Block 2 | Conv2D(64) → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25) |
| Block 3 | Conv2D(128) → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.30) |
| Block 4 | Conv2D(256) → BN → ReLU → MaxPool → Dropout(0.30) |
| Head    | Flatten → Dense(512) → BN → ReLU → Dropout(0.50) → Dense(6, softmax) |

### Transfer Learning (MobileNetV2)

| Component   | Details |
|-------------|---------|
| Backbone    | MobileNetV2, pre-trained on ImageNet, `include_top=False` |
| Phase 1     | Backbone frozen; only custom head trained |
| Phase 2     | Last 30 backbone layers unfrozen; fine-tuned at LR=1e-5 |
| Custom Head | GAP → Dense(512) → BN → ReLU → Dropout(0.40) → Dense(256) → BN → ReLU → Dropout(0.30) → Dense(6, softmax) |

---

## Preprocessing Summary

| Step                    | Details |
|-------------------------|---------|
| Resize                  | 150×150 (scratch) / 224×224 (TL) |
| Normalization           | [0,1] for scratch; [-1,1] for MobileNetV2 |
| Corrupted image check   | cv2.imread returns None check |
| Data Split              | Train: 12,632 / Val: 1,402 / Test: 3,000 (separate folders) |
| Augmentation            | Rotation ±20°, horizontal flip, zoom 15%, brightness [0.8–1.2], shifts |
| Class balance           | Checked — all 6 classes are well-balanced (~2,100 each in train) |

---

## Evaluation Metrics

Both notebooks compute:
- Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted)
- Confusion Matrix (heatmap)
- ROC / AUC curves (One-vs-Rest, one curve per class)
- Training vs. Validation accuracy & loss curves
- Misclassified examples visualisation

---

## Deployment (Bonus)

**Streamlit web app** (`app/app.py`) allows real-time inference:
- Upload any JPG/PNG image
- Get predictions from both models side-by-side
- Confidence bar charts for all 6 classes
- Model agreement indicator

---

## Team Members

*(Fill in your team names here)*

---

## Grading Compliance

| Requirement | Status |
|-------------|--------|
| Real dataset >= 1,000 images | ✅ ~17,034 images |
| No built-in library datasets | ✅ Downloaded from Kaggle |
| CNN from scratch (layer by layer) | ✅ Notebook 1 |
| Transfer learning manually built | ✅ Notebook 2 (no drag-and-drop) |
| Backbone freeze → unfreeze | ✅ Phase 1 & Phase 2 |
| Preprocessing documented | ✅ |
| Train/Val/Test split | ✅ Separate folders |
| Class balance addressed | ✅ |
| Evaluation metrics | ✅ All required metrics |
| Confusion matrix (heatmap) | ✅ |
| Loss & accuracy curves | ✅ |
| ROC/AUC (Bonus) | ✅ |
| Model comparison table | ✅ |
| Deployment (Bonus) | ✅ Streamlit app |
| README | ✅ This file |
| requirements.txt | ✅ |
