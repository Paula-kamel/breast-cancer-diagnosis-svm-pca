# Breast Cancer Diagnosis — SVM & K-NN with PCA Dimensionality Reduction

A machine learning study that classifies breast tumors as **benign** or **malignant** from clinical measurements, and investigates how Principal Component Analysis (PCA) affects model performance. The best model reaches **98.25% accuracy** with only **2 misclassifications out of 114 test cases**.

---

## Overview

Early, accurate diagnosis is critical in breast cancer outcomes. This project builds and compares classification models on diagnostic measurements taken from digitized images of fine-needle aspirate (FNA) samples, and studies the trade-off between model accuracy and feature-space dimensionality.

The work covers a full, leakage-safe ML workflow:

- Data exploration and a correlation-driven feature-selection rationale
- Normalization fitted on training data only (no test-set leakage)
- PCA for dimensionality reduction and visualization
- Model building and tuning for **SVM** (linear & RBF kernels) and **K-Nearest Neighbours**
- A like-for-like comparison of every model **before vs. after** PCA

## Dataset

- 569 samples, 30 numerical features describing cell-nucleus morphology and texture (radius, texture, perimeter, area, smoothness, concavity, concave points, symmetry, fractal dimension — each as mean, standard error, and worst value).
- Target: Benign (~63%) vs. Malignant (~37%). No missing values.
- Source: [Diagnostic Breast Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/diagnostic-breast-cancer-dataset)

## Approach

| Stage | What was done |
|-------|---------------|
| Preprocessing | Dropped the non-informative ID column; label-encoded the target (Benign→0, Malignant→1). |
| Feature selection | Ranked all 30 features by absolute point-biserial correlation with the label; chose a size feature (Mean Radius) and an independent shape feature (Worst Concave Points). |
| Normalization | StandardScaler fitted on the training set only, then applied to the test set. |
| Split | Stratified 80/20 train/test split (455 train / 114 test) preserving class balance. |
| PCA | Fitted on the normalized training set; retained 10 components capturing ≥95% of variance (a 66.7% reduction from 30 features). |
| Models | SVM (linear: C ∈ {2,10,27,52}; RBF: tuned C and γ) and K-NN (K ∈ {3,5,7,…}). |
| Evaluation | Accuracy, precision, recall, F1, and confusion matrices — with emphasis on recall, since a missed malignant case is the costliest error. |

## Key Results

**Best model overall: SVM with RBF kernel (C=10, γ=0.01) on the full 30 features**

| Metric | Score |
|--------|-------|
| Accuracy | 98.25% |
| Precision | 1.0000 (zero false positives) |
| Recall | 0.9524 |
| F1 | 0.9756 |
| Errors | 2 / 114 (both false negatives) |

**Before vs. after PCA (best model of each type):**

| Model | Features | Accuracy | F1 | Errors |
|-------|----------|----------|------|--------|
| SVM RBF (C=10, γ=0.01) | Original (30) | 0.9825 | 0.9756 | 2 |
| SVM Linear (C=2) | PCA (10) | 0.9737 | 0.9639 | 3 |
| K-NN (K=5) | Original (30) | 0.9561 | 0.9383 | 5 |
| K-NN (K=5) | PCA (10) | 0.9561 | 0.9383 | 5 |

### Findings

- Reducing the feature space by **66.7% (30 → 10 components)** cost **under 1.2% in F1** — PCA is a highly practical preprocessing step here.
- PCA *improved* the linear SVM slightly, by removing correlated/redundant dimensions.
- SVM consistently outperformed K-NN, which is expected given K-NN's sensitivity to high dimensionality.
- The RBF kernel benefited from the original space (non-linear boundaries); the linear kernel benefited from PCA's noise removal.

## Tech Stack

Python · scikit-learn · pandas · NumPy · matplotlib · seaborn · Jupyter

## How to Run

1. Download the dataset from the Kaggle link above and place the CSV next to the notebook (or update the path in the data-loading cell).
2. Open `paula_hany_breast_cancer_notebook.ipynb` in Jupyter and run all cells.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook paula_hany_breast_cancer_notebook.ipynb
```

---

*An independent machine learning study by Paula Hany Joseph.*
