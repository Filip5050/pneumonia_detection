# Pneumonia Detection — Comparing ML Approaches on Chest X-rays

A comparison of three different machine learning approaches for detecting pneumonia 
from chest X-ray images. The goal is to explore how unsupervised feature extraction, 
linear dimensionality reduction, and end-to-end supervised learning compare on a 
real-world medical imaging task.

## Approaches

| Model | Description |
|---|---|
| Autoencoder + Logistic Regression | Convolutional autoencoder compresses images into a latent vector, logistic regression classifies on top |
| CNN (end-to-end) | Direct supervised learning — raw images in, prediction out |
| PCA + MLP | Linear dimensionality reduction followed by a neural network classifier |

## Results (Test set, 624 images)

| Model | Accuracy | NORMAL recall | PNEUMONIA recall |
|---|---|---|---|
| Autoencoder + LogReg | 75% | 0.33 | 0.99 |
| CNN | 76% | 0.36 | 0.99 |
| PCA + MLP | 77% | 0.42 | 0.98 |

> For medical use, **PNEUMONIA recall** is the most critical metric.
> Missing a sick patient is more dangerous than a false positive.
> On this metric, Autoencoder and CNN both achieve 0.99 — 
> meaning they miss only 1% of pneumonia cases.

## Dataset

Download the dataset from Kaggle and place it in the `data/` folder:

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

The expected folder structure is:
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).
```powershell
uv sync
```

That's it — uv will install all dependencies including the correct CUDA version of PyTorch automatically.

If you don't have a CUDA-capable GPU, edit `pyproject.toml` and remove the 
`[tool.uv.sources]` and `[[tool.uv.index]]` sections, then run:
```powershell
uv pip install torch torchvision
uv sync
```

## Project structure
```
pneumonia_detection/
├── Notebooks/
│   ├── pneumonia_detection_AE.ipynb    # Autoencoder + Logistic Regression
│   ├── pneumonia_detection_NN.ipynb    # CNN end-to-end
│   └── pneumonia_detection_PCA.ipynb   # PCA + MLP
├── data/                               # not included — download from Kaggle
├── pyproject.toml
├── uv.lock
└── README.md
```

## Key findings

- All three approaches perform similarly on this dataset (~75–77% accuracy)
- Class imbalance (~3x more pneumonia than normal cases) causes low NORMAL recall across all models
- PCA + MLP wins on raw accuracy, but Autoencoder and CNN are better suited for 
  medical use due to higher pneumonia recall
- A pretrained model like ResNet or EfficientNet would likely outperform all three approaches

## Next steps

- [ ] Transfer learning with ResNet or EfficientNet
- [ ] Fix class imbalance with weighted loss or oversampling
- [ ] Tune classification threshold to maximize pneumonia recall
- [ ] Train autoencoder longer with data augmentation