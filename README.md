# Handwritten Alphanumeric Recognition

A deep-learning project to classify **handwritten English characters and digits** (62 classes: A-Z, a-z, 0-9) using Convolutional Neural Networks (CNNs) trained and evaluated on Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sachinadk2011/handwritten-alphanumeric-recognition/blob/main/handwritten_recognition.ipynb)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Models](#models)
- [Model Comparison](#model-comparison)
- [Key Findings](#key-findings)
- [Future Plans](#future-plans)
- [Dependencies](#dependencies)

---

## Project Overview

This project explores multi-class classification of 62 handwritten alphanumeric characters using CNN architectures built with Keras/TensorFlow. The primary challenges tackled are:

- **Overfitting** on a small original dataset (~2,000 training images)
- **Generalization** improvement via data augmentation (~13,000 images)
- **Architecture comparison** across 6 CNN variants
- **Hybrid approach** testing: CNN feature extractor + Random Forest classifier head

Training runs on **Google Colab** with the dataset stored on **Google Drive**. Throughout all experiments, `ModelCheckpoint` saves the best weights by `val_loss` and `EarlyStopping` (patience=10, `restore_best_weights=True`) halts training early. All reported results reflect the model at its most reliable checkpoint, not its peak training accuracy.

---

## Dataset

**Source:** Handwritten English Characters and Digits dataset

| Split | Images (exact) |
|---|---|
| Training (original) | 2,728 |
| Validation | 682 |
| Augmented training | 13,640 |

- **Classes:** 62 -- A-Z (uppercase), a-z (lowercase), 0-9 (digits)
- **Image size:** 128 x 128, RGB
- **Batch size:** 32 (64 for augmented set)
- **Augmentation:** small rotations, shifts, and zooms via Keras `image_dataset_from_directory`
- **Normalization:** pixel values scaled to `[0, 1]`
- **Reproducibility:** global seed 42 set across `os`, `random`, `numpy`, and TensorFlow

> **Note:** The augmented dataset (`augmented_images/`) is not pushed to GitHub due to size. Generate it from the original data or upload it to Google Drive as described in the notebook.

---

## Setup

### Option 1 - Google Colab (Recommended)

1. Click the **Open in Colab** badge above.
2. Mount Google Drive when prompted.
3. Place the dataset on your Drive at:

```
MyDrive/handwritten_english_character_and_digits/
handwritten-english-characters-and-digits/combined_folder/
├── train/
└── test/
```

4. Run all cells top to bottom.

### Option 2 - Local (Windows)

**Prerequisite:** Python 3.9+ available on PATH.

**Automated setup (recommended):**

```powershell
git clone https://github.com/sachinadk2011/handwritten-alphanumeric-recognition.git
cd handwritten-alphanumeric-recognition
task setup
```

`task setup` (via `taskipy`) runs `setup_venv.ps1`, which:

- Creates a `venv/` virtual environment
- Installs all packages from `requirements.txt`
- On re-runs: detects and syncs any newly installed packages back into `requirements.txt`

**Manual setup:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Then open `handwritten_recognition.ipynb`.

> **Note:** Notebook data paths point to `/content/drive/...`. Update them to your local dataset paths before running locally.

---

## Project Structure

```
handwritten-alphanumeric-recognition/
├── handwritten_recognition.ipynb        # Main notebook: training, evaluation, analysis
├── requirements.txt                     # Pinned Python dependencies
├── pyproject.toml                       # taskipy task definitions (setup / start)
├── setup_venv.ps1                       # Smart Windows venv setup script
├── augmented_images/
│   └── augmented_images1/               # Augmented training images (class subfolders)
├── handwritten-english-characters-and-digits/
│   └── combined_folder/
│       ├── train/                       # Original training images
│       └── test/                        # Validation images
└── images/                              # Reference screenshots
```

---

## Models

All models share this base configuration:

| Setting | Value |
|---|---|
| Input | 128 x 128 x 3 |
| Output | Dense(62, softmax) |
| Loss | Sparse Categorical Cross-Entropy |
| Callbacks | ModelCheckpoint (best `val_loss`) + EarlyStopping (patience=10) |

---

### Model 1 - 4-Block ReLU CNN (with padding)

Four Conv blocks using `padding="same"` and a larger Dense head. Established the baseline and confirmed overfitting on the small original dataset.

```
Conv2D(32,  3x3, relu, padding=same) -> MaxPooling2D
Conv2D(64,  3x3, relu, padding=same) -> MaxPooling2D
Conv2D(128, 3x3, relu, padding=same) -> MaxPooling2D
Conv2D(160, 3x3, relu, padding=same) -> MaxPooling2D
Flatten -> Dense(128, relu) -> Dense(62, softmax)
Optimizer: Adam (lr=0.001)
```

---

### Model 2 - 4-Block ReLU CNN (no padding, stride=2)

Same four-block depth as Model 1, but removes `padding="same"`, uses explicit `strides=(2,2)` on MaxPooling, and shrinks the Dense head to 64 units. Outperforms Model 1 and remains the **overall best by augmented `val_loss`** (0.6099).

```
Conv2D(32,  3x3, relu) -> MaxPooling2D(stride=2)
Conv2D(64,  3x3, relu) -> MaxPooling2D(stride=2)
Conv2D(128, 3x3, relu) -> MaxPooling2D(stride=2)
Conv2D(160, 3x3, relu) -> MaxPooling2D(stride=2)
Flatten -> Dense(64, relu) -> Dense(62, softmax)
Optimizer: Adam (lr=0.001)
```

---

### Model 3 - BatchNorm + Dropout + Low LR

Adds BatchNormalization after each Conv block (activation applied post-BN), Dropout(0.3) before the classifier, and a lower learning rate with gradient clipping. Achieves the **best augmented val accuracy (81.23%)** among the regularized models.

```
Conv2D(32)  -> BN -> ReLU -> MaxPooling2D(stride=2)
Conv2D(64)  -> BN -> ReLU -> MaxPooling2D(stride=2)
Conv2D(128) -> BN -> ReLU -> MaxPooling2D(stride=2)
Conv2D(160) -> BN -> ReLU -> MaxPooling2D(stride=2)
Flatten -> Dense(64, relu) -> Dropout(0.3) -> Dense(62, softmax)
Optimizer: Adam(lr=1e-4, clipnorm=1.0)
```

---

### Model 4 - LeakyReLU + BatchNorm

Replaces ReLU with `LeakyReLU(alpha=0.1)` to address dying-ReLU behavior observed in Model 3's early training, while keeping BatchNorm and Dropout. Best result on **original** data but **worst augmented performance** and steepest overfitting curve.

```
Conv2D(32/64/128/160) -> BN -> LeakyReLU(0.1) -> MaxPooling2D(stride=2)  [x4]
Flatten -> Dense(64, relu) -> Dropout(0.3) -> Dense(62, softmax)
Optimizer: Adam(lr=1e-4, clipnorm=1.0)
```

---

### Model 5 - LeakyReLU, no BatchNorm

Tests LeakyReLU without BatchNorm to isolate its contribution. Gradients stayed alive without BN, but performance sits between Model 3 and Model 4 -- the BatchNorm + Dropout combination remains the stronger regularizer.

```
Conv2D(32/64/128/160) -> LeakyReLU(0.1) -> MaxPooling2D(stride=2)  [x4]
Flatten -> Dense(64, relu) -> Dropout(0.3) -> Dense(62, softmax)
Optimizer: Adam(lr=1e-4, clipnorm=1.0)
```

---

### Model 6 - LeakyReLU + Wider Dense Head

Same as Model 5 but replaces `Dense(64)` with `Dense(128)`. The extra capacity added no generalization benefit -- worst of the LeakyReLU variants.

```
Conv2D(32/64/128/160) -> LeakyReLU(0.1) -> MaxPooling2D(stride=2)  [x4]
Flatten -> Dense(128, relu) -> Dropout(0.3) -> Dense(62, softmax)
Optimizer: Adam(lr=1e-4, clipnorm=1.0)
```

---

### Hybrid - Model 3 Conv + Random Forest Head

To isolate whether overfitting came from the Dense head or the conv feature extractor, the best-checkpoint conv stack from Model 3 was frozen and its 5,760-dim flattened features were fed into a `RandomForestClassifier` (300 trees, `max_depth=30`, `min_samples_leaf=2`) instead of the Dense+Dropout head.

**Conclusion:** The RF head performed worse on every metric -- lower val accuracy and dramatically higher log-loss. This useful negative result rules out the classifier head as the cause of overfitting and points to the **conv feature extractor and/or data diversity** as the true bottleneck.

---

## Model Comparison

| Model | Architecture | LR | Original val acc | Original best epoch | Aug val acc | Aug best epoch | Aug val_loss | Notes |
|---|---|---|---|---|---|---|---|---|
| Model 1 | 4× Conv(32→160, same) + Dense(128) | 1e-3 | 65.54% | 3 | 80.06% | 3 | 0.6812 | Baseline |
| Model 2 | 4× Conv(32→160, stride=2) + Dense(64) | 1e-3 | 67.30% | 5 | 80.50% | 2 | **0.6099** | **Best overall val_loss** |
| Model 3 | 4× [Conv+BN+ReLU] + Dropout(0.3) + Dense(64) | 1e-4 | **68.18%** | 38 | **81.23%** | 16 | 0.6414 | **Best aug val acc** |
| Model 4 | 4× [Conv+BN+LeakyReLU] + Dropout(0.3) + Dense(64) | 1e-4 | 68.04% | 48 | 75.95% | 10 | 0.8319 | Worst aug; highest orig val_loss |
| Model 5 | 4× [Conv+LeakyReLU] + Dropout(0.3) + Dense(64) | 1e-4 | 66.28% | 32 | 80.06% | 19 | 0.6849 | No BN |
| Model 6 | 4× [Conv+LeakyReLU] + Dropout(0.3) + Dense(128) | 1e-4 | 66.13% | 22 | 77.86% | 11 | 0.7661 | Wider head, no BN |
| Model 3 + RF (original feats) | Frozen Model 3 conv → RF(300 trees, depth=30) | -- | -- | -- | 74.93% | -- | 1.8313 (log-loss) | Negative result |
| Model 3 + RF (aug feats) | Frozen Model 3 conv → RF(300 trees, depth=30) | -- | -- | -- | 77.57% | -- | 1.9347 (log-loss) | Negative result |

> All val metrics are taken from **actual epoch training logs** at the minimum `val_loss` epoch. Models 3-6 use `Adam(lr=1e-4, clipnorm=1.0)`. Models 1-2 use default `Adam(lr=1e-3)`. Original-data val_loss at best checkpoints: M1=1.3075, M2=1.2282, M3=1.1632, M4=1.2029, M5=1.4287, M6=1.3743.

---

## Key Findings

1. **Data is the primary bottleneck, not architecture.** Augmentation lifted val accuracy from the 65-68% range to 75-81% across all models. No single architectural change (BatchNorm, Dropout, LeakyReLU, wider Dense) produced a gain of that magnitude on its own.

2. **`val_loss` is a more reliable stopping signal than `val_accuracy`.** In every model, peak val_accuracy appeared at later epochs than the minimum val_loss checkpoint. Those later checkpoints are overfitted. The `restore_best_weights=True` callback consistently selects the more trustworthy checkpoint.

3. **Simpler Model 2 beats the more complex Models 3-6 on augmented `val_loss`.** Model 2 (0.6099) outperforms all regularized variants despite having no BatchNorm or Dropout. Architecture additions shifted *when* overfitting kicks in but not *how much* the gap closes.

4. **Overfitting is in the conv feature extractor, not the Dense head.** Swapping the Dense head for a Random Forest on frozen Model 3 conv features made performance *worse* (val acc 74-77% vs 81.23%, log-loss nearly 3× higher). This rules out the classifier head as the primary cause of overfitting.

5. **Model 3 has the best augmented val accuracy (81.23%) and is the strongest candidate for further tuning.** It also achieved the highest original-data val accuracy (68.18% at epoch 38), marginally ahead of Model 4 (68.04%).

---

## Future Plans

- [ ] **Larger, more diverse augmentation** -- expand beyond 13,640 images with stronger rotation/shift/zoom and additional transforms (brightness, contrast jitter, shear) to close the remaining train/val gap.
- [ ] **Collect new training data from external sources** -- gather additional handwritten character samples from other publicly available datasets (e.g. EMNIST, NIST SD19, Kaggle alternatives) and fine-tune Model 3 further on the combined dataset to improve generalization beyond what augmentation alone can provide.
- [ ] **Focus effort on Model 3** -- targeted Dropout tuning and L2 weight regularization, rather than spreading experiments across all six architectures.
- [ ] **Transfer learning** -- fine-tune a pretrained backbone (MobileNetV2, EfficientNet) on this dataset to leverage pre-learned visual features.
- [ ] **Systematic hyperparameter search** -- grid or random search over LR, batch size, and Dropout rate for the top-performing models.
- [ ] **Grad-CAM visualization** -- inspect which image regions the model focuses on and identify failure modes, especially for visually similar character pairs (O/0, I/l/1).
- [ ] **Per-class accuracy analysis** -- find the most-confused character pairs among 62 classes and address them with targeted augmentation or class-balancing.

---

## Dependencies

Key packages -- see [`requirements.txt`](requirements.txt) for the full pinned list.

| Package | Version | Purpose |
|---|---|---|
| tensorflow | 2.21.0 | Model training |
| keras | 3.15.1 | High-level API |
| scikit-learn | 1.9.0 | Random Forest, metrics |
| numpy | 2.5.2 | Array operations |
| matplotlib | 3.11.1 | Training curve plots |
| pillow | 12.3.0 | Image handling |
| joblib | 1.5.3 | RF model serialization |
| jupyterlab | 4.6.3 | Notebook environment |
