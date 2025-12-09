Here’s a clean, professional **README.md** designed for GitHub, based on your report.
You can copy-paste directly — let me know if you want badges, logo, images, or a shorter version.

---

# Fine-Grained Visual Categorization Using FeathersV1

This repository presents experimentation and results on **fine-grained feather image classification** using the **FeathersV1 dataset**, containing over **28K feather images** from **595 bird species**. We benchmark multiple models and training strategies (EfficientNet-B4) to evaluate whether single feather images can accurately identify bird species.

---

## 🐦 Overview

Fine-grained visual categorization aims to distinguish between visually similar object classes. Bird feather identification can support:

* Ornithology and taxonomy
* Ecology and biodiversity monitoring
* Aviation safety (bird strike investigations)
* Conservation efforts

Unlike traditional bird datasets (whole body), **FeathersV1 focuses solely on feathers**, making it a unique fine-grained classification benchmark.

---

## 📂 Dataset: FeathersV1

* **Images:** 28,272
* **Species:** 595
* **Orders:** 23
* **Goal:** Predict species directly from single feather images

### Distribution

| Subset          | Train Images | Val Images |
| --------------- | ------------ | ---------- |
| Top-50 Species  | 8,251        | 2,063      |
| Top-100 Species | 11,953       | 2,988      |
| All 595 Species | 22,618       | 5,654      |

### Characteristics

* Large variation between species
* High intra-species variability (age, sex, genetics)
* Significant class imbalance (from 2 to 620 images per species)

---

## 🧠 Model & Training

We use **EfficientNet-B4** pretrained on ImageNet.

### Data Processing

**Training Augmentations**

* Random Resized Crop
* Random Horizontal Flip
* Normalization (ImageNet mean/std)

**Validation Preprocessing**

* Resize + Center Crop
* Normalization

### Loss & Optimization

* Batch size: **32**
* Initial LR: **0.005**
* Class imbalance handled using **Class Weights**
* Learning rate scheduler applied

---

## 🚧 Challenges

* **High inter-species similarity**
* **Large intra-species variation**
* **Severe class imbalance**

### Mitigation Strategies

* Weighted loss
* Ensuring single-instance classes appear in both train/test
* Wide augmentation
* Gradual fine-tuning of model layers

---

## 📊 Results

We ran **four main experiments** evaluating:

* Full dataset vs. top 100 classes
* 50 vs. 128 epochs
* Gradual fine-tuning depths

### Summary Table

| Classes | Epochs | Accuracy   | F1 Score   | Precision  | Recall     | Top-5 Acc  |
| ------- | ------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| All     | 128    | 63.29%     | 63.79%     | 72.66%     | 63.29%     | 90.33%     |
| Top 100 | 128    | **74.84%** | **74.77%** | **78.45%** | **74.84%** | **95.03%** |
| All     | 50     | 58.10%     | 58.43%     | 70.70%     | 58.10%     | 82.71%     |
| Top 100 | 50     | 71.84%     | 71.10%     | 78.95%     | 71.84%     | 93.37%     |

### Key Takeaways

* Longer training (128 epochs) improves results significantly.
* Restricting to the top 100 most represented species yields the best accuracy.
* Gradual unfreezing (incremental fine-tuning) gives strong performance gains.

---

## 🚀 Future Work

* Hierarchical classification (Order → Species)
* Transformer-based models (ViT, Swin)
* Advanced augmentation and sampling
* Cross-domain transfer learning
* Few-shot learning for rare species

---

## 📁 Repository Structure (Suggested)

```
|-- results/
├── EfficientNetB4_top_100_50_epochs/
│   ├── best.pt                     # Best model checkpoint
│   ├── classification_report.xlsx  # Per-class precision, recall, F1 scores
│   ├── loss_curve.png              # Training/validation loss visualization
│   └── accuracy_curve.png          # Training/validation accuracy visualization
│
├── EfficientNetB4_top_100_128_epochs/
│   ├── best.pt
│   ├── classification_report.xlsx
│   ├── loss_curve.png
│   └── accuracy_curve.png
│
├── EfficientNetB4_all_50_epochs/
│   ├── best.pt
│   ├── classification_report.xlsx
│   ├── loss_curve.png
│   └── accuracy_curve.png
│
├── EfficientNetB4_all_128_epochs/
│   ├── best.pt
│   ├── classification_report.xlsx
│   ├── loss_curve.png
│   └── accuracy_curve.png
├── trainer.py
|-- README.md
```

---

## 📜 Citation

If you use FeathersV1 or this work, please cite:

> Belko et al., “Feathers dataset for Fine-Grained Visual Categorization,” arXiv:2004.08606v1, 2020

---

## 🙌 Acknowledgments

Thanks to the FeathersV1 authors for releasing this benchmark dataset and enabling fine-grained classification research.


