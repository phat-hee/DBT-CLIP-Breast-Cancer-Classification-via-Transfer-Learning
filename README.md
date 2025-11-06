# DBT-CLIP: Breast Cancer Classification via Transfer Learning

A robust, leak-safe machine learning pipeline for classifying breast cancer screening images (Digital Breast Tomosynthesis) using CLIP embeddings and an MLP classifier.

## 🎯 Overview

This project implements a **two-stage transfer learning approach** for multi-class breast cancer classification:
1. **Feature Extraction**: Uses pre-trained CLIP (ViT-B/32) as a frozen feature extractor
2. **Classification**: Trains a lightweight MLP on extracted 512-dimensional embeddings

### Key Features

✅ **Fully Reproducible** - Auto-installs dependencies and downloads dataset from Kaggle  
✅ **Leak-Safe Design** - Data splitting performed BEFORE any augmentation or resampling  
✅ **Imbalance-Aware** - Combines SMOTETomek, Focal Loss, and ENS class weights  
✅ **Comprehensive Evaluation** - Generates confusion matrices, ROC/PR curves, and per-class metrics  
✅ **Professional Documentation** - Exports results as JPG figures and Word documents  

---

## 📊 Dataset

**Source**: [Breast Cancer Screening DBT Dataset](https://www.kaggle.com/datasets/gabrielcarvalho11/breast-cancer-screening-dbt)

**Classes** (4):
- Benign
- Actionable
- Cancer
- Normal

The dataset is automatically downloaded via `kagglehub` when you run the notebook.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Kaggle account (for dataset download)

### Installation & Running

The notebook handles all dependencies automatically! Just run:

```bash
# Clone the repository
git clone https://github.com/phat-hee/dbt-clip-breast-cancer.git
cd dbt-clip-breast-cancer

# Run the notebook
jupyter notebook breast_cancer_dbt_clip_pipeline.ipynb
```

All required packages will be installed automatically on first run.

---

## 🏗️ Architecture

### Pipeline Overview

```
Raw Images → [CLIP Encoder (frozen)] → 512-d Embeddings → [MLP (trained)] → 4 Classes
              └─ ViT-B/32 pre-trained                      └─ Focal Loss + ENS weights
```

### Model Components

1. **CLIP (ViT-B/32)** - Pre-trained vision encoder (frozen)
   - Extracts 512-dimensional feature vectors
   - No training/fine-tuning performed

2. **MLP Classifier** - 2-layer feedforward network
   - Input: 512-d CLIP features
   - Hidden layers: 256 → 128
   - Output: 4 classes
   - Regularization: BatchNorm + Dropout (0.5)

3. **Loss Function** - Focal Loss with α weights
   - Handles class imbalance
   - Focuses learning on hard examples
   - α computed via Effective Number of Samples (ENS)

---

## 📈 Methodology

### 1. Data Splitting (Leak-Safe)
```
All Images (100%)
    ↓
Train (64%) / Temp (36%)
    ↓
Train (64%) / Val (16%) / Test (20%)
```
- **Stratified splits** maintain class distributions
- Splitting done **BEFORE** any preprocessing

### 2. Imbalance Handling (Train Only)
- **SMOTETomek**: Synthetic minority oversampling + Tomek link removal
- **Focal Loss**: γ=2.0 focuses on hard examples
- **ENS Weights**: Class-wise α weights based on effective samples

### 3. Augmentation (Train Only, Minority Classes)
- Horizontal/Vertical flips
- Random 90° rotation
- Brightness/Contrast adjustment
- Shift/Scale/Rotate transforms

### 4. Training
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Scheduler: CosineAnnealingLR
- Epochs: 30
- Batch size: 32
- Best model selected by validation loss

---

## 📁 Project Structure

```
dbt-clip-breast-cancer/
├── breast_cancer_dbt_clip_pipeline.ipynb   # Main notebook
├── README.md                                # This file
├── data/                                    # Dataset (auto-downloaded)
│   └── Breast-Cancer-Screening-DBT/
│       ├── Benign/
│       ├── Actionable/
│       ├── Cancer/
│       └── Normal/
└── results/                                 # All outputs
    ├── Figure_Class_Distribution_All.jpg
    ├── Figure_Class_Distribution_All.docx
    ├── Figure_Training_Loss.jpg
    ├── Figure_Validation_Confusion_Matrix.jpg
    ├── Figure_Test_ROC_PR.jpg
    ├── Table_Test_Overall_Metrics.docx
    ├── Table_Test_PerClass_Metrics.docx
    └── Summary_Experiment.docx
```

---

## 📊 Results

The pipeline generates comprehensive evaluation metrics:

### Overall Metrics
- Accuracy
- Balanced Accuracy
- Macro F1 Score
- Macro Precision/Recall
- Macro AUC-ROC

### Per-Class Metrics
- Precision
- Recall
- F1 Score
- Specificity
- AUC-ROC

### Visualizations
- Class distribution plots
- Training/validation loss curves
- Normalized confusion matrices
- ROC curves (one-vs-rest)
- Precision-Recall curves

All outputs are saved as:
- **JPG images** (300 DPI, publication-ready)
- **Word documents** (.docx) with captions and descriptions

---

## 🔬 Technical Details

### Why This Approach?

**Transfer Learning Benefits:**
- ✅ Leverages CLIP's knowledge from 400M image-text pairs
- ✅ Reduces training time (minutes vs. hours)
- ✅ Works well with limited medical imaging data
- ✅ Lower computational requirements

**Frozen vs. Fine-tuning:**
- This implementation uses **frozen CLIP** (feature extraction only)
- Fine-tuning CLIP end-to-end could potentially improve results, but requires:
  - More GPU memory
  - Longer training time
  - Risk of overfitting on small datasets

### Reproducibility

- Fixed random seeds (RANDOM_STATE=42)
- Deterministic operations where possible
- Complete environment captured via auto-installation
- Dataset version pinned (v1)

---

## 🛠️ Dependencies

Automatically installed packages:
- `torch`, `torchvision` - Deep learning framework
- `clip-anytorch` - OpenAI CLIP model
- `scikit-learn` - ML utilities and metrics
- `imbalanced-learn` - SMOTETomek resampling
- `albumentations` - Image augmentation
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `python-docx` - Word document export
- `kagglehub` - Dataset download
- `opencv-python-headless` - Image processing

---

## 📝 Usage Examples

### Running the Full Pipeline
```python
# Just run all cells in the notebook!
# Everything is automated from installation to results export
```

### Customizing Parameters
```python
# In the configuration cell, modify:
RANDOM_STATE = 42        # Change for different splits
EPOCHS = 30              # Training epochs
BATCH_SIZE = 32          # Batch size
LEARNING_RATE = 1e-3     # Initial learning rate
```

### Using Custom Dataset
```python
# Replace the Kaggle download section with:
DATASET_DIR = Path("/path/to/your/dataset")
# Ensure folder structure: DATASET_DIR/ClassName/images.jpg
```

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{dbt_clip_2024,
  title={DBT-CLIP: Breast Cancer Classification via Transfer Learning},
  author={N. Alipour, M. Faramarzi, M. Gholami, M. Fathi , N. Deravi},
  year={2024},
  url={https://github.com/phat-hee/dbt-clip-breast-cancer}
}
```

**Dataset Citation:**
```bibtex
@dataset{carvalho2024dbt,
  title={Breast Cancer Screening DBT Dataset},
  author={Carvalho, Gabriel},
  year={2024},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/gabrielcarvalho11/breast-cancer-screening-dbt}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Fine-tuning CLIP end-to-end
- [ ] Experimenting with other CLIP variants (ViT-L/14, ResNet-50)
- [ ] Ensemble methods
- [ ] Grad-CAM visualization
- [ ] Cross-validation
- [ ] Hyperparameter optimization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This is a research tool and **NOT approved for clinical use**. Always consult healthcare professionals for medical diagnosis and treatment decisions.

---

## 📧 Contact

- **Author**: Mohammad Fathi
- **Email**: mohammad.s.fathi98@gmail.com
- **GitHub**: [@phat-hee](https://github.com/phat-hee)
- **Project Link**: [https://github.com/phat-hee/dbt-clip-breast-cancer](https://github.com/yourusername/dbt-clip-breast-cancer)

---

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- Kaggle community for the dataset
- PyTorch team for the deep learning framework


