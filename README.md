# 🏥 PKD Classification System

**AI-Powered Polycystic Kidney Disease Detection from CT Scans**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io)

---

## 📋 Project Overview

This project implements a deep learning-based system for classifying kidney CT scan images into four categories:

| Class | Description | Samples |
|-------|-------------|---------|
| 🟢 **Normal** | Healthy kidney with no abnormalities | 5,077 |
| 🟠 **Cyst** | Fluid-filled sac in kidney tissue | 3,709 |
| 🔴 **Tumor** | Abnormal tissue growth | 2,283 |
| 🟣 **Stone** | Kidney stones (calculi) | 1,377 |

**Total Dataset:** 12,446 CT scan images

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     PKD Classification System                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │
│  │   CT Scan   │───▶│ Preprocessing│───▶│  Data Augmentation  │   │
│  │   Images    │    │  (Resize,   │    │ (Spatial, Color,    │   │
│  │             │    │  Normalize) │    │  Noise transforms)  │   │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘   │
│                                                    │              │
│                                                    ▼              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    EfficientNet-B0 Encoder                   │ │
│  │              (Pretrained on ImageNet)                        │ │
│  └─────────────────────────────┬───────────────────────────────┘ │
│                                │                                  │
│                                ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Custom Classification Head                      │ │
│  │     FC(1280→512) → BN → ReLU → Dropout                      │ │
│  │     FC(512→256)  → BN → ReLU → Dropout                      │ │
│  │     FC(256→4)    → Softmax                                   │ │
│  └─────────────────────────────┬───────────────────────────────┘ │
│                                │                                  │
│                                ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │     Output: [Normal, Cyst, Stone, Tumor] probabilities       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
pkd/
├── model_training/
│   └── pkd_classification_training.py   # Colab training script
├── backend/
│   ├── main.py                          # FastAPI server
│   └── requirements.txt
├── frontend/
│   ├── app.py                           # Streamlit UI
│   └── requirements.txt
├── outputs/                             # Generated after training
│   ├── models/                          # Saved model checkpoints
│   ├── plots/                           # Publication-quality plots
│   └── metrics/                         # Training metrics JSON
├── ct-kidney-dataset-normal-cyst-tumor-and-stone/
│   ├── kidneyData_fixed.csv             # Dataset metadata
│   └── CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
│       └── CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
│           ├── Cyst/
│           ├── Normal/
│           ├── Stone/
│           └── Tumor/
├── imp.txt                              # Project information
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Training on Google Colab

1. Upload the entire `pkd` folder to Google Drive
2. Open Google Colab and mount your drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Navigate to the project:
   ```python
   %cd /content/drive/MyDrive/pkd
   ```
4. Run the training script:
   ```python
   %run model_training/pkd_classification_training.py
   ```

> ⚠️ **Important:** Change runtime to **T4 GPU** before training:
> `Runtime → Change runtime type → T4 GPU`

### 2️⃣ Running the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3️⃣ Running the Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Web UI will be available at: http://localhost:8501

---

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | EfficientNet-B0 |
| Input Size | 224×224 |
| Batch Size | 16 (effective: 32) |
| Epochs | 5 per fold |
| Cross-Validation | 5-Fold Stratified |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |
| Mixed Precision | ✅ Enabled |

---

## 📈 Generated Outputs

After training, the following files are generated:

### Models
- `outputs/models/best_model_fold_1.pth` to `best_model_fold_5.pth`

### Plots (Publication Quality, 300 DPI)
1. `class_distribution.png` - Dataset class balance
2. `training_curves.png` - Loss/accuracy per epoch
3. `confusion_matrices.png` - Per-fold and average
4. `roc_curves.png` - Multi-class ROC with AUC
5. `class_performance.png` - Per-class metrics
6. `fold_comparison.png` - Cross-validation comparison

### Metrics
- `training_metrics.json` - All metrics in JSON format
- `classification_report.txt` - Detailed text report

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/predict` | POST | Single image prediction |
| `/predict/batch` | POST | Batch prediction (up to 10) |
| `/model/info` | GET | Model information |

### Example API Usage

```python
import requests

# Single prediction
with open("kidney_scan.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    print(response.json())

# Output:
# {
#     "class_name": "Normal",
#     "confidence": 0.9543,
#     "all_probabilities": {
#         "Cyst": 0.0212,
#         "Normal": 0.9543,
#         "Stone": 0.0089,
#         "Tumor": 0.0156
#     }
# }
```

---

## 👥 Team

- **Aniket Waghela** (60009220033)
- **Bhavya Jappi** (60009220004)
- **Kerul Kidecha** (60009220064)

**Guide:** Prof. Kanchan Dabre

**Department:** Computer Science and Engineering (Data Science)

---

## 📚 Dataset Citation

The CT-KIDNEY-DATASET was collected from PACS in hospitals in Dhaka, Bangladesh.

---

## ⚠️ Disclaimer

This system is developed for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare providers for medical advice.

---

## 📄 License

This project is part of an academic curriculum. Please contact the team for usage permissions.
