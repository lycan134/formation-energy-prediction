# 🧪 Crystal Structure Formation Energy Prediction  
[![DOI](https://img.shields.io/badge/DOI-10.1088%2F2053--1591%2Fae22cb-blue)](https://doi.org/10.1088/2053-1591/ae22cb)  
![Python](https://img.shields.io/badge/Python-3.8+-yellow)  
![License](https://img.shields.io/badge/License-MIT-green)  
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)

A **Deep Learning pipeline** for predicting the **formation energy per atom** of inorganic crystalline materials using their **chemical composition** and **crystallographic symmetry**.  
This repository accompanies the published paper:

> **Torlao et al., Materials Research Express (2025)**  
> DOI: **10.1088/2053-1591/ae22cb**

The workflow features preprocessing, DNN training, stability classification, evaluation, and SHAP interpretability.

---

## ✨ Features

- 🔬 Formation energy prediction using a deep neural network  
- 🧭 Stability classification (stable / metastable / unstable) via **Energy Above Hull**  
- ⚙️ End-to-end ML workflow: preprocessing → training → evaluation → plots  
- 📊 Model interpretability with **SHAP**  
- 🏗️ Crystallographic space group encoding (1–230)  
- 📈 Automatically generates publication-ready figures  
- 🌐 Ready for future Streamlit deployment  

---

## 📘 Overview

This project uses a DNN trained on the **Materials Project** database to:

- Predict formation energy per atom  
- Classify material stability  
- Analyze feature contributions (elements, physical descriptors, symmetry)

---

## 📂 Dataset

The processed and labeled dataset (used in the publication) is hosted at Zenodo:

👉 **[Zenodo Dataset (DOI)](https://zenodo.org/records/17504632)**

Place the file inside the `data/` directory.

---

## 🧭 Stability Classification

| Energy Above Hull (eV/atom) | Label        |
|-----------------------------:|--------------|
| ≤ 0.025                     | Stable       |
| 0.025–0.100                 | Metastable   |
| > 0.100                     | Unstable     |

---

## 🧮 Feature Engineering

### **1. Elemental Features**
Fractional composition of all elements (H–Lr) found in the dataset.

### **2. Physical Descriptors**
- `n_atoms`, `n_elements`  
- Mean atomic mass  
- Electronegativity: mean, max, min, range  
- Covalent radius: mean  
- Electron affinity: mean, max, min, range  

### **3. Crystallographic Symmetry**
- Space group (1–230), one-hot encoded  

---

## 🛠️ Preprocessing Workflow

1. Remove formation energy outliers (±5σ)  
2. Deduplicate lowest-energy entries per formula + space group  
3. Normalize atomic & physical features  
4. Impute missing values  
5. Encode space groups and stability labels  
6. Export:
   - `X_preprocessed.csv`
   - `y_preprocessed.csv`

---

## ⚙️ Installation

```bash
git clone https://github.com/lycan134/formation-energy-prediction.git
cd formation-energy-prediction
pip install -r requirements.txt
Python 3.8+ recommended.

---

## 🚀 Usage
1. Preprocessing
python preparation.py
2. Training
python train.py
3. Prediction
python predict.py
4. Evaluation
python evaluate.py
Generates metrics (MAE, RMSE, R²) and publication-ready figures.

## 📊 SHAP Interpretability
SHAP is used to identify feature importance for:

Elemental contributions

Physical properties

Space group effects

All plots are saved in the figures/ directory.

📁 Project Structure
formation-energy-prediction/
│
├── data/
│   └── (dataset here)
├── models/
│   └── saved_checkpoints/
├── figures/
├── preparation.py
├── train.py
├── predict.py
├── evaluate.py
├── requirements.txt
└── README.md
🧾 Citation
If you use this repository, model, or dataset, please cite:

Torlao V.C., et al. "Formation energy prediction of material crystal structures using deep learning."
Materials Research Express (2025).
DOI: 10.1088/2053-1591/ae22cb
📜 License
Released under the MIT License.
