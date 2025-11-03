# 🧪 Crystal Structure Formation Energy Prediction

A **Deep Learning workflow** to predict **formation energy per atom** of materials based on their **chemical composition** and **crystallographic symmetry**.  
It also includes **stability classification** using energy above hull.

Built with **PyTorch**, the workflow covers data preprocessing, model training, evaluation, and feature interpretability via **SHAP**.

---

## 📘 Overview

This project demonstrates a **Deep Neural Network (DNN)** trained on the **Materials Project** dataset to predict formation energy and classify material stability.

**Core features:**

- 🧹 **Data preprocessing:** Outlier removal, stability labeling, and encoding of composition and symmetry features.  
- ⚙️ **Model training:** Deep feedforward network with **early stopping** and **checkpoint saving**.  
- 🔍 **Interpretability:** **SHAP**-based feature importance analysis.  
- 📈 **Evaluation:** Metrics include **MAE**, **RMSE**, and **R²**, with figures auto-generated for publication use.  
- 🌐 **Future integration:** Compatible with **Streamlit** for interactive predictions.

---

## 📂 Dataset Preparation

The dataset originates from the **Materials Project** and is preprocessed for ML compatibility.

### 🧭 Stability Classification

Materials are labeled according to **Energy Above Hull (Eᵃʰ):**

| Energy Above Hull (eV/atom) | Stability Label |
|-----------------------------:|----------------:|
| ≤ 0.025                     | Stable          |
| 0.025 – 0.100               | Metastable      |
| > 0.100                     | Unstable        |

---

## 🧮 Features and Data Representation

Each material sample is represented through **elemental composition**, **aggregate atomic descriptors**, and **space group symmetry**.

### 🔢 Elemental Features
Elements included in the dataset:

`'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',  
'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',  
'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',  
'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',  
'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',  
'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',  
'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',  
'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',  
'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',  
'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',  
'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',  
'Es', 'Fm', 'Md', 'No', 'Lr'`

Each of these is represented by its **fractional composition** in the material formula.

---

### ⚛️ Physical Descriptors

| Descriptor | Meaning |
|-------------|----------|
| `n_atoms` | Total number of atoms in the compound |
| `n_elements` | Number of unique elements present |
| `avg_atomic_mass` | Mean atomic mass (weighted by composition) |
| `en_mean`, `en_max`, `en_min`, `en_range` | Mean, max, min, and range of electronegativity |
| `avg_covalent_radius` | Average covalent radius |
| `ea_mean`, `ea_max`, `ea_min`, `ea_range` | Mean, max, min, and range of electron affinity |

---

### 🧩 Crystallographic Symmetry

Space group symmetry is **one-hot encoded** to incorporate structural information.  
Each material is tagged with its **space group number (1–230)**.

---

### 🧰 Preprocessing Summary

1. **Filter formation energy:** Remove outliers beyond ±5σ from the mean.  
2. **Deduplicate:** Keep the lowest-energy entry for each formula and space group.  
3. **Feature scaling:** Normalize atomic and physical features.  
4. **Handle missing values:** Fill with mean or zero.  
5. **Encode categorical variables:** Space groups and stability labels.  
6. **Output:** Final ML tensors `X` (features) and `y` (formation energy).

---


### Dataset Access

Due to size constraints, the dataset is hosted externally:

👉 [Download Dataset (Zenodo DOI)](https://www.google.com/search?q=https://zenodo.org/record/xxx)

Place the downloaded CSV in the `data/` folder.

## ⚙️ Setup and Installation

Clone the repository and install dependencies:

git clone https://github.com/lycan134/formation-energy-prediction.git 
cd formation-energy-prediction 
pip install -r requirements.txt


Recommended Python version: 3.8+

## 🏃 Usage

### Preprocessing

python preparation.py


Cleans dataset, adds stability labels, encodes categorical features, and saves preprocessed `X_preprocessed.csv` and `y_preprocessed.csv`.

### Training

python train.py


Trains the DNN with early stopping and checkpointing.

### Prediction / Evaluation

python predict.py # Predict formation energy on new data python 
evaluate.py # Compute MAE, RMSE, R², and generate figures


## 📊 Interpretability

Uses [SHAP](https://shap.readthedocs.io/en/latest/index.html) to analyze feature importance and understand model decisions.

Publication-ready plots saved automatically in the `figures/` folder.

## ⚡ Key Highlights

* Integrates composition, space group, and stability information as features.

* Fully implemented in PyTorch with ML-ready pipeline.

* Ready for interactive deployment in Streamlit.

