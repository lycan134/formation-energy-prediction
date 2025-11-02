# 🧪 Crystal Structure Stability Prediction

This repository contains the **PyTorch implementation** for predicting the **formation energy per atom** of crystalline materials using deep learning.  
The model integrates **chemical composition**, **space group symmetry**, and **stability labels** (ground state, metastable, unstable) to improve prediction accuracy.

---

## 📘 Overview

This project demonstrates a **Deep Neural Network (DNN)** built with PyTorch, trained using **k-fold validation**, **early stopping**, and **checkpoint saving**.  
It predicts the *formation energy per atom* from the [Materials Project](https://materialsproject.org/) dataset and evaluates performance using standard regression metrics.

### 🔑 Key Highlights
- Incorporates **space group** and **stability label** as key input features  
- Implements a **deep feedforward neural network** with multiple hidden layers  
- Uses **MAE**, **RMSE**, and **R²** for real-world evaluation  
- Automatically saves trained models and normalization stats in the `models/` directory  
- Generates **publication-ready figures** (`.eps`, `.svg`) in the `figures/` folder  
- Designed for future integration into a **Streamlit web application**

---

## 📂 Dataset

The dataset is derived from the [**Materials Project**](https://materialsproject.org/) and preprocessed for machine learning.  
Due to its large size, the dataset is hosted externally on Zenodo:

👉 [**Download Dataset (Zenodo DOI)**](https://zenodo.org/records/17504632)

After downloading, create a `data/` folder in the root of the repository (if it doesn’t exist) and place the downloaded CSV files inside:


> ⚠️ Make sure the file names match exactly; the scripts (`train.py`, `evaluate.py`) expect these exact names.

---

## ⚙️ Setup and Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/lycan134/formation-energy-prediction.git
cd formation-energy-prediction
pip install -r requirements.txt




