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

After downloading:

1. Create a folder named `data/` in the root of the repository (if it doesn’t exist).  
2. Place the downloaded CSV files inside `data/`, e.g.:

📁 formation-energy-prediction/
├── data/
│   ├── X_preprocessed.csv
│   ├── y_preprocessed.csv
│
├── models/
│   ├── best_model_full.pt
│   ├── normalization_stats.pth
│
├── figures/
│   ├── true_vs_predicted_plot.eps
│   ├── true_vs_predicted_plot.svg
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md

