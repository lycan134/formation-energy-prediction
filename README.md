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

The dataset is derived from the **Materials Project** and preprocessed for machine learning.  
Due to file size limits, it is hosted externally on Zenodo:

👉 [**Download Dataset (Zenodo DOI)**](https://zenodo.org/records/17218766)

After downloading, place the dataset inside the `data/` folder:

