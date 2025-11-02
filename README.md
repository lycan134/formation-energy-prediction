# 🧪 Crystal Structure Stability Prediction

This repository contains the **PyTorch implementation** for predicting **formation energy** of materials using deep learning.  
The project integrates **chemical composition**, **crystallographic symmetry information (space group)**, and **stability labels** (ground state, metastable, unstable) to improve prediction accuracy.

---

## 📘 Overview
This project demonstrates a **Deep Neural Network (DNN)** trained with early stopping and checkpoint saving to predict the *formation energy per atom* from the Materials Project dataset.  
It also includes analysis connecting **composition**, **crystal symmetry**, and **thermodynamic stability**.

**Key highlights:**
- Uses **space group** and **stability label** as part of input features  
- Implements **deep feedforward neural networks** in PyTorch  
- Includes **evaluation scripts** with real-world metrics (MAE, RMSE, R²)  
- Automatically saves **publication-ready figures** to the `figures/` directory  
- Designed for future integration into a **Streamlit web app**

---

## 📂 Dataset
The dataset is derived from the [**Materials Project**](https://materialsproject.org/) and preprocessed for machine learning.

Due to file size limits, the dataset is hosted externally on **Zenodo**:  
👉 [**Download Dataset (Zenodo DOI)**](https://zenodo.org/records/17218766)

After downloading, place the dataset inside the `data/` folder:
