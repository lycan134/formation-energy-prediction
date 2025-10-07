# 🧪 Crystal Structure Stability Prediction

This repository contains the code and trained models for predicting **formation energy** of materials using deep learning.  
The study integrates **chemical composition**, **crystallographic symmetry information** (crystal system, point group, and space group), and **stability labels** (ground state, metastable, unstable) to improve prediction accuracy.

---

## 📘 Overview
This project implements a **Deep Neural Network (DNN)** optimized using **Bayesian optimization** and **early stopping**.  
The model was trained to predict *formation energy per atom* from the Materials Project dataset and to analyze the relationship between **composition, structure, and thermodynamic stability**.

**Key features:**
- Incorporates **space group symmetry** and **stability label** as additional input descriptors  
- Includes **SHAP-based interpretability analysis** to ensure physically meaningful model behavior  
- Reproducible experiments with controlled **train–test random splits** for reliability assessment  

---

## 📂 Dataset
The dataset is derived from the [**Materials Project**](https://materialsproject.org/) and preprocessed into a machine-learning–ready format.

Due to file size limitations, the data is hosted on **Zenodo**:  
👉 [**Download Dataset (Zenodo DOI)**](https://zenodo.org/records/17218766)

After downloading, place the dataset inside the `data/` folder.

---

## ⚙️ Setup and Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/lycan134/formation-energy-prediction.git
cd formation-energy-prediction
pip install -r requirements.txt
