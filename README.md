# 🧪 Crystal Structure Stability Prediction

This repository contains the **Jupyter Notebook** for predicting **formation energy** of materials using deep learning.  
The study integrates **chemical composition**, **crystallographic symmetry information** (crystal system, point group, and space group), and **stability labels** (ground state, metastable, unstable) to improve prediction accuracy.

---

## 📘 Overview
The notebook demonstrates a **Deep Neural Network (DNN)** trained with **Bayesian optimization** and **early stopping** to predict the *formation energy per atom* from the Materials Project dataset.  
It also includes analysis connecting **composition**, **crystal symmetry**, and **thermodynamic stability**.

**Key highlights:**
- Uses **space group symmetry** and **stability label** as input features  
- Includes **SHAP-based interpretability analysis** for understanding model decisions  
- Demonstrates **training, testing, and visualization** within a single notebook  

---

## 📂 Dataset
The dataset is derived from the [**Materials Project**](https://materialsproject.org/) and preprocessed for machine learning.

Due to file size limits, the dataset is hosted on **Zenodo**:  
👉 [**Download Dataset (Zenodo DOI)**](https://zenodo.org/records/17218766)

After downloading, place the dataset inside the `data/` folder.

---

## ⚙️ Setup and Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/lycan134/formation-energy-prediction.git
cd formation-energy-prediction
pip install -r requirements.txt
