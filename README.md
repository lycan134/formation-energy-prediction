# formation-energy-prediction
Neural network with hyperparameter tuning (Bayesian optimization + k-fold CV) for materials property prediction.
# 🧪 Crystal Structure Stability Prediction

This repository contains the **Jupyter Notebook** for predicting **formation energy** of materials using deep learning.  
The study integrates **chemical composition**, **crystallographic symmetry information** (space group), and **stability labels** (ground state, metastable, unstable) to improve prediction accuracy.
This repository contains the **PyTorch implementation** for predicting **formation energy** of materials using deep learning.  
The project integrates **chemical composition**, **crystallographic symmetry information (space group)**, and **stability labels** (ground state, metastable, unstable) to improve prediction accuracy.

---

## 📘 Overview
The notebook demonstrates a **Deep Neural Network (DNN)** trained with **Bayesian optimization** and **early stopping** to predict the *formation energy per atom* from the Materials Project dataset.  
This project demonstrates a **Deep Neural Network (DNN)** trained with early stopping and checkpoint saving to predict the *formation energy per atom* from the Materials Project dataset.  
It also includes analysis connecting **composition**, **crystal symmetry**, and **thermodynamic stability**.

**Key highlights:**
- Uses **space group symmetry** and **stability label** as input features  
- Includes **SHAP-based interpretability analysis** for understanding model decisions  
- Demonstrates **training, testing, and visualization** within a single notebook  
- Uses **space group** and **stability label** as part of input features  
- Implements **deep feedforward neural networks** in PyTorch  
- Includes **evaluation scripts** with real-world metrics (MAE, RMSE, R²)  
- Automatically saves **publication-ready figures** to the `figures/` directory  
- Designed for future integration into a **Streamlit web app**

---

## 📂 Dataset
The dataset is derived from the [**Materials Project**](https://materialsproject.org/) and preprocessed for machine learning.

Due to file size limits, the dataset is hosted on **Zenodo**:  
Due to file size limits, the dataset is hosted externally on **Zenodo**:  
👉 [**Download Dataset (Zenodo DOI)**](https://zenodo.org/records/17218766)

After downloading, place the dataset inside the `data/` folder.

---

## ⚙️ Setup and Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/lycan134/formation-energy-prediction.git
cd formation-energy-prediction
pip install -r requirements.txt
