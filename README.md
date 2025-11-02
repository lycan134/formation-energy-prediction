# 🧪 Crystal Structure Formation Energy Prediction

This repository implements a deep learning workflow to predict formation energy per atom for materials using chemical composition and crystallographic symmetry. It also classifies material stability based on energy above hull.

The workflow is built in PyTorch and includes data preprocessing, model training, evaluation, and interpretability analysis.

## 📘 Overview

The project demonstrates a Deep Neural Network (DNN) trained to predict formation energy from the Materials Project dataset. Key aspects of the workflow:

* **Data preprocessing:** Outlier removal, stability classification, one-hot encoding of space groups and stability labels.

* **Model training:** Deep feedforward neural network with early stopping and checkpoint saving.

* **Interpretability:** SHAP analysis to understand feature contributions.

* **Evaluation:** Metrics include MAE, RMSE, and R², with publication-ready figures automatically saved.

* **Future integration:** Designed to integrate into a Streamlit web app for interactive predictions.

## 📂 Dataset Preparation

The dataset is derived from the Materials Project and preprocessed for machine learning.

### Stability Classification

Materials are labeled based on energy above hull:

| Energy Above Hull (eV/atom) | Stability Label | 
 | ----- | ----- | 
| $\le 0.025$ | Stable | 
| $0.025 – 0.100$ | Metastable | 
| $> 0.100$ | Unstable | 

### Preprocessing Steps

* **Filter formation energy:** Remove outliers beyond $\pm 5\sigma$ of the mean.

* **Deduplicate:** Keep only the lowest-energy entry for each material formula and space group.

* **Feature selection:** Includes elemental fractions, physical descriptors (atomic mass, electronegativity, covalent radius, electron affinity), and one-hot encoded space group and stability label.

* **Handle missing values:** Options for filling missing values with zeros or mean.

* **Prepare ML tensors:** Output $X$ (features) and $y$ (target: formation energy) ready for training.

### Dataset Access

Due to size constraints, the dataset is hosted externally:

👉 [Download Dataset (Zenodo DOI)](https://www.google.com/search?q=https://zenodo.org/record/xxx)

Place the downloaded CSV in the `data/` folder.

## ⚙️ Setup and Installation

Clone the repository and install dependencies:
