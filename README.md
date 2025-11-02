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

python predict.py # Predict formation energy on new data python evaluate.py # Compute MAE, RMSE, R², and generate figures


## 📊 Interpretability

Uses [SHAP](https://shap.readthedocs.io/en/latest/index.html) to analyze feature importance and understand model decisions.

Publication-ready plots saved automatically in the `figures/` folder.

## ⚡ Key Highlights

* Integrates composition, space group, and stability information as features.

* Fully implemented in PyTorch with ML-ready pipeline.

* Ready for interactive deployment in Streamlit.

## 📄 License & Contribution

**License:** [MIT](https://www.google.com/search?q=LICENSE)

Contributions are welcome via GitHub Pull Requests. Please ensure code style consistency and documentation.
