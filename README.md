# Crystal Structure Stability Prediction

This project trains a deep learning model with Bayesian optimization and early stopping 
to predict formation energies of materials.

## Data
The dataset is derived from [Materials Project](https://materialsproject.org/) 
and featurized. Due to size restrictions, data is hosted on Zenodo:  
👉 [Download here](https://zenodo.org/records/17218766)

Place the dataset inside the `data/` folder.

## Setup
```bash
git clone https://github.com/lycan134/formation-energy-prediction.git
cd formation-energy-prediction
pip install -r requirements.txt
