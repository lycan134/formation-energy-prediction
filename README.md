git clone https://github.com/lycan134/formation-energy-prediction.git cd formation-energy-prediction pip install -r requirements.txt


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
