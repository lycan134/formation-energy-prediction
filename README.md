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

data/
├── X_preprocessed.csv
└── y_preprocessed.csv

yaml
Copy code

---

## ⚙️ Setup and Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/lycan134/formation-energy-prediction.git
cd formation-energy-prediction
pip install -r requirements.txt
To verify CUDA availability (for GPU acceleration):

python
Copy code
import torch
print(torch.cuda.is_available())
🧩 Repository Structure
kotlin
Copy code
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
🚀 Usage
🔧 Train the Model
Run the training script to create and save model weights and normalization statistics:

bash
Copy code
python train.py
This will automatically save:

Trained model → models/best_model_full.pt

Normalization parameters → models/normalization_stats.pth

Training output example:

yaml
Copy code
=== K-Fold Summary ===
Average Train Loss: 0.0372
Average Test Loss:  0.0661
✅ Model and normalization stats saved successfully.
📈 Evaluate the Model
Run the evaluation script to compute real-world denormalized metrics and generate plots:

bash
Copy code
python evaluate.py
The script will:

Load the trained .pt model and normalization stats

Compute MAE, RMSE, and R²

Save a True vs Predicted plot in both .eps and .svg formats in the figures/ folder

Example output:

yaml
Copy code
=== Denormalized Performance Metrics ===
MAE  : 1.7001
RMSE : 1.7203
R²   : -0.4247
✅ Evaluation complete.
🎨 Example Output
True vs Predicted Formation Energy (eV/atom)

The diagonal line shows ideal predictions (y = x)

The blue regression line indicates model fit

The metrics box (MAE, RMSE, R²) is displayed directly on the figure

Saved automatically in:

bash
Copy code
figures/true_vs_predicted_plot.eps
figures/true_vs_predicted_plot.svg
🌐 Streamlit App (Coming Soon)
You will soon be able to interactively predict formation energy by uploading preprocessed CSV files.
Once complete, the app will be launched using:

bash
Copy code
streamlit run app.py
Planned features:

File upload and preprocessing preview

Model-based prediction using best_model_full.pt

Visual output (scatter plots, regression metrics, etc.)

🧠 Future Work
 Build and deploy Streamlit app for inference

 Add SHAP-based feature interpretation for model explainability

 Include Docker support for reproducibility and deployment

⚙️ Requirements
All dependencies are listed in requirements.txt.
Example contents:

shell
Copy code
torch>=2.3.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.5.0
matplotlib>=3.8.0
streamlit>=1.37.0
tqdm>=4.66.0
📜 License
This project is released under the MIT License — you are free to use, modify, and distribute with proper attribution.

👨‍💻 Author
Virginio Torlao
Materials Informatics Researcher | Machine Learning Developer
📧 Contact: [your email or GitHub profile link]
🌐 GitHub: lycan134

yaml
Copy code

---

✅ Just copy the entire block above into your `README.md` file — it’s clean, Markdown-formatted, and GitHub-render ready.  

Would you like me to add an optional section for **“📊 Demo Results Preview”** (so once you push your 
