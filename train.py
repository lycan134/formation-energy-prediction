# ==========================================================
# 🧠 Deep Learning Model Training with K-Fold Cross Validation
# ==========================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# ⚙️ 1. Setup and Data Loading
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧩 Using device: {device}")

os.makedirs("models", exist_ok=True)
eps = 1e-8

# Load preprocessed data
X = pd.read_csv("data/X_preprocessed.csv")
y = pd.read_csv("data/y_preprocessed.csv")

# Flatten y if it’s a DataFrame with one column
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Convert to tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

# ==========================================================
# 📊 2. Normalize Inputs and Targets
# ==========================================================
X_mean, X_std = X_tensor.mean(dim=0), X_tensor.std(dim=0)
y_mean, y_std = y_tensor.mean(), y_tensor.std()

X_tensor = (X_tensor - X_mean) / (X_std + eps)
y_tensor = (y_tensor - y_mean) / (y_std + eps)

# Save normalization stats for inference later
torch.save({
    "X_mean": X_mean.cpu(),
    "X_std": X_std.cpu(),
    "y_mean": y_mean.cpu(),
    "y_std": y_std.cpu()
}, "models/normalization_stats.pth")

# ==========================================================
# 🧩 3. Define the Neural Network
# ==========================================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    def forward(self, x):
        return self.layers(x)

# ==========================================================
# 🔁 4. K-Fold Cross Validation Setup
# ==========================================================
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

best_overall_loss = float('inf')
best_overall_model_state = None
fold_train_losses, fold_test_losses = [], []

# ==========================================================
# 🏋️ 5. Training Loop with Early Stopping
# ==========================================================
for fold, (train_idx, test_idx) in enumerate(kfold.split(X_tensor)):
    print(f"\n===== Fold {fold + 1}/{k_folds} =====")

    train_dataset = Subset(TensorDataset(X_tensor, y_tensor), train_idx)
    test_dataset = Subset(TensorDataset(X_tensor, y_tensor), test_idx)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = NeuralNetwork(X_tensor.shape[1], 1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_test_loss = float('inf')
    counter = 0
    patience = 10
    num_epochs = 500
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        # ----- Training -----
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # ----- Validation -----
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(-1, 1))
                test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_loader.dataset)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # ----- Early Stopping -----
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"⏸️ Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train: {train_loss:.4f} | Test: {test_loss:.4f}")

    fold_train_losses.append(train_losses[-1])
    fold_test_losses.append(best_test_loss)

    # Save best model across folds
    if best_test_loss < best_overall_loss:
        best_overall_loss = best_test_loss
        best_overall_model_state = model.state_dict()
        torch.save(best_overall_model_state, "models/best_model_overall.pth")
        print(f"✅ Saved new best model from Fold {fold+1} (Loss={best_test_loss:.4f})")

# ==========================================================
# 📊 6. Plot Loss Summary
# ==========================================================
plt.figure(figsize=(8, 6))
plt.plot(range(1, k_folds+1), fold_train_losses, 'o--', label='Train Loss')
plt.plot(range(1, k_folds+1), fold_test_losses, 'o--', label='Test Loss')
plt.xlabel("Fold", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Average Train & Test Loss per Fold", fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("models/kfold_loss_summary.svg", format='svg', dpi=300)
plt.show()

print("\n=== K-Fold Summary ===")
print(f"Average Train Loss: {np.mean(fold_train_losses):.4f}")
print(f"Average Test Loss:  {np.mean(fold_test_losses):.4f}")

# ==========================================================
# 🧩 7. Save Best Model as .pt (Full Model)
# ==========================================================
print("\n💾 Saving best model as .pt ...")
best_model = NeuralNetwork(X_tensor.shape[1], 1).to(device)
best_model.load_state_dict(torch.load("models/best_model_overall.pth", map_location=device))
best_model.eval()

# Save full model (.pt)
torch.save(best_model, "models/best_model_full.pt")

print("✅ Model successfully saved as:")
print(" - models/best_model_overall.pth (weights only)")
print(" - models/best_model_full.pt (architecture + weights)")

print("\n🎉 Training complete! Ready for inference or deployment.")
